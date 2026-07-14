"""
cropper.py — 16:9 landscape -> 9:16 portrait smart cropper.

The horizontal crop centre X(t) is decided by a multi-stage algorithm:

  1. Per-frame analysis (single decode pass):
       - person boxes      : YOLOv8n  (ultralytics)
       - face boxes        : OpenCV Haar cascade  (MediaPipe optional)
       - saliency          : spectral-residual saliency  (core cv2 only)
       - dense optical flow: Farneback between consecutive frames

  2. Terminal-frame target (frame N):
       Look at the last 10 frames and pick the crop centre that maximises the
       *enclosed content* with a strict hierarchy:  faces > people > saliency.

  3. Backward propagation (frame N-1 .. 0):
       Propagate the crop centre from t+1 back to t using ONLY the optical-flow
       horizontal displacement of the pixels inside the crop window.  Newly
       appearing objects are ignored during this pass — the goal is pixel
       continuity, letting the crop "track" whatever it locked onto at the end.

  4. Temporal smoothing:
       Forward EMA over the target trajectory + boundary clamping so the 9:16
       window never leaves the 16:9 frame.

  5. Render + JSON metadata export.

Run:
    python cropper.py --video /path/to/clip.mp4
    python cropper.py --folder /path/to/folder --limit 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from collections import deque

import cv2
import numpy as np

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# Bump when the algorithm changes so old vs new results are distinguishable.
# v2: detector-persistence terminal decision + centred tie-break + even H.264.
# v3: fast pipeline — detect only on the trailing window + storyboard frames,
#     optical flow at a stride with keyframe interpolation.
ALGO_VERSION = 3

DEFAULT_ALPHA = 0.15        # EMA weight on the raw target (lower = smoother)
DEFAULT_PROC_WIDTH = 480    # width used for flow / saliency (speed vs accuracy)
DEFAULT_FLOW_STRIDE = 5     # compute optical flow every k-th frame, interpolate between
TERMINAL_WINDOW = 10        # number of trailing frames used for the terminal decision
STORYBOARD_TIMES = [0, 3, 6, 9, 12, 15]   # seconds sampled for the storyboard
PORTRAIT_AR = 9.0 / 16.0    # width / height of the output


# --------------------------------------------------------------------------- #
# Detectors
# --------------------------------------------------------------------------- #

class PersonDetector:
    """YOLOv8n person detector (class 0)."""

    def __init__(self, model_path="yolov8n.pt", conf=0.35):
        from ultralytics import YOLO  # lazy import (heavy)
        self.model = YOLO(model_path)
        self.conf = conf

    def detect(self, frame):
        res = self.model(frame, verbose=False, classes=[0], conf=self.conf)
        boxes = []
        if res and res[0].boxes is not None:
            for b in res[0].boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                c = float(b.conf[0])
                boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1), round(c, 3)])
        return boxes  # [x, y, w, h, conf]


class FaceDetector:
    """OpenCV Haar cascade face detector (frontal + profile)."""

    def __init__(self):
        base = cv2.data.haarcascades
        self.frontal = cv2.CascadeClassifier(base + "haarcascade_frontalface_default.xml")
        self.profile = cv2.CascadeClassifier(base + "haarcascade_profileface.xml")

    def detect(self, gray):
        faces = []
        for cascade in (self.frontal, self.profile):
            if cascade.empty():
                continue
            found = cascade.detectMultiScale(gray, scaleFactor=1.15,
                                             minNeighbors=6, minSize=(40, 40))
            for (x, y, w, h) in found:
                faces.append([int(x), int(y), int(w), int(h)])
        return _dedupe_boxes(faces)  # [x, y, w, h]


def _dedupe_boxes(boxes, iou_thr=0.4):
    """Cheap greedy NMS so frontal/profile overlaps don't double count."""
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    kept = []
    for b in boxes:
        if all(_iou(b, k) < iou_thr for k in kept):
            kept.append(b)
    return kept


def _iou(a, b):
    ax1, ay1, aw, ah = a[:4]
    bx1, by1, bw, bh = b[:4]
    ax2, ay2, bx2, by2 = ax1 + aw, ay1 + ah, bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


# --------------------------------------------------------------------------- #
# Saliency (spectral residual — works with plain opencv-python)
# --------------------------------------------------------------------------- #

def saliency_map(gray):
    """Spectral-residual saliency, normalised to [0, 1] at the input resolution."""
    h, w = gray.shape
    small = cv2.resize(gray, (64, 64)).astype(np.float32)
    dft = np.fft.fft2(small)
    log_amp = np.log(np.abs(dft) + 1e-8)
    phase = np.angle(dft)
    avg = cv2.blur(log_amp, (3, 3))
    spectral_residual = log_amp - avg
    combined = np.exp(spectral_residual + 1j * phase)
    sal = np.abs(np.fft.ifft2(combined)) ** 2
    sal = cv2.GaussianBlur(sal, (0, 0), sigmaX=2.5)
    sal = cv2.resize(sal, (w, h))
    mn, mx = float(sal.min()), float(sal.max())
    if mx - mn > 1e-8:
        sal = (sal - mn) / (mx - mn)
    return sal.astype(np.float32)


# --------------------------------------------------------------------------- #
# Optical flow
# --------------------------------------------------------------------------- #

def flow_column_dx(prev_gray, cur_gray):
    """Column-averaged horizontal displacement (prev -> cur), length = width."""
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, cur_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )
    return flow[..., 0].mean(axis=0)  # mean over rows -> per-column dx


# --------------------------------------------------------------------------- #
# Crop-position algorithm
# --------------------------------------------------------------------------- #

def _window_argmax(profile, crop_w):
    """Centre x that maximises the summed profile over a crop_w-wide window.

    When several window positions tie (e.g. a small subject fits many ways), the
    central one is chosen so the subject ends up centred rather than jammed
    against an edge. An empty profile returns the frame centre.
    """
    W = len(profile)
    crop_w = min(crop_w, W)
    csum = np.concatenate([[0.0], np.cumsum(profile)])
    # window [i, i+crop_w) sum = csum[i+crop_w] - csum[i]
    win = csum[crop_w:] - csum[:W - crop_w + 1]
    mx = float(win.max())
    if mx <= 0:
        return W // 2
    near = np.flatnonzero(win >= mx - 1e-9)   # all (near-)optimal left edges
    i = int(near[len(near) // 2])             # central one -> centres the content
    return i + crop_w // 2


def _boxes_profile(boxes_per_frame, W):
    """1-D importance profile: each box contributes its height to its columns."""
    prof = np.zeros(W, dtype=np.float64)
    for frame_boxes in boxes_per_frame:
        for b in frame_boxes:
            x, y, w, h = b[:4]
            x0, x1 = max(0, x), min(W, x + w)
            if x1 > x0:
                prof[x0:x1] += h
    return prof


def decide_terminal_target(faces_pf, people_pf, sal_cols, W, crop_w, min_frac=0.3):
    """
    Pick X_target for the terminal frame from the last TERMINAL_WINDOW frames.
    Strict hierarchy: faces > people > saliency.

    A detector only "wins" if it fires in at least ``min_frac`` of the trailing
    frames, so a single-frame false positive (common with Haar cascades on
    high-contrast textures) does not hijack the decision away from saliency.
    Returns (x_target, source).
    """
    n = max(1, len(sal_cols))
    need = max(1, int(round(min_frac * n)))
    face_hits = sum(1 for f in faces_pf if len(f))
    people_hits = sum(1 for p in people_pf if len(p))

    if face_hits >= need:
        return _window_argmax(_boxes_profile(faces_pf, W), crop_w), "faces"
    if people_hits >= need:
        return _window_argmax(_boxes_profile(people_pf, W), crop_w), "people"
    prof = np.sum(np.asarray(sal_cols), axis=0)
    return _window_argmax(prof, crop_w), "saliency"


def backward_propagate(x_terminal, col_dx, n, W, crop_w):
    """
    col_dx[t] = per-column horizontal flow for transition t -> t+1 (len W).
    Propagate the crop centre backward using only flow inside the window.
    """
    x = np.zeros(n, dtype=np.float64)
    x[n - 1] = x_terminal
    half = crop_w // 2
    for t in range(n - 2, -1, -1):
        c = int(round(x[t + 1]))
        lo, hi = max(0, c - half), min(W, c + half)
        dx = float(np.mean(col_dx[t][lo:hi])) if hi > lo else 0.0
        # pixels at column c in frame t+1 were at c-dx in frame t
        x[t] = x[t + 1] - dx
    return x


def backward_propagate_kf(x_terminal, keyframes, gap_dx, W, crop_w):
    """
    Keyframe version of the backward pass (used by the fast pipeline).
    keyframes : ascending frame indices where flow was sampled (first = 0).
    gap_dx[j] : per-column horizontal flow from keyframes[j] -> keyframes[j+1].
    Returns the crop centre at each keyframe; interpolate for the rest.
    """
    m = len(keyframes)
    xk = np.zeros(m, dtype=np.float64)
    xk[-1] = x_terminal
    half = crop_w // 2
    for j in range(m - 2, -1, -1):
        c = int(round(xk[j + 1]))
        lo, hi = max(0, c - half), min(W, c + half)
        dx = float(np.mean(gap_dx[j][lo:hi])) if hi > lo else 0.0
        xk[j] = xk[j + 1] - dx      # content at keyframes[j+1] was here at keyframes[j]
    return xk


def ema_smooth(x_target, alpha, W, crop_w):
    """Forward EMA + boundary clamp of the crop centre."""
    half = crop_w // 2
    lo, hi = half, W - half
    xs = np.zeros_like(x_target)
    xs[0] = np.clip(x_target[0], lo, hi)
    for t in range(1, len(x_target)):
        xs[t] = alpha * x_target[t] + (1 - alpha) * xs[t - 1]
    return np.clip(xs, lo, hi)


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #

def process_video(path, out_root, alpha=DEFAULT_ALPHA, proc_width=DEFAULT_PROC_WIDTH,
                  flow_stride=DEFAULT_FLOW_STRIDE, detect_window=TERMINAL_WINDOW,
                  model_path="yolov8n.pt", render=True, verbose=True, full=False):
    """
    Fast pipeline:
      * heavy detectors (YOLO person + Haar face + saliency) run ONLY on the
        trailing `detect_window` frames (used for the terminal decision) and on
        the 6 storyboard frames (used for the dashboard overlays). `full=True`
        restores per-frame detection.
      * dense optical flow is sampled every `flow_stride` frames; the crop
        trajectory is propagated backward across those keyframes and linearly
        interpolated for the frames in between.
    """
    name = os.path.splitext(os.path.basename(path))[0]
    t_start = time.time()
    if full:
        flow_stride = 1

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    crop_w = int(round(H * PORTRAIT_AR))
    crop_w = min(crop_w, W)                 # never wider than the frame
    crop_w -= crop_w % 2                    # even width (required by H.264 yuv420p)
    scale = W / float(proc_width)           # proc-space -> full-space factor
    proc_h = max(1, int(round(H / scale)))
    xp = np.linspace(0, W - 1, proc_width)  # proc-column -> full-column mapping

    if verbose:
        print(f"[{name}] {W}x{H} @ {fps:.1f}fps  crop_w={crop_w}  "
              f"flow_stride={flow_stride}", flush=True)

    person = PersonDetector(model_path)
    face = FaceDetector()

    def detect_frame(frame, gray, small):
        faces = face.detect(gray)
        people = person.detect(frame)
        sal = saliency_map(small)
        col_full = np.interp(np.arange(W), xp, sal.sum(axis=0))
        return faces, people, sal, col_full, (float(sal.mean()), float(sal.max()))

    story_targets = {int(round(s * fps)) for s in STORYBOARD_TIMES}
    story_dir = os.path.join(out_root, "frames", name)
    os.makedirs(story_dir, exist_ok=True)
    storyboard = []

    dets = {}                          # frame index -> (faces, people, (sal_mean, sal_max))
    sal_col_by_i = {}                  # frame index -> saliency column profile (len W)
    keyframes, gap_dx = [], []         # flow sample indices + per-gap column dx (len W)
    prev_kf_small, prev_kf_idx = None, None
    last_small, last_idx = None, None
    ring = deque(maxlen=detect_window)  # trailing (idx, frame, gray, small)

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (proc_width, proc_h))

        # ---- optical flow at keyframes ----
        if prev_kf_small is None:
            keyframes.append(i)                     # first keyframe, no preceding gap
            prev_kf_small, prev_kf_idx = small, i
        elif i - prev_kf_idx >= flow_stride:
            dx = np.interp(np.arange(W), xp, flow_column_dx(prev_kf_small, small)) * scale
            keyframes.append(i)
            gap_dx.append(dx)
            prev_kf_small, prev_kf_idx = small, i
        last_small, last_idx = small, i

        # ---- detection: storyboard frames now, trailing window after the loop ----
        want = full or (i in story_targets)
        if i in story_targets:
            faces, people, sal, col_full, stt = detect_frame(frame, gray, small)
            dets[i] = (faces, people, stt)
            sal_col_by_i[i] = col_full
            fpath = os.path.join(story_dir, f"f_{i}.jpg")
            spath = os.path.join(story_dir, f"s_{i}.png")
            cv2.imwrite(fpath, frame)
            cv2.imwrite(spath, (sal * 255).astype(np.uint8))
            storyboard.append({"t": round(i / fps, 2), "i": i,
                               "frame": os.path.relpath(fpath, out_root),
                               "saliency": os.path.relpath(spath, out_root)})
        elif full:
            faces, people, sal, col_full, stt = detect_frame(frame, gray, small)
            dets[i] = (faces, people, stt)
            sal_col_by_i[i] = col_full

        ring.append((i, frame, gray, small))
        i += 1

    cap.release()
    n = i
    if n == 0:
        raise RuntimeError(f"no frames decoded: {path}")

    # close the final flow gap so the last frame is a keyframe
    if keyframes[-1] != last_idx:
        dx = np.interp(np.arange(W), xp, flow_column_dx(prev_kf_small, last_small)) * scale
        keyframes.append(last_idx)
        gap_dx.append(dx)
    if not gap_dx:                                   # single-frame guard
        keyframes = [0, max(0, n - 1)]
        gap_dx = [np.zeros(W)]

    # ---- detect the trailing window (for the terminal decision) ----
    for idx, frame, gray, small in ring:
        if idx not in dets:
            faces, people, sal, col_full, stt = detect_frame(frame, gray, small)
            dets[idx] = (faces, people, stt)
            sal_col_by_i[idx] = col_full

    w0 = max(0, n - detect_window)
    faces_win = [dets[k][0] for k in range(w0, n) if k in dets]
    people_win = [dets[k][1] for k in range(w0, n) if k in dets]
    sal_win = [sal_col_by_i[k] for k in range(w0, n) if k in sal_col_by_i]
    x_terminal, source = decide_terminal_target(faces_win, people_win, sal_win, W, crop_w)

    # ---- backward propagation across keyframes + interpolation + smoothing ----
    xk = backward_propagate_kf(x_terminal, keyframes, gap_dx, W, crop_w)
    x_target = np.interp(np.arange(n), keyframes, xk)
    x_smooth = ema_smooth(x_target, alpha, W, crop_w)

    # ---- render 9:16 output ----
    out_video_rel = None
    if render:
        out_dir = os.path.join(out_root, "outputs")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{name}_portrait.mp4")
        _render(path, out_path, x_smooth, crop_w, H, fps)
        out_video_rel = os.path.relpath(out_path, out_root)

    # ---- metadata JSON (detections present only where computed) ----
    half = crop_w // 2
    frames_meta = []
    for t in range(n):
        c = float(x_smooth[t])
        x0 = int(np.clip(round(c - half), 0, W - crop_w))
        faces, people, stt = dets.get(t, ([], [], (0.0, 0.0)))
        frames_meta.append({
            "i": t,
            "t": round(t / fps, 3),
            "x_target": round(float(x_target[t]), 2),
            "x_smooth": round(c, 2),
            "dx": round(float(x_target[t] - x_target[t - 1]), 3) if t > 0 else 0.0,
            "crop": [x0, 0, x0 + crop_w, H],
            "faces": faces,
            "people": people,
            "saliency": {"mean": round(stt[0], 4), "max": round(stt[1], 4)},
            "detected": t in dets,
        })

    meta = {
        "video": os.path.basename(path),
        "path": path,
        "width": W, "height": H, "fps": round(fps, 3), "n_frames": n,
        "algo_version": ALGO_VERSION,
        "crop_width": crop_w, "crop_height": H, "aspect": "9:16",
        "terminal": {"source": source, "x_target": int(x_terminal),
                     "window": min(detect_window, n)},
        "params": {"alpha": alpha, "proc_width": proc_width,
                   "flow_stride": flow_stride, "detect_window": detect_window,
                   "full": full, "model": os.path.basename(model_path)},
        "output_video": out_video_rel,
        "storyboard": storyboard,
        "frames": frames_meta,
        "elapsed_sec": round(time.time() - t_start, 1),
    }

    meta_dir = os.path.join(out_root, "metadata")
    os.makedirs(meta_dir, exist_ok=True)
    meta_path = os.path.join(meta_dir, f"{name}.json")
    with open(meta_path, "w") as fh:
        json.dump(meta, fh)

    if verbose:
        print(f"[{name}] done in {meta['elapsed_sec']}s  source={source}  "
              f"-> {os.path.relpath(meta_path, out_root)}", flush=True)
    return meta


def _render(src_path, out_path, x_smooth, crop_w, H, fps):
    """Crop each frame and write an H.264 (yuv420p, faststart) mp4 so the result
    plays in browsers / the Streamlit dashboard. Falls back to the avc1 writer
    when ffmpeg is not on PATH."""
    import shutil
    import subprocess

    ffmpeg = shutil.which("ffmpeg")
    cap = cv2.VideoCapture(src_path)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # With ffmpeg we write a raw mp4v temp and transcode; without it we ask cv2
    # for avc1 (H.264) directly.
    tmp = out_path + ".tmp.mp4" if ffmpeg else out_path
    fourcc = cv2.VideoWriter_fourcc(*("mp4v" if ffmpeg else "avc1"))
    writer = cv2.VideoWriter(tmp, fourcc, fps, (crop_w, H))
    half = crop_w // 2
    t = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        c = x_smooth[min(t, len(x_smooth) - 1)]
        x0 = int(np.clip(round(c - half), 0, W - crop_w))
        writer.write(frame[:, x0:x0 + crop_w])
        t += 1
    writer.release()
    cap.release()

    if ffmpeg:
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-i", tmp,
             "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
             "-movflags", "+faststart", out_path],
            check=True,
        )
        os.remove(tmp)


# --------------------------------------------------------------------------- #
# Batch helpers / CLI
# --------------------------------------------------------------------------- #

def list_videos(folder):
    """os.listdir (not glob/stat) so it works under restricted metadata access."""
    exts = (".mp4", ".mov", ".avi", ".mkv")
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(exts))


def main():
    ap = argparse.ArgumentParser(description="16:9 -> 9:16 smart cropper")
    ap.add_argument("--video", help="single video file")
    ap.add_argument("--folder", help="folder of videos")
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)),
                    help="project root for metadata/outputs/frames")
    ap.add_argument("--limit", type=int, default=0, help="process first N (0=all)")
    ap.add_argument("--skip", type=int, default=0, help="skip first N videos")
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    ap.add_argument("--proc-width", type=int, default=DEFAULT_PROC_WIDTH)
    ap.add_argument("--flow-stride", type=int, default=DEFAULT_FLOW_STRIDE,
                    help="compute optical flow every k-th frame (interpolate between)")
    ap.add_argument("--detect-window", type=int, default=TERMINAL_WINDOW,
                    help="number of trailing frames to run detectors on")
    ap.add_argument("--full", action="store_true",
                    help="run detectors on every frame (slow, spec-faithful)")
    ap.add_argument("--model", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "models", "yolov8n.pt"))
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args()

    if args.video:
        targets = [args.video]
    elif args.folder:
        names = list_videos(args.folder)
        names = names[args.skip:]
        if args.limit:
            names = names[:args.limit]
        targets = [os.path.join(args.folder, n) for n in names]
    else:
        ap.error("provide --video or --folder")

    print(f"processing {len(targets)} video(s)", flush=True)
    ok = 0
    for p in targets:
        try:
            process_video(p, args.out, alpha=args.alpha, proc_width=args.proc_width,
                          flow_stride=args.flow_stride, detect_window=args.detect_window,
                          full=args.full, model_path=args.model,
                          render=not args.no_render)
            ok += 1
        except Exception as e:
            print(f"[ERROR] {os.path.basename(p)}: {e}", flush=True)
            traceback.print_exc()
    print(f"finished: {ok}/{len(targets)} ok", flush=True)


if __name__ == "__main__":
    main()
