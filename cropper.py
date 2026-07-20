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
# v4: saliency re-anchoring during the backward pass to correct flow drift.
# v5: epipole (focus-of-expansion) tracking for forward motion + median-filter
#     stability; 15-frame storyboard; landscape green-frame overlay render.
# v6: motion saliency blended into the saliency signal; weak/partial person
#     detections fall through to it (so the crop can follow the moving action).
# v8: anchor the whole trajectory to a detected PERSON (not just the terminal
#     frame) when it's a single consistent subject — fixes crops that drift off
#     the person (e.g. 5127498). Gated by subject-consistency; faces/scattered
#     detections keep the stable v7 flow+saliency path.
# v7: epipole from ~1s-apart frames + EMA (stable heading); final output video
#     stabilization via long-term feature tracking (removes residual jitter).
ALGO_VERSION = 8

DEFAULT_STABILIZE = True     # feature-tracking stabilization pass on the rendered output
STAB_ZOOM = 1.04            # slight zoom to hide stabilization warp borders
STAB_RADIUS_SEC = 1.0       # moving-average radius (seconds) for the long-term trajectory
STAB_MIN_SHAKE = 0.8        # px: skip stabilization when the source is already steady
                            # (avoids adding resample noise; sub-pixel crop still applies)
EPIPOLE_EMA = 0.9           # epipole smoothing: new = 0.9*last + 0.1*measured
DEFAULT_MOTION_WEIGHT = 0.5  # blend: saliency = w*motion + (1-w)*static (0 = static only)
DEFAULT_ALPHA = 0.15        # EMA weight on the raw target (lower = smoother)
DEFAULT_PROC_WIDTH = 480    # width used for flow / saliency (speed vs accuracy)
DEFAULT_FLOW_STRIDE = 5     # compute optical flow every k-th frame, interpolate between
DEFAULT_ANCHOR = 0.5        # blend toward per-keyframe saliency to fight flow drift (0=pure flow)
TERMINAL_WINDOW = 10        # number of trailing frames used for the terminal decision
STORYBOARD_COUNT = 16       # number of evenly-spaced storyboard frames
FWD_MIN_FRAC = 0.5          # >= this fraction of keyframes showing a focus-of-expansion => forward motion
MEDIAN_K = 19              # temporal median window (keyframes) for trajectory de-spiking
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
    return flow_columns(prev_gray, cur_gray)[0]


def flow_columns(prev_gray, cur_gray):
    """One Farneback pass -> (dx_col, motion_col) per column.

    dx_col     : column-averaged *signed* horizontal flow (for epipole/backprop).
    motion_col : column-averaged flow *magnitude* sqrt(u^2+v^2) -> motion saliency
                 (high where things move, e.g. reaching hands over a static table).
    """
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, cur_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )
    dx = flow[..., 0].mean(axis=0)
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).mean(axis=0)
    return dx, mag


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


def _people_weak(people_pf, W, H, conf_thr=0.5):
    """True when person detections are unreliable — low YOLO confidence OR mostly
    small/edge-touching boxes (an arm/hand reaching in, not a full body). Such
    clips fall through to the (motion-blended) saliency instead of tracking a box."""
    boxes = [b for pf in people_pf for b in pf]
    if not boxes:
        return True
    confs = [b[4] for b in boxes if len(b) > 4]
    mean_conf = float(np.mean(confs)) if confs else 0.0

    def partial(b):
        x, y, w, h = b[:4]
        return x <= 2 or (x + w) >= W - 2 or y <= 2 or (y + h) >= H - 2
    frac_partial = float(np.mean([partial(b) for b in boxes]))
    return mean_conf < conf_thr and frac_partial > 0.5


def decide_terminal_target(faces_pf, people_pf, sal_cols, W, crop_w, H=10 ** 9,
                           min_frac=0.3):
    """
    Pick X_target for the terminal frame from the last TERMINAL_WINDOW frames.
    Hierarchy: faces > people (if not weak) > saliency (motion-blended).

    A detector only "wins" if it fires in at least ``min_frac`` of the trailing
    frames (rejects single-frame false positives), and people additionally must
    not be "weak" (see _people_weak) — otherwise the crop follows the moving,
    salient action instead of a flimsy person box. Returns (x_target, source).
    """
    n = max(1, len(sal_cols))
    need = max(1, int(round(min_frac * n)))
    face_hits = sum(1 for f in faces_pf if len(f))
    people_hits = sum(1 for p in people_pf if len(p))

    if face_hits >= need:
        return _window_argmax(_boxes_profile(faces_pf, W), crop_w), "faces"
    if people_hits >= need and not _people_weak(people_pf, W, H):
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


def backward_propagate_kf(x_terminal, keyframes, gap_dx, W, crop_w,
                          kf_sal=None, anchor_x=None, anchor=0.0):
    """
    Keyframe version of the backward pass (used by the fast pipeline).
    keyframes : ascending frame indices where flow was sampled (first = 0).
    gap_dx[j] : per-column horizontal flow from keyframes[j] -> keyframes[j+1].

    Pure optical-flow integration drifts over long clips, so when ``anchor`` > 0
    each step is a blend of the flow-propagated centre and a drift-correcting
    anchor. The anchor per keyframe is ``anchor_x[j]`` when given (e.g. the
    tracked subject centre — keeps the crop ON a detected person/face without the
    jumpiness of using the raw track directly), otherwise the saliency-optimal
    centre from ``kf_sal[j]``. ``anchor=0`` = pure flow.
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
        flow_pos = xk[j + 1] - dx    # content at keyframes[j+1] was here at keyframes[j]
        ap = None
        if anchor > 0:
            if anchor_x is not None:
                ap = float(anchor_x[j])
            elif kf_sal is not None and kf_sal[j] is not None:
                ap = _window_argmax(kf_sal[j], crop_w)
        xk[j] = (anchor * ap + (1 - anchor) * flow_pos) if ap is not None else flow_pos
    return xk


def focus_of_expansion(col_dx, W):
    """
    Estimate the horizontal focus of expansion (epipole) from the column-averaged
    horizontal optical flow of one keyframe gap.

    Forward camera/drone motion makes the flow diverge: horizontal flow is
    negative left of the heading and positive to its right. The FOE is the
    neg->pos zero-crossing. Returns the FOE x (full-frame px) or None when the
    divergence pattern is absent (i.e. the clip is not moving forward here).
    """
    if col_dx is None or len(col_dx) < 8:
        return None
    k = max(3, (W // 40) | 1)                       # odd smoothing window
    u = np.convolve(col_dx, np.ones(k) / k, mode="same")
    left = u[:int(W * 0.4)].mean()
    right = u[int(W * 0.6):].mean()
    if not (left < 0 < right):                       # require divergence
        return None
    zc = np.where((u[:-1] < 0) & (u[1:] >= 0))[0]
    if len(zc) == 0:
        return None
    return float(zc[len(zc) // 2])


def _median_filter(x, k):
    """1-D temporal median filter (odd k); de-spikes the target trajectory."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < 3 or k < 3:
        return x
    k = min(k | 1, n if n % 2 else n - 1)
    h = k // 2
    out = np.empty(n)
    for i in range(n):
        out[i] = np.median(x[max(0, i - h):min(n, i + h + 1)])
    return out


def stability_grade(x_smooth, W, fps):
    """
    Grade how steady the crop trajectory is (A best .. F worst). Penalises the
    'crop jumps sides' behaviour: large-amplitude direction reversals (swings
    across the frame), plus overall pan busyness. A slow single pan scores well;
    repeated full-frame oscillation scores badly.
    Returns {score 0-100, grade, big_swings, swing_rate, travel_per_sec, range_frac}.
    """
    xs = np.asarray(x_smooth, dtype=np.float64)
    n = len(xs)
    dur = max(1e-6, n / (fps or 30.0))
    if n < 3 or W <= 0:
        return {"score": 100, "grade": "A", "big_swings": 0,
                "swing_rate": 0.0, "travel_per_sec": 0.0, "range_frac": 0.0}
    d = np.diff(xs)
    travel_ps = (np.abs(d).sum() / W) / dur          # frame-widths panned / sec
    rng = (xs.max() - xs.min()) / W
    # amplitude between successive turning points -> count "big" reversals
    ext = [xs[0]]
    for i in range(1, n - 1):
        if (xs[i] - xs[i - 1]) * (xs[i + 1] - xs[i]) < 0:
            ext.append(xs[i])
    ext.append(xs[-1])
    amps = np.abs(np.diff(np.asarray(ext))) / W if len(ext) > 1 else np.array([0.0])
    big = int((amps > 0.25).sum())
    swing_rate = big / dur
    swing_pen = min(1.0, swing_rate / 0.20)          # >=0.2 big swings/sec = maxed
    travel_pen = min(1.0, travel_ps / 0.25)
    instability = 0.75 * swing_pen + 0.25 * travel_pen
    score = int(round(100 * (1 - instability)))
    grade = ("A" if score >= 85 else "B" if score >= 70 else
             "C" if score >= 55 else "D" if score >= 40 else "F")
    return {"score": score, "grade": grade, "big_swings": big,
            "swing_rate": round(swing_rate, 3),
            "travel_per_sec": round(travel_ps, 3), "range_frac": round(rng, 3)}


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
                  anchor=DEFAULT_ANCHOR, motion_weight=DEFAULT_MOTION_WEIGHT,
                  stabilize=DEFAULT_STABILIZE, model_path="yolov8n.pt", render=True,
                  verbose=True, full=False):
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

    total_est = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total_est > 1:
        story_targets = {int(round(x))
                         for x in np.linspace(0, total_est - 1, STORYBOARD_COUNT)}
    else:
        story_targets = set()   # unknown length; storyboard stays sparse
    story_dir = os.path.join(out_root, "frames", name)
    os.makedirs(story_dir, exist_ok=True)
    storyboard = []

    def sal_col(small_gray):
        return np.interp(np.arange(W), xp, saliency_map(small_gray).sum(axis=0))

    dets = {}                          # frame index -> (faces, people, (sal_mean, sal_max))
    sal_col_by_i = {}                  # frame index -> saliency column profile (len W)
    keyframes, gap_dx, gap_motion, kf_sal, kf_small = [], [], [], [], []
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

        # ---- optical flow (dx + motion) + saliency at keyframes ----
        if prev_kf_small is None:
            keyframes.append(i)                     # first keyframe, no preceding gap
            kf_sal.append(sal_col(small))
            kf_small.append(small)
            prev_kf_small, prev_kf_idx = small, i
        elif i - prev_kf_idx >= flow_stride:
            dxc, magc = flow_columns(prev_kf_small, small)
            gap_dx.append(np.interp(np.arange(W), xp, dxc) * scale)
            gap_motion.append(np.interp(np.arange(W), xp, magc))
            keyframes.append(i)
            kf_sal.append(sal_col(small))
            kf_small.append(small)
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
        dxc, magc = flow_columns(prev_kf_small, last_small)
        gap_dx.append(np.interp(np.arange(W), xp, dxc) * scale)
        gap_motion.append(np.interp(np.arange(W), xp, magc))
        keyframes.append(last_idx)
        kf_sal.append(sal_col(last_small))
        kf_small.append(last_small)
    if not gap_dx:                                   # single-frame guard
        keyframes = [0, max(0, n - 1)]
        gap_dx = [np.zeros(W)]
        gap_motion = [np.zeros(W)]
        kf_sal = [kf_sal[0] if kf_sal else np.zeros(W)] * 2
        kf_small = [kf_small[0] if kf_small else None] * 2

    # ---- blend motion saliency into the per-keyframe saliency profiles ----
    kf_motion = [gap_motion[0]] + gap_motion         # align to keyframes

    def _norm(p):
        p = np.asarray(p, dtype=np.float64)
        s = p.sum()
        return p / s if s > 0 else p
    kf_sal_blend = [motion_weight * _norm(kf_motion[j])
                    + (1 - motion_weight) * _norm(kf_sal[j])
                    for j in range(len(keyframes))]

    # ---- detect the trailing window (for the terminal decision) ----
    for idx, frame, gray, small in ring:
        if idx not in dets:
            faces, people, sal, col_full, stt = detect_frame(frame, gray, small)
            dets[idx] = (faces, people, stt)
            sal_col_by_i[idx] = col_full

    # ---- focus-of-expansion (epipole) from ~1s-apart frames + EMA smoothing ----
    # Larger temporal baseline (1s) makes the divergence -> FOE estimate far more
    # stable than consecutive keyframes; an exponential filter removes residual jitter.
    kback = max(1, int(round((fps or 30) / max(1, flow_stride))))   # keyframes ~= 1s apart
    raw_foe = []
    for j in range(len(keyframes)):
        a = j - kback
        if a < 0 or kf_small[j] is None or kf_small[a] is None:
            raw_foe.append(None)
            continue
        dxc, _ = flow_columns(kf_small[a], kf_small[j])
        raw_foe.append(focus_of_expansion(np.interp(np.arange(W), xp, dxc) * scale, W))
    kf_foe, ema = [None] * len(keyframes), None
    for j, f in enumerate(raw_foe):
        if f is not None:
            ema = f if ema is None else EPIPOLE_EMA * ema + (1 - EPIPOLE_EMA) * f
        kf_foe[j] = ema                                   # EMA-smoothed epipole (carried through gaps)
    valid_foe = [f for f in raw_foe if f is not None]
    foe_fraction = len(valid_foe) / max(1, len(raw_foe))
    is_forward = foe_fraction >= FWD_MIN_FRAC and len(valid_foe) >= 2

    # ---- choose target trajectory: epipole WINS on forward motion ----
    w0 = max(0, n - detect_window)
    faces_win = [dets[k][0] for k in range(w0, n) if k in dets]
    people_win = [dets[k][1] for k in range(w0, n) if k in dets]
    # trailing motion-blended saliency profiles (for the saliency terminal pick)
    sal_win = [kf_sal_blend[j] for j in range(len(keyframes)) if keyframes[j] >= w0]
    if not sal_win:
        sal_win = kf_sal_blend[-2:]

    if is_forward:
        source = "epipole"
        idx = [j for j, f in enumerate(kf_foe) if f is not None]
        xk = np.interp(np.arange(len(keyframes)), idx, [kf_foe[j] for j in idx])
        x_terminal = float(xk[-1])
    else:
        x_terminal, source = decide_terminal_target(
            faces_win, people_win, sal_win, W, crop_w, H)
        # When a subject (faces/people) wins, ANCHOR the whole trajectory to it —
        # not just the terminal frame. Flow gives smooth continuity; the subject
        # anchor (detected centres, interpolated across the clip) stops the crop
        # drifting off the person. Anchoring (vs using the raw track) avoids the
        # jumpiness of sparse/multi-subject detections.
        # Track PEOPLE only (reliable YOLO). Haar faces false-positive on textures,
        # so face-source clips keep the stable v7 flow+saliency path.
        key = 1 if source == "people" else None
        det_idx = [k for k in sorted(dets) if key is not None and dets[k][key]]
        det_x = ([_window_argmax(_boxes_profile([dets[k][key]], W), crop_w)
                  for k in det_idx] if det_idx else [])
        # Only TRACK the subject when it is a single, consistent one; if the
        # detections are scattered (multiple people / flickering false faces),
        # tracking them jitters the crop, so fall back to the stable v7 anchor.
        single_subject = len(det_x) >= 2 and float(np.std(det_x)) < 0.20 * W
        if single_subject:
            anchor_x = _median_filter(np.interp(keyframes, det_idx, det_x), 9)
            x_terminal = float(anchor_x[-1])
            xk = backward_propagate_kf(x_terminal, keyframes, gap_dx, W, crop_w,
                                       anchor_x=anchor_x, anchor=DEFAULT_ANCHOR)
        else:
            xk = backward_propagate_kf(x_terminal, keyframes, gap_dx, W, crop_w,
                                       kf_sal=kf_sal, anchor=anchor)

    # ---- de-spike (both paths) + interpolate + smooth ----
    xk = _median_filter(xk, MEDIAN_K)
    x_target = np.interp(np.arange(n), keyframes, xk)
    x_smooth = ema_smooth(x_target, alpha, W, crop_w)

    # per-frame focus of expansion (for the dashboard heading marker)
    if valid_foe:
        fidx = [keyframes[j] for j, f in enumerate(kf_foe) if f is not None]
        foe_frame = np.interp(np.arange(n), fidx, [kf_foe[j] for j in range(len(kf_foe))
                                                   if kf_foe[j] is not None])
    else:
        foe_frame = None

    # ---- per-frame mathematical quality score ----
    # Object coverage (0.7): fraction of detected person/face box area kept inside
    # the crop (1.0 when nothing is detected). Saliency coverage (0.3): saliency
    # mass inside the crop over total. Computed where signals exist, interpolated
    # to every frame (exact per-frame with --full).
    half = crop_w // 2

    def crop_x0(c):
        return int(np.clip(round(c - half), 0, W - crop_w))

    def object_coverage(faces, people, c):
        boxes = [b[:4] for b in faces] + [b[:4] for b in people]
        if not boxes:
            return 1.0
        x0 = crop_x0(c); x1 = x0 + crop_w
        num = den = 0.0
        for (x, y, w, h) in boxes:
            ov = max(0, min(x + w, x1) - max(x, x0))   # crop spans full height
            num += h * ov
            den += h * w
        return num / den if den > 0 else 1.0

    def saliency_coverage(profile, c):
        tot = float(profile.sum())
        if tot <= 0:
            return 1.0
        x0 = crop_x0(c)
        return float(profile[x0:x0 + crop_w].sum() / tot)

    obj_idx = sorted(dets.keys())
    if obj_idx:
        obj_vals = [object_coverage(dets[k][0], dets[k][1], x_smooth[k]) for k in obj_idx]
        obj_cov = np.interp(np.arange(n), obj_idx, obj_vals)
    else:
        obj_cov = np.ones(n)

    sal_profiles = {keyframes[j]: kf_sal[j] for j in range(len(keyframes))}
    sal_idx = sorted(sal_profiles.keys())
    sal_vals = [saliency_coverage(sal_profiles[k], x_smooth[k]) for k in sal_idx]
    sal_cov = np.interp(np.arange(n), sal_idx, sal_vals) if sal_idx else np.ones(n)

    frame_scores = 0.7 * obj_cov + 0.3 * sal_cov

    # ---- render 9:16 portrait output + landscape green-frame overlay ----
    out_video_rel = out_overlay_rel = None
    if render:
        out_dir = os.path.join(out_root, "outputs")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{name}_portrait.mp4")
        correction = None
        if stabilize:
            correction = _estimate_stab_correction(
                path, proc_width, radius=max(1, int(round((fps or 30) * STAB_RADIUS_SEC))))
        _render(path, out_path, x_smooth, crop_w, H, fps, correction=correction)
        out_video_rel = os.path.relpath(out_path, out_root)
        ov_path = os.path.join(out_dir, f"{name}_overlay.mp4")
        _render_overlay(path, ov_path, x_smooth, foe_frame, crop_w, W, H, fps)
        out_overlay_rel = os.path.relpath(ov_path, out_root)

    # ---- metadata JSON (detections present only where computed) ----
    frames_meta = []
    for t in range(n):
        c = float(x_smooth[t])
        x0 = crop_x0(c)
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
            "score": round(float(frame_scores[t]), 4),
            "foe_x": (round(float(foe_frame[t]), 1) if foe_frame is not None else None),
        })

    meta = {
        "video": os.path.basename(path),
        "path": path,
        "width": W, "height": H, "fps": round(fps, 3), "n_frames": n,
        "algo_version": ALGO_VERSION,
        "crop_width": crop_w, "crop_height": H, "aspect": "9:16",
        "terminal": {"source": source, "x_target": int(x_terminal),
                     "window": min(detect_window, n)},
        "motion": {"forward": bool(is_forward),
                   "foe_fraction": round(foe_fraction, 3),
                   "foe_mean_x": (round(float(np.mean(valid_foe)), 1)
                                  if valid_foe else None)},
        "params": {"alpha": alpha, "proc_width": proc_width,
                   "flow_stride": flow_stride, "detect_window": detect_window,
                   "anchor": anchor, "motion_weight": motion_weight,
                   "stabilize": stabilize, "full": full,
                   "model": os.path.basename(model_path)},
        "output_video": out_video_rel,
        "output_overlay": out_overlay_rel,
        "storyboard": storyboard,
        "frame_scores": [round(float(s), 4) for s in frame_scores],
        "quality": {
            "avg_score": round(float(np.mean(frame_scores)), 4),
            "avg_object_coverage": round(float(np.mean(obj_cov)), 4),
            "avg_saliency_coverage": round(float(np.mean(sal_cov)), 4),
        },
        "stability": stability_grade(x_smooth, W, fps),
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


def _moving_avg(x, radius):
    """Box moving average (edge-clamped) — the long-term trajectory smoother."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n == 0 or radius < 1:
        return x
    out = np.empty(n)
    for i in range(n):
        out[i] = x[max(0, i - radius):min(n, i + radius + 1)].mean()
    return out


def _estimate_stab_correction(src_path, proc_w, radius):
    """
    Long-term feature-tracking stabilization: track corners frame-to-frame
    (goodFeaturesToTrack + LK), fit a similarity transform (shift+rotate+zoom),
    accumulate the camera trajectory, smooth it over a long window, and return the
    per-frame correction (dx, dy, da in full-res px / radians) that cancels the
    high-frequency jitter while preserving the intended slow motion.
    """
    cap = cv2.VideoCapture(src_path)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    sc = W / float(proc_w)
    ok, prev = cap.read()
    if not ok:
        cap.release()
        return None
    ph = max(1, int(round(prev.shape[0] / sc)))
    pg = cv2.cvtColor(cv2.resize(prev, (proc_w, ph)), cv2.COLOR_BGR2GRAY)
    transforms = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(f, (proc_w, ph)), cv2.COLOR_BGR2GRAY)
        dx = dy = da = 0.0
        pts = cv2.goodFeaturesToTrack(pg, maxCorners=200, qualityLevel=0.01,
                                      minDistance=15, blockSize=3)
        if pts is not None and len(pts) >= 6:
            npts, stt, _ = cv2.calcOpticalFlowPyrLK(pg, g, pts, None)
            if npts is not None:
                gp, gn = pts[stt == 1], npts[stt == 1]
                if len(gp) >= 6:
                    M, _ = cv2.estimateAffinePartial2D(gp, gn)
                    if M is not None:
                        dx, dy = M[0, 2] * sc, M[1, 2] * sc
                        da = float(np.arctan2(M[1, 0], M[0, 0]))
        transforms.append((dx, dy, da))
        pg = g
    cap.release()

    n = len(transforms) + 1
    traj = np.zeros((n, 3))
    for i, (dx, dy, da) in enumerate(transforms):
        traj[i + 1] = traj[i] + (dx, dy, da)
    smooth = np.stack([_moving_avg(traj[:, k], radius) for k in range(3)], axis=1)
    correction = smooth - traj                        # correction per frame
    # Lightly smooth the correction: cancels sustained drift/rotation without
    # re-injecting the per-frame estimation noise (the sub-pixel crop handles HF).
    correction = np.stack([_moving_avg(correction[:, k], 2) for k in range(3)], axis=1)
    # only stabilize when there is meaningful shake to remove; otherwise skip so
    # the (already steady) clip isn't degraded by warp-resample noise.
    shake = float(np.hypot(correction[:, 0], correction[:, 1]).std())
    return correction if shake >= STAB_MIN_SHAKE else None


def _render(src_path, out_path, x_smooth, crop_w, H, fps, correction=None):
    """Crop each frame and write an H.264 (yuv420p, faststart) mp4 so the result
    plays in browsers / the Streamlit dashboard. Falls back to the avc1 writer
    when ffmpeg is not on PATH. When ``correction`` is given, each source frame is
    first stabilized (warp by the per-frame correction + a slight zoom to hide
    borders) before cropping, removing residual jitter."""
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
    half = crop_w / 2.0
    t = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        c = float(x_smooth[min(t, len(x_smooth) - 1)])
        c = min(max(c, half), W - half)
        # Sub-pixel crop (removes the ~1px integer-quantization shimmer): affine that
        # maps source column c -> output centre. Optionally composed with the
        # feature-tracking stabilization warp (rotate+zoom+shift) in one resample.
        crop_m = np.array([[1.0, 0.0, half - c], [0.0, 1.0, 0.0]])
        if correction is not None and t < len(correction):
            dx, dy, da = correction[t]
            S = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), float(np.degrees(da)), STAB_ZOOM)
            S[0, 2] += dx
            S[1, 2] += dy
            S3 = np.vstack([S, [0, 0, 1]])
            C3 = np.vstack([crop_m, [0, 0, 1]])
            crop_m = (C3 @ S3)[:2]
        out = cv2.warpAffine(frame, crop_m, (crop_w, H),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
        writer.write(out)
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


def _render_overlay(src_path, out_path, x_smooth, foe_frame, crop_w, W, H, fps):
    """Write the ORIGINAL 16:9 video with the moving green crop rectangle drawn on
    each frame (and a heading marker where the epipole is known). Lets the viewer
    see what is kept vs discarded over time. H.264, same transcode as _render."""
    import shutil
    import subprocess

    ffmpeg = shutil.which("ffmpeg")
    cap = cv2.VideoCapture(src_path)
    tmp = out_path + ".tmp.mp4" if ffmpeg else out_path
    fourcc = cv2.VideoWriter_fourcc(*("mp4v" if ffmpeg else "avc1"))
    writer = cv2.VideoWriter(tmp, fourcc, fps, (W, H))
    half = crop_w // 2
    t = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        c = x_smooth[min(t, len(x_smooth) - 1)]
        x0 = int(np.clip(round(c - half), 0, W - crop_w))
        cv2.rectangle(frame, (x0, 0), (x0 + crop_w - 1, H - 1), (0, 255, 0), 4)
        if foe_frame is not None:
            fx = int(np.clip(round(foe_frame[min(t, len(foe_frame) - 1)]), 0, W - 1))
            cy = H // 2                                   # heading crosshair (magenta)
            cv2.drawMarker(frame, (fx, cy), (255, 0, 255),
                           cv2.MARKER_CROSS, 40, 3)
        writer.write(frame)
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
    ap.add_argument("--anchor", type=float, default=DEFAULT_ANCHOR,
                    help="0..1 blend toward per-keyframe saliency (0=pure flow)")
    ap.add_argument("--motion-weight", type=float, default=DEFAULT_MOTION_WEIGHT,
                    help="0..1 blend of motion vs static saliency (0=static only)")
    ap.add_argument("--no-stabilize", action="store_true",
                    help="disable the feature-tracking output stabilization pass")
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
                          anchor=args.anchor, motion_weight=args.motion_weight,
                          stabilize=not args.no_stabilize,
                          full=args.full, model_path=args.model,
                          render=not args.no_render)
            ok += 1
        except Exception as e:
            print(f"[ERROR] {os.path.basename(p)}: {e}", flush=True)
            traceback.print_exc()
    print(f"finished: {ok}/{len(targets)} ok", flush=True)


if __name__ == "__main__":
    main()
