"""
app.py — Streamlit visualization dashboard for the smart cropper.

Reads the JSON metadata produced by cropper.py (metadata/*.json) plus the cached
storyboard frames (frames/<video>/). For each video it renders:

  * a storyboard strip of 15 frames across the clip with:
      - green rectangle = final 9:16 crop window (x_smooth)
      - bounding boxes + labels for faces (cyan) and people (yellow)
      - red saliency heatmap overlay
      - magenta crosshair = epipole / heading (forward-motion clips)
  * an interactive frame-by-frame inspector (slider) that scrubs the whole clip
  * the original 16:9 video with the moving green crop frame (+ portrait output)

Run:
    streamlit run app.py
"""

import json
import os

import cv2
import numpy as np
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
META_DIR = os.path.join(ROOT, "metadata")

FACE_COLOR = (255, 255, 0)     # cyan (BGR)
PERSON_COLOR = (0, 255, 255)   # yellow (BGR)
CROP_COLOR = (0, 255, 0)       # green (BGR)
FOE_COLOR = (255, 0, 255)      # magenta (BGR) — epipole / heading


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def load_index():
    # Not cached: metadata is overwritten live by the batch, so always read fresh.
    if not os.path.isdir(META_DIR):
        return {}
    out = {}
    for f in sorted(os.listdir(META_DIR)):
        if f.endswith(".json"):
            try:
                with open(os.path.join(META_DIR, f)) as fh:
                    out[f[:-5]] = json.load(fh)
            except Exception:
                pass
    return out


def frame_by_index(meta, idx):
    """Frame metadata by integer index (frames list is ordered)."""
    frames = meta["frames"]
    idx = max(0, min(idx, len(frames) - 1))
    return frames[idx]


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #

def apply_saliency_overlay(img, sal_path, strength=0.5):
    if not sal_path or not os.path.exists(sal_path):
        return img
    sal = cv2.imread(sal_path, cv2.IMREAD_GRAYSCALE)
    if sal is None:
        return img
    sal = cv2.resize(sal, (img.shape[1], img.shape[0]))
    heat = np.zeros_like(img)
    heat[..., 2] = sal                       # red channel (BGR)
    a = (sal.astype(np.float32) / 255.0 * strength)[..., None]
    return (img * (1 - a) + heat * a).astype(np.uint8)


def draw_overlays(img, fmeta, draw_sal_path=None, label=True):
    if draw_sal_path:
        img = apply_saliency_overlay(img, draw_sal_path)
    for (x, y, w, h) in fmeta.get("faces", []):
        cv2.rectangle(img, (x, y), (x + w, y + h), FACE_COLOR, 2)
        if label:
            cv2.putText(img, "face", (x, max(0, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, FACE_COLOR, 2)
    for b in fmeta.get("people", []):
        x, y, w, h = b[:4]
        cv2.rectangle(img, (x, y), (x + w, y + h), PERSON_COLOR, 2)
        if label:
            conf = b[4] if len(b) > 4 else ""
            cv2.putText(img, f"person {conf}", (x, max(0, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, PERSON_COLOR, 2)
    x0, y0, x1, y1 = fmeta["crop"]
    thick = max(6, img.shape[1] // 60)   # scale with frame so it survives downscaling
    cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), CROP_COLOR, thick)
    foe = fmeta.get("foe_x")
    if foe is not None:
        fx = int(max(0, min(img.shape[1] - 1, round(foe))))
        cv2.drawMarker(img, (fx, img.shape[0] // 2), FOE_COLOR,
                       cv2.MARKER_CROSS, max(24, img.shape[0] // 12), 3)
    return img


def to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #

st.set_page_config(page_title="Landscape → Portrait Smart Crop", layout="wide")
st.title("🎬 16:9 → 9:16 Smart Crop — Evaluation Dashboard")

index = load_index()
if not index:
    st.warning(f"No metadata found in `{META_DIR}`. "
               "Run `python cropper.py --folder <videos> --limit 5` first.")
    st.stop()

CURRENT_ALGO = 5   # keep in sync with cropper.ALGO_VERSION

names = list(index.keys())
st.sidebar.header("Videos")
n_new = sum(1 for m in index.values() if m.get("algo_version", 1) >= CURRENT_ALGO)
st.sidebar.caption(f"{len(names)} processed · {n_new} on current algo (v{CURRENT_ALGO})")
# mark each entry in the sidebar list so old results are obvious
labels = {n: f'{"✅" if index[n].get("algo_version",1) >= CURRENT_ALGO else "⚠️"} {n}'
          for n in names}
choice = st.sidebar.radio("Select", names, format_func=lambda n: labels[n],
                          label_visibility="collapsed")
meta = index[choice]

ver = meta.get("algo_version", 1)
if ver >= CURRENT_ALGO:
    st.success(f"✅ Algorithm v{ver} (current — epipole tracking + stability).")
else:
    st.warning(f"⚠️ Algorithm v{ver} (OLDER result — reprocess to update).")

# ---- evaluation summary cards (prominent) ----
scores = meta.get("frame_scores", [])
avg_math = (sum(scores) / len(scores)) if scores else None
ai_score = meta.get("ai_score")

e1, e2 = st.columns(2)
e1.metric("📐 Average Math Score",
          f"{avg_math * 100:.1f}%" if avg_math is not None else "—",
          help="Mean of the per-frame quality score: 0.7·object-coverage + "
               "0.3·saliency-coverage inside the crop window.")
e2.metric("🤖 AI Director Score",
          f"{ai_score:.1f} / 5.0" if isinstance(ai_score, (int, float)) else "—",
          help="VLM rating of the storyboard (subject retention + temporal "
               "flow). Run the batch with --ai-eval to populate.")
if meta.get("ai_reasoning"):
    (st.info if isinstance(ai_score, (int, float)) else st.caption)(
        f'**AI reasoning:** {meta["ai_reasoning"]}')

# ---- technical summary ----
fwd = meta.get("motion", {}).get("forward")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Resolution", f'{meta["width"]}×{meta["height"]}')
c2.metric("Frames", meta["n_frames"])
c3.metric("Crop width", meta["crop_width"])
c4.metric("Target", meta["terminal"]["source"] + (" ✈️" if fwd else ""))

# ---- storyboard (15 frames across the clip, single row) ----
st.subheader("Storyboard — 15 frames across the clip"
             + ("  ✈️ forward motion (epipole tracked)" if fwd else ""))
st.caption("🟩 crop · 🟦 faces · 🟨 people · 🔴 saliency"
           + ("  · ✚ heading (epipole)" if fwd else ""))

story = meta.get("storyboard", [])
if story:
    per_row = (len(story) + 1) // 2          # two rows (e.g. 8 + 7)
    for group in (story[:per_row], story[per_row:]):
        cols = st.columns(per_row)
        for k, s in enumerate(group):
            fpath = os.path.join(ROOT, s["frame"])
            spath = os.path.join(ROOT, s["saliency"])
            if not os.path.exists(fpath):
                cols[k].caption(f"{s['t']}s —")
                continue
            img = cv2.imread(fpath)
            fmeta = frame_by_index(meta, s["i"])
            img = draw_overlays(img, fmeta, draw_sal_path=spath, label=False)
            cols[k].image(to_rgb(img), caption=f'{s["t"]}s',
                          use_container_width=True)

# ---- interactive inspector ----
st.subheader("Frame-by-frame inspector")
n = meta["n_frames"]
idx = st.slider("Frame", 0, n - 1, 0)
fmeta = frame_by_index(meta, idx)

src = meta.get("path")
cap = cv2.VideoCapture(src) if src else None
frame = None
if cap is not None and cap.isOpened():
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()

left, right = st.columns([3, 1])
if frame is not None:
    frame = draw_overlays(frame, fmeta, draw_sal_path=None)
    left.image(to_rgb(frame), caption=f'frame {idx} · t={fmeta["t"]}s',
               use_container_width=True)
else:
    left.info("Source video not reachable from here — showing metadata only. "
              "(Storyboard above uses cached frames and always works.)")

right.markdown("**Crop / tracking**")
right.write({
    "t (s)": fmeta["t"],
    "x_smooth": fmeta["x_smooth"],
    "x_target (raw)": fmeta["x_target"],
    "epipole x": fmeta.get("foe_x"),
    "forward motion": bool(meta.get("motion", {}).get("forward")),
    "faces": len(fmeta.get("faces", [])),
    "people": len(fmeta.get("people", [])),
    "saliency mean": fmeta["saliency"]["mean"],
})

# ---- trajectory + quality charts ----
cta, ctb = st.columns(2)
with cta:
    st.subheader("Crop-centre trajectory X(t)")
    xs = [f["x_smooth"] for f in meta["frames"]]
    xt = [f["x_target"] for f in meta["frames"]]
    st.line_chart({"x_smooth (rendered)": xs, "x_target (raw backward pass)": xt})
with ctb:
    st.subheader("Per-frame quality score")
    if scores:
        st.line_chart({"quality": scores})
        q = meta.get("quality", {})
        st.caption(f'avg {avg_math*100:.1f}%  ·  object cov '
                   f'{q.get("avg_object_coverage", 0)*100:.0f}%  ·  saliency cov '
                   f'{q.get("avg_saliency_coverage", 0)*100:.0f}%')
    else:
        st.info("No frame_scores — reprocess this video with the current cropper.")

# ---- output: landscape with moving green crop frame (main) + portrait ----
st.subheader("Rendered output")
ov = meta.get("output_overlay")
ov_path = os.path.join(ROOT, ov) if ov else None
if ov_path and os.path.exists(ov_path):
    st.caption("Original 16:9 with the moving green crop frame 🟩"
               + (" and heading marker ✚" if fwd else "")
               + " — shows what is kept vs discarded.")
    st.video(ov_path)
else:
    st.info("Landscape overlay not found — reprocess with the current cropper.")

out = meta.get("output_video")
out_path = os.path.join(ROOT, out) if out else None
if out_path and os.path.exists(out_path):
    with st.expander("▶ 9:16 portrait output (the actual conversion)"):
        vcol, _ = st.columns([1, 4])
        vcol.video(out_path)
