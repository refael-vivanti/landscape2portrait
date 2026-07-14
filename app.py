"""
app.py — Streamlit visualization dashboard for the smart cropper.

Reads the JSON metadata produced by cropper.py (metadata/*.json) plus the cached
storyboard frames (frames/<video>/). For each video it renders:

  * a storyboard matrix of 6 frames (0,3,6,9,12,15s) with:
      - green rectangle = final 9:16 crop window (x_smooth)
      - bounding boxes + labels for faces (cyan) and people (yellow)
      - red saliency heatmap overlay
  * an interactive frame-by-frame inspector (slider) that scrubs the whole clip
  * the rendered 9:16 output video

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


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

@st.cache_data(show_spinner=False)
def load_index():
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
    cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), CROP_COLOR, 4)
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

names = list(index.keys())
st.sidebar.header("Videos")
st.sidebar.caption(f"{len(names)} processed")
choice = st.sidebar.radio("Select", names, label_visibility="collapsed")
meta = index[choice]

# ---- summary ----
c1, c2, c3, c4 = st.columns(4)
c1.metric("Resolution", f'{meta["width"]}×{meta["height"]}')
c2.metric("Frames", meta["n_frames"])
c3.metric("Crop width", meta["crop_width"])
c4.metric("Terminal target", meta["terminal"]["source"])

# ---- storyboard ----
st.subheader("Storyboard (0, 3, 6, 9, 12, 15 s)")
st.caption("🟩 crop window · 🟦 faces · 🟨 people · 🔴 saliency heatmap")

story = meta.get("storyboard", [])
cols = st.columns(3)
for k, s in enumerate(story):
    fpath = os.path.join(ROOT, s["frame"])
    spath = os.path.join(ROOT, s["saliency"])
    if not os.path.exists(fpath):
        cols[k % 3].info(f"t={s['t']}s frame missing")
        continue
    img = cv2.imread(fpath)
    fmeta = frame_by_index(meta, s["i"])
    img = draw_overlays(img, fmeta, draw_sal_path=spath)
    cols[k % 3].image(to_rgb(img), caption=f't = {s["t"]} s  (frame {s["i"]})',
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
    "flow dx": fmeta["dx"],
    "faces": len(fmeta.get("faces", [])),
    "people": len(fmeta.get("people", [])),
    "saliency mean": fmeta["saliency"]["mean"],
})

# ---- trajectory chart ----
st.subheader("Crop-centre trajectory X(t)")
xs = [f["x_smooth"] for f in meta["frames"]]
xt = [f["x_target"] for f in meta["frames"]]
st.line_chart({"x_smooth (rendered)": xs, "x_target (raw backward pass)": xt})

# ---- output video ----
st.subheader("Rendered 9:16 output")
out = meta.get("output_video")
out_path = os.path.join(ROOT, out) if out else None
if out_path and os.path.exists(out_path):
    # portrait video at full width is huge — pin it to a narrow column so it fits
    vcol, _ = st.columns([1, 4])
    vcol.video(out_path)
else:
    st.info("Rendered video not found (run cropper without --no-render).")
