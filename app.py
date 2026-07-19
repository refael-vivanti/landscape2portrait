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
from collections import Counter

import cv2
import numpy as np
import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
META_DIR = os.path.join(ROOT, "metadata")
HIST_PATH = os.path.join(ROOT, "history", "versions.json")


def load_history():
    try:
        with open(HIST_PATH) as fh:
            return json.load(fh)
    except Exception:
        return None

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
# Overview page
# --------------------------------------------------------------------------- #

def render_overview(index, history, current_algo):
    st.header("📊 Overview — all videos")

    # ---- current-version aggregate stats ----
    maths, stabs, grades, ais = [], [], [], []
    for m in index.values():
        q = m.get("quality", {}).get("avg_score")
        if q is not None:
            maths.append(q * 100)
        s = m.get("stability", {})
        if s.get("score") is not None:
            stabs.append(s["score"]); grades.append(s.get("grade"))
        ai = m.get("ai_score")
        if isinstance(ai, (int, float)):
            ais.append(ai)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Videos", len(index))
    c2.metric("Mean math score", f"{np.mean(maths):.1f}%" if maths else "—")
    c3.metric("Mean stability", f"{np.mean(stabs):.0f}/100" if stabs else "—")
    c4.metric("Mean AI score",
              f"{np.mean(ais):.2f}" if ais else "—",
              help=f"{len(ais)} of {len(index)} scored")
    gc = Counter(grades)
    st.caption("Stability grade counts — "
               + " · ".join(f"**{g}**: {gc.get(g, 0)}" for g in "ABCDF"))

    # ---- version comparison ----
    if history and history.get("versions"):
        vers, data, vids = history["versions"], history["data"], history["videos"]

        rows = []
        for ver in vers:
            d = data.get(ver, {})
            st_ = [d[v]["stability"] for v in vids if v in d and d[v].get("stability") is not None]
            ma_ = [d[v]["math"] for v in vids if v in d and d[v].get("math") is not None]
            ai_ = [d[v]["ai"] for v in vids if v in d and d[v].get("ai") is not None]
            gc2 = Counter(d[v]["grade"] for v in vids if v in d)
            rows.append({"version": ver,
                         "mean math %": round(np.mean(ma_), 1) if ma_ else None,
                         "mean stability": round(np.mean(st_), 1) if st_ else None,
                         "A": gc2.get("A", 0), "B": gc2.get("B", 0), "C": gc2.get("C", 0),
                         "D": gc2.get("D", 0), "F": gc2.get("F", 0),
                         "mean AI": round(np.mean(ai_), 2) if ai_ else None})
        st.subheader("Version comparison — means across all videos")
        st.dataframe(pd.DataFrame(rows).set_index("version"), use_container_width=True)

        st.subheader("Per-video comparison across versions")
        st.caption("Each line is one version across all videos (x = video index). "
                   "Lines moving up = improvement; a version dipping below the others "
                   "at some x = a per-video regression there.")
        for metric, label in [("stability", "Stability score (0–100)"),
                              ("math", "Math score (%)")]:
            df = pd.DataFrame(
                {ver: [data[ver].get(v, {}).get(metric) for v in vids] for ver in vers},
                index=range(len(vids)))
            st.markdown(f"**{label}**")
            st.line_chart(df)

        # ---- biggest per-video changes between the last two versions ----
        if len(vers) >= 2:
            a, b = vers[-2], vers[-1]
            deltas = []
            for v in vids:
                if v in data[a] and v in data[b]:
                    da, db = data[a][v].get("stability"), data[b][v].get("stability")
                    if da is not None and db is not None:
                        deltas.append({"video": v, f"{a}": da, f"{b}": db, "Δ": db - da})
            dd = pd.DataFrame(deltas).sort_values("Δ")
            st.subheader(f"Biggest stability changes  {a} → {b}")
            lc, rc = st.columns(2)
            lc.caption("⬇️ Regressions")
            lc.dataframe(dd.head(8).set_index("video"), use_container_width=True)
            rc.caption("⬆️ Improvements")
            rc.dataframe(dd.tail(8).iloc[::-1].set_index("video"), use_container_width=True)
    else:
        st.info("No history/versions.json — run `python build_history.py` to enable "
                "version comparison.")

    # ---- full current grades table ----
    st.subheader(f"All videos — current (v{current_algo})")
    trows = []
    for name, m in index.items():
        s = m.get("stability", {}); q = m.get("quality", {}).get("avg_score")
        ai = m.get("ai_score")
        trows.append({"video": name, "grade": s.get("grade"),
                      "stability": s.get("score"),
                      "math %": round(q * 100, 1) if q is not None else None,
                      "AI": ai if isinstance(ai, (int, float)) else None,
                      "source": m.get("terminal", {}).get("source"),
                      "forward": m.get("motion", {}).get("forward"),
                      "big swings": s.get("big_swings")})
    st.dataframe(pd.DataFrame(trows).set_index("video"),
                 use_container_width=True, height=520)


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

CURRENT_ALGO = 6   # keep in sync with cropper.ALGO_VERSION
OVERVIEW = "📊 Overview"

names = list(index.keys())
st.sidebar.header("Videos")
n_new = sum(1 for m in index.values() if m.get("algo_version", 1) >= CURRENT_ALGO)
st.sidebar.caption(f"{len(names)} processed · {n_new} on current algo (v{CURRENT_ALGO})")
# mark each entry: algo freshness (✅/⚠️) + stability grade colour dot
_GDOT = {"A": "🟢", "B": "🟢", "C": "🟡", "D": "🟠", "F": "🔴"}


def _sidebar_label(n):
    if n == OVERVIEW:
        return OVERVIEW
    m = index[n]
    fresh = "✅" if m.get("algo_version", 1) >= CURRENT_ALGO else "⚠️"
    dot = _GDOT.get(m.get("stability", {}).get("grade"), "")
    return f"{fresh}{dot} {n}"


choice = st.sidebar.radio("Select", [OVERVIEW] + names, format_func=_sidebar_label,
                          label_visibility="collapsed")

if choice == OVERVIEW:
    render_overview(index, load_history(), CURRENT_ALGO)
    st.stop()

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

stab = meta.get("stability", {})
grade = stab.get("grade")
gcolor = {"A": "🟢", "B": "🟢", "C": "🟡", "D": "🟠", "F": "🔴"}.get(grade, "")

e1, e2, e3 = st.columns(3)
e1.metric("📐 Average Math Score",
          f"{avg_math * 100:.1f}%" if avg_math is not None else "—",
          help="Mean of the per-frame quality score: 0.7·object-coverage + "
               "0.3·saliency-coverage inside the crop window.")
e2.metric("🤖 AI Director Score",
          f"{ai_score:.1f} / 5.0" if isinstance(ai_score, (int, float)) else "—",
          help="VLM rating of the storyboard (subject retention + temporal "
               "flow). Run the batch with --ai-eval to populate.")
e3.metric("📈 Stability grade",
          f"{gcolor} {grade}" if grade else "—",
          help="Camera-work steadiness (A best … F worst) from the crop "
               "trajectory: penalises large side-to-side swings + pan busyness.")
if grade:
    e3.caption(f"{stab.get('score')}/100 · {stab.get('big_swings')} big swings")
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
st.caption(f"Frame {idx} of {n - 1}  ·  t = {round(idx / (meta['fps'] or 30), 2)} s")
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
    if grade:
        st.caption(f"Stability {gcolor} **{grade}** ({stab.get('score')}/100) · "
                   f"{stab.get('big_swings')} big side-swings · "
                   f"{stab.get('travel_per_sec')} frame-widths panned/s")
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
