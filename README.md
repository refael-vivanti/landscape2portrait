# landscape2portrait — 16:9 → 9:16 smart video cropper

Convert 16:9 landscape videos into 9:16 portrait by sliding a full-height vertical
crop window horizontally over time, so it keeps the *important* content in frame
and moves smoothly like a real camera operator. The only quantity the algorithm
decides is the horizontal **crop centre `X(t)`** for every frame
(`crop_width = round(H · 9/16)`, full height, no letterboxing).

> **Two-minute tour:** watch a clip in `sample_outputs/` (the `*_overlay.mp4`
> shows the original 16:9 with the moving green crop box — the clearest way to see
> what the algorithm decided), read the **decision tree** below, then open the
> **Streamlit dashboard** to explore all results and the version-over-version
> comparison. Full methodology & rationale: **[APPROACH.md](APPROACH.md)**.

---

## Quickstart

```bash
# 0. system dependency: ffmpeg (H.264 encode/mux).  macOS: brew install ffmpeg
# 1. python env
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # opencv, ultralytics (torch), streamlit, ...

# 2. convert 16:9 -> 9:16  (YOLOv8n weights auto-download on first run)
python cropper.py --video /path/to/clip.mp4          # one clip
python cropper.py --folder /path/to/videos           # a whole folder

# 3. explore the results
streamlit run app.py
```

Outputs land in `outputs/`: `<name>_portrait.mp4` (the 9:16 deliverable) and
`<name>_overlay.mp4` (original 16:9 with the moving crop box, for review).
Ten pre-rendered examples — two per crop-strategy category — are in
`sample_outputs/`.

Assumptions (per the brief): single continuous shot (no scene-cut handling),
full-height crop (no letterboxing), **audio omitted** (optional per spec).

---

## The decision tree (v8)

```
For each clip:

  ANALYZE (single decode pass; cheap — see "Performance"):
    YOLOv8n person · Haar face · spectral-residual saliency · Farneback optical flow
    (detectors on the trailing 10 + 16 storyboard frames; flow at stride-5 keyframes)
    estimate the epipole / focus-of-expansion from ~1s-baseline column flow, EMA-smoothed

  CHOOSE the target trajectory X(t):
    ┌ forward camera motion?            → track the EPIPOLE (where we're heading)
    │   (FOE present in ≥50% of keyframes)         e.g. aerial road, drone flythrough
    ├ strong, consistent PERSON?        → track the PERSON across the whole clip
    │   (in ≥40% of detected frames, x-std < 0.20·W)   e.g. kayak, skateboarder, a walker
    └ else → terminal decision on the last 10 frames:
              faces  >  people (if not weak)  >  saliency (motion-blended)
              then backward optical-flow propagation + saliency anchor
                                                  e.g. centered/off-center anchors, pans, handoffs

  SMOOTH:  temporal median (19 keyframes)  +  EMA (α=0.15)  +  clamp to frame
  RENDER:  sub-pixel crop  +  adaptive feature-tracking stabilization  →  H.264 9:16
```

Why this shape: deciding each frame independently makes the crop jump between
subjects; instead we pick one intent and hold it. A reliable **person** (YOLO) is
tracked directly; a **flythrough** follows its natural target (the epipole); every
other clip anchors on the terminal frame and walks **backward via optical flow**
for pixel continuity, corrected by a saliency anchor. See **[APPROACH.md](APPROACH.md)**
for the derivation, the per-category mapping, and the 8-version insight log.

---

## Evaluation (how we iterated)

Rather than eyeballing, every result is scored so regressions and improvements
are visible at a glance across 118 videos and 8 algorithm versions:

- **Math grade** — `0.7 · object_coverage + 0.3 · saliency_coverage`: how much of
  the detected people/faces and salient mass the crop keeps in frame. A blunt but
  fast proxy — the *lowest* scores are almost always genuinely bad crops.
- **Stability grade (A–F)** — from the crop trajectory: penalises large
  side-to-side swings and pan busyness. A volatile crop is bad UX even if it's
  "informative".
- **AI Director score (VLM-as-judge)** — Gemini rates the storyboard 1–5 on
  subject retention + temporal flow (provider-agnostic: `GEMINI_API_KEY`,
  `OPENAI_API_KEY`, or `LLAMA_API_KEY`).

The **Streamlit dashboard** has an **Overview** page (bird's-eye: grades table,
aggregate stats, and per-metric charts comparing every version across all clips —
so you see whether a change helped globally or regressed specific videos) and a
**per-video** page (root-cause: 16-frame storyboard with overlays, a frame
slider, the `X(t)` trajectory, and the crop-box overlay video).

### Results across versions (all 118 videos)

| ver | math % | stability | grades A / F | headline change |
|-----|:------:|:---------:|:------------:|-----------------|
| v4  | 58.9 | 75.6 | 67 / 15 | baseline: terminal decision + backward flow + saliency anchor; fast pipeline |
| v5  | 58.6 | 81.2 | 82 / 13 | epipole (forward-motion) + median stabilisation + storyboard/overlay |
| v6  | 58.8 | 91.5 | 98 / 2  | **median window 5→19 (big stability win)** + motion-saliency (terminal-only) |
| v7  | 58.3 | 92.0 | 98 / 1  | sub-pixel crop + adaptive feature-tracking stabilization + stable epipole (1s+EMA) |
| v8  | **64.6** | 90.3 | 92 / 3 | **person subject-tracking** (any source) → keeps people in frame (+coverage) |

Stability climbed sharply through v6; v8 then traded ~1.5 stability points for
**+5.7 coverage** by actually keeping the subject in frame. v8 source mix:
person 70, saliency 23, epipole 14, faces 7, people 4.

---

## Running it

```bash
# a whole folder (resumable: skips clips that already have metadata)
python batch.py --folder /path/to/videos

# the fixed 10-clip smoke set used for fast iteration (2 per category)
python batch.py --folder /path/to/videos --list smoke_set.txt --force

# with the VLM judge (needs an API key in .env)
python batch.py --folder /path/to/videos --ai-eval
```

Useful `cropper.py` flags: `--anchor` (0..1 saliency pull vs pure flow),
`--motion-weight` (motion vs static saliency), `--flow-stride`, `--detect-window`,
`--no-stabilize`, `--full` (detect every frame, slower), `--no-render`.

**Performance:** detectors + saliency run only on the trailing + storyboard frames
and flow at a stride, so a ~520-frame clip processes in ~10 s (~18× faster than
per-frame) with a near-identical trajectory; `--full --flow-stride 1` for the
exhaustive version. Rendering adds a stabilization pass.

---

## Limitations (short)

- **Cumbersome / over-engineered** for a "few-hours" brief — a lot of moving parts
  and heuristic thresholds rather than one learned model.
- Haar faces false-positive on textures; the design works around this (track the
  reliable YOLO person) but it's a patch.
- No identity-consistent multi-object tracking — "narrative handoff" clips are
  handled heuristically.
- The stability grade *penalises correct subject-following* (a moving subject means
  a moving crop), so it's read alongside the coverage grade, not alone.
- Horizontal reframing only; assumes a single shot; the math grade is a proxy, not
  ground truth.

Full limitations + **future work** in **[APPROACH.md](APPROACH.md)**.

---

## Repo layout

```
landscape2portrait/
├── cropper.py        # core algorithm + CLI (the decision tree lives here)
├── batch.py          # resumable folder / smoke-set runner
├── evaluate.py       # VLM-as-judge (Gemini/OpenAI/Llama) + storyboard grid
├── app.py            # Streamlit dashboard (Overview + per-video root-cause)
├── build_history.py  # rebuilds the version-comparison data for the Overview
├── smoke_set.txt     # the fixed 10-clip iteration set (2 per category)
├── requirements.txt
├── APPROACH.md       # methodology, KPIs, insight log, decision tree, limitations
├── sample_outputs/   # 10 ready-to-watch results (portrait + overlay)
├── metadata/         # <video>.json per clip (committed — the analysis output)
├── history/          # versions.json (v4→v8 metrics for the Overview)
├── outputs/ frames/ models/   # rendered videos, cached frames, weights (git-ignored)
```
