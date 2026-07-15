# landscape2portrait — 16:9 → 9:16 Smart Video Cropper

Convert 16:9 landscape videos into 9:16 portrait by intelligently sliding a
vertical crop window horizontally over time so it keeps the *important* content
(faces → people → salient regions) in frame, while moving smoothly like a real
camera operator.

The crop keeps the full frame height `H` and uses a window of width
`W_crop = round(H · 9/16)`. The only quantity the algorithm decides is the
horizontal **crop centre `X(t)`** for every frame.

---

## Algorithm

The centre trajectory is built in five stages (`cropper.py`).

### 1. Per-frame analysis (single decode pass)
For each frame we extract four signals:

| Signal | Method | Used for |
|---|---|---|
| **People** | YOLOv8n (`ultralytics`), class `person` | terminal target |
| **Faces** | OpenCV Haar cascade (frontal + profile) | terminal target |
| **Saliency** | Spectral-residual saliency (core `cv2`, no contrib needed) | terminal target |
| **Optical flow** | Dense Farneback between consecutive frames | backward propagation |

Flow and saliency are computed at a reduced `--proc-width` (default 480px) for
speed and rescaled back to full resolution; detection runs on the full frame.

### 2. Terminal-frame target (frame *N*)
The **last frame's** crop centre is chosen first, by inspecting the last
`10` frames and maximising *enclosed content* under a **strict hierarchy**:

1. **Faces** — if any face is present, pick the window that encloses the most
   face area.
2. **People** — else, the window enclosing the most person area.
3. **Saliency** — else, the window with the highest summed saliency.

Mechanically this builds a 1-D importance profile over image columns and takes
the sliding-window argmax (`_window_argmax` via a cumulative sum).

### 3. Backward propagation via optical flow (frame *N-1 → 0*)
This is the core idea. Instead of independently re-deciding a target for every
frame (which causes jitter and identity switches), we **anchor on the terminal
frame and walk backward in time**, letting optical flow tell us where that same
content *was* one frame earlier:

```
X[N-1] = X_terminal
for t = N-2 … 0:
    dx    = mean horizontal flow (t → t+1) over the columns inside the window at X[t+1]
    X[t]  = X[t+1] − dx      # pixels at X[t+1] in frame t+1 were at X[t+1]−dx in frame t
```

Only optical flow is used here — newly appearing objects are **deliberately
ignored** during the backward pass. The result is a target trajectory that
maximises pixel continuity: the crop "locks on" to whatever mattered at the end
and tracks it consistently throughout the clip.

### 4. Temporal smoothing
A forward Exponential Moving Average removes residual jitter, followed by
boundary clamping so the window never leaves the frame:

```
X_smooth[0] = clamp(X_target[0])
X_smooth[t] = clamp( α · X_target[t] + (1 − α) · X_smooth[t−1] )
```

`α` (default `0.15`) trades responsiveness for smoothness (lower = smoother).

### 5. Render + metadata
Each frame is cropped at `X_smooth[t]` and written to a 9:16 `.mp4`
(`outputs/`). All per-frame data is exported to `metadata/<video>.json`.

---

## JSON schema (`metadata/<video>.json`)

```jsonc
{
  "video": "7747235.mp4",
  "width": 1920, "height": 1080, "fps": 30.0, "n_frames": 450,
  "crop_width": 607, "crop_height": 1080, "aspect": "9:16",
  "terminal": { "source": "faces|people|saliency", "x_target": 950, "window": 10 },
  "params":   { "alpha": 0.15, "proc_width": 480, "stride": 1, "model": "yolov8n.pt" },
  "output_video": "outputs/7747235_portrait.mp4",
  "storyboard": [ { "t": 0, "i": 0, "frame": "frames/7747235/f_0.jpg",
                    "saliency": "frames/7747235/s_0.png" }, … ],
  "frames": [
    {
      "i": 0, "t": 0.0,
      "x_target": 951.2,       // raw backward-pass centre
      "x_smooth": 948.7,       // rendered centre (EMA + clamp)
      "dx": 1.4,               // flow displacement used at this frame
      "crop": [645, 0, 1252, 1080],
      "faces":  [[x, y, w, h], …],
      "people": [[x, y, w, h, conf], …],
      "saliency": { "mean": 0.11, "max": 1.0 }
    }, …
  ]
}
```

`metadata/*.json` is the committed deliverable. `outputs/`, `frames/`, and model
weights are reproducible artifacts and are git-ignored.

---

## Evaluation & analytics

Two complementary quality signals are produced per video.

### 1. Mathematical quality score (per frame)

For every frame `t`, `cropper.py` computes:

- **Object coverage (70%)** — fraction of detected person/face box *area* that
  falls inside the crop window (`1.0` when nothing is detected).
- **Saliency coverage (30%)** — saliency mass inside the crop over the frame's
  total saliency.
- `Quality[t] = 0.7·object_coverage[t] + 0.3·saliency_coverage[t]`

Saved to the JSON as `frame_scores` (1-D array, one per frame) plus a `quality`
summary (`avg_score`, `avg_object_coverage`, `avg_saliency_coverage`). Object
coverage is measured on the detected frames and interpolated between them
(`--full` makes it exact per frame); saliency coverage uses the per-keyframe
saliency profiles.

### 2. AI (VLM) director score

`evaluate.py` composes the 6 storyboard frames (with green crop overlays) into a
grid and asks a vision LLM for a 1.0–5.0 rating of subject retention + temporal
flow. The parsed `ai_score` / `ai_reasoning` are merged into the video's JSON.

It is **provider-agnostic**, auto-selecting by whichever key is set
(`llama > openai > gemini`), or force one with `--ai-provider`:

| Provider | Key | Notes |
|---|---|---|
| `llama` | `LLAMA_API_KEY` | Meta Llama API (OpenAI-compatible, `https://api.llama.com/compat/v1`). Recommended for Meta FTEs: high rate limits, data stays on approved infra. |
| `openai` | `OPENAI_API_KEY` | Any OpenAI-compatible endpoint (`OPENAI_BASE_URL` to override). |
| `gemini` | `GEMINI_API_KEY` / `GOOGLE_API_KEY` | Google Gemini (free tier is heavily rate-limited). |

```bash
pip install openai google-genai
# pick a provider by exporting its key (put it in .env — git-ignored):
export LLAMA_API_KEY=...               # or OPENAI_API_KEY / GEMINI_API_KEY
# during batch:
python batch.py --folder <videos> --ai-eval               # auto-detect provider
python batch.py --folder <videos> --ai-eval --ai-provider llama
# or standalone on one result:
python evaluate.py --meta metadata/<video>.json --provider llama
```

Without a key it degrades gracefully (`ai_score=null`, explanatory
`ai_reasoning`) so the batch never breaks. The dashboard shows both the
**Average Math Score** and **AI Director Score** as summary cards, the AI
reasoning, and a per-frame quality chart.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> Saliency uses a self-contained spectral-residual implementation, so plain
> `opencv-python` is enough (no `opencv-contrib`). YOLOv8 weights
> (`yolov8n.pt`) download automatically on first run into `models/`.

## Run the cropper

```bash
# one clip
python cropper.py --video /path/to/clip.mp4

# first 5 clips in a folder
python cropper.py --folder /Users/refaelv/Documents/CV/smart_crop_videos --limit 5

# the rest, in the background (resumable — skips already-processed clips)
mkdir -p logs
nohup python batch.py --folder /Users/refaelv/Documents/CV/smart_crop_videos \
      --skip 5 > logs/batch.log 2>&1 &

# canonical smoke test — the fixed 10-video set in smoke_set.txt
python batch.py --folder /Users/refaelv/Documents/CV/smart_crop_videos \
      --list smoke_set.txt --force
```

Useful flags: `--alpha` (smoothing), `--proc-width` (flow/saliency resolution),
`--flow-stride` (compute flow every k-th frame, interpolate between),
`--detect-window` (trailing frames to detect on), `--anchor` (0..1 blend toward
per-keyframe saliency to correct flow drift; `0` = pure backward flow),
`--full` (detect every frame, slow/spec-faithful), `--no-render` (metadata only).

### Drift correction (`--anchor`)

Pure optical-flow integration from the terminal frame accumulates error over a
long clip, so the crop can end up off the subject in the early frames. Because
saliency is nearly free, it is computed at every flow keyframe and the backward
pass blends the flow-propagated centre with that keyframe's saliency-optimal
centre (`--anchor`, default `0.5`). This keeps the salient subject centred
throughout while flow preserves continuity. `--anchor 0` reproduces the strict
optical-flow-only backward pass from the spec.

### Performance

Detection (YOLO) and saliency are only *needed* for the terminal-frame decision
(the last `--detect-window` frames) and for the 6 storyboard overlays, so by
default they run on ~16 frames instead of every frame. Optical flow is sampled
every `--flow-stride` frames and the crop trajectory is interpolated across those
keyframes (EMA smooths the rest). This cuts a ~520-frame clip from **~185 s to
~10 s (~18×)** with a near-identical trajectory. Use `--full --flow-stride 1`
for the exhaustive per-frame version.

## Run the dashboard

```bash
streamlit run app.py
```

The dashboard shows, per video: a 6-frame storyboard (0/3/6/9/12/15 s) with the
green crop window, face/person boxes, and a red saliency heatmap; an interactive
frame-by-frame slider; the `X(t)` trajectory; and the rendered 9:16 video.

---

## Project layout

```
landscape2portrait/
├── cropper.py        # core algorithm + CLI
├── batch.py          # resumable folder/background runner
├── app.py            # Streamlit visualization dashboard
├── requirements.txt
├── metadata/         # <video>.json  (committed)
├── outputs/          # rendered 9:16 mp4 (git-ignored)
├── frames/           # cached storyboard frames (git-ignored)
└── models/           # yolov8n.pt (git-ignored, auto-downloaded)
```

## Design notes / trade-offs

- **Why anchor on the last frame and go backward?** Deciding each frame
  independently makes the crop jump between subjects. Anchoring once and tracking
  via flow yields a temporally coherent shot; the terminal frame is chosen
  because it best reflects where the action resolves.
- **Column-averaged flow** — since the crop only moves horizontally, we reduce
  the dense flow field to a per-column mean `dx`, which is compact and robust.
- **Spectral-residual saliency** avoids the `opencv-contrib` dependency while
  giving a reasonable importance map for the fallback case.
- **`stride`** lets you trade detection density for speed on long batches
  without changing the flow-based tracking.
