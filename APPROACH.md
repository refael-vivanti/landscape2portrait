# Approach & Methodology

How I turned "crop 16:9 → 9:16" into a reframing that mimics a native portrait
capture — and, more importantly, **how I knew whether it was getting better.**

The brief expected a few hours. I spent the first hour on the algorithm and the
rest building an **evaluation harness** so I could iterate with evidence instead
of vibes. That harness — three complementary scores + a dashboard — is the part
I'd most want reviewed, because it's what let 8 versions each be a measured step
rather than a guess.

---

## 1. Problem framing

- Output is 9:16, **full height**, no letterboxing → the *only* free parameter is
  the horizontal crop centre `X(t)`, one value per frame.
- Assumptions (given): single continuous shot (no cuts), 16:9 in. Audio optional
  → omitted.
- "No single correct solution" → success = the crop keeps the *right* thing in
  frame and *moves like an operator* (smooth, purposeful), not like a jitter bot.

Those two goals — **keep the subject** and **move smoothly** — became my two
numeric KPIs (coverage + stability), plus a VLM "director" for the gestalt.

---

## 2. Method: evaluation-first

### 2.1 Categorise the footage
I first asked an assistant to cluster the test videos into the distinct *cropping
strategies* they demand. Five emerged — this taxonomy drove everything:

| # | Crop strategy | What it needs | Smoke reps |
|---|---------------|---------------|-----------|
| 1 | **Fixed center anchor** | hold a centered subject while the background moves | `10398657` aerial road, `13965282` Victory Column orbit |
| 2 | **Fixed off-center anchor** | hold an off-center subject (rule-of-thirds) | `14627395` coastal cliff, `12142075` honeycomb + bees |
| 3 | **Dynamic subject follow** | pan the window to follow one moving subject | `11353413` red kayak, `16436843` skateboard trick |
| 4 | **Pan/reveal follow** | ride a camera pan, re-centering on what's revealed | `10388072` night city pan, `14369070` indoor soccer |
| 5 | **Narrative handoff / priority switch** | change subject over time | `11967303` woman → overlook, `16577316` aerial kite beach |

### 2.2 A fixed smoke set (`smoke_set.txt`)
Ten clips — **two per category** — became the fast iteration loop (~1 min to
re-score all ten). The full 118 clips are the validation set (~35–80 min to
render). Keeping the smoke set *fixed* means version-to-version numbers are
comparable.

### 2.3 Three KPIs (the differentiator)

Each catches what the others miss:

1. **Math (coverage) grade** — per frame:
   `0.7 · object_coverage + 0.3 · saliency_coverage`, where object_coverage is the
   fraction of detected person/face box area kept inside the crop (1.0 when nothing
   is detected) and saliency_coverage is the salient mass inside the crop over the
   total. *Why:* absolute value isn't meaningful (you can't keep everything), but
   the **lowest-scoring clips are reliably the broken crops**, and the **trend**
   shows whether a change helped. This is what made bug-hunting fast.
2. **Stability grade (A–F)** — from the `X(t)` trajectory: counts large
   side-to-side swings + total pan per second. *Why:* a crop can be perfectly
   on-subject and still be nauseating; smoothness is a first-class UX property.
3. **AI Director score (VLM-as-judge)** — a vision LLM sees the 16-frame
   storyboard (with the crop boxes drawn) and rates 1–5 on subject retention +
   temporal flow, with a one-line rationale. *Why:* a cheap "senior reviewer"
   opinion that catches gestalt problems the math misses. Notably it does **not**
   reliably catch fine jitter — which is exactly why the stability grade exists.

### 2.4 The dashboard (`app.py`)
- **Overview (bird's-eye):** grades table for all clips, aggregate stats, and
  **per-metric line charts with one line per version across all 118 clips** — a
  version sitting above the others = global improvement; a dip at some x = a
  per-video regression to investigate. Plus a v4→v8 comparison table.
- **Per-video (root-cause):** 16-frame storyboard with crop box + face/person
  boxes + saliency heatmap + heading marker; a frame slider; the `X(t)` trajectory
  (raw vs smoothed); the landscape overlay video. This is where I diagnosed each
  bad clip.

This BEV-plus-drilldown is what turned "it looks off" into "clip 5127498, source
= people, terminal x = 553 but rendered x = 202 → the backward pass drifted".

---

## 3. The algorithm (final, v8)

The full decision tree is in the [README](README.md#the-decision-tree-v8); here is
the rationale for each branch and how it serves the five categories.

**Per-frame signals** (one decode pass): YOLOv8n `person`, Haar faces
(frontal+profile), spectral-residual saliency (core OpenCV — no contrib needed),
dense Farneback flow. Detectors run only on the trailing 10 + 16 storyboard
frames; flow at stride-5 keyframes (interpolated) → ~18× faster than per-frame
with a near-identical trajectory.

**Target selection (priority order):**

1. **Forward motion → epipole.** For a flythrough the flow diverges from the
   *focus of expansion* (FOE) — the point we're heading toward. It's the
   neg→pos zero-crossing of the column-averaged horizontal flow; estimated over a
   **~1s baseline** and **EMA-smoothed** (`0.9·last + 0.1·new`) for a steady
   heading. Serves **category 1** (aerial road) and drone reveals.
2. **Strong, consistent person → track the person.** If a YOLO person is present
   in ≥40% of detected frames with low horizontal spread (x-std < 0.20·W), anchor
   the *whole* trajectory to the person's interpolated, median-damped centre.
   Serves **category 3** (kayak, skateboarder) and any single clear subject
   (`5127498`, `4114693`, `4385153`). Gated on **spatial consistency, not
   confidence** — a lying-down subject scores low confidence but is very
   consistent, whereas scattered swimmers score higher confidence but jump around.
3. **Else → terminal decision + backward flow.** Pick the last frame's centre by
   `faces > people(if not weak) > saliency(motion-blended)`, then walk backward in
   time via optical flow (pixel continuity) with a per-keyframe saliency anchor to
   correct drift. Serves **categories 1/2/4/5** — fixed anchors, pans, and
   handoffs, where "lock onto where the action resolves and hold it" is the right
   behaviour.

**Smoothing:** temporal median over 19 keyframes (kills erratic target jumps) →
forward EMA (α=0.15) → clamp to frame bounds.

**Render:** sub-pixel crop (an affine warp, not an integer slice — removes the
~1px per-frame quantization shimmer) + an **adaptive** feature-tracking
stabilization pass (goodFeaturesToTrack + LK → similarity transform → long-term
trajectory smoothing → warp), skipped when the source is already steady so it
never adds resample noise. H.264 / yuv420p / faststart so it plays anywhere.

---

## 4. The 8-version insight log

Each version was a measured response to what the KPIs/dashboard surfaced.

| ver | Root-cause insight | Change | Impact |
|-----|--------------------|--------|--------|
| v2  | A single-frame Haar false-positive can hijack the whole crop; argmax ties hug the left edge | Require a detector to fire in ≥30% of the window; center-not-edge tie-break; web-playable H.264 | fixed popcorn-clip mis-crop |
| v3  | Detection/saliency are only *needed* at the terminal + storyboard frames | Detect there only; flow at a stride + interpolate | **~18× faster** (185 s → 10 s / clip) |
| v4  | Pure backward optical-flow integration **drifts** over long clips | Anchor each keyframe to its saliency-optimal centre | crop stays on subject; enables full 118 run |
| v5  | Forward/drone clips have no "subject" but a natural target | Track the **epipole**; median stabilisation; storyboard + overlay viz | stability 75.6 → 81.2 |
| v6  | Erratic saliency-target jumps are the dominant instability; motion-in-anchor chases waves | **Median window 5 → 19**; motion-saliency terminal-only; stricter weak-people | **stability 81 → 91.5**, F-clips 13 → 2 |
| v7  | Residual "small jitter" is integer-crop quantization; epipole estimate is noisy | **Sub-pixel crop**; adaptive feature-tracking stabilization; epipole 1s-baseline + EMA | jitter ~1px → ~0.08px; fewer false-forwards |
| v8  | People/faces only set the *terminal* frame, so the crop drifts off the subject elsewhere; Haar faces unreliable | **Track a consistent YOLO person across the whole clip, regardless of terminal source**; gate on spatial consistency | **coverage 58.9 → 64.6**; fixed 5127498/4114693/4385153 |

Distilled insights:
1. Single-frame detector false positives must be gated by **persistence**.
2. Tie-breaks must **center** the subject, not hug an edge.
3. You only need heavy detection at a few frames → huge speedup, same result.
4. Backward optical-flow **drifts** → correct it with an anchor.
5. A **flythrough's** natural target is the **epipole**; use a long baseline + EMA.
6. The biggest stability lever was a **long temporal median**, not fancier logic.
7. **Motion** in the per-keyframe anchor chases background (waves); keep motion for
   *choosing* the target, not for the smooth path.
8. Most "jitter" complaints are **sub-pixel quantization** — fix at render time.
9. Haar faces are unreliable on texture; prefer the **reliable YOLO person**.
10. A subject must drive the **whole trajectory**, not just the terminal frame.
11. Gate person-tracking on **spatial consistency, not confidence**.

### Autonomous overnight iteration
For two of the versions I let the agent run unattended with a simple loop:
*find the lowest-graded clip → diagnose the root cause on the dashboard data →
apply a targeted fix → re-score the smoke set → keep only if the mean improved.*
This is exactly how v6's median-window win and several regression fixes were
found — the eval harness made autonomous, evidence-gated iteration possible.

---

## 5. Limitations / drawbacks (honest)

- **Cumbersome and over-engineered** for the brief. It's a pipeline of heuristics
  with hand-tuned thresholds (`FWD_MIN_FRAC`, `MEDIAN_K`, anchor/motion weights,
  the 0.20·W consistency gate), not one principled model. Each threshold is a place
  it can be wrong.
- **Haar face false positives** on textured scenes (bees, foliage); worked around
  rather than solved.
- **No identity-consistent multi-object tracking.** Category 5 (narrative handoff)
  is handled by the terminal-decision heuristic, not a real "which subject now"
  policy.
- **The stability grade penalises correct subject-following** (moving subject →
  moving crop). It must be read *with* the coverage grade, not alone — a single
  headline number would be misleading.
- **Greedy, causal-ish trajectory.** Median+EMA smooth it but it isn't a globally
  optimal path; a true optimum would trade coverage vs. motion over the whole clip.
- **Horizontal only**; **single-shot assumption** (no cut detection); **no audio**.
- **Metadata stores absolute source paths**, so the dashboard's *live* frame
  inspector needs the originals (the storyboard + sample outputs are self-contained).
- The **math grade is a proxy**, not ground truth — good for catching bad crops and
  tracking trend, not for claiming absolute quality.

## 6. What I'd do with more time

- **Replace heuristics with a learned model**: a lightweight saliency/subject-of-
  interest network (or a small model distilled from the VLM judge) to score
  candidate crop columns directly.
- **Proper multi-object tracker** (e.g. ByteTrack) for identity-consistent follow
  and principled narrative handoff (switch when a higher-priority track appears).
- **Global trajectory optimization**: solve `X(t)` as an optimization
  (coverage reward − motion/jerk penalty − out-of-bounds) over the whole clip,
  instead of greedy + smoothing — the "operator" behaviour would come out naturally.
- **A small labelled ground-truth set** (human-drawn ideal crops) to turn the math
  grade into a real accuracy metric and enable proper A/B testing.
- **Scene-cut detection** to drop the single-shot assumption; **audio muxing**;
  **2D framing** (vertical too); **real-time / streaming** version.
- Consolidate the threshold zoo into a few documented, cross-validated parameters.
