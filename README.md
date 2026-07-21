# landscape2portrait — turning wide videos into tall ones, intelligently

This project takes an ordinary 16:9 landscape video and turns it into a 9:16
portrait video, the shape you'd film for a phone. It does this purely by cropping:
it keeps the full height of the original frame and slides a tall, narrow window
left and right over time so that whatever matters most stays in view. The goal is
for the result to feel like it was shot in portrait to begin with — the window
should move the way a thoughtful camera operator would move, not jump around.

Because we always keep the full height, the only thing the software has to decide
is *where to place that tall window horizontally in each frame*. Everything below
is about making that one decision well, and about how we measured whether we were
actually making it well.

> **The fastest way to get a feel for it:** open one of the clips in
> `sample_outputs/`. The `_overlay` version shows the original wide video with the
> chosen crop drawn on top as a green box, so you can watch the decision being
> made. Then open the dashboard to browse all the results. If you want the full
> story of how we got here — the method, the trade-offs, the eight rounds of
> iteration — read [APPROACH.md](APPROACH.md).

## Getting it running

You'll need `ffmpeg` installed (it's what actually writes the video files); on a
Mac that's `brew install ffmpeg`. Then set up Python and run it:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# convert a single clip, or a whole folder of them:
python cropper.py --video /path/to/clip.mp4
python cropper.py --folder /path/to/videos

# then browse the results in your browser:
streamlit run app.py
```

The person-detection model downloads itself the first time you run it, so the
first clip is a little slower. Each converted video is written to `outputs/`: a
`_portrait` file (the finished 9:16 video) and a `_overlay` file (the original with
the crop box drawn on, which is handy for judging the result).

A couple of choices we made, following the brief: we assume each video is a single
continuous shot with no cuts, we always use the full frame height (no black bars),
and we drop the audio — the brief said audio was optional, so we left it out to
keep things simple.

## How it decides where to crop

The heart of the project is choosing, for every frame, where the tall window
should sit. We don't decide each frame in isolation — doing that makes the crop
twitch and hop between things. Instead we look at the whole clip, figure out what
kind of shot it is, and pick one consistent behaviour. There are three cases, and
we check them in order.

**First: is the camera flying forward?** In drone shots that fly down a road or
toward a building, everything in the frame streams outward from a single point on
the horizon — the point you're heading toward. Humans naturally want to look at
that point, so for these shots we simply keep the crop centred on it. We find that
point by looking at how the image flows between frames and locating where the
motion comes from, and we smooth that estimate heavily so the crop doesn't wobble.

**Second: is there one clear person to follow?** We run a person detector on the
video, and if it consistently finds a single person in roughly the same place
across the clip (one skateboarder, one kayaker, one person walking), we lock the
crop onto them and let it follow them the whole way through. The important detail
here is *consistency* rather than the detector's confidence: someone lying down or
filmed from behind scores low confidence but is clearly the subject, whereas a
beach full of scattered swimmers scores higher confidence but has no single
subject to follow — so we decide based on how steady the detected position is, not
how sure the detector feels.

**Third: everything else.** For the remaining shots — a fixed monument, a slow pan
revealing a street, a scene where the interesting thing changes over time — we do
something a little clever. We first decide what to frame in the *last* frame of the
clip (preferring faces if we see them, then people, then just the most
eye-catching region), and then we walk *backwards* through the video, using the
motion between frames to keep that same content in view all the way back to the
start. This gives a crop that locks onto whatever the shot resolves to and holds
it steadily, instead of second-guessing itself frame by frame. A gentle pull
toward the eye-catching parts of each frame keeps it from slowly drifting off
course.

Whichever case applies, we then smooth the resulting motion — a median filter to
throw out sudden jumps, followed by a light running average so the movement glides
— and clamp it so the window never runs off the edge of the frame. Finally, when
we write the video out, we extract the crop at sub-pixel precision (which removes a
subtle one-pixel shimmer you'd otherwise get) and run a light stabilization pass
that cancels any leftover camera shake, but only on clips that actually need it so
we never make a steady shot worse.

To keep it fast, we don't run the heavy detectors on every single frame — we only
need them at the end of the clip and at a handful of sampled frames — and we
measure the motion every few frames rather than all of them. That takes a typical
clip from a few minutes down to about ten seconds, with essentially the same
result.

## How we knew it was getting better

The trickiest part of a task like this is that there's no single right answer, so
"looks good" isn't enough to iterate on. We built three ways to score every result
so we could see, across all 118 videos and every version of the algorithm, whether
a change actually helped:

- **A coverage score** that asks how much of the important stuff — the people,
  faces, and eye-catching areas — the crop actually kept in frame. You can never
  keep everything, so the absolute number isn't the point; but the *lowest* scores
  reliably point straight at the genuinely broken crops, which made finding bugs
  fast, and the trend told us whether a change was an improvement.
- **A steadiness grade (A–F)** that looks at how much the crop swings from side to
  side. A crop can be perfectly on-subject and still be unpleasant to watch if it
  jitters, so we treat smoothness as its own goal.
- **An "AI director" score**, where we hand a vision model a storyboard of the clip
  with the crop drawn on and ask it to rate the result one to five, the way a
  reviewer might. It catches big-picture problems the math misses (and, tellingly,
  it does *not* catch fine jitter — which is exactly why we also have the steadiness
  grade).

The Streamlit dashboard ties these together. Its overview page is the bird's-eye
view — a table of every clip's grades, and charts that put each version on the same
axes so you can instantly see whether a change lifted everything or quietly broke a
few specific clips. Its per-video page is for digging in: a storyboard with all the
overlays, a frame slider, a plot of where the crop moved, and the overlay video.
That combination is what let us turn "this one looks off" into a precise diagnosis
and a fix.

## Results

Here's how the scores moved across the eight versions, measured on all 118 videos:

| version | coverage % | steadiness | grade A / F clips | the main change that version |
|---|:--:|:--:|:--:|---|
| v4 | 58.9 | 75.6 | 67 / 15 | the baseline: pick the ending, track backwards, nudge toward salient regions |
| v5 | 58.6 | 81.2 | 82 / 13 | follow the heading point on drone shots; smoother motion |
| v6 | 58.8 | 91.5 | 98 / 2 | a much stronger de-jitter step — the biggest single steadiness win |
| v7 | 58.3 | 92.0 | 98 / 1 | sub-pixel cropping + shake removal killed the last of the tiny jitters |
| v8 | **64.6** | 90.3 | 92 / 3 | follow a clear person through the whole clip — keeps people in frame |

The pattern tells the story: through v6 we mostly made the motion steadier, and
then in v8 we spent a little of that steadiness to actually keep people in frame,
which is why the coverage number jumps. In the final version, following a person
drives about 60% of the clips, the heading-point behaviour handles the drone shots,
and the rest use the pick-the-ending-and-track-back approach.

## Where it falls short

Being honest about the weak spots:

- It's more elaborate than the task strictly needed. It's a stack of rules with
  hand-tuned thresholds rather than one clean learned model, and every threshold is
  a place it can be wrong.
- The face detector occasionally sees "faces" in textures like foliage or a
  honeycomb. We work around this by preferring the more reliable person detector,
  but it's a patch, not a cure.
- It has no real sense of identity, so shots where the subject genuinely changes
  over time are handled by a heuristic rather than a proper "who matters now"
  decision.
- The steadiness grade actually punishes correctly following a moving person (a
  moving subject means a moving crop), so it has to be read alongside the coverage
  score, never on its own.
- It only moves the crop left and right, assumes a single shot, and the coverage
  score is a useful proxy rather than real ground truth.

There's a fuller discussion, along with what we'd do given more time, in
[APPROACH.md](APPROACH.md).

## What's in the repo

```
cropper.py        the core: turns a video into a 9:16 video (the crop logic lives here)
batch.py          runs the whole folder, resumable, with a fixed smoke-test option
evaluate.py       the "AI director" scoring, via Gemini / OpenAI / Llama
app.py            the Streamlit dashboard (overview + per-video)
build_history.py  rebuilds the version-comparison data the dashboard shows
smoke_set.txt     the fixed ten clips we iterated on (two per category)
APPROACH.md       the full write-up: method, scoring, the eight rounds, limitations
sample_outputs/   ten finished results you can watch right away
metadata/         one JSON per clip with everything we computed (committed)
history/          the version-by-version scores for the dashboard
outputs/ frames/ models/   generated videos, cached frames, model weights (not committed)
```
