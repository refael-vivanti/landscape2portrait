# How we approached this

This is the longer story behind the project: how we thought about the problem, how
we decided whether we were doing well, and how the algorithm grew over eight
rounds of iteration.

The brief said to expect a few hours of work. We spent the first hour on a working
cropper, and most of the rest on something less obvious but more useful: a way to
*measure* the results. When there's no single correct answer, "it looks good" is a
weak thing to iterate on. So we built a small evaluation setup — three ways of
scoring each result, plus a dashboard to explore them — and that's the part we'd
most want a reviewer to look at, because it's what turned every later change from a
guess into a measured step forward.

## The problem, plainly

We're given a wide (16:9) video and we have to produce a tall (9:16) one, using the
full height and no black bars. That means the only real freedom we have is *where
to put the tall window from left to right*, and we have to choose that for every
single frame. Two things make a crop good: it keeps the right thing in view, and it
moves smoothly, like a real camera operator rather than a nervous one. Those two
ideas became the two numbers we track — coverage and steadiness — with a third,
more holistic "director's opinion" on top.

## Step one: understand the footage

Before writing much cropping logic, we sorted the test videos by the *kind of crop
they need*. Five distinct behaviours came out of that, and this grouping shaped
everything afterwards. For each one we kept two representative clips as examples:

| # | The kind of shot | What a good crop has to do | Two examples |
|---|---|---|---|
| 1 | **Hold something in the centre** | keep a centred subject in view while the background moves around it | `10398657` road seen from above, `13965282` a monument the drone circles |
| 2 | **Hold something off-centre** | keep an off-centre subject (a nicer composition) steady through a tilt | `14627395` a coastal cliff, `12142075` a honeycomb covered in bees |
| 3 | **Follow one moving subject** | pan the window sideways to chase a single subject | `11353413` a red kayak, `16436843` a skateboarder doing a trick |
| 4 | **Ride a pan and reveal** | move with the camera and re-centre on whatever comes into view | `10388072` a night-time city pan, `14369070` an indoor soccer game |
| 5 | **Change subject over time** | hand off from one thing to another as the story develops | `11967303` a woman, then the view she's looking at; `16577316` kite-surfers coming and going |

We then picked ten clips — the two examples from each group — as a fixed
"smoke set" that we could re-score in about a minute. That let us iterate quickly,
while the full set of 118 videos was the slower, honest check we ran before
trusting a change. Keeping the ten clips fixed was deliberate: it meant the numbers
from one version were directly comparable to the next.

## Step two: decide how to measure quality

We score every result three ways, because each one catches something the others
miss.

The first is a **coverage score**. For each frame we ask a simple question: of the
things worth keeping — the people and faces we detected, and the visually
eye-catching regions — how much ended up inside the crop? We weight people and
faces more heavily than raw eye-catching-ness. You can never keep everything (that's
the whole point of cropping), so the absolute value isn't meaningful. But it turned
out to be a fantastic bug-finder: the clips with the *lowest* coverage were almost
always genuinely broken crops, so we could sort by it and go straight to the
problems. And watching the average move told us whether a change was really an
improvement.

The second is a **steadiness grade**, from A to F, based purely on how the crop
moved. It counts big side-to-side swings and general busyness. We added this because
a crop can be perfectly on-subject and still be unpleasant — a shaky or restless
window is bad to watch even when it's technically "correct." Smoothness deserves to
be a goal in its own right.

The third is an **AI director's opinion**. We assemble a storyboard of sixteen
frames from the clip, draw the chosen crop on each one, and hand it to a
vision-language model with a simple instruction: rate this one to five on whether it
kept the subject and whether the movement feels natural, and say why in a sentence.
It's like having a patient senior reviewer look at every result. Interestingly, it
does *not* reliably notice fine jitter — which is a nice illustration of why we keep
the steadiness grade separate rather than trusting any single judge.

## Step three: the dashboard

All of this lives in a Streamlit app with two views. The **overview** is the
bird's-eye picture: a table of every clip's grades, the overall averages, and charts
that plot each version of the algorithm on the same axes across all the clips. That
last part is what made progress legible — if a version's line sits above the others
everywhere, it helped across the board; if it dips at one point, that's a specific
clip that got worse and needs a look. The **per-video** view is for exactly that
kind of digging: it shows the storyboard with all the overlays, a slider to scrub
frame by frame, a plot of where the crop moved over time, and the overlay video. It
was this drill-down that turned vague impressions into precise diagnoses — for
example, seeing that on one clip the software had correctly decided to frame the
woman at the end but the backward tracking had drifted away from her everywhere
else.

## The algorithm we ended up with

For every frame we gather four cheap signals: where people are (from an off-the-shelf
person detector), where faces are (from a classic face detector, which we use to
*encourage* the crop to include faces when it reasonably can), how eye-catching each
part of the frame is (a lightweight "saliency" estimate that highlights bright,
high-contrast, or unusual regions), and how the image is moving between frames (an
optical-flow estimate, i.e. which way each patch of the picture is sliding). To keep
it fast we only run the detectors where we actually need them and we measure motion
every few frames rather than all of them.

Then, for each clip, we choose one of three behaviours, checked in this order.

**If the camera is flying forward,** we keep the crop on the point it's heading
toward. In a shot that flies down a road or toward a building, the whole picture
appears to stream outward from a single spot on the horizon; that spot is where a
viewer naturally wants to look. We locate it from the motion field and smooth it
heavily over about a second so the crop settles on the heading instead of twitching.

**If there's one clear person to follow,** we lock onto them for the whole clip.
When the person detector keeps finding a single person in roughly the same place
throughout — a skateboarder, a kayaker, someone walking — we anchor the crop to them
and let it track them from start to finish. The key judgement here is to trust
*consistency of position* rather than the detector's confidence. Someone lying down
or seen from behind gets low confidence but is unmistakably the subject; a beach
dotted with distant swimmers gets higher confidence but has no single person to
follow. So we decide based on how steady the detected position is, which cleanly
separates "one subject to follow" from "a scattered crowd."

**Otherwise,** we use a small trick that handles fixed shots, slow pans, and shots
where the subject changes. We first decide what to frame in the *final* frame —
preferring a face if there's a dependable one, then a person, and otherwise just the
most eye-catching region — and then we walk *backwards* through the clip, using the
frame-to-frame motion to keep that same content in view all the way to the
beginning. The effect is that the crop locks onto whatever the shot ultimately
settles on and holds it, rather than nervously re-deciding every frame. Because pure
backward tracking slowly drifts over a long clip, we add a gentle pull toward the
eye-catching parts of each frame to keep it honest.

Whatever behaviour we chose, we finish by smoothing the movement: a median filter to
discard sudden jumps, then a light running average so the motion glides, and a clamp
so the window never leaves the frame. When we render, we cut the crop at sub-pixel
precision — extracting it with a proper resample rather than snapping to whole
pixels, which removes a faint one-pixel shimmer — and we run a light shake-removal
pass, but only on clips that are actually shaky, so we never add noise to a shot
that was already steady.

## The eight rounds, and what each taught us

Every version was a response to something the scores or the dashboard showed us.

| version | what we saw | what we changed | the effect |
|---|---|---|---|
| v2 | one stray face detection on a single frame could hijack the whole crop, and ties in "where's the most content" hugged the left edge | only trust a detection if it shows up across several frames; break ties toward the centre | fixed a badly mis-framed clip |
| v3 | we were running heavy detection on every frame for no reason | detect only where it matters and measure motion every few frames | about 18× faster, same result |
| v4 | pure backward tracking slowly drifts off the subject on long clips | add the gentle pull toward eye-catching regions | crops stay on target; made the full run practical |
| v5 | drone shots have no "subject" but do have a natural target | follow the heading point; smoother motion; added the storyboard and overlay views | steadiness 75.6 → 81.2 |
| v6 | most of the remaining unsteadiness was the crop hopping between similar regions | a much stronger de-jitter step (a longer median window) | steadiness 81 → 91.5, and F-grade clips dropped from 13 to 2 — the biggest single win |
| v7 | a faint leftover jitter turned out to be pixel-snapping, and the heading estimate was noisy | cut the crop at sub-pixel precision, add adaptive shake removal, smooth the heading over a full second | the tiny jitters essentially disappeared |
| v8 | people and faces only influenced the *ending*, so the crop drifted off them earlier in the clip | follow a clear person across the whole clip, no matter which signal "won" | coverage 58.9 → 64.6 — people actually stay in frame now |

If we had to boil the whole journey down to a handful of lessons: don't let a
single noisy frame make a big decision; when you have to pick between equally good
options, pick the centred one; you only need expensive analysis at a few frames, not
all of them; tracking purely by motion drifts, so give it something to hold onto;
the natural target of a flythrough is the point it's heading toward; the single
biggest lever for smoothness was simply throwing out sudden jumps more aggressively;
most "jitter" complaints are actually sub-pixel rounding you can fix at render time;
lean on the reliable person detector rather than the twitchy face detector; and,
most importantly, a subject should guide the *whole* shot, not just its ending.

For two of these rounds we let the process run on its own overnight, with a simple
rule: find the worst-scoring clip, work out why it's bad from the dashboard's data,
try a targeted fix, re-score the smoke set, and keep the change only if the average
improved. That's how the big smoothness win and several of the regression fixes were
found — and it was only possible *because* the scoring existed to gate the changes.

## Where it falls short

We'd rather name the weaknesses than hide them.

The biggest one is that it's simply more elaborate than the task needed. It's a
collection of rules with hand-tuned thresholds — when to call something a flythrough,
how much to smooth, how consistent a person has to be — rather than one clean model
that learned the right behaviour. Every one of those thresholds is a place it can be
wrong on footage we didn't test.

The face detector is the classic weak component: it occasionally "sees" faces in
textures like leaves or a honeycomb. We mostly sidestep this by leaning on the
sturdier person detector, but that's a workaround, not a fix.

It also has no real sense of who's who over time, so the "subject changes mid-shot"
case is handled by a heuristic rather than a genuine decision about which subject
matters at each moment. Relatedly, the steadiness grade *penalises* correctly
following a moving person — if the subject moves, the crop has to move — so that
grade only makes sense read together with the coverage score, never alone.

Finally, it only moves the crop horizontally, it assumes each video is one
continuous shot, and the coverage score is a helpful stand-in for quality rather
than real ground truth.

## What we'd do with more time

The most valuable next step would be to replace the pile of heuristics with a small
learned model that scores possible crop positions directly — we could even train it
on the "AI director" judgements we're already collecting. Alongside that, a proper
object tracker would let us follow subjects reliably and make principled decisions
about when to hand off from one to another, which is the honest way to handle the
"subject changes over time" shots.

We'd also like to stop choosing the crop frame by frame and instead solve for the
whole path at once — balancing "keep the subject in view" against "don't move too
much" over the entire clip. That kind of global optimisation would make the smooth,
purposeful "camera operator" feel come out naturally instead of being bolted on with
smoothing.

Beyond that: a small set of human-drawn "ideal" crops would turn our coverage score
into a real accuracy measure and let us A/B test properly; detecting scene cuts would
remove the single-shot assumption; and there's obvious room to add audio, to reframe
vertically as well as horizontally, and to make a real-time version.
