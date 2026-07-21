# Sample outputs

Ten pre-rendered results — **two per crop-strategy category** — so you can watch
the algorithm's behaviour without running anything. Regenerate all 118 with
`python batch.py --folder /path/to/videos`.

For each clip:
- `<id>_portrait.mp4` — the **9:16 deliverable** (the converted video).
- `<id>_overlay.mp4` — the **original 16:9 with the moving green crop box** (and a
  magenta heading marker on forward-motion clips). This is the clearest way to see
  *what the algorithm decided*. (Downscaled/compressed here to keep the repo light;
  full-res versions come out of a fresh render.)

| Category (crop strategy) | Clip | What to look for |
|---|---|---|
| **1. Fixed center anchor** | `10398657` | aerial road held centered on the vanishing point |
| | `13965282` | Victory Column stays centered while the drone orbits |
| **2. Fixed off-center anchor** | `14627395` | coastal cliff held off-center (rule-of-thirds) during the tilt |
| | `12142075` | bee-covered honeycomb kept slightly right; blurred background dropped |
| **3. Dynamic subject follow** | `11353413` | crop pans to follow the red kayak across the beach |
| | `16436843` | tracks the skateboarder through the trick |
| **4. Pan/reveal follow** | `10388072` | rides the night-city pan, re-centering on the lit facades |
| | `14369070` | follows the indoor-soccer pan toward the densest player cluster |
| **5. Narrative handoff / priority switch** | `11967303` | woman first, then hands off to the overlook/water |
| | `16577316` | shifts between kite/person action clusters as they appear |
