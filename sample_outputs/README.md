# Sample outputs

These are ten finished results you can watch without running anything — two from
each of the five kinds of shot we identified, so you can see how the different
behaviours look in practice. To generate the rest, run
`python batch.py --folder /path/to/videos`.

For each clip there are two files. The `_portrait` one is the actual deliverable:
the video converted to tall 9:16. The `_overlay` one is the original wide video
with the chosen crop drawn on top as a green box (plus a small marker on drone
shots showing the point the camera is heading toward). The overlay is the clearest
way to *see the decision* — watch where the box goes and why. We've compressed the
overlays a little to keep the repository small; a fresh render produces them at full
resolution.

| The kind of shot | Clip | What to watch for |
|---|---|---|
| **Hold something centred** | `10398657` | the road stays centred on the point the drone flies toward |
| | `13965282` | the monument stays centred while the drone circles it |
| **Hold something off-centre** | `14627395` | the cliff is held off to one side through the tilt |
| | `12142075` | the bee-covered comb stays framed; the blurred background is dropped |
| **Follow one moving subject** | `11353413` | the crop slides along to follow the red kayak |
| | `16436843` | it tracks the skateboarder through the whole trick |
| **Ride a pan and reveal** | `10388072` | it moves with the night-city pan, settling on the lit storefronts |
| | `14369070` | it follows the pan toward wherever the players are densest |
| **Change subject over time** | `11967303` | it starts on the woman, then hands off to the view she's looking at |
| | `16577316` | it shifts between the kite-surfers as they come and go |
