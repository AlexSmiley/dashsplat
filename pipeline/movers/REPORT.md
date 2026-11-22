# Moving objects: what worked on the drive

**17 vehicle episodes in each scene.** Four at 80–140 s, four at 190–215 s and nine at 286–330 s. The bus, crossing SUV and black coupe carry photographed surfaces across camera handoffs. The black coupe gives the clearest chase-view result. No moving person was confirmed. These are selected episodes, not exhaustive traffic counts; the two black-coupe episodes may be the same physical vehicle, separated by more than 20 s without a credible continuous handoff.

## Compare the reconstruction with the photographs

Each projection plate has a row per associated camera: **magenta = source mask, yellow = fitted footprint, green = fitted box**. The overhead plot distinguishes released support from motion prediction. Model plates show the exported observed and inferred PLYs together at eight azimuths and three drive times. The first plate row is not the final model revealed early.

| Episode | Real camera comparison (live / hindsight) | Overhead track | Eight angles × three moments |
|---|---|---|---|
| Crossing SUV | [live](report/live-crossing-suv-projection.jpg) / [hindsight](report/stitched-crossing-suv-projection.jpg) | [track](report/live-crossing-suv-track.jpg) | [model](report/live-crossing-suv-model.jpg) |
| Passing bus | [live](report/live-passing-bus-projection.jpg) / [hindsight](report/stitched-passing-bus-projection.jpg) | [track](report/live-passing-bus-track.jpg) | [model](report/live-passing-bus-model.jpg) |
| Junction crossover | [live](report/live-junction-silver-crossover-projection.jpg) / [hindsight](report/stitched-junction-silver-crossover-projection.jpg) | [track](report/live-junction-silver-crossover-track.jpg) | [model](report/live-junction-silver-crossover-model.jpg) |
| Turning black coupe | [live](report/live-turning-black-coupe-projection.jpg) / [hindsight](report/stitched-turning-black-coupe-projection.jpg) | [track](report/live-turning-black-coupe-track.jpg) | [model](report/live-turning-black-coupe-model.jpg) |

The SUV contributes front and right-pillar views; the bus contributes front, left pillar, left repeater and back (the rear masker calls it a car). Coarse exterior shape and bus livery read from several angles; wheels, window boundaries and paint smear. Some bus colour samples contain tree/pole occlusion. Roofs and undersides are largely class fill. Hindsight gives a better silhouette fit but can hit the size-prior bound on the crossing SUV. There is no independent metric 3D box or surface ground truth.

## Projection errors and timed misses

Live / hindsight pairs at the 724 × 469 mask grid. The centre error measures distance between projected and mask-box centres; IoU compares their areas. Medians below use held masks **only while a track exists**. The photograph count is the number accepted by the surface bake, not the number of frames propagated with LK.

| Episode | Projectable held masks | Median centre error (px) | Median box IoU | Accepted photographs |
|---|---:|---:|---:|---:|
| crossing-suv | 3 / 4 | 13.7 / 4.7 | 0.35 / 0.75 | 17 / 18 |
| passing-bus | 26 / 26 | 5.3 / 2.9 | 0.73 / 0.81 | 55 / 57 |
| junction-silver-crossover | 3 / 3 | 7.0 / 2.5 | 0.62 / 0.77 | 28 / 29 |
| turning-black-coupe | 12 / 13 | 6.5 / 4.8 | 0.85 / 0.86 | 53 / 54 |
| oncoming-grey-suv | 3 / 4 | 13.4 / 1.2 | 0.65 / 0.89 | 17 / 18 |

| Window | Held masks | Live: no track / IoU < 0.5 | Hindsight: no track / IoU < 0.5 |
|---|---:|---:|---:|
| 80 s window | 66 | 5 / 21 | 0 / 5 |
| 190 s window | 11 | 0 / 5 | 0 / 0 |
| 286 s window | 59 | 17 / 8 | 0 / 4 |

Of 136 held mask exposures, live had no track for 22 and an IoU below 0.5 on 34 more. Hindsight missed 0 temporally and 9 spatially. These counts measure alignment on **retained identities**, not discovery recall. Nine-frame mask withholding still permits independent LK on the same photographic exposure. Held masks informed the shorter-history filter comparison for the two anchors, grey SUV and coupe; their final numbers are not an untouched test set. The depth cross-check has 553 mask/depth overlaps among 1,111 eligible training masks, not an independent depth truth. [All timings, errors and held counts](report/measurements.json) · [live surface counts](report/live-models.json) · [hindsight counts](report/stitched-models.json).

The live SUV and bus still have occasional large pose corrections: their largest single 36 Hz horizontal steps were 2.21 m and 3.75 m. Those peaks occur at 103.64 s and 134.31 s respectively. The rate is continuous, but these are estimator corrections, not plausible instantaneous road speeds. The HUD/scene checks cover [all 68 referenced PLYs and both tile caches](report/schema-check.json).

## Web viewer

Open the [live crossing](http://10.0.50.51:8090/?scene=live-v5-movers-v1&t=102.85&paused=1&stats=1), [live coupe](http://10.0.50.51:8090/?scene=live-v5-movers-v1&t=319.05&paused=1&stats=1&orbit=43,5,1.1), or [hindsight coupe](http://10.0.50.51:8090/?scene=stitched-opt-v2-movers-v1&t=318.8&paused=1&stats=1&orbit=40,4,1.1). Use ordinary playback. [Decoded clip contact sheet](report/viewer-motion.jpg). The HUD mover numerator includes prefetched nearby tracks and is not a count of objects visible on screen.

| Playback | Still / recording | What the rendered image establishes |
|---|---|---|
| Crossing SUV | [live still](report/viewer-suv.jpg) · [clip](report/viewer-suv.mp4) · [hindsight clip](report/viewer-offline-suv.mp4) | Small white vehicle crossing near the far bend. It is much less legible than its real-camera crop; roadside splats hide it during part of the handoff. |
| Bus | [live still](report/viewer-bus.jpg) · [clip](report/viewer-bus.mp4) · [hindsight clip](report/viewer-offline-bus-alt.mp4) | The live orbit shows a teal bus passing behind the near roadside reconstruction. The hindsight greenery nearly hides it, although its camera overlays and object model are present. |
| Junction | [live still](report/viewer-junction.jpg) · [clip](report/viewer-junction.mp4) · [hindsight still](report/viewer-offline-junction.jpg) | Two cars move apart ahead of the ego vehicle; the cross street lacks continuous visible ground, so they sometimes appear to float. |
| Late coupe | [live still](report/viewer-coupe.jpg) · [clip](report/viewer-coupe.mp4) · [hindsight still](report/viewer-offline-coupe.jpg) | The black coupe is distinct from ego and changes position through the turn. The surrounding road is sparse and the generic body shape is visible from close range. |

## Coverage and misses

| Seconds | Inventory fragments | Screened proposals | Unassigned proposals | Published episodes | Person/rider-like fragments |
|---|---:|---:|---:|---:|---:|
| 80–140 | 187 | 93 | 80 | 4 | 12 |
| 190–215 | 23 | 11 | 6 | 4 | 0 |
| 286–330 | 409 | 154 | 137 | 9 | 6 |

Screen: ≥4 key masks, ≥150 pixels at peak, ground-contact speed 1.8–34 m/s, residual below max(3 m, 0.15 × median range). Fragment counts are **not** physical-object counts: many are ego body/rim, signs, hedges, stationary traffic or duplicate IDs. [Counts by camera](report/coverage.json), [31 largest initially unassigned crops](report/unassigned.jpg), [person-like and handoff checks](report/review.jpg). All 18 person/rider-like fragments were inspected; none established a moving human with reliable ground contact. The sparse junction packet covered front and both pillars, not the rear or repeaters. Absolute missed people/vehicles outside these annotations are unknown.

The inspection recovered three camera fragments of the first black oncoming SUV, one later repeater fragment of the turning silver crossover, and [the white junction SUV's ID split](report/junction-identity.jpg). The blue SUV was visible and associated across three cameras but withheld: a single-camera causal fit suggested movement, the whole-episode velocity was only ~0.9 m/s, and the cameras disagreed. No reliable moving/parked verdict. The red parking-lot vehicle is also excluded: it would pass the motion score despite appearing stationary. The grey car at the wall was rejected by the score. [Control measurements](report/controls.json). Parked-looking objects are not automatically stripped from static content.

## What “live” proves

All 17 track/model fits use release times that cover the shutter and both interpolated ego samples. The largest release delay in the fitted episodes was 0.509 s relative to the foot row. State 0 denotes recent released evidence; state 1 is motion-only. Previously photographed surface remains as memory; surfaces never photographed remain in a separate predicted PLY. These are different statements about pose and appearance.

The live prefix audit reruns the estimator after physically removing observations released after 103.7 s (SUV) or 131.3 s (bus), then rebakes and compares positions, colours and visible temporal metadata. [Prefix results](report/prefix.json). This conditions on fixed identity annotations. Candidate inventory, initial identity review and optional association proposals use offline data, so this is **not** an end-to-end causal detector or online identity benchmark. Compute latency and the viewer’s smoothed ego chase camera are outside the audit.

The live scene links 1,396 unchanged source tiles and filters 776 tagged legacy rows. Hindsight links the 368 parent tiles plus 96 previously absent ground tiles from the matching road revision. No editor/renderer change is required. All 102 per-object proof plates and the model and measurement JSONs are in [report/](report/).
