# Traffic in the published v5 splat world

[Open the default chase at the pull-away](http://10.0.50.51:8090/?scene=stitched-opt-v5-movers-v1&photo=off&t=286). The three slices share one scene: 23 moving episodes and nine fixed models, all object-local Gaussian splats. All 924 tile PLYs and their 924 cache entries are the very same files as the static parent; the road and sky grade is unchanged. The public manifest SHA-256 is `b9f88e3ee3e39d029f4dc939f419cff87f03dc3e520e63c4fb69ba5470c59461`. No photographic planes or video backgrounds were active in the review.

## Default chase playback

| Range | What actually shows |
|---|---|
| [129–134 s · bus](bus.mp4) | The oncoming bus is behind the left median trees and leaves the forward field of view. The viewer does **not** show a readable bus pass. At 129.25, 130.25 and 130.75 s, toggling the bus alone changed no above-threshold pixels with the scenery loaded. Hiding the static field exposes the window band, wheels and livery on the same actor; thin markings are still soft. [Full scene / static hidden](bus-depth.jpg) · [pixel counts](occlusion.json). |
| [286–300 s · pull-away](pull-away.mp4) | A stopped grey Tesla stays beside ego; the dark coupe pulls away under the overpass. A velocity-derived yaw had spun the coupe as it began to move. The final fit advances it along the lane: 0.15 m travel in the first 1.6 s, zero backward steps over 2 cm, and a maximum heading change of 0.59° per 36 Hz step. The grey Tesla still has a blocky rear where its inferred face meets the original splats. The black coupe’s tyres cross a hole in the road surface left of ego; even a correct world z cannot make that contact look convincing. |
| 295–315 s · traffic | Traffic far ahead is lost in the white haze on the early straight. Three separately photographed lane-ahead vehicles and the oncoming models take shape on the later approach. No extrapolation is made across seven seconds without a usable lane-ahead camera token. |
| [316–326 s · junction](junction.mp4) | The dark coupe moves through the turn while the blue SUV holds its place across the road. The white queue resolves into two different bodies in the side cameras; the nearer car creeps and the farther car stays fixed. Static fragments can soften or double an outline. [Full scene / static hidden at the blue SUV](corner-depth.jpg). |
| 198–206 s · stop | The red crossing SUV is visible across the intersection; at chase distance it is a small, partly hazed silhouette. No moving vehicle is visible throughout the entire stop. |

The separate measured and inferred files contain 126,185 and 300,601 splats. Each loft face prefers a broad, frontal masked photograph over a glancing sliver. The texture stays fixed during each hindsight episode; source shutter numbers and per-donor counts are in [world-traffic.json](world-traffic.json). Both the v5 static fit and these exports use the same `pipeline/photo` corrected camera RGB (stitched-ground-v2), encode SH DC as (RGB/255 − 0.5)/0.28209479, and leave display grading to the viewer. Its grade gains were 1.156, 0.937, 0.937. Distinct measured and inferred geometry is preserved: a complete body does not imply every face was seen.

Ground height is fitted to local road samples under the wheels. Supported tyre-height residual P90: departing coupe 0.9 cm, fixed grey Tesla 0.7 cm, bus 7.2 cm. Only 39/996 bus wheel probes had dense ground support. Every model exports a ground-tested elliptical footprint, its size and peak opacity. In the recorded bundle the per-mover shadow shader was not present; it is not valid to count the ego car's shadow as a mover shadow. The bus comparison establishes foreground occlusion by the trees, not perfect sorting of every translucent pole or branch. No unambiguous draw-through of a solid pole was identified.

## Parked audit

Reviewed all 18 producer track IDs with parked decisions in the slices, including the merged white pair. “Coverage” is the median fraction of a real instance silhouette covered by projected parent splats inside the fitted 3-D footprint, **not** a car completeness score. Source crops are paired with the isolated static-parent footprint in [sheet 1](parked-sheet-0.jpg), [sheet 2](parked-sheet-1.jpg), [sheet 3](parked-sheet-2.jpg) and [the second white car](parked-sheet-3.jpg). Class/pose ambiguity is recorded instead of turning distant furniture into cars. Three stopped-in-traffic models are only shown around their observed stops. Every fixed model has constant x/y/z/yaw and both track states are zero.

| Producer track | Parent coverage | Decision |
|---|---:|---|
| 4 | 0.16 | Fixed 10004: silver sedan |
| 7 | 0.06 | Fixed 10007: burgundy sedan |
| 47 | 0.23 | same silver sedan as 4 across the island |
| 492 | 0.00 | Fixed 10492: distant blue vehicle |
| 497 | 0.00 | background behind trunks; a car is not established |
| 498 | 0.00 | occluded sliver at a concrete support; class and heading indeterminate |
| 502 | 0.00 | three distant dark fragments split by a post; class not established |
| 504 | 0.00 | hedge and pale facade; no reliable car outline |
| 587 | 0.44 | Fixed 10587: grey Tesla at the light |
| 589 | 0.95 | static splats already describe the bonnet, roof, flank and wheels |
| 593 | 0.16 | Fixed 10593: white Tesla across the intersection |
| 770 | 0.14 | roadside board overlaps the mask; vehicle extent not established |
| 789 | 0.56 | same blue SUV as 972 |
| 935 | 0.00 | Fixed 10935: red SUV at the roadside |
| 972 | 0.62 | Fixed 10972: blue SUV at the corner |
| 976 | 0.00 | pickup in the moving queue; poor fixed fit; do not freeze |
| 1040 | 0.86 | merged near and far white cars; near car creeps forward (17); far car separately fitted using two clean mask IDs (10401) |
| 1091 | 0.00 | Fixed 11091: silver SUV stationary at end of observation |
| 1040, far car | 0.45 | Fixed 20401: far white sedan behind the blue SUV queue |

[Playback stills: pull-away](pull-away-287.jpg) · [junction](junction-320.jpg) · [traffic ahead](lane-traffic-312.jpg) · [cross traffic](cross-traffic-204.jpg).

[Verified publication, ground footprints and time ranges](published.json) · [viewer playback receipts](playback.json) · [clip sizes, decoded frame counts and hashes](clips.json) · [spatial audit](parked.json). To reproduce on Studio in the existing environment, run `pipeline.movers.standing` then `pipeline.movers.world_traffic` from the reality-splat root with one BLAS thread; rebuild the scene's tile cache. These are hindsight models, not inputs to a causal live run. No eligible three-slice live parent was present at delivery.
