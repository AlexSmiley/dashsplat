# Movers: v3 demo and unseeded live replay

## Demo evidence

[Bus: actual v3 viewer beside synchronized dashcam](report2/bus-v3-vs-dashcam.mp4) · [318–322 s traffic in the actual viewer](report2/traffic-turn.mp4).

The v3 object overlay retains all 419 static tile inodes, the parent's map metadata, sky and available generated clip. It contains the four in-slice hindsight episodes. In ordinary playback the bus passes from the left pillar into the left repeater on the opposite carriageway, 23.6–25.5 m left of ego over the compared fits. The viewer bus is much less legible than the photographed bus: reconstructed greenery frequently covers its windows and wheels. The side-by-side comparison uses the nearest native shutter to the recorded viewer HUD clock, within 14 ms. The HUD updates less often than the render. [Decoded comparison](report2/bus-viewer-comparison.jpg) · [timing evidence](report2/bus-sync.json).

A height-only adjustment used 108 near-side wheel contours from fit exposures; it did not move the bus sideways to chase pole-split masks. On held exposures the median absolute wheel-to-mask-bottom difference was **1.93 px (pillar, 8 wheels)** and **0.52 px (repeater, 10 wheels)** on the 724 × 469 grid. Only 8 of 924 wheel positions had dense reconstructed ground within 1 m; those eight have ~7 cm absolute height residual. The static surface does not establish tyre clearance for the rest of the pass. [Tyre projections](report2/bus-v3-road.jpg) · [mesh support](report2/stitched-opt-v3-movers-v1-contact.json).

The traffic clip shows the black coupe turning across ego's path with another vehicle entering, over the live-v5 static scene **with hindsight movers**. Gaps in the cross-street ground are visible. This clip does not establish online identity recall.

## Photographed models

[Bus](report2/passing-bus-before-after.jpg) · [turning coupe](report2/turning-black-coupe-before-after.jpg) · [junction white SUV](report2/junction-white-suv-before-after.jpg) · [junction silver crossover](report2/junction-silver-crossover-before-after.jpg). Each sheet shows eight fixed azimuths at early, middle and late coverage, before and after. [Per-instance shape settings, image counts and generations](report2/shape-texture.json).

Hindsight fits length, height and cabin position to non-held silhouette contours, then reprojects photographs. Contour registration aligns body and trim; fine colours come from individual exposures ranked by face angle, native pixel density and local sharpness. Another exposure must corroborate a glass colour; otherwise inferred dark glass remains. Inferred rims, lamps, roofs and mirrored sides are kept separate and retire when measured support arrives. Early bodies are easier to read. Streaks on late bus livery and coupe paint remain. The underside and much of the roof lack direct photographs.

## Online discovery

Discovery receives release-ordered mask pixels, camera timing, released/extrapolated ego pose and class dimensions. Neither tracker reads reviewed subjects. Only the benchmark adapter removes the frozen 136 masks; the scorer matches outputs to reviewed episodes using **other** mask observations. Strict scoring permits one persistent live ID per truth episode. The optimistic fragmented score picks the best available associated fragment at each held exposure; it is not identity-continuity recall. Counts are exposures / no track / additional projected box IoU below 0.5.

The first pass (live-v5-movers-v2) used world bearing, paint and overlap assignment; it sometimes followed a different car on the same ID. The component pass (live-v5-movers-v3) preserves an unreviewed camera-local mask token, buffers the first measurements of a new token, and joins handovers after checking released world motion, colour and concurrent actors. A token never identifies an object across cameras; buffered information is released only at the resolving exposure. This is an information-causal replay, **not a claim of running every camera inference at 36 Hz wall-clock**.

The component filter starts with a class-sized ground-contact body, compares stationary and translating silhouettes, and checks interior image corners against plausible static depths and nearby background motion. An independent crop classifier filters road signs mislabelled as vehicles. Rejected updates do not snap the visible pose. The 36 Hz presentation advances bounded velocity across camera gaps and releases knots one tick late for causal interpolation. Motion prediction and photo visibility are separate: measured PLY runs indicate observed/remembered surfaces; mirrored or class material is inferred. Geometry-only release requires repeated translating fits when corner motion is ambiguous.

## Results on the same 136 masks

| Run | No matching track | Additional IoU < 0.5 | Aligned exposures |
|---|---:|---:|---:|
| Round 1 live, reviewed identities | 22 | 34 | 80 |
| First unseeded pass | 89 | 17 | 30 |
| Component-first live, strict persistent ID | 52 | 30 | 54 |
| Same output, allowing fragments | 32 | 37 | 67 |
| Updated hindsight | 0 | 10 | 126 |

Unseeded discovery improved over its first pass, **not over the reviewed-ID baseline**. Hindsight formerly had 0 temporal and 9 spatial misses. The 55 live outputs are episodes, not 55 verified distinct moving vehicles. The held panel only measures the 17 reviewed episodes and cannot estimate scene-wide precision. The bus's primary unseeded ID covers true left-pillar and left-repeater sightings; the front sighting remains separate. The crossing SUV carries one true identity from the front into the right pillar. In the later turn, the black coupe's primary live ID covers all 13 held times; 12 meet IoU 0.5. Its track later picks up unrelated observations. At junctions both splits and overlong/merged tracks remain.

The parked/ambiguous exclusion is reliable only in the reviewed demonstration. In the unseeded replay neither storefront nor fuel-price-sign control was published, but three IDs consumed observations of the withheld blue SUV. Its motion is still unconfirmed. Treat this as a live false-positive risk, not a newly established mover. [Control results](report2/component/stationary-controls.json). The white SUV and black coupe are easier to see than very distant actors; some distant live cars visibly float above sparse road geometry.

### Largest 36 Hz position step (3D metres)

| Reviewed episode | Live held / absent / low IoU | Matched live ID | Live max (m) | Hindsight max (m) |
|---|---:|---:|---:|---:|
| crossing-suv | 4 / 1 / 3 | 43 | 0.88 | 0.47 |
| passing-bus | 26 / 9 / 1 | 256 | 0.64 | 1.45 |
| black-oncoming-suv | 28 / 17 / 8 | 246 | 0.46 | 1.17 |
| red-oncoming-sedan | 8 / 3 / 3 | 279 | 1.08 | 1.55 **⚑** |
| junction-white-suv | 5 / 2 / 1 | 305 | 0.98 | 0.77 |
| junction-silver-crossover | 3 / 2 / 1 | 329 | 0.53 | 0.75 |
| junction-red-suv | 1 / 0 / 1 | 336 | 0.43 | 0.61 |
| junction-approach-suv | 2 / 0 / 0 | 307 | 0.39 | 0.33 |
| departing-coupe | 13 / 3 / 2 | 363 | 1.06 | 0.35 |
| oncoming-pickup | 4 / 3 / 1 | 641 | 1.09 | 0.49 |
| oncoming-minivan | 4 / 2 / 1 | 659 | 1.00 | 0.45 |
| oncoming-black-sedan | 5 / 1 / 1 | 662 | 1.09 | 0.93 |
| oncoming-grey-suv | 4 / 4 / 0 | 724 | 0.66 | 0.56 |
| turning-black-coupe | 13 / 0 / 1 | 547 | 1.08 | 0.48 |
| turning-white-sedan | 3 / 1 / 1 | 764 | 0.43 | 0.21 |
| turning-silver-crossover | 5 / 0 / 5 | 815 | 1.08 | 0.60 |
| junction-white-sedan | 8 / 4 / 0 | 845 | 0.71 | 0.64 |

All 55 live tracks stayed at or below **1.091 m**. This bounds display corrections; it does not prove the underlying positions or instantaneous speeds are correct. One hindsight step exceeds 1.5 m: the red oncoming sedan, **1.545 m at 138.723–138.751 s**. The rear-camera mask is about 14 × 8 px and constrains bearing much better than range. [Real frames at the peak](report2/fastest-hindsight-step.jpg). Full values including every unmatched live ID: [live metrics and steps](report2/component/online-evaluation.json) · [hindsight metrics and steps](report2/hindsight-evaluation.json) · [all published manifests, PLYs and tile caches](report2/component/published-audit.json).

The first association pass generated 2,747 tentative identities; preserving camera-local continuity reduced that to 917. Two coasting duplicates were retired only after different cameras repeatedly supported the same projected body and 3D velocity. The 318.83 s decision removed the second coupe by 319.28 s, verified in the [actual live viewer at 319.32 s](report2/online-traffic.jpg). The HUD includes prefetched tracks. [Decisions](report2/component/handover-retirements.json) · [prefix replay](report2/component/handover-prefix.json).

Truncations at 103.7, 131.3 and 319.05 s reproduced the released assignments and already-published pose and PLY generations at those cuts. [Prefix evidence](report2/component/causality.json) · [withheld pixel boxes absent from discovery input](report2/holdout-audit.json).

## Junction people and available inputs

99 segmented shutters per camera in 190–215 s (594 in six cameras). There were 54 person/rider/cycle-like detections across 49 fragments; just one fragment lasted three detections. Context-stabilized native crops show stationary street furniture, vegetation and ambiguous distant shapes. None establishes independent walking or cycling. No moving human was added to the demo. Human/cycle geometry has a separate head, torso, arms, thighs and shins; it does not estimate gait. [Inventory](report2/people.json) · [strongest candidates over time](report2/junction-person-motion.jpg) · [disk-verified inputs](report2/input-coverage.json).

At final inspection the later optimized static parents were not present. The live static base supplies the traffic demonstration meanwhile. Shared 286–330 s depth frame counts: front 960, back 967, left_pillar 954, right_pillar 0, left_repeater 0, right_repeater 0. The junction used separate 4 Hz instance masks; shared six-camera junction depth was not available. The Studio filesystem watcher is bounded to 20:40 EST on Nov 21 and writes a result only when an optimized parent and its cache are complete.

## Reproduce on Studio

From `~/dev/reality-splat` in the existing environment, set `OPENBLAS_NUM_THREADS=1` and `VECLIB_MAXIMUM_THREADS=1`. Run each named Python module as `nice -n 10 .venv/bin/python -m pipeline.movers.<module>`.

1. `upgrade` fits the 17 hindsight models and attaches in-slice movers to available parents. `inspection panel bus people coverage` recreates the corresponding evidence.
2. With `MOVERS_RUN=component` run `packets --holdout`, `handoff`, `motion`, `semantics`, `filter`, `online_scene publish` and `online_scene score`. Omit `--holdout` on unlabelled drives. Motion and semantic evidence can be computed independently. The category check uses Apple's CPU Vision classifier without new weights or packages.
3. Rebuild the cache with `cd viewer && bun run cache:studio` in the local checkout, then run `inspection audit prefix controls` on Studio. With the default run directory, `demo --follow-later 120` waits on filesystem events for complete optimized parents covering the later windows. It creates cached mover variants and exits when both are ready or two hours elapse. Results are written to `work/movers/round2/late-parents.json` as parents become ready.
