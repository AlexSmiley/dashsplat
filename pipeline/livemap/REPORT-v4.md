# Live world v4: 85–140 s

[10-second chase, 30 fps](report/chase-v4-94-104.mp4) · [held-out storefront](report/v4/camera-left_repeater.jpg) · [exit orbit](report/v4/orbit-exit.jpg) · [arterial orbit](report/v4/orbit-arterial.jpg)

## Published scene

Studio: `$RS/scene/live-v4/map.json`. 132 hard-linked original ground tiles; 423 non-ground tiles (202 transient). 7,566,996 immutable static generations and 250,528 instance-tagged transient generations; **7,817,524** records total, including retirements (750.7 MB of object PLYs). 5 Hz sampling found a peak of 579,855 extant records. Fusion including preparation ran at 29.7 camera frames/s on Studio; depth, export and rendering are excluded.

Distance extrapolation: **38.17M**. Time extrapolation: **50.96M**. Both include retired records, against a 60M limit. These are slice estimates, not a whole-drive measurement. The six-lens whole schedule needs 5,426 source exposures after 140 s; the input inventory below is independent of these projections.

## Held-out real cameras

Exactly the v3 set: ten scored physical exposures per camera, unseen by object depth windows and the original ground schedule. The additional storefront row is unscored because ground v2 saw it. Denominator: calibrated non-sky, non-road pixels **including movable instances** (a mover overrides a road label). Results are pooled by pixel within each lens. Geometry requires accumulated alpha >0.50; objects require non-ground alpha contribution >0.45 after depth compositing. The same slice panorama is behind both maps and contributes zero to both tests. PSNR uses the unmodified photographs, including misses; opacity coverage is not correct placement.

| Camera / comparison | v3 geom. | v4 geom. | Lowest v4 frame | v3 objects | v4 objects | Non-ground PSNR v3 → v4 | v4 objects over sky |
|---|---:|---:|---:|---:|---:|---:|---:|
| [front](report/v4/camera-front.jpg) | 40.4% | 96.1% | 92.3% | 23.4% | 80.5% | 13.47 → 18.67 dB | 15.2% |
| [left_pillar](report/v4/camera-left_pillar.jpg) | 35.6% | 91.9% | 82.3% | 15.9% | 80.5% | 14.77 → 17.71 dB | 23.8% |
| [right_pillar](report/v4/camera-right_pillar.jpg) | 47.6% | 95.4% | 90.6% | 17.3% | 75.6% | 12.11 → 17.44 dB | 28.5% |
| [left_repeater](report/v4/camera-left_repeater.jpg) | 34.6% | 95.4% | 88.1% | 16.1% | 88.3% | 12.76 → 17.19 dB | 33.5% |
| [right_repeater](report/v4/camera-right_repeater.jpg) | 45.5% | 96.1% | 90.6% | 20.4% | 81.3% | 12.82 → 17.14 dB | 35.1% |
| [back](report/v4/camera-back.jpg) | 41.7% | 94.5% | 87.3% | 15.2% | 78.1% | 16.16 → 19.41 dB | 29.0% |

**55/60 individual held-out images** exceed 90% geometry coverage. Scores on the stricter object-contribution test are shown separately. See [the measurements](measurements-v4.json) for every frame and the mover-only subset.

## World-space evidence and temporal behaviour

Both chase and orbit use the stored scene from a camera 6 m back and 3 m up. Each slow orbit freezes scene time and turns through 360° in 12 s; the chase video advances both the car and scene clock for 300 consecutive frames. No source photo, synthetic car, or unprojected backdrop is inserted. Studio videos: `$RS/work/livemap/orbit-v4-<name>.mp4` and `$RS/work/livemap/chase-v4-94-104.mp4`.

| Moment | Chase v3 / v4 | v4 orbit | Traffic detail |
|---|---|---|---|
| exit · 93.70 s | [chase](report/v4/chase-exit.jpg) | [film](report/v4/orbit-v4-exit.mp4) · [sheet](report/v4/orbit-exit.jpg) | — |
| first-road · 112.60 s | [chase](report/v4/chase-first-road.jpg) | [film](report/v4/orbit-v4-first-road.mp4) · [sheet](report/v4/orbit-first-road.jpg) | — |
| arterial · 135.00 s | [chase](report/v4/chase-arterial.jpg) | [film](report/v4/orbit-v4-arterial.mp4) · [sheet](report/v4/orbit-arterial.jpg) | — |
| crossing-car · 102.82 s | [chase](report/v4/chase-crossing-car.jpg) | [film](report/v4/orbit-v4-crossing-car.mp4) · [sheet](report/v4/orbit-crossing-car.jpg) | [crop sequence](report/v4/traffic-handover-crossing-car.jpg) |
| passing-bus · 131.78 s | [chase](report/v4/chase-passing-bus.jpg) | [film](report/v4/orbit-v4-passing-bus.mp4) · [sheet](report/v4/orbit-passing-bus.jpg) | [crop sequence](report/v4/traffic-handover-passing-bus.jpg) |

**What holds together:** the exit storefront, red fascia, planting island and opposite tree line stay recognisable through the turn. The forward arterial has a horizon again. Neither the sign lettering nor the signal gantry is reliably legible.

**What still moves:** during the 94–104 s chase, lane markings mostly stay put while curb vegetation and the farther trees change shape and slide relative to them. The quarter-second fade softens the static refresh, but around high-contrast trunks and façade columns it produces two soft edges and local flicker. Turning the 135 s orbit backwards shows retained roadsides; some verge and canopy sheets float above the asphalt, and several poles lean or stop short of it. This is a near-car reconstruction, not stable arbitrary-view geometry.

**Traffic:** the bus is recognisable across four updates, but its red roof and rear edge trail during handover. The white crossing SUV is almost entirely missing at its photographed location; pale fragments appear behind it. In this sequence its depth centroids move 2–5 m per update, and capture-to-release plus the fade delays the displayed position. The sidecar supplies measurements for a rigid-track model; these transients do not already implement one.

**Coverage tradeoff:** the producer already invalidates depth edges. The final pass erodes between two measured, disagreeing ranges rather than treating an absent neighbour as zero metres and eroding again. This recovered thin and distant fragments, but side-camera splats now cover up to 35% of held-out sky. The reserved photographs were used to diagnose this sampling choice; their pixels were never fused, but these are not blind model-selection results.

During the chase, consecutive frames were warped by the known change in camera pose and compared only where both depths agree. This measures image change on geometrically matched pixels, not position error against a photograph.

| Region | Mean RGB change /255 | Fraction changing by >25 | Mean of framewise p90 |
|---|---:|---:|---:|
| near | 0.51 | 0.01% | 1.33 |
| far | 0.67 | 0.01% | 1.65 |

Static agreement extended 10,145,124 visibility observations; 313,425 finer handovers, 5,063,845 free-space/sky retirements and 1,019,183 displaced front sheets were recorded. The quarter-second overlap is real; it does not guarantee constant opacity or perfect registration. Rear memory coarsening retired 439,793 old records. The playback tape still includes them.

For the traffic sequences, the green overlay is visible transient alpha >0.35 inside the tracked mask; red is visible transient outside **all** instance masks dilated by two pixels, within the crop. These are illustrative training-time photographs evaluated at capture time, not additional holdouts. The displayed scene can only use packets already released. A mask classification or timing error can also look like a trail.

- **crossing-car:** 4 photographic instants; mean transient spill outside all nearby masks 46.7%. Mean target-mask coverage by transients 0.6%.
- **passing-bus:** 4 photographic instants; mean transient spill outside all nearby masks 12.2%. Mean target-mask coverage by transients 78.4%.

## Inputs, range and sky

Requested 4.5 Hz moving, 2 Hz stopped, with a one-second warm-up from 84 s. Actual intervals vary because inputs must be complete and held-out frames cannot occur in depth windows. The scene clock uses calibrated release time, not an unreleased physical exposure.

| Camera | Packets | Moving Hz | Stopped Hz | p95 gap (s) | Max release lag (s) | Records born ≥100 m | Maximum range (m) |
|---|---:|---:|---:|---:|---:|---:|---:|
| front | 246 | 4.40 | — | 0.333 | 0.401 | 43,126 | 163.9 |
| left_pillar | 245 | 4.38 | — | 0.333 | 0.400 | 12 | 100.9 |
| right_pillar | 244 | 4.37 | — | 0.333 | 0.392 | 0 | 78.1 |
| left_repeater | 244 | 4.37 | — | 0.333 | 0.395 | 127 | 123.1 |
| right_repeater | 245 | 4.38 | — | 0.333 | 0.400 | 10 | 103.8 |
| back | 245 | 4.38 | — | 0.333 | 0.389 | 0 | 94.7 |

Sky colour is measured in 290 training exposures. Direct bins occupy 22.3% of the upper-hemisphere texels before wrapped filtering and smooth extension. The sky is a single non-causal slice-wide texture by design. No ground or object position is inferred from it.

All 7,817,524 PLY records passed header/extent, finite geometry, quaternion and lifetime checks. Every transient row has a source camera, ordinal and mask ID, with its PLY tile range and per-frame robust world centroid in `$RS/maps/live-v4/instances.json`. 65 segmented observations had no valid depth centroid and are explicitly null. IDs are camera-local; these are not rigid tracks.

Whole-drive inventory after 140 s at verification: 0/5,426 scheduled shared depths and 1/5,426 shared sky masks. See [reproduction](README.md) and [the updated live contract](DESIGN.md).
