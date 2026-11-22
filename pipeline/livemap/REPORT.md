# Static object layer: 85–140 s

The later [live v4 now + memory report](REPORT-v4.md) compares the same physical camera poses over sky and includes traffic and a continuous chase video.

Both scenes retain the whole-drive ground v2. **Objects are a sampled slice, not whole-drive coverage.** Dark openings in non-sky regions are gaps in the stored geometry; sky itself is intentionally blank. The first and last scheduled reference packet times are 85.355 and 139.740 s.

| Variant | Object records | Tiles | Object PLY bytes | Fusion (s) | Camera frames/s | Distance-scaled whole drive |
|---|---:|---:|---:|---:|---:|---:|
| live | 149,979 | 82 | 14,443,004 | 6.47 | 129.4 | 728,744 |
| stitched | 403,466 | 107 | 38,791,529 | 6.52 | 128.5 | 1,960,432 |

All 838 selected training packets per variant were processed. Live used released evidence from 85.701–139.808 s (maximum capture-to-release lag 0.389 s). The whole-drive object budgets are 40M live / 20M stitched. Distance extrapolation is ×4.86, not a whole-drive benchmark. Timings include JPEG/depth/mask I/O, row poses, radiometry, footprint preparation and native fusion; they exclude depth inference, audit, PLY export and proof. Ground is hard-linked and occupies no additional disk blocks.

## What the pictures show

The [93.7 s exit orbit](report/orbit-exit.jpg) is the strongest result: the planting island, shop columns, red sign and opposite wall retain their positions as the camera turns. The sign shape and tint survive; its lettering does not. On the [135 s arterial](report/chase-arterial.jpg) the stitched right verge has trees and posts, but the forward horizon is mostly missing. Live on the fast road is still essentially a ground reconstruction with scattered roadside objects. The slice demonstrates static memory, not complete street geometry.

- **Façades and poles.** At 92.7 s the roofline has gaps, and some of the opposite wall occupies two ranges. At 136.1 s the right-pillar photo has one thick pole; stitched has a displaced second outline. The 117.5 s utility pole is missing almost entirely in both. The spot checks below distinguish holes from duplicated depth and found no matching recent live contradiction retirement at the two sampled pole cores. Thin features lose their valid interiors under silhouette erosion; remaining estimates move laterally when range is wrong.
- **Foliage and terrain.** Canopies have thickness and survive edge-on, but the 60° exit and 240° first-road orbit show smeared branches and vegetation floating above gaps at the kerb. Thickness helps visibility, not positional accuracy. Grass and coarse terrain are deliberately sparse where neither detector nor ground fit supports them. Sky input pixels were excluded; splats drawn over held-out sky are silhouette errors (21% on stitched right pillar).
- **Lens decisions.** The stitched right repeater had zero independently supported bands, hence zero source records; other lenses still render in that viewpoint. Stitched rear births are limited to locally qualified 2–8 m packets inside 65°. On separate test tracks the outer rear under-ranges by 58% median (live: 61%). Short-range evidence and safe subjects are preferable to filling these gaps with wrongly placed geometry.
- **Scaling decision.** Object counts and CPU fusion speed leave ample headroom. The limiting factor is independent depth agreement. Prioritise side-camera/edge geometry and fast-road capture density before expanding this configuration to the whole drive. The frozen v2 ground and its existing exit stripe are unchanged.


## What is actually present

| Source lens | Train packets L/S | First–last physical time (s) | Longest gap (s) | Exported birth records L/S |
|---|---:|---:|---:|---:|
| front | 139 / 139 | 85.355–139.740 | 1.39 | 54,293 / 91,485 |
| left_pillar | 139 / 139 | 85.356–139.741 | 1.39 | 35,637 / 74,324 |
| right_pillar | 139 / 139 | 85.364–139.750 | 1.39 | 9,705 / 184,429 |
| left_repeater | 139 / 139 | 85.361–139.746 | 1.39 | 34,145 / 50,446 |
| right_repeater | 139 / 139 | 85.356–139.741 | 1.39 | 445 / 0 |
| back | 143 / 143 | 85.367–139.752 | 1.39 | 15,754 / 2,782 |

Live: 268 coarse terrain records; 56,985 thick foliage/rough patches; 4,906 supersessions, 656 contradiction retirements, 327 removals by prior hull occupancy, 5,577 visibility continuations. Supersession and contradiction counts are fusion events, not human-labelled errors. 2,689,318,255 bytes of original ground are shared by hard link.

Stitched: 807 coarse terrain records; 171,928 thick foliage/rough patches; 10,220 supersessions, 2,502 contradiction retirements, 324 removals by prior hull occupancy, 0 visibility continuations. Supersession and contradiction counts are fusion events, not human-labelled errors. 1,499,867,974 bytes of original ground are shared by hard link.

## Held-out photographs

Each plate is the unmodified native photo, live render, stitched render. Every scored physical frame was excluded as a fusion target, from all same-lens training depth windows and from the frozen ground schedule. Fixed pre-existing camera, trajectory and radiometry calibration are shared. Every lens has 10 scored frames; three are shown per plate. The additional storefront row is an object-only holdout that ground v2 saw; it is labelled and excluded from scores. No sky, car model or image backplate is composited. Geometry is evaluated at physical row times. The raw-photo PSNR penalises missing static non-road pixels as background; matched MAE and coverage are reported together so a sparse layer cannot win by staying blank.

| Camera | Ground → v3 PSNR gain dB L/S | Non-road PSNR dB L/S | Non-road object coverage L/S | Covered MAE L/S (RGB levels) | Depth too near L/S | Sky overlap L/S |
|---|---:|---:|---:|---:|---:|---:|
| [front](report/camera-front.jpg) | 0.95 / 2.61 | 14.2 / 16.2 | 23.5 / 55.0% | 24.7 / 23.7 | 10.0 / 24.1% | 0.7 / 10.6% |
| [left_pillar](report/camera-left_pillar.jpg) | 0.32 / 3.41 | 10.8 / 14.2 | 15.9 / 53.0% | 32.9 / 31.0 | 23.3 / 19.3% | 2.9 / 5.0% |
| [right_pillar](report/camera-right_pillar.jpg) | 0.41 / 1.91 | 14.5 / 16.5 | 17.4 / 50.1% | 26.2 / 27.8 | 9.4 / 23.6% | 7.1 / 21.0% |
| [left_repeater](report/camera-left_repeater.jpg) | 0.56 / 3.90 | 12.1 / 15.6 | 16.1 / 54.5% | 28.9 / 26.6 | 7.9 / 6.6% | 2.2 / 5.0% |
| [right_repeater](report/camera-right_repeater.jpg) | 0.29 / 1.31 | 14.2 / 15.5 | 20.5 / 56.3% | 28.1 / 30.7 | 5.1 / 20.5% | 4.6 / 20.4% |
| [back](report/camera-back.jpg) | 0.53 / 1.52 | 12.0 / 13.0 | 16.6 / 39.1% | 37.3 / 31.7 | 16.1 / 13.1% | 3.1 / 5.7% |

“Too near” compares the depth-sorted alpha-weighted range to **held-out DA3**, only where an object was drawn and valid reference range was 2–40 m. A violation is >20% and >0.5 m too near. It is a symptom, not independent truth. Sky overlap is rendered object coverage on held-out sky labels; sky was never a depth or free-space anchor. Raw and photometrically corrected MAEs, all-static PSNR and both depth tails are in the measurements.

## Depth qualification

Depth is physical optical-axis z. Divide by the original distorted grid’s unit ray z **before** comparing or fitting. The 59,665-point BA is distinct from DA3; its poses were inputs to DA3, its landmark positions were not. Transform original BA xyz through its original row camera into the v2 road gauge. For the second check, triangulate reserved tracks from their row-timed rays; exclude weak perpendicular baselines, high angular residual, dynamic/sky/edge pixels and extrapolation without a nearby observed track. Road checks intersect the separately fitted suspension/road tangent. Counts are correlated observations, not independent trials. All bands below use **reference range**; birth gates use measured/corrected range. Table entries: signed median error / robust scatter (1.4826 × MAD), metres, followed by N. Raw uncorrected tables and p90 absolute errors are in [measurements](measurements.json).

| Camera | Median applied a / b (1/m for b) |
|---|---:|
| live front | 1.00293 / 0.00188 |
| live left_pillar | 0.96541 / 0.00033 |
| live right_pillar | 0.97635 / 0.00005 |
| live left_repeater | 0.98244 / 0.00027 |
| live right_repeater | 0.85056 / 0.00199 |
| live back | 0.95632 / -0.00216 |
| stitched front | 0.99661 / 0.00167 |
| stitched left_pillar | 0.94135 / 0.00235 |
| stitched right_pillar | 0.95388 / 0.00085 |
| stitched left_repeater | 0.95851 / 0.00087 |
| stitched right_repeater | 0.83497 / 0.00300 |
| stitched back | 0.90381 / -0.00146 |

Live uses only earlier measured road and raw LK prefixes triangulated in the causal gauge. Its gates also use the current released prefix; an untested band can make *proposals* but needs independent confirmation. Stitched fits a regularised affine inverse-range model, retains identity if validation gain is under 1.5%, and quarantines local packets where separate bundle anchors disagree. Rejected calibration candidates are not used.

| Lens | Admitted packets by measured band: 2–8 / 8–15 / 15–25 / 25–40 / 40–70 m | Angular cap |
|---|---:|---:|
| live front | 100% / 99% / 99% / 83% / 96% | 27.9° |
| live left_pillar | 50% / 60% / 42% / 24% / 0% | 38.1° |
| live right_pillar | 71% / 23% / 6% / 15% / 0% | 38.4° |
| live left_repeater | 65% / 42% / 27% / 37% / 0% | 36.2° |
| live right_repeater | 1% / 0% / 1% / 9% / 0% | 36.6° |
| live back | 76% / 17% / 0% / 0% / 0% | 62.0° |
| stitched front | 100% / 99% / 100% / 98% / 98% | 27.9° |
| stitched left_pillar | 88% / 75% / 83% / 80% / 0% | 38.1° |
| stitched right_pillar | 82% / 0% / 81% / 0% / 0% | 38.4° |
| stitched left_repeater | 60% / 59% / 55% / 66% / 0% | 36.2° |
| stitched right_repeater | 0% / 0% / 0% / 0% / 0% | 36.6° |
| stitched back | 33% / 0% / 0% / 0% / 0% | 65.0° |

### Lens periphery

Outer 15% of the calibrated angular grid (rear: ≥65°); signed fractional range error, scatter and N on the separate test tracks. Includes unsupported distances. An empty cell is unmeasured, not a zero error.

| Lens | Rim starts | Fractional bias / scatter (N) |
|---|---:|---:|
| live front | 21.4° | -0.02 / 0.09 (19) |
| live left_pillar | 38.1° | -0.08 / 0.10 (20) |
| live right_pillar | 38.4° | -0.35 / 0.56 (13) |
| live left_repeater | 36.2° | -0.08 / 0.10 (29) |
| live right_repeater | 36.6° | -0.03 / 0.56 (9) |
| live back | 65.0° | -0.61 / 0.10 (48) |
| stitched front | 21.4° | -0.04 / 0.07 (19) |
| stitched left_pillar | 38.1° | -0.09 / 0.10 (20) |
| stitched right_pillar | 38.4° | -0.38 / 0.53 (13) |
| stitched left_repeater | 36.2° | -0.07 / 0.12 (29) |
| stitched right_repeater | 36.6° | +0.02 / 0.75 (9) |
| stitched back | 65.0° | -0.58 / 0.10 (48) |

### Far-to-near reprojection

A distant DA3 observation on a reserved track is lifted with the applied calibration, then projected onto a later *measured* feature in the same physical camera (native pixels). Require ≥1 m baseline and ≥15% nearer range; select at most one pair per track and source band. Turn: 85–110 s. Fast straight: 130–140 s, ≈21 m/s. A dash means that lens had no qualifying pair. Per-band values, counts and p90 are in measurements.json.

| Lens | Turn median / scatter (N), px | Straight median / scatter (N), px |
|---|---:|---:|
| live front | +36.17 / 19.42 (10) | +37.77 / 32.33 (5) |
| live left_pillar | +33.10 / 13.81 (2) | +159.85 / 123.77 (5) |
| live right_pillar | +505.86 / 469.65 (5) | +83.48 / 99.55 (8) |
| live left_repeater | — | — |
| live right_repeater | — | — |
| live back | — | — |
| stitched front | +14.18 / 15.72 (10) | +13.82 / 18.59 (5) |
| stitched left_pillar | +40.74 / 26.03 (2) | +188.81 / 152.01 (5) |
| stitched right_pillar | +498.59 / 496.86 (5) | +75.08 / 74.00 (8) |
| stitched left_repeater | — | — |
| stitched right_repeater | — | — |
| stitched back | — | — |

### Live: BA landmarks on reserved captures

| Camera | 2–8 m | 8–15 m | 15–25 m | 25–40 m | 40–70 m |
|---|---:|---:|---:|---:|---:|
| front | -0.27 / 0.38 (18) | -0.15 / 1.04 (62) | -0.82 / 1.46 (40) | +0.20 / 2.13 (34) | -2.87 / 4.50 (105) |
| left_pillar | -0.18 / 0.35 (85) | -0.25 / 0.84 (37) | +0.14 / 2.23 (25) | +0.18 / 2.44 (53) | -3.62 / 4.46 (230) |
| right_pillar | -0.13 / 0.28 (98) | -0.09 / 1.14 (65) | -1.30 / 3.88 (106) | -3.82 / 7.35 (132) | -23.55 / 14.97 (45) |
| left_repeater | -0.04 / 0.32 (33) | +0.13 / 0.67 (62) | -0.57 / 1.38 (39) | +0.33 / 2.44 (74) | -4.22 / 10.82 (187) |
| right_repeater | +1.89 / 3.50 (53) | +2.05 / 2.42 (90) | +1.96 / 4.59 (141) | -15.55 / 9.40 (158) | -26.31 / 7.36 (156) |
| back | -0.31 / 0.10 (8) | +1.89 / 6.18 (20) | -1.28 / 5.76 (45) | -11.39 / 7.70 (104) | -30.29 / 7.91 (179) |

### Live: Tracks never used for pose fitting, correction or band selection

| Camera | 2–8 m | 8–15 m | 15–25 m | 25–40 m | 40–70 m |
|---|---:|---:|---:|---:|---:|
| front | — | +50.12 / 0.00 (1) | -0.69 / 1.58 (6) | +0.25 / 5.75 (29) | -1.74 / 6.19 (99) |
| left_pillar | +0.23 / 0.00 (1) | — | -1.51 / 0.71 (3) | +0.56 / 1.88 (37) | -3.88 / 4.89 (165) |
| right_pillar | +0.22 / 0.13 (7) | +1.55 / 1.27 (14) | +1.48 / 1.86 (39) | -1.62 / 5.95 (27) | -27.05 / 10.84 (33) |
| left_repeater | +0.10 / 0.27 (13) | +0.09 / 1.24 (34) | -0.52 / 2.41 (56) | -2.43 / 3.22 (88) | -6.48 / 10.81 (205) |
| right_repeater | +6.67 / 10.30 (9) | +2.20 / 6.33 (32) | -0.17 / 5.15 (102) | -9.99 / 11.64 (121) | -24.58 / 10.01 (181) |
| back | +0.40 / 0.00 (1) | +0.36 / 0.30 (4) | -0.86 / 5.59 (30) | -9.34 / 8.01 (41) | -31.29 / 7.42 (115) |

### Live: Road: signed height above fitted surface

| Camera | 2–8 m | 8–15 m | 15–25 m | 25–40 m | 40–70 m |
|---|---:|---:|---:|---:|---:|
| front | +0.03 / 0.08 (2638) | +0.03 / 0.07 (2964) | +0.16 / 0.17 (156) | -0.24 / 0.20 (9) | — |
| left_pillar | -0.02 / 0.10 (5706) | +0.07 / 0.18 (114) | +0.39 / 0.36 (30) | +0.45 / 0.41 (8) | +1.03 / 0.03 (11) |
| right_pillar | -0.01 / 0.10 (5587) | -0.09 / 0.13 (83) | +0.17 / 0.31 (29) | +0.73 / 0.15 (13) | +1.01 / 0.01 (5) |
| left_repeater | -0.01 / 0.13 (4425) | -0.30 / 0.42 (53) | -0.01 / 0.17 (13) | +0.31 / 0.19 (5) | +0.57 / 0.02 (13) |
| right_repeater | -1.22 / 1.74 (5148) | -0.41 / 0.34 (242) | -0.02 / 0.26 (99) | +0.45 / 0.03 (3) | — |
| back | -0.10 / 0.11 (2031) | +0.33 / 0.11 (21) | +0.55 / 0.03 (4) | — | +0.58 / 0.00 (3) |

### Stitched: BA landmarks on reserved captures

| Camera | 2–8 m | 8–15 m | 15–25 m | 25–40 m | 40–70 m |
|---|---:|---:|---:|---:|---:|
| front | -0.22 / 0.19 (18) | -0.17 / 0.73 (61) | -0.27 / 1.34 (40) | +0.73 / 1.70 (34) | -2.37 / 3.94 (104) |
| left_pillar | +0.00 / 0.27 (85) | -0.19 / 0.50 (36) | -0.22 / 2.37 (24) | -0.24 / 2.01 (54) | -4.64 / 4.76 (230) |
| right_pillar | -0.03 / 0.30 (98) | +0.11 / 1.05 (65) | -0.61 / 3.62 (105) | -3.69 / 6.63 (132) | -24.24 / 13.45 (45) |
| left_repeater | +0.16 / 0.25 (33) | -0.04 / 0.58 (62) | -0.59 / 1.63 (39) | -0.61 / 3.02 (74) | -5.85 / 12.85 (188) |
| right_repeater | +1.40 / 2.67 (52) | +1.41 / 2.57 (91) | +1.27 / 3.81 (140) | -15.33 / 9.40 (158) | -26.52 / 6.90 (155) |
| back | +0.07 / 0.11 (8) | +2.23 / 6.74 (20) | -0.13 / 5.46 (45) | -10.55 / 8.66 (105) | -29.08 / 8.16 (179) |

### Stitched: Tracks never used for pose fitting, correction or band selection

| Camera | 2–8 m | 8–15 m | 15–25 m | 25–40 m | 40–70 m |
|---|---:|---:|---:|---:|---:|
| front | — | +53.92 / 0.00 (1) | -0.71 / 1.66 (6) | +0.58 / 5.65 (29) | -0.47 / 5.18 (99) |
| left_pillar | +0.54 / 0.00 (1) | — | -1.48 / 0.07 (3) | +0.12 / 1.84 (36) | -4.80 / 4.97 (164) |
| right_pillar | +0.42 / 0.11 (7) | +1.48 / 1.29 (14) | +1.67 / 2.34 (39) | -1.09 / 5.64 (27) | -26.56 / 10.99 (33) |
| left_repeater | +0.28 / 0.26 (13) | +0.14 / 0.86 (34) | -0.68 / 1.39 (56) | -3.69 / 3.50 (88) | -7.61 / 9.78 (205) |
| right_repeater | +6.52 / 10.17 (9) | +1.89 / 6.28 (32) | -0.40 / 5.15 (101) | -10.52 / 11.86 (121) | -25.51 / 9.57 (182) |
| back | +0.89 / 0.00 (1) | +0.57 / 0.28 (4) | -0.32 / 6.72 (30) | -7.83 / 7.73 (41) | -30.21 / 8.04 (114) |

### Stitched: Road: signed height above fitted surface

| Camera | 2–8 m | 8–15 m | 15–25 m | 25–40 m | 40–70 m |
|---|---:|---:|---:|---:|---:|
| front | +0.01 / 0.06 (2679) | +0.01 / 0.06 (2994) | +0.03 / 0.06 (90) | -0.20 / 0.00 (3) | — |
| left_pillar | -0.04 / 0.11 (5714) | +0.06 / 0.13 (101) | +0.27 / 0.39 (36) | +0.85 / 0.02 (3) | +0.97 / 0.04 (10) |
| right_pillar | -0.03 / 0.09 (5595) | -0.08 / 0.08 (77) | +0.22 / 0.58 (31) | +0.75 / 0.11 (9) | +1.01 / 0.06 (10) |
| left_repeater | -0.07 / 0.20 (4276) | -0.35 / 0.33 (51) | -0.04 / 0.15 (7) | +0.48 / 0.15 (22) | +0.21 / 0.12 (5) |
| right_repeater | -1.17 / 1.63 (5034) | -0.47 / 0.50 (250) | -0.09 / 0.30 (98) | +0.46 / 0.04 (6) | — |
| back | -0.16 / 0.14 (1933) | +0.29 / 0.16 (22) | +0.56 / 0.01 (4) | — | +0.51 / 0.01 (3) |

## Playback and visual evidence

Prefix replay: 61,856 live records born before 101 s reproduced float-for-float in XYZ, normal, colour, opacity, covariance and birth after independently refitting the prefix calibration and stopping fusion at 101 s. The clipped death/visibility endpoints agree through 100.5 s (outside the visibility-run join tolerance). Both manifests and every object PLY were read back; the exact ground tiles, counts, bounds, unit normals/quaternions and visibility ordering passed.

[Exit chase](report/chase-exit.jpg) · [first road](report/chase-first-road.jpg) · [arterial](report/chase-arterial.jpg). Stitched 360° orbit contact sheets: [exit](report/orbit-exit.jpg), [first road](report/orbit-first-road.jpg), [arterial](report/orbit-arterial.jpg). The corresponding 12 s, 216-frame side-by-side live/stitched videos are in $RS/maps/object-v3/orbits/.

### Two wall and two pole spot checks

For each wall, nine fixed image patches lie inside the photographed façade. A double-depth flag requires at least three surfel centres per group with a >20% range gap; missing patches are not counted as correct. This identifies potential doubled surfaces, not physically surveyed walls. For poles, count live see-through retirements during the preceding 6 s whose centres reproject into the photographed pole core and agree with its nonzero reference depth within 20%; supersessions and visibility continuations are excluded. These are candidates for wrong retirement, not proof that an entire pole was retired.

- Turn wall — live: 0/6 populated patch(es) flagged; stitched: 4/9 populated patch(es) flagged.
- Straight wall — live: 0/0 populated patch(es) flagged; stitched: 0/8 populated patch(es) flagged.
- Turn antenna — 0 recent compatible see-through retirement(s).
- Straight post — 0 recent compatible see-through retirement(s).
