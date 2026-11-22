# Chase-trained splats · v4 · 85–140 s

**I would not show this as a finished comparison with Tesla’s screen.** The 110–125 s drive has a convincing strip of asphalt, and paint is generally steadier as it approaches the car. The final recording also caught a one-frame loss of the entire 3D scene at 121.93 s; the HUD stayed visible. The roadside tells a different story: foliage goes soft and shifts as it passes; thin skyline features lose their supports; the distant white office at 128 s is barely there. The scene is usable as a moving reconstruction demo, not photographic ground truth.

[Play the published viewer at 110 s, photos off](http://10.0.50.51:8090/?scene=stitched-opt-v4&photo=off&t=110) · [Watch v3 | v4, 110–125 s](report-v4/chase-110-125.mp4). The side-by-side uses actual 1× Chrome playback for both scenes, aligned by the viewer’s recorded drive clock. No frame interpolation, offline render or image overlay is used for the scenery.

## Published viewer: compare the same pose

| Pose | v3 / v4 screenshots | Reading |
|---|---|---|
| 93 s | [v3](report-v4/v3-93.jpg) / [v4](report-v4/v4-93.jpg) | The island kerbs and the tan frontage are recognisable. The frontage has dark openings but not a reliably straight, fully resolved roofline. The nearby turn arrow is soft. |
| 100 s | [v3](report-v4/v3-100.jpg) / [v4](report-v4/v4-100.jpg) | The two turn lanes are legible. The yellow line at the near island is improved over the first v4 pass, but an existing apron gap remains at left. The near-right crown has little leaf texture. |
| 112 s | [v3](report-v4/v3-112.jpg) / [v4](report-v4/v4-112.jpg) | The centre dashed lines are easier to separate; right-verge turf meets the trunks. This is the strongest default-view improvement. The far lamps and frontage do not resolve. |
| 118 s | [v3](report-v4/v3-118.jpg) / [v4](report-v4/v4-118.jpg) | The fork markings and road remain single, coherent surfaces. Lane paint is more even than v3 in motion, but the turn-arrow head and outer kerb still blur. |
| 128 s | [v3](report-v4/v3-128.jpg) / [v4](report-v4/v4-128.jpg) | The median and nearby dashed lanes remain readable. The left foreground crown is smoky; the far white office never reads as a building with windows. |
| 135 s | [v3](report-v4/v3-135.jpg) / [v4](report-v4/v4-135.jpg) | Exposes the limit: overhead signal shapes, palm crowns and gaps around the junction are not a coherent skyline. No synthetic wires or lamps were added to conceal this. |
| 101 s, +60° | [v3](report-v4/v3-101-orbit.jpg) / [v4](report-v4/v4-101-orbit.jpg) | Orbiting gives real parallax, with the same severe foreground crown blur as v3. |
| 128 s, +60° | [v3](report-v4/v3-128-orbit.jpg) / [v4](report-v4/v4-128-orbit.jpg) | This caught the most important regression: the teacher warp carved through real trunks. A final guard restored grounded vertical structures from v3. Near and far rows remain distinct; intervening terrain and fine branches are incomplete. |

The published viewer was paused at each pose, photo layers disabled and runtime-checked, with ~15 s to stream tiles at 1440×900. The grade, haze and display antialiasing were left alone. The effect of extra capacity is not a visible hex grid; the remaining giveaway is missing or smeared scenery. Detail beyond the viewer's distance fade is invisible even when the independent compositor can measure it.

## What was fitted

The published map is one static, degree-zero Gaussian field and the photographed 1024×512 live-v5 sky. The camera, car, route ribbon, natural grade, tile loader and haze come from the shipping viewer. There are no per-frame textures, photographic cut-outs, projection layers or generated frames in the display. Orbiting changes the view of the same Gaussians.

1. **Photographic teachers.** Project the centres of v3 splats into the original 1440×900 RGB, confidence, sky and centimetre-depth arrays. Higher confidence and shorter base distance get more weight; offset views get less weight in the direct fit. Compare depth before accepting colour. Use the nearest high-confidence witness bounded by the multi-view distribution. Prune contradicted object opacity rather than placing a second surface. Depth makes a weak, bounded adjustment to supported object centres; no ground height is optimized. A first full-resolution sweep updated 3,329,503 ground colours and 261,956 object colours. A second sweep sampled newly split ground children at their own centres.
2. **Corridor detail.** The broad first subdivision replaced 1,470,195 ground parents with four coplanar children each. It improved held teacher error but was wasteful on plain asphalt and locally washed out paint. Retain 575,240 subdivisions with measured intra-footprint contrast, restore disputed native paint where it does not have a second nearby teacher edge, and merge the rest back into their original tangent footprints. On 409 held-in default stations, back-project the high-frequency error of rendered paint onto the same grounded centres. Require repeated observations with agreeing residual signs before correcting them; 726,372 matching centres received a correction. Four children replace a parent; they do not form a second ground layer.
3. **Objects and thin silhouettes.** Teacher-depth patches need real-camera surface support, three displaced teacher observations, distinct base photographs and consistent normal and colour. Of 396,659 candidate cells, 2,347 survived. A separate real-image pass tested whether neighbouring points across broad existing footprint axes actually had different colour. Across 1,665 held-in images, 57,193 coarse footprints qualified for a two-child replacement; unsupported texture was not added. The warp's sky and depth masks removed 217 elements of the v3 photo-validated uprights. Restore those and reinforce 425 weakened elements from the measured v3 shafts. These are 28 tracked uprights, not 28 classified lamps; there is no claim that all lamp heads or wires were recovered.
4. **Real-camera anchor and generation.** Optimize anti-aliased native-resolution crops against the render using RGB, image gradients, an object-opacity penalty on known sky, and a weak depth moment. The first 5,600 steps used 2,566 real-camera and 3,034 teacher crops. The second 2,400 used 908 real, 1,310 teacher and 182 Difix crops; 816 repaired views remained eligible from the completed 1,023-view chase-opt3 set. Difix has a 0.035 weight, agreement ≥220/255, a class/change/coverage gate, and a height gate >1.15 m above the ground. Its gradients also require object-dominated pixels; its update cannot change ground parameters, object geometry or opacity. Held-out real references and held-out teacher stations are excluded from its appearance loss.
5. **Published-viewer iteration.** The mixed-camera crop pass introduced pink/green patches on the otherwise neutral road at 112 s. Keep the teacher/mosaic low-frequency colour and retain supported fine luminance detail from the raster fit. In the same asphalt rectangle (x=492–589, y=490–733), the mean absolute deviation of red–green from its median was 1.77 levels in v3, 3.36 after mixed-view fitting, and 1.39 in the corrected checkpoint. No viewer grading or renderer setting changed. This last correction is to the asset's road colours, not a screen-space filter.

**Orbit correction.** Three trunk-base crops in the 128 s +60° orbit had object alpha .603/.945/.726 in v3 and zero in the mixed-view v4. Restore measured v3 splats only on grounded columns occupied at successive heights, matching existing elements first and excluding deliberately subdivided parents. This reinstated 9,491 missing elements and raised opacity on 50,905 weakened ones. The same crops now measure **.603/.940/.548**, and the published orbit shows the trunks again. Final asset: **5,570,788 ground + 1,207,602 object splats**, 419 static tiles, 224 MB compact cache. All published tile payloads matched the independently verified candidate; the sky, rig and drive are unchanged.

**Stopping point.** A final paint-only contrast experiment raised gradient recovery on 37 *training* views from .217 to .294 but increased RGB error on strong paint edges from 15.62 to 16.12 levels (ground overall 5.21 → 5.39). It was rejected. Pushing harder on the same soft or misregistered evidence makes bolder paint, not more correct paint. To materially improve the foliage and skyline, the useful additional input is confident, displaced photographic support for those surfaces specifically.

## Held-out real cameras

Same six lenses, physical exposure split, native row timing, valid static pixels and reference-selected LPIPS crop as v3. Each entry is v3 → v4. Lower LPIPS is better.

| Physical exposures | n | PSNR (dB) | SSIM | LPIPS | Coverage |
|---|---:|---:|---:|---:|---:|
| All | 852 | 19.96 → 19.74 | 0.6952 → 0.6912 | 0.1677 → 0.1714 | 93.67% → 92.97% |
| Clean | 361 | 20.15 → 19.90 | 0.7099 → 0.7059 | 0.1560 → 0.1601 | 94.02% → 93.26% |
| front | 142 | 19.88 → 19.77 | 0.7268 → 0.7435 | 0.1205 → 0.1057 | 93.77% → 92.57% |
| back | 142 | 20.56 → 20.43 | 0.7953 → 0.7992 | 0.1757 → 0.1727 | 94.38% → 94.04% |
| left pillar | 142 | 20.44 → 20.12 | 0.6540 → 0.6358 | 0.2295 → 0.2325 | 92.96% → 92.96% |
| right pillar | 142 | 18.78 → 18.32 | 0.6006 → 0.5747 | 0.2000 → 0.2198 | 92.00% → 90.54% |
| left repeater | 142 | 20.06 → 19.92 | 0.7156 → 0.7154 | 0.1366 → 0.1456 | 94.96% → 94.81% |
| right repeater | 142 | 20.03 → 19.88 | 0.6790 → 0.6786 | 0.1440 → 0.1521 | 93.92% → 92.89% |

The aggregate native-camera result is slightly worse, not a collapse. The front lens improved in LPIPS and SSIM; the right pillar shows the biggest perceptual regression.

## Held-out teacher views

Every tenth station **and all its offsets** was held out: 1,021 training views at 613 stations; 477 test views at 69 stations. Three held default views share a base photo with a neighbouring training teacher; the last row removes them. None of the teachers' base or neighbouring front frames belong to the native holdout. These are independent *poses*, not independently photographed chase-height ground truth. Scores are per-view means at confidence ≥.5, not whole-screen errors.

| Views | n | RGB MAE (0–255) | Ground MAE | Above-ground MAE | PSNR (dB) | Ground edge contrast / teacher | Lower MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| default | 69 | 8.00 → 6.32 | 7.41 → 5.84 | 19.34 → 15.31 | 25.77 → 27.32 | 0.325 → 0.277 | 66/69 |
| offset | 408 | 7.72 → 6.11 | 7.13 → 5.64 | 19.04 → 14.98 | 26.04 → 27.62 | 0.332 → 0.281 | 391/408 |
| nonshared default | 66 | 7.63 → 6.15 | 7.02 → 5.66 | 19.33 → 15.40 | 25.94 → 27.41 | 0.324 → 0.274 | 63/66 |

Default confidence-weighted MAE at all nonzero weights: **8.09 → 6.48**; mean composited alpha on flagged sky: **.0082 → .0016**. Ground means within 34 cm of measured ground height. The strong-ground-edge gradient ratio measures contrast at the teacher's edge locations; 1.0 matches the teacher. It went **down**, even while colour error and ordinary playback stability improved. The result is not teacher-sharp lettering.

**After-images:** In 23 held default views, the detector looks for same-facing outlines 3–30 pixels apart with tangent support, where only one outline matches the single photograph. v3: **1,677/20,963** sampled strong render edges flagged (8.0%); v4: **595/8,985** (6.6%). Self-comparison: 0/29,605; injected 9 px twin: 1,772/15,750. This is a suspected-pair diagnostic, not a count of actual ghost objects. The detector also finds fewer strong edges in v4, so fewer flags do not prove that blur or all after-images are gone.

## Motion, including the failure

Both scenes played in the published photo-off viewer from 110 through 125 s at 1×. The deliverable is **450 frames, 15.00 s, H.264, 1280×426, 2,885,203 bytes**. Synchronisation selects actual captured frames against each drive clock; maximum matching error was 18 ms for v3 and 39 ms for v4. The whole-scene blink at 121.93 s is **left in the clip** ([before / during / after](report-v4/motion-pop.jpg)). The final v4 recording shows one blank 3D frame with HUD intact; the matched v3 recording shows none. The evidence doesn't distinguish a viewer render hiccup from a Chrome capture hiccup.

Flow-compensated absolute luminance residual (0–255 levels), every third decoded frame at half CSS resolution. Flow must be consistent in both directions; fixed regions avoid the HUD, car and guidance ribbon. A blurrier scene can also score lower. Values include the blink.

| Region | Median v3 → v4 | Mean v3 → v4 | Worst v3 → v4 |
|---|---:|---:|---:|
| road | 2.47 → 1.86 | 2.48 → 3.26 | 3.03 → 103.65 |
| verge | 2.79 → 2.72 | 2.85 → 4.25 | 4.37 → 113.69 |
| distant | 1.80 → 1.63 | 2.01 → 2.97 | 4.36 → 72.45 |

Paint generally holds its place as the car accelerates; the grass and tree edges still shimmer, the 115–117 s pole is ragged, and the far tree line resolves and fades abruptly. There is no sustained screen-space projection sliding across the road. The single-frame interruption makes averages worse despite the better typical road frame.

## Extension: generate 286–330 s first

The published scene still covers **85–140 s**. Checked scheduled keyframes against actual posed depth arrays, producer-window metadata, and road/sky/dynamic masks: **968–971 complete records per lens at 286–330 s**, and **302 per lens at 190–215 s**. The only teacher manifest on Studio covers 85–140 s. No additional slice met all three conditions.

For 286–330 s, the official 1,321-frame, 1440×900, 30 Hz chase track is already on Studio at `work/stitch-v4/extension-286-330/chase-track.json`. Generate `work/ibr/teacher-286-330/manifest.json` with a default view at each metre (~632 stations), the same six perturbations every fifth station, single-photo RGB, confidence, sky, centimetre depth, intrinsics/extrinsics and base/neighbor exposure provenance. Include front posed depth and road/sky/dynamic masks for the **280.5–286 s pre-roll** so the first chase views can look back to their source; none of the 11 indexed pre-roll exposures currently has these in either depth location. Front moving-object removal exists on all 971 in-window front keyframes. Keep a fresh physical and teacher-station split, and select ground tiles covering this new corridor. No new six-lens depth run for the main 286–330 s window is needed.

## Reproduce the selected scene on Studio

Use the v2/v3 training caches and fixed physical split from [the v3 report](REPORT-v3.md). From the Reality Splat root, export `PYTHONPATH=$PWD/envs/stitch/lib/python3.12/site-packages` and use the project's `.venv/bin/python`. All output directories below are fresh; sources are immutable. `S=scene/stitched-opt-v3` and `W=work/stitch-v4`. `run` in the example invokes a Python module.

```sh
S=scene/stitched-opt-v3; W=work/stitch-v4
run() { name=$1; shift; .venv/bin/python -m "pipeline.stitch.$name" "$@"; }
run v4teacher "$S" "$W/subdivided" 1
run v4teacher "$W/subdivided" "$W/teacher-fit" 0
run v4detail
run v4detail "$W/teacher-fit" "$W/first-published"
run v4canopy "$W/first-published"
run v4corridor "$W/first-published" "$W/budget-published"
run v4edges "$W/budget-published"
run v4fit "$W/first-published" "$W/joint-A" 5600
run v4corridor "$W/joint-A" "$W/joint-budget"
run v4canopy "$W/joint-budget" "$W/joint-B"
run v4edges "$W/joint-B" "$W/phase-fit"
run v4fit "$W/phase-fit" "$W/final-fit" 2400 --gen
run v4finish "$W/final-fit" "$W/teacher-fit" "$W/finished"
run v4roots "$W/finished" "$W/rooted"
run v4proof "$S" v3
run v4proof "$W/rooted" rooted native
run v4outline "$S" v3; run v4outline "$W/rooted" rooted
```

Publish the verified `rooted` directory as `scene/stitched-opt-v4/`. From basebook's `viewer/` run `bun run cache:studio`. Browser captures use `?scene=stitched-opt-v4&photo=off&t=<seconds>&paused=1`; the two orbits add `&orbit=60,0,1`. Start the recording paused at 110 and play at 1× through 125. Full-size recordings, per-exposure scores, fit logs and intermediate scenes are on Studio under `work/stitch-v4/`.
