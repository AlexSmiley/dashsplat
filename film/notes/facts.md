# Reality Splat — fact sheet for the film

Every line is a number or event found in the repo, with its source. "path:line" is relative to the repo root.
"commit abc1234 (MM-DD HH:MM)" means the fact is in that commit's message body (git log --reverse, dates 2026).
Anything marked *(derived)* is simple arithmetic or a filename reading on my part, flagged so it can be checked.
Images marked ✓ were opened and checked; the others are described from their report's caption.

## Best numbers for the screen

- 6 cameras, 361.4 s, 3,324 m of driving, 12,927–12,937 frames per camera — commit 4ce6433 (11-21 15:45)
- Calibration from the footage alone: 0.81–0.97 px reprojection error on all six cameras — commit 6913bc0 (11-21 15:52); pipeline/calib/REPORT.md:17-22
- Lane-line edge: 4.50 cm in the splats → 1.23 cm in the road atlas, against 1.01 cm in the photograph — pipeline/groundtex/REPORT.md:16
- The whole 3.31 km road baked at 1–2 cm per pixel in 1,118 tiles, 57 MB — commit 1eb949e (11-21 20:12)
- 0 of 333.6 million road texels taken from a vehicle — commit 72cb278 (11-21 20:40)
- 18,211,768 splats in the v5 world — pipeline/stitch/REPORT-v5.md:21
- Every monocular "metric" depth model was 16–69% wrong in scale on this drive — commit be3b3f2 (11-21 17:55)
- After-image test for the re-projected photograph: 2.12% of strong edges doubled vs 0.15% for one real photo — commit 0af7e16 (11-21 19:04)
- Video diffusion (StreetNVS): 43.8 s of compute per second of video, and less faithful than its own input — commit 36515c3 (11-21 18:23)
- Image fixer (Difix-ref): 1.4–2.4 s per frame on the Mac; LPIPS 0.54 → 0.33 — commit 9493eba (11-21 20:18)
- 106 s of generated chase video published overnight at about 3 s per frame — pipeline/genfill/REPORT.md:238
- 53.1% of the ground within 40 m is predicted, every texel labelled — pipeline/groundtex/REPORT-V6.md:3
- 30 s of new footage → published scene in 35 min 52 s, no manual step — pipeline/run/README.md:96
- Power cut at 21:19; the run resumed without redoing finished stages — commit ec01f28 (11-21 21:31); pipeline/run/README.md:78
- 53 commits, first 11-21 15:45, last 11-22 04:15 — git log (commit count of git log --format=%h)

---

## 1. Input: the drive

- Six 1-minute clips, 2026-11-21 07:15:40 to 07:21:42, Jacksonville; one continuous 6-minute, 3.3 km drive — PLAN.md:34
- About 80 s parked at a drive-through window, slow car-park exit at t = 80–105 s, then arterial roads at up to 22 m/s with several 90° turns and two long stops at lights — PLAN.md:35-36
- Dawn, overcast, flat light, teal ISP tint, strong vignette, visible compression and motion blur — PLAN.md:37
- HW4 rig, Sony IMX963, rolling shutter, about 36 fps — PLAN.md:39
- Front (narrow, ~45°) 2896×1876; back fisheye and four side cameras (two B-pillar, two fender repeaters) 1448×938 — PLAN.md:43-46
- Telemetry embedded in every frame: speed, heading, lat, lon, steer, ax, ay, az, autopilot state, brake, blinkers, gear — PLAN.md:56
- No height, pitch or roll in the telemetry; acceleration has gravity removed — PLAN.md:73
- Decoded 12,927–12,937 frames per camera, 361.4 s, 3,324 m; dead reckoning from speed + heading ends 5.9 m from GPS — commit 4ce6433 (11-21 15:45) (PLAN.md:72 says 7.5 m)
- Streets (OpenStreetMap): Gate Parkway, Commerce Village Drive, Town Center Parkway — viewer/public/streets.json:9,17,33
- The red-light stop is at Burnt Mill Road — pipeline/groundtex/report/v7/README.md:3
- Sun 2.9–4.1° above the horizon at 30.25 N, 81.52 W; photographs show full overcast, no cast shadows — viewer/README.md:203-206
- Three stops where the car is exactly still: 5.67–81.83 s, 198.82–202.65 s, 257.70–286.67 s — pipeline/poses/REPORT.md:71

Images
- ✓ scratch/calib/debug/rectified-contact.jpg — six cameras rectified at the drive-through: palms, the block wall, an orange bollard; straight masonry joints.
- ✓ scratch/demo/sheet.jpg — four moments with the six-camera strip (front, pillars, repeaters, fisheye back) beside the viewer.
- pipeline/depth/report/masks_front_t335.jpg — close traffic at the lights, front camera (caption, pipeline/depth/REPORT.md:226).

## 2. Geometry: calibration, poses, depth, masks

- Calibrated from the footage alone: RootSIFT tracks within and across cameras, Ceres bundle adjustment over lenses, mounts, per-camera clock offsets and rolling-shutter readout; telemetry speed fixes scale — commit 6913bc0 (11-21 15:52)
- Fit on 164 synchronised rig frames from five windows; no learned models or factory parameters — pipeline/calib/REPORT.md:7,9
- Reprojection RMS: front 0.974, back 0.915, left/right pillar 0.815/0.823, left/right repeater 0.815/0.813 px — pipeline/calib/REPORT.md:17-22
- Clock offsets against the front camera: back −8.70 ms, right pillar −6.83 ms, left repeater −3.09 ms (others ≤1.24 ms); rolling-shutter readout 14.0–17.8 ms per frame — pipeline/calib/REPORT.md:17-22
- Split-half repeatability 0.29% focal, under 2.8 cm, under 0.15° — commit 6913bc0 (11-21 15:52)
- Held-out scale check: visual 15.97 m vs telemetry 15.64 m, +2.14% (misses the ±1% goal) — pipeline/calib/REPORT.md:50
- Poses: 6-DoF for all 12,927 master-clock samples, offline and causal (≤0.32 s lookahead) — commit 77c8a3b (11-21 16:03)
- Offline path 3,312.85 m vs fused GPS 3,319.26 m; median 0.39 m from GPS — pipeline/poses/REPORT.md:61,65
- First poses put the same road patch 0.8–6.7 m apart; pose v2 (suspension pitch/roll/heave fitted to road texture) brought held-out ground shift to 9–14 cm median — commit d0e104c (11-21 16:16)
- Depth bake-off on 13 moments of this drive; production: Depth Anything 3 Large with poses (front/rear: scale 1.04–1.05, shape error 3.9%, 0.041 s per image), MapAnything with poses (pillars/repeaters: shape error 5.5% vs 14.8% for DA3) — commit be3b3f2 (11-21 17:55); pipeline/depth/REPORT.md:12-13
- Masks: Mask2Former Swin-L (Cityscapes panoptic), 0.098 s per image; 1,148 front keyframes for 80–140 s — pipeline/depth/REPORT.md:56,251
- Moving vs parked: 472 of 3,952 objects decided (306 moving, 166 static), covering 69% of sightings — commit f6b7e9a (11-21 19:40)
- Honest real-time number: 2.84 s per six-camera keyframe, ~0.35 keyframes/s vs the 40/s needed at 20 m/s — pipeline/depth/REPORT.md:334,340

Images
- ✓ scratch/calib/debug/front-transfer.jpg — calibration check: gold measured features vs magenta points projected from another camera.
- ✓ pipeline/depth/report/prod_chase_t101.jpg — "6 cams, last 25 m fused, 7.4 M points": the car-park exit as a coloured point cloud.
- ✓ pipeline/depth/report/masks_examples.jpg — sky (blue), road (green) and numbered cars on four cameras.
- ✓ pipeline/road/report/compare-turn-only-turn.jpg — "ONLY" lettering under five trajectories; broken under offline-v1, whole under v2.

## 3. First ground: mosaic and ground splats

- Bird's-eye road mosaic at 2 cm per cell; "ONLY" lettering and turn arrows legible and single-edged — commit d0e104c (11-21 16:16)
- Ground v1 (whole drive): 15.0 M splats live and 11.4 M stitched, 93 tiles each, about 21 s to fuse and export — commit 67b67c1 (11-21 16:25)
- Cross-camera colour disagreement on held-out patches 10.3 → 6.6 levels (live) and 4.7 (stitched) — commit 67b67c1 (11-21 16:25)
- Ground v2: live 28.0 M splats, stitched 15.6 M, 132 tiles, 48 s to build — commit e30ee59 (11-21 16:38)
- Live recovery of painted strokes at 6–15 m: about 31% → 66–80%; ground exists 20/40/60 m ahead for 98/83/68% of moving time — commit e30ee59 (11-21 16:38)
- The patch under the car and the wedge between cameras are drawn from memory, not seen — commit e30ee59 (11-21 16:38)
- Viewer published on the Studio at 16:43 so the page is a built snapshot — commit 3679be9 (11-21 16:43)
- First demo: six-camera strip beside the viewer, t = 305–320 s; "ONLY" passes under the car at the same moment in camera and map — commit 01b6a94 (11-21 16:51)
- First viewer: 53–61 fps at 3200×2000 with 1.0–2.3 M splats drawn; each splat's seven time values in one texel, scrubbing costs no CPU per splat — commit e7f92d6 (11-21 16:53)

Images
- ✓ pipeline/road/report/v2/detail-only-318.jpg — front frame vs live v1 vs live v2 vs stitched v2 of the same 6–15 m strip of road.
- ✓ pipeline/road/report/v2/memory-93.7.jpg — age map under the car: green = seen <40 ms, rose/violet = remembered, with camera outlines.
- ✓ scratch/demo/sheet.jpg — first demo: road-live-v2 at 45/38/32/22 mph next to the six real cameras.

## 4. First world: objects from depth, live world v4/v5

- First static object layer (85–140 s): 149,979 live and 403,466 stitched object records; fusion at about 129 camera-frames/s — pipeline/livemap/REPORT.md:9-10
- Too sparse: objects covered 16–24% of non-road pixels live, 39–56% stitched; the right repeater's depth 15–17% short — commit 865b267 (11-21 17:02)
- Replaying to 101 s reproduced 61,856 live records float-for-float (no knowledge from the future) — pipeline/livemap/REPORT.md:196
- Live world v4 ("newest view wins"): coverage of non-sky pixels from held-out poses 92–96% per camera (v3: 16–24%) — commit 83c0eca (11-21 17:20)
- v4: 55 of 60 held-out photographs above 90% coverage; 7,817,524 records for the slice — pipeline/livemap/REPORT-v4.md:24,7
- v4 still wrong: distant foliage slides, poles float, a passing bus trails, a crossing SUV nearly absent — commit 83c0eca (11-21 17:20)
- Live v5: overhead slab at 112 s goes from 72.4% to 0% of its window — pipeline/livemap/REPORT-v5.md:19
- v5 whole-drive gate: fail (foliage turns into upright rectangular panels at 128 s); no whole-drive job started — pipeline/livemap/REPORT-v5.md:3

Images
- ✓ pipeline/livemap/report/v4/chase-arterial.jpg — v3 vs v4 at 135 s: empty horizon becomes a tree line and poles.
- ✓ pipeline/livemap/report/v4/orbit-exit.jpg — 360° orbit at the car-park exit, six angles: island tree, red fascia, smeared splats.
- pipeline/livemap/report/v5/112-+0.jpg — v4 vs v5 at 112 s, slab gone (pipeline/livemap/REPORT-v5.md:15).

## 5. Optimiser rounds v1–v5 (852 held-out real photographs unless noted)

- v1: photographs fitted on Metal (PyTorch-MPS + MetalSplat), 8 rolling-shutter row bands per frame; right-pillar p90 error 6.49 → 0.90 px — commit 866041a (11-21 17:06); pipeline/stitch/REPORT.md:22
- v1: PSNR 18.81 → 19.45 dB, SSIM 0.658 → 0.670, LPIPS 0.204 → 0.215 (worse); 4,141,060 splats; "not a photoreal orbit" — pipeline/stitch/REPORT.md:38,3
- v2 (dense start, ground frozen, judged from the viewer's chase pose): PSNR 19.50 → 19.72, SSIM .670 → .700, LPIPS .211 → .174 — pipeline/stitch/REPORT-v2.md:34
- v2: 5,140,321 splats, about 35 min for the 684 m slice; road differs from the mosaic by 0.108 levels — pipeline/stitch/REPORT-v2.md:67,65,18
- v3 (opaque crowns, grass under trees, poles): PSNR 19.716 → 19.959, LPIPS .1742 → .1677, coverage 89.93% → 93.67% — pipeline/stitch/REPORT-v3.md:30
- v3: crown opacity .211 → .688; uncovered verge 30.7% → 0.0%; 28 measured uprights; 5,033,798 splats — pipeline/stitch/REPORT-v3.md:19,18,21,22
- v4 (trained on 1,498 teacher views at the chase pose): error on 69 held-out chase stations 8.00 → 6.32 levels (better at 66/69); teacher PSNR 25.77 → 27.32 dB — pipeline/stitch/REPORT-v4.md:59
- v4: real-camera PSNR slips 19.96 → 19.74; lettering not teacher-sharp (edge contrast 0.325 → 0.277); one blank 3D frame at 121.93 s — pipeline/stitch/REPORT-v4.md:42,59,69
- v4 after-image detector on splats: v3 8.0% of strong edges flagged, v4 6.6% — pipeline/stitch/REPORT-v4.md:65
- v5 (standing world, three slices): PSNR 19.74 → 19.80, LPIPS .1714 → .1616, coverage 93.0 → 93.9% — pipeline/stitch/REPORT-v5.md:68
- v5: 8,419,622 object + 9,792,146 ground splats, 924 tiles; 85–140 s objects 3.38× v4; 78.4% of objects above opacity .8 (v4: 10.9%) — pipeline/stitch/REPORT-v5.md:88,25
- v5: the two named offices sit 305–474 m away; the viewer loaded 0 of 13 office tiles at 112 s: "Higher opacity cannot make these windows appear in this drive" — pipeline/stitch/REPORT-v5.md:54,56

Images
- ✓ scratch/opt3/viewer-112.jpg — v3 in the published viewer at 112 s (HUD "STITCHED-OPT-V3", 25 mph).
- ✓ pipeline/stitch/report-v5/112.jpg — same pose in v5: denser trees, a trunk, open lane.
- pipeline/stitch/report-v2/views-101.jpg — v1 vs v2 board at 101 s, four camera angles (pipeline/stitch/REPORT-v2.md:12).
- pipeline/stitch/report-v4/chase-110-125.mp4 — v3 | v4 side-by-side playback (pipeline/stitch/REPORT-v4.md:5).
- pipeline/stitch/report-v5/drive.mp4 and traffic.mp4 — v5 photo-off playback, 15 s each (pipeline/stitch/REPORT-v5.md:5).

## 6. Dead ends

Photographic cut-out / layer look
- ibr-v1: chase view drawn from the photographs nearest the virtual camera, as layers (sky, backdrop, upright cut-outs, movers) — commit 45470a7 (11-21 18:07)
- Remaining: pale gaps and ghosted silhouettes on the horizon, sky colour shifts, fewer near objects (edge strength 7.4 vs 12.5) — commit 45470a7 (11-21 18:07)
- First RGB-D version: "do not export this version"; p90 edge strength at 112 s source 22.0 / photo render 18.6 / splats 4.4 — pipeline/ibr/REPORT.md:3,14
- Rigid-layer version: "keep the result private"; 1,813 of 2,847 near-object identities appear in only one base keyframe, so trees change shape between photographs — pipeline/ibr/REPORT-popup.md:3,27

Re-projected photograph look ("morph")
- Every front frame warped to the chase pose; centre keeps 78% of the source's edge strength, 21% above the splats — pipeline/ibr/MORPH.md:31
- Flanks stretch over 6× on 2.0–8.9% of the image — pipeline/ibr/MORPH.md:32
- After-image gate, 110–125 s: suspected twin edges 0.15% single photograph, 1.47% published blend, 1.78% hard-cut — pipeline/ibr/MORPH.md:48
- Whole 85–140 s: 2.12% flagged (10.26% at 109.5 s) vs 0.15% — commit 0af7e16 (11-21 19:04)
- Pillar-flank round M2: 1.44% vs a 0.143% floor ("roughly ten times"); withheld — pipeline/ibr/report-m2/README.md:7
- Owner's decision: "Ryan does not want projection mapping on screen." Re-projection became a teacher for the splats instead — commit a4f6a06 (11-21 19:19)

Diffusion route (StreetNVS)
- Research verdict on Tesla: a hybrid, a 3D scene rendered and redrawn by a generator, ~65% confidence; StreetNVS first — research/05-diffusion-route.md:21-28
- Round 1: 15 s clip at 512×320, 10 fps; 10 min 57 s of inference, 43.8 s per delivered second; paint drifts median 6.1 px; 3 of 46 frames keep every patch within 3 px — commit 36515c3 (11-21 18:23)
- Round 1 vs real photos: error 15.52 (generated) vs 9.27 (its input); invented a road crest, vehicles, fog bands — pipeline/genvideo/REPORT.md:14-15
- Round 2 at native 832×480: best clip 8.95 vs input 8.62 levels; the autoencoder alone (no diffusion) beat every setting — pipeline/genvideo/round2/REPORT.md:14-15
- Round 2: 4× less stable (2.56 vs 0.57), 21.8 s per generated second; nothing published; Cosmos-Transfer2.5 package ($4–12 rented GPU) prepared, not run — commit 09f0703 (11-21 19:30)

Other rejected
- Distilling Difix into the splats: one pass changed nothing (12.14 vs 12.08) and made crowns frostier: "Do not publish fix1" — pipeline/genfill/REPORT.md:6,136
- Difix on the bird's-eye mosaic turns seams into fake paving — pipeline/genfill/REPORT.md:7
- Paint-only contrast push in v4: bolder but less correct paint, rejected — pipeline/stitch/REPORT-v4.md:34
- Grain on predicted ground: put a dot pattern on the field at 297–300 s, not published — pipeline/genfill/REPORT.md:249

Images
- ✓ pipeline/ibr/report/112-+0.jpg — v5 splats | photo-layer render | selected front photo | real photo, 112 s.
- ✓ pipeline/ibr/report-chase/road-switch.jpg — rigid layers: a roadside tree changes shape at a photo hand-over (grey box = ego).
- ✓ pipeline/ibr/report-m/110-125/stretch.jpg — morph frames at 110/112/117.5/125 s with the melting, >6× stretched flanks boxed.
- ✓ pipeline/genvideo/report/generated.jpg — StreetNVS round 1: magenta sky, fog columns, speckle.
- ✓ pipeline/genvideo/round2/report/three-column.jpg — dense input | round 1 | round 2, 110–118 s.
- pipeline/ibr/report-m/single-owner/110-125/worst-twins.jpg — annotated doubled edges (pipeline/ibr/MORPH.md:50).

## 7. Road atlas

- Road baked as a texture on the fitted ground: one source photograph per texel, registered, seams routed off paint, 1–2 cm per pixel — commit 1eb949e (11-21 20:12)
- Lane dash at 128 s, 230 transverse profiles, median 20–80% edge width: splat 4.50 cm, atlas 1.23 cm, front photograph (exposure 4590) 1.01 cm; contrast 30 / 52 / 44 levels; crops sampled at 1 cm — pipeline/groundtex/REPORT.md:16,9
- Whole drive, 3,312 m: 1,118 tiles of 10.24 m (822 at 1 cm, 296 at 2 cm); 57,084,343 bytes, 1.72 MB per 100 m — pipeline/groundtex/REPORT.md:50
- Driven corridor opaque on 99.95% of distance — pipeline/groundtex/REPORT.md:44
- Held-out photo error 4.82 levels vs a 2.94 floor between two real photos (85–140 s) — pipeline/groundtex/REPORT.md:26
- Doubled paint edges: 1.71% flagged vs 0.031% for a real photo — pipeline/groundtex/REPORT.md:33
- Front-strip registration: matched points 10.07 → 1.38 cm — pipeline/groundtex/REPORT.md:52
- v5: dash edge 1.26 cm vs 1.04 cm photo vs 4.18 cm v5 splats; flags 1.71% → 0.83%; 1,160 textures, 43.5 MB — pipeline/groundtex/REPORT-V5.md:18,31,9
- v5: zero of 136.2 M + 49.7 M + 147.7 M texels touch a vehicle or removal mask — pipeline/groundtex/REPORT-V5.md:45
- v6: ground out to 40 m with 2,052 new 4 cm tiles, fading to zero at 44 m — pipeline/groundtex/REPORT-V6.md:20,28
- v6: grey void pixels at 300 s 4,121 → 0; 53.1% of covered ground ≤40 m predicted (142,001 m² predicted vs 125,328 m² observed) — pipeline/groundtex/REPORT-V6.md:11,24
- Three bars at the red light: one 0.58 m stop line and two 0.30/0.29 m crosswalk lines, all in single pillar photographs at 198.99/199.00 s — "Changed tiles: none" — pipeline/groundtex/report/v7/README.md:3,5,7

Images
- ✓ pipeline/groundtex/report/detail-lane.jpg — the lane-dash triptych: v4 splats | atlas at 1 cm | one photo.
- ✓ pipeline/groundtex/report/v6/before-118.jpg / check-118.jpg — published viewer at 118 s: grey void beside the road, then continuous ground.
- ✓ pipeline/groundtex/report/v7/199-stationary-pillars.jpg — raw pillar crops showing all three stripes at once.
- pipeline/groundtex/report/v6/ground-rays-118.jpg — overlay: green observed, amber predicted ground (pipeline/groundtex/REPORT-V6.md:14).
- pipeline/groundtex/report/v5/road-strip-286-330.jpg — straightened road overview (pipeline/groundtex/REPORT-V5.md:39).

## 8. Vehicles

- Movers v1: 17 vehicle episodes (4 at 80–140 s, 4 at 190–215 s, 9 at 286–330 s); no moving person confirmed — pipeline/movers/REPORT.md:3
- Bus within 5.3 px (live) / 2.9 px (hindsight) median of its photographed centre; crossing SUV 13.7 / 4.7 px — pipeline/movers/REPORT.md:24-25
- Test set: 136 held-out mask exposures; live had no track for 22, poor overlap on 34 more — pipeline/movers/REPORT.md:36
- Movers v2: bus passes 23.6–25.5 m to the left; tyre contacts within 1.93 px (pillar) and 0.52 px (repeater) — pipeline/movers/ROUND2.md:7,9
- v2: every live step ≤1.091 m per 36 Hz frame; 2,747 tentative identities → 917 — pipeline/movers/ROUND2.md:63,65
- No pedestrian or cyclist in 594 junction frames — pipeline/movers/ROUND2.md:71
- Movers v3: unseeded live tracking aligns 80 of 136 (reviewed baseline 77–80); first detection → renderable 0.79 → 0.44 s; max step 1.0911 m across 113 episodes — pipeline/movers/ROUND3.md:14,20,42
- Removal masks for 22,268 shutters (7,469 + 9,403 + 5,396) — pipeline/movers/ROUND3.md:50-52
- Parked blue SUV classified parked on 149 of 149 delayed decisions — pipeline/movers/ROUND3.md:58
- On the v5 world: 23 moving episodes and nine fixed vehicle models — pipeline/movers/report4/README.md:3
- Clean models: saloons, coupes and Teslas drawn as the Model 3; SUVs, buses etc. as procedural bodies; no third-party models — viewer/README.md:221-225
- Scanned models dropped as default "because they read as blocky close to the camera and broke up in the generated clip" — commit c9cf385 (11-22 00:26)

Images
- ✓ pipeline/movers/report2/bus-viewer-comparison.jpg — viewer (left) vs dashcam (right): the bus hidden by splat trees in the viewer.
- ✓ pipeline/movers/report4/pull-away-287.jpg — scanned black coupe and grey Tesla under the overpass, blocky.
- ✓ scratch/night/n2-287.jpg — same moment with clean vehicle models.
- pipeline/movers/report2/turning-black-coupe-before-after.jpg — coupe model at eight angles filling in (pipeline/movers/ROUND2.md:15).

## 9. Image model (Difix-ref / genfill)

- Model: NVIDIA Difix-ref (nvidia/difix_ref), a single-step SD-Turbo fixer that takes a render plus one reference photograph; non-commercial licence — pipeline/genfill/REPORT.md:3,14
- Inputs: the splat render from the chase pose + the nearest training photograph (median 9.7 m away at the default chase view) — pipeline/genfill/REPORT.md:38,48
- Speed: 1.4–2.4 s per 1408×896 view on the Studio's MPS — pipeline/genfill/REPORT.md:32
- 60 held-out photos: LPIPS 0.541 → 0.326, PSNR 21.13 → 20.61 dB, faint lane paint found 24% → 59% — pipeline/genfill/REPORT.md:61-62
- Repaired views disagree with neighbours by 10.3 levels vs 3.8 for plain renders — pipeline/genfill/REPORT.md:80
- Stabilisation: fixed seed, held reference photo (switches ≤ every 2 s), previous frame warped forward by depth, observed atlas road put back exactly — pipeline/genfill/REPORT.md:163-167
- Flicker 0.81–1.16 levels vs 1.36–1.59 for the real front camera — commit 9a3d01a (11-21 23:47); pipeline/genfill/REPORT.md:185-186
- Provenance mask under every frame: 255 observed, 170 repaired, 85 filled, 0 none — pipeline/genfill/REPORT.md:168; viewer/README.md:170-173
- Night clips: 93–140 s (1,411 frames), 190–215 s (751), 296–330 s (1,021) — pipeline/genfill/REPORT.md:212,226-228
- The c093/c190/c296 names are not in the reports *(derived from filenames scratch/night3/g093/g190/g296-sheet.jpg and pipeline/genfill/report-clip/clip-093-140 etc.)*; they match those three stretches
- Stopped at the light (199–203 s): frames differ by 0.30 levels — pipeline/genfill/REPORT.md:230
- Cost: about 3 s per frame, 2.8 h for 106 s — pipeline/genfill/REPORT.md:238
- 03:00–04:15 repair: near-camera objects dropped whole, purple post / pink column / signal mast removed by hand, haze band at 315–328 s filled; 1,145 frames redone — pipeline/genfill/REPORT.md:244-247,259; commit 3b7a42b (11-22 04:15)

Images
- ✓ pipeline/genfill/report/15-opt3-112-chase.jpg — render | Difix-ref repaired | reference photo | change map, 112 s.
- ✓ scratch/night3/g093-sheet.jpg — 15 frames of the 93–140 s clip in the viewer (Generative mode, 6 → 45 mph).
- ✓ pipeline/genfill/report-clip/viewer-playback-107-122.jpg — two moments, three frames each, in the published viewer.
- ✓ pipeline/genfill/report-clip/viewer-repaired-307-324.jpg — after the 04:15 repair, through the 90° turn and "ONLY".
- pipeline/genfill/report-clip/clip-093-140-side-by-side.mp4 — render vs repaired, side by side.

## 10. Finished view

- Ego car lit by an environment built from the scene and the frame on screen, captured four times a second — viewer/README.md:188-198
- Overcast contact shadow, no sun shadow (a sun shadow would be 20–30 m long) — viewer/README.md:204-209
- One grade for splats and video, measured from the asphalt: gains 1.16, 0.94, 0.94 — viewer/README.md:72-76
- Generative mode: playback enters and leaves clips automatically; 0.25 s fade in, 1 s fade out, adjacent clips join with no fade — viewer/README.md:128-136
- Sync: exact frame when paused or scrubbing; at 1× 73% exact, otherwise the neighbour — viewer/README.md:141-142
- Chase camera matched to Tesla's frames: 16.2 m, 16.9° up, horizon 28% from the top — viewer/README.md:36,114
- Verified 60 fps, worst 1% of frames 21.5–23 ms — commit 2e87073 (11-21 22:28)

Images
- ✓ scratch/night3/default-morning.jpg — finished Generative view at 46 mph: trees, poles, clean grade.
- scratch/now/g6-118.jpg — published-page still at 118 s (not opened).

## 11. Pipeline

- One command: drive.py run data/clips --name hw4-drive --quality full — pipeline/run/README.md:30
- Preview tier on fresh footage (224–254 s): 425.61 s = 7 min 06 s; 2.89 M ground splats, 114 atlas tiles, 28 vehicle tracks — pipeline/run/README.md:90; commit a522a80 (11-21 23:36)
- Full tier, same 30 s: preview + full 2,152.39 s = 35 min 52 s of stage time — pipeline/run/README.md:96
- "71 min 45 s per selected minute" — sum of successful stage times, not elapsed time with queues — pipeline/run/README.md:96 *(≈72× the driving time, derived: 2,152 s ÷ 30 s)*
- Full scene: 2.9 M splats in 161 tiles, 36 tracks, a 4 s clip at 233–237 s; final resume check skipped all 17 finished stages — commit 4aba1fc (11-22 02:48); pipeline/run/README.md:128,130
- Power cut at 21:19 EST: the unattended full-tier run restarted and raw stages were cache hits — pipeline/run/README.md:78
- Viewer host did not come back after the power cut (started by hand); a launchd agent now restarts it — commit ec01f28 (11-21 21:31)
- Lock records PID, process start and kernel boot time; after a reboot rerun the same command — pipeline/run/README.md:48
- Whole drive: 249.53 moving seconds, 14 fit windows, 70 clip chunks — pipeline/run/README.md:60
- Whole-drive preview published 04:50 EST: 13.6 min per moving minute — pipeline/run/README.md:66
- 06:30 checkpoint: 49.86 s of full geometry, 32.86 s of video published; 8 h 15 min projected to finish — pipeline/run/README.md:72,74
- Rig check: stored rig kept unless pooled error exceeds 5.36 px; proof window scored 1.53 px — pipeline/run/README.md:118

Images
- ✓ scratch/night3/full-234-gen.jpg — automatic full-tier scene "HW4-RAW-224-254", Generative, red lights, blob cars.
- scratch/night3/full-228.jpg, full-245.jpg, hw4-drive-100.jpg — the automatic scenes (not opened).

## 12. Known remaining flaws

- The sky's clouds are invented (coded filled) — pipeline/genfill/REPORT.md:236
- The black coupe is drawn as the Model 3 (coupes use the Highland Model 3) — viewer/README.md:221 *(derived: the reports don't call it a stand-in)*
- No clip under the overpass (286–296 s): it stays a blobby wall, the road dark and streaked — pipeline/genfill/REPORT.md:212
- Still visible after 04:15: pale crowns 101–102 s, blue patch 102–106 s, stepped grey fill 110–118 s and 296–300 s, a line at 317–326 s, an invented apartment block 322–325 s — pipeline/genfill/REPORT.md:252-257
- Lettering legible only where it is atlas road paint (e.g. "ONLY" at 318 s) — pipeline/genfill/REPORT.md:236
- Garbled sign faces in the automatic pipeline — pipeline/run/README.md:62
- Named offices never visible from the default chase — pipeline/stitch/REPORT-v5.md:56
- Ground atlas above the single-photo floor for doubled paint — pipeline/groundtex/REPORT-V5.md:35
- Rear camera depth good only to about ±15% — pipeline/depth/REPORT.md:154
- Automatic full tier uses the v4 recipe: speckled trees, floating splats — commit 4aba1fc (11-22 02:48)
- Not tested on a real iPhone — commit e7f92d6 (11-21 16:53)

Images
- ✓ pipeline/stitch/report-v5/290.jpg — the overpass in v5: scalloped deck, streaked road.
- ✓ scratch/night/n2-287.jpg — clean cars at the overpass stop (no generated clip there).
