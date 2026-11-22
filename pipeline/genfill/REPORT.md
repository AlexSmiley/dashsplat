# Generative repair of novel views: Difix-ref on Apple Silicon

**Decision.** NVIDIA's **Difix-ref** (`nvidia/difix_ref`: single-step SD-Turbo fixer that takes a render plus one reference photograph) runs unmodified on the Studio's MPS at **1.4-2.4 s per 1408x896 chase view** and is packaged in `fix_views.py` with per-pixel change, class and cross-view agreement maps, and a pseudo-view manifest in the stitch optimiser's existing `--pseudo` format. It turns our soft renders into clean, photoreal pictures, mostly by importing the reference photo's appearance into the render's layout. It also invents: road arrows, paving texture under smeared ground, bare trees inside near-camera foliage blobs. Each repaired view is about **three times less consistent with its neighbours** than the plain render (10.3 vs 3.8 RGB levels after warping 2 m). It is usable for distillation only through the agreement-weighted mask, at weak weight; see the recommendation at the end. Generated content is **predicted**, never observed.

- **On the best scene (stitched-opt-v3, chase-track poses)** it does what we hoped to the defects we have: dark smoky crowns become leafy trees with trunks in the same places, smeared poles become clean poles (with wires borrowed from the reference), purple fringes and specks go, asphalt and grass gain texture, the blank band at the horizon becomes sky. 93 % of checked rendered pixels agree with neighbouring repaired views (83 % on live-v4). It also invents: sidewalks and kerbs on verges, building lettering ("DO" for "DD"), palms where the far field is haze.
- **One distillation pass through the optimiser's existing `--pseudo` interface changed nothing measurable** (`scene/stitched-opt-v3-fix1` vs the no-pseudo control `scene/stitched-opt-v3-fix0`: 12.14 vs 12.08 RGB levels from the repaired targets), and the refit itself made v3's crowns frostier in the viewer. Section 8 lists the four changes the optimiser needs.
- **Ground mosaics in bird's-eye space: no.** Paint does not move (<= 0.66 px), but the mosaics' real defects are seams and brightness bands, which Difix hardens into "paving panels"; Real-ESRGAN only blurs ([ground/REPORT.md](ground/REPORT.md)).
- **The image-based prototype is a worse fixer input than the splats**: Difix leaves its sharp shards untouched.

## 1. Survey (November 2026)

| Candidate | Conditions on | Steps, backbone | Resolution | Licence | Weights | Without CUDA |
|---|---|---|---|---|---|---|
| **Difix-ref / Difix** (NVIDIA, CVPR 2025) | render + one reference image (cross-view attention); Difix: render only | 1 step, SD-Turbo UNet, t = 199 | trained 512 / 576x1024; any multiple of 64 | NVIDIA License, **non-commercial** (+ SD-Turbo licence) | HF, ungated, 5.2 GB | **Yes**: plain diffusers 0.25; ran on MPS unmodified |
| NVIDIA **Fixer** (Nov 2025) | render only | 1 step, Cosmos-Predict2 0.6B DiT + Wan VAE, t = 250 | 576x1024 | NVIDIA Open Model License (commercial OK) | HF, ungated, 4 + 1.4 GB | **No** in practice: code imports `transformer_engine`, needs `cosmos-predict2==1.0.9` (megatron-core, triton; flash-attn/natten on CUDA) inside NVIDIA's CUDA container. In-domain (trained on AV 3DGUT renders) but no reference input. Not ported |
| **ArtiFixer** (NVIDIA, SIGGRAPH 2026) | render video, opacity mixing with observed content | few-step causal autoregressive video; Wan2.1 1.3B / 14B or Cosmos3-Nano | video | weights NVIDIA non-commercial, code Apache-2.0 | HF, ungated, 6.7 GB (1.3B) / 68 GB (14B) | No: Linux + Ampere/Hopper/Blackwell, CUDA containers. The natural consistency upgrade on a rented GPU |
| **GSFixer** (ICML 2026) | render video + reference views (VGGT geometry, DINOv2 features) | many-step video diffusion, CogVideoX-5B-I2V (+BLIP-2) | short clips | no licence file in repo; CogVideoX licence | HF, ungated, 14 GB | Impractical: about 51 GB and 146 min per scene on an A100 (ArtifactWorld, arXiv 2604.12251) |
| GSFix3D (3DV 2026) | render (+ mesh render), per-scene fine-tune | multi-step latent diffusion | scene-specific | see repo LICENSES | code Nov 2025, models partly "coming soon" | Its 3DGS part needs a CUDA rasteriser; needs per-scene fine-tuning |
| 3DGS-Enhancer (NeurIPS 2024) | render video + references | many-step SVD | 512 | n/a | code was "coming soon" | not pursued |
| Driving-specific: OneFixer (42dot, Sep 2026), ConFixGS (May 2026), SPVC (Aug 2026), ReconDreamer DriveRestorer (CVPR 2025) | lane/actor states, boxes + HD map, or world model | one-step AR video to many-step | n/a | n/a | OneFixer, ConFixGS, SPVC: paper only, no code/weights found; DriveRestorer CUDA video | no |

Difix-ref is the only open, ungated (render, reference) single-step image fixer that preserves the render's layout and runs on MPS. Fixer is the in-domain alternative if we rent CUDA; ArtiFixer if per-frame inconsistency proves fatal.

## 2. Bring-up on MPS (M5 Ultra, shared GPU)

| Setting | Seconds per image | Metal driver memory |
|---|---:|---:|
| Difix-ref, fp16, 1024x576, idle GPU (bench) | **1.63** (first call 13.6) | 13.3 GB (3.0 GB tensors) |
| Difix-ref, fp32, 1024x576, idle GPU | 1.57 | 23.5 GB (6.1 GB tensors) |
| Difix-ref, fp16, 1024x576, with depth inference running | 2.1-3.0 | 13.5-14.4 GB |
| Difix (no reference), fp16, 1024x576, contended | 1.4-1.8 | 8.1 GB |
| Difix-ref, fp16, **1408x896** chase, contended | **1.4-2.4** (3.6-3.9 under heavy contention) | 30-32 GB |

fp32 is no faster and doubles memory; fp16 shows no NaNs or visible difference. Contention came from depth inference, other engineers' proof renders and a video-prep job. **Volume:** the 85-140 s slice is 342 stations x 5 chase views = 1,710 views, about 45-70 min per pass on the Studio. The whole 3.3 km drive is about 8,300 views per pass; a Difix3D+-style progressive schedule of about four passes is about 33,000 fixes: **13-22 h on the Studio**. CUDA fallback (not rented): Difix runs 76 ms per 576x1024 frame on an A100 (paper), roughly 0.15-0.2 s at 1408x896, so the four-pass whole drive is about 1.5-2 GPU-hours, **about $2-3** at research/05's $1.39/h A100. Renting only pays for the video fixers (ArtiFixer, GSFixer), at an estimated tens of dollars per drive.

## 3. Render sets and reference photographs

`views.py` renders from contract scenes with the shared rasteriser (`maps/object-v3/raster.dylib`, static tiles only, sky panorama behind the splats as in the viewer, no haze) and picks, for every view, the training photograph (stitch fit split, never held-out) that sees most of its rendered surface from a similar direction and distance.

- **Chase**: the viewer's default camera from `viewer/tools/chase-track.ts` (16.2 m from the pivot, 16.9 deg up, 5.3 deg tilt, 50 deg vertical FOV), one station per 2 m of travel, 342 stations for 85-140 s, plus orbits of 30 and 60 deg either side through a Python replica of `placeChase` (agrees with the tool to 0.08 mm and 0.0001 deg over the slice). 1408x896 (1440x900 on Difix's 64-px grid), at each station's playback time.
- **Real**: 60 held-out exposures (10 per camera; the live-map hold set, which live-v4 never fused, minus ground-scheduled frames), pinhole crops at 1024x576 with eight rolling-shutter bands, photo as truth; references at least 2 m away ("near") or 10 m away ("far").

How far the best photograph is from each view, which is what limits conditioning:

| Set | Mean reference score | Median reference distance |
|---|---:|---:|
| Real, near (>= 2 m) | 0.52 | 3.1 m |
| Chase default (mid) | 0.46 | 9.7 m |
| Chase orbit -30 / +30 | 0.30 / 0.29 | 13.9 / 12.6 m |
| Real, far (>= 10 m) | 0.26 | 11.2 m |
| Chase orbit -60 / +60 | 0.14 / 0.17 | 16.0 m |

So the default chase view is conditioned between the two real-pose settings, the 30-degree orbits like "far", and the 60-degree orbits worse than anything we can score.

## 4. Real poses: how much it helps and how much it invents (live-v4, 60 held-out photos)

Static, non-sky, valid pixels; "rendered" = splat alpha >= 0.5, "holes" = the rest. Paint = bright road markings within 45 m (white top-hat), recall/precision against the photo's paint within 3 px.

| Output vs photo | PSNR | PSNR rendered | PSNR holes | SSIM | LPIPS | LPIPS holes | Paint recall / precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| Render | 21.13 | 21.30 | 16.78 | 0.719 | 0.541 | 0.656 | 0.24 / 0.52 |
| Difix-ref t199, near ref | 20.61 | 20.71 | 17.10 | 0.660 | **0.326** | 0.420 | **0.59** / 0.54 |
| Difix-ref t99, near ref | 21.06 | 21.22 | 16.66 | 0.695 | 0.407 | 0.514 | 0.46 / 0.51 |
| Difix-ref t199, far ref | 19.59 | 19.69 | 16.57 | 0.634 | 0.392 | 0.469 | 0.57 / 0.45 |
| Difix-ref t99, far ref | 20.87 | 21.04 | 16.48 | 0.693 | 0.424 | 0.530 | 0.45 / 0.49 |
| Difix, no reference, t199 | 19.12 | 19.27 | 14.87 | 0.567 | 0.494 | 0.555 | 0.66 / **0.32** |

On the sparse stitched-opt-v1 scene (different hold set) the pattern was the same: LPIPS 0.485 -> 0.288 at t199, PSNR +0.2 dB.

- It **looks** much closer to the photo (LPIPS -40 % near, -28 % far) while matching it **pixel for pixel slightly worse** (-0.5 dB near, -1.5 dB far at t199): plausible detail in roughly the right place, not the right detail. t99 is the gentle setting: PSNR within 0.3 dB of the render, LPIPS -22 to -25 %.
- Faint real paint becomes visible (recall 0.24 -> 0.59) without more false paint on average (precision 0.52 -> 0.54), but individual cases go wrong: a smeared, wavy line is sharpened as a wavy line ([03](report/03-real-fail-wrong-paint-sharpened.jpg)), and arrow fragments are copied in from the reference where the road is blank ([04](report/04-real-fail-arrow-invented.jpg)).
- Without a reference the prior alone hurts everything and invents paint (precision 0.32): never use plain Difix here.

## 5. Chase poses on live-v4 (no truth; 40 stations at 96-103 s and 129-131 s, 200 views)

Cross-view consistency: warp each view into its neighbour through the rendered depth (occlusion-tested, rendered pixels only) and compare. The render's own error is the baseline (view-dependent splats, resampling).

| Mean absolute difference after warping (RGB levels) | Render | Difix-ref t99 | Difix-ref t199 |
|---|---:|---:|---:|
| 2 m of travel, same pose type (190 pairs) | 3.8 | 5.6 | 10.3 |
| ... default pose / orbit 30 / orbit 60 | 3.3 / 3.3 / 4.3-4.8 | 4.8 / 4.9-5.4 / 6.3-6.5 | 8.5 / 9.1-10.2 / 11.8-12.1 |
| Default pose vs 30-degree orbit, same station (80 pairs) | 4.2 | 6.1 | 11.0 |

Per-pixel classes after the consistency pass (`consistency.py`, tolerance 10 levels of excess over the render's own disagreement), fraction of all pixels:

| Pose | Agreeing (of rendered, checked) t199 / t99 | Kept | Repaired | Fixer-only | Sky |
|---|---:|---:|---:|---:|---:|
| Default | 0.86 / 0.99 | 0.56 / 0.78 | 0.09 / 0.01 | 0.19 / 0.07 | 0.15 |
| Orbit -30 / +30 | 0.82-0.84 / 0.96-0.98 | 0.54-0.56 / 0.76-0.79 | 0.09-0.12 / 0.01-0.02 | 0.20 / 0.06-0.08 | 0.13-0.15 |
| Orbit -60 / +60 | 0.77-0.79 / 0.96-0.97 | 0.49-0.57 / 0.79 | 0.10-0.15 / 0.01-0.02 | 0.21-0.23 / 0.06-0.08 | 0.12-0.13 |

Road paint relative to the render (45 m around the car): t199 keeps 89 % of the render's paint but 42 % of its paint pixels have no render paint within 3 px (t99: 99.5 % kept, 21 % new). Some of that is faint paint made legible; some is invented, e.g. a straight arrow at the exit ([06](report/06-chase-0024-mid.jpg)). `fix_views.py` now marks any new road paint as fixer-only.

What it does by region, from looking at the sheets ([06](report/06-chase-0024-mid.jpg)-[11](report/11-consistency-orbit.jpg)):
- **Road and lane lines**: lines stay where they are; asphalt gets texture. Smeared ground becomes paving slabs or invented markings.
- **Far field and sky**: the biggest visible win. Floating far fragments become plausible signals, palms and office blocks, largely in the reference photo's style; sky becomes clouds. This is exactly the content that is least consistent between views (bright edges in [10](report/10-consistency-travel.jpg)).
- **Foliage**: blotchy crowns become leafy trees in the same places; leaf texture differs per frame. A smear right in front of the camera (orbiting into a roadside tree) stays a smear or becomes an invented tree ([08](report/08-chase-0030-orb+60.jpg), [09](report/09-chase-0236-orb-30.jpg)).
- **Buildings**: become clean façades at the right place and size; windows and lettering are invented.
- **Cars, signs, lettering**: no whole cars were added in the views inspected; sign faces and building lettering are invented when the render has a blob there.

## 6. Which input repairs better: splats or the image-based prototype

Nine frames of `pipeline/ibr/report/{road-105-120,exit-93-104}.mp4` and live-v4 splat renders from exactly the same camera (the controller replay that made those videos), same reference photo ([12](report/12-ibr-vs-splat-112.jpg), [13](report/13-ibr-vs-splat-97.jpg)). Difix-ref leaves the prototype almost untouched (mean change 0.024, median CIELAB difference 2.6) while it rebuilds the splat renders (0.156, 5.3): it treats sharp photographic shards as clean, so tears, seams, the pink cast and the grey car box all survive. **The splat render is the better fixer input.** Torn-but-sharp input would need a fixer trained on reprojection artefacts.

## 7. The best scene: stitched-opt-v3 at the chase-track poses

1,023 repaired views: every 2 m of 85-140 s, default pose and 30-degree orbits either side (the 60-degree orbits are excluded from distillation: their best photo is 16 m away). 2.0 s per view median, 44 min wall for the slice with other GPU jobs running.

| Pose | Agreeing (of rendered, checked) | Kept | Repaired | Fixer-only | Sky | Mean change on rendered |
|---|---:|---:|---:|---:|---:|---:|
| Default | 0.93 | 0.63 | 0.04 | 0.11 | 0.23 | 0.07 |
| Orbit -30 / +30 | 0.90 / 0.93 | 0.58 / 0.65 | 0.06 / 0.04 | 0.13 / 0.11 | 0.24 / 0.20 | 0.11 / 0.07 |

Against live-v4 the fixer has much less to invent (mean change 0.07 vs 0.18; fixer-only 0.11 vs 0.19). Defect by defect, from the slice and the sheets [14](report/14-opt3-100-chase.jpg), [15](report/15-opt3-112-chase.jpg), [16](report/16-opt3-128-chase.jpg):

| Defect | What the fixer does | Trust |
|---|---|---|
| Soft, dark, smoky foliage | Leafy crowns with visible trunks and branches, same position and size | Shape yes; leaf texture differs between neighbouring views (amber/red in the class maps) |
| Smeared poles | Clean straight poles at the same place; wires and lamp heads copied from the reference photo | Pole yes; wires and heads predicted |
| Floating specks, purple fringes | Removed or absorbed into foliage | Yes |
| Soft asphalt, grass | Fine texture; lane lines stay put | Texture predicted, lines kept |
| Foggy far field, holes below the horizon | Clouds, palms, office blocks in the reference's style | Predicted (fixer-only), never geometry |
| Verges and kerbs | Invents sidewalks and hard kerbs | No |
| Signs and building lettering | Redrawn plausibly but wrong ("DO" for "DD" at 89.7 s) | No |

## 8. Distillation into the splats: one pass, minimal version

`pipeline/stitch/v2fit.py` already takes weighted pinhole pseudo-views (`--pseudo`), so I ran it unmodified (`distil.py fit`) from `scene/stitched-opt-v3`, 14,400 steps as published, with the 1,023 views (mean weight 0.55: agreement map, zero on fixer-only and sky), and the same run without pseudo-views as a control. Both scenes are in `$RS/scene/` (`stitched-opt-v3-fix1`, `stitched-opt-v3-fix0`) with v3's ground, sky and map keys, and the viewer cache is rebuilt. 873 s and 924 s, peak 3.3 GB.

| Scene | Weighted error to the repaired views, 128 poses | Error to v3 renders |
|---|---:|---:|
| stitched-opt-v3 | 10.11 | 0 |
| fix0: refit, no pseudo | 12.08 | 5.02 |
| fix1: refit + pseudo | 12.14 | 5.01 |

In the published viewer ([17](report/17-viewer-distil.jpg), 100/112/128 s, unretouched captures at half size) fix1 is indistinguishable from fix0, and both refits are worse than v3: crowns frostier at 100 s and smokier at 128 s. **The fixer content did not reach the splats, and re-running the full refit on v3 undoes v3's crown and pole work. Do not publish fix1.** What the optimiser needs to change, in order:

1. **Cadence and gain.** v2fit draws a pseudo view on 1 update in 16 at 0.10 of a photograph's loss: about 900 weak patch updates against 13,500 photo updates. Make both arguments; start at 1 in 3 and 0.4 on weight-1 pixels.
2. **A polish mode on v3** instead of a refit: train DC colour, opacity and small scale changes only, with no XYZ freedom and no split/clone/prune, so v3's crowns, trunks and poles survive. Alternatively apply the pseudo loss inside `v3build` before its final pruning and pole insertion.
3. **Progressive rounds** (Difix3D+): render the polished scene at the same poses, repair, re-run `consistency.py`, distil again; two or three rounds, about 1.5 h each on the Studio.
4. **Provenance.** Splats whose colour or opacity is driven mainly by pseudo pixels must go to a `...-predicted` layer. That needs a per-splat accumulator of pseudo versus photo gradient, which v2fit does not have.

Unchanged and already right: pinhole pseudo poses with one timestamp; pseudo pixels never vote on depth, exposure, sky or densification.

## 9. Ground texture and small general models

Run by a helper, details in [ground/REPORT.md](ground/REPORT.md). Three 30.7 m patches at 2 cm/px (turn arrows 93-101 s, bike lane 112 s, "ONLY" 318 s), tiled with feathered overlap. Paint never moved (phase correlation 0.07-0.19 px mean, 0.66 px max). Difix without a reference sharpens paint (the second "ONLY" becomes crisp) but turns keyframe seams into hard "pavement panels", redraws stroke ends and changes 15-24 % of road pixels by more than 10 levels; with a reference (only the same mosaic exists) it changes little and looks no better. The alternative, **Real-ESRGAN x2plus** (BSD-3, one pass, runs on MPS, 0.2-0.3 s per 512-px tile), acts as a mild blur. On the chase renders it changes 1.7-2.4 levels and fixes nothing ([sheet](ground/report/chase-render-difix-esrgan.jpg)). The mosaics' real defects are seams and brightness bands, which belong in `pipeline/road/mosaic.py` (seam-aware blending), not in a generator; nothing needs re-exporting. Instruction-edit models with a reference input were not run: those I found were gated or 12-20 B parameters, not a cheap test on MPS.

## 10. Recommendation

- **Distil, but not yet**: Difix-ref at timestep 199, one reference, through the agreement-weighted manifest, at a pseudo weight well below a photograph's, only after the optimiser changes in section 8. Use t99 if a run must be strictly conservative: it keeps PSNR within 0.3 dB of the render and agrees with neighbours 96-99 % of the time, but adds little.
- **It will fix**: crown shape and trunks, smeared poles, fringes and specks, asphalt and grass texture, the sky band. **It will not and must not be trusted for**: road paint and lettering (new paint is forced to fixer-only), signs and building text, anything within a few metres of the camera (orbiting into roadside trees), the far field and holes below the horizon (always predicted), ground-mosaic seams.
- **Viewpoint gap**: at the default pose the best photo is about 10 m away and conditioning still works; at 30-degree orbits (13-14 m) it is like the "far" real test (PSNR -1.5 dB, LPIPS still -28 %); at 60 degrees (16 m, score 0.14-0.17) it is mostly invention, so those poses are excluded.
- **If per-frame inconsistency limits distillation** after the changes above, the upgrade is a video fixer (ArtiFixer 1.3B) on a rented GPU, not a different image model.


## 11. Generated chase clips for the viewer's Generative mode

`clip.py` turns the default chase view into a repaired, stable video that the viewer plays under the car, ribbon and HUD (`mode=generative`). Two clips are in both v5 scenes (`generated/genfill-chase-110-125.mp4`, `generated/genfill-chase-286-300.mp4`, entries in `generated.json` with `"provenance": "below"`), and the Studio cache is rebuilt.

**Pipeline, per frame** (poses from `viewer/tools/chase-track.ts` at 30 fps, 1440x900, vfov 50; rendered on a 1472x960 canvas with the same focal length because Difix needs multiples of 64, then cropped):
1. **Input**: sky panorama, ground splats, the ground atlas v6 (1 cm road, 4 cm ring to 40 m, with its observed/predicted label), object splats depth-tested against the ground, tracked vehicle models at their interpolated poses on 286-300 s. No ego car.
2. **Reference**: the best-scoring training photograph, held: it changes only to a later photo, at most every 2 s, when one scores 1.25x better. Mostly the front camera 10-16 m behind; 7 switches in 110-125 s (14 without the hold).
3. **Holes**: below the horizon with no surface inside 40 m, the input starts from the previous repaired frame carried through the local ground plane, else the reference photo projected onto that plane, else a low-frequency fill. Beyond 40 m it keeps the render's horizon haze, as the viewer does. With atlas v6 such holes fell to 0.0 % of the frame on both stretches.
4. **Difix-ref**, timestep 199, fixed seed and noise.
5. **Feed-forward**: the previous repaired frame, warped by the render's depth (sky at infinity), is blended into the input (0.5) and output (0.5) wherever the warp is visible and agrees with the render; into filled and atlas-predicted ground at 1.0 in and 0.8 out; into the sky only at the output (0.7). Feeding the sky into the input grew round pale blobs; that was removed.
6. **Truth back on top**: wherever the atlas is opaque and observed, the atlas road is put back exactly (lane paint cannot move, appear or vanish; recall and precision against the atlas are 1 by construction). Atlas-predicted ground keeps the atlas colour and only 45 % of the fixer's texture, because the fixer turned it into slabs and planks.
7. **Provenance** per frame (`<tag>-prov/`, and in the video under the picture): 255 observed (atlas road or render kept), 170 repaired, 85 filled (ground the atlas predicted, holes, and sky the fixer redrew), 0 none.

**Stability.** Flicker = mean RGB-level difference between consecutive frames after moving the previous frame onto the current one, inside 1440x900. "Depth" warps by the render's depth and pose (sky at infinity, holes on the ground plane). "Flow" uses DIS optical flow computed on the sequence itself, which is the only one that can be applied to the real front-camera video.

| Setting (110-113 s, ground atlas v5) | Depth | Flow |
|---|---:|---:|
| Plain splat render | 0.48 | 0.50 |
| Difix t199, per frame, fixed seed, held reference | 1.93 | 1.82 |
| ... plus feed-forward (0.5 in, 0.5 out) | 1.28 | 1.36 |
| ... t149 with feed-forward | 0.82 | 0.90 |
| ... t99 with feed-forward | 0.87 | 0.90 |
| Real front-camera video, same seconds | | 1.55 |

At t149 and t99 the fixer barely changes the picture, so the delivered clips use **t199 with feed-forward**, which keeps the look of sheets 14-16 and flickers less than the real camera.

| Delivered clip (full length, atlas v6) | Render depth / flow | Repaired depth / flow | On filled pixels | Real video flow |
|---|---:|---:|---:|---:|
| 110-125 s, 451 frames | 0.46 / 0.47 | **0.81 / 0.90** | 1.49 | 1.36 |
| 286-300 s, 421 frames, tracked vehicles drawn in | 0.55 / 0.57 | **1.16 / 1.22** | 0.56 | 1.59 |

Holding the reference cut switches from 14 to 7 (110-125 s) and from 66 to 7 (286-300 s, flow flicker 1.42 -> 1.22).

**Completeness and provenance** (mean share of each 1440x900 frame):

| Clip | Observed | Repaired | Filled | of which atlas-predicted ground / holes | Sky (redrawn, coded filled) | Worst frame filled |
|---|---:|---:|---:|---:|---:|---:|
| 110-125 s | 68 % | 1 % | 5.4 % | 5.3 % / 0.0 % | 25 % | 9.1 % at 117.4 s |
| 286-300 s | 70 % | 4 % | 8.7 % | 8.7 % / 0.0 % | 17 % | 12.2 % at 295.9 s |

"Repaired" counts pixels whose colour or structure changed noticeably (CIELAB distance after a 1.5 px blur); the fixer's sharpening of trees and poles mostly stays below that threshold. No white or grey voids remain inside 40 m. The worst-fill frames ([110](report-clip/clip-110-125-worst-fill.jpg), [286](report-clip/clip-286-300-worst-fill.jpg)) show what fills are: smooth, plausible fields beyond the verges, steady in motion but without detail. Beyond 40 m the ground fades into the horizon haze, as in the viewer.

**In the published viewer** (`?mode=generative`, played at 1x and recorded, frames in [110](report-clip/viewer-playback-110.jpg) and [286 + Predicted overlay](report-clip/viewer-playback-286-and-predicted-118.jpg)): the clip plays under the car, ribbon and HUD, in sync. 110-125 s reads as a coherent street from behind the car: leafy trees, poles, grass, clouds and a road that is exactly the atlas. What breaks the illusion:
- **Lettering**: the 286-300 s overpass carries invented, garbled text ("NEXIC GaoloAGINIAN") and a redrawn green sign. Signs are wrong whenever they are large in frame.
- **Tracked vehicles**: a badly reconstructed model at 286-289 s stays a blob; Difix does not turn it into a car. Clean models (290 s) look fine. If the viewer also draws its own mover overlay on these scenes, vehicles appear twice; the manifest needs a flag for "vehicles already in the clip".
- **Residual shimmer**: foliage edges and invented sky texture. It is lower than the real camera's frame-to-frame change by these measures, but it is visible in places. The reference switching every 2 s shifts the style subtly.
- **Soft fields** where the atlas predicted ground.
- Not seen in sampled frames, but not systematically measured: invented vehicles off the road. On the road the atlas is put back, so none can appear there.

**Cost.** Render 1.0 s per frame (8 CPU workers, atlas sampling included), reference choice 0.3 s, repair 1.8-2.3 s (Difix 1.5-1.7 s on MPS with other jobs running), encode and checks about 0.4 s: **about 3.5-4 s per frame end to end, 25-30 min per 15 s clip**. A whole-drive clip (361 s, 10,840 frames) would take about 11-12 h on the Studio; the map covers only 85-140, 190-215 and 286-330 s today. **Live?** Difix runs 76 ms per 576x1024 frame on an A100 (paper). Our 1472x960 frame is 2.4 times larger: about 0.18 s on an A100 and roughly 0.1 s on an H100-class GPU, so **5-10 fps on one high-end GPU at this resolution**. That rises to about 13-25 fps at 576x1024; the splat render and warps add milliseconds. Not 30 fps without a smaller or distilled fixer or fixing every few frames and warping in between.

Everything per frame (renders, depth, references, repaired frames, provenance, flicker) stays on the Studio under `work/genfill/clip/c110v6` and `c286v6`; all stages are resumable (frames whose outputs decode are skipped).

## 12. Night run: published clips, 22 November

Live in both v5 scenes (`generated.json`, `"provenance": "below"`, `"vehicles": "overlay"`): **93-140 s** as three continuous clips (93-108, 108-123, 123-140, cut from one run so the last repaired frame carries across the cuts), **190-215 s** (190-205, 205-215) and **296-330 s** (296-311, 311-326, 326-330). **286-296 s has no clip**: the overpass stays a blobby wall, the road beneath is dark and streaked, and a stopped car's splats in the static map remain a blob even at timestep 99. The old 110-125 and 286-300 clips are withdrawn.

Changes to the input render, all in a per-stretch copy (the published tiles are untouched):
- **Near the chase camera**: object splats fade out with distance from the camera, gone inside 5.5 m and full beyond 9 m. Splats in a 2.2 m tube from the camera to the car are dropped.
- **Contradicted by the photographs**: each object splat is projected into front and pillar keyframes around the stretch (every third, 60 m range). A splat is dropped if it lands on sky in at least 60 % of its 3 or more views, or on a vehicle instance in at least half. A whole upright column (0.75 m cells, more than 1 m above the road) is dropped when its views see sky behind at least 20 % of the time. A lower per-splat threshold only broke smeared poles into floating fragments. Dropped: 19,723 of 3.99 M splats on 93-140 s, 8,172 of 3.22 M on 296-330 s, 2,656 of 1.11 M on 190-215 s. No splat is elongated beyond 1.8:1 (99.9th percentile), so nothing was shrunk.
- **No vehicles**: tracked models are no longer drawn in; splats that the photographs show as vehicles are dropped.

Changes to the repair:
- The observed atlas counts as truth only within 9 m of the driven path. Off it (car parks, side roads) it is ordinary ground.
- On all ground that is not the observed corridor, thin bright strokes are removed after repair, so the fixer cannot write lines and garbled car-park paint is wiped.
- Atlas-predicted ground keeps 25 % of the fixer's texture.

| Stretch | Frames | Flicker, depth / flow (render) | Real front video, flow | Filled (worst) | Sky | Reference switches |
|---|---:|---:|---:|---:|---:|---:|
| 93-140 s | 1,411 | 1.11 / 1.26 (0.51 / 0.60) | 1.73 | 4.4 % (10.4 % at 125.8 s) | 25 % | 21 |
| 190-215 s | 751 | 0.91 / 0.94 (0.37 / 0.36) | 1.26 | 5.3 % (16.2 % at 214.2 s) | 23 % | 8 |
| 296-330 s | 1,021 | 1.13 / 1.14 (0.47 / 0.46) | 1.63 | 6.4 % (11.9 % at 320.8 s) | 24 % | 17 |

While stopped at the light (199-203 s) consecutive repaired frames differ by 0.30 RGB levels on average (max 1.0): the picture holds still. Played in the published viewer at 1x under the car and HUD ([107-122 s](report-clip/viewer-playback-107-122.jpg), [197-206 s](report-clip/viewer-playback-197-206.jpg), [315-326 s](report-clip/viewer-playback-315-326.jpg)). Still wrong, by time:
- **106-108 s** (exit bend): a roadside tree passing through the 5.5-9 m fade zone shows as a translucent grey ghost at the left edge for about a second, with a dark wedge below it.
- **110-111 s**: a smeared purple post on the right verge. It is about 10 m from the camera, so it is not faded, and the photographs do not reject it.
- **112-125 s**: the pale median lamp post at about 18 m is a real pole and stays, crisper than before.
- **296-330 s**: far-field trees float as dark smudges above the horizon at 317-326 s. The turn holds (reference from the front camera throughout; score dips to 0.14 at 322.5 s). The fixer invents an apartment block on the right at 322-325 s.
- **190-215 s**: the stop line is drawn as three parallel bars; that is the observed atlas there, kept as truth. The right verge has odd green shapes at 213-215 s.
- **Everywhere**: the sky's clouds are invented (coded filled). Lettering is only legible where it is atlas road paint (e.g. "ONLY" at 318 s).

**Cost**: render 0.8 s per frame (8 CPU workers, the photo test cached per stretch), repair 1.9-2.3 s (Difix 1.4-1.6 s on MPS beside the pipeline engineer's run), encode and checks about 0.2 s: **about 3 s per frame, 2.8 h for the 106 s published tonight** with renders overlapping repairs. Side-by-side videos and contact sheets: [93-140](report-clip/clip-093-140-contact.jpg), [190-215](report-clip/clip-190-215-contact.jpg), [296-330](report-clip/clip-296-330-contact.jpg).

## 13. Targeted repairs, 22 November 03:00-04:15

Only the worst seconds were re-rendered and re-repaired. Each run was seeded with the published frame just before its range (so the carry-forward continues) and spliced back with a 0.5 s cross-fade at both ends; the replaced frames are in `<tag>-orig/`. Only the clips containing those frames were re-encoded and republished: 093-108, 108-123, 205-215, 296-311, 311-326 and 326-330.

- **Close to or over the camera, as whole objects.** Object splats are grouped into connected pieces: 0.5 m voxels, more than 1 m above the road, split into 8 m cells so a row of touching crowns is not one piece. A piece is not drawn in a frame if any of its splats is inside 7 m of the camera, in the camera-to-car tube, above the camera or in the top third of the frame inside 15 m, or seen from nearer than 0.6x the closest photograph of it. A per-splat fade left translucent ghosts and cut crowns flat, so it was replaced. Gone: the signal arm across the top at 208-210 s, the trunk at the right edge at 308 s, the tall near pole at 319-320 s, the ghost tree at 106-108 s and the near tree and dark smear at 103-105 s.
- **By hand** (one object each, world x, y, radius): the purple post at 110-111 s (-88.05, 40.48, 0.6 m), a pink column at the right edge at 110 s (-77.28, 39.03, 0.5 m) and the smeared signal mast at 320-322 s (-1556.1, 885.2, 0.8 m). The mast was 15 m from the camera, just outside the close-range rule.
- **Washed-out crowns** (101-105 s): crown splats closer in colour to the sky than to the scene's typical foliage, and brighter, take the colour of their 16 nearest non-pale neighbours. This was only partly effective: the left crowns at 101-102 s are still pale.
- **Floating trees above the horizon** (317-326 s): the far trees were not floating splats (they reach the ground in their cells). They looked afloat because the ground beyond 40 m was horizon haze. For 315.5-328 s the holes out to 150 m are filled with plain ground colour plus the fixer's fine grain only. Without that limit the fixer drew a railing along the horizon. The reference there is scored with far points weighted 3x; it now switches 5 times in that range.
- **Not changed.** The apartment block at 322-325 s is still there (it follows the reference photographs). The stop line at 197-205 s is real paint (confirmed by the ground engineer).
- **Item 5, not published**: quiet grain on the atlas-predicted ground (110-119 s, 296-301 s). At 110-119 s the change is barely visible; at 297-300 s it put a regular dot pattern on the left field. Kept the published frames.

Played in the published viewer at 1x after each republish: [100-111 s and 205-211 s](report-clip/viewer-repaired-100-111-and-205-211.jpg), [307-324 s](report-clip/viewer-repaired-307-324.jpg). Still wrong:
- **101-102 s**: pale crowns on the left.
- **102-106 s**: a blue, water-like patch on the right verge where the near tree's ground was never seen.
- **106 s**: grey fill with odd structure on the left where trees were removed.
- **110-118 s and 296-300 s**: staircase-edged flat grey predicted ground.
- **317-326 s**: a smudgy line where the filled band meets the observed ground.
- **322-325 s**: the apartment block.

**Cost**: about 3 s per re-done frame end to end, the same as the night run. Tonight's repairs re-did 1,145 frames (385 + 135 + 90 + 375 + a 375-frame redo of 315.5-328 s). The item-5 trial added 435 frames that were not used.

## Images

Each under 500 KB. Real: render | repaired over truth | reference. Chase: render | repaired over reference | change classes (amber repaired, red fixer-only).
[02](report/02-real-buildings-from-reference.jpg) buildings from the reference (good) · [03](report/03-real-fail-wrong-paint-sharpened.jpg) wrong paint sharpened (failure) · [04](report/04-real-fail-arrow-invented.jpg) arrow invented (failure) · [05](report/05-real-far-reference.jpg) same view, reference 10 m away · [06](report/06-chase-0024-mid.jpg) exit, invented arrow · [07](report/07-chase-0240-mid.jpg) arterial (good) · [08](report/08-chase-0030-orb+60.jpg) 60-degree orbit into planting (failure) · [09](report/09-chase-0236-orb-30.jpg) near trunk smear survives · [10](report/10-consistency-travel.jpg) / [11](report/11-consistency-orbit.jpg) neighbour disagreement, render vs repaired · [12](report/12-ibr-vs-splat-112.jpg) / [13](report/13-ibr-vs-splat-97.jpg) prototype vs splats as input. · [14](report/14-opt3-100-chase.jpg) / [15](report/15-opt3-112-chase.jpg) / [16](report/16-opt3-128-chase.jpg) stitched-opt-v3 at 100/112/128 s · [17](report/17-viewer-distil.jpg) published viewer: v3, refit, refit + pseudo-views.
