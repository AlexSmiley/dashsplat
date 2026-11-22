# Reality Splat · 85–140 s photograph fit

**Verdict: useful optimisation, not a photoreal orbit.** The exported scene has **4,141,060 splats** in 173 verified tiles. On all 852 reserved physical images, PSNR gained **0.65 dB** and SSIM gained **0.012**; perceptual crop LPIPS worsened by **0.011**. Road geometry survives, and several false foreground patches disappear. The 101 s orbit still intersects broad sheets of foliage, and distant buildings are missing. Do not scale this pass expecting a photoreal whole drive.

## Why this trainer

**Chosen: custom constrained PyTorch-MPS trainer with MetalSplat's differentiable 3DGS kernels** (MIT, pinned at `84e8c3a433f71c107e9387d3f72a765436c11f6e`). Model 3 poses never move. We import each contract Gaussian's position, DC colour, opacity, covariance, quaternion and visibility. A native 5 m spatial index makes the MPS workload a frustum crop rather than a multi-million-row Adam update. Ground tangent scales/colour/opacity can change; orientation and thickness stay fixed. Normal displacement is bounded. Objects can be added, split or removed only with geometric/photographic evidence. The viewer reads the actual optimised DC; there are no hidden higher-order colours or exposure parameters needed at display time.

**Actual-data bake-off.** Eight rectified front photos near 93 s, 336×216, 41,691 of the *same* ground centres and colours within 30 m, fixed poses, 300 warm-timed steps, CPU image readback every 30 steps. This deliberately small fixture tests import and control. The stock msplat 1.1.4 Python trainer ran **574 steps/s**, 18.59 dB on the in-sample static mask. The MetalSplat DC/opacity-only comparator preserved the imported discs and ran **182 steps/s**, 16.72 dB. Not an apples-to-apples loss: stock msplat can change geometry, trains all pixels, and reconstructs scale/rotation/opacity from a point seed. The latter deliberately freezes that geometry and fits the mask. The stock API accepted the real pinholes but exposed no contract-Gaussian initialiser, layerwise anchors, depth/exposure losses or gradient/mask hook. A faster in-sample fit is insufficient for this orbit. Brush and OpenSplat were reviewed, not benchmarked; the decision does not rely on invented performance figures for them. The authoritative application timings are the multi-million-splat rows below.

## Camera and shutter proof

Physical test frames are assigned **before** rectification. Training uses one 44° front pinhole, one 80° pinhole for each pillar/repeater, and three 83° rear pinholes yawed −52°, 0°, +52°. Eight native sensor-row bands each get the pose of their middle exposure row; a rectified pixel's *physical* row chooses its band. Fitting virtual horizontal strips instead would be wrong especially on the yawed back view. Colour/exposure are calibrated before supervision. Learned residual exposure is a physical-frame scalar (bounded log gain ±0.04); it is never fitted to test frames or applied to the chase camera.

Reproject actual offline BA landmarks to original measured native features, after transporting the stored v1 landmarks into the v2 vertical gauge through the observation camera. These are in-fit points: the **exact-row residual is the calibration floor, not an independent localisation accuracy claim**. Numbers below are native pixels. Virtual-direction-to-native round trip is <1.1×10⁻¹² px, as expected analytically; the meaningful experiment is against the measured features and row poses.

| Lens | BA observations | Feature residual p90, midpoint → 8 bands → exact | Shutter-only error vs exact p90, midpoint → 8 bands |
|---|---:|---:|---:|
| Front | 18,878 | 5.65 → 4.08 → 4.01 | 5.53 → 0.66 |
| Back | 20,380 | 3.18 → 2.98 → 2.96 | 1.23 → 0.20 |
| Left pillar | 29,346 | 5.75 → 3.21 → 2.92 | 4.85 → 0.95 |
| Right pillar | 29,552 | 6.89 → 3.25 → 2.99 | 6.49 → 0.90 |
| Left repeater | 23,646 | 3.76 → 3.35 → 3.32 | 1.73 → 0.35 |
| Right repeater | 23,946 | 4.19 → 3.45 → 3.38 | 2.13 → 0.56 |

## Held-out native photographs

Reserve keyframe ordinals 7, 15, … **on each lens**, never using their RGB for the optimiser, stereo births, clearance or residual exposure. Exclude training depth whose producer window touched a reserved exposure. There are 142 reserved physical images per camera, **852 total**. The already-fused input and calibrations pre-exist this split. A second, stricter column removes images found in v2 ground's exact rescheduled physical exposures or any v3 object input window. It is still same-drive testing with shared pose/photometric calibration, not unseen-drive generalisation. Uniform per-frame mean, both models evaluated with the independent CPU native fisheye renderer and 33 sensor-row poses. Sky, body and moving instances are excluded from RGB metrics; uncovered static pixels are counted, and no render-coverage mask or per-test gain is allowed. SSIM is measured only on complete 11×11 valid windows. LPIPS is SqueezeNet on a wholly valid, central 96×96 reference-selected crop; strictly a local perceptual measure. Both perceptual comparison renders are quality-88 JPEGs; PSNR/SSIM are computed before JPEG.

| Lens (142 each) | PSNR dB ↑ | SSIM ↑ | LPIPS ↓ | Clean n | Clean PSNR dB ↑ | Clean SSIM ↑ | Clean LPIPS ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| front | 19.02 → 19.32 | 0.688 → 0.693 | 0.158 → 0.183 | 63 | 18.87 → 19.16 | 0.694 → 0.700 | 0.145 → 0.172 |
| back | 19.86 → 20.61 | 0.776 → 0.794 | 0.176 → 0.207 | 70 | 20.06 → 20.75 | 0.789 → 0.807 | 0.164 → 0.196 |
| left pillar | 18.17 → 19.08 | 0.594 → 0.602 | 0.297 → 0.296 | 57 | 18.11 → 19.02 | 0.606 → 0.614 | 0.280 → 0.283 |
| right pillar | 18.12 → 18.51 | 0.565 → 0.573 | 0.251 → 0.252 | 57 | 18.41 → 18.78 | 0.578 → 0.587 | 0.237 → 0.237 |
| left repeater | 18.66 → 19.49 | 0.685 → 0.696 | 0.167 → 0.178 | 57 | 18.69 → 19.51 | 0.698 → 0.707 | 0.145 → 0.161 |
| right repeater | 19.01 → 19.70 | 0.639 → 0.662 | 0.173 → 0.173 | 57 | 19.09 → 19.76 | 0.652 → 0.675 | 0.156 → 0.158 |
| **Frame-weighted mean** | **18.81 → 19.45** | **0.658 → 0.670** | **0.204 → 0.215** | **361** | **18.91 → 19.54** | **0.674 → 0.687** | **0.186 → 0.201** |

All six lenses gain in PSNR and SSIM, including the clean subset. Valid static pixels above 50% rendered opacity increase from 87.6% to 89.3%. **LPIPS regresses**, most on the front and rear crops. Those figures do not support a claim of improved fine photographic texture.

Camera plates: [front](report/camera-front.jpg), [rear](report/camera-back.jpg), [left pillar](report/camera-left_pillar.jpg), [right pillar](report/camera-right_pillar.jpg), [left repeater](report/camera-left_repeater.jpg), [right repeater](report/camera-right_repeater.jpg). Plates were selected by proximity to stated drive times, not best score.

## Novel views: what changed, what did not

At each instant the car and map are fixed; the proof camera circles at a 6 m radius and 3 m height. No car mesh, synthetic sky or original background photograph is composited. The dark clear colour makes missing geometry visible. Exact pairs, 96 azimuths per pose (8 s, 12 fps), were rendered from exported PLYs. The baseline used during training was the completed all-six-lens staging snapshot: its index and all 403,466 object records matched the subsequently published `scene/stitched-v3` byte-for-byte; ground tiles were the same files.

- **93 s:** The palm/building boundary across the junction separates better; a foreground smear near the left island recedes. Turn arrows remain identifiable, not sharper in every stroke. Geometry still misses the far building outline.
- **101 s:** The centre opens enough to see the trees and roadway behind the former blob. At 90° the red-roofed structure becomes more visible, and at 180° the large foreground patch shrinks. At 0° and 270° a broad, translucent-looking sheet still crosses the image. This is the clearest failure: the camera gains are real but do not fix canopy depth from above. The left-pillar example gains 2.72 dB and *still* does not recover the real trunk or branches.
- **112 s:** A floating horizontal patch near the skyline mostly disappears and the dashed lane remains flat. The distant building visible in the front photograph is still absent. Broken verges and foliage silhouettes remain.
- **128 s:** The right verge is less purple and a few detached roadside patches disappear. The orbit still sees through the verge, and tree backs and distant scenery are incomplete. Broad cool stripes in the road are more apparent in places; appearance optimisation does not recover asphalt grain.

On 24 clean native exposures nearest the four sites (six lenses each), mean splat opacity over eroded observed sky fell **22.2% → 14.7%**, and composited object opacity over eroded observed road **1.91% → 1.23%**. In the 101 s chase centre, object-dominated pixels fell 52.3% → 48.6%; this is a screen-space diagnostic, not a truth label. The independent CPU and MPS renderers agree to a mean absolute **0.46–0.99 RGB levels** on covered pixels of the four chase poses. On 37,389 sampled ground records, centre displacement was **0.26/0.90/3.58 mm** (median/p90/max), with maximum tangent displacement 0.019 mm. This is not an improvement claim for foliage geometry.

Chase: [93 s](report/chase-93.jpg), [101 s](report/chase-101.jpg), [112 s](report/chase-112.jpg), [128 s](report/chase-128.jpg). Orbit contact sheets: [101 s](report/orbit-101.jpg), [128 s](report/orbit-128.jpg). On Studio, `$RS/work/stitch/report/orbit-{93,101,112,128}.mp4` contains the paired full revolutions; `before/` and `after/` contain all native and orbit frames and per-frame measurements.

## Measured throughput and the whole-drive plan

Apple M5 Ultra, 96 GB unified memory, shared with active Studio jobs. The first fit overlapped object-layer production; CPU evaluation and a short GPU bake-off also overlapped parts of the runs. Step rates varied, so these are contended wall timings, not isolated GPU benchmarks. The slice travels **684.03 m**. Its spatial halo included **3,737,013 ground + 403,466 object** records at initialisation. The fit cache held 2,177 training physical exposures (2,907 virtual views); 596 physical exposures had a usable holdout-safe depth product, with strong anchoring elsewhere. Median/p90 absolute learned exposure in pass 1 were 0.68%/2.00%.

| Fit | Iterations | Wall training | Bands/s | Full 336×216 view-area equivalents/s¹ | Mean candidate / updated splats per step | Peak process RSS / MPS driver |
|---|---:|---:|---:|---:|---:|---:|
| Fused, anchored | 6,800 | 211.2 s | 32.2 | 4.22 | 32,987 / 17,746 | 7.97 / 2.19 GB |
| Stereo + visibility, compensated raster | 3,600 | 138.6 s | 26.0 | 3.42 | 32,982 / 17,708 | 5.56 / 2.19 GB |

¹Each iteration processes one sampled image band, not a complete image. Dividing the exactly replayed valid native-mapped band area by 336×216 gives 0.131/0.132 views per iteration respectively; excluding sky/body/movers leaves 3.65/2.98 full-view RGB equivalents/s. RSS and MPS allocations overlap in unified memory; do not add them. The first on-slice 40-step trial without individual frustum culling processed 194,068 candidates/step at 13.7 steps/s; the 80-step culled trial processed 33,007 candidates/step at 23.4. These short trials include warmup and are not the full-run rate.

Pass 1 split 3,144 and pruned 9,399 objects. Independent optical tracks proposed 47,082 elements, 26,034 received agreement from another triplet, and 22,475 remained after existing geometry and voxel deduplication. The repeated clear-ray test removed 15,611 elements from the combined scene; none required a car-hull veto. Pass 2 split 160 and pruned 188. Final: **3,737,013 ground + 404,047 objects**, 397,636,860 PLY bytes. Every PLY was reopened and its header, count, bounds, DC range, normals/quaternions, lifetime and run ordering checked; drive and rig were verified against source. Init, fits and exports took 213.4 s and 141.6 s respectively; view preparation 8.2 s, validated stereo 27.8 s and clear-ray pruning 14.7 s. Native-and-orbit evaluation took 31.0 s before and 30.5 s after, plus 8.5 s for LPIPS.

For 3.3 km use **300 m ownership cores with 50 m observation overlap on both ends**, rendered with the same 84 m context-tile halo. Assign each stable input surfel to one canonical world tile/core (project to route distance with an explicit branch ID at self-intersections). Optimise the core with halo geometry present; freeze non-owned geometry in the final border pass. In overlap, combine DC in the common photometry gauge and logit/tangent updates with a smooth distance taper using stable IDs. New stereo/split proposals in the overlap are resolved once by distance/normal/colour and parent ID; keep one representative/union of supported visibility runs. Never concatenate duplicate overlap sheets. This merge is a plan, not an implemented or tested whole-route result. Stream one chunk at a time, build a global tile index and review both sides of every seam before publishing.

**Projected cost with inputs ready:** 405.6 s measured per 684.03 m, × (3,300/684.03) × (400/300 overlap factor) = **about 44 min** for preparation, stereo/clearance and fitting. Allow **35–65 min** for scene density and machine contention; add roughly 7–10 min for comparable evidence renders. This assumes equivalent input sampling, not a full-resolution trainer or every 36 Hz frame. Use the actual whole-drive 15,622,867 ground records, then scale only the objects: 15,622,867 + 404,047 × 3,300/684.03 ≈ **17.57 million**. Do not scale the tile halo or duplicate overlap. The whole drive was **not run**.

**Recommended next experiment before scaling:** use a short lateral-baseline stereo reconstruction of the island tree and kerb at 101 s, with visibility-qualified thin-surface depths and higher-resolution colour. Require the 90°/270° orbit sheets to disappear *and* perceptual crop quality to hold before adding more route. More iterations on the current anchor sheet do not supply the missing depth.

**Outstanding risks:** weak front parallax and wrong depth at long range; canopy that looks opaque to one lens but thin or displaced from above; unresolved unseen backs of surfaces; 9–14 cm cross-pass ground registration and motion blur; dawn colour/ISP variation. The mask/depth/geometry production time outside this slice is not measured here. Do not treat the training-time projection as an end-to-end ingestion ETA.
