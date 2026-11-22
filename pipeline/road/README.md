# Road-relative pose and orthographic imagery

The public geometry uses ENU metres and **COLMAP pixel centres**. It always evaluates the exposure time of the original image row. From the Reality Splat root:

```python
from pipeline.common.rig import load_rig, Trajectory, ground_plane
from pipeline.road.geometry import pixels_to_ground, ground_to_pixels
from pipeline.road.mosaic import accumulate_window

cam = load_rig()['left_repeater']
tr = Trajectory('poses/ego.npz')            # or poses/ego_causal.npz
n, h = ground_plane(320.0, tr)              # unit upward normal in EGO coordinates
# n @ point_ego + h == 0; h is perpendicular metres above the road
world_xyz, valid = pixels_to_ground(cam, tr, frame_time_s, uv, max_range=12)
uv2, valid2, world_z = ground_to_pixels(cam, tr, frame_time_s, world_xyz[..., :2])
grid = accumulate_window(316, 326, tr, cell=0.02, save='road/example')
```

1. **Coordinates.** `pixels_to_ground` takes arbitrary arrays of image centres and broadcastable frame times. `ground_to_pixels` takes one physical exposure and an array of world XY positions; it finds each point's local road height and iterates project → original row → exposure. The renderer samples this mapping every 8 cm before fine resampling; the measured patch shifts use original pixels and the full source-row mapping. Public validity includes the original rig's body mask, signed ray-plane intersection and ego distance. Legacy files use a world-level plane at z=0. Do not combine a V2 orientation with a V1 ground plane.
2. **Rasters.** `MosaicGrid(origin,basis,size,cell)` maps column/row centres to `origin + basis @ ([column,row]*cell)` (ENU XY). `accumulate(box,bgr,weight,exposure_time,camera_index)` updates the best view, first/last exposure and repeated-view appearance support. The default 2 cm grid uses every scheduled keyframe for all six lenses; it does not generate points for missing timestamps. `best.jpg` is the multi-view-consensus image, `pavement.jpg` further restricts it to connected road-colour pavement around the driven lane, and `road-mask.png` marks the fixed evaluation support. Report crops use `display-mask.png`, which also admits actual observed pixels on small enclosed neutral-colour islands; no missing paint is synthesised. `grid.npz` carries world origin/basis/cell and first_seen/last_seen arrays (NaN = never observed). No learned or external dynamic mask is required. Parked vehicles/curb edges are not perfectly separable by this test.
3. **Products.** `poses/ego.npz` and `poses/ego_causal.npz` keep their original field names. Road and suspension pitch/roll are **radians**, positive nose-up/left-up. Road angles describe the world road tangent resolved along/across the current heading, not fixed east and north Euler angles. `susp_heave` is signed metres along the road normal. `n_road_pairs` counts training matches ending in the previous 350 ms; bit 8 of `quality_flag` means fewer than four while moving. The retained V1 flags/counts and uncertainty proxies are not calibrated absolute errors. `ground_plane` is the canonical API for consuming the decomposition.
4. **Reproduction.** On Studio, use `bash pipeline/road/run.sh` from `$RS` with the project's existing environment and the completed V1 pose workspace (schedule, original final Ceres problem and accepted feature observations). It preserves V1 once, evaluates staged candidates before atomically installing them, then renders the comparison figures. Set `ROAD_THREADS` for renderer parallelism; Python/BLAS is single-threaded per worker and Ceres uses the original CPU thread budget. Patch-block caches are **drive/rig specific**; use a fresh scratch directory for different captures. No source images or full rasters are copied into this repository.

See [REPORT.md](REPORT.md) for actual coverage, pair-level errors and the limits of the vertical gauge. Full-resolution Studio imagery is at `$RS/road/mosaics/{planar,offline-v1,causal-v1,offline,causal}/{park,turn,straight}/`.

## Whole-drive ground scenes

On Studio, the two viewer roots are `$RS/scene/road-live-v1/` and `$RS/scene/road-stitched-v1/`. Open the root containing `map.json`; each tile is a binary little-endian 3DGS PLY in ENU metres. The live car follows the causal V2 poses and the stitched car follows the offline V2 poses. The 50 m tile grid is shared. Each record has a thin disc (8 mm normal sigma, tangent sigmas 18.18, 9.09 or 4.55 cm), opacity 0.88, SH DC colour and the seven viewer time fields.

**Rebuild or render.** From `$RS`, run `bash pipeline/road/run_ground.sh` with the native frames, fixed rig, V2 poses and Task C ground-pair/pose-lag workspace present. It uses the existing Python environment and macOS C++ compiler, with 14 tile workers by default. Both variants reuse a snapshot of available mask paths in `maps/ground-work/mask-index.json`. The renderer reads the exported PLY, manifest and drive track, not the source mosaic or source frames. For one image:

```sh
.venv/bin/python -m pipeline.road.render --scene live --time 200.2 --view chase \
  --mode age --output maps/ground-debug/stop-age.jpg
```

**Radiometry.** The shared interface takes a physical camera name, original frame number and an RGB image. Pass RGB in JPEG code values (0–255), not OpenCV BGR or linear radiance. The fixed shading profile is zero in log gain at the sensor centre; the first front exposure anchors the gain. The offline model uses robust repeated-patch constraints with smooth gain knots. The online model updates gains from earlier and same-packet overlaps, learns shading only from moving viewpoints in completed prefixes, and freezes shading after 105 s. It never applies the hindsight profile on early frames. A rerun with everything at or after 99 s removed gave **identical gains and shading for all 394 retained packets**. Shared calibration files and held-out RGB metrics are in `$RS/photo/`.

```python
from pipeline import photo
rgb = photo.apply("front", n, image_rgb, variant="live", root=RS)
model = photo.load("live", RS)
ready_time = model.available_at("front", n)  # includes causal pose availability
corrected_samples = model.sample("front", n, original_uv, rgb_samples)
```

**Acceptance and height.** At each cell, compare corrected colour, chroma and local patch structure across baselines. Test the local V2 road tangent against two above-road parallax hypotheses (32 and 84 cm). Sky and nonzero instance IDs exclude pixels when masks exist; road masks and confirmed neighbours supply extra support near the path. Failing cells stay empty. Three or more strong patch pairs at separated times can support a smooth local height offset off the road; taper it to zero in the driven core. There were 22 live and 27 offline supported 1.5 m height bins. These are sparse local corrections, not a recovered kerb along the whole drive.

**Existence versus visibility.** The 2,235 packets cover 0–361.42 s. After the first packet of each stop, live updates only visibility at about 4 Hz. The latest camera row and causal pose release gate birth. Available observations, not future observations, set the initial resolution; four or sixteen children cover the entire parent cell before its 0.25 s fade. The fourth disjoint live visibility run starts an identical continuation, not a geometry refinement. Age is camera recency, not existence: the cell under the car can still draw. Stitched keeps one record per final disc, with birth 0 and no retirement. Where its visibility exceeds the PLY's three slots, the PLY keeps birth, first revisit and last revisit; additional true intervals are stored under `maps/ground-work/stitched/<tile>.rgb.extra-runs` as repeated little-endian `(uint32 vertex_index, float32 start, float32 end)` records. A seven-field-only reader therefore understates stitched recency on those intermediate revisits. Geometry is present throughout.

**Evidence.** Eight paired overhead/chase renders use the same playback times in both scenes. [First road](report/ground-first-road.jpg), [stop](report/ground-light-stop.jpg) and [arterial turn](report/ground-arterial-turn.jpg) show the lane-to-verge boundary. [Held-camera refinement](report/ground-refinement.jpg) follows one physical patch as coarse orange gives way to medium teal and fine blue. [Photometry](report/ground-photo-before-after.jpg) compares raw and corrected fusion using the same observations and geometry. Full per-tile summaries, rendered-view statistics, the 0.5 s coverage probe and its definition are under `$RS/maps/ground-{work,debug}/`. Coverage measures sampled 22.7 cm cells, not the rendered screen area or semantic precision.

### Measured on the 361.42 s capture

| Scene | Discs / records | 50 m tiles | Total on disk | Fusion + tile export |
|---|---:|---:|---:|---:|
| Live | 15,034,568 | 93 | 1,444,677,313 bytes | 20.91 s |
| Stitched | 11,420,416 | 93 | 1,097,719,970 bytes | 20.44 s |

Times are Studio wall-clock with the fitted photometry and height anchors already available. The validator read every PLY, checked counts, bounds, DC colour, quaternion, normal, and run ordering, then replayed the car corridor at 0.5 s intervals. In a separate playback probe, all 1,162 sampled finer handovers had four half-pitch children born 0.25 s before retirement. At 200.2 s all 1,004 live splats in the inner underbody rectangle were remembered, not visible (median age 6.48 s). There are 1,822 live visibility continuations. Stitched stores 218,973 additional visibility intervals for 214,413 discs in the sidecars.

Held-out cross-camera median mean-absolute RGB error went from 10.33 to **6.61** levels live (60,711 pairs) and 10.33 to **4.66** stitched (60,035 pairs); the 90th percentiles moved from 28.67 to 22.45 and 27.00 to 17.33, respectively. The original dawn colour is retained. A periodic shadow/brightness stripe is reduced, not eliminated, between the arrows at 93–101 s. The live map still contains some curb/wall-coloured fragments along the drive-through and interrupted paint at 318–322 s. The latter is continuous in the stitched view.

The core probe is a 3.2 m wide swath extending 2 m behind to 10 m ahead, so it measures both remembered and newly exposed road. Live core coverage at playback is **77.3%** over the drive, stitched **98.1%**. Live reaches 99.8% on the first arterial section, 100% at the later light stop and 98.2% through the arterial turn. The initial 5.67–81.83 s stop is the outlier (6.1% live / 92.3% stitched): the body occludes part of the lane and a stationary rig offers no new baseline to confirm the nearby surface. At 181–181.5 s there is a brief live core dip to 44%; at 360–361 s the stitched forward probe reaches the end of the observed route and falls to 49%. Beyond 6 m lateral offset, only 26% live / 35% stitched is occupied at playback: dark regions are deliberately empty where no useful, consistent view supports ground. No object/sky layer or cosmetic hole-filling is included.

## Ground v2: what the front camera can actually resolve

On Studio, `bash pipeline/road/run_ground_v2.sh` builds `$RS/scene/road-{live,stitched}-v2/` using v1's paired-ground workspace and height anchors. The v1 scenes and photometry stay on disk. See the [side-by-side proof images](report/v2/): left to right is live v1, live v2, stitched v2. The detail strips (exit, lettering on approach, the first capture at or after 318 s, and 318–322 s) compare a rectified **native** front-camera frame against all three scenes at the same 6–15 m road-relative crop. Timestamps are selected by native paint contrast, except for the explicitly fixed 318 s capture. The rendered result is never used to choose the frame.

**Source of the softness.** At 318.6 s the native front projection resolves 0.23 cm across / 1.44 cm along the road at 10 m, and 0.50 / 6.91 cm at 20 m. V1 decoded half-resolution colour, promoted according to height divided by squared ego range, and emitted one isotropic scale for both axes. Its median exported sigma at 6–15 m was 9.1 cm in either direction. Its 19.8 m cutoff was unconditional. At the exit v1 also froze colours while the exposure gauge was still adapting.

**Footprints and reach.** V2 reads full-resolution front colour. The two singular values and directions of the local ground-to-sensor Jacobian provide separate transverse and longitudinal pixel footprints. The radial variance also includes the pitch term `sigma_theta * range² / camera_height`. Near the car, pitch comes from recent causal suspension-pitch innovations; at range, the pose uncertainty proxy enters with a floor and partial reduction for separated agreeing views. It is a working uncertainty model, not an absolute pitch calibration. A far splat needs uninterrupted road evidence along its bearing, a bounded credible range interval and either a matching semantic road surface or repeated appearance from independent baselines. Bending roads restrict maskless extrapolation. Eligible front radius is 82 m, other cameras 30 m, lateral limit 20 m. Stretched splats have lower peak opacity as their footprint expands. Front observations inside 26 m can refine cross-track sampling before resolving the range; two additional immutable handovers are allowed on high-contrast paint when the range estimate or measured colour changes. The [approach strip](report/v2/approach-refinement.jpg) holds the *same ground patch* and crop fixed at 60, 40, 20, 10 and 5 m of forward distance. Close-range contradictions may retire false provisional surfaces. Colour and pose inputs are never taken from a future keyframe.

**Holes, boundaries and colour.** Enclosed holes preferentially recover previously observed colour. For a genuinely missing cell, nearby accepted ground must close the rim on all four cardinal sides and three diagonals, agree in height within 7.5 cm and have a small luminance spread. The inferred underlay uses their local plane and colour; it does not synthesise lettering. The live repair runs when the rim closes and cannot propagate through other inferred rims. At the outer edge, opacity and kernel width taper with neighbour support. Periodic robust prefix photometric fits re-anchor all six cameras; a running equal-camera road-colour estimate fixes the common low-frequency RGB gauge in both variants. The reference is observed illumination, not a grey or white target. The exported transforms are available as `photo.load("live-ground-v2", RS)` and `photo.load("stitched-ground-v2", RS)`.

**Masks and memory.** The mask snapshot covers approximately 80.0–140.0 s on the six cameras, not the whole drive. The [mask producer](../masks/README.md) marks road and sky separately and uses nonzero uint16 instance IDs for movable classes. The builder may pick an actually segmented physical frame up to 55 ms **earlier** than the scheduled one; it projects that image at its own row times. Outside mask coverage, reprojection consistency remains the gate. The final schedule used matching sky, road and instance masks on 323 of 2,235 packets per lens (14.5%); 58 of these used the earlier segmented frame. The memory plates use independent calibrated fisheye projection, body/instance/sky masks and each camera's useful range. Outlines depict camera footprints at the packet time; colour depicts age from the PLY runs. Sub-250 ms gaps can be merged, and moving frustum boundaries are sampled, not continuous.

Diagnostics and tile summaries live in `$RS/maps/ground-v2/`. The reach probe samples all moving time at 0.25 s intervals, by distance along **current heading** (not along the known future route). Its centre result asks if the centre cell actually exists at playback; the companion lane result samples 13 offsets from −1.5 to +1.5 m. Neither awards distant coverage just because a large Gaussian tail overlaps the probe.

**Causality and position.** Removing observations and pose samples from 99 s onward, refitting the live colour gauge, and rerunning exit tile `-1_0` reproduced all 394 retained packet transforms and 418,241 previously emitted records byte-for-byte in geometry, RGB, alpha and birth time (comparison through 98.61 s). None of the six cameras’ rescheduled row-pose inputs was released after its packet’s birth. Applying the causal and offline poses to the same real front pixels at 6–15 m gave median absolute along/across position differences of 34/1.2 cm at the exit and 29/0.9 cm during the lettering approach; these compare two estimates, not ground truth. The live packet delay relative to the front capture was 234 ms median at 91–94 s and 95 ms at 316–322 s, about 1.0 and 1.2 m of vehicle travel. These effects limit live along-track registration even where the native strokes resolve cleanly.

### Whole-drive v2 result

| Export | PLY records | Tiles | PLY bytes | Studio fusion | Lane-centre coverage at playback |
|---|---:|---:|---:|---:|---:|
| Live v2 | 28,012,974 | 132 | 2,689,318,255 | 47.91 s | 81.4% (v1: 77.3%) |
| Stitched v2 | 15,622,867 | 132 | 1,499,867,974 | 47.87 s | 100.0% (v1: 98.1%) |

The validator read all 264 tiles and checked the drive/rig, all 24 PLY fields, record bounds, positive unit-normal geometry, lifetimes and visibility-run ordering. The numbers above are fusion times with inputs ready, not end-to-end calibration/render times. Across all 994 moving-time probes, live centre presence at 20/40/60/75 m was **97.6/83.3/68.3/0.4%**; the corresponding 3 m lane-wide fractions were 97.0/83.2/68.6/0.4%. On the 293–310 s straight, 60 m reached 97.1%, but 75 m was never reliable. A sample of 29,345 stretched live records born over 32 m away had median/p90 major-axis error of 0.06°/0.21° relative to the bearing at release. At 18 and 19.5 m either side, 0–10 m ahead, 28.2% live / 48.3% stitched of probed cells existed; this measures occupancy, **not** ground-classification precision.

| Native stroke recovered in 6–15 m crop | Live v1 | Live v2 | Stitched v2 |
|---|---:|---:|---:|
| [Exit arrows, 91.72 s](report/v2/detail-exit.jpg) | 31.7% | 73.8% | 91.0% |
| [ONLY approaching, 317.53 s](report/v2/detail-only-early.jpg) | 29.5% | 66.3% | 92.8% |
| [First capture after 318 s, 318.09 s](report/v2/detail-only-318.jpg) | 31.5% | 80.4% | 86.1% |
| [Highest native paint contrast, 318–322 s, at 318.76 s](report/v2/detail-only.jpg) | 35.3% | 46.8% | 74.9% |

Recall is the fraction of native bright high-pass stroke pixels with a rendered bright response within a 7×7 pixel window (±7.5 cm per axis). It is not OCR accuracy. At 318.6 s, v1 median tangent sigmas in the front lane were 9.09/9.09 cm, versus 7.34/4.10 cm live v2 and 4.13/4.09 cm stitched v2. Tracking the same lettering patch from 60 to 5 m, median long sigma fell from 5.07 m to 5.7 cm; its short sigma fell from 16.4 to 4.1 cm. At the [moving exit](report/v2/memory-93.7.jpg) and the [light stop](report/v2/memory-200.2.jpg), every sampled underbody and inter-camera-gap record was remembered. Underbody median ages were 1.27/6.24 s; 99.75%/100.00% of records well inside the camera footprints were current (<40 ms).

**Remaining defects.** Live arrows are identifiable and lettering is more legible, but approaching strokes are weaker and sometimes broken versus stitched. The initial stop still has just 21.5% of the lane centre live (v1: 6.1%). Median road-pixel colour difference to stitched at 87.9/93.7/101.2 s fell from ΔE76 12.5/7.6/4.9 to 5.6/3.3/3.5, but a cool edge and a lightness band remain: at 93.7 s the 90th-percentile centred lightness residual **worsened** from 4.8 to 11.1 L*. Stitched is cleaner. Large diagonal ribbons from the first v2 build are gone at 101.2/112.5 s; stretched verge fragments and openings still occur at turns. No 80 m reach or complete edge classification is claimed. [Full measurements](report/v2/measurements.json) include the 11 same-time chase/top triptychs’ sample statistics, reach, poses, colours and age tests.
