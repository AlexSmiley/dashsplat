# Live map: now + memory (v4)

The live view is a reconstruction for a camera near the car, not a map certified from arbitrary viewpoints. The following contract supersedes the v2/v3 **live non-ground** birth, range and retirement policies. Ground v2 and stitched v3 retain their own contracts.

1. **Capture and release.** Schedule six physical cameras at 4.5 Hz while moving and 2 Hz below 0.2 m/s. Keep v3 photographic holdouts out of targets and depth windows. Use causal pose-conditioned depth, row-timed physical rays and released exposure corrections. The 85–140 s evaluation has a 1 s warm-up. The panorama is the explicitly permitted slice-wide exception to causal playback.

2. **Now.** Every supported non-sky depth region is eligible, including movers and the far horizon. No independent-confirmation birth gate, lens band veto or scene range cutoff. Convert optical z to range and apply the latest released per-lens inverse-range fit. Clamp the denominator at 0.60 and freeze the inverse offset beyond 40 m to avoid a correction pole. Keep a two-pixel erosion around measured 4% depth jumps. A missing neighbour is not a zero-range surface: its producer-side erosion has already happened. Sample each angular cell, range octave and instance (3/4/4/5 depth-grid pixels at <15/<42/<85/≥85 m); tangent axes follow the surface differential, scaled by range. Clip grazing stretch to 2.25×. Ground v2 owns its occupied road cells; road outside them can supply temporary terrain.

3. **Newest evidence wins.** Reproject the extant set with calibrated rolling-shutter fisheye geometry. Surface agreement is within `max(0.20 m, range × min(0.16, 0.075+0.0007×range))`. Extend the same immutable disc unless the new footprint is at least 1.3× finer. Immediately start a 0.25 s cross-fade on positive free space, an eroded sky witness, a finer sample, or displacement of the previous visible sheet. For a nearer observation, preserve a distinctly occluded background; only replace the old back sheet if it was frontmost, appearance matches and depth separation is modest. This last distinction is a heuristic, not a claim of scene understanding.

4. **Memory and traffic.** No observation leaves the last visibility endpoint in place. Fewer than three separated revisits extend existing runs; a fourth starts an immutable continuation. After five unseen seconds and >85 m distance, coarsen the rear memory by position, orientation and colour. Retired records stay in the playback tape. Every movable-mask sample is transient, even a parked object or a segmentation error. Redraw it at the next same-camera packet or clear it earlier on overlapping camera evidence. Per-camera IDs, per-frame robust 3D centroids, and contiguous PLY row ranges live in `maps/live-v4/instances.json`; IDs are not associated across cameras.

5. **Sky and proof.** Bin corrected sky colour by world direction (centre east, top +z); wrap longitude, low-pass measured areas and extend unobserved pixels smoothly. It contributes zero geometry alpha. Judge from native camera poses and a 6 m back/3 m up chase, with identical sky under v3 and v4. Measure both any opaque-enough geometry and the non-ground layer’s contribution on non-sky, non-road reference pixels, including movers. Count all generations against the 60M whole-drive budget.

## Historical design: ground and v3 static fusion (v2)

Goal: turn the stream of calibrated frames into a growing world of Gaussian splats, using only data up to
the current moment, and record for every splat when it was created, when cameras could see it, and when it
was retired. The same engine run with hindsight gives the first version of the stitched map.

Tesla's version is a feed-forward network (about 220 ms, no initialisation, handles movers). Ours has the same
shape: a feed-forward depth model per frame, then a deterministic fusion step that could run in real time.
No per-scene optimisation in the live path.

v2 folds in two design reviews (geometry and performance). The main changes from v1: far geometry is
provisional and gets superseded instead of averaged; splats are immutable once born; depth is corrected from
independent anchors only; the see-through test is conservative; footprints are anisotropic; visibility is
stored as runs, not one timestamp.

## Inputs per keyframe

For each camera: image, depth (range along the calibrated ray), sky mask, movable-object instance mask, and the
camera model from `pipeline/common/rig.py` (intrinsics, `T_ego_cam`, clock offset, rolling-shutter readout), plus
the trajectory. Poses are evaluated per image row: `t_row = t + offset + (v/H - 0.5) * readout`, both when
unprojecting pixels and when projecting surfels (project with the mid-row pose, then re-project once with the
pose of the row it landed on).

Keyframes: spacing `max(0.5 m, v / 9 Hz)` while moving; static fusion is skipped below 0.2 m/s (stopped frames
only update visibility and the mover layer, at about 4 Hz). The first stopped frame seeds the map.

## Surfels

A surfel is an oriented elliptical disc: position, normal, two tangent axes with radii, colour, confidence
(log-odds), generation footprint `r`, birth uncertainty, source camera, `t_birth`, `t_death` and up to three
visibility runs. **Geometry and colour never change after birth.** Anything better arrives as a new surfel
that supersedes the old one. This keeps playback honest: scrubbing back never shows knowledge from the future.

Footprint: intersect the rays of neighbouring samples with the local tangent plane. Along the image direction
that grazes the surface the footprint stretches by `1 / |ray . n|` (clamped to 3.5x), so a road seen from 1.3 m
up is covered without gaps when viewed from 3 m up. Export as a Gaussian with tangent sigmas about 0.9 of the
footprint radii, a thin normal sigma (`max(0.08 r, 8 mm)`) on smooth surfaces and a fat one (`0.5 r`) where
depth variance says foliage, and opacity about `0.7 * sigmoid(confidence)`.

Density: `r_min` is 4-5 cm in the driven corridor and grows with lateral distance. Each depth band is sampled
from the image-pyramid level whose footprint is in `[r_min, 2 r_min)`. Fusion runs on depth at about 724x469;
colour is sampled from the full-resolution image.

Range limits: surfels are born out to `R_max` = 40 m at first (later 80 m). Road surfels are only born where
the incidence angle is at least 3.5 degrees (about 22 m ahead of the car for the front camera).

## Depth preparation

1. **Edge erosion.** Invalidate depth where the relative jump to a neighbour exceeds 4 percent, dilated by
   two pixels. Depth ramps across silhouettes are the source of streaks that only show from above.
2. **Correction from independent anchors.** Per camera, a slowly varying correction in inverse range,
   `rho' = a rho + b` (b shrunk towards 0), fitted to: (a) static points tracked between keyframes and
   triangulated with the known baseline, weighted by `B_perp / (s^2 sigma_theta)` and rejected near the epipole
   and on movers; (b) the road next to the car at the calibrated camera height, as a few robust patch anchors,
   not hundreds of pixels. The map is never used as a scale anchor for the depth that built it.
3. **Ground manifold.** A smooth height surface fitted to road samples 3-10 m from the car in the ego frame
   (and to the tyre track behind the car, which is ground by definition). Road surfel heights and normals are
   regularised towards it.

## Per keyframe, per camera

1. Take the hot set: alive surfels within range of the frustum (8 m Morton-sorted chunks; ego-centred float32).
2. Project them (full fisheye model, row-timed pose). Render nearest-surface depth over footprints, not centres.
3. For each projected surfel compare against the prepared depth, using the point-to-plane residual
   `e = n . (X_obs - X_map)` and a gate that grows with depth error, incidence angle, pose and timing
   uncertainty, starting from `tau = max(10 cm, 8% z) / max(cos theta, 0.25)`:
   - **Confirmed** (within the gate): extend the current visibility run to now, raise confidence (+2.0).
   - **Seen through**: only if the minimum depth in a 3x3 neighbourhood exceeds `z + 2 tau` across the surfel's
     core, away from depth and colour edges, and the view is not grazing. Lower confidence (-0.4). Retire when
     confidence falls below a floor after violations from distinct viewpoints.
   - **Occluded, masked, outside every image, or uncertain**: nothing changes. The visibility run ends.
     This is the memory.
4. Coverage buffer: per pixel, the finest confirmed footprint and its surfel id.
5. **Birth.** A sampled pixel with valid depth that is not sky and not a mover gives a new surfel when it is
   uncovered, or when the confirmed surfel there is at least 1.5x coarser or was born with much higher
   uncertainty (a far, provisional estimate). No birth within `2 tau` in front of or behind a confident surfel of
   similar footprint: that is the same surface with depth noise. Newborn depth is snapped towards the parent's
   plane within the gate to avoid two sheets a few centimetres apart.
6. **Supersede.** A parent is retired only when at least 80 percent of its projected disc is covered by finer
   confirmed or newborn surfels. `t_death` is set 0.25 s after that so the hand-over cross-fades.
7. Exposure: fixed per-camera vignette from calibration data; per-keyframe gain per channel, estimated jointly
   for the six cameras on road they share over time, smoothed, and applied before colours are sampled.

## Movers

v1: pixels of movable-object instances are cut out of the static map. v2: instances judged to be moving become
transient splats (`t_birth` = now, `t_death` = next update, or last seen + 1.5 s linger so a car entering a
blind spot remains as an ageing ghost). The motion test tracks corners inside each eroded instance, lifts them
with depth, predicts them with the known ego motion and compares both pixel displacement and range; when the
evidence is too weak (a lead car at matched speed) the answer is "unknown" and it stays transient.

## Sky and far field

Sky pixels go to a rotation-only panorama. Finite far scenery is not put on the panorama (a building at 80 m
has real parallax over this drive); it gets coarse surfels in the far band once the near field is right.

## Visibility runs and playback

Per surfel: `t_birth, t_death, e0, s1, e1, s2, e2`. Run 0 is `[t_birth, e0]`, runs 1 and 2 are `[s1, e1]` and
`[s2, e2]` (unused runs are set after the end of the drive). Gaps shorter than 0.25 s are merged. A fourth run
retires the surfel and births an identical continuation. At playback time T:

    seen = min(T, e0); if (T >= s1) seen = max(seen, min(T, e1)); if (T >= s2) seen = max(seen, min(T, e2));
    visible = t_birth <= T < t_death;   age = T - seen

## Export

Per 25 m tile: merge coexisting same-generation surfels born within 0.3 s of each other into an adaptive
4-12 cm grid by moment matching (mean position, summed covariance), then write the PLY tile with the time
properties above. Static tiles are indexed by space, transient tiles by time bucket.

## Stitched map from the same engine

With hindsight the immutability rule is dropped: every surface keeps a weighted average of its finest, most
frontal, sharpest observations (`cos theta / z^2`, less for motion blur), and only the final generation is
exported, with `t_birth` = start of the drive. That is the initialisation and depth prior for the optimised splat.

## Engineering

First version in plain PyTorch on MPS at reduced resolution, per camera, with a preallocated hot pool on the
GPU and retired or distant surfels evicted to CPU storage. Expected 20-35 camera-frames per second, which is
fine while depth inference (15-35 camera-frames per second) dominates. A single Metal kernel
(`torch.mps.compile_shader`) that tests all six frustums per surfel and writes coverage atomically is the
path to hundreds of frames per second when it matters.

## Build order

1. t = 85-110 s, front camera only: prepared depth, surfels with anisotropic footprints, export, look at it in the viewer's chase camera.
2. Add pillars and repeaters with row-timed poses, depth correction from anchors, joint exposure.
3. Confirm / see-through / supersede with visibility runs; mover cut-out; export merge.
4. Whole drive with eviction and tiles.
5. Mover layer, far band, sky panorama, back camera.

## Acceptance

Static tracks held out from pose fitting and depth correction: far-to-near reprojection and range residuals by
camera and distance on the car-park turn and a fast straight; count of doubled walls and wrongly retired poles
at the same places; and the chase-view render compared with the reference frames.

## Implemented v3 slice: amendments from the photographs

- **Depth quantity and provenance.** The file stores physical camera z, not unit-ray range: use `s=z/ray_z` on the distorted grid. The producer uses past images but can still be conditioned on *offline poses*. Live therefore gets a separate prediction with causal poses and waits for every contributing pose row and the photometry release. Every tenth selected exposure is withheld; skip a training target if its DA3 source window would contain a reserved image. The reference camera score additionally excludes ground-v2 target frames. The producer's rear call is joint across three yawed virtual views with three past/current physical frames, not five on each view.
- **Anchor policy.** Use the stored BA points transported from their original vertical gauge for offline correction; score separately on genuinely reserved static tracks. A different, strictly prefix-based inverse-range fit uses triangulated forward LK history and a handful of near-road patch medians live. Shrink the intercept, and qualify *measured* range bands. A frame can give a good road scale at 8 m and place a 36 m feature at 64 m. At the first arterial section this produced a false dark wall; the independent local band veto is more useful than rescaling the whole frame. Rear peripheral predictions and unreliable repeater ranges are gated even if files are complete. A passed front far band extends to 70 m; sides do not inherit it.
- **Birth and retirement.** Edge erosion is on *range*, not z, after cutting out calibration holes, sky and every nonzero uint16 movable ID. The 4% discontinuity is dilated by two grid pixels. A clean near surface or a frontal wall can be born from an already multi-image depth window. Unknown bands and ambiguous far patches stay proposals until independent parallax/range agree; flat distant walls get a wider gate only with matching photographic colour. Rasterized oriented footprints suppress duplicate sheets. Nine core probes must reach 80% finer support to retire a parent. Multiple distinct clear, non-grazing core see-throughs are required for negative evidence. The *measured car hull* is also positive free-space evidence: a foliage patch once appeared in a lane later physically occupied by the car. Stitched rejects these contradictions; live never shortens an interval before the relevant pose release. Every road grid cell in the conservative XY bounding box of a coarse terrain one-sigma core must be unowned at birth. A later ground claim starts a 0.25 s live handover; confirmation never shifts a stitched terrain core into a neighbouring owned cell. Gaussian tails are not cell boundaries.
- **Sampling and implementation.** This is a 55 s prototype using a minimum 0.56 m **and** 0.28 s spacing, not all 0.5 m depth keyframes. The earlier 80–85 s interval warms the calibration. Pyramidal lattice stride follows range and lateral distance; equivalent tangent radii come from intersecting neighbouring calibrated unit rays with the local plane. Native sensor RGB is sampled before JPEG-domain radiometric correction. CPU vector preparation and an 8 m XY chunk index with a native, chronological per-camera core replaced the planned MPS pool. Distant/retired records stay in RAM but do not enter the hot projection pass. No hindsight moment merge is applied to live records: projected coverage and weighted stitched colour/normal-direction position updates provide consolidation without rewriting live births. Foliage volume is a local roughness/colour heuristic, not a tree detector. Stitched overflow chronology is retained outside the viewer’s three-run contract.
- **Limits of the proof.** Raw photographic comparison and independent range checks are different tests. Reserved DA3 depth can diagnose a projected occluder but is not ground truth. Thin silhouette eroders sacrifice narrow poles; dark renders do not imply empty real space. Object count projections scale by travel distance, not by assuming this slice has the same object density as the whole drive.
