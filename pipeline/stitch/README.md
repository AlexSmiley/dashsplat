# Photograph-constrained stitched map (Apple Silicon)

The output is a viewer scene with DC colour and the seven time properties. [REPORT.md](REPORT.md) contains the measured results and the remaining novel-view defects.

## Run on the Studio

Push just this folder from the repository root: `ferry cp ./pipeline/stitch anomastudio:dev/reality-splat/pipeline/`. The following commands run on the Studio, from `~/dev/reality-splat`, with completed calibration, poses, v2 ground, v3 objects, masks, physical keyframes and photometry. The stereo/clearance passes also use v3's input-frame schedule to audit exposure provenance. Run into **new output directories** for a different slice or a different experiment: ground tiles in intermediate scenes can be hard-linked to immutable inputs.

~~~sh
cd ~/dev/reality-splat
export RS=$PWD
P=$RS/.venv/bin/python
# One-time setup. The numerical/rendering stack stays in the shared environment.
git clone https://github.com/tchauffi/metalsplat third_party/metalsplat
git -C third_party/metalsplat checkout 84e8c3a433f71c107e9387d3f72a765436c11f6e
uv venv -p "$P" envs/stitch
uv pip install --python envs/stitch/bin/python --no-deps msplat==1.1.4 lpips==0.1.4
export PYTHONPATH="$RS/envs/stitch/lib/python3.12/site-packages:${PYTHONPATH:-}"
~~~

~~~sh
$P -m pipeline.stitch.views                         # Physical train/test split + virtual pinhole cache
$P -m pipeline.stitch.shutter                       # BA proof, native pixel units
$P -m pipeline.stitch.bakeoff                       # Stock msplat comparison
$P -m pipeline.stitch.refine scene/stitched-v3 work/stitch/seeded-runs
$P -m pipeline.stitch.fit --source scene/stitched-v3 --steps 6800 --out work/stitch/pass1
$P -m pipeline.stitch.visibility work/stitch/pass1 work/stitch/pass2-input --seeds work/stitch/seeded-runs
$P -m pipeline.stitch.fit --source work/stitch/pass2-input --aa --steps 3600 --out scene/stitched-opt-v1
$P -m pipeline.stitch.verify scene/stitched-opt-v1
$P -m pipeline.stitch.evaluate before --source scene/stitched-v3
$P -m pipeline.stitch.evaluate after --source scene/stitched-opt-v1
$P -m pipeline.stitch.evaluate compare
$P -m pipeline.stitch.perceptual
$P -m pipeline.stitch.diagnose
~~~

## How the fit works

1. **Cameras and colour.** Read exact physical exposures from the depth manifest, not the frame number of the nearest front image. Reserve indices 7, 15, … of each physical lens in the slice before sampling or rectifying. The back camera has three virtual views (−52°, 0°, +52°); the other five have one. Every training pixel has an original sensor row: assign it to one of eight equal-native-row time bands; render the enclosing virtual pinhole crop at the centre exposure time, and mask curved band boundaries. Intrinsics and trajectory are fixed. Ground-v2 photometry is applied in JPEG RGB units, followed by a physical-exposure log gain bounded to ±0.04 and shrunk towards zero. Rear views share that gain. Held-out images never get fitted gains.
2. **Differentiable splats.** MetalSplat's MPS project/bin/sort/front-to-back Gaussian renderer differentiates robust RGB residuals. DC colours, opacity logits and the two tangent scales move. Orientation and thickness remain tied to the fused normal; only normal-directed translation is allowed (≤6 mm ground, ≤9 cm objects per fit). A near-surface depth residual in *physical optical z*, not range or virtual z, is used only where observed training depth exists, its producer window excludes our withheld frames and the rendered surface agrees. Opaque object centroids in eroded sky receive a separate penalty; sky RGB, movers and camera body never supervise DC. Off-surface displacement and shape/appearance changes have restoring penalties. No panorama, image planes, view-dependent radiance or pose optimisation.
3. **Spatial and topology controls.** Load whole cells around the active chunk (84 m input tile halo); sort Gaussians into 5 m bins; bound their kernels conservatively, then reject individual spheres outside the cropped frustum with a tiny native CPU kernel. Only active MPS parameters get Adam state updates. Split repeatedly observed, textured, enlarged object primitives along a tangent, preserving aggregate centre opacity and inherited chronology; prune repeatedly sky-contradicted ones. Ground never splits. The second pass uses mip opacity compensation to agree more closely with the independent proof renderer.
4. **Geometry from photographs.** Fill selected holes with new static surface elements only when three physically distinct camera poses, forward/backward patch flow, held-centre reprojection and two independently triangulated tangent directions agree. A different triplet must corroborate world position, colour and normal. Sky, body, movers and road are rejected; raw depth, if available without holdout leakage, can veto a conflicting layer. Deduplicate in 11 cm voxels and avoid existing object centres. Merge confirmed exposure times separated by ≤0.25 s; store first, first revisit and last run if more than three. Existing visibility fields survive fitting; children inherit the parent. Multi-baseline road/sky and clear-depth rays prune contradicted objects, with positive depth observations acting as a veto. The actually traversed car volume is an additional free-space test. These are evidence tests, not hole-painting or chase-specific masks.
5. **Evaluation.** Score all valid, static, non-sky pixels of 362×235 *native distorted* frames, including black holes, with the independent v3 CPU compositor at 33 sensor-row poses. SSIM uses Gaussian 11×11 windows wholly inside the valid mask. The separate clean subset excludes known exposure overlap with fused ground and object DA3 windows. For each of four chase poses, render 96 paired azimuths at 544×320, 6 m behind and 3 m up, at fixed drive time. No target photos enter novel rendering. LPIPS-Squeeze uses a reference-selected, wholly valid central 96×96 crop; it reads the same-quality proof JPEGs for both models, while PSNR/SSIM use unclipped-then-clamped float renders before JPEG coding.

All intermediate data and full evidence live under `$RS/work/stitch/`. Markdown and representative JPEGs live under `pipeline/stitch/`. Use `STITCH_WORK` to isolate caches, compiler output and reports for separate chunks. The frozen input manifest is reused when present. The scripts accept `STITCH_START` and `STITCH_END` for frame selection; full-route depth, masks and object fusion must exist first. The report gives the route-partition and merge plan; automatic seam merging and a whole-drive run are not implemented here.

Licences: MetalSplat MIT; msplat Apache-2.0 (bake-off only). The fit inherits DA3-Large-1.1 depth (CC BY-NC 4.0); it is a personal-project scene. SqueezeNet/LPIPS are for evaluation only.

## Dense optimisation (v2)

[Results and four-site acceptance](REPORT-v2.md). Run on the Studio from the Reality Splat root with the v1 MetalSplat environment. Keep the upstream depth products and their index files together until targets, seed and clearance complete. The first invocation snapshots the physical split. The unreserved ground is the original fused mosaic, not v1's fitted ground; every export takes the original tile records and metadata.

	source .venv/bin/activate
	export PYTHONPATH="$PWD/envs/stitch/lib/python3.12/site-packages"

1. Run `python -m pipeline.stitch.v2targets` and `python -m pipeline.stitch.v2seed`. They use the same preassigned schedule in `work/stitch-v2/exposures.json`. The rectified target cache is immutable for this experiment. Missing input depths or depth windows touching a reserved physical camera frame do not become synthetic supervision. Prepared targets include RGB-only views.
2. Run `python -m pipeline.stitch.v2clear`, then `python -c "from pipeline.stitch.v2fit import V,compact; compact(V/'dense-prior')"`. Initialise the visibility and colour priors with `python -m pipeline.stitch.v2posterior work/stitch-v2/packed work/stitch-v2/photo-seed --seed`. After target preparation, this stage uses the immutable cached corrected depths. It does not substitute once-missing files from subsequently regenerated depth windows.
3. Run `python -m pipeline.stitch.v2fit --source work/stitch-v2/photo-seed --out work/stitch-v2/photo-fit --steps 14400`. Real RGB plus finite spatial differences, eight physical row bands, anti-aliased RGB raster, differentiable auxiliary log-depth moments, calibrated visibility/depth/planarity and distortion priors. The renderer's ordinary depth is forward-only. Full object XYZ, DC, anisotropic covariance and opacity are trainable. Split/clone/prune at ⅓ and ⅔, export DC only. Ground is copied straight from the mosaic; none of its attributes is trained.
4. Form two guarded candidates. Run `python -m pipeline.stitch.v2guard work/stitch-v2/photo-fit work/stitch-v2/photo-guard`; `python -m pipeline.stitch.v2posterior work/stitch-v2/photo-fit work/stitch-v2/photo-filtered`; and `python -m pipeline.stitch.v2guard work/stitch-v2/photo-filtered work/stitch-v2/photo-clean`. The second candidate rechecks displaced-photo colour, depth and projected silhouette tips against eroded observed sky. The guard tightens or retires object ellipses whose conservative three-sigma enclosure crosses the swept hull or local ground. Select using the four viewer poses before calculating held-out scores; the report names the selected candidate. For the selected mixed model, run `python -m pipeline.stitch.v2select work/stitch-v2/photo-fit work/stitch-v2/photo-filtered work/stitch-v2/distance-mix` and `python -m pipeline.stitch.v2guard work/stitch-v2/distance-mix work/stitch-v2/distance-guard`, then `python -m pipeline.stitch.v2colour work/stitch-v2/distance-guard scene/stitched-opt-v2 6300`. The 25–65 m lateral-distance blend protects far buildings while giving repeated sky/free-space witnesses priority near the orbit.
5. Generate the official 16.2 m chase track on basebook: from `viewer/` run `bun tools/chase-track.ts ../scenes/road-stitched-v2 85 140 30 1440 900 > /path/to/chase-track.json`. From the repo root, run `bun pipeline/stitch/v2camera.ts scenes/road-stitched-v2/drive.json /path/to/chase-track.json /path/to/viewer-cameras.json`. Copy both JSON files to `work/stitch/report/v2/` on the Studio. The default pose is the tool's actual output, cross-checked against the pure viewer function. Orbit uses the same function with azimuth offsets and elevation offsets relative to 16.9°; 40° is absolute elevation. Both paths include the time-only heading, speed and height smoothing. The superseded 6 m and 19.5 m poses are not used for acceptance. For independent contract proof run `python -m pipeline.stitch.v2eval build`, `python -m pipeline.stitch.v2eval ground work/stitch-v2/packed quick`, `python -m pipeline.stitch.v2eval v1 scene/stitched-opt-v1`, `python -m pipeline.stitch.v2eval v2 scene/stitched-opt-v2`, `python -m pipeline.stitch.v2eval boards v1 v2`, `python -m pipeline.stitch.v2support` and `python -m pipeline.stitch.v2tracks`. Both models use the same 215 m compositor, controller matrices and panorama. Keep the full 1440×900 renders of the default, ±60° and elevated 40° poses, as well as the four 96-frame paired orbit videos. Camera metrics use the constant v1 scoring background and count holes.

**Weighted pseudo-view interface (optional).** Pass `--pseudo manifest.json` to the fit. The JSON is an array of objects with `rgb` (RGB image path), `weight` (same-height/width float32 NPY [0,1] or single-channel uint8/uint16 image), `K` (four numbers: fx, fy, cx, cy in pixels), and `T_world_cam` (4×4 camera-to-world transform, OpenCV camera frame). Pseudo poses are pinholes with a single timestamp pose. One in sixteen updates may use a pseudo target; pixel confidence multiplies the loss relative to the **entire patch**, at an additional 0.10 gain. Pseudo pixels never vote on metric depth, exposure, sky, or densification statistics. The published run used zero pseudo images. A direct weighted update was exercised separately; no genfill install or call is required.
