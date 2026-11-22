# Live world (v4)

[Slice results and photographs](REPORT-v4.md) · [chase video, 94–104 s](report/chase-v4-94-104.mp4) · [now + memory contract](DESIGN.md). Studio scene: `$RS/scene/live-v4/map.json`. The temporary mover layer is indexed by camera-local instance IDs and zero-based contiguous PLY row ranges in `$RS/maps/live-v4/instances.json`; frames with no valid depth report a null centroid.

From `$RS=/Users/anomaly/dev/reality-splat` on Studio:

1. `bash pipeline/livemap/run.sh v4-inputs` selects 4.5 Hz moving / 2 Hz stopped, excludes the original holdouts from every same-camera DA3 input window, and runs the unchanged depth producer with causal poses in a private root. Each physical camera needs a complete depth index and sky/road/instance masks before fusion.
2. `bash pipeline/livemap/run.sh v4-final` compiles the CPU engine, builds the scene, links the v2 ground, writes the panorama, and verifies tile data, playback times and the instance ↔ PLY rows. Physical z is converted to ray range; the camera’s last released v3 inverse-range fit is applied without the old birth veto.
3. `bash pipeline/livemap/run.sh v4-proof` renders the 60 held-out photos, five chase/orbit moments, transient overlays and the continuous 30 fps chase; it regenerates the report and measurements. Full orbit videos live in `$RS/work/livemap/`.
4. The later `bash pipeline/livemap/run.sh v4-whole` path uses a private whole-drive mask/depth workspace and a 60M generation cap. It writes a separate `scene/live-v4-whole/` candidate and `maps/live-v4/whole-instances.json` so a partial full-drive run cannot replace the verified slice. Before the first released correction it uses identity; after the last it carries the most recent fit forward.

# Static object layer (v3)

Builds the live and stitched **ground + static objects** scenes on anomastudio. The existing whole-drive ground is retained; object evidence on this run starts at 85 s and ends at 140 s. [Results, comparisons and weaknesses](REPORT.md) · [Implementation design](DESIGN.md).

**Studio scenes:** `$RS/scene/live-v3/map.json` and `$RS/scene/stitched-v3/map.json`. Start with the [exit orbit](report/orbit-exit.jpg) and [left-repeater photograph comparison](report/camera-left_repeater.jpg). The [arterial chase](report/chase-arterial.jpg) shows the live coverage limit.

## Reproduce on Studio

From `$RS=/Users/anomaly/dev/reality-splat`, use its existing `.venv/bin/python`. The input movie, rig, V2 poses including the causal release ledger, raw LK histories, fitted and reserved tracks, ground PLYs, photometry, masks, and depth-agent environment must be present. No model training or dependency installation is required.

1. `bash pipeline/livemap/run.sh inputs` snapshots the depth schedule and invokes the **unchanged** producer in two private roots under `maps/object-v3/{live,stitched}/depth`. It reuses matching existing offline files and never overwrites the shared depth. Check the per-camera index and physical frame IDs, not a directory's modification time. DA3-LARGE-1.1, 518 px, 1 m window spacing, five images per lens except the rear’s three frames × three virtual views; physical output is 724×469 optical-axis z, scale 1.0. Pose-conditioning on offline poses is not used on the live path.
2. `bash pipeline/livemap/run.sh audit` compares independent landmarks and ground, produces qualification profiles and fits the live prefix anchor model. Calibration and diagnostics are separate: the live builder cannot read future landmark fits. Poorly supported regions can be absent even when a physical depth file exists. See the numbers before increasing range.
3. `bash pipeline/livemap/run.sh front`, then `... sides`, then `... back`. Inspect the native-photo and chase plates under `maps/object-v3/stages/` between stages. Front is 85–110 s; the other stages use 85–140 s. Static geometry from a single uncertain observation is kept out of the PLY until a separated viewpoint corroborates it. Masked, out-of-frustum and occluded samples cannot vote to retire a record.
4. `bash pipeline/livemap/run.sh final` requires all scheduled training inputs and their window metadata. It publishes `scene/live-v3` and `scene/stitched-v3`. `bash pipeline/livemap/run.sh proof` verifies every tile, runs a 101 s prefix replay, renders the genuinely held-out physical lenses, pairs chase views and produces three slow 360° orbit videos. Text and stills go to `pipeline/livemap/`; videos stay in `maps/object-v3/orbits/`.

## Scene contract

Ground PLYs, rig and drive are hard links to the corresponding v2 export, with identical ground manifest entries. Object PLYs are independent 25 m tiles named `tiles/obj_<x>_<y>.ply`, indexed as static, layer `objects`. They use the same 24 little-endian float properties as the ground: world XYZ and normal; RGB as SH DC; opacity logit; log tangent/tangent/normal sigmas; unit quaternion (w,x,y,z); seven playback times. Smooth patches are thin; canopy patches have normal thickness. Confirmations change bookkeeping, never the live record’s position, colour, opacity or size. Retirement cross-fades at supported refinements; a fourth separated visibility run makes a continuation. Stitched exports final geometry once, born at zero. Extra stitched visibility intervals live in `maps/object-v3/stitched/final-extra-runs.npz`, addressed by object tile coordinates and record index.

The encoder and reference renderer work at original distorted camera angles, with the source-row exposure time and full rotated 3D covariance. The renderer reads only published PLYs; its dark clear colour means no stored surface. Held-out photos are unmodified. The chase viewpoint is 6 m back and 3 m up, 60° horizontal field of view. All scope and extrapolation figures are for static objects including coarse terrain; movers, sky and the viewer are separate projects.
