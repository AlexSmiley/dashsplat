# Photographic ground atlas

Source, export contract and independent ground-only comparisons for Reality Splat. [Viewer format](FORMAT.md) · [40 m ground extension and published viewer stills](REPORT-V6.md) · [V5 result and comparisons](REPORT-V5.md) · [Original v4 measurement](REPORT.md). All commands below run from the Studio Reality Splat root, with its existing Python environment. The build never modifies the rig, poses, source footage, splat PLYs or renderer.

### Build

Use fresh work/output directories for a new drive or changed solve. OpenCV and BLAS use one thread per worker. The initial pass can extend its own world-grid state across disjoint time intervals; camera IDs and native exposure numbers are deduplicated. Run the intervals in the stated order, publishing a manifest after each, with immutable texture filenames. The quality pass reads the completed initial state; its candidate remains separate until verified.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
py=.venv/bin/python
for interval in '85 140' '286 330' '0 85' '140 286' '330 361.42'; do
  set -- $interval; "$py" -m pipeline.groundtex.atlas "$1" "$2" --workers 5
done
export GT_WORK=work/groundtex-final/state GT_OUT=work/groundtex-final/ground GT_REPORT=work/groundtex-final/report
"$py" -m pipeline.groundtex.refine 0 361.42 --workers 6
"$py" -m pipeline.groundtex.verify
for mode in detail chase metrics hold; do "$py" -m pipeline.groundtex.proof "$mode"; done
"$py" -m pipeline.groundtex.profile
"$py" -m pipeline.groundtex.proof hold --interval 286 330
export PYTHONPATH="$PWD/envs/stitch/lib/python3.12/site-packages"
"$py" -m pipeline.groundtex.audit 85 140
"$py" -m pipeline.groundtex.sourcecheck
"$py" -m pipeline.groundtex.publish "$GT_OUT" "$GT_REPORT/verified.json"
```

This is POSIX shell syntax. The optional `prepare` driver starts the whole-drive fit while the base pass is still building; it reuses the early candidate's local warp, and recomputes its colours from the native frames. It expects that early candidate under `work/groundtex-candidate/{ground,state}`. Use the direct `refine` invocation above for a fresh standalone run.

### What is sampled

1. Select physically distinct exposures from *all* native frame times: one front view per 1.5 m of travel and one per 3.5 m for each other lens. Prefer a frame with actual semantic and mover masks when available. Omit the stitch physical holdout and reserve every 29th frame (index modulo 29 = 7). Use the row-timed camera-to-ground map, calibrated vignette/exposure, Jacobian resolution in its worst direction, actual image detail, range and mask margin. No assumed shutter speed, blur inversion or sharpening filter.
2. Register overlaps on a 4 cm working grid using forward/backward tracked photographic corners and robust 2D similarities. Solve binary graph cuts at 8 cm, penalizing cuts through high gradients or disagreement. With insufficient constraints keep the nominal mapping rather than estimating an unconstrained transform. There is **one source index per cell**, not six blended cameras.
3. Refine against a low-frequency geometry witness made from the existing verified ground splats. The original v4 ground provides the witness where available; elsewhere use stitched ground v2. High-pass patch correlation provides bounded displacements on a 32 cm smooth control lattice. A stripe with an ambiguous tangent constrains only the normal direction. Assign a whole connected marking and its pavement margin to one observed exposure wherever one can see it all. No splat colour is copied into the texture.
4. Estimate broad illumination residuals using neutral, flat, measured pixels across overlaps, with higher weight for good front observations. These gains are smooth in metres: they correct exposure/chroma, never blend detailed images. Per-frame residual similarities, robust registration errors and the local displacement/gain field are retained in `$GT_WORK/observations.json` and `$GT_WORK/fields/` (world origin XY, ENU warp in metres and BGR channel gain, 0.32 m pitch, axes south→north / west→east). These are ground mapping residuals, not an independently identified physical suspension or clock error.
5. Intersect rig validity, sky, dynamic instance and available mover-removal masks with semantic road or independent appearance support on the fitted road surface. The quality pass also requires nearby measured ground geometry. Recover a hole only from a surviving real photograph. The alpha rim is a signed-distance taper on support, not fabricated pavement. Texture each tile directly from native JPEG samples and export the shared height grid. The 4 cm ownership grid is never upsampled as the road's RGB.

### Reading the proof

`proof.py` decodes the actual WebPs and premultiplies for sampling. Chase renders intersect the exact SW–NE height triangles, apply mip sampling and four anisotropic taps, and retain v4 splats through transparent ground. Comparison poses are the viewer's actual 1440×900 transforms, not an approximation. The local v3 and Studio v4 drive files have identical SHA-1, `d4dc02dd4937a5dcb380a5d1a7363562556fdf97`. The ground-only plates deliberately omit HUD, ego and objects; they are not browser screenshots.

The camera holdout is reprojected in its **nominal** calibrated geometry. A different nearby unused photograph, projected to the same pixels, supplies the comparison floor. The paired-outline audit uses the existing v4 detector, an independently photographed reference, paint-selected regions and independent solo, self and injected controls. `sourcecheck.py` tests a different question: whether both sides of a paired outline already exist in either contributing, unmixed photograph. It checks across different owners and never compares a seam just to the already-composited atlas. Neither a smaller pixel pitch nor having only one owner proves the floor was reached. See [the report](REPORT.md) for raw counts and failures.

### V5 rebuild on Studio

The committed whole-drive v4 state is the input. Run the diagnostic baseline against it first; the original 85–140, 190–215 and 286–330 per-tile files are also in [report/v5/before](report/v5/before). Set the ground splat reference to v5, and use a fresh v5 state for reconstruction. `finish_v5` refines the cached exposure state in place; it is a stage, not an idempotent filter on its own output.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTHONPATH="$PWD/envs/stitch/lib/python3.12/site-packages" GT_SPLATS=scene/stitched-opt-v5
py=.venv/bin/python; base=work/groundtex-v5
export GT_WORK=work/groundtex-final/state GT_OUT=work/groundtex-final/ground GT_REPORT=$base/before
for w in '85 140' '190 215' '286 330'; do set -- $w; "$py" -m pipeline.groundtex.audit "$1" "$2"; done
"$py" -m pipeline.groundtex.proof detail --patches pipeline/groundtex/patches-v5.json
export GT_WORK=$base/state GT_OUT=$base/ground GT_REPORT=$base/final
"$py" -m pipeline.groundtex.v5 6
"$py" -m pipeline.groundtex.risk
"$py" -m pipeline.groundtex.finish_v5
"$py" -m pipeline.groundtex.entrance
"$py" -m pipeline.groundtex.verify
"$py" -m pipeline.groundtex.mask_audit
for w in '85 140' '190 215' '286 330'; do
  set -- $w; "$py" -m pipeline.groundtex.audit "$1" "$2"; "$py" -m pipeline.groundtex.audit "$1" "$2" 1
  "$py" -m pipeline.groundtex.sourcecheck "$1" "$2"; "$py" -m pipeline.groundtex.proof hold --interval "$1" "$2"
done
"$py" -m pipeline.groundtex.proof detail --patches pipeline/groundtex/patches-v5.json
"$py" -m pipeline.groundtex.proof chase; "$py" -m pipeline.groundtex.proof metrics
"$py" -m pipeline.groundtex.profile; "$py" -m pipeline.groundtex.compare_v5
"$py" -m pipeline.groundtex.surface_checks "$base/before"
for scene in stitched-opt-v5 stitched-opt-v5-movers-v1; do
  "$py" -m pipeline.groundtex.publish "$GT_OUT" "$GT_REPORT/verified.json" --dest "scene/$scene/ground" --drive-reference scene/stitched-opt-v4/drive.json
done
```

From the basebook checkout, run `cd viewer && bun run cache:studio` after publication. The RSB index is only for splat payloads; the viewer loads the atlas separately from the ground manifest.

### 40 m outer extension

The final v6 release depends on both the staged v4 and v5 photographic states on Studio, the existing 724×469 metric depths and masks, the calibrated pose stream, and the v5 object splats. Work in a fresh v6 work directory. This is an offline reconstruction; no unknown class is ever promoted to observed because it looks plausible in a render.

```sh
export GT_WORK=work/groundtex-v6 GT_OUT=work/groundtex-v6/ground-final GT_REPORT=work/groundtex-v6/final-report GT_TERRAIN=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 HF_HUB_OFFLINE=1
export PYTHONPATH="$PWD/envs/stitch/lib/python3.12/site-packages"; py=.venv/bin/python
"$py" -m pipeline.groundtex.outer 6          # native-depth evidence and registered source fields
"$py" -m pipeline.groundtex.terrain_classes # full Cityscapes labels, including low terrain vs vegetation
"$py" -m pipeline.groundtex.terrain_select  # source ownership after cross-view and semantic checks
"$py" -m pipeline.groundtex.ring_export 5   # one RGB owner or a labelled class continuation
"$py" -m pipeline.groundtex.ring_check verify
"$py" -m pipeline.groundtex.ring_check       # both alternating physical paint holdout splits
for scene in stitched-opt-v5 stitched-opt-v5-movers-v1; do
  "$py" -m pipeline.groundtex.publish "$GT_OUT" "$GT_REPORT/verified.json" --dest "scene/$scene/ground" --drive-reference scene/stitched-opt-v4/drive.json
done
```

Rebuild the cache from the basebook viewer checkout as above. After capturing the *published* default view at all four paused times, `ring_views` can project labels onto those frames using saved browser camera matrices; it requires the pre/post browser JPEGs and the pose JSON, not synthetic ground-only plates.
