# Rigid movers

Two additive viewer scenes for this drive: `live-v5-movers-v1` and `stitched-opt-v2-movers-v1`. Use ordinary playback for either; the viewer’s **Stitched** display mode deliberately hides all movers. [Results and limitations](REPORT.md).

## Studio run

Run from `~/dev/reality-splat` on anomastudio using the existing Python environment. Inputs: camera frames/tables, rig, causal and offline ego trajectories, `work/road/causal-lag.npy`, ground-height anchors, PNG instance IDs with their JSON sidecars, and available z-depth maps/indexes. Shared masks and private masks have **different ID namespaces**. Object seeds in [subjects.json](subjects.json) are specific to this drive and those namespace revisions.

~~~sh
export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
py=.venv/bin/python
$py -m pipeline.movers.inventory
$py -m pipeline.movers.associate
nice -n 10 $py -m pipeline.movers.solve       # optionally supply object IDs
nice -n 10 $py -m pipeline.movers.publish
nice -n 10 $py -m pipeline.movers.proof
nice -n 10 $py -m pipeline.movers.proof --prefix
~~~

After both manifests have been published, run `cd viewer && bun run cache:studio` on basebook. Proofs and counters are in `work/movers/proof`; per-mode bake counts are in `work/movers/{live,stitched}-models.json`. No parent scene manifest or tile is rewritten. Unchanged PLYs/caches are hard-linked; tagged legacy dynamic slices matched to published IDs are filtered into the live copy. Missing ground tile IDs are linked from `road-stitched-v2` into the hindsight copy. The later windows have ground but do not acquire a new static scenery reconstruction from this pipeline.

**Private junction inputs.** `python -m pipeline.movers.extra` creates a private keyframe schedule and input symlinks under `work/movers/extra`. Run the project's mask producer in that working directory with the repo on `PYTHONPATH`, for front and both pillars at 190–215 s, before running inventory. The private depth supplement uses the unchanged depth producer with five past same-camera frames, `--every 5 --trajectory poses/ego_causal.npz` for front and left pillar. Exact depth intersections are counted in the measurements. Depth is a diagnostic in live mode and a conditional occlusion check in hindsight.

## What a record means

1. **Evidence.** Each real mask supplies an image extent and a robust bottom contact; forward/backward-checked pyramidal LK carries it through up to six native frames. Withheld mask boxes do not reset LK. The tracker compares a translating 3D silhouette against a stationary, free-yaw fit. Both use fisheye calibration, row-time camera twist, contact on the local road tangent and available raised-ground anchors.
2. **Track.** Each pose is a rigid object frame on the ground (x forward, y left, z up). Live waits for the entire shutter and both master pose samples, fits released evidence at 12 Hz and advances at 36 Hz. The playback knots are delayed one tick so the viewer's interpolation does not use a future fit. Hindsight uses a full-episode size/velocity estimate and a local two-sided fit. The nominal tail is 0.72–0.75 s after last support; there is no long speculative bridge between separated episodes.
3. **Surface.** The loft, wheel sidewalls, end caps and underside are class priors. Masked, front-facing, z-buffered samples from real photographs create measured colour records on that geometry. Closer views can replace coarse discs with finer ones; replacement normally overlaps for 0.25 s. Each record has at most three visibility runs. The predicted PLY contains prior fill and mirrored colours, retired where real coverage arrives. An unseen roof is not a photographed roof.
4. **Causality boundary.** Live motion and colour accumulation are conditional on the supplied identity stream. Seeds are hand-reviewed, and optional association proposals use the offline trajectory, geometry and paint. This is a reconstruction with release-causal track/model updates, **not a demonstrated online object-discovery or re-identification system**. The prefix check drops unreleased observations, refits and rebakes before comparing track samples and PLY generations. Ground and ego inputs are precomputed with explicit availability; detector/compute wall-clock latency is not simulated.
