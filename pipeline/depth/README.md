# Depth and masks tools

Metric depth, sky masks and movable-object instance masks for every keyframe of every camera.
Results, bake-off numbers and the model choice are in [REPORT.md](REPORT.md).

All commands run on anomastudio from `$RS` (`~/dev/reality-splat`) with `.venv/bin/python`; masks scripts live
in `pipeline/masks/`. Camera maths comes from `pipeline/common/rig.py`; intrinsics, distortion, mounts, time
offsets and rolling shutter from `calib/rig.json`; poses from `poses/ego.npz` (planar telemetry if absent).
Every tool is resumable: existing outputs are skipped unless `--overwrite`. All files are written atomically (temp
file + rename); depth files are written before the `index.json` that lists them, so readers should treat a depth
frame as available only once it is in `index.json`, and a mask frame once its `.json` exists.

## Run

Whole drive, politely, in priority order (resumable; waits while free memory is below 20 %, one model per process):

```sh
nice -n 10 .venv/bin/python pipeline/depth/produce.py --phases 1,2,3
cat work/depth/produce_status.json; tail work/depth/produce.log    # progress
```

Single steps (all accept `--t0/--t1` to restrict to a time window):

```sh
# 1. keyframes: every 0.5 m of travel on the front clock, plus 2 Hz time-based keyframes while slower than 1 m/s
#    (kind "time"; --time-only for the parked start); other cameras take the nearest frame in time
.venv/bin/python pipeline/depth/keyframes.py --t0 286 --t1 330 --cams all
# 2. sky / road / movable-object masks with per-camera track ids          (~0.1 s per image)
.venv/bin/python pipeline/masks/run_masks.py --cams all
# 3. depth, live-map setting (--model auto: DA3-LARGE-1.1 on front and rear, MapAnything on pillars and
#    repeaters; pose-conditioned causal window of 5 keyframes >= 1 m apart, rear as 3 views x 3 frames, 518 px)
.venv/bin/python pipeline/depth/run_depth.py --cams all
#    offline/quality setting for front and rear: DA3-LARGE-1.1 at 1036 px (about 4x slower)
.venv/bin/python pipeline/depth/run_depth.py --cams front,back --width 1036 --overwrite
# 4. moving vs parked flags for every instance track (needs depth)
.venv/bin/python pipeline/masks/run_masks.py --cams all --motion
```

Use `--trajectory poses/ego_causal.npz` on `run_depth.py` and `run_masks.py --motion` for a strictly causal live map.

## Outputs

| Path | Content |
|---|---|
| `depth/<cam>/<n:06d>.npy` | float16 metres along the physical camera's z axis, 0 = invalid |
| `depth/<cam>/index.json` | per frame: kind, window frames, window span, model, road-plane height ratio, scale applied. A frame is complete when it is listed here |
| `depth/keyframes.json` → `depth.cameras.<cam>` | grid, virtual views (size, K, R virtual→camera), model and settings used for that camera |
| `depth/keyframes.json` | keyframe lists per camera (`frames`: n, t, t_key, odo, speed, kind = distance or time), and `depth`: quantity, grid, views, model, licence, trajectory |
| `masks/sky/<cam>/<n:06d>.png` | uint8, 255 = sky |
| `masks/road/<cam>/<n:06d>.png` | uint8, 255 = road surface (extra; used for scale checks) |
| `masks/dyn/<cam>/<n:06d>.png` | uint16 track id of movable objects, 0 = background |
| `masks/dyn/<cam>/<n:06d>.json` | per instance: id, class, score, area, bbox, object_id, moving, speed_mps. Written last: its existence marks the frame's masks complete |
| `masks/dyn/<cam>/tracks.json` | counts; objects (tracks linked across gaps) with speed, moving true/false/null; per-track observations with world positions |
| `masks/masks.json` | mask model, licence, grid |

**Grid.** Depth and masks share one grid per camera: the original (distorted) image resized to 724x469 (front 1/4,
others 1/2). Intrinsics for that grid are the `rig.json` params with fx, cx scaled by 724/W and fy, cy by 469/H;
distortion unchanged; COLMAP pixel centres (j+0.5, i+0.5). To unproject pixel (j, i): native uv =
((j+0.5)·W/724, (i+0.5)·H/469), ray r = `Camera.unproject(uv)` (unit), point = r · z / r_z.

**How depth is made.** Each image is rectified to virtual pinhole views (one for front, pillars and repeaters with the
lens's own principal point; for the rear fisheye three level views yawed -55/0/+55 degrees about the ego vertical,
20 degrees down, processed in one joint call). The model predicts z in each
view; that is converted to range along the ray, resampled (nearest, flying pixels removed first) onto the original grid,
and converted to z = range · r_z. Rays more than 88 degrees off the axis (rear fisheye rim), sky, the calibration mask
and depth edges are 0. Movable objects keep their depth (the transient layer needs it).

## Scripts

| Script | Purpose |
|---|---|
| `depth/rscommon.py` | paths, frame tables, virtual pinhole views, view-to-original resampling (on top of `common/rig.py`) |
| `depth/models.py` | uniform wrappers: DA3 (metric, any-view, nested), MapAnything, MoGe-2, UniDepth v2 |
| `depth/keyframes.py` | keyframe selection into `depth/keyframes.json` |
| `depth/run_depth.py` | production depth |
| `depth/bakeoff.py`, `depth/evalgeo.py` | bake-off: road plane, triangulation, temporal consistency, chained alignment |
| `depth/render.py` | colourised depth grids and fused chase-view renders |
| `depth/bench_speed.py` | speed / memory table (model calls) |
| `depth/bench_e2e.py` | end-to-end seconds per six-camera keyframe (masks + depth, production code path) |
| `depth/produce.py` | whole-drive producer in priority order, with status file |
| `depth/crosscam.py` | cross-camera and cross-time consistency of the delivered depth |
| `depth/render_prod.py` | chase-view and top-down renders of the delivered depth, all cameras fused |
| `depth/verify_outputs.py` | completeness / readability check of depth and masks against keyframes.json |
| `depth/apply_scale.py` | idempotent per-camera rescale of written depth (not used: no correction applied) |
| `depth/viz_depth.py` | image + colourised depth for given frames |
| `depth/summarize.py` | bake-off tables |
| `masks/seg.py`, `masks/run_masks.py` | Mask2Former panoptic segmentation, tracking, moving flags |
| `masks/viz_masks.py`, `masks/example_grid.py` | overlay for one frame; 2x2 example grid for the report |

## Models and licences

| Model | Used for | Licence |
|---|---|---|
| Depth Anything 3 DA3-LARGE-1.1 (`depth-anything/DA3-LARGE-1.1`) | **production depth, front and rear** | weights CC BY-NC 4.0, code Apache-2.0 |
| DA3METRIC-LARGE | fallback when a frame has no baseline behind it; bake-off | Apache-2.0 |
| DA3-BASE, DA3-SMALL | bake-off (commercial-safe alternative) | Apache-2.0 |
| DA3NESTED-GIANT-LARGE-1.1 | bake-off | CC BY-NC 4.0 |
| MapAnything (`facebook/map-anything`) | **production depth, pillars and repeaters** (`mapanything-nomask`) | weights CC BY-NC 4.0, code Apache-2.0 |
| MapAnything-apache (`facebook/map-anything-apache`) | bake-off (commercial-safe alternative) | Apache-2.0 |
| MoGe-2 ViT-L / ViT-B (`Ruicheng/moge-2-vitl-normal`) | bake-off | MIT code; see model card |
| UniDepth v2 ViT-L (`lpiccinelli/unidepth-v2-vitl14`) | bake-off | CC BY-NC 4.0 |
| Mask2Former Swin-L Cityscapes panoptic (`facebook/mask2former-swin-large-cityscapes-panoptic`) | **production masks** | MIT code; trained on Cityscapes (non-commercial dataset terms) |

The production set is non-commercial (personal project). The commercial-safe fallback is `--model mapanything-apache`
on every camera (close on the side cameras, worse on the front) or `--model da3-base` (about 2x worse on the front);
see REPORT.md.

## Environment notes

- Model repos are cloned in `$RS/third_party/` and installed editable with `--no-deps` into `.venv` (torch 2.15 and
  numpy untouched; extra packages were installed with the venv's existing versions as constraints).
- DA3 imports moviepy/open3d at module load for export features we do not use: `rscommon.stub_optional_modules`
  stubs them. UniDepth imports wandb the same way. MoGe-2 runs in fp32 (fp16 autocast fails on MPS).
- DA3's own pose alignment (evo Umeyama Sim3) fails on collinear camera centres (driving straight); `models.py`
  replaces it with a scale from the ratio of predicted to input camera baselines.
- MapAnything (NC weights) without poses returns a degenerate near-constant depth; always give it poses. The
  `mapanything-nomask` entry turns off its internal validity/edge mask so coverage matches the DA3 output.
- `KMP_DUPLICATE_LIB_OK=TRUE` is set (two OpenMP runtimes in the venv); CPU threads are capped at 8.
