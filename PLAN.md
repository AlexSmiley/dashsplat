# dashsplat: plan and data contract

Written on 21 November 2026 under the working name Reality Splat, which the stage reports still use.

Our own version of Tesla's Reality View: an orbitable Gaussian-splat world around the car, built from
the car's six dashcam cameras and the telemetry embedded in every frame.

Two products from one drive:

- **Live map (causal).** Built frame by frame using only data up to the current moment. Every splat carries
  the time it was first created (`t_birth`) and the time a camera last saw it (`t_seen`), so the viewer can
  play the drive back, scrub time, and show memory of places the cameras can no longer see (under the car,
  behind hedges, blind spots) with their age.
- **Stitched map (offline).** One globally consistent, optimised splat of the whole drive.

The viewer is a web app that looks and behaves like the Tesla screen: chase camera above the car, a car
model, planned-path ribbon, speed and status UI, drag to orbit.

Reference frames from Tesla's promo were in `reference/` (removed from the public repository: they are Tesla's images). Research is in `research/`.

## Where things live

| What | Where |
|---|---|
| Source (this repo) | basebook `/Users/vogel/dev/experiments/2026-11-21-tesla-reality-splat` |
| Compute, data and all artefacts | anomastudio `/Users/anomaly/dev/reality-splat` (called `$RS` below). M5 Ultra, 96 GB, no CUDA |
| Python env | `$RS/.venv` (uv, Python 3.12, torch 2.15 with MPS). Add packages with `uv pip install --python .venv/bin/python <pkg>` |

Code is edited on basebook and pushed per folder, for example
`ferry cp ./pipeline/calib anomastudio:dev/reality-splat/pipeline/`. Run with `ferry sh anomastudio '...'`
(`--detach` for long jobs, then `ferry cat anomastudio:<log>`). Each workstream owns its own folder and never
edits another one's. Nothing under `$RS/data` or any other artefact folder is ever committed.

## The drive

Six 1-minute clips, 2026-11-21 07:15:40 to 07:21:42, Jacksonville. One continuous 6-minute, 3.3 km drive:
about 80 s parked at a drive-through window, a slow turning exit through the car park (t = 80-105 s),
then suburban arterial roads at up to 22 m/s with several 90-degree turns and two long stops at lights.
Dawn, overcast, flat light, teal ISP tint, strong vignette, visible compression and motion blur.

Cameras (HW4, Sony IMX963, rolling shutter, about 36 fps):

| Name | Size | Notes |
|---|---|---|
| front | 2896x1876 | Narrow main camera. A 2026 paper fitted f = 3638.7 px (about 45 degrees wide) |
| back | 1448x938 | Full-circle fisheye, bumper visible at the bottom |
| left_pillar, right_pillar | 1448x938 | B-pillar, look sideways and forward, vignetted corners |
| left_repeater, right_repeater | 1448x938 | Front fender, look backward along the body; car body visible at the inner edge |

## Data contract

All paths are relative to `$RS`.

### Frames and telemetry (ingest)

- `data/drive/frames/<cam>/<n:06d>.jpg`: every decoded frame. `n` is a running index per camera.
- `data/drive/cams/<cam>.json`: one row per frame: `n, clip, idx, pts, seq, t`, the raw telemetry
  (`speed, heading, lat, lon, steer, ax, ay, az, ap, brake, blink_l, blink_r, gear`; a missing key means 0 or false)
  and the dead-reckoned ego pose at that frame's time: `ego_x, ego_y, ego_yaw, ego_speed`.
- `data/drive/timeline.npz`: master clock from the front camera at about 36 Hz:
  `t, seq, x, y, yaw, speed, heading, gps_x, gps_y, lat, lon, steer, ax, ay, az, ap, brake, blink_l, blink_r, gear`.
- `data/drive/meta.json`: fps, origin lat/lon, per-camera frame-counter offsets, image sizes.

Conventions:

- **World**: ENU metres, x east, y north, z up, origin at the first front frame.
- **Ego**: x forward, y left, z up, origin at the rear-axle centre on the ground.
- **Camera**: OpenCV (x right, y down, z forward). `T_ego_cam` maps camera coordinates to ego coordinates.
- **yaw**: counter-clockwise from east, radians. Tesla `heading` is degrees clockwise from north.
- **Time**: `t` is seconds on the front camera's frame counter, `(seq - seq0) / fps`. Other cameras are placed
  on it by matching identical telemetry tuples, which is good to about one frame (28 ms). Side cameras share
  the front counter; the back camera's counter runs about 8.6 frames ahead. Sub-frame offsets per camera are
  still unknown and are part of calibration.
- `x, y` are integrated from speed and heading (smooth). They end 7.5 m from the GPS track after 3.3 km.
  The telemetry has no height, pitch or roll, and its acceleration has gravity removed.

### Calibration (`calib/`)

- `calib/rig.json`:

      {"cameras": {"front": {"model": "OPENCV_FISHEYE", "width": 2896, "height": 1876,
                             "params": [fx, fy, cx, cy, k1, k2, k3, k4],
                             "T_ego_cam": [[4x4 row-major]], "time_offset_s": 0.0,
                             "rs_readout_s": null, "mask": "masks/front.png"}}}

  Models are COLMAP's `OPENCV` or `OPENCV_FISHEYE` with COLMAP parameter order. A frame with table time `t`
  was exposed at ego time `t + time_offset_s`.
- `calib/masks/<cam>.png`: 255 where the pixel sees the world, 0 for car body, vignette and image-circle border.

### Poses (`poses/`)

- `poses/ego.npz`: `t` (N) and `T_world_ego` (N,4,4), the refined 6-DoF trajectory on the master clock.

### Per-frame products

- `depth/<cam>/<n:06d>.npy`: float16 metres, 0 = invalid, on the original image grid (possibly downscaled). `depth/keyframes.json`
  lists the keyframes and states the quantity (range along the calibrated ray, or z) and the resolution.
- `masks/sky/<cam>/<n:06d>.png` and `masks/dyn/<cam>/<n:06d>.png` (instance ids of movable objects, 0 = background).

### Viewer scene (`scene/<name>/`)

- `drive.json`: `{fps, duration, origin: {lat, lon}, t: [], x: [], y: [], z: [], yaw: [], pitch: [], roll: [], speed: [], steer: [], ap: [], brake: [], blink_l: [], blink_r: [], gear: [], lat: [], lon: []}`, world frame as above.
- `rig.json`: a copy of the calibration, for the camera-view mode and frustum overlays.
- `map.json`: `{t_end, tiles: [{id, kind, file, count, bbox: [minx, miny, minz, maxx, maxy, maxz], t_min, t_max}], sky: "sky.jpg" or null}`.
  `kind` is `static` (indexed by space) or `transient` (moving objects, indexed by the time span `t_min..t_max`).
- `sky.jpg`: optional equirectangular panorama in the world frame (centre column faces east, up is +z).
- Tile files: standard 3DGS PLY (`x y z`, `scale_0..2` log, `rot_0..3` wxyz, `opacity` logit, `f_dc_0..2`) plus seven
  float properties in seconds: `t_birth, t_death, e0, s1, e1, s2, e2`. A splat exists while `t_birth <= T < t_death`.
  Cameras could see it during up to three runs: `[t_birth, e0]`, `[s1, e1]`, `[s2, e2]`. Unused values are
  `1e9`. At playback time `T`:

      seen = min(T, e0); if (T >= s1) seen = max(seen, min(T, e1)); if (T >= s2) seen = max(seen, min(T, e2));
      age = T - seen      // 0 while a camera sees it, otherwise how long ago it was last seen

  Splats are immutable: when the map learns something better, the old splat gets a `t_death` and a new one is
  born, with about 0.25 s of overlap for a cross-fade.

### Moving objects (`objects.json` in the scene folder)

Movers are not re-drawn from scratch each frame. Each tracked vehicle or person is one rigid object with its
own splats in its own frame, and a smooth trajectory. As the car sees more of it, its model fills in, like a
photo scan that improves; the viewer moves the whole model along the trajectory at display rate, so motion is
smooth however sparse the keyframes are.

    {"objects": [{"id": 17, "class": "car", "size": [4.6, 1.9, 1.5],
                  "file": "objects/17.ply", "predicted_file": "objects/17.pred.ply",
                  "t_first": 201.3, "t_last": 214.9,
                  "track": {"t": [], "x": [], "y": [], "z": [], "yaw": [], "state": []}}]}

- **Object frame**: x forward, y left, z up, origin on the ground under the centre of the object's box.
  `track` gives the pose of that frame in the world (yaw counter-clockwise from east, radians), sampled at
  10 Hz or better and smoothed; the viewer interpolates. `state` is 0 while a camera sees the object and 1
  while its pose is predicted from its last motion (blind spots, occlusion). The object exists from
  `t_first` to `t_last` (which includes any predicted tail) and fades in and out.
- **`file`**: a PLY with the same properties as a map tile, positions in the object frame. The time properties
  are in drive time: a part of the car first scanned at 203.1 s has `t_birth` = 203.1, coarse splats are
  superseded by finer ones as with the static map, so the model visibly gains detail as it is observed.
- **`predicted_file`** (optional): splats that were never observed, inferred from the object's left-right
  symmetry or a class shape prior, so a car seen only from one side is not a hollow shell. The viewer may
  draw them softer; they are superseded where real observations arrive.
- Parked vehicles are static map content, not objects, once the motion test says they do not move.

## Reasoning the map should do

Every layer should use three kinds of evidence, and say which one a splat rests on:

- **Spatial (what is seen now, including far ahead).** The front camera sees the road 60-80 m ahead long before
  it can be drawn sharply. That far view is a suggestion for the ground ahead: where the road goes, where the
  lane lines are (bearing is accurate even when range is not), how the surface slopes. Roads are continuous, so
  where nothing has been seen yet the map may extend the surface and continue lane lines a short way as a
  prediction.
- **Historical (what was seen before).** Earlier observations are kept, not thrown away when a closer one
  arrives: a far observation that was sure about sideways position and unsure about distance should be fused
  with the near one direction by direction. Places no camera can see now (under the car, between cameras,
  behind things) are drawn from memory with their age.
- **Temporal (how things change).** Movers keep moving when no camera sees them: a car that enters a blind
  spot is carried forward with its last velocity instead of freezing, with growing uncertainty, until it is
  seen again or the prediction expires.

Three states result, and the viewer may style them differently: **seen** (a camera sees it now), **remembered**
(seen earlier; has an age) and **predicted** (never directly seen at this time; inferred from continuity or
motion). Predicted content goes in its own tiles with `"layer": "...-predicted"` so it can never be mistaken
for an observation, and is superseded as soon as a real observation arrives.

## Two clocks

The world and the things moving in it are updated at different rates, and the compute budget follows:

- **Static world: slow, detailed.** The ground and fixed scenery live in world coordinates, so between updates
  nothing needs recomputing: the viewer simply renders the same splats from the car's new position at display
  rate. Updates can therefore be infrequent (every half metre to a metre of travel, a few per second) and spend
  their time on detail: the heavier depth model, higher resolution, more views.
- **Movers: fast, light.** Vehicles and people get their own splat models and a track. The track is updated at
  camera rate (36 Hz) from cheap evidence (masks, 2D tracking, ground contact, a light depth model), and the
  viewer interpolates it at display rate; the model itself fills in at the slow rate as more of the object is
  seen.

A real-time version budgets one heavy static update and several light mover updates per cycle.

## Stages

1. **Ingest**: clips to frames, telemetry, dead-reckoned trajectory. `pipeline/ingest`
2. **Calibration**: intrinsics, distortion, camera positions on the car, time offsets, body masks. `pipeline/calib`
3. **Poses**: 6-DoF trajectory for the whole drive from telemetry plus vision. `pipeline/poses`
4. **Depth and masks**: metric depth, sky, moving objects. `pipeline/depth`, `pipeline/masks`
5. **Live map**: causal fusion into time-stamped splats. `pipeline/livemap`
6. **Stitched map**: optimisation of the whole drive. `pipeline/stitch`
7. **Viewer**: `viewer/`

## Licences

Personal project. Several of the best models (VGGT, the large Depth Anything 3 models, MapAnything's
default weights, DUSt3R/MASt3R) are non-commercial. Each stage's README notes the licence of every model it uses.
