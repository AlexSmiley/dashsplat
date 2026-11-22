# 03 · Offline, globally consistent, highest-accuracy reconstruction of a whole drive

Research date: 2026-11-21. Scope: one 6-minute, 3.3 km Tesla Model 3 (HW4) drive, 6 cameras at ~36 fps, no LiDAR, no boxes, no factory calibration. Covers calibration, whole-drive poses, camera-only dynamic driving splats, camera models, large-scale training, tooling on CUDA vs Apple Silicon, and multi-drive fusion. Every claim has a URL. "Estimate" marks my own arithmetic. "Not found / unverified" marks things I could not confirm.

Search-tool note: Parallel search hit its free-tier rate limit early, so much of the later evidence comes from primary sources fetched directly (GitHub READMEs, COLMAP docs and changelog, arXiv API, Hugging Face API, and the two 2026 Tesla ISPRS PDFs).

---

## 0. Key findings

1. **Someone has already done this with a Highland Model 3 and published it.** Barazzetti et al. (ISPRS 3D-ARCH, Feb 2026) calibrated the Tesla camera rig with bundle adjustment from parking-lot drives, then built Gaussian splats of roads in Lombardy. Their findings:
   - Intrinsics and extrinsics are stable across sessions: baselines agree to 1–2 cm and rotations to 0.1–0.2°.
   - The rear camera gave unreliable estimates and was dropped.
   - Open 3 km sequences drift "several meters" without GNSS.
   - Splats show artifacts "along the road" because straight-line forward motion gives poor geometry.

   Source: https://isprs-archives.copernicus.org/articles/XLVIII-2-W12-2026/25/2026/isprs-archives-XLVIII-2-W12-2026-25-2026.pdf
2. **Tesla now ships per-frame telemetry inside the dashcam MP4s, as H.264 SEI.** The fields are speed, steering-wheel angle, lat/lon, heading, 3-axis linear acceleration, gear and `frame_seq_no`. There is **no gyro / angular rate**. This needs firmware ≥ 2025.44.25, and data may be missing while the car is parked. Sources: https://github.com/teslamotors/dashcam · https://raw.githubusercontent.com/teslamotors/dashcam/master/dashcam.proto
3. **COLMAP is now a complete rig-aware, prior-aware SfM stack**:
   - 4.0 (Mar 2026) merged GLOMAP as `global_mapper` and added ALIKED/LightGlue.
   - 4.1 (Jun 2026) added Caspar, a GPU bundle-adjustment backend with rig support, plus the EUCM camera model.
   - 4.2 (Aug 2026) added LoMa matching and a tangent-Sampson error for wide-FoV cameras, and fixed pose priors combined with rigs.

   Sources: https://github.com/colmap/colmap/releases/tag/4.0.0 · https://github.com/colmap/colmap/releases/tag/4.1.0 · https://colmap.github.io/changelog.html
4. **gsplat + 3DGUT is the practical rasteriser for Tesla optics.** It trains directly on distorted pinhole, fisheye (Kannala-Brandt) and f-theta cameras, with rolling shutter. Since Mar 2026 it also models "external distortion (e.g. windshield-style rigs)", which matters for the front camera behind the glass. It also has PPISP exposure/ISP compensation and multi-GPU support. **It is CUDA-only.** Sources: https://raw.githubusercontent.com/nerfstudio-project/gsplat/main/docs/3dgut.md · https://github.com/nerfstudio-project/gsplat (README news)
5. **Camera-only dynamic driving splats exist, but most code still assumes Waymo/PandaSet LiDAR and calibration.** The closest fits are:
   - IDSplat (CVPR Findings 2026, Apache-2.0): annotation-free; uses Grounded-SAM-2 masks and a rolling-shutter gsplat fork.
   - D2GS (NeurIPS 2025, MIT): LiDAR-free; uses metric-depth init and a road node.
   - AD-GS, DIAL-GS and DeSiRe-GS: self-supervised.

   None ships a "bring your own rig" loader. Expect to write a dataset adapter.
6. **Apple Silicon can train static, pinhole-only splats well** (Brush, msplat, splatx-metal, MetalSplat, OpenSplat). I found **no Metal trainer with fisheye, rolling shutter, or dynamic-actor support**. On the Mac, use undistorted or virtual-pinhole views. Do the serious training on the RTX 5070 or a rented GPU.

---

## 1. Calibrating the unknown rig from the footage

### 1.1 What the sensors are (as far as published)

| Camera (dashcam file) | Resolution | Lens evidence | Source |
|---|---|---|---|
| Front (main) | 2896×1876 | Barazzetti fit f ≈ 3638.7 px with an equidistant model, which gives HFOV ≈ 45.6° (estimate: 2·1448/3638.7 rad). SurveyXR (Model Y) calls the front camera "~45°" and uses Brown-Conrady with k3 fixed. | Barazzetti PDF (above); SurveyXR: https://isprs-archives.copernicus.org/articles/XLIX-B1-2026/341/2026/isprs-archives-XLIX-B1-2026-341-2026.pdf |
| Rear | 1448×938 | "~130°", Kannala-Brandt (`cv2.fisheye`) | SurveyXR PDF |
| L/R B-pillar, L/R repeater | 1448×938 | Treated as "fisheye-type", equidistant model | Barazzetti PDF |
| All | ~36 fps, compressed MP4, ~60 s files whose lengths differ by a few ms and sometimes by 1–2 frames | — | Barazzetti PDF |
| Sensor | HW4 "5 MP", reportedly Sony IMX963 | — | https://teslanorth.com/2025/12/02/tesla-might-be-swapping-cameras-heres-what-we-know/ (press, unverified) |
| Rolling-shutter readout time | **not found / unverified** | — | — |

Sync matters. One frame (27.8 ms) at 22 m/s is **0.61 m** of travel (estimate). The 1–2 frame length differences that Barazzetti observed are therefore far from negligible.

### 1.2 Lens models

- **COLMAP models available (Nov 2026):** `OPENCV` and `FULL_OPENCV` for pinhole with distortion; `OPENCV_FISHEYE` (Kannala-Brandt, k1–k4), `RADIAL_FISHEYE`, `THIN_PRISM_FISHEYE` and `RAD_TAN_THIN_PRISM_FISHEYE`; `FISHEYE` (equidistant, no distortion); `FOV`; `DIVISION`; `EUCM` (added in 4.1); and spherical/equirectangular. Sources: https://raw.githubusercontent.com/colmap/colmap/main/doc/cameras.rst · https://colmap.github.io/changelog.html
- **Double-sphere:** not found in COLMAP (unverified). EUCM is the nearest model.
- **gsplat/3DGUT models:** pinhole with OpenCV radial/tangential/thin-prism, `fisheye` (radial coefficients), `ftheta` and `ortho`. The 2026 "sensors library" lists pinhole, FTheta, fisheye and LiDAR. Sources: https://raw.githubusercontent.com/nerfstudio-project/gsplat/main/docs/3dgut.md · https://github.com/nerfstudio-project/gsplat
- **Recommendation:**
  - Use **OPENCV_FISHEYE** for all five 1448×938 cameras.
  - For the front camera, try both **OPENCV** and **OPENCV_FISHEYE**. Barazzetti found the two equally good on reprojection error and ground control, but the fisheye model gave "smaller and more balanced" coefficients, so they kept it.
  - Pick models that exist in both COLMAP and gsplat: OPENCV and OPENCV_FISHEYE. That way intrinsics carry straight into training.

### 1.3 Rig extrinsics with COLMAP

COLMAP's documented "unknown rig sensor poses" recipe works like this (https://raw.githubusercontent.com/colmap/colmap/main/doc/rigs.rst):

1. Lay out `rig1/<cam>/<frame>.jpg` with identical filenames per frame, then run `feature_extractor --ImageReader.single_camera_per_folder 1`.
2. Reconstruct without rig constraints. A partial reconstruction is fine, as long as each camera has at least one registered image in the same frame as a registered reference-camera image.
3. Run `rig_configurator --input_path <partial model>`. It averages `cam_from_rig` across registered frames.
4. Either run rig `bundle_adjuster`, or re-run `mapper` from scratch with rig constraints ("may lead to more accurate reconstruction results").
5. Configure rigs **before** sequential matching, so consecutive frames are matched across cameras.
6. Freeze the rig later with `--Mapper.ba_refine_sensor_from_rig 0`.

Related features:

- `pycolmap.apply_rig_config` allows finer control.
- 4.x lets you keep specific cameras or rigs constant (https://colmap.github.io/changelog.html).
- Caspar BA (4.1) supports rigs and is "1–2 orders of magnitude faster than Ceres CUDA" on medium and large problems (https://github.com/colmap/colmap/releases/tag/4.1.0).
- A practitioner write-up from June 2026 covers COLMAP rigs for extrinsic calibration: https://medium.com/@r.siddhesh96/extrinsic-camera-calibration-part-1-colmap-rigs-370d39c2b26f

**Why it works with little overlap.** As the car moves forward, a point seen by the front camera is later seen by the repeater and then the B-pillar camera. That temporal overlap ties the cameras together, which is how Barazzetti calibrated without shared simultaneous views. Their recipe:

- A parking lot, driven at 5–7 km/h in closed loops, in both directions, with transverse passes.
- Frames extracted at 1 fps.
- 252–441 rig frames per project.
- Joint BA of intrinsics and extrinsics.

Results:

- Reprojection RMS of 0.48–0.73 px.
- 3D check-point RMSE of 1.5–3.3 cm against total-station ground control.
- On reuse, they held intrinsics fixed and extrinsics soft-constrained at ±1.5 cm / ±0.1°.

### 1.4 Degeneracies with mostly-forward, planar motion

- **Planar (Ackermann) motion makes parts of camera-to-vehicle calibration unobservable.** OpenGV 2.0 targets exactly these "partial unobservabilities in the transformation variables that commonly occur for Ackermann-motion" for surround-view rigs with "a single camera facing into either direction and very limited overlap". It uses non-holonomic motion priors and a continuous-time back end. https://arxiv.org/abs/2503.03230 (code availability not verified)
- **Classic camera–odometry hand-eye:** CamOdoCal handles intrinsics plus extrinsics of multiple non-overlapping generic cameras plus odometry. It was tested on a car with 185° fisheyes. https://github.com/hengli/camodocal · https://people.inf.ethz.ch/marc.pollefeys/pubs/HengIROS13.pdf
- **Infrastructure-based variant** (map first, then localise each camera): https://people.inf.ethz.ch/marc.pollefeys/pubs/HengICRA14.pdf
- **Target-based option for non-overlapping rigs (2026):** Calousel uses pure rotation and one board, validated on a full-scale vehicle, with code released. Not targetless, but a cheap fallback. https://arxiv.org/abs/2606.25646
- **Practical consequences (my analysis; consistent with the sources above):**
  - Rotations between cameras, and each camera's rotation relative to the vehicle, are well constrained. On straights, a camera's translation direction gives the vehicle x-axis. The ground plane gives the vertical.
  - Camera height above the road, and vertical offsets between cameras, are weak under pure planar motion. They are only fixed by scene structure seen across cameras, which is why loops, turns and both driving directions matter.
  - Focal length versus distortion is ambiguous on straight, low-parallax segments. Calibrate on low-speed, high-turn segments.

### 1.5 Learned initialisers (useful for bootstrapping, not as final calibration)

| Tool | What it gives | Licence | Note |
|---|---|---|---|
| COLMAP `single_view_calibrator` (AnyCalib) | Per-image learned intrinsics before matching | Apache-2.0 (AnyCalib) | Built into COLMAP 4.x CLI: https://raw.githubusercontent.com/colmap/colmap/main/doc/cli.rst · https://github.com/javrtg/AnyCalib |
| COLMAP `view_graph_calibrator` | Focal lengths from two-view geometry, intended for `global_mapper` | BSD | Same doc |
| GeoCalib | Single-image focal length and gravity | Apache-2.0 | https://github.com/cvg/GeoCalib |
| VGGT / VGGT-X | Feed-forward poses and depth; VGGT-X scales to dense NVS | VGGT-1B weights CC-BY-NC (a separate commercial checkpoint exists) | https://github.com/Linketic/VGGT-X · https://huggingface.co/facebook/VGGT-1B-Commercial |
| VGGT-Long | Chunked, loop-closed, km-scale monocular sequences | see repo | https://github.com/DengKaiCQ/VGGT-Long |
| DriveVGGT | VGGT variant that uses rig calibration and rigid extrinsics for AD | not verified | https://arxiv.org/abs/2511.22264 |
| MapAnything | Metric multi-view reconstruction; accepts calibration, poses, depth | CC-BY-NC weights (`facebook/map-anything`) | https://github.com/facebookresearch/map-anything |
| π³ (Pi3) | Reference-free feed-forward geometry | BSD-2 (HF tag) | https://github.com/yyfz/Pi3 |
| Depth Anything 3 | Multi-view depth and pose | `DA3METRIC-LARGE` Apache-2.0; Giant/Large/Nested CC-BY-NC | https://huggingface.co/depth-anything/DA3METRIC-LARGE |

All of these assume pinhole-ish input. Feed them undistorted crops.

### 1.6 Practical recipe

1. **Frames for calibration:**
   - Drive-through manoeuvres, turns, any parking-lot section, and both directions on the same street if they exist.
   - Speed below 8 m/s, sampled at 1–2 fps.
   - Ideally record a **dedicated 5-minute parking-lot loop drive**, as Barazzetti did. It is the single highest-value action.
2. **Masks:**
   - A static ego-body mask per camera. COLMAP `camera_mask_path`; black means ignored.
   - Per-image dynamic-object masks via `mask_path`.

   Source: https://raw.githubusercontent.com/colmap/colmap/main/doc/faq.rst
3. **Scale:** integrate SEI `vehicle_speed_mps` between frames and compare with SfM translation. Alternatively, convert GPS to ENU and pass it as pose priors; COLMAP's prior-based alignment keeps metric scale. Barazzetti's rig-derived scale was within 1% of laser scan (ICP scale ≈ 0.99).
4. **Validate:**
   - Reprojection RMS below 0.8 px.
   - Extrinsics within a few cm of the physical layout and repeatable across two calibration subsets.
   - Watch the **rear camera**. Barazzetti found it inconsistent and excluded it.

---

## 2. Whole-drive pose estimation

### 2.1 Tools

- **`pose_prior_mapper`:** the incremental mapper with position priors. Set covariance with `--overwrite_priors_covariance` and `--prior_position_std_{x,y,z}` (default 1 m). https://raw.githubusercontent.com/colmap/colmap/main/doc/faq.rst
- **Rigs plus priors:** 4.2 fixed "pose-prior alignment and scale preservation for multi-camera rigs", so the combination is supported. https://colmap.github.io/changelog.html (a user issue on the combination: https://github.com/colmap/colmap/issues/3554)
- **Other COLMAP aids:**
  - Spatial matching from priors.
  - Sequential matching with loop detection.
  - Gravity priors.
  - Structure-less registration fallback through the generalized relative pose estimator (4.0), which helps frames without 3-view overlap.
  - Multi-component `global_mapper` (4.2).

  Sources: https://github.com/colmap/colmap/releases/tag/4.0.0 · https://colmap.github.io/changelog.html
- **Orientation (heading) priors in COLMAP:** not found. The docs only describe position priors and gravity. Treat as unverified.
- **Feeding SEI data:** COLMAP reads GPS from EXIF automatically. You can also write `PosePrior`s into the database with pycolmap. That is easier than writing EXIF; the exact pycolmap API should be checked against 4.2.

### 2.2 Building the prior trajectory from SEI (my design)

- Fuse SEI lat/lon (already Tesla-fused), heading, speed and acceleration in a small factor graph or EKF:
  - Speed × dt gives along-track odometry.
  - Heading gives yaw.
  - GPS gives absolute position at roughly 1–3 m.
- Output ENU positions with per-frame σ. Use σ ≈ 0.3–0.5 m relative and 2–3 m absolute (assumption).
- Pass these as COLMAP position priors. They bound the multi-metre open-loop drift that Barazzetti saw over 3 km without GNSS.
- SurveyXR (Model Y, external PPK GNSS) got 20–40 cm direct georeferencing residuals on the front camera alone, and 7–20 cm with ground control points. That is a realistic ceiling for GNSS-anchored global accuracy. Tesla's own fused GPS will be worse (unverified by how much).

### 2.3 Stationary segment (~2 min)

- Zero baseline means SfM cannot triangulate from those frames.
- Keyframe by distance, not time. Use SEI speed and a threshold of about 0.2 m/s. Keep about 1 frame per 2–3 s while stopped, just enough for the dynamic model. Excluding those frames from SfM and the static fit is my suggestion.
- Moving traffic past a stopped car can drag static Gaussians. Mask those frames' dynamic objects strictly.

### 2.4 Rolling shutter and time offsets

- **COLMAP:** no rolling-shutter camera model found in the docs or 4.x changelog (unverified absence).
- **Literature:** COLMAP poses alone are a poor basis for later rolling-shutter compensation, so joint refinement is better. https://arxiv.org/html/2403.13327v1 · Rolling-shutter self-calibration (2026): https://www.alphaxiv.org/abs/2608.01509
- **Where to handle it:** in splat training.
  - gsplat 3DGUT takes `rolling_shutter` plus `viewmats_rs` (the end-of-exposure pose).
  - SplatAD's gsplat fork takes per-camera `rolling_shutter_time` and direction.
  - XSIM (Feb 2026) generalises rolling shutter for AD on top of 3DGUT.

  Sources: https://raw.githubusercontent.com/nerfstudio-project/gsplat/main/docs/3dgut.md · https://github.com/carlinds/splatad/blob/main/gsplat/rendering.py · https://arxiv.org/abs/2602.05617
- **Readout time:** Tesla's is unknown, so treat it as a learnable per-camera scalar or sweep it (my suggestion).
- **Per-camera time offset:** model it explicitly as one learnable dt per camera, interpolating the rig pose. gsplat has "fused SE(3) … trajectory interpolation" helpers (README, Jul 2026).

### 2.5 Expected accuracy (from the Tesla papers)

| Setting | Result | Source |
|---|---|---|
| Calibrated 5-cam rig, parking lot, with GCPs | 1–3 cm 3D RMSE | Barazzetti |
| Dense cloud vs laser scan, church, two passes | 4.9 ± 5.1 cm | Barazzetti |
| 3 km open sequence, no GNSS, sequential matching | "several meters" accumulated error | Barazzetti |
| Front camera + PPK GNSS, direct georeferencing | 20–40 cm; 7–20 cm with 10 GCPs | SurveyXR |
| Surround-view driving SfM using rig constraints and road-plane filtering | qualitative, no numbers verified | MRASfM https://arxiv.org/abs/2510.15467 · MGSfM https://arxiv.org/abs/2507.03306 |

---

## 3. Driving-scene Gaussian splatting with dynamic objects, camera-only

### 3.1 Method comparison

Fit is for this project (1 = poor, 5 = excellent). "Boxes" means GT 3D boxes or tracklets.

| Method | Year / venue | Needs LiDAR? | Needs boxes? | Camera models | Dynamic handling | Code | Licence | VRAM (reported) | Apple Silicon | Fit |
|---|---|---|---|---|---|---|---|---|---|---|
| Street Gaussians | ECCV 2024 | Init uses LiDAR + SfM; SfM-only possible | Yes (tracked boxes, refined) | Pinhole (undistorted) | Per-vehicle rigid nodes | https://github.com/zju3dv/street_gaussians | Custom (NOASSERTION) | not verified | No | 2 |
| Street Gaussians w/o 3D tracker | ICCV 2025 | Waymo pipeline (LiDAR in prep, unverified) | No: 2D tracking foundation model + DINOv2 | Pinhole | Rigid nodes from 2D tracks | https://github.com/lolrudy/tracker-free-street-gaussian | Custom | not verified | No | 3 |
| OmniRe / drivestudio | ICLR 2025 | LiDAR init and supervision in paper; configurable (unverified for zero LiDAR) | Yes (instances/boxes; SMPL for humans) | Pinhole, multi-cam | Rigid vehicles, SMPL humans, deformable others; sky masks required | https://github.com/ziyc/drivestudio | MIT | not verified | No | 3 (best framework; needs boxes) |
| PVG | IJCV 2026 | LiDAR for init/depth in default configs (unverified) | No | Pinhole | Unified periodic-vibration Gaussians (no instances) | https://github.com/fudan-zvg/PVG | not verified | not verified | No | 3 |
| S³Gaussian | 2024 (repo updated Jan 2026) | LiDAR init in default (unverified) | No | Pinhole | 4D hex-plane deformation | https://github.com/nnanhuang/S3Gaussian | Custom | not verified | No | 2 |
| DeSiRe-GS | CVPR 2025 | Waymo LiDAR in configs (unverified) | No | Pinhole | Self-supervised motion masks + PVG-style, plus surface regularisation | https://github.com/chengweialan/DeSiRe-GS | not stated | not verified | No | 3 |
| EMD (plug-in) | ICCV 2025 | Inherits base (OmniRe / S³G) | No for S³G variant | Pinhole | Explicit motion embedding | https://github.com/qingpowuwu/emd | Apache-2.0 | not verified | No | 2 |
| HUGS (urban) | CVPR 2024 | No | Noisy 3D detections OK (unicycle model) | Pinhole | Rigid vehicles with physics prior | https://github.com/hyzhou404/HUGS | Custom | not verified | No | 3 |
| AutoSplat | ICRA 2025 | Yes (paper) | Yes + 3D templates | Pinhole | Road/sky constraints, reflected-Gaussian vehicles | Project page only: https://autosplat.github.io/ (code not found) | — | — | No | 1 |
| SplatAD | CVPR 2025 | Designed for LiDAR + camera | Yes | Pinhole + **rolling shutter** | Rigid actors | https://github.com/carlinds/splatad | Apache-2.0 | not verified | No | 2 (borrow its rolling-shutter gsplat fork) |
| IDSplat | CVPR Findings 2026 | Uses LiDAR on WOD/PandaSet; camera-only unverified | **No**: Grounded-SAM-2 tracked masks | Pinhole + rolling shutter (SplatAD fork) | Per-instance rigid dynamic Gaussians with learned motion | https://github.com/zenseact/idsplat | Apache-2.0 | README supports RTX 50-series wheels | No | **4** |
| AD-GS | ICCV 2025 (arXiv v3) | not verified | No | Pinhole | B-spline + trig motion, pseudo-2D segmentation | https://arxiv.org/abs/2507.12137 (code not verified) | — | — | No | 3 |
| DIAL-GS | 2025 | not verified | No | Pinhole | Instance-aware 4D Gaussians from warp inconsistency | https://arxiv.org/abs/2511.06632 (code not verified) | — | — | No | 3 |
| D²GS | NeurIPS 2025 | **No** (LiDAR-free) | Released config is Waymo 3-cam (uses dataset boxes, unverified) | Pinhole | drivestudio-style graph + **road node** + metric-depth init + diffusion depth enhancer | https://github.com/les1a/D2GS | MIT | not verified | No | **4** (geometry ideas) |
| SSTD-GS | 2026 (journal) | not verified | No | Pinhole | Spatiotemporal deformation + uncertainty masks | https://www.tandfonline.com/doi/full/10.1080/20964471.2025.2595830 | — | — | No | 2 |
| XSIM | Feb 2026 | Uses LiDAR (sim) | Yes (datasets) | 3DGUT + generalised rolling shutter | — | Code not verified: https://arxiv.org/abs/2602.05617 | — | — | No | 2 (ideas) |
| NVIDIA NuRec | product, 2025–26 | Ingests camera + LiDAR | Tracks in package | 3DGUT-based | Yes | https://docs.nvidia.com/nurec/ | NVIDIA terms | Datacentre GPUs (unverified) | No | 2 |
| SurfaceGS / DBM-GS / STHC-GS | CVPR-F / ICASSP 2026 | DBM-GS uses LiDAR pseudo-supervision | Varies | Pinhole | Surface-embedded / shadow-aware | CVPR-F 2026 papers; code not verified | — | — | No | 1–2 |
| Flux4D | Dec 2025 | Trains across many scenes | No | Pinhole | Unsupervised flow, feed-forward | https://arxiv.org/abs/2512.03210 | — | — | No | 1 (other researcher's mode) |

Benchmarks: driving methods are evaluated on Waymo NOTR / PandaSet with 3 front cameras at about 640×960 to 1600×1066, on a single A100. https://proceedings.iclr.cc/paper_files/paper/2025/file/d4938496368e1c5806138bea8837595c-Paper-Conference.pdf · https://arxiv.org/html/2605.21032. None of the above has been shown on 6-camera fisheye consumer footage (not found).

### 3.2 Getting boxes, masks and depth without LiDAR

- **Masks:**
  - SAM 2 (Apache-2.0) video segmentation: https://github.com/facebookresearch/sam2
  - IDSplat uses Grounded-SAM-2 tracked instance masks and documents the format: https://raw.githubusercontent.com/zenseact/idsplat/main/README.md
  - drivestudio requires sky masks (SegFormer) and optionally fine dynamic masks: https://raw.githubusercontent.com/ziyc/drivestudio/main/docs/Waymo.md
- **Pseudo-depth (licence-safe options):**
  - Depth Anything 3 `DA3METRIC-LARGE`, Apache-2.0: https://huggingface.co/depth-anything/DA3METRIC-LARGE
  - MoGe-2, MIT weights: https://huggingface.co/Ruicheng/moge-2-vitl-normal
  - Metric3D, BSD-2 code: https://github.com/YvanYin/Metric3D
  - UniDepthV2: https://github.com/lpiccinelli-eth/UniDepth (weights licence not verified)
  - UniK3D handles fisheye directly and was shown to beat SfM init for >180° fisheye: https://arxiv.org/abs/2508.06968
  - MTGS uses UniDepth for metric depth: https://github.com/OpenDriveLab/MTGS
- **3D boxes from cameras:** I did not verify a ready-made, licence-clean monocular 3D detector + tracker for fisheye. Practical routes:
  - (a) Lift SAM2 tracks into 3D with metric depth and fit boxes, as HUGS tolerates noisy boxes.
  - (b) Skip boxes and use IDSplat- or DeSiRe-style self-supervised decomposition.

### 3.3 Sky, exposure and road

- **Sky:** a masked sky/environment model is standard (drivestudio sky masks; AutoSplat sky constraints: https://arxiv.org/abs/2407.02598).
- **Exposure / ISP:** Tesla auto-exposure and tone mapping vary per camera and per frame. Three options:
  - **PPISP** (NVIDIA, CVPR 2026). Physically based ISP correction plus a controller that predicts parameters for novel views. Integrated into gsplat (Jan 2026) and works with MCMC (Jul 2026). https://github.com/nv-tlabs/ppisp · https://arxiv.org/abs/2601.18336
  - **Bilateral grids** (gsplat).
  - **Multi-scale bilateral grids unified with appearance codes, for driving** (NeurIPS 2025, MIT code). https://github.com/bigcileng/bilateral-driving
- **Road:** D²GS road node and plane priors; MRASfM plane filtering of road points (https://arxiv.org/abs/2510.15467). Barazzetti reports the road surface as the weakest area from linear forward motion.

---

## 4. Camera models in the rasteriser

| Option | Camera support | Rolling shutter | Trains on | Licence | Verdict |
|---|---|---|---|---|---|
| gsplat 3DGUT (`--with_ut --with_eval3d`, MCMC only) | Pinhole + OpenCV distortion, fisheye (KB), f-theta, ortho; **external distortion (windshield)** since Mar 2026 | Yes (`rolling_shutter`, `viewmats_rs`) | CUDA | Apache-2.0 | **Use this.** https://raw.githubusercontent.com/nerfstudio-project/gsplat/main/docs/3dgut.md |
| gsplat classic EWA | Pinhole, fisheye (Fisheye-GS-style), ortho | Partial (docs mention RS with 3DGUT) | CUDA | Apache-2.0 | Fine for undistorted pinhole |
| NVIDIA 3DGRUT (3DGRT ray tracing + 3DGUT) | OpenCV pinhole/fisheye, rolling shutter; NCore v4 AV datasets | Yes | CUDA + OptiX | Apache-2.0 | Reference implementation. https://github.com/nv-tlabs/3dgrut · https://research.nvidia.com/labs/sil/projects/ncore/ |
| Fisheye-GS | Equidistant fisheye (first-order) | No | CUDA | not stated | Author says use 3DGUT for strong distortion. https://github.com/zmliao/fisheye-gs |
| GOFish (Oct 2026) | Full FoV >180°, plus mesh | not stated | CUDA (unverified) | code not verified | Watch. https://arxiv.org/abs/2610.37818 |
| nerfstudio splatfacto | Pinhole only ("Currently, no" for fisheye) | No | CUDA | Apache-2.0 | Avoid for raw fisheye. https://docs.nerf.studio/nerfology/methods/splat.html |
| Metal trainers (Brush, msplat, splatx-metal, MetalSplat, OpenSplat) | Pinhole (no fisheye/distortion mentioned in READMEs) | No | Metal | various | Undistorted input only |

**Is undistorting to pinhole good enough?**

- **Front camera (~46° HFOV):** yes. Undistorting loses almost nothing. Barazzetti got nearly identical results from pinhole and fisheye models.
- **Side and rear fisheyes:** a single pinhole either crops the periphery or stretches it badly. Splitting each fisheye into 2–3 virtual pinhole views (cubemap-style) works with any trainer, including on the Mac. The cost is more images, seam handling, and resampling blur.
- **Evidence that native distorted training is better:**
  - On fisheye vehicle captures, 3DGUT+MCMC beat undistorted 3D-GS by **7.78 dB PSNR** (RigidDriveGS, Mar 2026; different domain, same optics issue): https://arxiv.org/abs/2603.26638
  - For ultra-wide lenses, about 160° of usable FoV was the sweet spot in a 200° study: https://arxiv.org/abs/2508.06968
  - Native rasterisation also lets rolling shutter and time offsets be modelled per pixel row, which virtual views lose.

---

## 5. Large scale: 3.3 km on a 12 GB GPU

### 5.1 Approaches

| Method | Year | Idea | Code / licence | Fit |
|---|---|---|---|---|
| Hierarchical 3DGS | SIGGRAPH 2024 | Coarse global "scaffold" → per-chunk training → consolidate into an LOD hierarchy and optimise intermediate nodes | https://github.com/graphdeco-inria/hierarchical-3d-gaussians (custom/Inria) | 4 (chunking recipe) |
| CityGaussian v1/v2 | ECCV 2024 / ICLR 2025 | Partitioned training; V2 adds better densification, geometry, lower memory | https://github.com/Linketic/CityGaussian (custom) | 3 (aerial-oriented) |
| Octree-GS / Scaffold-GS | TPAMI 2025 / CVPR 2024 | Anchor-based neural Gaussians; octree LOD | https://github.com/city-super/Octree-GS (custom) | 3 |
| Grendel-GS | ICLR 2025 | Multi-GPU partitioned training | https://github.com/nyu-systems/grendel-gs | 2 (cloud only) |
| gsplat multi-GPU dense 3DGS | Jul 2026 | Distributed rendering path | https://github.com/nerfstudio-project/gsplat | 3 (cloud) |
| Budgeted-GS | Oct 2026 | Post-hoc factoring-tree LOD sized to device memory; "budget-centred training" | https://arxiv.org/abs/2610.03162 (code not verified) | 4 (viewer LOD) |
| DOOM-GS | Nov 2026 | Out-of-core training for **dynamic** massive GS | https://arxiv.org/abs/2611.22805 (code not verified) | watch |
| UrbanGS | Feb 2026 | Scalable, geometry-accurate urban 3DGS | https://arxiv.org/abs/2602.02089 | watch |
| MRASfM aggregation | Oct 2025 | Multi-scene association and assembly at SfM level | https://arxiv.org/abs/2510.15467 | 3 |

### 5.2 Memory budget (estimate)

- **Per Gaussian at SH degree 3:** about 59 floats = 236 B of parameters. Gradients and two Adam moments add about 3× that, so roughly 0.95 KB per Gaussian while training. On top of that come rasteriser buffers that scale with image size.
- **On 12 GB:** about 3–5 M Gaussians at full 2896×1876 front resolution is a realistic ceiling.
- **Mitigations:** downscale front frames to about 1448 wide, use MCMC with a hard `cap_max`, use SH degree 2 for chunks, and use gsplat `packed` mode.
- **Whole-scene count:** street scenes typically need several million Gaussians per few hundred metres (assumption). The whole 3.3 km drive therefore needs **chunking**: about 250–400 m chunks with 50 m overlap, giving roughly 10–14 chunks.
- **Merging:** all chunks share one COLMAP world frame. Merge by cropping each chunk to its core region and concatenating, as Hierarchical-3DGS does, then build LOD.

### 5.3 Regularisation for low-parallax forward driving

- **Geometry-first primitives:** 2DGS (https://github.com/hbb1/2d-gaussian-splatting), PGSR planar GS (https://github.com/zju3dv/PGSR), and MetalSplat's 2DGS mode on the Mac.
- **Monocular metric depth as dense supervision:**
  - Align scale and shift per frame to SfM points.
  - Use as a dense init plus a depth loss.
  - D²GS does exactly this with progressive pruning and a road node (https://arxiv.org/abs/2510.25173).
  - Alternatively, VAD-GS densifies via MVS in unreliable regions (https://arxiv.org/html/2510.09364).
- **Diffusion repair for orbit / extrapolated views:**
  - **Difix3D+** (CVPR 2025): single-step diffusion fixer with progressive distillation back into 3D. https://github.com/nv-tlabs/Difix3D. The NVIDIA licence is not verified for commercial use. NuRec Fixer is the AV-post-trained variant: https://developer.nvidia.com/blog/accelerating-av-simulation-with-neural-reconstruction-and-world-foundation-models/
  - **GSFixer** (ICML 2026): reference-guided video diffusion. https://github.com/GVCLab/GSFixer
  - **SPVC** (Aug 2026): driving video fixing, but conditioned on 3D boxes and HD maps. https://arxiv.org/abs/2608.17420
  - **Lane-Switch GS:** plug-in for lateral extrapolation. https://lty2226262.github.io/lane-switch-gaussian-splatting/
  - Expect the orbit view above or behind the car to be the hardest extrapolation. Fixers help visually, but hallucinate.

---

## 6. Tooling and compute

### 6.1 CUDA (RTX 5070, 12 GB, Blackwell sm_120)

- **gsplat main:** needs PyTorch ≥ 2.7 and supports CUDA 12.8/13.x. Windows install is documented by a community tutorial. https://github.com/nerfstudio-project/gsplat
- **IDSplat:** README defaults to 50-series-compatible wheels.
- **Older repos** (drivestudio, Street Gaussians, PVG, DeSiRe, CityGaussian, Hierarchical-3DGS) pin older CUDA and PyTorch (e.g. cu118 / torch 2.2). They need porting for sm_120. This is a risk. Example: https://raw.githubusercontent.com/lolrudy/tracker-free-street-gaussian/main/README.md
- **COLMAP 4.0** fixed empty PatchMatch on compute ≥ 10.0 (Blackwell). Caspar requires compute ≥ 7.0. https://github.com/colmap/colmap/releases/tag/4.0.0 · https://colmap.github.io/changelog.html
- **3DGRUT:** needs CUDA + OptiX. https://radiancefields.com/platforms/nvidia
- **Linux vs Windows:** I recommend running everything under Ubuntu (dual-boot or WSL2) rather than native Windows. Not verified per repo.

### 6.2 Apple Silicon (M5 Ultra, 96 GB)

| Trainer | Tech | Camera models | Extras | Reported speed | Licence |
|---|---|---|---|---|---|
| Brush | Rust / Burn / WebGPU | COLMAP/Nerfstudio input; fisheye not mentioned | Masks, MCMC variant | v0.3 Sep 2025; CLI 1.0 community build Oct 2026 (https://huggingface.co/Merlef/brush-cli-builds) | Apache-2.0 https://github.com/ArthurBrussee/brush |
| msplat | Pure Metal kernels, C++ / Python / Swift | Pinhole | — | Mip-NeRF 360 7K iterations in ~59–80 s on M4 Max | Apache-2.0 https://github.com/rayanht/msplat |
| splatx-metal | Metal, numerically matches gsplat @77ab983 | Pinhole | MCMC, `--auto-cap` memory budget | garden 30K iterations, 65.7 min on M4 24 GB | Apache-2.0 https://github.com/PoSTMEDIA-AI/splatx-metal |
| MetalSplat | PyTorch MPS + Metal kernels | Pinhole | 3DGS and **2DGS** | garden 15K iterations, 42 min | MIT https://github.com/tchauffi/metalsplat |
| OpenSplat | libtorch, MPS | Pinhole (fisheye not mentioned) | Takes OpenSfM / COLMAP / nerfstudio input | — | **AGPL-3.0** https://github.com/pierotofy/OpenSplat |
| gsplat-mlx / gsplat-mps | MLX port / old gsplat 0.1.3 port | Pinhole | — | — | https://github.com/RobotFlow-Labs/gsplat-mlx · https://github.com/iffyloop/gsplat-mps |

None of these Metal trainers lists fisheye, rolling shutter, per-image ISP, depth loss (except MetalSplat's depth output) or dynamic actors (README review; absence unverified). The Mac is excellent for:

- Preprocessing: ffmpeg/VideoToolbox decode and the SEI parse.
- COLMAP on CPU. Slower, since there is no CUDA SIFT or Caspar; ALIKED/LightGlue run through ONNX, and their Metal acceleration is unverified.
- Depth and mask inference via PyTorch MPS (unverified per model).
- Fast static previews on virtual-pinhole views.
- Viewing.

**Training-time estimates (assumptions):** one 300 m chunk with about 1.5k images at 1448 width and 30K iterations is roughly 30–60 min on a 5070 with gsplat (no benchmark found for this setup). The whole drive in 10–14 chunks is roughly 6–12 GPU-hours on the 5070. A rented 80 GB GPU allows fewer, larger chunks and the dynamic models.

---

## 7. Multi-drive fusion (brief)

- **Localise a new drive into the existing map:** COLMAP's documented flow is `image_registrator` + `vocab_tree_matcher` with an image list (https://raw.githubusercontent.com/colmap/colmap/main/doc/faq.rst). SEI GPS priors give coarse pre-alignment; then run a joint BA.
- **MTGS:** multi-traversal GS with a shared static node, traversal-specific dynamic nodes and per-traversal appearance. Auto-reconstruction pipeline released Jan 2026. Apache-2.0. https://github.com/OpenDriveLab/MTGS
- **ADM-GS (Apr 2026):** separates traversal-invariant material from traversal-dependent illumination. https://arxiv.org/abs/2604.05908
- **Others:**
  - 3DGM: self-supervised, camera-only multi-traverse mapping that separates permanent from ephemeral content. https://arxiv.org/abs/2405.17187
  - GSA (CVPR 2026): splat-to-splat similarity alignment. https://bgu-cs-vil.github.io/GSA-project/
- **Barazzetti:** driving both directions and on multiple passes markedly improves road and façade geometry; lighting changes are the main obstacle.

---

## 8. Recommended pipeline

Legend: **[Mac]** = M5 Ultra · **[5070]** = RTX 5070 PC (Ubuntu or WSL2) · **[Cloud]** = rented 48–80 GB GPU (A6000/L40S/A100/H100).

1. **[Mac] Ingest.**
   - Decode all 6 streams to lossless PNG, or JPEG quality 95. Keep native resolution.
   - Extract SEI per frame with `teslamotors/dashcam` `sei_extractor.py`.
   - Build a per-camera frame table: file start time, frame index, `frame_seq_no`.
   - Check whether `frame_seq_no` aligns across cameras (open question). Flag files whose frame counts differ.
2. **[Mac] Prior trajectory and keyframes.**
   - Fuse lat/lon + heading + speed (+ accel) into a smoothed ENU trajectory: a small GTSAM or EKF script.
   - Choose rig keyframes **by distance**: about every 1.0 m for SfM and every 0.5 m for splat training. While stopped, keep about 1 frame per 3 s.
3. **[5070] Masks.**
   - One hand-made ego-body mask per camera.
   - Grounded-SAM-2 / SAM 2 tracked masks for vehicles, people and bikes on all keyframes.
   - Sky masks (SegFormer as in drivestudio, or SAM2-based).
4. **[5070] Rig calibration (COLMAP 4.2.x, CUDA, Caspar).**
   1. Prefer a **dedicated parking-lot loop drive** (5–7 km/h, both directions, 1 fps). Otherwise use the drive-through and turning segments of this drive.
   2. Models: `OPENCV_FISHEYE` on the five 1448 cameras. Front: OPENCV vs OPENCV_FISHEYE, initialised at f ≈ 3640 px.
   3. Run ALIKED+LightGlue (or SIFT) with masks, then `rig_configurator` (no extrinsics), sequential + loop matching, `mapper`, `rig_configurator --input_path`, and rig `bundle_adjuster` refining intrinsics and extrinsics.
   4. Validate: repeat on two disjoint subsets; extrinsics should agree within about 2 cm / 0.2°. If the rear camera is unstable, drop it from SfM as Barazzetti did, and only use it in splat training later with its intrinsics frozen.
5. **[5070] Whole-drive SfM.**
   - Write SEI ENU positions as pose priors (σ ≈ 2–3 m).
   - Run `pose_prior_mapper` with the rig frozen (`--Mapper.ba_refine_sensor_from_rig 0`), using sequential + spatial + loop matching.
   - Fallback or comparison: `global_mapper` (`view_graph_calibrator` not needed, since intrinsics are fixed).
   - Final rig BA with extrinsics soft-refined.
   - Check: front-camera trajectory vs GPS, and road-plane flatness.
6. **[5070 or Mac] Dense priors.**
   - Undistort or split each view into virtual pinholes.
   - Run DA3METRIC-LARGE (Apache) or MoGe-2 (MIT). For raw fisheye, consider UniK3D.
   - Fit per-frame scale and shift to SfM points.
   - Fuse into a voxel-downsampled static init cloud, excluding dynamic and sky masks.
   - Fit a road plane / height field.
7. **[5070] Static world, v1 (the core deliverable).**
   - gsplat `simple_trainer mcmc --with_ut --with_eval3d` on native distorted images, with `camera_model="fisheye"` for side and rear cameras.
   - Add PPISP (or bilateral grid), masks (ego and dynamic objects ignored), sky as background/env, depth loss from step 6, and pose refinement.
   - Add per-camera rolling-shutter time and time offset as learnable scalars (`viewmats_rs` interpolation).
   - Chunks of about 300 m with 50 m overlap, front downscaled to about 1448 px, `cap_max` around 3M.
   - Merge by core-crop + concatenate. Build LOD (Hierarchical-3DGS-style, or Budgeted-GS when code appears).
8. **[Cloud] Dynamic actors, v2.**
   - Port IDSplat (annotation-free, Grounded-SAM-2, rolling-shutter gsplat fork, Apache-2.0) or drivestudio/OmniRe (MIT).
   - Write a "Tesla" dataset adapter: rig poses, fisheye intrinsics, masks, mono-depth in place of LiDAR.
   - Train per chunk on 48–80 GB, then composite actors over the static v1 world.
   - Borrow D²GS's road node and depth-enhancer ideas for the ground.
9. **[Cloud or 5070] Orbit-view polish (optional).**
   - Render orbit trajectories and run Difix3D+ (check the licence) or GSFixer.
   - Distill fixed views back into the splat for a few thousand iterations.
   - Keep the un-fixed version for honesty checks.
10. **[Mac] Viewing and QA.** Use MetalSplatter, Brush or splatx-metal's viewer for PLY/SPZ. For quick iteration on chunks, the Mac can retrain static pinhole-only versions (msplat or splatx-metal) from step 5 outputs.

## 9. Top risks

1. **Calibration observability:**
   - Mostly-forward planar motion leaves camera heights and distortion weakly constrained.
   - The rear camera failed in the only published Tesla calibration.
   - Mitigation: a dedicated loop drive.
2. **Time sync and rolling shutter:**
   - Per-file length jitter and 1–2 frame mismatches can mean about 0.6 m per frame at 22 m/s.
   - The readout time is unknown.
   - Without per-camera dt and rolling-shutter modelling, cross-camera seams and ghosting are likely.
3. **Low-parallax floaters and road artifacts.** Barazzetti saw them even after a careful photogrammetric pipeline. The orbit view exposes them. Mitigate with depth/plane priors, 2DGS/PGSR regularisation and diffusion fixers.
4. **Dynamic objects without boxes:**
   - Self-supervised decomposition is fragile on consumer footage, especially during the 2-minute stationary segment.
   - All dynamic codebases assume Waymo/PandaSet loaders and often LiDAR, so expect adapter work.
5. **Toolchain and licences:**
   - Older research repos may not build for sm_120.
   - Several strong priors are non-commercial: VGGT-1B, MapAnything, the DA3 Giant/Large/Nested models, and possibly Difix.
   - OpenSplat is AGPL.

## 10. Open questions

1. Does `frame_seq_no` in SEI share one clock or counter across the six cameras? Is SEI present in Sentry/Dashcam clips while stationary in Drive (the docs only say it may be missing when parked)?
2. What is the HW4 cameras' rolling-shutter readout time and direction (not found)? Can it be self-calibrated well from this footage?
3. Can Ryan record a 5-minute calibration loop in an empty parking lot (both directions, slow)? This would make calibration much more reliable.
4. Is Tesla's SEI lat/lon accurate enough (σ?) to use as tight position priors, and how is it filtered? Is `heading_deg` true-north or GPS course?
5. Does COLMAP 4.2 accept orientation (yaw) priors, or only position and gravity? If only position, should yaw be fused externally?
6. Will IDSplat and D²GS train without any LiDAR once a depth-only init is supplied (camera-only not verified)?
7. Does the windshield measurably distort the front camera beyond the lens model? gsplat 3DGUT's external-distortion hook exists, but there is no Tesla-specific evidence.
8. Licensing goal: is this personal or research only, or could it ship? The answer decides whether the CC-BY-NC priors (VGGT, DA3 Giant, MapAnything) and Difix are usable.
