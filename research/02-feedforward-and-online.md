# 02 · Feed-forward and online / streaming reconstruction (mode A) — state of the art, Nov 2026

Research helper report for this project (then called Reality Splat). Scope: feed-forward geometry models, driving-specific feed-forward Gaussian splatting, online/SLAM-style splatting, temporal memory, and ego-motion priors. Researched 2026-11-21. Every claim carries a source URL; "unverified" means I could not confirm it from a primary source in this session.

**Method note.** Licence and repo facts come from the GitHub API / raw LICENSE files / Hugging Face model API fetched on 2026-11-21. Some web-search tools hit rate limits mid-session, so a few secondary facts (marked) could not be cross-checked.

---

## TL;DR

- **The 2026 field has split into three layers you can mix:** (1) *geometry foundation models* that turn a window of images into poses + metric depth/point maps (VGGT-Ω, MapAnything, Depth Anything 3, Pi3X, WorldMirror 2.0, DVGT); (2) *streaming wrappers* that make them run on endless video with bounded memory (DA3-Streaming, VGGT-Long/Pi-Long, StreamVGGT/InfiniteVGGT/OVGGT, LongStream, VGGT-SLAM 2.0, AMB3R-SLAM); (3) *Gaussian map builders*, either feed-forward-online (ReCoSplat, StreamSplat, UniSplat, OF³GS) or optimisation-online (on-the-fly NVS, its multi-camera-rig version, HIBS).
- **Nothing off the shelf does what you want end-to-end** (6-camera Tesla rig incl. fisheye + dynamic objects + persistent timestamped memory + Metal). The closest single papers are **UniSplat** (driving, feed-forward, keeps a *persistent memory of static Gaussians* beyond current camera coverage) and **On-the-fly Large-scale 3D Reconstruction from Multi-Camera Rigs** (calibration-free online 3DGS from a rig). Neither handles fisheye natively.
- **Licences are the main trap.** Most of the best weights are non-commercial (VGGT-1B, VGGT-Ω, Pi3/Pi3X, DA3 Large/Giant, MapAnything default, StreamVGGT, CUT3R, DUSt3R/MASt3R, DGGT, all NVIDIA-licence repos). Permissive options: **MapAnything-apache**, **DA3-Small/Base**, **VGGT-1B-Commercial** (custom AUP licence), **AnySplat (MIT)**, **DVGT-1 (Apache)**, **UniSplat code (Apache)**.
- **Apple Silicon is now realistic for the geometry layer**: MLX ports exist for VGGT-Ω, MapAnything (both licences) and DA3 incl. a DA3-Streaming port; MPS ports exist for VGGT and DA3. The splat *training/optimisation* layer is the weak spot on Metal (OpenSplat and Brush run on Metal; gsplat/CUDA-kernel-based research code does not, MPS status unverified).
- **Recommendation (detail at the end):** build mode A as *your own incremental Gaussian map* fed by a windowed metric geometry model that accepts your odometry/rig priors — **MapAnything-apache** first, **DA3 (Streaming + pose-conditioning + GS head)** second — and use **UniSplat** (or DGGT) as the reference design / optional learned head for dynamic-vs-static separation and memory.

---

## 1. Feed-forward geometry / pose foundation models

### 1.1 VGGT (CVPR 2025) and VGGT-Ω (CVPR 2026 Oral)
- **Outputs:** cameras (extrinsics + intrinsics), depth, point maps, tracks in one pass (https://huggingface.co/facebook/VGGT-1B). VGGT-Ω simplifies to a single dense head, adds *register attention* (inter-frame exchange through registers, partly replacing global attention), supports **dynamic scenes**, uses ~30% of VGGT's training memory and is 1.6× faster at inference (https://arxiv.org/abs/2605.15195, https://cvpr.thecvf.com/virtual/2026/oral/40381).
- **Priors in:** none (images only) — not found for VGGT-Ω.
- **Memory/frames:** VGGT-Ω-1B-512 peak memory on A100 at 624×416: 6.0 GB (1 frame), 9.7 GB (50), 13.4 GB (100), 20.8 GB (200), 43.2 GB (500) (https://github.com/facebookresearch/vggt-omega README). Paper: VGGT/VGGT-Ω handle ~1250 frames where DA3 OOMs at ~750; an all-register variant cuts 1000-frame runtime from 240 s to 11.7 s at lower accuracy (https://vggt-omega.github.io/assets/paper/preview_v3.pdf). Blog benchmark: ~24 ms/frame for small batches (https://learnopencv.com/vggt-vs-vggt-%CF%89-vggt-omega-a-complete-guide-to-feed-forward-3d-reconstruction) (secondary source).
- **Metric scale:** not claimed (unverified).
- **Licence:** VGGT-1B weights CC-BY-NC-4.0 (https://huggingface.co/facebook/VGGT-1B); **VGGT-1B-Commercial** under a custom "vggt-aup-license", gated (https://huggingface.co/facebook/VGGT-1B-Commercial). VGGT-Ω code + weights: **FAIR Noncommercial Research License**, gated (https://github.com/facebookresearch/vggt-omega/blob/main/LICENSE, https://huggingface.co/facebook/VGGT-Omega).
- **Apple:** VGGT on MPS via community port (https://github.com/jmanhype/vggt-mps); **VGGT-Ω in MLX** (https://huggingface.co/mlx-community/VGGT-Omega-1B-512-bf16).

### 1.2 π³ / Pi3X (ICLR 2026)
- Permutation-equivariant (no reference frame), predicts affine-invariant poses and scale-invariant local point maps (https://github.com/yyfz/Pi3).
- **Pi3X (Dec 2025):** conv head (fewer grid artifacts), better confidence, **optional conditioning on camera poses, intrinsics and depth**, and **approximate metric scale**; has a VO example (`example_vo.py`) (https://github.com/yyfz/Pi3).
- **Licence:** code BSD-3; **weights CC BY-NC 4.0 (strictly non-commercial)** (https://github.com/yyfz/Pi3, https://huggingface.co/yyfz233/Pi3X).
- **Apple:** no MLX/MPS port found (unverified).
- Long-sequence: **Pi-Long** applies VGGT-Long's chunk/align/loop framework to Pi3 (https://github.com/DengKaiCQ/Pi-Long; repo carries the VGGT License text).

### 1.3 MapAnything (Meta, 3DV 2026)
- **Inputs:** images plus *any subset* of intrinsics **or per-pixel ray directions**, depth, camera poses (OpenCV cam2world) and an `is_metric_scale` flag; one model covers 12+ tasks (https://map-anything.github.io/, https://github.com/facebookresearch/map-anything README).
- **Outputs:** metric point maps, depth along ray, ray directions, pinhole intrinsics, cam2world poses, confidence, metric scaling factor (README).
- **Why it matters here:** the `ray_directions` input is the cleanest published way to feed a *calibrated fisheye* (or any central camera) without undistorting, and the pose input is where your odometry goes. Whether its training covered strongly distorted ray maps is **unverified** — test it.
- **Scale:** memory-efficient mode runs "up to 2000 views on 140 GB" (README).
- **Licence:** code Apache-2.0; weights **facebook/map-anything (CC-BY-NC-4.0)** and **facebook/map-anything-apache (Apache-2.0, commercial OK)**, same API (README; https://huggingface.co/facebook/map-anything-apache).
- **Apple:** MLX ports for both licences (https://huggingface.co/mlx-community/map-anything-apache-fp16, https://huggingface.co/mlx-community/map-anything-fp16); listed as "Stable" in appautomaton/mlx-spatial (https://github.com/appautomaton/mlx-spatial).
- Also a useful *harness*: the repo wraps VGGT, VGGT-Ω, Pi3/Pi3X, DA3, DUSt3R, MASt3R, MUSt3R, MoGe behind one output format for benchmarking (README).

### 1.4 Depth Anything 3 (ByteDance Seed, ICLR 2026)
- **Inputs/outputs:** any number of views with or without poses; depth + ray maps → poses, intrinsics, point clouds; **pose-conditioned depth**; a **3D Gaussian head** (Giant models); nested metric series (any-view Giant + metric Large) for real-world scale; sky segmentation (https://github.com/ByteDance-Seed/Depth-Anything-3).
- **Retrained "-1.1" checkpoints** fix a training bug and are "much better for street scenes" (README).
- **Speed / capacity (A100 80 GB, 504×336):** DA3-Giant 900–1000 images max, 37.6 FPS; Large 1500–1600, 78.4 FPS; Base 2100–2200, 126.5 FPS; Small 4000–4100, 160.5 FPS; VGGT reference 34.1 FPS (https://arxiv.org/html/2511.10647, Table 8).
- **DA3-Streaming:** VGGT-Long-style sliding-window chunking with Sim(3) alignment and loop closure, "ultra-long video … with less than 12 GB GPU memory" (https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/da3_streaming/README.md) — fits the RTX 5070.
- **Licence (per model):** Nested-Giant-Large(-1.1), Giant(-1.1), Large(-1.1): **CC BY-NC 4.0**; **Small and Base: Apache-2.0**; code Apache-2.0 (README table). Note the GS head is only on Giant/Nested (non-commercial).
- **Apple:** MPS benchmarks exist (https://github.com/DEEPESH468/Depth-Anything3-Benchmark; https://pypi.org/project/awesome-depth-anything-3); **mlx-swift port including DA3-Streaming** (https://github.com/mnmly/mlx-swift-da3, Apache-2.0, very new: 1 star); Apple's own Core AI model zoo reportedly lists depth-anything-3-small for macOS (https://ithome.me/post/2026/10/26/on-device-depth-estimation-coreml-vision, secondary).

### 1.5 WorldMirror (ICML 2026) / WorldMirror 2.0 (HY-World 2.0, Apr 2026)
- **Inputs:** images plus any subset of camera poses, intrinsics, depth ("any-prior prompting"); **prior injection yields gains across all tasks** (https://arxiv.org/html/2510.10726).
- **Outputs:** point clouds, depth, normals, cameras, **3DGS attributes** (https://github.com/Tencent-Hunyuan/HY-World-2.0/blob/main/DOCUMENTATION.md). 2.0 adds normalized RoPE (flexible resolution), depth-mask prediction, sequence-parallel/FSDP/BF16 inference; default target size 952 px; video extraction defaults to ≤32 frames (same doc).
- **Licence:** Tencent community licences that **do not apply in the EU, UK and South Korea** (https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror License.txt; HY-World-2.0 License.txt).
- **Apple:** no port found (unverified).

### 1.6 DUSt3R / MASt3R, CUT3R and the streaming/recurrent line
- **DUSt3R/MASt3R:** pairwise point maps + global alignment; **CC BY-NC-SA 4.0** (https://github.com/naver/dust3r, https://github.com/naver/mast3r LICENSE). Superseded for this use by multi-view models.
- **CUT3R (CVPR 2025 Oral):** recurrent model with persistent state; online metric point maps; **16.58 FPS on KITTI at 512×144 on A100** (https://arxiv.org/html/2501.12387, Table 2). Code **CC BY-NC-SA 4.0** (https://github.com/CUT3R/CUT3R). Degrades past its training length; **TTT3R** (ICLR 2026) adds a closed-form confidence-based state-update rule to improve length generalisation (https://arxiv.org/html/2509.26645; repo carries CUT3R's CC BY-NC-SA + MIT, https://github.com/Inception3D/TTT3R). **Mem3R** (WACV 2027) splits tracking (TTT fast-weight MLP) from mapping (token state), −39% ATE vs CUT3R+TTT3R on long TUM sequences (https://arxiv.org/abs/2604.07279).
- **StreamVGGT (ICLR 2026):** causal VGGT with cached tokens (https://github.com/wzzheng/StreamVGGT); code CC BY-NC-SA, weights CC-BY-NC (https://huggingface.co/lch01/StreamVGGT). KV cache grows linearly → OOM within hundreds of frames (https://arxiv.org/abs/2603.05959).
- **Bounded-memory fixes (2026):** **InfiniteVGGT** (Apache-2.0, https://github.com/AutoLab-SAI-SJTU/InfiniteVGGT), **OVGGT** (training-free, constant VRAM, https://arxiv.org/abs/2603.05959, code https://github.com/VAISR/OVGGT), **IncVGGT** (ICLR 2026, https://proceedings.iclr.cc/paper_files/paper/2026/hash/0a4096f09ab230082df548f901d4f79e-Abstract-Conference.html), **STAC** cache compression (CVPR 2026, https://openaccess.thecvf.com/content/CVPR2026/papers/Wang_STAC_Plug-and-Play_Spatio-Temporal_Aware_Cache_Compression_for_Streaming_3D_Reconstruction_CVPR_2026_paper.pdf), **STream3R** (ICLR 2026, S-Lab non-commercial licence, https://github.com/NIRVANALAN/STream3R).
- **LongStream (CVPR 2026):** keyframe-relative poses (no first-frame anchor), disentangled metric scale, cache refresh; **"stable, metric-scale reconstruction over kilometer-scale sequences at 18 FPS"** (https://openaccess.thecvf.com/content/CVPR2026/html/Cheng_LongStream_Long-Sequence_Streaming_Autoregressive_Visual_Geometry_CVPR_2026_paper.html). Code/licence: not found.

### 1.7 Long-sequence chunk-and-align
- **VGGT-Long (ICRA 2026):** chunk → overlap Sim(3) align → loop closure; km-scale KITTI without calibration (https://arxiv.org/abs/2507.16443). Repo carries the VGGT License (https://github.com/DengKaiCQ/VGGT-Long). It is the template behind DA3-Streaming and Pi-Long.

### 1.8 Driving-specific geometry transformers
- **DVGT (CVPR 2026):** metric global point map in ego frame + ego poses from unposed multi-view sequences, **no intrinsics/extrinsics needed**, arbitrary camera configs; trained on nuScenes, OpenScene, Waymo, KITTI, DDAD (https://arxiv.org/abs/2512.16919). **DVGT-2** (Apr 2026): streaming (causal attention + sliding-window cache), joint geometry + planning (https://arxiv.org/abs/2604.00813). Code Apache-2.0 (https://github.com/wzzheng/DVGT); DVGT-1 weights Apache-2.0 (1.7B), DVGT-2 weights (1.8B) with no licence tag (https://huggingface.co/api/models/RainyNight/DVGT-2). Tested with CUDA 12.8 (README). Fisheye generalisation: unverified (none of its training sets is fisheye-heavy).
- **DriveVGGT:** uses known calibration (camera baselines → absolute scale), per-camera temporal attention, windowed multi-camera attention, factorised rig/ego-motion heads; −49% inference time vs VGGT on long sequences (https://arxiv.org/abs/2511.22264). Code: not found.
- **Rig3R (Wayve, NeurIPS 2025):** VGGT-style model conditioned on optional camera ID, timestamp and rig-relative raymap, and can *discover* the rig when metadata is missing (https://arxiv.org/abs/2506.02265, https://wayve.ai/thinking/rig3r/). No official code; community reimplementation (https://github.com/engichang1467/Open-Rig3R).

### 1.9 Non-pinhole / fisheye-capable feed-forward models (important for the Tesla back camera)
- **Any3R** (Oct 2026): conditions positional encoding and loss on the dense ray field, any central camera, mixed camera types (https://arxiv.org/abs/2610.36150). Code: unverified.
- **Camera3R** (NeurIPS 2026): LoRA (0.5% params, <1 day training) that adapts a perspective 3R model to other camera models; code at https://github.com/android-xr/camera3r (https://arxiv.org/abs/2611.13453).
- **MEOW / "Many Eyes, One World"** (Sep 2026): mixed perspective/fisheye/panorama, no calibration, synthetic-only training, zero-shot to real (https://arxiv.org/abs/2609.35658).
- **CAM3R** (Mar 2026): two-view camera-agnostic model + ray-aware global alignment (https://arxiv.org/abs/2603.22631).
- **X-Lens** (2026): real-time metric depth from calibrated fisheye + pinhole views (https://arxiv.org/html/2607.12993v2).
- **MASt3R-SLAM** assumes only a unique camera centre, no parametric model (https://edexheim.github.io/mast3r-slam/).
- Practical fallback: undistort the fisheye into 2–3 virtual pinhole crops; or feed MapAnything `ray_directions`.

### 1.10 Feed-forward Gaussian heads on general scenes
- **AnySplat (SIGGRAPH Asia 2025):** unposed, uncalibrated images → 3D Gaussians + poses in one pass; MIT code and weights (https://github.com/OpenRobotLab/AnySplat, https://huggingface.co/lhjiang/anysplat). Voxel-aligned; reported to produce millions of Gaussians and floaters on large scenes (AnchorSplat comparison: 5.39 M Gaussians, 7.5 s; https://openaccess.thecvf.com/content/CVPR2026/papers/Zhang_AnchorSplat_Feed-Forward_3D_Gaussian_Splatting_With_3D_Geometric_Priors_CVPR_2026_paper.pdf).
- DA3-Giant GS head, WorldMirror GS head (above).

---

## 2. Feed-forward driving-scene Gaussian splatting from surround cameras

| Model | Key idea | Poses needed? | Dynamics | Trained on | Code / weights / licence |
|---|---|---|---|---|---|
| **DrivingForward** (AAAI 2025) | Self-supervised pose + depth + Gaussian nets, small overlap, flexible single/multi-frame input, real-time claim (https://fangzhou2000.github.io/projects/drivingforward) | No extrinsics in training; uses intrinsics (paper says pose/depth nets self-supervised without depth GT or extrinsics, https://www.alphaxiv.org/abs/2409.12753) | None explicit | nuScenes | MIT (https://github.com/fangzhou2000/DrivingForward) |
| **Omni-Scene** (CVPR 2025) | Pixel + volume "Omni-Gaussians" for ego-centric sparse views; diffusion extension for generation (https://cvpr.thecvf.com/virtual/2025/poster/34857) | Yes (unverified detail) | None explicit | nuScenes | MIT (https://github.com/WU-CVGL/Omni-Scene) |
| **STORM** (ICLR 2025) | Transformer predicts Gaussians + **per-Gaussian velocities**, aggregates all frames via self-supervised scene flow → amodal reconstruction; emergent dynamic masks; 200 ms for a 2-s clip (https://arxiv.org/html/2501.00602) | **Yes** (Plücker ray maps from intrinsics + extrinsics, same paper) | Velocity / scene flow | Waymo, nuScenes, AV2 | NVIDIA licence (non-commercial); **no official checkpoint** ("cannot share an official checkpoint") (https://github.com/NVlabs/GaussianSTORM) |
| **EVolSplat** (CVPR 2025) | Metric depth → global point cloud → sparse 3D CNN volume → Gaussians; hemisphere model for far field (https://xdimlab.github.io/EVolSplat) | Yes (posed images) | Static only | KITTI-360, Waymo | Code linked from project page; licence unverified |
| **Flux4D** (Waabi, 2025) | Unsupervised, predicts Gaussians + motion in 3D from multi-sensor input in seconds (https://chatpaper.com/zh-CN/paper/215778) | Uses LiDAR ("multi-sensor") | Motion params | driving datasets | Code: not found |
| **UniSplat** (ICLR 2026) | 3D latent scaffold from foundation models, spatio-temporal fusion in the scaffold, point + voxel decoder, **per-Gaussian dynamic score and a persistent memory bank of static Gaussians for streaming scene completion beyond current coverage** (https://arxiv.org/abs/2511.04595, https://alphaxiv.org/abs/2511.04595) | Yes (calibrated rig, unverified detail) | Dynamic probability per Gaussian; memory stores only static | Waymo | **Code Apache-2.0**, Waymo weights on HF (https://github.com/chenshi3/UniSplat, https://huggingface.co/chenchenshi/UniSplat); training code released May 2026; scene-completion demo "soon" (README) |
| **DGGT** (CVPR 2026) | VGGT-based, **pose-free**, per-frame Gaussian maps + cameras, dynamic head, **lifespan head** (visibility over time), diffusion refinement; zero-shot across datasets; ~0.4 s (https://arxiv.org/abs/2512.03004, https://openaccess.thecvf.com/content/CVPR2026/papers/Chen_DGGT_Feedforward_4D_Reconstruction_of_Dynamic_Driving_Scenes_using_Unposed_CVPR_2026_paper.pdf) | **No** | Dynamic head + lifespan; uses external TAPIP3D for tracking (https://github.com/xiaomi-research/dggt/issues) | Waymo, nuScenes, AV2 | Code Apache-2.0; **weights CC-BY-NC-4.0** (https://github.com/xiaomi-research/dggt, https://huggingface.co/api/models/xiaomi-research/dggt); only Waymo 1-view checkpoint released so far (README) |
| **ReconDrive** (Mar 2026) | Extends VGGT with hybrid Gaussian heads + static-dynamic composition with velocities (https://arxiv.org/abs/2603.07552) | unverified | Velocity | nuScenes | Code: not found |
| **PointForward / V2** (2026) | Sparse point-aligned 3D queries; V2 adds static/dynamic registers, 4D PE (https://arxiv.org/abs/2610.28120) | Yes | **Needs 3D boxes** for dynamic instances (https://papernotes.org/NeurIPS2026/autonomous_driving/pointforward_feedforward_driving_reconstruction_through_point-aligned_representa/) | Waymo etc. | Code: not found |
| **FreeForward4D** (Nov 2026) | **RGB-only, annotation-free**: VGGT-Ω windows calibrated to metric with MoGe-2, registered across time; Grounding-DINO + SAM 2 tracks associated in 3D; object velocity + local Gaussian corrections; 0.21 s per 3-view 2-s clip + 13 ms per render on RTX 4090 (given precomputed conditions); beats STORM, DGGT, etc. on Waymo (https://arxiv.org/abs/2611.24007, https://arxiv.org/html/2611.24007v1) | No (estimates its own) | Object-level velocity | Waymo; RGB-only fine-tune on nuScenes, AV2 | "Code will be released" (paper) |
| **IGRecon4D** (Nov 2026) | Instance-prior distillation + LiDAR-anchored Gaussians (https://arxiv.org/abs/2611.21126) | Yes + sparse LiDAR | Dynamic masks | Waymo → nuScenes | Not applicable (needs LiDAR) |
| **Drive2Gauss / WorldSplat / FreeGen** (2026) | Generative-video → 4D Gaussians (https://arxiv.org/abs/2610.26585; https://media.eventhosts.cc/Conferences/ECCV2026/pdfs/3841.pdf) | — | — | nuScenes | Generation-oriented; relevant for gap filling only |

**Generalisation to a different rig with fisheye.** No paper in this family evaluates on a fisheye surround rig (none found). DGGT and FreeForward4D report cross-dataset (Waymo↔nuScenes↔AV2) transfer, but all three datasets are pinhole rigs (https://arxiv.org/abs/2512.03004, https://arxiv.org/abs/2611.24007). DVGT claims arbitrary camera configurations (https://arxiv.org/abs/2512.16919). Expect to undistort the Tesla back camera (and possibly repeaters) to pinhole or to fine-tune; the FreeForward4D recipe (RGB-only self-supervision from foundation-model priors) is the most transferable *training* recipe for an unlabelled Tesla dataset.

**Licence warning (all Waymo-trained weights):** weights trained on the Waymo Open Dataset are typically bound by its non-commercial terms — not verified in this session; check https://waymo.com/open/terms/ before any non-research use.

---

## 3. Online / incremental / SLAM-style Gaussian splatting

### 3.1 Optimisation-online (Gaussians spawned and refined during capture)
- **On-the-fly NVS (Meuleman et al., SIGGRAPH/TOG 2025):** sequential capture, learned matcher + fast pose init, Gaussians spawned per keyframe and refined briefly; live viewer. Constraints: ordered frames with >2/3 overlap, **pinhole only (optimises focal only, no fisheye/distortion)**, **no loop closure**, best at 1–2 MP. CUDA (tested CUDA 11.8/12.8 on Ubuntu/Windows) (https://github.com/graphdeco-inria/on-the-fly-nvs README). Licence: Inria non-commercial-style (LICENSE.md; "Inria holds all ownership rights").
- **On-the-fly Large-scale 3D Reconstruction from Multi-Camera Rigs** (Dec 2025): first on-the-fly framework for rigs; **no calibration needed** (hierarchical camera initialisation picks a central camera and aligns others), lightweight multi-camera BA, redundancy-free Gaussian sampling, frequency-aware scheduler; "hundreds of meters within 2 minutes"; new RigScapes dataset (5 helmet cameras, 1 km road) (https://arxiv.org/abs/2512.08498, https://arxiv.org/html/2512.08498v1). Notes: experiments limited to ≤3 cameras because baselines drift; captures avoided "excessive dynamic objects" (same page). Code: not found.
- **HIBS — Hierarchical Incremental Beta Splatting** (WACV 2027, Franke — co-author of on-the-fly NVS): beta kernels as a confidence dial, confidence-aware spawning from multi-view depth agreement, *error-driven relocation recycles stale primitives*, closed-form merging gives an **LOD hierarchy** (https://arxiv.org/abs/2611.08288). Code: unverified.
- **MCGS-SLAM** (ICRA 2026): multi-camera Gaussian-splatting SLAM (ETH), code link on page (https://mcgs-slam.github.io/).
- **GLAM-SLAM** (IROS 2026): real-time large-scale outdoor monocular GS-SLAM with feature-based front end, sparse anchor grid, scene partitioning; KITTI, Oxford RobotCar, Málaga (https://arxiv.org/abs/2607.21416, code https://github.com/pmermigkas/GLAM-SLAM/).
- **KiloGS-SLAM** (Jun 2026): km-scale monocular GS-SLAM with *lifecycle-managed Gaussian mapping* (probabilistic init, chunked densify/prune) (https://arxiv.org/abs/2606.30436).
- **LingGS** (Oct 2026): frozen streaming 3D foundation model as "virtual RGB-D sensor" + explicit geometric tracking + streaming Gaussian mapper, ~7 FPS on RTX 4090 (https://arxiv.org/abs/2610.36853).
- **MonoGS** (CVPR 2024; Imperial licence, https://github.com/muskie82/MonoGS) and **Splat-SLAM** (Google, https://github.com/google-research/Splat-SLAM) are indoor-oriented predecessors; not a fit for long forward driving.

### 3.2 Feed-forward online (no per-scene optimisation)
- **ReCoSplat** (Mar 2026): online feed-forward GS with **posed or unposed input, with or without intrinsics**; render-and-compare conditioning; KV-cache compression (>90%); **256-view streams at 45.1 FPS average input throughput on RTX 6000 Ada**; built on a π³-style backbone; code + weights released (https://arxiv.org/abs/2603.09968). Licence: unverified (likely inherits π³ weight restrictions — check).
- **StreamSplat (Aug 2026):** calibrated views, **voxel-aligned causal cache** so memory grows with explored geometry not stream length, history-projected depth anchoring; code "upon acceptance" (https://arxiv.org/abs/2608.01659).
- **StreamSplat (ICLR 2026, different paper):** online *dynamic* 3DGS from uncalibrated video with bidirectional deformation and adaptive fusion of emerging/vanishing Gaussians (https://arxiv.org/abs/2506.08862).
- **OF³GS / FreeStreamGS** (Jun 2026): on-the-fly feed-forward GS from unposed images; voxel-based global Gaussian state cache (https://arxiv.org/abs/2606.03254, https://arxiv.org/pdf/2606.03254v1).
- **SaLon3R** (Oct 2025): online generalizable GS, 50+ views at >10 FPS, 50–90% redundancy removal via saliency-aware anchor quantisation (https://arxiv.org/abs/2510.15072; MIT, https://github.com/wrld/SaLon3R).
- **LongSplat (AAAI 2026, Huang et al.):** global 3DGS set + streaming update that compresses redundant history (https://ojs.aaai.org/index.php/AAAI/article/view/42504/46465). (Not the NVIDIA LongSplat, which is per-scene optimisation for casual long videos, NVIDIA licence: https://github.com/NVlabs/LongSplat.)
- **DynStream** (Oct 2026): windowed 4D Gaussians incrementally aligned and fused into a global scene from unposed video (https://arxiv.org/abs/2610.09720).
- **UniSplat** (§2) is the driving-specific member of this group.

### 3.3 Feed-forward SLAM back ends (poses + dense geometry, not splats)
- **VGGT-SLAM 2.0** (RSS 2026): VGGT submaps on an SL(4) factor graph without the 15-DoF drift, attention-based loop verification, real time on Jetson Thor; BSD-2 repo (https://arxiv.org/abs/2601.19887, https://github.com/MIT-SPARK/VGGT-SLAM, https://gtsam.org/2026/06/24/vggt-slam.html). GTSAM back end makes odometry/GPS factors easy to add.
- **MASt3R-SLAM** (CVPR 2025): ~15 FPS, generic central camera; CC BY-NC-SA (https://opencv.org/blog/mast3r-slam/, https://github.com/rmurai0610/MASt3R-SLAM).
- **AMB3R-SLAM** (Sep 2026): km-scale, >10k frames on one consumer GPU, hierarchical back end, **handles dynamic scenes without static-world BA**, optional LiDAR (https://arxiv.org/abs/2609.19518; project https://hengyiwang.github.io/projects/amber-slam).
- **F²SLAM** (Oct 2026): turns feed-forward geometry into persistent factors in a dense factor graph (https://arxiv.org/abs/2610.05207).
- **SLAM3R** (CVPR 2025 Highlight; CC BY-NC-SA, https://github.com/PKU-VCL-3DV/SLAM3R).

---

## 4. Temporal memory, persistence, dynamics, generative fill

**Per-Gaussian time attributes already in the literature**
- **PVG** (IJCV 2026): each Gaussian has a life peak time and lifespan (temporal opacity) plus periodic vibration; MIT code (https://fudan-zvg.github.io/PVG/, https://github.com/fudan-zvg/PVG).
- **STORM:** per-Gaussian velocities, aggregates all timesteps for amodal views (https://arxiv.org/html/2501.00602).
- **DGGT:** explicit **lifespan head** modulating visibility over time (https://arxiv.org/abs/2512.03004).
- **UniSplat:** per-Gaussian dynamic probability; **memory bank keeps only static Gaussians**, transformed into the current frame, to complete regions outside current coverage (https://alphaxiv.org/abs/2511.04595). This is the closest published analogue of "this was seen N seconds ago".
- **Temporal confidence:** TCGSplat renders per-Gaussian temporal confidence via alpha compositing (https://www.researchsquare.com/article/rs-10579135/v1.pdf); O-SCD++ treats change as a persistent property of a 3DGS map, >10 FPS online (https://arxiv.science/abs/2610.24462).
- **Not found:** a published system that stores an explicit "last observed" timestamp per Gaussian and renders staleness. It is a simple attribute to add (update on every frame where the Gaussian is in a camera frustum and passes a depth test) — design work, not research.

**Static vs transient separation**
- Self-supervised: STORM emergent masks; DeSiRe-GS extracts 2D motion masks from 3DGS residuals (https://github.com/chengweialan/DeSiRe-GS); DGGT dynamic head.
- Prior-based: FreeForward4D uses Grounding DINO + SAM 2 tracks lifted to 3D (https://arxiv.org/html/2611.24007v1); PointForward needs 3D boxes.
- For a car-mounted system, combine (a) class prior (vehicle/pedestrian masks from a segmenter), (b) ego-motion-compensated flow residual, (c) multi-view depth disagreement; only promote to the static memory after repeated consistent observations.

**Generative fill for never-seen regions**
- **Difix3D+** (CVPR 2025 Oral): single-step SD-Turbo fixer; **76 ms per frame on A100** as a render-time post-process, and used to distill pseudo-views back into the 3D model (https://arxiv.org/html/2503.01774); NVIDIA licence (non-commercial) (https://github.com/nv-tlabs/Difix3D). DGGT ships a Difix-style refiner checkpoint (https://github.com/xiaomi-research/dggt).
- Driving-specific fixers: ConFixGS (confidence-aware diffusion fix for feed-forward driving GS, listed in https://github.com/Awesome3DGS/3D-Gaussian-Splatting-Papers/blob/main/Changelog.md; details unverified), FreeGen (reconstruction–generation co-training, ECCV 2026, https://media.eventhosts.cc/Conferences/ECCV2026/pdfs/3841.pdf).
- Video-diffusion → 3D: Lyra (NVIDIA, built on Cosmos/GEN3C; https://research.nvidia.com/labs/toronto-ai/lyra), HY-World 2.0 WorldStereo (https://arxiv.org/abs/2604.14268).
- **Practicality:** real-time per-frame fixers (~76 ms on A100) are viable as an *optional viewer layer* for mode A on the RTX 5070 at reduced resolution (unverified on 5070/M5). Video-diffusion completion is offline-only (mode B) and risks hallucinating road layout — always flag generated content in the UI.

---

## 5. Using known ego motion (speed, yaw/heading, GPS)

What the odometry gives you, concretely:
1. **Metric scale** for free. Monocular/feed-forward models drift in scale over long sequences (motivation in https://openaccess.thecvf.com/content/CVPR2026/html/Cheng_LongStream_Long-Sequence_Streaming_Autoregressive_Visual_Geometry_CVPR_2026_paper.html and https://arxiv.org/abs/2603.05959); integrated wheel speed fixes it directly, and FreeForward4D shows how much effort is otherwise spent recovering metric scale (MoGe-2 calibration of VGGT-Ω windows, https://arxiv.org/html/2611.24007v1).
2. **A strong pose prior** (planar SE(2) from speed + yaw rate, with GPS as a weak absolute constraint). Methods that *accept* pose priors: **MapAnything** (poses + metric flag), **Pi3X** (pose/intrinsics/depth conditioning), **DA3** (pose-conditioned depth), **WorldMirror/2.0** (any-prior prompting; "prior injection yields universal gains"), **Rig3R** (rig raymaps, timestamps), **DriveVGGT** (calibration constraints), **ReCoSplat** (posed mode), **StreamSplat** (calibrated views), **STORM/UniSplat/EVolSplat** (require poses) — sources in §1–§3.
3. **Rig rigidity.** Extrinsics are constant across the drive, so estimate them once (offline, mode B) and reuse; DriveVGGT and Rig3R are built around this (https://arxiv.org/abs/2511.22264, https://arxiv.org/abs/2506.02265).
4. **Factor-graph fusion.** VGGT-SLAM 2.0 (GTSAM), F²SLAM and AMB3R-SLAM have explicit back ends where odometry and GPS factors can be added (https://gtsam.org/2026/06/24/vggt-slam.html, https://arxiv.org/abs/2610.05207, https://arxiv.org/abs/2609.19518).

**Not found:** a 2026 paper quantifying the gain from *wheel odometry + heading* specifically as input to a VGGT-class model. Expectation (my inference, not a cited result): odometry fixes scale and gross drift; vision still needs to supply pitch/roll, lateral slip and fine alignment.

---

## 6. Comparison table

Fit = fit for mode A on this project (1 poor – 5 excellent). NC = non-commercial.

| Method | Year | Inputs | Outputs | Dynamic handling | Rig / fisheye | Speed | VRAM | Code / weights | Licence | Apple Silicon | Fit |
|---|---|---|---|---|---|---|---|---|---|---|---|
| VGGT-1B | 2025 | images | cams, depth, points, tracks | weak | no / no | 34 FPS @504×336 A100 (DA3 paper) | ~400–500 imgs on 80 GB | github.com/facebookresearch/vggt | NC; Commercial variant (AUP) | MPS port | 3 |
| VGGT-Ω | 2026 | images | cams, depth (single head), registers | yes (trained on dynamic) | no / no | ~24 ms/frame small batches (secondary) | 9.7 GB@50, 13.4 GB@100 frames | github.com/facebookresearch/vggt-omega | FAIR NC, gated | MLX port | 3 |
| Pi3 / Pi3X | 2025–26 | images (+pose, K, depth on Pi3X) | poses, local points, approx metric | trained w/ dynamic (per VGGT-Ω paper) | no / no | unverified | unverified | github.com/yyfz/Pi3 | code BSD-3, weights NC | not found | 3 |
| **MapAnything** | 2025–26 | images + any of K / **ray dirs** / depth / **poses** / metric flag | metric points, depth, rays, K, poses, conf | none explicit | via ray-dir input (unverified on fisheye) | fast; 2000 views on 140 GB mem-efficient | scales w/ views | github.com/facebookresearch/map-anything | **Apache variant** + NC variant | **MLX (both)** | **5** |
| **Depth Anything 3** (+Streaming) | 2025 | images (+poses) | depth, rays, poses, K, metric (nested), **3DGS (Giant)**, sky | trained w/ dynamic | no / no | Base 126 FPS, Giant 38 FPS (A100) | Streaming <12 GB | github.com/ByteDance-Seed/Depth-Anything-3 | Small/Base Apache; Large/Giant NC | **MPS + MLX-Swift incl. Streaming** | **4–5** |
| WorldMirror 2.0 | 2026 | images + poses/K/depth | points, depth, normals, cams, 3DGS | none explicit | no / no | unverified | unverified | github.com/Tencent-Hunyuan/HY-World-2.0 | Tencent community (excl. EU/UK/KR) | not found | 3 |
| CUT3R (+TTT3R, Mem3R) | 2025–26 | image stream | online metric pointmaps, poses | some | no / no | 16.6 FPS KITTI A100 | unverified | github.com/CUT3R/CUT3R | CC BY-NC-SA | not found | 2 |
| StreamVGGT / InfiniteVGGT / OVGGT | 2025–26 | image stream | cams, depth, points | some | no / no | real-time class | OVGGT constant VRAM | see §1.6 | NC / Apache (InfiniteVGGT) | not found | 3 |
| LongStream | 2026 | image stream | metric poses + geometry, km scale | unverified | no / no | 18 FPS | unverified | not found | unverified | not found | 3 (if code appears) |
| VGGT-Long / Pi-Long | 2025–26 | images | chunked poses + points + loops | no | no / no | offline-ish | bounded by chunk | github.com/DengKaiCQ/VGGT-Long | VGGT licence | not found | 3 (mode B more) |
| DVGT / DVGT-2 | 2025–26 | multi-view seq, no calib | metric ego-frame points, ego poses; DVGT-2 streaming | unverified | **any rig config** / fisheye unverified | unverified | 1.7–1.8 B params | github.com/wzzheng/DVGT | Apache (DVGT-1 weights); DVGT-2 weights untagged | not found | 4 |
| DriveVGGT | 2025 | rig video + calib | poses, depth, metric | no | **yes** / no | −49% vs VGGT | unverified | not found | — | — | 2 (no code) |
| Rig3R | 2025 | images + optional rig metadata | poses, points, rig discovery | no | **yes** / no | unverified | unverified | unofficial reimpl. only | — | — | 2 |
| Any3R / Camera3R / MEOW | 2026 | any central camera | poses, points | no | **fisheye yes** | unverified | unverified | Camera3R: github.com/android-xr/camera3r | unverified | not found | 3 (fisheye enabler) |
| AnySplat | 2025 | unposed images | 3DGS + poses | no | no / no | seconds | unverified | github.com/OpenRobotLab/AnySplat | **MIT** | not found | 2 |
| DrivingForward | 2025 | surround imgs | 3DGS | none | nuScenes rig / no | real-time claim | unverified | github.com/fangzhou2000/DrivingForward | MIT | not found | 2 |
| Omni-Scene | 2025 | surround imgs | 3DGS | none | nuScenes / no | unverified | unverified | github.com/WU-CVGL/Omni-Scene | MIT | not found | 2 |
| STORM | 2025 | posed multi-cam clip | 3DGS + velocities | **scene flow / velocity** | Waymo-like / no | 200 ms per 2-s clip | unverified | github.com/NVlabs/GaussianSTORM (no official ckpt) | NVIDIA NC | not found | 3 |
| **UniSplat** | 2025–26 | rig images + foundation priors | dynamic-aware 3DGS + **static memory bank** | dynamic score per Gaussian | Waymo / no | unverified | unverified | github.com/chenshi3/UniSplat | **code Apache**, Waymo weights | not found | **4** |
| DGGT | 2025–26 | unposed images | 3DGS + cams + dynamics + **lifespan** + diffusion refine | dynamic head, lifespan | cross-dataset / no | ~0.4 s | unverified | github.com/xiaomi-research/dggt | code Apache, **weights NC** | not found | 4 |
| FreeForward4D | 2026 | RGB only | metric 4D Gaussians, cams | object velocity + local corrections | Waymo→nuScenes/AV2 / no | 0.21 s + 13 ms render (4090) | unverified | "will be released" | — | — | 4 (recipe) |
| ReconDrive / PointForward V2 | 2026 | posed rig (+boxes for PF) | 4DGS | velocity / instances | nuScenes/Waymo | unverified | unverified | not found | — | — | 2 |
| On-the-fly NVS | 2025 | ordered pinhole frames | 3DGS + poses, live | no | no / **no fisheye** | live | CUDA GPU | github.com/graphdeco-inria/on-the-fly-nvs | Inria NC | CUDA only | 3 |
| **On-the-fly multi-camera rig** | 2025 | rig video, no calib | 3DGS + rig trajectory | no (avoids movers) | **yes** / no | 100 m in ~2 min | unverified | not found | — | — | 4 (if code) |
| HIBS | 2026 | online frames | beta splats with LOD, stale-primitive recycling | no | no | real-time render | unverified | unverified | — | — | 3 (ideas) |
| ReCoSplat | 2026 | posed/unposed stream | online 3DGS | no | no / no | 45 FPS input (RTX 6000 Ada) | KV cache −90% | released (per paper) | unverified | not found | 4 |
| StreamSplat (2026) | 2026 | calibrated stream | online 3DGS, voxel cache | no | no / no | unverified | bounded by geometry | "upon acceptance" | — | — | 3 |
| VGGT-SLAM 2.0 | 2026 | RGB stream | poses + dense submaps, loops | no | no / no | real-time (Jetson Thor) | unverified | github.com/MIT-SPARK/VGGT-SLAM | BSD-2 (VGGT weights NC) | not found | 3 |
| AMB3R-SLAM | 2026 | mono (+LiDAR) | km-scale poses + map | handles dynamics | no / no | real-time, 10k frames consumer GPU | unverified | code linked (project page) | unverified | not found | 3 |
| Difix3D+ | 2025 | renders + refs | fixed renders | — | — | 76 ms/frame A100 | unverified | github.com/nv-tlabs/Difix3D | NVIDIA NC | not found | 3 (viewer add-on) |

---

## 7. Recommended shortlist for the online pipeline (mode A)

**Architecture assumption:** Gaussians live in your own persistent, voxel-hashed world map (per-Gaussian: position, covariance, colour/SH, opacity, `first_seen`, `last_seen`, `n_obs`, `confidence`, `dynamic_prob`). Each new multi-camera time-step window is turned into metric geometry by a foundation model using odometry + rig priors; new Gaussians are spawned only where the map is empty or disagrees (on-the-fly-rig/HIBS/StreamSplat-style voxel dedup); a few optimisation iterations run on the active region; Gaussians not in any current frustum keep their `last_seen` and are rendered with a staleness cue. Movers are kept in a short-lived transient layer and never promoted into memory.

### 1. MapAnything (apache weights) as the per-window geometry engine — **top pick**
- **Why:** the only strong model that takes *exactly* your side information in one call: intrinsics **or per-pixel ray directions** (fisheye without undistortion), **camera poses** (odometry × rig extrinsics), and a **metric-scale flag**; outputs metric depth, poses and confidence (https://github.com/facebookresearch/map-anything). Apache-2.0 weights exist (https://huggingface.co/facebook/map-anything-apache) and **MLX ports** exist for the Mac Studio (https://huggingface.co/mlx-community/map-anything-apache-fp16). Same repo benchmarks VGGT-Ω/DA3/Pi3X, so you can swap engines.
- **Risks:** fisheye ray maps may be out of distribution (test the back camera first, fall back to virtual-pinhole crops); not causal, so you run it on sliding windows (e.g. 6 cams × 4–8 timesteps) and must align windows yourself; the Apache model is weaker than the NC one (README says the NC one is "best performance"); no dynamics handling — you add masks.

### 2. Depth Anything 3 (Streaming + pose-conditioned; GS head as a research option) — **best on the Mac today**
- **Why:** fastest family (DA3-Base 126 FPS, Small 160 FPS on A100, https://arxiv.org/html/2511.10647); **pose-conditioned depth** mode for your odometry; nested metric models; **DA3-Streaming under 12 GB** fits the RTX 5070 (https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/da3_streaming/README.md); "-1.1" checkpoints much better on street scenes (README); **mlx-swift port including streaming** for an eventual native Metal app (https://github.com/mnmly/mlx-swift-da3). Its GS head gives a ready-made way to seed Gaussian attributes.
- **Risks:** best checkpoints (Giant, Nested, GS head) are **CC BY-NC**; only Small/Base are Apache; pinhole assumption (no ray-map input found); the MLX-Swift port is a one-person, brand-new project.

### 3. UniSplat (with DGGT as an alternative) as the driving-specific reference for dynamics + memory
- **Why:** the only published feed-forward driving splatting model with a **persistent memory of static Gaussians for regions outside current coverage** plus per-Gaussian dynamic scores — essentially the "seen N seconds ago" behaviour (https://arxiv.org/abs/2511.04595). Apache-2.0 code with training code released (https://github.com/chenshi3/UniSplat). DGGT adds **pose-free** operation and a **lifespan head**, with cross-dataset zero-shot results (https://arxiv.org/abs/2512.03004) but NC weights. FreeForward4D (code pending) shows how to self-train such a model on *unlabelled* RGB video, i.e. on your own Tesla footage (https://arxiv.org/abs/2611.24007).
- **Risks:** trained on Waymo (pinhole, 5-camera front-biased rig, Waymo licence terms); needs calibrated poses (UniSplat); CUDA-only custom rasterizers (`diff-gaussian-rasterization-feature`, `pointops2`) → runs on the RTX 5070, not on Metal; 12 GB VRAM may limit window size (unverified). Treat as a design reference / fine-tuning base, not a drop-in.

**Honourable mentions:** ReCoSplat (online feed-forward GS at 45 FPS, posed mode — good if its licence permits; https://arxiv.org/abs/2603.09968); On-the-fly multi-camera-rig 3DGS (exactly the right optimisation-online design, but no code found; https://arxiv.org/abs/2512.08498); DVGT/DVGT-2 (Apache, metric, rig-agnostic, streaming; https://github.com/wzzheng/DVGT); VGGT-SLAM 2.0 as the pose back end with GTSAM odometry factors (https://github.com/MIT-SPARK/VGGT-SLAM).

---

## 8. Open questions

1. Do MapAnything's `ray_directions` inputs produce sane metric depth on the Tesla back fisheye, or do we need virtual-pinhole undistortion / Camera3R-style LoRA adaptation? (Needs a quick experiment on real footage.)
2. Licence intent: is this strictly personal research? If any commercial path is possible, the usable set shrinks to MapAnything-apache, DA3-Small/Base, VGGT-1B-Commercial (AUP), AnySplat, DVGT-1, UniSplat code (weights trained on Waymo still need checking).
3. ReCoSplat's code/weight licence and backbone (π³ weights are NC) — not verified.
4. Code availability for: On-the-fly multi-camera-rig 3DGS, LongStream, FreeForward4D, HIBS, StreamSplat (2026), DriveVGGT. Worth a watch list.
5. Does gsplat (or any CUDA-free differentiable rasterizer) run fast enough on Metal for online refinement? Known Metal-capable options: OpenSplat (https://github.com/pierotofy/OpenSplat) and Brush (WebGPU/Burn, trains on macOS, https://github.com/ArthurBrussee/brush); gsplat MPS status not verified.
6. Real throughput of MapAnything/DA3 on the M5 Ultra via MLX and on the RTX 5070 (12 GB) for a 6-camera × N-timestep window at the native Tesla resolutions — no published numbers for either device.
7. How good is the Tesla per-frame metadata (speed, heading, lat/lon) in time sync and noise? Determines whether it is a hard pose prior or a weak factor.
8. How to estimate the six cameras' extrinsics once (offline, mode B) so mode A can treat the rig as fixed — Rig3R-style discovery vs. calibration from a static-scene drive.
9. Is a diffusion fixer (Difix-class, ~76 ms/frame on A100) acceptable in the live viewer, and how should generated content be marked?
