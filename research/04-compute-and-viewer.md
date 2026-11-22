# 04 · Compute platform and viewer/runtime (state as of 2026-11-21)

Scope: where each stage of this project (then called Reality Splat) should run (Mac Studio M5 Ultra, RTX 5070 box, rented GPUs), and which viewer/runtime to build on. The reconstruction algorithms themselves are covered elsewhere.

Research method: web search plus direct reads of GitHub READMEs, changelogs, PyPI metadata, vendor pricing pages and docs, done on 2026-11-21. Every claim has a link. "Unverified" means I could not find a primary source. Numbers marked **(estimate)** are my own arithmetic, not published figures.

---

## 0. TL;DR recommendation

1. **The Mac Studio can do almost the whole pipeline.** That covers SfM (COLMAP 4.2 with the built-in global mapper, rig support and ONNX/CoreML learned features), feed-forward depth and pose models (PyTorch 2.15 MPS, plus MLX/Core AI ports where they exist), splat training (Brush or msplat, both native Metal/wgpu), and serving the viewer.
2. **Build the trainer on Brush (Rust).** It is the only Apple-capable trainer that has all of these: it is a Rust library (with a C FFI and a JS/wasm wrapper), MCMC-style densification with a splat cap, masks, KB4/fisheye camera models (merged May 2026), LOD baking, Rerun logging and active development. Its gaps are that depth supervision is an open PR, a fisheye focal-length bug is open, and there is no documented API for adding cameras mid-training. **msplat** is the speed fallback on Metal: it has the fastest published Mac numbers and a step-wise Python/Swift API, but it looks pinhole-only (unverified).
3. **What genuinely needs CUDA:** gsplat (no official Metal backend; its 3DGUT path is the best-supported way to train with real fisheye + rolling shutter + external/windshield distortion, plus PPISP, batching and LiDAR), driving-specific dynamic-scene frameworks such as drivestudio/OmniRe, and most research repos with custom CUDA kernels. Use the **RTX 5070** for small segments (12 GB is tight: gsplat's own 30k-iteration *bicycle* run peaks at about 11.6 GB). **Rent an L40S/A100/H100** for the full 3.3 km offline run (about $1–4/h; a 4–8 h job costs roughly $5–35).
4. **Viewer: Spark 2.x (World Labs, three.js, WebGL2), written in TypeScript.** It is the only web renderer that combines streamed LOD (.RAD, HTTP range requests, a paged GPU splat pool), splat objects added or removed at any time, per-splat programmable shader graphs ("dyno") with editable vertex and fragment shaders, correct global sorting across many objects, three.js mesh compositing (car model, frustums) and iPhone support (a 1–3 M splat budget). Plan B is the **PlayCanvas engine**, which has a WebGPU compute renderer, streamed SOG LOD, per-splat vertex shader hooks and procedural `GSplatContainer`. Watch **Spark Radix**, the experimental WebGPU version (1.8× faster at 10 M splats, but no LOD yet). Use **Rerun** (≥0.36 shows Gaussian splats on a timeline) as the debugging viewer.
5. **Formats:** keep PLY internally. Ship **.RAD** (Spark LOD) or **Streamed SOG** for the offline map, use SPZ v4 for compact interchange, and use **glTF `KHR_gaussian_splatting`** (now ratified) plus Cesium 3D Tiles if you want GPS-anchored, geospatial tiling. There is no standard for 4D or streamed-update splats, so the live mode needs a custom websocket delta protocol.

---

## 1. Hardware baseline

| Machine | GPU compute | Memory | Notes |
|---|---|---|---|
| anomastudio: M5 Ultra, 64-core GPU, 96 GB | Full 80-core M5 Ultra is estimated at ~82 FP16 TFLOPS ([jdhodges](https://www.jdhodges.com/blog/apple-cpu-compared-m1-m3-m3-m4-m5-max/)) and 33 TFLOPS FP32, vendor claimed ([flopper](https://flopper.io/gpu/apple-m5-ultra)). The 64-core bin should be ~80% of that **(estimate)**. Each GPU core has a Neural Accelerator for matmul ([Apple ML research](https://machinelearning.apple.com/research/mlx-distributed-inference)) | 96 GB unified. The binned 30-CPU/64-GPU variant starts at 96 GB ([jdhodges](https://www.jdhodges.com/blog/apple-cpu-compared-m1-m3-m3-m4-m5-max/)). The full chip has 1.2 TB/s; the bandwidth of the 64-core bin is unverified | Apple announced it on 2026-08-25 ([contextstudios](https://www.contextstudios.ai/blog/mac-studio-m5-ultra-local-ai-guide)). Metal 4 adds `MTLTensor` and ML passes ([Apple docs](https://developer.apple.com/documentation/metal/machine-learning-passes)); the WWDC26 Metal Tensor API / MPP Tensor Ops session targets the M5 Neural Accelerators ([WWDC26-330](https://developer.apple.com/videos/play/wwdc2026/330/)) |
| exon-desk: RTX 5070 | 6,144 CUDA cores at 2.51 GHz ([TechPowerUp](https://www.techpowerup.com/gpu-specs/nvidia-geforce-rtx-5070-founders-edition.b12306)), so ~30.9 TFLOPS FP32 **(estimate: 6144×2×2.512 GHz)** | 12 GB GDDR7, 672 GB/s ([TechSpot](https://www.techspot.com/specs/gpu/305060-nvidia-geforce-rtx-5070.html)); 128 GB system RAM | Blackwell sm_120 needs CUDA ≥12.8 ([NVIDIA forum](https://forums.developer.nvidia.com/t/software-migration-guide-for-nvidia-blackwell-rtx-gpus-a-guide-to-cuda-12-8-pytorch-tensorrt-and-llama-cpp/321330)) |
| Modal / RunPod | see §6 | 24–141 GB | per-second billing |

Raw FP32 is in the same ballpark on both boxes. The Mac's advantages are 8× the memory and no VRAM ceiling ([Laan Labs](https://labs.laan.com/blogs/gaussian-splat-training-on-mac-with-metal) on scene size being bounded by unified memory). CUDA's advantage is software maturity.

Data scale for this project: Tesla saves six videos per clip: front, rear, left/right repeater and left/right pillar ([Tesla owner's manual](https://www.tesla.com/ownersmanual/models/en_pr/GUID-F311BBCA-2532-4D04-B88C-DBA784ADEE21.html)). At 36 fps, 3.3 km at ~40 km/h is ~300 s, or about 65 k frames across six cameras **(estimate, speed assumed)**. Training will need keyframe subsampling regardless of platform.

---

## 2. Gaussian-splat training on Apple Silicon

### 2.1 Comparison

| Trainer | Stack | Maturity / latest | Speed evidence | Camera models | Depth sup. | Masks | Appearance / exposure | Densification | Library / incremental use |
|---|---|---|---|---|---|---|---|---|---|
| **Brush** ([repo](https://github.com/ArthurBrussee/brush)) | Rust, Burn, wgpu; kernels ported to CubeCL `#[cube]` ([CHANGELOG](https://raw.githubusercontent.com/ArthurBrussee/brush/main/CHANGELOG.md)) | 5.3k★, pushed 2026-11-21. Last tagged release is v0.3.0 (2025-09-14); a large "Unreleased" changelog sits on main ([releases](https://github.com/ArthurBrussee/brush/releases)) | v0.2 "trains as fast as gsplat" ([0.2 notes](https://github.com/ArthurBrussee/brush/releases)). A Nov 2026 paper says Brush takes ~3× the training time of a Vulkan trainer at comparable quality on a workstation ([Cross3DGS, arXiv 2611.22240](https://arxiv.org/abs/2611.22240)). In-browser Brush 0.3 reaches 28.63 PSNR at 30k on Mip-NeRF360 ([SpawnScene bench](https://github.com/LostBeard/SpawnScene/blob/master/Docs/benchmarks.md)). No published M5 numbers | Pinhole, plus **KB4 and RT8 fisheye models merged 2026-05-20** ([PR #434](https://github.com/ArthurBrussee/brush/pull/434)). Open bug: KB4 trains at the wrong focal length when the distortion polynomial folds ([#564](https://github.com/ArthurBrussee/brush/issues/564), opened 2026-11-17). 3DGUT not implemented ([#124](https://github.com/ArthurBrussee/brush/issues/124)) | **Open PR**, "gsplat-like depth loss" ([#497](https://github.com/ArthurBrussee/brush/pull/497)) | Yes: alpha and mask folders, several layouts ([CHANGELOG](https://raw.githubusercontent.com/ArthurBrussee/brush/main/CHANGELOG.md)) | Not found | MCMC-like with growth and a max-splat cap ([v0.3.0](https://github.com/ArthurBrussee/brush/releases)); Mip-Splatting 2D filter; post-training LOD baking ([CHANGELOG](https://raw.githubusercontent.com/ArthurBrussee/brush/main/CHANGELOG.md)) | `brush-c` C FFI and `brush-js` wasm, with zero-copy access to WebGPU splat buffers ([CHANGELOG](https://raw.githubusercontent.com/ArthurBrussee/brush/main/CHANGELOG.md)). Datasets bigger than RAM ([v0.3.0](https://github.com/ArthurBrussee/brush/releases)). `units_per_meter` for metric scale ([#545](https://github.com/ArthurBrussee/brush/pull/545)). Rerun logging ([README](https://github.com/ArthurBrussee/brush)). Adding cameras mid-run: **unverified**, would need a fork of the training loop |
| **msplat** ([repo](https://github.com/rayanht/msplat)) | C++ + fused Metal kernels, no PyTorch; Python (nanobind), Swift and CLI | v1.1.4 (2026-08-25) on PyPI, macOS 14+ arm64 wheels ([PyPI](https://pypi.org/pypi/msplat/json)) | M4 Max, Mip-NeRF360 at 7k iterations: 59–80 s per scene versus ~335 s for gsplat on a TITAN RTX (different hardware). Garden at 30k: 700 s, 27.14 PSNR versus gsplat's 27.32 ([README](https://raw.githubusercontent.com/rayanht/msplat/main/README.md)) | Formats: COLMAP, Nerfstudio, Polycam. No fisheye found in the repo (search returned 0 hits), so **probably pinhole-only (unverified)** | Not found | Not found | Not found | Classic split/clone/cull, GPU-resident ([README](https://raw.githubusercontent.com/rayanht/msplat/main/README.md)) | `GaussianTrainer.step()` in Swift, checkpoint save/resume, `render_from_pose` ([README](https://raw.githubusercontent.com/rayanht/msplat/main/README.md)). The dataset is fixed at construction, so incremental training is unverified |
| **splatx-metal** ([repo](https://github.com/PoSTMEDIA-AI/splatx-metal)) | C++/Metal + Swift app; a "numerical reproduction" of gsplat checked against golden tensors | New (macOS 26 / iOS 26) | M4 24 GB: garden 7k MCMC in 14.9 min; 30k in 65.7 min ([README](https://raw.githubusercontent.com/PoSTMEDIA-AI/splatx-metal/main/README.md)) | Not stated | Not found | Not found | Not found | MCMC | CLI / Swift |
| **metalsplat** (tchauffi) | PyTorch MPS + MSL kernels compiled through `torch.mps.compile_shader` | v0.3.0 (Sep 2026) fixed opacity-reset and filter bugs ([radiancefields](https://radiancefields.com/metalsplat-v0.3.0-trains-3dgs-and-2dgs-on-apple-silicon)) | No published speed ([platform page](https://radiancefields.com/platforms/metalsplat-pytorch)) | Pinhole via pycolmap (unverified beyond that) | Not found | Not found | Not found | AbsGS-style, SparseAdam, Mip 3D filter, 2DGS | Python API: `GaussianModel`, `Camera`, `render`. The easiest to write a custom incremental loop around, but probably the slowest |
| **OpenSplat** ([repo](https://github.com/WebODM/OpenSplat)) | C++/libtorch, with CUDA, ROCm, Metal and CPU paths | Moved to the WebODM org ([repo](https://github.com/WebODM/OpenSplat)) | Metal support since 2024 ([reddit](https://www.reddit.com/r/GaussianSplatting/comments/1c4buxi/opensplat_now_supports_training_gaussian_splats)) | Fisheye request still **open** ([#213](https://github.com/pierotofy/OpenSplat/issues/213)) | No | — | — | Classic | CLI |
| gsplat-mlx / MLX trainers | MLX | Research quality | gsplat-mlx: 273 ms fwd+bwd at 64×64 with 500 Gaussians ([repo](https://github.com/RobotFlow-Labs/gsplat-mlx)), far too slow. A native C++ Metal MLX build reported 38.5 it/s ([Ghif Lab](https://ghif.github.io/posts/2026-02-19-production-grade-3d-gaussian-splatting-apple-silicon)) | — | — | — | — | — | Not recommended |
| **gsplat official** | CUDA only | v1.6.0 on main, not on PyPI (PyPI is still 1.5.3) ([README](https://raw.githubusercontent.com/nerfstudio-project/gsplat/main/README.md), [PyPI](https://pypi.org/pypi/gsplat/json)) | — | — | — | — | — | — | An MPS PR was closed unmerged ([#832](https://github.com/nerfstudio-project/gsplat/pull/832)) and the "MPS Support?" issue is still open ([#163](https://github.com/nerfstudio-project/gsplat/issues/163)). The old AGPL `gsplat-mps` fork is based on gsplat 0.1.3 ([repo](https://github.com/iffyloop/gsplat-mps)) |
| **nerfstudio** | PyTorch + gsplat | Last PyPI release 1.1.5 (2024-11-11) ([PyPI](https://pypi.org/pypi/nerfstudio/json)) | — | — | — | — | — | — | `splatfacto` fails on MPS ([#3290](https://github.com/nerfstudio-project/nerfstudio/issues/3290)). Effectively CUDA-only on Mac |
| 3D Splat App (Laan Labs) | Closed-source app, C++/Metal port of "a modern CUDA splat trainer" | v0.3, Sep 2026 ([blog](https://labs.laan.com/blogs/gaussian-splat-training-on-mac-with-metal)) | 2 / 5 / 10–20 min presets | 360° video is converted to pinhole views | — | Yes | — | — | An app, not a library. Useful as a sanity baseline |

### 2.2 Apple's own work
- **SHARP** (Dec 2025) is single-image → metric 3DGS in under 1 s "on a standard GPU" ([Apple ML](https://machinelearning.apple.com/research/sharp-monocular-view), [repo](https://github.com/apple/ml-sharp)). It could fill gaps or give quick per-frame previews. It does not replace multi-view fusion.
- **RealityKit `GaussianSplatComponent`** (iOS/iPadOS/macOS/visionOS 27) renders splats natively from USD or PLY ([sample](https://developer.apple.com/documentation/visionos/rendering-gaussian-splats-with-realitykit), [API](https://developer.apple.com/documentation/realitykit/gaussiansplatcomponent), [WWDC26 RealityKit](https://developer.apple.com/videos/play/wwdc2026/279)). It has **no developer-visible shaders**, so stale regions can't be tinted per splat, and there is an internal total-splat limit ([API docs](https://developer.apple.com/documentation/realitykit/gaussiansplatcomponent)); a Reddit user reports a ~200 k cap, unverified ([reddit](https://www.reddit.com/r/GaussianSplatting/comments/1u0omkk/apple_now_natively_supports_gaussian_splats_in)). Buffers are referenced rather than copied, so you can update the `LowLevelBuffer`s to animate ([GaussianSplatResource](https://developer.apple.com/documentation/realitykit/gaussiansplatresource)). It is fine for a later native iPad/Vision Pro app, but not for the main viewer.
- No Apple splat *trainer* API was found.

### 2.3 Assessment
- For a Rust developer, Brush is the natural base: its crates, CubeCL kernels and FFI make an online trainer that "adds keyframes and splats over time" a matter of forking the training loop rather than writing kernels. CubeCL has a native Metal runtime besides wgpu ([CubeCL](https://github.com/tracel-ai/cubecl)). There is a reported high kernel-launch overhead for CubeCL through wgpu on Metal ([cubecl#1215](https://github.com/tracel-ai/cubecl/issues/1215)), which matters for small incremental steps. Which runtime Brush uses on macOS is unverified.
- Fisheye: Tesla's front wide camera and possibly the pillar and repeater cameras are wide-angle. Brush's KB4 support is new and has an open bug. The safe path is to undistort to pinhole for the Mac trainers and keep native fisheye or rolling shutter for gsplat 3DGUT on CUDA (§4).

---

## 3. Feed-forward models on Apple Silicon

### 3.1 PyTorch MPS in late 2026
- **PyTorch 2.15.0** was released 2026-10-28 with macOS arm64 wheels ([PyPI](https://pypi.org/pypi/torch/json)). The 2.15 MPS changes add FP8 (`float8_e4m3fn`, `scaled_mm`), native Metal GEMV, grouped matmul, small-matrix inverse kernels (useful for camera geometry), faster conv/padding/interpolation backward, much lower video VAE memory and public MPS streams. MPS now **requires macOS 15+** ([HF blog, 2026-10-20](https://huggingface.co/blog/mps-pytorch-2-15)). The 2.16 preview has attention running entirely on Metal (same source).
- The same blog admits that "at the start of 2026, running even small models on MPS often meant working around unsupported operations, performance cliffs and unexpected memory growth" ([HF blog](https://huggingface.co/blog/mps-pytorch-2-15)). PyTorch 2.13 and 2.14 moved many ops from MPSGraph to native Metal ([2.14 release blog](https://pytorch.org/blog/pytorch-2-14-release-blog)).
- There is no float64 or complex128 on MPS ([MPS docs](https://docs.pytorch.org/docs/main/notes/mps.html)). Unsupported ops fall back to CPU with a warning when fallback is enabled ([MPSFallback.mm](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/mps/MPSFallback.mm?plain=1)).
- **Core AI** is Apple's successor to Core ML in iOS/macOS 27. PyTorch models are exported with `coreai-torch` to `.aimodel` and run on the GPU or ANE ([HF model card](https://huggingface.co/mlboydaisuke/Depth-Anything-3-CoreAI)). COLMAP added CoreML as an ONNX execution provider (§4).

### 3.2 Per-model status

| Model | Apple Silicon path | Precision notes | Published Mac throughput |
|---|---|---|---|
| VGGT-1B | Upstream README selects `cuda` or CPU only ([README](https://github.com/facebookresearch/vggt)). Community `vggt-mps` adds sparse covisibility attention ([repo](https://github.com/jmanhype/vggt-mps)); another trimmed port exists ([repo](https://github.com/YichongSUN/VGGT_MAC_Apple-Silicon)) | vggt-mps runs **float32 only**: "MPS does not support float16 autocast for this model" ([README](https://raw.githubusercontent.com/jmanhype/vggt-mps/main/README.md)) | Not found |
| π³ (Pi3) | Upstream picks dtype by CUDA capability only ([README](https://github.com/yyfz/Pi3)) | — | Not found |
| MapAnything | Upstream is PyTorch with bf16 AMP (falls back to fp16). Memory-efficient mode handles up to 2000 views on 140 GB ([README](https://github.com/facebookresearch/map-anything)) | bf16 on MPS untested by me | Not found. With 96 GB unified memory, roughly 1000+ views could fit **(estimate)** |
| Depth Anything 3 | `da3-engine` provides PyTorch-MPS and **MLX** backends with identical outputs ([repo](https://github.com/tg1482/da3-engine)). A Core AI conversion of DA3-Small/Base exists ([HF](https://huggingface.co/mlboydaisuke/Depth-Anything-3-CoreAI)). DA3-Streaming runs long videos in <12 GB ([DA3 README](https://github.com/ByteDance-Seed/Depth-Anything-3)) | fp16 | **M2 Pro 16 GB**, one SLAM step (new frame + 5 keyframes, 504×378): SMALL 166 ms, BASE 371 ms, LARGE 1076 ms; ray-pose RANSAC adds ~550 ms on CPU ([da3-engine](https://raw.githubusercontent.com/tg1482/da3-engine/main/README.md)). An M5 Ultra should be several times faster **(estimate, not measured)** |
| SAM 2 / 2.1 | Runs on MPS (community setups) ([blog](https://ice-ice-bear.github.io/posts/2026-04-08-sam2-mac/)); MLX quantized ports exist ([X post](https://x.com/neural_avb/status/2056734166717538382)) | Upstream examples autocast bf16 on CUDA ([README](https://github.com/facebookresearch/sam2)) | Not found |
| SAM 3 | MLX port (image) at `mlx-community/sam3-image` ([HF](https://huggingface.co/mlx-community/sam3-image)) | — | Not found |
| CoTracker3 | README: "A GPU is strongly recommended" ([README](https://github.com/facebookresearch/co-tracker)); MPS status unverified | — | Not found |
| MASt3R / MASt3R-SfM | No MPS mention in the README ([README](https://github.com/naver/mast3r/blob/mast3r_sfm/README.md)); unverified | — | Not found |

Takeaway: for anything without a port, expect to patch `device="cuda"` assumptions and autocast blocks, run fp32 or bf16, and hit occasional CPU fallbacks. **No published M-series Ultra throughput numbers were found for VGGT, π³, MapAnything, MASt3R, SAM 3 or CoTracker.** Benchmark them yourself early.

---

## 4. COLMAP / GLOMAP on macOS

- **COLMAP 4.2.1** (2026-09-29) is current, with a Homebrew arm64 bottle ([changelog](https://colmap.github.io/changelog.html), [brew](https://formulae.brew.sh/api/formula/colmap.json)) and `pycolmap` 4.2.1 macOS arm64 wheels ([PyPI](https://pypi.org/pypi/pycolmap/json)).
- **GLOMAP has been merged** into COLMAP 4.0 (2026-03-14) as `global_mapper` / `automatic_reconstructor --mapper GLOBAL`. The standalone repo is archived ([changelog](https://colmap.github.io/changelog.html), [motionmaster3d](https://www.motionmaster3d.com/colmap-glomap-openmvs-blender)). 4.2.0 added multi-component global mapping.
- Learned features are built in: **ALIKED + LightGlue** (4.0), **LoMa** (4.2) via ONNX, and **CoreML as an ONNX execution provider on macOS** with CPU fallback ([changelog](https://colmap.github.io/changelog.html)). Running hloc separately is optional. Hloc's own performance on MPS is unverified.
- Features relevant to driving: rig support (rig configurator; the old rig bundle adjuster has been removed), sequential pairing for rigs that avoids unrelated cross-sensor pairs, FISHEYE and DIVISION camera models, and tangent-Sampson error that "particularly improves wide-field-of-view" cameras ([changelog](https://colmap.github.io/changelog.html)).
- **SIFT on GPU:** upstream SiftGPU is CUDA or OpenGL. The fork `byplay-io/colmap-metal` adds Metal SIFT: 4.3 s versus 26.6 s on CPU for 53 images at 1920×1536, with 154 MB versus 11.6 GB of RAM ([repo](https://github.com/byplay-io/colmap-metal)). Laan Labs also moved COLMAP extraction and matching to Metal inside their app ([blog](https://labs.laan.com/blogs/gaussian-splat-training-on-mac-with-metal)). Bundle adjustment on Mac uses CPU Ceres (the GPU BA solver is CUDA) ([changelog](https://colmap.github.io/changelog.html)).
- Practical note: with known rig extrinsics plus GPS/heading priors, the global mapper on a 36-core CPU should be fine. No timing for this data size was found.

---

## 5. Using the RTX 5070 box

- **Blackwell support:** PyTorch builds with CUDA 12.8+ support sm_120 ([PyTorch forum](https://discuss.pytorch.org/t/nvidia-geforce-rtx-5070-ti-with-cuda-capability-sm-120/221509)). Old stacks fail: nerfstudio's documented cu118 / PyTorch 2.1 path does not recognise sm_120 on a 5070 ([nerfstudio#3732](https://github.com/nerfstudio-project/nerfstudio/issues/3732)). Nerfstudio 1.1.5 pins gsplat 1.4.0, which predates sm_120 ([reddit](https://www.reddit.com/r/GaussianSplatting/comments/1r1zvvk/what_are_my_options_with_a_rtx_5060_graphics_card/)).
- **gsplat main (1.6.0)** requires PyTorch 2.7+, has "improved CUDA 12.8 and 13.2 build compatibility" and CUDA 13 support, but is **not on PyPI**, so build it from source ([README](https://raw.githubusercontent.com/nerfstudio-project/gsplat/main/README.md)). Windows prebuilt wheels exist on `docs.gsplat.studio/whl`, but the docs give old pt/cu examples ([INSTALL_WIN](https://github.com/nerfstudio-project/gsplat/blob/main/docs/INSTALL_WIN.md)). Whether there are sm_120 wheels is unverified, so assume a source build with VS Build Tools.
- gsplat 2026 additions that matter here ([README](https://raw.githubusercontent.com/nerfstudio-project/gsplat/main/README.md)):
  - 3DGUT external distortion ("windshield-style rigs")
  - per-ray inputs
  - fisheye, FTheta and LiDAR sensor models
  - PPISP exposure/ISP compensation working with MCMC
  - sparse rasterization
  - multi-GPU
  - native MCMC noise injection
  - ~30% faster 3DGUT+MCMC
  - HiGS inference renderer

  3DGUT gives distorted cameras and rolling shutter ([NVIDIA blog](https://developer.nvidia.com/blog/revolutionizing-neural-reconstruction-and-rendering-in-gsplat-with-3dgut/)).
- **Windows versus WSL2:** gsplat documents a native Windows build. Community guides recommend Ubuntu/WSL2 for RTX 50 setups ([graphdeco#1313](https://github.com/graphdeco-inria/gaussian-splatting/issues/1313)). WSL2 GPU passthrough is claimed at 90–100% of native ([insiderllm](https://insiderllm.com/guides/wsl2-local-ai-windows-guide/)); file I/O across the Windows/Linux boundary is the usual bottleneck ([itctshop](https://itctshop.com/wsl2-performance-for-ai/)). Recommendation: use WSL2 Ubuntu, with data stored on the Linux filesystem.
- **What fits in 12 GB:** gsplat's benchmark shows 30k-iteration *bicycle* at 11.59 GB and *garden* at 11.06 GB peak training memory ([gsplat benchmark](https://rocm.docs.amd.com/projects/gsplat/en/latest/reference/benchmark-evaluation.html)). A 3.3 km street scene won't fit in one go. Use segments or chunks with an MCMC cap, or rent. drivestudio/OmniRe VRAM needs: not found ([repo](https://github.com/ziyc/drivestudio)).

---

## 6. Rented GPUs (prices read 2026-11-21)

| GPU (VRAM) | Modal (per-second rate → $/h) ([pricing](https://modal.com/pricing)) | RunPod Pods ([pricing](https://www.runpod.io/pricing)) |
|---|---|---|
| L4 (24 GB) | $0.000222/s → **$0.80/h** | $0.69/h |
| A10 (24 GB) | $0.000306/s → $1.10/h | — |
| RTX 4090 (24 GB) | — | $0.89/h |
| RTX 5090 (32 GB) | — | $1.39/h |
| L40S (48 GB) | $0.000542/s → **$1.95/h** (matches [gpueconomy](https://gpueconomy.com/providers/modal/l40s)) | **$1.09/h** |
| A100 40 GB / 80 GB | $2.10/h / $2.50/h | A100 PCIe 80 GB $2.09/h, SXM $2.19/h |
| H100 (80 GB) | $0.001097/s → **$3.95/h** | PCIe $2.89/h, SXM $4.39/h |
| RTX PRO 6000 (96 GB) | $3.03/h | $2.69/h |

The RunPod numbers are the default tab rendered on the pricing page. Whether that tab is Community or Secure Cloud is unverified. Serverless rates are higher (for example L40S at $1.89/h) ([pricing](https://www.runpod.io/pricing)).

**Rough job cost (estimate):** an offline 3.3 km reconstruction split into ~10 chunks of ~30–60 min each on an L40S comes to 5–10 GPU-hours, about **$5–11 on RunPod or $10–20 on Modal**. On an H100 it would be ~$15–40. Storage and egress are extra (unverified). Modal's per-second billing and scale-to-zero suit bursty jobs ([budgetforge](https://www.budgetforge.dev/tools/modal-pricing-2026)).

---

## 7. Web viewers and runtimes

### 7.1 Capability matrix

| Runtime | Tech | Millions of splats + LOD / chunk streaming | Live add / remove / update | Per-splat time or custom shaders | Mesh + frustum compositing | iPhone/iPad Safari | Formats |
|---|---|---|---|---|---|---|---|
| **Spark 2.x** (World Labs, MIT; current 2.4.0 per [radiancefields](https://radiancefields.com/spark-radix-brings-webgpu-splat-rendering-to-threejs)) | three.js, **WebGL2** | **Yes:** LoD splat tree, fixed per-frame budget N (500 K–2.5 M by device), composite LoD across objects, **.RAD** streaming via HTTP range, shared LRU page table (16 M-splat pool), "100 M or 1 G+ splats" on mobile ([2.0 features](https://sparkjs.dev/docs/new-features-2.0/), [World Labs blog](https://worldlabs.ai/blog/spark-2.0)) | **Yes:** "Splat objects can be added and removed dynamically at any time". `PackedSplats.setSplat/pushSplat`, GPU `generate()`, on-demand LoD regeneration ([2.0 features](https://sparkjs.dev/docs/new-features-2.0/), [PackedSplats](https://sparkjs.dev/docs/packed-splats/)) | **Yes:** dyno shader graphs, `objectModifiers` / `worldModifiers` per SplatMesh ([SplatMesh](https://sparkjs.dev/docs/splat-mesh/), [dyno](https://sparkjs.dev/docs/dyno-overview/)), editable vertex and fragment shaders in 2.0 ([2.0 features](https://sparkjs.dev/docs/new-features-2.0/)), SDF edits for colour or opacity ([editing](https://sparkjs.dev/docs/splat-editing/)). There is no built-in per-splat timestamp attribute (unverified), so encode it per chunk or in extra data | **Yes:** fused with the three.js pipeline ([README](https://github.com/sparkjsdev/spark)) | Yes, WebGL2. iPhone budget 1–3 M splats ([perf](https://sparkjs.dev/docs/performance/)) | PLY, SPZ, SPLAT, KSPLAT, SOG, RAD ([PackedSplats](https://sparkjs.dev/docs/packed-splats/)) |
| **Spark Radix 0.0.1** (experimental) | three.js r186 **WebGPURenderer**, with WebGL2 fallback | No LoD selection yet ([radiancefields](https://radiancefields.com/spark-radix-brings-webgpu-splat-rendering-to-threejs), [docs](https://sparkjs.dev/radix/docs/index.html)) | GPU arrays and collections | Projections `ewa`, **`3dgut`**, `htgs`, `raygs`; TSL shaders ([docs](https://sparkjs.dev/radix/docs/index.html)) | three.js | WebGPU is on by default in Safari 26 ([metalmancy](https://metalmancy.tech/guides/turning-on-webgpu-in-your-browser/)) | PLY, SPZ, RAD, RDX |
| **PlayCanvas engine** (2.19+, API v2.23.3) / SuperSplat | WebGPU compute renderer + WebGL2 fallback | **Yes:** Streamed SOG LODs with a device-tuned Gaussian budget. M4 Max WebGPU runs 35 M splats at 76 fps; iPhone 13 Pro Max runs 4 M at 42 fps ([blog](https://blog.playcanvas.com/new-in-supersplat-webgpu-and-streaming-bring-huge-performance-wins/)) | `GSplatContainer` creates splat data programmatically ([docs](https://developer.playcanvas.com/user-manual/gaussian-splatting/building/procedural-splats/), [API](https://api.playcanvas.com/engine/classes/GSplatContainer.html)). Live streaming updates unverified | **Yes:** `gsplatModifyVS` runs per splat to "move, rotate, scale, hide or tint" ([docs](https://developer.playcanvas.com/user-manual/gaussian-splatting/building/custom-shaders/vertex/)); fragment hooks and varying streams; fisheye rendering page ([docs nav](https://developer.playcanvas.com/user-manual/gaussian-splatting/building/custom-shaders/vertex/)) | Yes (full engine) | Yes | PLY, SOG, Streamed SOG, glTF splats ([reddit](https://www.reddit.com/r/GaussianSplatting/comments/1umbwnn/playcanvas_engine_now_loads_gaussian_splats_from/)) |
| Babylon.js 9.0 (2026-03-26) | WebGL/WebGPU | Multiple splats combined; LOD unverified | Unverified | Node Material integration (since v8), shadows ([radiancefields](https://radiancefields.com/babylon.js-v9.0-3dgs-gets-shadows-sogs-and-triangle-splatting-support-announced), [babylonjs.com](https://www.babylonjs.com/)) | Yes | Yes | PLY, splat, SPZ, SOG/SOGS |
| Viser 1.1.1 | Python server driving a WebGL client | No LOD found | `add_gaussian_splats`; dynamic buffer updates flicker ([#615](https://github.com/viser-project/viser/issues/615)) | Limited | Yes: frustums, meshes, GUI ([docs](https://viser.studio/main/examples/scene/gaussian_splats/)) | Browser | numpy arrays |
| **Rerun** ≥0.36 (current 0.43.2) | Rust viewer, native + web | No LOD found | Log per time step | **Time-indexed by design**: `GaussianSplats3D` archetype, PLY drop-in, anisotropic scale, opacity, SH ([0.36](https://rerun.io/docs/changelog/changeset-0-36), [archetype](https://rerun.io/docs/reference/types/archetypes/gaussian_splats3d), [PyPI](https://pypi.org/pypi/rerun-sdk/json)) | Yes: Pinhole, Mesh3D, transforms | Web viewer; iOS performance unverified | PLY, SDK |
| Cesium / 3D Tiles | CesiumJS, Unreal, ion | **Yes:** hierarchical LOD over 3D Tiles, glTF payload, SPZ compression ([Cesium blog, 2026-04-27](https://cesium.com/blog/2026/04/27/3d-gaussian-splats-lod/)) | Tile-level only | Limited | Globe context, GPS-anchored | Yes | glTF KHR splats + SPZ |
| Brush web / `brush-js` | wasm + WebGPU | URL streaming | Zero-copy buffer access from JS ([CHANGELOG](https://raw.githubusercontent.com/ArthurBrussee/brush/main/CHANGELOG.md)) | "PLY with delta frames" playback for 4D ([v0.2](https://github.com/ArthurBrussee/brush/releases)) | Limited | WebGPU | PLY, compressed PLY |
| Visionary | WebGPU + ONNX Runtime | — | Per-frame ONNX Gaussian generators (4DGS, avatars) ([repo](https://github.com/Visionary-Laboratory/visionary)) | Yes, via ONNX | Meshes | WebGPU | ONNX |
| MetalSplatter 1.0.x (native) | Swift/Metal: iOS, macOS, visionOS | — | Library | SH0–3 ([radiancefields](https://radiancefields.com/metalsplatter-releases-v1.0)) | App code | Native, not a browser | PLY, SPZ, .splat ([README](https://raw.githubusercontent.com/scier/MetalSplatter/main/README.md)) |

### 7.2 Fit to Ryan's requirements
- **Live, growing splat over a websocket:** Spark's dynamic SplatMesh/PackedSplats plus add/remove at any time fits this best. Design the server to emit *spatial chunks* (for example 20–50 m tiles keyed to the drive), each a SplatMesh. Replace a chunk when the trainer updates it.
- **"Last seen N seconds ago" fade or a time scrubber:** keep a `lastSeen` (and optional `firstSeen`) per chunk as a uniform fed into a dyno `worldModifier` or the 2.0 custom fragment shader, which multiplies colour or opacity. Per-chunk granularity avoids needing a per-splat attribute. If per-splat timing is needed, pack it into Spark's extra splat data, which is unverified. On PlayCanvas the same idea works with `gsplatModifyVS`.
- **Car mesh and camera frustums:** standard three.js GLTF plus `LineSegments`, composited with splats.
- **iPhone/iPad:** Spark targets WebGL2 with a 1–3 M splat iPhone budget, so use LoD. PlayCanvas's WebGPU path roughly doubles iPhone fps compared with its WebGL2 path ([blog](https://blog.playcanvas.com/new-in-supersplat-webgpu-and-streaming-bring-huge-performance-wins/)).
- **Fisheye-correct display (optional):** Radix's `3dgut` projection could render the original camera views for side-by-side comparison.

---

## 8. File formats and compression

| Format | Status (Nov 2026) | Notes |
|---|---|---|
| PLY (3DGS layout) | De facto interchange | Large; use internally |
| **SPZ** (Niantic) | **Version 4** is the default: ZSTD, independent parallel attribute streams, configurable SH bits, coordinate-system extension ([README](https://raw.githubusercontent.com/nianticlabs/spz/main/README.md)) | Compact interchange; loaders exist in Spark, MetalSplatter and Babylon |
| **SOG v2** (PlayCanvas) | ~15–20× smaller than PLY, lossy, bundled `.sog` zip ([spec](https://developer.playcanvas.com/user-manual/gaussian-splatting/formats/sog-v2/)). **Streamed SOG** adds LOD chunks ([blog](https://blog.playcanvas.com/new-in-supersplat-webgpu-and-streaming-bring-huge-performance-wins/)) | Built by `splat-transform`, which works as CLI or library, in Node or the browser ([repo](https://github.com/playcanvas/splat-transform)) |
| **RAD** (Spark 2.0) | LoD splat tree, HTTP range chunk streaming, coarse-to-fine ([2.0 features](https://sparkjs.dev/docs/new-features-2.0/)) | `bhatt-lod` offline, `tiny-lod` in-browser |
| RDX (Radix) | Chunked, range reads ([docs](https://sparkjs.dev/radix/docs/index.html)) | Experimental |
| KSplat | Legacy (mkkellogg); Spark still loads it ([PackedSplats](https://sparkjs.dev/docs/packed-splats/)) | Avoid for new work |
| **glTF `KHR_gaussian_splatting`** | **"Complete, Ratified by the Khronos Group"** ([spec README](https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_gaussian_splatting/README.md)); RC was 2026-02-03 ([Khronos](https://www.khronos.org/news/press/gltf-gaussian-splatting-press-release)) | Compression is handled by companion extensions (SPZ). Cesium deprecated the older `KHR_spz_gaussian_splats_compression` draft ([cesium#12837](https://github.com/CesiumGS/cesium/issues/12837)) |
| 3D Tiles + glTF splats | Shipping in Cesium ([blog](https://cesium.com/blog/2026/04/27/3d-gaussian-splats-lod/)) | Good for the GPS-anchored offline map |
| Apple USD / PLY | RealityKit loads both ([sample](https://developer.apple.com/documentation/visionos/rendering-gaussian-splats-with-realitykit)) | Native Apple only |
| 4D / streamed updates | **No standard found.** Brush uses "PLY with delta frames" ([v0.2](https://github.com/ArthurBrussee/brush/releases)); Visionary uses ONNX generators; commercial 4DGS streaming exists ([radiancefields 4DGS](https://radiancefields.com/4d-gaussian-splatting)) | Define your own websocket chunk/delta protocol |

---

## 9. Concrete recommendation

### 9.1 What runs where
| Stage | Where | Why |
|---|---|---|
| Decode 6×36 fps video, keyframe selection, masks (sky/ego-car/dynamic objects via SAM 2/3 MLX or MPS) | Mac (VideoToolbox, MPS/MLX) | Local, fast enough; MLX ports exist |
| Poses: GPS/heading priors + COLMAP 4.2 rig + `global_mapper`, ALIKED/LightGlue via ONNX+CoreML; optionally feed-forward (MapAnything/VGGT/π³/DA3) | Mac | All available on macOS; big memory helps feed-forward models with many views |
| Online splat (grows as playback runs) | Mac: a **forked Brush** trainer in Rust that streams chunk deltas to the viewer | Native, Rust, has a library API; you own the incremental loop |
| Offline high-accuracy 3.3 km splat | Mac Brush/msplat at 96 GB for a first pass. **CUDA gsplat 3DGUT+MCMC+PPISP** (RTX 5070 per chunk, or a rented L40S/A100/H100 for the full run) for the best fisheye, rolling-shutter and exposure handling | gsplat has no Metal backend |
| Driving-specific dynamic-scene methods (drivestudio/OmniRe and similar) | CUDA only (rented, or the 5070 for small scenes) | Custom CUDA kernels |
| Viewer server + static LOD assets | Mac (Rust server; websocket + HTTP range for .RAD), shared over Tailscale | |
| Viewer client | Browser: **Spark 2.x + three.js in TypeScript** | LOD, dynamic objects, shaders, iOS |
| Debugging | Rerun (time-indexed splats, frustums, depth) | Brush already logs to Rerun |

### 9.2 What genuinely needs CUDA
- gsplat in any form, including 3DGUT distorted and rolling-shutter cameras, PPISP, external distortion and batching.
- nerfstudio splatfacto.
- drivestudio-style multi-object dynamic driving reconstructions.
- Research repos built on `diff-gaussian-rasterization` or tiny-cuda-nn.
- Feed-forward models only when you need their fast bf16/FlashAttention paths at scale. They run on MPS with patching, at fp32 if needed.

### 9.3 Viewer build sketch
1. Use the three.js scene with a `SparkRenderer`. Give each spatial chunk a `SplatMesh` (ExtSplats for float32 centres over 3.3 km, to avoid float16 striping ([2.0 features](https://sparkjs.dev/docs/new-features-2.0/))).
2. On the websocket, the server sends `{chunkId, version, lastSeen, splatBytes}`. The client swaps that chunk's PackedSplats, adding or removing meshes.
3. A dyno `worldModifier` reads a per-chunk `age = now - lastSeen` uniform and desaturates or tints it; a time scrubber sets `now`.
4. For the offline map, pre-bake `.RAD` with `bhatt-lod` and stream it by range.
5. Overlays: car GLTF mesh, camera frustums and trajectory as three.js objects.

---

## 10. Risks
- **Brush maturity for this use:** depth loss isn't merged ([#497](https://github.com/ArthurBrussee/brush/pull/497)); the fisheye focal bug is open ([#564](https://github.com/ArthurBrussee/brush/issues/564)); no release has been tagged since 2025-09; incremental training isn't a documented feature; a third-party paper reports it is ~3× slower than a Vulkan trainer ([Cross3DGS](https://arxiv.org/abs/2611.22240)).
- **MPS gaps:** float32-only paths (VGGT on MPS), CPU fallbacks and memory growth are acknowledged by the MPS maintainers themselves ([HF blog](https://huggingface.co/blog/mps-pytorch-2-15)). No Ultra-class throughput numbers have been published for the vision models.
- **12 GB VRAM** is borderline even for standard benchmark scenes ([gsplat bench](https://rocm.docs.amd.com/projects/gsplat/en/latest/reference/benchmark-evaluation.html)), and the Blackwell toolchain is fiddly ([nerfstudio#3732](https://github.com/nerfstudio-project/nerfstudio/issues/3732)).
- **Spark** is WebGL2 with a worker-side sort, so sort lag shows during fast camera motion. The WebGPU successor (Radix) is experimental and lacks LoD ([radiancefields](https://radiancefields.com/spark-radix-brings-webgpu-splat-rendering-to-threejs)).
- **Per-splat time attributes** in Spark are unverified, so plan for per-chunk timestamps.
- **Large-coordinate precision** over 3.3 km: use ExtSplats or per-chunk local origins.
- **Licences:** `gsplat-mps` is AGPL ([repo](https://github.com/iffyloop/gsplat-mps)); Brush is Apache-2.0 ([repo](https://github.com/ArthurBrussee/brush)); Spark is MIT ([radiancefields](https://radiancefields.com/platforms/spark)).

## 11. Open questions
1. Does Brush on macOS use CubeCL's native Metal runtime or wgpu→Metal, and how bad is per-step launch overhead for small incremental updates ([cubecl#1215](https://github.com/tracel-ai/cubecl/issues/1215))?
2. Can Brush's training loop (or `brush-c`) accept new cameras and splats mid-run without a restart? This needs a code read.
3. Real M5 Ultra 64-core numbers: Brush and msplat it/s on a street scene, and VGGT/MapAnything/DA3 latency at Tesla resolutions. None have been published.
4. What are Tesla's camera intrinsics and distortion per camera (fisheye versus pinhole), and does the rolling shutter matter at speed? This decides whether CUDA 3DGUT is essential.
5. Does Spark support a custom per-splat float attribute in a dyno modifier (for a per-splat `lastSeen`), or only per-object uniforms?
6. Are the RunPod prices above Community or Secure Cloud, and what do storage and egress cost on each provider?
7. Does gsplat 1.6 publish sm_120 Windows wheels, or is a source build required?
8. What is the memory bandwidth of the 64-core / 96 GB M5 Ultra bin?
