# CUDA port inventory — fe29f83, 224–254 s on the front SEI clock

**First target:** one secure-cloud A40 (48 GB, advertised $0.59/h, high stock). The $0.34/h community 4090 has 24 GB, about the reported MPS allocation for Difix fp32 before increasing resolution or batch. A40 has room for both precisions at native size; never load all stages at once. Use Runpod's versioned CUDA 12.8.1/Torch 2.9.1 Ubuntu 24.04 development image (runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404). No network volume. Container disk ~60 GB for components; for whole-drive runner generations provision only on a filesystem with verified reflink support or budget separately for physical copies. No CUDA resource requirement in this document is a measurement.

## Production dependencies and minimum changes

| Stage | Mac dependency or installation assumption | Linux path |
|---|---|---|
| Inventory/ingest | Six-file minute groups, PyAV H.264 with original Tesla SEI, frame-counter time origin; runner scans entire raw source. | CPU FFmpeg/PyAV. A stream cut must preserve original SEI/global clock; do not export entire bracketing minutes or unrelated leading GOP frames. |
| Runner | transactions.boot_id uses Darwin sysctl; stages/transactions/growing use BSD cp clone options. lstart/process detection and shared-Studio contention assumptions. RS_INSTALL and symlinks into installed binaries, envs, calibration, chase-camera bundle/cache, viewer host. | Read Linux boot_id, use flock/atomic rename, use cp --reflink=auto -a (never hardlink mutable outputs); size physical fallback. Build exe/envs locally, separate offline scene/cache validation from Studio publication. |
| Calibration, poses, road, photo, groundtex | Native Ceres bundle programs, C++ ground helpers; prebuilt arm64 binaries and installed rig. Groundtex/road/photometric estimation run on CPU. | Build rig_ba and trajectory_ba from pipeline/{calib,poses}/CMakeLists.txt using libceres-dev/Eigen and a release compiler; compile road C++ on host. OpenCV, NumPy, SciPy, Pillow. No direct Apple Accelerate dependency in these CMake targets. |
| Depth and masks | depth/models.py and masks/seg.py select MPS else CPU even on CUDA. rscommon sets fallback/OpenMP env and calls memory_pressure; run_depth + bench_e2e + bakeoff/speed use torch.mps cache, timing or driver statistics. bench_e2e uses BSD ps and expects report/cache directories. | Dispatch CUDA before MPS, keep CPU fallback, use CUDA synchronization and reset/read allocated/reserved peak each case; Linux host memory separately from VRAM. Model time must include window-frame decode, resampling and writes to compare to 2.84 s. |
| Movers | movers/semantics.py compiles classify.swift: VNClassifyImageRequest, CPU-only Apple Vision/Framework. Geometry, association/filtering and skin otherwise CPU. | No portable weights or exact replacement for this independent category check. A Linux classifier needs labelled agreement testing. Cached Mac labels or empty labels would not be a fresh equivalent end-to-end benchmark. |
| Stitch | Two different renderers: pipeline/livemap/raster.cpp is portable CPU C++ with extern C raster, used by genfill/views.py, run/clips.py, several IBR, livemap and stitch evaluation tools as maps/object-v3/raster.dylib. v2eval.build makes a modified wide variant. Actual optimizer uses MetalSplat project, bin/sort and differentiable render, including private raster ops in stitch/v2field and v4fit; v5fit calls v4fit. Device allocations in model/v2field, metrics in fit/v2fit; auxiliary cull, sample, visibility, match, vote and index dylibs use -dynamiclib. | For CPU rendering build c++ -std=c++17 -O3 -fPIC -shared pipeline/livemap/raster.cpp -o raster.so and select extension at loaders/builds. For CUDA training use gsplat CUDA project/raster/backward with explicit view, alpha, colour, filter, covariance and gradient parity. A stand-in iteration is not the full v5 fit. |
| Generated clip | genfill/difix.py defaults MPS/CPU, sync/driver metrics MPS-only; fix_views also reports MPS memory. Uses separate env, offline HF cache and Difix3D/src. Temporal RGB, depth and provenance carried between frames; batches of independent pairs are **not** clip throughput. CPU renderer, rectified reference choice, occlusion/road warps and libx264 encoding remain. | CUDA device and timing, same seed/timestep/weights. Upstream mv_unet assumes consecutive render/reference pairs; its pipeline concatenates views instead and chunks output, so naive batch>1 silently mixes scenes. Validate cardinality and compare each batch result to individual inference before reporting. |
| Publish | runner requires a live viewer and checks HTTP against Studio addresses; Bun-generated chase camera/cache scripts and scene pointer are installation assets. | Validate local files, transfer only verified scene privately and rebuild at the Studio. No external public viewer. |

Source inventory (repeatable): git grep -nI -E 'torch.mps|"mps"|raster.dylib|dynamiclib|swiftc|sysctl|metalsplat|videotoolbox|Accelerate|/opt/homebrew|/bin/cp|memory_pressure' -- pipeline. Includes off-path bakeoffs, perceptual LPIPS, genvideo and ESRGAN diagnostics; do not install these to benchmark the drive.

## Weights and source

Production weights are Hugging Face snapshots. Code and weights have different licences; making runner code public does not make the weights open source or grant hosted commercial use.

| Model | Weights (Studio revision) | Terms / role |
|---|---|---|
| DA3-LARGE-1.1 | depth-anything/DA3-LARGE-1.1 @ 0e109ae307c5982f319a67cf6f9f99ccdc0ec97c | CC BY-NC 4.0; posed front/rear. |
| DA3METRIC-LARGE | depth-anything/DA3METRIC-LARGE @ 4010e39f3634a45bc60553321fb49fb760bd594e | Apache-2.0; conditional short-baseline fallback. |
| MapAnything | facebook/map-anything @ a1d87e9086706fb9974f3be5a3e3a0ca5401c5aa | CC BY-NC 4.0; four side lenses, BF16, no validity mask. |
| Mask2Former | facebook/mask2former-swin-large-cityscapes-panoptic @ 544d76fe93971ee046dacae19b6d4f6ecb5d9088 | MIT model code, Cityscapes non-commercial training data; do not assume unrestricted weights. |
| Difix-ref | nvidia/difix_ref @ d4830559772a5795c9d136302c5b197d6418d3fb | NVIDIA research/non-commercial + SD-Turbo terms; nvidia/difix no-ref is optional, not chase production. |
| VNClassifyImageRequest | Built into Apple's Vision framework | Proprietary OS-delivered, no separate redistributable weights. |

Studio code revisions: Depth-Anything-3 3d835ec1a5802d64a8b8b15f817a1ab54809bfe4; facebookresearch/map-anything 3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9; nv-tlabs/Difix3D c76edc595586e16732c91ddee82f3a6d83a8a9cc; metalsplat 84e8c3a433f71c107e9387d3f72a765436c11f6e. Mac Python 3.12/torch 2.15/torchvision 0.29.2; genfill pins NumPy 1.26, diffusers 0.25.1, transformers 4.38.0, PEFT 0.9.0, hub 0.25.1. Keep genfill isolated from the depth/seg environment. Optional models and alternatives are catalogued in depth/models.py and genfill/REPORT.md; none are substitutes in the production measurements.

## Disk and memory: measured artefacts, not speculative peaks

Studio du -sm, active generation under work/drives/hw4-raw-224-254/1361f01da6e9b73f. Rounded MiB, excludes previous private generations, models and package caches. RSS/VRAM were not logged per stage in the reference manifest. CPU-only stages have no model VRAM requirement, **not** zero host RAM. Do not substitute filesystem size for RSS.

| Stage | Retained working/output disk, MiB | Available peak evidence |
|---|---:|---|
| Raw / JPEG ingest | inventory <1; verified JPEG drive 2,134 | CPU single decoder |
| Calibration / pose / road | 39 / 125 / 9 | CPU; no RSS receipt |
| Keyframes / preview masks | <1 / 15 | Swin-L 5.2 GB MPS at 724×469 |
| Photo / ground / atlas | photo 8; ground scene 135 stitched, 265 live; shared run intermediates 418 | CPU; no RSS receipt |
| Movers / publish | preview mover scratch 12; attached scene 697 | CPU; Vision CPU; no RSS receipt |
| Production depth/masks | 1,497 / 37 | DA3 6.1, MapAnything 9.1 GB MPS for six 518px images; rear production uses nine views, not six |
| Removal / teachers | static masks 18; mover scratch 22; first chunk teachers 332 | CPU; no RSS receipt |
| Fit / attach / merge | two chunks 5,122 + 1,937, incl targets and teachers; full scene tree 1,786, object assets 396 | Earlier 336px band fit 5.56–7.97 GB RSS, 2.19 GB MPS; v5 peak unknown |
| Chase / cache | 446 for the old **4 s** clip incl aux & repaired frames | Difix ~14 GB MPS fp16/~24 fp32 at reported settings; native CUDA and batches unknown |

Measured historical Studio preview 425.61 s; preview + full 2152.39 s for 30 driving seconds, but full produced only **121 clip frames (233–237 s)**, not 30 s at 30 fps. Do not treat this historical full time as a Studio baseline for --clip all. The separate depth benchmark was measured at 300 s under contention; compare settings and explicitly label the different window. No c296 images are eligible to upload; select 233–237 s real inputs and independently verify reference shutter times.

## Confirmed after the runs

The Studio mover log records **570 independent Vision crop scores** on this window; the executable is Mach-O arm64. An empty-candidate shortcut cannot apply. Linux/CUDA cannot produce an equivalent fresh preview with device/build plumbing alone. The GPU full runner was not invoked and no CUDA release or viewer cache was published. A separate raw boundary issue: the front stream has IDRs at 223.977 and 224.477 s. Starting the first GOP unchanged leaks an out-of-window picture. A private raw pack must preserve native SEI, pixels for the partial leading GOP, and the original clock. The component pack does not claim raw ingest equivalence.

The Studio reference has 161 tiles, 2,906,939 splats, 36 tracks, one chase MP4, and 771 MiB in the full published release tree (702 MiB preview). Earlier v2 geometry training reported 6.89 GB process RSS / 3.30 GB MPS driver on a different slice; no v5 peak was logged. The chase compositor links libc++/libSystem only. Compiling and loading its CPU C ABI succeeded on both Linux pods. MapAnything's giant encoder uses facebookresearch/dinov2 via Torch Hub (Apache-2.0 for that upstream project); MapAnything weights are still NC. The timed run fetched Torch Hub main without logging its commit. The replay pins the main SHA resolved immediately after teardown: 7764ea0f912e53c92e82eb78a2a1631e92725fc8.

The Studio manifest for this window explicitly invokes `run_depth.py --cams all --model da3-large-1.1` and runs `v4fit` for 1,600 iterations on each of two chunks. The separate requested `bench_e2e.py --configs production` uses DA3 on front/rear and MapAnything on four side cameras. Neither its timings nor the gsplat surrogate represent those historical runner stages.

## Replacement effort (planning estimates only, one engineer)

**Six-camera depth is host-bound.** On L40S, 1.713 of 2.525 s/keyframe was host processing and file I/O; on A40 it was 5.126 of 6.399 s. The first change to try is to prefetch and reuse decoded, rectified camera windows across adjacent keyframes, with a bounded queue overlapping preparation and output writes with inference. Preserve window membership and fisheye calibration; time reader, rectification and writer separately before choosing worker counts. This is a proposal, not a measured speed-up; no depth code was changed.

- **Apple Vision mover corroboration:** 1–2 days to capture the 570 original crop decisions and run a licensed Linux classifier against them; 3–8 further days to calibrate thresholds and verify release times, track counts and false positives on held-out scenes. Exact equivalence to proprietary Vision is not guaranteed; labelled disagreement and model rights are the deciding risks.
- **Differentiable MetalSplat fit:** 3–5 days for a CUDA/gsplat forward adapter and matched camera/filter/alpha/depth outputs, then roughly 1–3 weeks for gradient parity, optimizer trajectories and a representative fitted-scene comparison. Assumes reusable CUDA backward kernels; custom kernels or undocumented sort/tie semantics could extend this. The portable CPU chase compositor is a separate component.

### Portable mover corroboration — 224–254 s Studio replay

`MOVERS_CLASSIFIER=vision|siglip2` (default Vision on macOS, SigLIP 2 elsewhere). The portable backend uses [google/siglip2-base-patch16-224](https://huggingface.co/google/siglip2-base-patch16-224/tree/75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2), model card **license: apache-2.0**, 1.50 GB weights. Uses torch, transformers and Pillow; pre-cache the pinned revision for offline use. Device is CUDA → MPS → CPU, overridable with `MOVERS_CLASSIFIER_DEVICE`. Use `--from movers` when switching backends on a cached drive: backend choice is not a runner cache key.

The mask supplies the class. The classifier corroborates vehicle/person/cycle and emits a release time; a sign or stronger competing label can veto a crop. On all 570 byte-identical JPEGs in `hw4-raw-224-254`, the Vision replay reproduced the saved ranked scores exactly. All candidates were **car**; no truck, bus, person or cycle candidate coverage. Agreement on the unchanged tracker gate was **393/570 (68.9%)**. Native SigLIP sigmoids with these Vision thresholds agreed on only 182/570; the shipped backend compares relative scores across vehicle, person, cycle, sign and background prompts. Its scores are not calibrated Vision probabilities.

| Vision decision | SigLIP accepts | SigLIP rejects | Agreement |
|---|---:|---:|---:|
| Accept | 366 | 1 | 99.7% |
| Reject | 176 | 27 | 13.3% |

[Disagreement sheet](../movers/classifier-disagreements.jpg) (+ = backend that accepted). Visual review of all 177 disagreements, with wider frames for ambiguous targets: 151 favor SigLIP (partial cars, distant vehicles and vans), 2 favor Vision (roadside display; a visible car denied by SigLIP), 24 unresolved. These are crop judgments, not independent vehicle counts. Studio M5 Ultra, low-priority CPU, 570 crops including cached model startup, excluding crop extraction: Vision 19.04 s / **33.4 ms per crop**; SigLIP 28.80 s / **50.5 ms per crop**. No rental; CUDA/MPS inference not measured.

Isolated filter + publisher replay retained all 28 reference car IDs and their full motion-state histories (final states: 4 parked, 3 moving, 21 unknown). SigLIP also published 12 previously uncorroborated IDs, all unknown-motion: **40 tracks, not track-count parity**. Of the shared tracks, 27 had identical sampled trajectories; ID 73 appeared 0.611 s earlier. Linux integration still needs a target-device smoke test; treating this as Vision-equivalent requires a held-out, multi-class gate evaluation.
