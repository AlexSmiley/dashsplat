# Third-party models, assets and libraries

dashsplat's own code is under the Apache License 2.0 ([LICENSE](LICENSE)). That licence covers only the code
written for this project. It grants nothing for the models, weights, assets and libraries below, which keep their
own terms. **Several of the models the default pipeline uses are licensed for non-commercial use only.** Running
the pipeline with them, or using what they produce, is bound by those terms whatever licence the code is under.

No model weights are included in this repository. They are downloaded from Hugging Face at the pinned revisions
below (on Linux by `pipeline/cuda/drive_models.py`, which prints each licence line before downloading; on a Mac by
each stage's own loader or setup script). Licence lines were read from the model cards and upstream repositories on
21 and 22 November 2026 (see [research/06-hugging-face.md](research/06-hugging-face.md) section 3 and
[pipeline/cuda/PORT.md](pipeline/cuda/PORT.md)). Check them again before relying on them: model terms change.

## Models used by the one-command pipeline

| Stage | Model and source | Licence (as stated upstream) | What it means |
|---|---|---|---|
| Panoptic masks (sky, road, vehicles, people) | [facebook/mask2former-swin-large-cityscapes-panoptic](https://huggingface.co/facebook/mask2former-swin-large-cityscapes-panoptic) @ `544d76fe` | Card: "license: other". Mask2Former code: MIT. Trained on Cityscapes, whose terms are non-commercial | Treat the weights as non-commercial |
| Metric depth, posed (front and rear; all six cameras in the runner) | [depth-anything/DA3-LARGE-1.1](https://huggingface.co/depth-anything/DA3-LARGE-1.1) @ `0e109ae3`; code [ByteDance-Seed/Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3) @ `3d835ec1` | The card says Apache-2.0; the official Depth Anything 3 README lists CC BY-NC 4.0 | Conflicting labels: treat as non-commercial until the authors say otherwise |
| Depth fallback for short baselines | [depth-anything/DA3METRIC-LARGE](https://huggingface.co/depth-anything/DA3METRIC-LARGE) @ `4010e39f` | Apache-2.0 | Permissive |
| Depth, four side cameras (depth stage `--model auto`, live map) | [facebook/map-anything](https://huggingface.co/facebook/map-anything) @ `a1d87e90`; code [facebookresearch/map-anything](https://github.com/facebookresearch/map-anything) @ `3d10cf7a` | Weights: CC BY-NC 4.0. Its encoder loads [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) through Torch Hub (Apache-2.0) @ `7764ea0f` | Non-commercial |
| Vehicle check on macOS | Apple Vision `VNClassifyImageRequest` (built into macOS; `pipeline/movers/classify.swift` calls it) | Proprietary, part of the operating system; no separate weights | Runs only on macOS; nothing to redistribute |
| Vehicle check elsewhere | [google/siglip2-base-patch16-224](https://huggingface.co/google/siglip2-base-patch16-224) @ `75de2d55` | Apache-2.0 | Permissive |
| Generated chase frames (one-step fixer) | [nvidia/difix_ref](https://huggingface.co/nvidia/difix_ref) @ `d4830559`; code [nv-tlabs/Difix3D](https://github.com/nv-tlabs/Difix3D) @ `c76edc59` | "Ready for research and development/non-commercial use only", NVIDIA License, for both model and code; built on [stabilityai/sd-turbo](https://huggingface.co/stabilityai/sd-turbo) and subject to the Stability AI Community License | Non-commercial. The generated frames in the reports and any you make come from this model |

Permissive replacements and what each costs in quality are compared in
[research/06-hugging-face.md](research/06-hugging-face.md) section 3.6 (for example DA3-BASE, MapAnything-apache and
NVIDIA Fixer). None of them is wired in as a switch yet.

## Models used only in experiments and benchmarks

These appear in stage reports and scripts but not in `pipeline/run`.

| Where | Model | Licence (as stated upstream) |
|---|---|---|
| Depth bake-off (`pipeline/depth`) | depth-anything/DA3-BASE, DA3-SMALL | Apache-2.0 |
| Depth bake-off | depth-anything/DA3NESTED-GIANT-LARGE-1.1 | CC BY-NC 4.0 |
| Depth bake-off | facebook/map-anything-apache | Apache-2.0 |
| Depth bake-off | Ruicheng/moge-2-vitl-normal, moge-2-vitb-normal | MIT code; weights per model card |
| Depth bake-off | lpiccinelli/unidepth-v2-vitl14 | CC BY-NC 4.0 |
| Fixer survey (`pipeline/genfill`) | nvidia/difix (no reference image) | NVIDIA License, non-commercial |
| Fixer comparison (`pipeline/cuda/ROUND3.md`) | [nvidia/Fixer](https://huggingface.co/nvidia/Fixer) | NVIDIA Open Model License (commercial use allowed); code Apache-2.0 |
| Ground refinement test (`pipeline/genfill/ground`) | Real-ESRGAN x2plus, loaded with spandrel | BSD-3-Clause; spandrel MIT |
| Video diffusion (`pipeline/genvideo`) | [StreetNVS](https://github.com/StreetNVS/StreetNVS) @ `fd002eaf` with Wan2.2-TI2V-5B and umt5-xxl | StreetNVS code Apache-2.0, parts of its preprocessing research/non-profit only, trained on Waymo data (likely non-commercial); Wan2.2 and umt5 Apache-2.0 |
| Video diffusion round 2 | VACE-Wan2.1 1.3B through DiffSynth-Studio | Apache-2.0 |
| Video diffusion round 2, prepared but not run | nvidia-cosmos/cosmos-transfer2.5 | Code Apache-2.0; weights NVIDIA Open Model License |
| Sky fill (`pipeline/envmap`) | kandinsky-community/kandinsky-2-2-decoder-inpaint and kandinsky-2-2-prior | Apache-2.0 |
| Sky fill bake-off | diffusers/stable-diffusion-xl-1.0-inpainting-0.1 | CreativeML OpenRAIL++-M |
| Image metrics | LPIPS (AlexNet) | BSD-2-Clause |

## Assets

| File | What | Licence and attribution |
|---|---|---|
| `viewer/public/models/model3.glb` | The ego car and the body of every saloon in the viewer; point-sampled in the film | **"2024 Tesla Model 3" by RBLXSupercars (the credit embedded in the file; the project's notes also recorded "Tesla model 3 2024" by brandonleong28)**, [Sketchfab](https://sketchfab.com/3d-models/tesla-model-3-2024-36c52f3f89f6439c90310f14e8ff33f2), licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Modified: baked into the car's frame, wheels split into separate nodes, materials renamed, textures resized, meshopt-compressed (`viewer/tools/prepare-car.ts`). CC BY covers the model's copyright only; the car's design and the Tesla and Model 3 names belong to Tesla, Inc. |
| `viewer/public/streets.json` | Street names along the drive for the HUD | Data © OpenStreetMap contributors, from Nominatim, [ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/) |
| Inter Variable (`@fontsource-variable/inter`, installed by `bun install`) | The viewer's typeface | SIL Open Font License 1.1 |
| Berkeley Mono | The film's typeface | Commercial licence from U.S. Graphics Company, held by the author. **Not included.** See below |

### The film's fonts

`film/render` compiles every file in `film/render/media/` into the renderer (`include_media_dir!` in
`film/render/src/lib.rs`) and draws all lettering in Berkeley Mono. That folder is ignored by git because the fonts
are licensed to the author. To build the film without them:

1. Create `film/render/media/` and put the TTF files of any monospaced font you may use there (JetBrains Mono and
   IBM Plex Mono are free under the SIL Open Font License; regular, bold and black weights are used).
2. Change the family name in `film/render/src/draw.rs` (`pub const MONO: &str = "Berkeley Mono";`) and in
   `film/render/src/main.rs` (`default_font: "Berkeley Mono"`) to that font's family name.

Letter widths differ between fonts, so some labels may need new positions. `CLEAN=1` renders the film with no
lettering at all.

## Libraries

The libraries that matter most, with their licences. Exact versions are pinned in `viewer/bun.lock`,
`film/render/Cargo.lock`, `pipeline/cuda/constraints-*.txt` and `pipeline/cuda/setup-drive.sh`.

| Library | Used for | Licence |
|---|---|---|
| [three.js](https://github.com/mrdoob/three.js) | Viewer rendering | MIT |
| [Spark](https://github.com/sparkjsdev/spark) (`@sparkjsdev/spark` 2.4.0) | Gaussian splats in three.js | MIT |
| Vite, TypeScript, Bun | Viewer build, scripts, tile cache, local host | MIT, Apache-2.0, MIT |
| glTF Transform, meshoptimizer, sharp | Preparing the car model | MIT, MIT, Apache-2.0 (sharp ships libvips, LGPL-3.0) |
| [MetalSplat](https://github.com/tchauffi/metalsplat) @ `84e8c3a4` (`msplat` 1.1.4) | Differentiable splat fit on Apple GPUs; its projection is reused by the CUDA fit | MIT |
| PyTorch, torchvision | Models and the splat fit | BSD-3-Clause |
| Hugging Face transformers, diffusers, huggingface_hub, peft, accelerate | Loading and running models | Apache-2.0 |
| NumPy, SciPy, OpenCV, Pillow | Image and geometry work throughout | BSD-3-Clause, BSD-3-Clause, Apache-2.0, MIT-CMU |
| PyAV and FFmpeg | Decoding dashcam video and reading its SEI telemetry; encoding clips | PyAV BSD-3-Clause; FFmpeg LGPL-2.1 or GPL depending on the build (x264 is GPL) |
| [Ceres Solver](http://ceres-solver.org), Eigen | Calibration and trajectory bundle adjustment | BSD-3-Clause, MPL-2.0 |
| [fframes](https://crates.io/crates/fframes) 1.2.0 | The film renderer | MIT. The film enables its `libav-agree-gpl` feature, so a compiled renderer binary carries FFmpeg's GPL terms |

Tesla's dashcam tools (`teslamotors/dashcam`) document the SEI telemetry format. That repository has no licence
file, so none of its code is copied here; `pipeline/ingest/ingest.py` reads the format with its own code.
