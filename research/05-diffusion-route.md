# 05: How Tesla renders Reality View, and how we build the diffusion route

Researched 2026-11-21. Labels used throughout, as in report 01:

- **[official]**: Tesla, Tesla staff or Tesla job postings.
- **[paper]** / **[repo]**: a paper, model card or code repository.
- **[community]**: press, forum or third-party write-up.
- **[observed]**: our own frame analysis of the promo video (paths below).
- **[inference]**: our reasoning or an estimate. Treat as a hypothesis to test.
- **Not found / unverified**: searched and found nothing citable, or not measured.

Frame crops live in `scratch/tesla-method/` (gitignored, so local only). Absolute folder:
`/Users/vogel/dev/experiments/2026-11-21-tesla-reality-splat/scratch/tesla-method/`. Paths below are relative to
that folder. `full/` holds full-resolution 4K frames (`s<time>_f<k>.jpg` = frame k after the given second),
`seq/` holds consecutive-frame runs, `crops/` holds the crops and tiles.

---

## TL;DR

- **Tesla's method: a hybrid.** It is generative, but anchored to a 3D scene. Most likely the driving network
  produces a learned 3D scene state (very probably the "Generative Gaussian Splatting" head Tesla described in 2025–26, or its
  successor). That state is rendered from the virtual camera, and its appearance is regenerated continuously by a learned generator.
  It is **not** a free-running 2D video-diffusion world model, and **not** a static splat map either.
  Confidence about 65%. Feed-forward Gaussians re-predicted every frame with no separate 2D stage: about 20%.
  A pure 2D video/world-model decoder: about 15%. Tommy Rossi and Ryan are both partly right.
- **Top three approaches for us:**
  1. **StreetNVS**, offline. It is a driving-specific video diffusion model built on Wan2.2-5B. It conditions on rendered points
     plus surround reference images plus poses, demonstrates bird's-eye and elevated trajectories, and its weights were released
     this month.
  2. **Cosmos-Transfer2.5-2B**, offline. It makes our splat render photoreal through edge, depth and blur controls and locks the
     layout hardest.
  3. **StreamDiffusionV2 (Wan 1.3B)** or the Difix/Fixer per-frame fixers. These cover near-real-time refinement of the live splat
     render. Only Cosmos3-Nano-Sim-Transfer is truly real-time, and it needs a datacenter Blackwell GPU.
- **First experiment (one day):** StreetNVS on a rented H100 80 GB (about $2.69/h). Input is our existing `pipeline/genfill` chase
  render set for t = 105–120 s: the splat RGB, alpha and z, plus five undistorted reference photos and poses. Output is two 81-frame,
  832×480 clips at 10 fps. Judge them against the splat render and against Tesla's fingerprints (section 3.2).

---

## Part 1. What is Tesla doing?

### 1.1 Material and caveats

- **Source.** `reference/tesla-v15-promo.mp4`: H.264, 3840×2160, 23.976 fps, about 16 Mb/s (ffprobe).
- **What it is.** A graded, edited marketing film of the car's screen, filmed by a separate camera. Every observation below passes
  through four layers:
  1. the screen's own rendering;
  2. the screen's refresh rate versus the camera's 1/48 s-ish exposure, so blended refreshes appear;
  3. lens focus and motion blur;
  4. H.264 compression and the grade.
- **Consequence.** Fine texture differences of one or two pixels are not evidence. Only large, structured changes count.
- **Not measurable.** Nothing here reveals latency or the on-car resolution (section 1.2, O9).
- **How the frames were taken.** The contact sheet `sheet_12-46.jpg` (2 fps, 12–46 s) maps the shots. Then, per shot:
  - 4K frames at offsets 1, 2, 6 and 11 (adjacent, 5 apart, 10 apart);
  - 16–30 consecutive frames for the shots that matter;
  - for the illuminated sign, a crop window that moves about 14 px per frame to follow it (`crops/s15p5_sign_tracked_16frames.jpg`).

### 1.2 Observations

| # | What we see | Where (crops) | What it implies | Confidence |
|---|---|---|---|---|
| O1 | **Pseudo-text that morphs over time.** Grant Ave at night (15.5 s): the lit sign on a parked van reads as crisp, high-contrast glyphs that are not letters. Over 16 consecutive frames (0.67 s) the glyph shapes change progressively ("ᓇ〇·" becomes "ᘏᔑ"-like shapes). A black plate nearby reads "7··99 1" in one frame and "TUR" in the next. | `crops/s15p5_vansign_f01-02-06-11.jpg`, `crops/s15p5_sign_tracked_16frames.jpg`, `crops/s15p5_sign_24frames.jpg`, `crops/s15p5_plate_f01-02-06-11.jpg` | The look of observed surfaces is **decoded by a generative network, not copied from camera pixels**. A splat fitted or predicted from the real photos would show blurry but stable real text. Crisp, wrong glyphs that evolve are the classic signature of a learned decoder working from a low-resolution latent (diffusion, a distilled one-step generator, or a GAN-like decoder). It does not prove the generator is diffusion specifically. | High that it is generative; medium on which kind |
| O2 | **Road and sign symbols morph.** A white road pictogram near the ego car during the side orbit (25.8 s) changes shape over 10 frames, from a swirl to something car-like. The green "P" parking sign (40.6 s) changes its inner glyph between frames 1, 6 and 11. Large road text ("ONLY", read as "NLY" next to the finger) stays legible and in place. | `crops/s25p8_screen_f01.jpg`, `f06`, `f11`; `crops/s40p6_f01-06-11.jpg` | The same as O1, but large, well-observed markings keep their **layout**. Generation is strongly anchored to geometry: small detail is invented, structure is not. | Medium–high |
| O3 | **Content grows in after a fast orbit.** At 25.3 s the virtual camera swings about 70° from the rear chase view to a side view in about 4 frames (about 170 ms). Over the next roughly 20 frames (about 0.8 s), with the camera nearly still, the Golden Gate towers across the bay materialise. They start as faint translucent red lines and become solid towers. | `crops/s25p3_orbit_30frames.jpg` (tile, frames 0–29), `crops/s25p8_screen_f01.jpg`, `f06`, `f11` | The scene is **not a fixed, finished 3D map**. When the view changes, newly visible content is filled in over time. That matches a generator with temporal state (autoregressive or recurrent), or a 3D representation that keeps accumulating and regenerating primitives for the new viewpoint. | Medium–high |
| O4 | **Geometry stays rigid while the camera moves.** During that same 170 ms swing, the road, guard rail, ego car, shadow and hills stay geometrically coherent in every intermediate frame. In the Palace of Fine Arts shot (40.6 s), the overpass, house, sign post and kerb keep rigid 3D relationships over 10 frames of forward motion. | `crops/s25p3_orbit_f06-09.jpg`, `crops/s40p6_f01-06-11.jpg` | There is an **explicit 3D representation**, or a renderer that is strongly 3D-conditioned. A pure 2D video model asked for a 70° jump in 4 frames typically smears or loses structure. The rigid parts behave like rasterised geometry. | Medium–high |
| O5 | **Fingerprints of 3D reprojection and splats in the far field.** At Lombard St from high above (33.2 s), the rooftops the car never saw from above appear as **stretched, torn slabs of texture** with slivers, and the palm trunk is semi-transparent. They are stable across adjacent frames. In the orbit (25.8 s), the distant bridge is a **ghostly, semi-transparent** structure and the far hills carry **white brush-stroke streaks**. | `crops/s33p2_lombard_roofs_f01-02.jpg`, `crops/s33p2_screenwide_f01.jpg`, `crops/s25p8_screen_f01.jpg` | Stretched texture along view rays, translucent ghosts and needle-like streaks are textbook artifacts of **splat or depth-reprojection rendering** far from the source views. A free diffusion model would normally invent clean, plausible roofs instead. So whatever generative stage exists, it does not fully repaint the far field. | Medium |
| O6 | **Unseen surfaces are completed plausibly.** Other cars seen from above have complete, plausible roofs (Grant Ave, the freeway pickup, Lombard). Vehicle shapes are complete from viewpoints the cameras never had. | `crops/s15p5_center_f01.jpg`, `f11`; `crops/s30p2_screenwide_f01.jpg`; `crops/s23_screen_f01.jpg` | A **generative prior over object shape**. Tesla said its generative GS "generat[es] the correct vehicle types and shapes" off the training views ([AINEXT summary of ScaledML 2026](https://ain3xt.com/en/posts/20260210-tesla-end-to-end-foundation-model)). A diffusion model would also do this, so it does not separate the hypotheses. | High that a prior exists |
| O7 | **Fast, close movers ghost.** On the freeway at 52 mph (29.7 s), a car passing at the lower-left edge renders as a translucent bluish blob that dissolves within about 8 frames. Slower cars ahead stay solid. | `crops/s29p7_freeway_16frames.jpg` | Either temporal blending in a generator, or splat floaters for poorly observed fast objects. Filming blur adds to it. | Low |
| O8 | **The ego car, blue path and chevrons are overlays.** The ego Model Y is the same clean CG asset in every shot. It has studio-like specular shading and a hard, dark projected shadow (very large in the orbit). The blue path ribbon runs under it. The turn chevrons float above the road with UI-style glow. | `crops/s25p3_orbit_f06-09.jpg`, `crops/s30p2_screenwide_f01.jpg`, `crops/s40p6_screenwide_f01.jpg`, `crops/s15p5_center_f01.jpg` | **Composited, conventionally rendered 3D assets** over (or fed into) the generated world. This fits Ashok Elluswamy's note that "the arrows show the turns FSD intends to take, the pins show its intent to park" ([X](https://x.com/aelluswamy/status/2124229978560835880)). | High |
| O9 | **Frame rate.** At 22.9 s the filming camera is nearly still for 5 frames: the UI region changes by a mean absolute difference of 1.4–2.4 grey levels per frame, while the world region changes by 8–10 on **every** frame. In the tracked-sign tile, alternate frames look double-exposed, so the screen updates faster than 24 Hz and its content differs between refreshes. Appearance changes roughly every displayed frame, not in 220 ms (4.5 Hz) steps. **Latency** could not be measured: the windshield is out of focus, and shots are edited. | ffmpeg `tblend` difference measurement, run in this session; `crops/s15p5_sign_tracked_16frames.jpg` | The displayed stream runs at about 24 fps or faster, and appearance is regenerated at about display rate. Either the 220 ms GGS figure from 2025 is not what drives the display, or it is interpolated or rendered between updates. | Medium |
| O10 | **Style.** Clouds are flat and painterly, the sky is uniform, distant hills look stylised, and the grade is consistent. | `crops/s23_screen_f01.jpg`, `crops/s25p8_screen_f06.jpg` | A learned decoder at moderate resolution, upscaled. A plain splat render of real photos would carry the camera's actual sky and cloud texture. | Medium |
| O11 | **Not evidence.** The hard horizontal top edge in `reference/ref_34.jpg` (night, Bush St) is a UI wipe into the "12:37 arrival" screen, not the edge of a render. The bright flare at the top right of the 40.6 s shot may be sunlight on the screen glass. | `crops/s27-30_appear_sheet.jpg`, `crops/s40p6_screenwide_f01.jpg` | Ignore both. | n/a |

Embedded, the five crops that carry most of the argument:

![O1: sign glyphs morph across 16 consecutive frames](../scratch/tesla-method/crops/s15p5_sign_tracked_16frames.jpg)
![O3: bridge grows in after a 70-degree orbit](../scratch/tesla-method/crops/s25p3_orbit_30frames.jpg)
![O4: geometry stays coherent mid-orbit](../scratch/tesla-method/crops/s25p3_orbit_f06-09.jpg)
![O5: stretched, torn rooftops at Lombard St](../scratch/tesla-method/crops/s33p2_lombard_roofs_f01-02.jpg)
![O2: P sign changes, road layout and ONLY stay](../scratch/tesla-method/crops/s40p6_f01-06-11.jpg)

### 1.3 What has been said

| Who / when | Statement | Source | Type |
|---|---|---|---|
| Tesla release notes, 2026-11-21 | "Tesla Reality View: New real-time generative driving visualization that continuously streams a view from an arbitrary virtual camera situated above your vehicle…" | [Teslarati, full notes](https://www.teslarati.com/tesla-full-self-driving-v15-rollout-begins-full-release-notes-improvements) | [official] |
| Same notes (engineering list) | "Introduced Tesla Reality View [also printed as 'Tesla Augmented Reality'], replacing all of the explicit model signals into a single end-to-end visualization using less total compute." | [@TeslaSI](https://x.com/TeslaSI/status/2124229673056113100), [Electrek](https://electrek.co/2026/11/21/tesla-fsd-v15-rollout-25x-safety-reality-view/) | [official] |
| Ashok Elluswamy, 2026-11-21 | "The same end-to-end driving model also renders a realistic view of the world around the vehicle… The arrows show the turns FSD intents to take, the pins show its intent to park…" | [X](https://x.com/aelluswamy/status/2124229978560835880), [Drive Tesla Canada](https://driveteslacanada.ca/news/tesla-reality-view/) | [official] |
| Elon Musk, 2026-11-21 | "The new user interface will re-render reality in a mind-bending way." | [Not a Tesla App](https://www.notateslaapp.com/news/5057/tesla-fsd-v15-coming-next-month-with-holiday-update-musk) (via report 01) | [official] |
| Elluswamy, ICCV 2025 | Tesla's Generative Gaussian Splatting "runs in ~220 ms, doesn't require initialization, can model dynamic objects and can be jointly trained with the end-to-end AI model" | [X article](https://x.com/aelluswamy/status/1981644831790379245) | [official] |
| Elluswamy, ScaledML, Jan 2026 | Generative GS "finishes in a few hundred milliseconds", produces "semantically correct 3D scenes when viewed from off-training angles, generating the correct vehicle types and shapes", and "is embedded directly inside the driving neural network". The same talk covers a neural world simulator that generates all cameras. | [AINEXT write-up](https://ain3xt.com/en/posts/20260210-tesla-end-to-end-foundation-model) of [YouTube LFh9GAzHg1c](https://www.youtube.com/watch?v=LFh9GAzHg1c) | [community summary of official talk] |
| Tesla job posting, "SI Engineer, World Modeling & Video Generation" | "Design and train action-conditioned video generation models…", "Integrate 3D generative techniques such as Gaussian Splatting and volumetric rendering for high-fidelity realism", "…real-time inference" | [tesla.com careers 255816](https://www.tesla.com/careers/search/job/si-engineer-world-modeling-and-video-generation-tesla-si-255816) | [official] |
| Tommy D. Rossi (@__morse), reply to Ryan | "from the original video they seem to use a diffusion model instead of gaussian splatting" | From Ryan's brief. **The post was not retrieved** (search found only his unrelated posts) | [community] |
| Industry prior art (not Tesla) | NVIDIA patent application "Depth-based vehicle environment visualization using generative AI": depth maps become a 3D surface topology, which is textured and then treated with generative AI for a surround view | [US 2025/0292498 A1](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/20250292498) | [paper] |
| greentheonly / firmware teardown of v15 | **Not found** (searched 2026-11-21; the rollout began the same day) | n/a | n/a |
| Tesla patent on Reality View or neural rendering for the display | **Not found** | n/a | n/a |
| On-car latency, resolution, frame rate, compute share | **Not found** in any Tesla statement | n/a | n/a |

Nothing from Tesla says "diffusion". Tesla's own vocabulary is "generative", "renders", "arbitrary virtual camera" and "generative
Gaussian splatting". Their world-model hiring explicitly pairs video generation with Gaussian splatting.

### 1.4 Verdict

**A hybrid: a 3D-anchored generative renderer. Confidence about 65%.** The best single story that fits O1–O10 and the statements:

1. The end-to-end driving network keeps a 3D scene state. It is very probably Gaussian-like, from the "generative Gaussian splatting"
   head (O4, O5, ICCV and ScaledML talks, the job posting).
2. That state is rendered from the virtual camera at display rate. This explains the rigid geometry through a 70° swing (O4) and the
   stretched, translucent far field (O5).
3. A learned generator with temporal state produces or refreshes the appearance. This explains the evolving pseudo-text (O1, O2),
   the content growing in after the orbit (O3), the plausible roofs (O6) and the painterly sky (O10). It can be a decoder of Gaussian
   attributes, or an image-space refiner after rasterisation.
4. The ego car, path, chevrons and pins are composited as conventional 3D assets (O8).
5. "Using less total compute" and 24+ fps on the car argue against multi-step video diffusion. A one-step or few-step decoder,
   splat rasterisation, or both, is far more plausible [inference].

Alternatives:

- **About 20%: purely feed-forward generative Gaussians, re-predicted continuously, with no separate 2D stage.** Pseudo-text is
  possible if Gaussian colours are decoded from latents rather than copied from pixels.
- **About 15%: a pure 2D world-model or video decoder conditioned on the virtual camera.** O4 and O5 argue against it.
- **Under 5%: classic per-scene splats or a textured mesh.** O1–O3 rule it out.

**What would change our mind:**

1. **A clean owner screen recording.** Park, orbit to a pose and back, and compare. If identical geometry reappears with different
   fine texture, that supports 3D plus generative appearance. If structure itself differs, that supports a pure 2D model. If a
   Gaussian ellipsoid shows in a paused frame, that supports splats.
2. **A firmware teardown** (greentheonly or similar) showing a rasteriser or splat shader in the UI process, versus only a network
   output buffer.
3. **A Tesla talk or post** saying "diffusion", "world model decoder" or "Gaussian" about Reality View specifically.
4. **Higher-bitrate footage.** It would show whether appearance changes in about 220 ms steps (a Gaussian re-prediction rate) or
   continuously (a per-frame generator).

**What this means for us:** Tesla's look is geometry-anchored generation, plus CG overlays. Our splat track already provides the
anchor. The diffusion track should be the generative appearance layer **on top of our geometry**, not a free-running video model.

---

## Part 2. The diffusion route: state of the art, November 2026

Our inputs:

- calibrated intrinsics and extrinsics, and 6-DoF ego poses;
- per-frame metric depth;
- a coarse splat or point reconstruction;
- six cameras: a narrow front pinhole-ish camera and five fisheyes.

`pipeline/genfill/views.py` already renders **chase render sets**: pinhole RGB, alpha and z from 6 m behind and 3 m above, plus
ranked reference photos, for 85–140 s. `pipeline/genfill/setup.sh` already installs Difix and Difix-ref on the Studio.

### 2.1 Driving-specific generative novel-view synthesis

| Model | Inputs / conditioning | Output | Consistency and layout | Code / weights | Licence | VRAM / speed | M5 Ultra? | RTX 5070? |
|---|---|---|---|---|---|---|---|---|
| **StreetNVS** (Stanford + NVIDIA, arXiv Jun 2026, v2 Aug; weights Nov 2026) | An aggregated point cloud rendered into the **target** view (RGB, validity mask, normalised depth), up to 5 **reference images** from the rig cameras, and **per-frame camera poses** via "Reference-Enhanced Camera Attention". Trained dense-to-sparse, so robust down to 0.1–0.001 point sampling ([paper](https://arxiv.org/html/2606.01590v2)) | 832×480, 49 or 81 frames at 10 fps ([repo](https://github.com/StreetNVS/StreetNVS)) | Demonstrates **elevation to bird's-eye view, lane shift, pull-back and rotation**. Rigid scenes under extreme shifts, where StreetCrafter "drifts" ([site](https://streetnvs.github.io)). Beats FreeVS, GEN3C, StreetCrafter and Wan-VACE-14B on Waymo | [github.com/StreetNVS/StreetNVS](https://github.com/StreetNVS/StreetNVS), [hf zfkuang/StreetNVS](https://huggingface.co/zfkuang/StreetNVS) (Wan2.2-TI2V-5B finetune) | Code Apache-2.0. Preprocessing parts are research and non-profit only (StreetCrafter- and Inria-derived), per [NOTICE](https://github.com/StreetNVS/StreetNVS/blob/main/NOTICE). The weights are trained on Waymo, so the [Waymo Open Dataset terms](https://waymo.com/open/terms/) (non-commercial) very likely apply [inference]. Fine for personal use | "At least 64 GB of GPU memory" for 81 frames, tested on H100 80 GB ([repo](https://github.com/StreetNVS/StreetNVS)). Runtime **not stated**; 20 steps with CFG 5 by default | **Plausible, unverified.** 96 GB unified memory exceeds 64 GB, and Wan2.2-5B already runs on Apple silicon ([mlx-video](https://github.com/Blaizzy/mlx-video)). StreetNVS uses PyTorch and DiffSynth, so it would need MPS. We would feed our own renders, which avoids the CUDA point rasteriser | No (12 GB) |
| **GeoV2V** (SJTU, arXiv Oct 2026) | LiDAR and depth-derived 3D priors plus the **full source video**, on Wan 2.1. Trained with the synthetic multi-lane set Para4D ([paper](https://arxiv.org/html/2610.25245v1)) | not verified | State of the art claimed for lane shifts on Waymo, nuScenes and Para4D | **Code not found** | n/a | n/a | n/a | n/a |
| **SPVC** (arXiv Aug 2026) | Video **fixer** for 3DGS renders, conditioned on the reference video, target pose, BEV/HD map and 3D boxes. Built on Wan-2.2 ([paper](https://arxiv.org/html/2608.17420v1)) | 800×448, 25 frames | "Structured": anchors road topology and object placement while repairing artifacts | Code **not found** | n/a | 305 s per 25-frame sequence on an H20 at 50 steps (253 s optimised) | n/a | n/a |
| **GA-Drive** (CUHK/SenseTime, Feb 2026) | Pseudo-views from decoupled geometry and appearance, segment-wise on CogVideoX I2V, chained by the last frame ([paper](https://arxiv.org/html/2602.20673v2)) | not verified | Free-viewpoint, editable | Code not found | Paper CC BY 4.0 | n/a | n/a | n/a |
| **StreetCrafter** (CVPR 2025) | LiDAR point renders as pixel-level conditions. Built on Vista (SVD-based) ([paper](https://arxiv.org/html/2412.13188)) | Vista/SVD resolution (not re-verified) | Good near the trajectory; drifts at extreme shifts (StreetNVS comparison) | [zju3dv/street_crafter](https://github.com/zju3dv/street_crafter). Weights on Google Drive, front-camera and multi-camera Waymo | Educational, research and non-profit (per StreetNVS NOTICE); Waymo terms | Tested on A100/A800 80 GB ([README](https://github.com/zju3dv/street_crafter)) | Unlikely without porting | No |
| **FreeVS** (ICLR 2025) | "Pseudo-images" of coloured LiDAR projected into the target view. SVD-based ([repo](https://github.com/esdolo/FreeVS)) | SVD resolution | Front camera; good for small shifts | [esdolo/FreeVS](https://github.com/esdolo/FreeVS), weights [Esdolo/FreeVS_WOD](https://huggingface.co/Esdolo/FreeVS_WOD) | Waymo terms (the README tells you to check them) | not stated | Unlikely | Possibly at reduced size [inference] |
| ReconDreamer / DriveDreamer4D / DiST-4D / MagicDrive-V2 / Vista | Mostly *generate* scenes from layouts (boxes and maps), or restore renders for specific scenes | various | Hallucinate scene content from the layout rather than re-render our pixels | [ReconDreamer](https://github.com/GigaAI-research/ReconDreamer) (weights only for some Street Gaussians scenes), [DriveDreamer4D](https://github.com/GigaAI-research/DriveDreamer4D), [DiST-4D](https://github.com/royalmelon0505/dist4d), [MagicDrive-V2](https://github.com/flymin/MagicDrive-V2) (AGPL-3.0) | various | not checked | Not pursued | Not pursued |
| **Cosmos3-Nano-Transfer-Auto / Sim-Auto** (NVIDIA, 2026-11-05) | Per-camera "World Scenario Map" renders (lane markings plus object cuboids), text, an optional first RGB frame. Up to 11 cameras on NVIDIA's Hyperion rig ([model card](https://huggingface.co/nvidia/Cosmos3-Nano-Transfer-Auto)) | Multi-view video, plus LiDAR | Generates *a* scene matching the layout, not *our* scene | [HF Cosmos 3 collection](https://huggingface.co/collections/nvidia/cosmos3) | OpenMDW-1.1, commercial use allowed ([blog](https://huggingface.co/blog/nvidia/cosmos-interactive-world-simulation)) | 16B; Sim-Auto runs 16.7 FPS × 7 views on **16 GB200s** ([Cosmos Lab](https://research.nvidia.com/labs/cosmos-lab/cosmos-simulation/)) | No | No |
| GAIA-class (Wayve) | n/a | n/a | n/a | **Public weights not found** | n/a | n/a | n/a | n/a |

### 2.2 3D-cache and control-conditioned general video models

| Model | Inputs / conditioning | Output | Layout preservation | Code / weights | Licence | VRAM / speed | M5 Ultra? | RTX 5070? |
|---|---|---|---|---|---|---|---|---|
| **Cosmos-Transfer2.5-2B** (NVIDIA, Oct 2025, updated 2026) | Up to 4 control videos (Canny **edge**, **blurred RGB** "vis", **segmentation**, **depth**) plus text ([model card](https://huggingface.co/nvidia/Cosmos-Transfer2.5-2B)) | 720p, 16 fps, 93 frames (5.8 s) | **Strong**. It is a multi-ControlNet: geometry comes from the controls, appearance is regenerated. Feeding our splat render as blur and edge plus our depth should lock lanes and road text | [cosmos-transfer2.5](https://github.com/nvidia-cosmos/cosmos-transfer2.5) | Code Apache-2.0. Weights under the NVIDIA Open Model License per [Classmethod, Aug 2026](https://dev.classmethod.jp/articles/nvidia-cosmos-transfer-2-5-driving-video-weather/); the Cosmos docs now say OpenMDW-1.1 ([docs](https://docs.nvidia.com/cosmos/latest/license.html)). Commercial use allowed either way | **65.4 GB**. **286 s on a B200, 719 s on an H100 NVL** per 93-frame clip ([model matrix](https://docs.nvidia.com/cosmos/latest/transfer2.5/model_matrix.html)). A distilled edge model for low latency followed in Feb 2026 ([repo news](https://github.com/nvidia-cosmos/cosmos-transfer2.5)) | No (CUDA Cosmos stack) [inference] | No |
| **GEN3C** (NVIDIA, CVPR 2025) | Renders of a **3D cache** (point cloud from depth) along the new camera path, plus the seed image(s) ([paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Ren_GEN3C_3D-Informed_World-Consistent_Video_Generation_with_Precise_Camera_Control_CVPR_2025_paper.pdf)) | 121 frames; input 720×1080 recommended ([model card](https://huggingface.co/nvidia/GEN3C-Cosmos-7B)); autoregressive extension in 121·N−1 frames | Strong where the cache has points; generative power goes to unobserved regions. Shown on driving lane changes | [nv-tlabs/GEN3C](https://github.com/nv-tlabs/GEN3C), [GEN3C-Cosmos-7B](https://huggingface.co/nvidia/GEN3C-Cosmos-7B). A driving-finetuned model was "planned"; **release not found** | Code Apache-2.0, weights NVIDIA Open Model License | 7B (Cosmos-Predict1). **VRAM and runtime not found**; likely 80 GB class [inference] | No | No |
| **TrajectoryCrafter** (Tencent, ICCV 2025) | Dual stream: point-cloud render of the new trajectory plus the **source video** (monocular) | not re-verified | Good for re-rendering one camera's video | [repo](https://github.com/TrajectoryCrafter/TrajectoryCrafter) | Custom Tencent licence ([License](https://github.com/TrajectoryCrafter/TrajectoryCrafter/blob/main/License)) | "VRAM ≥ 28GB" ([README](https://github.com/TrajectoryCrafter/TrajectoryCrafter)) | Possible on MPS, unverified | No |
| **ViewCrafter** (TPAMI 2025) | Point-cloud renders plus images | 576×1024 × 25 frames, or 320×512 | Older; short clips | [repo](https://github.com/Drexubery/ViewCrafter) | Apache-2.0 | 23.5 GB and 120 s on an A100 (576×1024); 13.8 GB and 50 s (320×512) ([README](https://github.com/Drexubery/ViewCrafter)) | Possible, unverified | 320×512 just too large for 12 GB [inference] |
| **ReCamMaster** (Kuaishou, ICCV 2025) | **Source video plus a target camera trajectory**, no explicit 3D | not re-verified | Hallucinates more freely, since nothing ties it to our geometry | [KwaiVGI/ReCamMaster](https://github.com/KwaiVGI/ReCamMaster) | MIT (code) | n/a | Possible (Wan-based) | Maybe |
| **Wan2.1-VACE-1.3B / 14B** | Video plus **mask** plus controls (depth, pose and so on): an all-in-one video edit and inpaint model ([ComfyUI docs](https://docs.comfy.org/tutorials/video/wan/vace)) | 480p (1.3B), 480p/720p (14B) ([Wan2.1](https://github.com/wan-video/wan2.1)) | Masked inpainting keeps the unmasked render untouched; it only fills holes. Used as a baseline by StreetNVS | [Wan-AI/Wan2.1-VACE-1.3B](https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B) | Apache-2.0 | The Wan T2V-1.3B base needs **8.19 GB**; 5 s at 480p in about 4 min on a 4090 ([Wan2.1](https://github.com/wan-video/wan2.1)) | Possibly (no MLX VACE port found) | **Yes**, 1.3B at 480p [inference] |
| **Lyra / Lyra 2.0** (NVIDIA) | GEN3C-style camera-controlled video lifted to 3DGS | 3D scenes | Generative 3D world from images | [nv-tlabs/lyra](https://github.com/nv-tlabs/lyra) | Lyra 2.0 weights: **NVIDIA internal scientific research licence**, no distribution or deployment ([HF](https://huggingface.co/nvidia/Lyra-2.0)) | n/a | No | No |
| **Fixer** (NVIDIA, Nov 2025) / **Difix** (v1, Jun 2025) | Per-frame: a novel-view render, plus a reference view for difix_ref | Per image | Repairs artifacts while keeping the render's layout; **per-frame, so temporal flicker** | [nv-tlabs/Fixer](https://github.com/nv-tlabs/Fixer), [nvidia/Fixer](https://huggingface.co/nvidia/Fixer) (0.6B, Cosmos-Predict based), [nvidia/difix](https://huggingface.co/nvidia/difix) | Code Apache-2.0, weights NVIDIA Open Model License ([card](https://huggingface.co/nvidia/Fixer)) | Difix3D+: 76 ms per frame on an A100 ([paper](https://arxiv.org/html/2503.01774), via report 02). Fixer speed not found | **Difix v1 is already set up** in `pipeline/genfill` on the Studio. Fixer needs the CUDA Cosmos container | Difix yes [inference] |

### 2.3 Real-time and streaming diffusion

| System | What it does | Speed (as published) | Geometry-conditioned? | Code / licence | Fit for us |
|---|---|---|---|---|---|
| **Cosmos3-Nano-Sim-Transfer** (NVIDIA, 2026-11-05) | Causal, distilled, **streaming** video from simulator-rendered **depth, segmentation, edge or blurred-RGB** controls plus text, generated in small latent chunks with cached history ([card](https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Transfer)) | **48.1 FPS, 166 ms latency, 480p, one GB300** ([Cosmos Lab](https://research.nvidia.com/labs/cosmos-lab/cosmos-simulation/)) | **Yes**, the closest open analogue of "render to photoreal in real time" | OpenMDW-1.1, open weights | Datacenter Blackwell only. Speed on H100 or B200 **not found** |
| **StreamDiffusionV2** (MLSys 2026 best paper) | Training-free streaming **video-to-video** on Wan2.1-1.3B (causal DMD), 1–4 steps ([paper](https://arxiv.org/html/2511.07399v2), [repo](https://github.com/chenfengxu714/StreamDiffusionV2)) | Read from Fig. 8 of the [MLSys PDF](https://proceedings.mlsys.org/paper_files/paper/2026/file/698cfaf72a208aef2e78bcac55b74328-Paper-Conference.pdf), so approximate: **one RTX 4090, 480p: about 8.6 FPS at 1 step, about 5.6 FPS at 4 steps; 512×512: about 12.9 FPS. One H100, 480p: about 20.5 FPS at 1 step.** Four GPUs reach about 40–64 FPS | Indirectly: it restyles an input video stream, so our splat render can be the input. Noise strength trades layout fidelity against repair | Apache-2.0; "Linux with NVIDIA GPU" | **Best near-real-time option on consumer NVIDIA.** On a 5070 12 GB: plausible at 480p with offloaded text embeddings, maybe 4–8 FPS [inference, unmeasured] |
| **FastFrames** (Apple and Cornell, arXiv 2026-11-17) | Autoregressive few-step video with the VAE decoder replaced by a light pixel decoder; camera-conditioned demo ([paper](https://arxiv.org/html/2611.23226)) | 1.3B: **36 ms per frame (28 FPS) at 480p on one B200**; 14B: 45 ms on 8× B200 | Camera poses (no 3D cache) | Code and weights **not found** | Watch it |
| **ABot-World-0** (2026-07) | Keyboard-action interactive world | **720p, up to 16 FPS, 1.2 s latency, about 19 GiB, one RTX 5090** ([paper](https://arxiv.org/html/2607.19191)) | No | not checked | Shows that a 5090 can do it; not our conditioning |
| **Waypoint-1.5** (Overworld, 2026-09) | 1.28B game world model, keyboard and mouse | "Real time on a wide range of consumer GPUs", 512×1024 / 256×512 ([paper](https://arxiv.org/html/2609.37107v2)) | No | Paper CC BY-NC-ND | Not our conditioning |
| Matrix-Game 3.0, Runway GWM Worlds 2, Helios | Interactive or long video | Matrix-Game 3.0: "720p real-time" ([abstract](https://arxiv.org/html/2604.08995)); GWM Worlds 2: 720p at 24 fps, closed ([Runway](https://runway.com/research/introducing-gwm-worlds-2)); Helios: 19.5 FPS on one H100 ([Wan2.2 README](https://github.com/Wan-Video/Wan2.2)) | No | various | Context only |
| **Apple Silicon** | **FastMetal-QAD** (Hao AI Lab, Aug 2026): three-step DMD, INT8 Wan on MLX ([blog](https://haoailab.com/blogs/fastmetal/)) | On an **M4 Max 36 GB**: 1.3B, 832×480×81 frames in **110 s** (45 s fast mode); 5B, 1280×704×81 in **151 s** (47 s fast) | No (text-to-video and image-to-video) | Apache-2.0 | **No real-time video diffusion on Apple Silicon was found.** An M5 Ultra might be 2–4× an M4 Max, so still only about 1–5 FPS at 480p [inference]. Interactive on the Mac means per-frame image fixers (Difix) at a few FPS, unmeasured |

### 2.4 Rental prices and cost per clip

RunPod on-demand prices, 2026-11-12, per GPU-hour, community / secure
([prompt20 review of RunPod's list](https://blog.prompt20.com/posts/runpod-review)):

| GPU | Community / secure |
|---|---|
| H100 SXM 80 GB | $2.69 / $4.39 |
| H100 NVL 94 GB | $2.59 / $3.19 |
| B200 180 GB | $5.98 / $7.99 |
| A100 80 GB | $1.19–1.39 / $2.09–2.19 |
| RTX 5090 32 GB | $0.69 / $1.39 |
| RTX 4090 24 GB | $0.34 / $0.89 |

The cheapest marketplace H100 is $1.67/h on Vast.ai ([Inference Hub](https://inferencehub.org/blog/h100-rental-prices-2026)).

Cost of generated footage. All runtimes marked [inference] are unmeasured and should be replaced by measurements from experiment E1.
Each line adds about 1 h of setup and download (about $3 on an H100).

| Approach | Unit | Unit time | 60 s of chase view | Whole 6-min drive |
|---|---|---|---|---|
| StreetNVS on H100 SXM ($2.69) | 81 frames at 10 fps = 8.1 s (about 72 new frames with overlap) | **Not published.** Estimate 1–3 min (Wan2.2-5B, about 8k target tokens, 20 steps × CFG) [inference] | 9 clips, 10–30 min, about **$0.5–1.5** | 50 clips, 1–2.5 h, about **$3–7** |
| Cosmos-Transfer2.5-2B on B200 ($5.98) | 93 frames at 16 fps = 5.8 s | 286 s ([NVIDIA](https://docs.nvidia.com/cosmos/latest/transfer2.5/model_matrix.html)) | 11 clips, 52 min, about **$5.2** | 63 clips, 5.0 h, about **$30** |
| Cosmos-Transfer2.5-2B on H100 NVL ($2.59) | same | 719 s (same source) | 11 clips, 2.2 h, about **$5.7** | 63 clips, 12.6 h, about **$33** |
| StreamDiffusionV2 on RTX 4090 ($0.34) | streaming, 480p | about 8.6 FPS at 1 step (Fig. 8) | 1,800 frames at 30 fps, about 3.5 min, **< $0.1** | about 21 min, **about $0.15** |
| Difix per-frame on A100 ($1.39) | 1 frame | 76 ms (Difix3D+ paper) | about 2.3 min, **< $0.1** | about 14 min, **about $0.3** |

Multiply by 2–3 for seeds and retries. Even then, the whole drive through StreetNVS costs about $10–20 in GPU time.

---

## Part 3. Recommendation

### 3.1 Shape of the diffusion track

Treat diffusion as Tesla appears to: **a generative appearance layer on top of our geometry.** Three uses, in order of value:

1. **Offline generative chase video (StreetNVS first, Cosmos-Transfer2.5 second).** Splat-rendered conditioning plus surround
   references produce photoreal chase-view clips along the viewer's default camera path.
2. **Generative densification.** Feed the generated chase frames back as pseudo-views to optimise the stitched splat, in the style
   of Difix3D+ and Lyra. Orbiting near the default path then improves with **no diffusion at runtime**. `pipeline/genfill` already
   records a per-pixel "kept, repaired, fixer-only" class map, so invented content stays labelled "predicted".
3. **Live refinement** of the splat render: per-frame Difix on the Mac now, StreamDiffusionV2 on NVIDIA as a streamed demo.

The ego car, planned path and turn arrows stay **our CG overlays**, as Tesla's are (O8). Generated video must never draw the ego car.

### 3.2 Experiment E1: within one day

**Goal.** A 15 s (t = 105–120 s) photoreal chase-camera clip from StreetNVS, compared side by side with our splat render of the same
camera path.

**Machine.** A rented H100 SXM 80 GB, RunPod community cloud, about $2.69/h. Budget 3–4 h, under $15. The repository asks for 64 GB
or more, and the M5 Ultra cannot run CUDA. In parallel, at no cost: try the same inference on the M5 Ultra with PyTorch MPS. 96 GB is
enough memory, but MPS support in DiffSynth is unverified, so don't block on it.

**Inputs, all from our pipeline.**

1. **Target trajectory.** The viewer's chase camera (6 m behind, 3 m above, viewer pitch, as in `pipeline/genfill/views.py`),
   sampled at **10 Hz in time** rather than per metre of travel: 150 poses for 105–120 s. That gives two 81-frame clips with
   12 frames of overlap (or three 49-frame clips). Use the pinhole K that StreetNVS expects (832×480; pick an FOV matching the
   viewer's).
2. **"LiDAR" render in the target view.** Use our **splat render** RGB, with **alpha as the validity mask** and **z as normalised
   depth**, at 832×480. In the conditioning, **mask out the ego-car footprint** (validity 0), so the model fills road there and our CG
   car is composited later. Also run a second variant: a render of the coloured metric-depth points only, subsampled to about 10%.
   That is closer to StreetNVS's sparse-LiDAR training distribution.
3. **Reference images.** Five undistorted, photometry-corrected pinhole views from the clip's first frame: front, left and right
   pillar, left and right repeater, mapped to their camera IDs 0–4. `views.py` already produces pinhole reference photos. Skip the
   back fisheye, or resample it.
4. **Poses.** Use `T_world_cam` from our trajectory and rig in the OpenCV convention, metric, in StreetNVS's `camera_data.json`
   layout.
5. Command: `python -m streetnvs.infer --camera <our_traj> --reference-cameras 0 1 2 3 4 --num-frames 81 --steps 20 --cfg 5`
   ([repo](https://github.com/StreetNVS/StreetNVS)). The output is 10 fps. Interpolate to 30 fps (RIFE) only for viewing.

**Fallback the same day**, if the domain gap kills StreetNVS. StreetNVS was trained on Waymo pinhole cameras with real LiDAR; ours
are dashcam fisheyes with monocular depth and a teal ISP tint. Run **Cosmos-Transfer2.5-2B** on the same H100: the blur control is the
splat render, the depth control is our z, and edge comes from the render. That is 3 clips of 5.8 s, about 12 min each.

**How to judge it.**

- **Against our splat render:**
  1. A/B flip and side-by-side contact sheet at the same timestamps.
  2. **Layout fidelity.** Lane and road-marking masks from `pipeline/road` reprojected into the target camera, against markings
     detected in the generated frames. Pass if markings sit within about 3 px on at least 90% of frames.
  3. **Temporal stability.** Warp frame t into t+1 with our known camera motion and depth, and measure the photometric error on
     static, valid pixels. It must be lower than the splat render's flicker, or at worst no higher.
  4. **Hallucination budget.** The share of valid-mask pixels changed by more than ΔE 20 (reuse `genfill`'s change map).
  5. **Ground truth where it exists.** Also run StreetNVS along the *real* front-camera path (their "camera 0" mode) and score
     LPIPS/PSNR against the real front frames, held out. Do the same for the splat render.
- **Against Tesla.** Apply the Part 1 checklist to both outputs at a similar chase angle:
  - Is text pseudo or legible?
  - Does fine detail morph over 10 frames?
  - Are car roofs and the unseen sides of objects plausible?
  - Is the far field stretched or torn (O5)?
  - Does geometry stay rigid across the clip?
  - Is the sky painterly or real?

  Put Tesla crops and our frames on one contact sheet. Success means our clip reads as "the same family": rigid layout, plausible
  completion, and no worse texture boiling than Tesla's.

### 3.3 From clips to something interactive

| Stage | What | Feasibility (honest) |
|---|---|---|
| **A. "Generative" mode, pre-rendered** (days) | Pre-generate the default chase path for the whole drive (about 50 StreetNVS clips, about $5–20), plus optionally 2–3 presets (left-high, right-high, top-down). The viewer plays the generated video in sync with the timeline while the camera sits at a preset. When the user orbits, it cross-fades in about 200 ms to the live splat render, and fades back when the camera springs home. About 50–100 MB per preset at 480p H.264 [inference] | **Feasible.** The weak point is the visible pop at handover, because generated and splat appearance differ. Colour-match the splat to the generated clip, and do stage B |
| **B. Distil generated views into the splat** (about a week) | Use the generated chase frames (and orbit offsets) as pseudo-ground-truth when optimising the stitched splat, in the style of Difix3D+ "progressive update" and Lyra's self-distillation. Labels from `genfill/class` keep invented regions marked | **Feasible offline**, and the best complement to the splat track: orbit quality near the default path rises with zero runtime cost. The risk is baking hallucinations into the map, which is why the class map matters |
| **C. Streamed refinement** (weeks) | Viewer pose goes to a server, which renders the splat, runs **StreamDiffusionV2** (1–2 steps, 480p) or **Difix/Fixer** per frame, and sends the result back over WebRTC to the viewer | **Demo-grade.** About 8–13 FPS at 480p on one 4090 (paper figures), with latency of about 0.2–0.5 s [inference]. The 5070 is unmeasured, perhaps 4–8 FPS. The M5 Ultra manages per-frame Difix only, a few FPS, with flicker (unmeasured) |
| **D. Tesla-class: at least 30 FPS, about 200 ms, geometry-locked** | **Cosmos3-Nano-Sim-Transfer** takes depth, edge or blurred-RGB controls from our renderer. 48 FPS at 480p, 166 ms, on **one GB300** | **Not feasible on our hardware.** Possible as a rented exploration if GB200/GB300 capacity is available; price and availability not checked |

---

## Open questions (for Ryan)

1. **Budget.** Is a GPU rental of about $15 for E1 approved? About $20–60 more would generate the whole drive and a couple of presets.
2. **Labelling.** Should generated pixels be visibly marked in the viewer (a "predicted" tint or toggle)? Tesla doesn't mark them.
   `genfill` already computes the mask.
3. **Clean footage.** Can someone with v15 record a clean screen capture (park, orbit away and back, pause)? That one test
   (section 1.4, point 1) would settle the splat-versus-diffusion question with far more certainty than the promo can.
4. **Fixed presets or free orbit?** Is pre-rendered "Generative" mode with fixed presets (stage A) acceptable as the deliverable, or is
   free orbit with generation required? The second forces stage C or D.

## Verified sources not cited above

- Wan2.2 TI2V-5B runs on a 24 GB GPU with offload ([Wan2.2](https://github.com/Wan-Video/Wan2.2)).
- mlx-video supports LTX-2 and Wan2.1/2.2 on Apple silicon ([repo](https://github.com/Blaizzy/mlx-video)).
