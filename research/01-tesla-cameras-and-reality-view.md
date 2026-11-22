# 01: Tesla cameras, dashcam metadata and Reality View (research report)

Researched 2026-11-21. Every claim has a source URL. Labels used throughout:

- **[official]**: Tesla itself (tesla.com, Tesla/TeslaSI/Ashok Elluswamy posts, teslamotors GitHub, Tesla service manual).
- **[peer-reviewed]**: published paper.
- **[community]**: forum, blog, hacker, press, or third-party tool. Not verified by Tesla.
- **[inference]**: our own reasoning from cited numbers. Treat as a hypothesis to test.
- **Not found**: searched and found nothing citable.

Search limits: the Parallel search tool hit its free-tier rate limit partway through, and tesla.com / service.tesla.com return HTTP 403 to scripted fetches. Some service-manual and owner's-manual facts below therefore come from search-engine excerpts of those pages, not full-page reads. Reddit and X threads were mostly unreadable, so they are cited by title or excerpt only.

---

## 1. Tesla Reality View (FSD v15) and Tesla's Gaussian splatting work

### 1.1 The feature (announced today, 2026-11-21)

| Item | Finding | Source |
|---|---|---|
| Name | "Tesla Reality View" in Tesla's announcement; "Tesla Augmented Reality" in the v15 release notes | [official, via Electrek](https://electrek.co/2026/11/21/tesla-fsd-v15-rollout-25x-safety-reality-view/) quoting [@Tesla](https://x.com/Tesla/status/2124222447855743019) and [@TeslaSI release notes](https://x.com/TeslaSI/status/2124223658680061958) |
| Official description | "New real-time generative driving visualization that continuously streams a view from an arbitrary virtual camera situated above your vehicle, enabling you to see and control what FSD is planning to do. Select between fullscreen visualizations, maps or a split…" | [official @TeslaSI, quoted by TeslaNorth](https://teslanorth.com/2026/11/21/tesla-fsd-v15-first-rollout-reality-view) |
| Release-note line | "Introduced Tesla Augmented Reality, replacing all of the explicit model signals into a single end-to-end visualization using less total compute." | [official, quoted in full by Electrek](https://electrek.co/2026/11/21/tesla-fsd-v15-rollout-25x-safety-reality-view/) |
| Rollout | Began 2026-11-21 to a first group of customers; Musk says wide release in December with the Holiday Update | [Electrek](https://electrek.co/2026/11/21/tesla-fsd-v15-rollout-25x-safety-reality-view/), [Not a Tesla App](https://www.notateslaapp.com/news/5057/tesla-fsd-v15-coming-next-month-with-holiday-update-musk) |
| Hardware | AI4 (HW4). HW3 is on a separate v14 "Lite" branch, and no v15 has been announced for it | [TeslaNorth](https://teslanorth.com/2026/11/21/tesla-fsd-v15-first-rollout-reality-view), [tslna.com](https://tslna.com/en/tesla-hw3-vs-hw4-fsd) [community] |
| Musk teaser | "The new user interface will re-render reality in a mind-bending way." | [@elonmusk](https://x.com/elonmusk/status/2124051536170914023) via [Not a Tesla App](https://www.notateslaapp.com/news/5057/tesla-fsd-v15-coming-next-month-with-holiday-update-musk) |
| Layouts | Fullscreen, beside the map, or split; follows upcoming turns | [Techlantis](https://thetechlantis.com/news/tesla-fsd-v15-rollout-reality-view) [community summary of official notes] |
| Is it Gaussian splats? | **Not stated officially.** The release notes say "generative" and "arbitrary virtual camera" but never "Gaussian". Linking it to the 2025 "Generative Gaussian Splatting" work is a strong **[inference]**: same "generative" wording, an end-to-end network output, and novel virtual viewpoints. | n/a |
| Latency / frame rate on car | **Not found.** | n/a |
| What it renders | The street around the car rendered "far beyond the current FSD visualization", plus the planned path | [Electrek](https://electrek.co/2026/11/21/tesla-fsd-v15-rollout-25x-safety-reality-view/) [community description of the official video] |

### 1.2 The underlying tech: "Generative Gaussian Splatting"

**ICCV 2025 talk (Ashok Elluswamy, "Building an Autonomous Future", WDFM-AD workshop, Honolulu, Oct 2025).**
- Video: [YouTube IRu-cPkpiFk](https://www.youtube.com/watch?v=IRu-cPkpiFk). Tesla's written abridgement: [X article](https://x.com/aelluswamy/status/1981644831790379245) [official].
- Direct quotes from the official article:
  - Traditional 3DGS "depends on camera views with large baseline… Typical vehicle motion… is quite linear, and running traditional Gaussian Splatting results in poor reconstruction quality, especially from novel viewpoints."
  - Traditional 3DGS needs "good initialization from other pipelines and the total optimization time can be on the order of 10s of minutes."
  - "Tesla's Generative Gaussian Splatting… has great generalization, **runs in ~220 ms, doesn't require initialization, can model dynamic objects and can be jointly trained with the end-to-end AI model**."
  - "All of these gaussians are generated based on the production vehicle configuration's cameras."
  - The article shows a novel-view render and "generative gaussians (bottom) produced from the camera videos (top row)".
- Implications, all **[inference]**:
  - "Jointly trained with the end-to-end model" plus "no initialization" means a **feed-forward network** that predicts Gaussians from camera video, not per-scene optimisation. The Chinese write-up on [Zhihu](https://zhuanlan.zhihu.com/p/2081761056779659186) [community] calls it "前馈生成式高斯泼溅" (feed-forward generative GS), about 30 min → 220 ms, with no 3D keypoint initialisation.
  - Input is "camera videos", meaning several frames, so some temporal accumulation is implied. The **number of frames, the history length, and how memory works are not found**.
  - The **hardware behind the 220 ms figure is not stated**. It is not known whether that time was measured on the car's AI4 computer.
  - Poses: production cameras with Tesla's own calibration. Whether the network takes calibration and ego-motion as explicit inputs is **not found**.
- Third-party summaries of the talk: [Medium (WDFM-AD organisers)](https://medium.com/@byaman019/a-peek-into-teslas-autonomous-future-core-tech-revealed-by-vp-ashok-elluswamy-at-iccv-2025-906df7ae95de), saying the splats "generalize much better and do not rely on traditional initialization techniques" and are used to debug the system; [Teslarati](https://www.teslarati.com/tesla-vp-explains-why-end-to-end-ai-future-self-driving); [NextBigFuture](https://www.nextbigfuture.com/2025/10/tesla-vp-ashok-describes-technology-of-fsd.html) [community].

**ScaledML 2026 talk (Jan 2026).** Write-up by [AINEXT](https://ain3xt.com/en/posts/20260210-tesla-end-to-end-foundation-model) [community]; source video [YouTube LFh9GAzHg1c](https://www.youtube.com/watch?v=LFh9GAzHg1c).
- Traditional GS takes ">30 minutes" and "only looks right near the training views". Tesla's version "finishes in a few hundred milliseconds".
- It "still produces semantically correct 3D scenes when viewed from off-training angles, generating the correct vehicle types and shapes". That points to a generative prior that completes unseen geometry, especially of vehicles.
- The reconstruction "is embedded directly inside the driving neural network".
- The same talk describes Tesla's neural world simulator, which generates all 8 cameras at 36 FPS and 5 MP. This is separate from the GS work.

**CVPR 2026 keynote (June 2026)**, "Building Foundational Models for Robotics at Tesla": [YouTube 2V0ddkF1Svo](https://www.youtube.com/watch?v=2V0ddkF1Svo) and [i2hB_8BeueM](https://www.youtube.com/watch?v=i2hB_8BeueM). A Chinese recap ([OFweek](https://www.ofweek.com/auto/2026-06/ART-70109-8110-30689565.html)) says it largely reused older slides. **No new GS numbers found** (the transcript was not readable).

**Patents / papers.**
- No Tesla patent or paper on Gaussian splatting found.
- Related Tesla patent applications: "Vision-Based Occupancy Determination", [US 2026/0073549 A1](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/20260073549) (inventors include Elluswamy), and "Vision-Based Surface Determination", summarised by [Not a Tesla App](https://www.notateslaapp.com/news/2773/how-teslas-fsd-builds-a-3d-world-from-pixels-part-3). These cover voxel occupancy and road surfaces, not splats.

**Summary of concrete technical facts**

| Question | Answer | Confidence |
|---|---|---|
| Feed-forward vs optimised | Feed-forward/generative network, jointly trained with the E2E model | High (official wording; "feed-forward" is our and Zhihu's reading) |
| Run time | ~220 ms (ICCV 2025); "few hundred ms" (ScaledML 2026); hardware unknown | High for the number, unknown for the hardware |
| Dynamic objects | "Can model dynamic objects". Method not disclosed | High that it does; method not found |
| Needs initialisation / SfM | No | High |
| Needs poses | Uses production camera configuration; explicit pose input not found | Medium |
| Novel-view quality | Better off-trajectory than classic 3DGS; hallucinates plausible vehicle shapes | Medium (demo videos only) |
| Temporal accumulation | Input is "camera videos"; mechanism and history length not found | Low |
| On-car latency of Reality View | Not found | n/a |

---

## 2. Model 3 HW4 camera rig

### 2.1 Which cameras exist (Model 3 "Highland", 2024+, HW4/AI4)
- The [ISPRS paper on a Model 3 Highland](https://isprs-archives.copernicus.org/articles/XLVIII-2-W12-2026/25/2026/isprs-archives-XLVIII-2-W12-2026-25-2026.pdf) [peer-reviewed] lists six recorded units: "one above the rear license plate, one on each B-pillar, two near the windshield above the rearview mirror (only one of which was used in this study), and one on each front fender."
- HW4 board camera connectors per greentheonly ([thread, Feb 2023](https://threadreaderapp.com/thread/1625905179282354194.html)) [community]: "Wide, Main, Backup, L-FF-Side, R-FF-Side, L-FF-Rear, R-FF-Rear, F-SVC/L-SVC/R-SVC (bumper), Selfie", plus a new GPS module "with triband antenna".
- HW4 has two forward windshield cameras ("Bi-Forward Facing"). HW3 had three: Narrow 35°, Main 50°, Wide 120°.
  - HW3 figures: [Electrek 2016](https://electrek.co/2016/10/20/tesla-new-autopilot-hardware-suite-camera-nvidia-tesla-vision) [community, quoting Tesla]; the [archived tesla.com/autopilot page](https://web.archive.org/web/20230509141041/https://www.tesla.com/autopilot) [official] gives Wide 120° fisheye (60 m), Main (150 m), Narrow (250 m), forward-looking side 90° (80 m), rearward-looking side (100 m), rear (50 m).
  - HW4 change: [Reddit, HW4 service-manual find](https://www.reddit.com/r/teslamotors/comments/11m64ks/hw4_installation_instructions_have_started/) [community].
- Newer US cars add a front bumper camera ([AutoPilot Review](https://www.autopilotreview.com/tesla-hardware-4-rolling-out-to-new-vehicles)) [community]. It is not in the dashcam set.

### 2.2 Sensor, shutter, frame rate
- **Sensor:** Sony **IMX963**, about 5.4 MP, **2896×1876**, 3.0 µm pixels, 120 dB HDR, flicker mitigation, **rolling shutter**. Believed to be a custom IMX490 variant. Source: [Not a Tesla App, Dec 2025](https://www.notateslaapp.com/news/3381/tesla-may-be-upgrading-fsd-cameras-soon) and [Teslarati](https://www.teslarati.com/tesla-making-change-exterior-cameras-potential-upgrade/), based on greentheonly firmware findings [community].
- These sources say "AI4 cameras use the IMX963". They do not confirm it per camera position.
- A successor sensor, "IMX00N", was found in firmware in Dec 2025 ([Electrek](https://electrek.co/2025/12/02/tesla-new-camera-upgrade-casting-more-doubt-full-self-driving-promises/)). Newer cars may differ.
- **Frame rate:**
  - The dashcam records about 36 fps on HW4 today. ISPRS (firmware 2025.20.3) found about 36 fps. Harper Shaw found 2,175 SEI frames in a 60.394 s front clip, which is 36.0 fps ([teslaseidecoder](https://teslaseidecoder.netlify.app/), [teslasei2](https://teslasei2.netlify.app)) [community].
  - Early HW4 firmware in 2023 recorded 24 fps; HW3 recorded 36 fps at 1280×960 ([blogordie 2023](https://www.blogordie.com/2023/09/hw4-tesla-new-self-driving-hardware/)) [community].
  - Tesla says its E2E net runs at 36 Hz ([Chinese research note on the ICCV talk](https://pdf.dfcfw.com/pdf/H3_AP202511051775434131_1.pdf?1762334029000.pdf)) and the world simulator outputs 36 FPS. So 36 fps is very likely the native capture rate **[inference]**.
  - Frame intervals are **not guaranteed even**. Harper Shaw explicitly did not test this.

### 2.3 Field of view, projection and calibration evidence
The most rigorous public source is the ISPRS 2026 paper by Barazzetti, Previtali and Roncoroni (Politecnico di Milano) [peer-reviewed]:
- **Method:** a 5-camera rig calibrated in Agisoft Metashape bundle adjustment from parking-lot drives, with 23 total-station control points. The rear camera was **excluded** because it "proved inconsistent, with unreliable parameter estimation", cause unknown.
- **Projection:** all five cameras modelled as **equidistant fisheye**, including the front.
  - For the front camera, pinhole and fisheye fit equally well (RMS 0.48 vs 0.59 px), but pinhole "required unusually large distortion coefficients".
  - **Front focal length: 3638.74 px (fisheye) / 3638.51 px (pinhole)** at 2896×1876.
- **Rig stability:** extrinsics repeat to 1–2 cm and 0.1–0.2° across projects. Prior σ used: ±0.015 m and ±0.1°. Reprojection RMS was about 0.4–0.7 px.
- **Gap:** the numeric intrinsics and extrinsics of the other four cameras are **not printed in the paper text** we retrieved (Figure 2 shows the layout only).
- **Derived front FOV [inference]:**
  - Pinhole: HFOV = 2·atan(1448/3638.6) ≈ **43.4°**, VFOV ≈ **28.9°**.
  - Equidistant: HFOV ≈ 2896/3638.7 rad ≈ **45.6°**, VFOV ≈ **29.5°**.
  - So the front dashcam stream is the narrower **"Main"** windshield camera, not the 120° "Wide". It also looks like the full sensor (2896×1876 = IMX963 native).

### 2.4 Per-camera spec table

| Camera (dashcam name) | Sensor | Native res | Dashcam res | HFOV / VFOV | Projection | Mount location | x/y/z, yaw/pitch | Shutter | Confidence |
|---|---|---|---|---|---|---|---|---|---|
| front | IMX963 [community] | 2896×1876 [community] | 2896×1876 ([ISPRS](https://isprs-archives.copernicus.org/articles/XLVIII-2-W12-2026/25/2026/isprs-archives-XLVIII-2-W12-2026-25-2026.pdf)) | ~43–46° / ~29°, derived from ISPRS f = 3638.7 px [inference] | Near-rectilinear; equidistant fits slightly better (ISPRS) | Behind windshield, top centre, above rear-view mirror (ISPRS) | **Not found** | Rolling [community] | Res: high. FOV: medium (one paper). "Main not Wide": medium |
| back | IMX963 assumed [community, not per-camera] | likely 2896×1876 [inference] | 1448×938 | **Not found.** Much wider than HW3 ([Not a Tesla App](https://www.notateslaapp.com/news/1564/tesla-s-hw3-and-hw4-cameras-comparing-the-differences-in-quality-and-hardware)); "180°" claim on [Reddit](https://reddit.com/r/TeslaModelY/comments/1j34m5b/thank_you_for_adding_rear_cross_traffic_chime) | Strong fisheye ([Not a Tesla App](https://www.notateslaapp.com/news/1564/tesla-s-hw3-and-hw4-cameras-comparing-the-differences-in-quality-and-hardware); our observation) | Above rear licence plate (ISPRS) | **Not found** | Rolling (assumed) | Low. ISPRS could not calibrate it reliably |
| left_pillar / right_pillar | IMX963 assumed | likely 2896×1876 [inference] | 1448×938 | HW3 official 90° ([archived tesla.com](https://web.archive.org/web/20230509141041/https://www.tesla.com/autopilot)); **HW4 not found** | Fisheye (equidistant per ISPRS) | B-pillars (ISPRS) | **Not found** | Rolling (assumed) | Low–medium |
| left_repeater / right_repeater | IMX963 assumed | likely 2896×1876 [inference] | 1448×938 | **Not found** (HW3: 100 m range, FOV unpublished) | Fisheye (equidistant per ISPRS) | Front fenders, facing rearward (ISPRS; [Reddit](https://reddit.com/r/TeslaFSD/comments/1nngrhd/blindspots_with_hw4_camera)) | **Not found** | Rolling (assumed) | Low–medium |

**Downscale vs crop [inference].** 1448×938 is exactly half of 2896×1876 on both axes, so the aspect ratio is identical (1.5437). That fits a 2× decimation of the full sensor rather than a crop, but this is unconfirmed. Test it by fitting intrinsics and checking that the implied FOV matches the in-car Service-mode camera preview. ISPRS's ability to fit five cameras at 0.5 px RMS shows the streams are at least geometrically clean (no rolling crop or stabilisation).

### 2.5 HW3 vs HW4 Model 3 differences relevant to us
- **HW3:** 1.2 MP (1280×960). Dashcam recorded 4 cameras at 1280×960, 36 fps ([blogordie](https://www.blogordie.com/2023/09/hw4-tesla-new-self-driving-hardware/), [tslna](https://tslna.com/en/tesla-hw3-vs-hw4-fsd)). Three forward cameras.
- **HW4:** 5 MP. Front recorded at 2896×1876, the others at 1448×938. Two forward cameras; red-tinted lenses; rear camera has a wider fisheye.
- **B-pillar recording:** added by a 2025 software update. Before that, only front/back/repeaters were recorded ([ISPRS](https://isprs-archives.copernicus.org/articles/XLVIII-2-W12-2026/25/2026/isprs-archives-XLVIII-2-W12-2026-25-2026.pdf); [blogordie update note](https://www.blogordie.com/2023/09/hw4-tesla-new-self-driving-hardware/)).

### 2.6 Model 3 Highland body dimensions (for extrinsic priors)

| Dimension | Value | Source |
|---|---|---|
| Length | 4720 mm | [ev-database](https://ev-database.org/car/3403/Tesla-Model-3-RWD) |
| Width (body / mirrors extended) | 1850 / 2089 mm | [ev-database](https://ev-database.org/car/3403/Tesla-Model-3-RWD), [TeslaOracle](https://www.teslaoracle.com/2023/09/06/tesla-model-3-highland-detailed-specs-and-features/) |
| Height | 1440 mm (owner's manual; other sources 1441–1442) | [Owner's manual excerpt](https://www.tesla.com/ownersmanual/model3/en_us/GUID-56562137-FC31-4110-A13C-9A9FC6657BF0.html), [evspecifications](https://evspecifications.com/en/comparison/061d8b70) |
| Wheelbase | 2875 mm | owner's manual excerpt (above) |
| Front / rear overhang | 868 / 977 mm | owner's manual excerpt (above) |
| Ground clearance | 138 mm | owner's manual excerpt (above) |
| Track F/R (non-Performance) | 1584 / 1584 mm (Performance 1582/1560) | [Service manual excerpt](https://service.tesla.com/docs/Model3/ServiceManual/2024/en-us/GUID-0E6A27EF-310C-4287-809B-CAD939F95581.html), owner's manual excerpt |

Published camera positions (x/y/z, yaw/pitch) relative to any origin: **Not found** in any official or community source. The forward-camera pitch-verification procedure uses a target with its bottom edge 90 cm above the ground ([service manual excerpt](https://service.tesla.com/docs/Model3/ServiceManual/2024/en-us/GUID-3D66A5D0-2B1E-4B3F-AECB-CE9C9A98234E.html)). That number is about the target, not the camera height.

---

## 3. Existing camera calibration for Tesla dashcam footage

| Source | What it gives | Usable? |
|---|---|---|
| **ISPRS 2026, Barazzetti et al.** ([PDF](https://isprs-archives.copernicus.org/articles/XLVIII-2-W12-2026/25/2026/isprs-archives-XLVIII-2-W12-2026-25-2026.pdf), [video](https://youtu.be/pOFpUp-vlHU)) [peer-reviewed] | Full method; front f = 3638.7 px; equidistant model for all; extrinsic repeatability 1–2 cm / 0.1–0.2°; rear camera fails; GS via Jawset Postshot (MCMC profile) using Metashape poses | Method yes. **Numbers for other cameras not published** in the text, so the authors would have to be asked |
| **Rami Tamimi (surveyor), "I Made a Point Cloud Using Tesla Cameras"**, 2026-06-11 ([YouTube](https://youtube.com/watch?v=RYq16soBiuo)) [community] | Model Y: checkerboard intrinsics per camera, extrinsics from surveyed control points, lever arms to a roof GNSS, Pix4Dmatic point cloud | Workflow only; no published numbers found |
| **Trînc Gabriel, "Can Tesla Cameras See in 3D?"**, 2026-10-07 ([YouTube](https://www.youtube.com/watch?v=JfeoHimMOck)) [community] | 4 cameras locked as one rig; used the **SEI GPS telemetry to fix the camera path** after "every 3D software failed"; splatted with Spirula Studio | Workflow only |
| Tesla self-calibration | The car self-calibrates over 20–25 miles (up to 100); "Clear Calibration" is available in Controls > Service ([Tesla DIY doc](https://service.tesla.com/docs/Public/diy/modely/en_us/GUID-D6F7D1BC-193D-4A2E-99B0-E3BA3D41BDE2.html)) [official] | Values are **not exposed** to owners. Not found on CAN, in the dashcam files, or in Fleet API |
| Service-mode pitch verification | Front-camera pitch check against a printed target (Model 3 service manual) [official] | Gives a procedure, not numbers |
| GitHub calibration repos with Tesla intrinsics / undistortion / stitching | **Not found.** All TeslaCam tools found (tesla_dashcam, Sentry-Six/Sentry Studio, TDashcamStudio, teslareplay, teslacam-viewer) only play back or grid-mosaic. None undistort or stitch geometrically | n/a |
| Dashcam-video 3DGS papers (general, not Tesla-specific) | [DC-Gaussian](https://arxiv.org/html/2405.17705v3) (windshield reflections/occlusions), [RDC-GS (CGF 2026)](https://onlinelibrary.wiley.com/doi/10.1111/cgf.70315), [RIDE-Splat](https://github.com/ridesplat/ride-splat) | Relevant to the front camera behind glass |

Note: the ISPRS authors reported "the absence of synchronized GNSS or odometry data" as a major limitation. Their firmware (2025.20.3) predates the SEI telemetry (firmware 2025.44.25+). **We have what they lacked: per-frame GPS, heading, speed and acceleration.**

---

## 4. Tesla dashcam SEI metadata

### 4.1 Official spec
- Repo: [teslamotors/dashcam](https://github.com/teslamotors/dashcam) (created 2025-12-01, 3 commits, no licence). Contents: `dashcam.proto`, `sei_extractor.py`, `dashcam-mp4.js`, `sei_explorer.html`; online explorer at [teslamotors.github.io/dashcam](https://teslamotors.github.io/dashcam/sei_explorer.html) [official].
- [`dashcam.proto`](https://github.com/teslamotors/dashcam/blob/master/dashcam.proto): `version`(1), `gear_state`(2: PARK/DRIVE/REVERSE/NEUTRAL), `frame_seq_no`(3, uint64), `vehicle_speed_mps`(4, float), `accelerator_pedal_position`(5, float), `steering_wheel_angle`(6, float), `blinker_on_left/right`(7/8), `brake_applied`(9), `autopilot_state`(10: NONE/SELF_DRIVING/AUTOSTEER/TACC), `latitude_deg`(11, double), `longitude_deg`(12, double), `heading_deg`(13, double), `linear_acceleration_mps2_x/y/z`(14–16, double).
- **The proto has no comments on units beyond the field names, no axis definitions, and no sign conventions.** Tesla has documented none of these [official, by absence].
- **Requirements** (README): firmware **2025.44.25+**, **HW3+**; "If car is parked, SEI data may not be present."
- **Encoding** ([`sei_extractor.py`](https://github.com/teslamotors/dashcam/blob/master/sei_extractor.py)): H.264 NAL type 6, SEI payload type 5 (user_data_unregistered). The payload starts after a run of `0x42` bytes ending in `0x69`. Emulation-prevention bytes are stripped and the last byte dropped. There is one SEI per video frame (2,174–2,175 per 1-min front clip per [Harper Shaw](https://teslasei.netlify.app)).

### 4.2 Field semantics: what the community has established (none officially confirmed)

| Field | Best available reading | Evidence | Confidence |
|---|---|---|---|
| `vehicle_speed_mps` | m/s | 11.688889 m/s shown as 26.1 mph ([teslasei2](https://teslasei2.netlify.app)) | High |
| `accelerator_pedal_position` | **0–100 (percent)** in real clips, despite tools/docs saying 0–1 | Real values 15.60, 14.0 ([teslasei](https://teslasei.netlify.app), [teslasei2](https://teslasei2.netlify.app)); teslareplay draws a "0–100% bar" ([README](https://github.com/martyearthy/teslareplay)); [DeepWiki](https://deepwiki.com/teslamotors/dashcam/3.2-usage-and-examples) and [heise](https://www.heise.de/en/news/Tesla-New-dashcam-feature-displays-detailed-telemetry-data-in-video-11112252.html) claim 0–1 | Medium. Check our data |
| `steering_wheel_angle` | Degrees at the wheel (not road-wheel angle). Sign is **contested**: several tools say + = right / − = left | [teslareplay](https://github.com/martyearthy/teslareplay), [video-fsd-enabler](https://github.com/sioakim/video-fsd-enabler), [teslacam-replay DeepWiki](https://deepwiki.com/ben-z/teslacam-replay/6.2-telemetry-data-format) (range ±450) | Low–medium. Verify against heading rate |
| `heading_deg` | 0–360°, clockwise from true north | [teslareplay README](https://github.com/martyearthy/teslareplay) [community]; example 253.47° ([teslasei](https://teslasei.netlify.app)) | Medium. Verify against the GPS track bearing |
| `latitude_deg` / `longitude_deg` | WGS84 decimal degrees | [teslareplay](https://github.com/martyearthy/teslareplay) [community] | Medium |
| lat/lon raw vs fused, update rate | **Not found.** HW4 has a new tri-band GNSS module ([greentheonly](https://threadreaderapp.com/thread/1625905179282354194.html)) but nothing on rate or filtering | n/a | Measure ourselves |
| `linear_acceleration_mps2_x/y/z` | m/s², **gravity apparently removed** (z ≈ +0.18–0.19 at rest-ish, not 9.8). **x = lateral, y = longitudinal** (not ISO 8855) | [Issue #11](https://github.com/teslamotors/dashcam/issues/11) claims x = lateral, y = longitudinal; Harper Shaw data shows y = +0.5675 while accelerating at 14% pedal and x = 0.0325 ([teslasei2](https://teslasei2.netlify.app)) | Medium. Signs (left vs right, up vs down) not found |
| `frame_seq_no` | **Per-camera private frame counter**, not a shared timestamp | [teslasei2](https://teslasei2.netlify.app): five cameras share range 150,030–152,204; the **back camera is offset by +17** and starts about 5 frames (~140 ms) later in real time (one clip) | Medium (one clip) |
| `version` | Schema version (value 1 seen) | [DeepWiki](https://deepwiki.com/teslamotors/dashcam/3.2-usage-and-examples); Teslemetry's [annotated proto](https://github.com/Teslemetry/tesla-protocol/blob/main/proto/dashcam/dashcam_sei.proto) says "meaning beyond that is not confirmed" | Medium |

Cross-camera telemetry is identical at matching moments. The same vehicle-bus record appears in all cameras, and a frame missing from one camera's CSV is still in the other cameras ([teslasei2](https://teslasei2.netlify.app)). The bus is therefore the shared clock: **align cameras by matching telemetry fingerprints** (e.g. acceleration spikes), not by frame number or file start.

### 4.3 Known caveats
1. **Clip length varies** by milliseconds and by 1–2 frames per camera (59.889–59.893 s) ([ISPRS](https://isprs-archives.copernicus.org/articles/XLVIII-2-W12-2026/25/2026/isprs-archives-XLVIII-2-W12-2026-25-2026.pdf)). The authors attribute some reconstruction artefacts to "slight temporal desynchronization between cameras".
2. **The back camera starts later** and has a frame-number offset (see the table above).
3. **The official extractor can silently drop SEI frames.** Harper Shaw found one frame present in the binary but missing from the explorer CSV; their raw byte scanner found it ([teslasei2](https://teslasei2.netlify.app)).
4. **Timing between the SEI sample and the exposure is unverified.** The one-to-one count is confirmed; the exact synchronisation is not ([teslaseidecoder](https://teslaseidecoder.netlify.app/)).
5. **Exports from the Tesla mobile app strip the SEI** on iOS ([#12](https://github.com/teslamotors/dashcam/issues/12)) and Android ([#10](https://github.com/teslamotors/dashcam/issues/10)). Use files straight off the USB drive.
6. **Format drift.** One HW4 car on firmware 2026.27.11 produced **HEVC 1112×720 at 30 fps** clips whose SEI did not decode as the published proto ([#14](https://github.com/teslamotors/dashcam/issues/14)). Later "Tesla has fixed the telemetry issue". A draft H.265 parser exists in [PR #15](https://github.com/teslamotors/dashcam/issues/15). An Australian-branch firmware also broke the tools ([#7](https://github.com/teslamotors/dashcam/issues/7)).
7. **Sentry/parked:** SEI may be absent when parked (README). Lat/lon of 0/NaN should be handled; teslareplay normalises invalid heading ([README](https://github.com/martyearthy/teslareplay)).
8. There is **no altitude and no gyro/yaw rate** (requested in [#4](https://github.com/teslamotors/dashcam/issues/4)). Pitch and roll must come from vision.

### 4.4 Open-source parsers
- Official: [`sei_extractor.py`](https://github.com/teslamotors/dashcam/blob/master/sei_extractor.py), [`dashcam-mp4.js`](https://github.com/teslamotors/dashcam).
- Protobuf packages (npm/PyPI/Go) from [Teslemetry/tesla-protocol](https://github.com/Teslemetry/tesla-protocol).
- Raw-scan forensic decoder: [Harper Shaw / TVN Labs](https://teslasei2.netlify.app).
- Viewers with SEI support:
  - [martyearthy/teslareplay](https://github.com/martyearthy/teslareplay): 6-cam sync, G-meter, GPS route.
  - [kandotrun/teslacam-viewer](https://github.com/kandotrun/teslacam-viewer): Rust/GPUI Mac app, 6-cam sync.
  - [DeaglePC/TDashcamStudio](https://github.com/DeaglePC/TDashcamStudio) / [TeslaCamPlayer](https://github.com/DeaglePC/TeslaCamPlayer).
  - [Sentry-Six/Sentry Studio](https://github.com/ChadR23/Sentry-Six).
  - [JeandreRoux/Tesla-Dashcam-Telemetry-Viewer](https://github.com/JeandreRoux/Tesla-Dashcam-Telemetry-Viewer).
  - [ehendrix23/tesla_dashcam #231](https://github.com/ehendrix23/tesla_dashcam/issues/231) (overlay support request).
  - `sentrysearch` [metadata.py](https://raw.githubusercontent.com/ssrajadh/sentrysearch/refs/heads/master/sentrysearch/metadata.py).

---

## 5. Prior art: a "Reality View" from consumer car cameras

| Who / what | Data | Method | Result | Source |
|---|---|---|---|---|
| Barazzetti, Previtali, Roncoroni (PoliMi), ISPRS 3D-ARCH, Feb 2026 | Model 3 Highland, 5 of 6 dashcam streams, 4 fps extraction, no GNSS | Metashape rig BA (fixed intrinsics, soft extrinsic constraints) → Postshot 3DGS (MCMC) | Works, but "visible artifacts… particularly evident along the road" (linear trajectory). Mesh accuracy 4.9 ± 5.1 cm vs TLS; several metres of drift over 3 km without GNSS; Street View used for georeferencing | [PDF](https://isprs-archives.copernicus.org/articles/XLVIII-2-W12-2026/25/2026/isprs-archives-XLVIII-2-W12-2026-25-2026.pdf), [video](https://youtu.be/pOFpUp-vlHU) |
| Trînc Gabriel (YouTube), Oct 2026 | Tesla dashcam, 4 cameras | SEI GPS for the path, rigid 4-cam rig, Spirula Studio GS | A 3D street; video covers "where it falls apart" | [YouTube](https://www.youtube.com/watch?v=JfeoHimMOck) |
| Reddit r/GaussianSplatting, 2026 | "I did reconstruct a street in 3D using only Tesla's cameras" | Not readable | n/a | [thread](https://www.reddit.com/r/GaussianSplatting/comments/1x11vmc/i_did_reconstruct_a_street_in_3d_using_only/) |
| Rami Tamimi (surveyor), Jun 2026 | Model Y dashcam + external RTK GNSS | Checkerboard intrinsics, control-point extrinsics, Pix4Dmatic | Point cloud, proof of concept | [YouTube](https://youtube.com/watch?v=RYq16soBiuo) |
| Tesla itself | 8 production cameras | Generative feed-forward GS in the E2E net | ~220 ms, dynamic objects, now shipping as Reality View (GS link is our inference) | §1 |

Real-time / feed-forward research that is the closest open analogue to Tesla's approach. Listed for context; a separate method survey should go deeper.
- **[RigRecon](https://github.com/zju3dv/RigRecon)** (ZJU + Xiaomi EV, 2026): rig-aware feed-forward reconstruction from *calibrated multi-camera video*, >1000 images per pass, optional Gaussian heads via gsplat, weights on Hugging Face.
- **Rig3R** ([arXiv 2506.02265](https://arxiv.org/abs/2506.02265)): rig-metadata-conditioned feed-forward reconstruction; suggested by the ISPRS authors as future work alongside **VGGT-Long** ([GitHub](https://github.com/DengKaiCQ/VGGT-Long)).
- **MapAnything** ([site](https://map-anything.github.io)): metric feed-forward reconstruction that accepts optional intrinsics and poses.
- Feed-forward driving GS:
  - UniSplat ([arXiv 2511.04595](https://arxiv.org/html/2511.04595v1)): dynamic, 3D latent scaffold, temporal fusion.
  - DrivingScene ([arXiv 2510.24734](https://arxiv.org/html/2510.24734)): online, from two consecutive surround frames.
  - DrivingRecon ([PDF](https://www.yingcong.me/publication/2025-lu-drivingrecon/2025-lu-drivingrecon.pdf)).
  - WorldSplat ([ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/6fa4a1def9d60b5bf6078e13c900c45c-Abstract-Conference.html)).
  - MCSplat ([arXiv 2610.25345](https://arxiv.org/abs/2610.25345)).
  - PointForward ([arXiv 2605.11594](https://arxiv.org/html/2605.11594)).
  - VGGD single-frame surround view ([arXiv 2608.10682](https://arxiv.org/abs/2608.10682)).
  - These are trained on nuScenes/Waymo/Argoverse rigs. **Applying them to Tesla's rig (fisheye side cameras, different FOVs) is untested publicly.** Not found.
- Other consumer surround-view rigs (Rivian, GM Surround Vision, etc.): **Not found.** Search quota ran out before this was covered thoroughly. Sentry Studio does read GM Surround Vision footage ([GitHub topics](https://github.com/topics/sentry-mode)), but no reconstructions were found.

---

## Open questions / things we must measure ourselves

**Calibration**
1. **Intrinsics of all six streams.** Fit equidistant (Kannala-Brandt) models per camera. Check the front against f ≈ 3638.7 px at 2896 px width. A checkerboard or ChArUco board in a parking lot, as in Tamimi's workflow, is the robust route; self-calibration via COLMAP/GLOMAP with a shared-intrinsics rig is the fallback.
2. **Is 1448×938 a 2× downscale or a crop** of the sensor? Compare our fitted FOV to the full-frame Service-mode camera preview.
3. **Back-camera model.** ISPRS failed with it, so test a more general fisheye model (more Kannala-Brandt terms or a unified/double-sphere model). It may be >180°.
4. **Rig extrinsics.** Nothing is published. Solve with rig-constrained bundle adjustment, using body dimensions as bounds (wheelbase 2.875 m, height 1.44 m, width 1.85 m). Then pin metric scale with SEI speed × dt.

**Timing**

5. **Per-camera time offsets.** Match telemetry fingerprints across files, check the back camera's +17 frame offset in our clips, and estimate the start offset (~5 frames) and any drift within a clip.
6. **Frame-interval regularity.** Read the MP4 `stts` per camera and check whether it is a constant 1/36 s.

**SEI telemetry**

7. **GPS update rate and smoothing.** Count how often lat/lon actually change across 36 fps frames, and check for steps vs interpolation. Compare heading to the GPS track bearing and to the vision-estimated yaw.
8. **Sign conventions.** Correlate steering-angle sign and accel x/y signs with turn direction (from heading rate) and braking/acceleration (from speed derivative).
9. **Pedal units** (0–1 vs 0–100) in our firmware version.
10. **Rolling shutter.** The readout time is unknown. Estimate it from straight vertical edges during fast yaw. It matters at speed for the 1448-px side cameras.

**Reality View and data**

11. **Reality View specifics.** Whether it is literally Gaussian splats, its on-car latency, and how much history it uses are all unknown. Watch for Tesla AI posts and owner videos over the December wide release.
12. **ISPRS rig numbers.** The paper prints only the front focal length. Ask the authors (luigi.barazzetti@polimi.it per the paper) for the full P3 intrinsics and extrinsics. That would give a validated Highland prior for the five non-rear cameras.
13. **Our firmware/codec.** Confirm our clips are H.264 with SEI that decodes as the published proto, given the 2026 HEVC 1112×720 / 30 fps report in issue #14. Keep a raw-scan fallback parser.
