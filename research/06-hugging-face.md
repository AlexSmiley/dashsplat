# 06 · Hosting Reality Splat on Hugging Face: compute, models and a design

Researched 2026-11-22 (desk research and reading this repo; nothing was run, created or bought). Sources are numbered
[S1]… and listed at the end with URLs; all were accessed on 2026-11-22. Labels:

- **[measured]**: a number from this repo's reports (Studio M5 Ultra or the rented A40), with the file it comes from.
- **[doc]**: Hugging Face or vendor documentation, model card or licence file.
- **[community]**: forums or third-party write-ups.
- **[estimate]**: our arithmetic or inference. Treat as a hypothesis until measured.
- **Unverified**: searched and not confirmed.

---

## 0. Summary

- **Shape.** One public **Gradio Space on ZeroGPU** as the front door, one free **static Space** for the viewer, and
  **Hugging Face Jobs billed to the signed-in user** for anything heavy. Uploads and results live in the **user's own
  private Storage Bucket**, written with their OAuth token, never in ours. A Dockerfile and "Duplicate this Space" cover
  people who want to run it on their own GPU or account.
- **Quick tier fits ZeroGPU; full tier does not.** The quick preview's only heavy GPU stage is panoptic masks (about
  4 A40-minutes per driving minute, [measured] projection). The full tier is roughly 50 compute-minutes per driving minute
  of geometry plus 66 A40-minutes per minute of generated video ([measured]). That is far beyond a 5- or 40-minute
  daily ZeroGPU quota, and 2-4x cheaper on a Job than on ZeroGPU credits.
- **Three facts changed the plan while researching.**
  1. Tesla **encrypts dashcam clips by default since software 2026.20** (May 2026). Users must decrypt in the car or at
     dashcam.tesla.com before uploading [S30][S31]. The page must say so up front and must never ask for Tesla credentials.
  2. **"Tesla Reality View" became Tesla's official feature name on 2026-11-21** (FSD v15) [S33][S34]. The public project
     needs its own name and its own look.
  3. The **DA3-LARGE-1.1 model card says Apache-2.0 but the official Depth Anything 3 README says CC BY-NC 4.0** [S16][S17].
     Treat it as non-commercial until the authors confirm.
- **Two Apple-only stages still block any Linux run** (`pipeline/cuda/PORT.md`): the Apple Vision vehicle classifier
  and the MetalSplat optimiser. Both have permissive CUDA replacements (SigLIP 2 zero-shot or RF-DETR; gsplat), but
  neither replacement has been checked against our outputs.

---

## 1. Compute on Hugging Face (as documented on 2026-11-22)

### 1.1 Spaces

| Item | What the docs say | Source |
|---|---|---|
| Who can create | Static Spaces are free for everyone. **Gradio and Docker Spaces on compute need a paid plan (PRO $9/month)**, except that free accounts in good standing can host **up to 2 ZeroGPU Gradio Spaces** | [S1][S3][S2] |
| Default container | 2 vCPU, 16 GB RAM, **50 GB ephemeral disk**, lost on restart or stop | [S3][S5] |
| CPU tiers | CPU Basic free; CPU Upgrade 8 vCPU / 32 GB, $0.03/h | [S4] |
| GPU tiers (per hour) | T4 small $0.40, T4 medium $0.60, L4 $0.80, A10G small $1.00, A10G large $1.50, L40S $1.80, A100 80 GB $2.50, RTX PRO 6000 96 GB $2.75, multi-GPU variants up to 8x; **H100 removed December 2025** | [S4][S2] |
| Billing | **Per minute while Starting or Running**, whether used or not; not during build; paused time free | [S4][S3] |
| Sleep and wake | Free hardware sleeps after about 48 h idle. Paid hardware runs forever unless you set a **custom sleep time**; asleep is not billed; the next visitor wakes it (cold start including model loading: not documented, unmeasured) | [S4] |
| Replicas | Paid hardware only, each billed | [S4] |
| Persistence | Attach **Storage Buckets as volumes** (read-write or read-only). The older "persistent storage" add-on is not mentioned on the current page; unclear whether it is still sold | [S5] |
| Visibility | Public, **protected** (PRO; app reachable, code private, not clonable) or private | [S36] |
| Duplication | Private by default; secrets are not copied; the target account needs a paid plan, with the same 2-ZeroGPU exception | [S3] |

### 1.2 ZeroGPU

| Item | What the docs say | Source |
|---|---|---|
| GPU | **NVIDIA RTX PRO 6000 Blackwell.** `large` (default) = half a card, 48 GB, 1x quota; `xlarge` = full card, 96 GB, **2x quota** and longer queues. The Space API still reports the flavour as `zero-a10g`; forum posts from April 2026 say H200, so the hardware changed this year | [S1][S25] |
| Per-call duration | Default **60 s**; `@spaces.GPU(duration=N)` or a callable for input-dependent durations. **Each visitor tier has a per-call cap that is not published**: exceeding it fails at once with "ZeroGPU illegal duration". Community reports: 120 s safe for free users, 300 s rejected for free users in 2026, longer works for PRO; InstantSplat's Space declares 600 s | [S1][S6][S7][S25] |
| Daily quota | Unauthenticated **2 min**, free account **5 min**, PRO **40 min** (extensible), Team 40, Enterprise 60. The window resets 24 h after first use. The **requested** duration is checked against remaining quota, not the actual run time | [S1][S6] |
| Beyond quota | PRO/Team/Enterprise only: **$1 per 10 GPU-minutes** from prepaid credits (= $6/h for half a card, $12/h for `xlarge`) | [S1] |
| Queue | Node-level; remaining quota and a shorter declared duration raise priority | [S1][S6] |
| Frameworks | **Gradio SDK only**; Python 3.10.13 or 3.12.12; PyTorch 2.8-2.13; no `torch.compile` (ahead-of-time compilation is supported); models go to `cuda` at import time | [S1] |
| CUDA extensions | **Spaces cannot compile at boot.** Prebuilt sm_120 wheels exist in a community dataset (xformers, nvdiffrast, Inria `diff_gaussian_rasterization`, pytorch3d, flash-attn), **not gsplat**; gsplat's own wheel index stops at older torch/CUDA | [S8][S9] |
| Host CPU, RAM, disk | Listed as "dynamic"; not documented | [S2] |

**What this means for us.** Long work must be cut into calls of about 60-120 s, each sized by a duration callable. Quota
is per visitor, so the free tier is limited by the visitor's 5 minutes, not by our bill. Whether a background thread
that outlives the visitor's request can still spend that visitor's quota is **unverified**; assume the tab must stay
open, and make every unit resumable.

### 1.3 Jobs (the batch product)

| Item | What the docs say | Source |
|---|---|---|
| Who | Anyone with a **positive credit balance**; PRO's monthly included credits count | [S10] |
| Flavours (per hour) | cpu-basic $0.01, cpu-upgrade (8 vCPU/32 GB) $0.03, cpu-xl (16 vCPU/124 GB/1 TB) $1.00, cpu-performance (32 vCPU/256 GB) $1.90; t4 $0.40-0.60, l4x1 $0.80, a10g $1.00-1.50, **l40sx1 $1.80**, a100-large $2.50, **rtx-pro-6000 $2.75**, h200 $5.00, multi-GPU up to $40 | [S10][S11] |
| Billing | **Per minute** while Starting or Running; failing Jobs are suspended | [S10] |
| Timeout | **Default 30 min**; any value can be set (`timeout="3h"`, `"1d"`); no documented maximum | [S10][S11] |
| Images | Any Docker image, including `hf.co/spaces/<id>`; `uv run` scripts with an inline launch config | [S11] |
| Storage | Mount model/dataset repos or **buckets as volumes** (read-write by default) | [S11] |
| Triggers | `run_job` from Python/HTTP, schedules (cron), and **webhooks on a bucket** that start a Job per upload (limit 1,000 triggers per 24 h) | [S11][S12] |
| On behalf of a user | Spaces can request the OAuth **`jobs`** scope ("Run jobs"). Jobs bill the namespace that runs them by default, so a Job started with the visitor's token bills the visitor. Both halves are documented; the combination is **unverified** | [S13][S10] |

### 1.4 Inference Endpoints and Inference Providers: not a fit

- **Inference Endpoints** are dedicated, autoscaling servers with custom containers at the same hourly GPU prices
  (L40S $1.80, A100 $2.50, RTX PRO 6000 $2.75, H200 $5.00, B200 $9.25) [S2][S14]. They suit request/response serving.
  Our work is a long batch per drive, which Jobs do with less machinery and no idle server.
- **Inference Providers** are serverless calls to catalogue models (LLMs, image and video generation, embeddings,
  classic tasks) through partners [S15]. None of our stages is in that catalogue. Not relevant.

### 1.5 Uploads and storage

| Path | Practical ceiling | Source |
|---|---|---|
| Gradio upload into the Space | `max_file_size` defaults to **no limit**; files land in the Gradio cache on the Space's 50 GB ephemeral disk and vanish on restart. **Any file in the Gradio cache is reachable by URL by any user of the app**, so the cache is wrong for private dashcam footage | [S26][S27] |
| Browser → user's bucket or dataset | `@huggingface/hub` uploads to buckets and repos with a token (Xet chunked, deduplicated). Python uploads are documented as resumable by re-running; resumability of the **browser** path is unverified | [S28][S29] |
| Repos | No per-repo size limit; files under 200 GB recommended, 500 GB hard; under 10k files per folder; Git history keeps deleted files until squashed (quota updates within 36 h) | [S19] |
| Storage Buckets | S3-like, **non-versioned, mutable**; **deletion is immediate and permanent**; public or private; webhooks and live change feeds; S3-compatible API | [S18] |
| Storage price | Free users 100 GB private storage; PRO 1 TB private; beyond that $18/TB/month private, $12/TB/month public; egress and CDN included | [S19][S20] |
| Reading from a browser | Checked here with one 100-byte request each: public **dataset** and public **bucket** `resolve` URLs return **206 Partial Content with CORS**; a private file without a token returns 401. Private reads with an `Authorization` header from the viewer: unverified | [measured here] |

A Tesla minute is six MP4 files (four on HW3/AI3, sometimes seven), "typically tens of megabytes" per camera [S32]. Our
runner reads H.264 with SEI telemetry; Tesla puts SEI in H.264/H.265 streams on firmware 2025.44.25 or later, HW3 and
above [S35][S37]. So **a one-minute, six-camera save is roughly 0.2-0.5 GB** [estimate].

---

## 2. Identity, privacy and storage

1. **Sign in with Hugging Face.** Request only what is needed: `openid profile` (always), **`contribute-repos`**
   ("create repositories and access those created by this app", nothing else) and **`jobs`** for the full tier [S13].
   Avoid `write-repos` and `manage-repos`, which reach all of a user's repositories. Buckets are documented as "a repo
   type" [S18], so `contribute-repos` probably covers a bucket the app creates; **unverified**. Tokens last 8 h by
   default, 30 days at most (`hf_oauth_expiration_minutes`) [S13]. In Gradio, `gr.LoginButton` plus a `gr.OAuthToken`
   parameter gives the token; a login button does not by itself block anonymous use [S38].
2. **Results go to the user's own private bucket**, `<user>/reality-splat`, created by the app. Our account never
   stores a user's footage. A Job started for the user mounts that bucket as a volume, so no token has to be passed into
   the Job [S11].
3. **Heavy use = duplicate or self-host.** The README offers "Duplicate this Space" (private by default, runs on the
   duplicator's ZeroGPU or paid GPU) and the Docker image for any CUDA box [S3].
4. **What is logged.** Space stdout and stderr are visible to the Space owner and anyone with write access; Gradio
   keeps uploads and outputs in its cache, reachable by URL [S27]. Hugging Face retains information "as long as
   necessary" and may access private content for security or legal reasons [S39]. Log stage names, durations and hashes
   only, never file names, coordinates or frames, and turn Gradio analytics off.
5. **Deletion we can guarantee.** Raw uploads in a bucket are deleted by our code when processing ends (or after a
   user-chosen 7 days), and bucket deletion is permanent [S18]. Anonymous examples are removed from the Space cache with
   `delete_cache=(3600, 3600)` [S26]. We **cannot** guarantee deletion from Hugging Face's own backups; say so in the
   privacy note. Never put footage in a Git-backed dataset unless the user publishes it on purpose.
6. **Faces and number plates.** Blur both **before** reconstruction, so the splats, atlas and reference photos never
   hold them. **EgoBlur Gen 2** (Meta) face and plate models are Apache-2.0, Faster R-CNN, about 400 MB and 104 M
   parameters each [S40][S41]. They were tuned on head-mounted cameras; recall on Tesla footage is unmeasured. Publishing
   a scene publicly should also require trimming the first and last few hundred metres and dropping absolute lat/lon from
   `drive.json`, since a drive often starts at home.

---

## 3. Models on the Hub, stage by stage

Licence lines are quoted from the model card or official README. Sizes are the Hub repository's stored bytes (several
repos hold more than one checkpoint). "ZeroGPU" means it plausibly fits a `large` (48 GB) call with plain PyTorch;
nothing here was run on ZeroGPU.

### 3.1 Depth and multi-view geometry

| Model (Hub id) | Licence line | Size | Our measurement | ZeroGPU |
|---|---|---|---|---|
| `depth-anything/DA3-LARGE-1.1` | Card: "license: apache-2.0", "**License** Apache 2.0". **Official README table: "CC BY-NC 4.0"**; the `DA3-LARGE` card (no 1.1): CC BY-NC 4.0 [S16][S17] | 1.6 GB, 0.35 B | Production front/rear: shape error 3.9 %, temporal 1.8 / 3.7 % (1 / 4 m), 0.041 s/image on MPS; 6.1 GB | Yes |
| `depth-anything/DA3-BASE`, `DA3-SMALL`, `DA3METRIC-LARGE` | "Apache 2.0" (README table and cards) [S17] | 0.5 / 0.1 / 1.3 GB | DA3-BASE posed: front temporal 4.8 / 7.4 %, scale 1.16 | Yes |
| `depth-anything/DA3NESTED-GIANT-LARGE-1.1`, `DA3-GIANT-1.1` | "⚠️ Non-commercial use only due to CC BY-NC 4.0 license" | 6.8 / 5.4 GB | Not better than posed Large here | Yes |
| `facebook/map-anything` | "license: cc-by-nc-4.0" [S16] | 12 GB, 1.2 B | Production side cameras: repeater shape 5.5 % vs 14.8 % for DA3-Large | Yes (9.1 GB at 518 px) |
| `facebook/map-anything-apache` | "license: apache-2.0" | 16.6 GB, 1.2 B | Pillar temporal 3.7 % (vs 3.3 % NC); front 7.7 % at 4 m | Yes |
| `facebook/VGGT-1B` / `VGGT-1B-Commercial` | CC BY-NC 4.0 / custom "vggt-aup-license", gated (manual approval) [S16] | 20 / 10 GB | Not tested (no gain expected over posed DA3) | Yes |
| `facebook/VGGT-Omega` | "fair-noncommercial-research-license", gated | 24 GB | Not tested | Yes (Meta runs a Space) |
| `naver/MASt3R_…_metric`, `DUSt3R_…` | "The code is distributed under the CC BY-NC-SA 4.0 License… make sure to agree to the license of all the public training datasets" | 2.3-2.8 GB | Superseded | Yes |
| `yyfz233/Pi3X` | cc-by-nc-4.0 | 5.4 GB | Not tested | Yes |
| `Ruicheng/moge-2-vitl-normal` | mit | 2.7 GB | Road scale 1.69x off on the front | Yes |

Feed-forward driving or multi-camera splat models with released weights (from `research/02`, rechecked on the Hub):
`chenchenshi/UniSplat` (apache-2.0 card, 1.1 B, Waymo-trained), `xiaomi-research/dggt` (cc-by-nc-4.0),
`RainyNight/DVGT` (apache-2.0, 13.9 GB) and `DVGT-2` (no licence tag), `lhjiang/anysplat` (MIT, used by several ZeroGPU
Spaces), DrivingForward (MIT, nuScenes, weights on Google Drive). FreeForward4D (2026-11-18) has no code yet [S42].
**Waymo-trained weights very likely inherit the Waymo Open Dataset's non-commercial terms** (not verified). None has
been evaluated on a fisheye Tesla rig. They are candidates for a future "instant" tier, not for the default.

### 3.2 Segmentation, vehicle classification, tracking, anonymisation

| Stage | Model | Licence | Notes |
|---|---|---|---|
| Panoptic masks (current) | `facebook/mask2former-swin-large-cityscapes-panoptic` | Card "license: other"; code MIT [S43]; **Cityscapes training data is non-commercial** (PORT.md) | 2.6 GB; 0.84 s per six-camera keyframe on the A40 [measured] |
| Panoptic, MIT code | `tue-mps/cityscapes_semantic_eomt_large_1024`, `shi-labs/oneformer_cityscapes_swin_large` | mit | Same Cityscapes-data caveat; unmeasured |
| Instances, permissive data | `Roboflow/rf-detr-seg-preview` (instances), `roboflow/rf-detr-base` (boxes) | apache-2.0 (cards and repo [S44]) | COCO classes include car, truck, bus, motorcycle, bicycle, person; no road/sky "stuff", so pair with DA3METRIC's Apache sky output. Unmeasured |
| Vehicle class check (replaces Apple Vision) | `google/siglip2-so400m-patch14-384` zero-shot on crops (or the base model) | apache-2.0 | An independent second opinion like Apple's classifier. Needs the agreement test PORT.md asks for: the 570 Apple-labelled crops of 224-254 s are a ready benchmark |
| Tracking | Our own IoU + 3D association (no model); optional `facebook/sam2.1-hiera-*` mask propagation | apache-2.0 | Avoid `facebook/cotracker3` (cc-by-nc-4.0) and `facebook/sam3` (custom licence, gated) |
| Faces and plates | EgoBlur Gen 2 | "Both the Gen 2 face and license plate models are licensed under Apache 2.0" [S41] | Weights sit behind an email form on projectaria.com, not on the Hub; re-hosting needs a read of its model licence agreement first |

### 3.3 Splat rasteriser and optimiser

| Library | Licence | Fit |
|---|---|---|
| **gsplat** (`nerfstudio-project/gsplat`) | Apache-2.0 [S45] | CUDA training, with 3DGUT for fisheye and rolling shutter; PyPI is 1.5.3 (July 2025), 1.6.0 on main. No sm_120 wheel for ZeroGPU, so either build and host our own wheel or train only in Jobs (a Docker build can compile) |
| **3dgrut** (NVIDIA) | Apache-2.0 [S46] | Ray-traced and unscented splats; heavier; the reference for fisheye |
| **Brush** | Apache-2.0 [S47] | Rust/wgpu, runs on Metal and CUDA; the Mac path for contributors |
| Inria `diff-gaussian-rasterization` | Inria licence, non-commercial (research/02) | Has a ZeroGPU wheel [S8] but **do not use**: licence, and no gradients for our extra losses |
| Our `pipeline/livemap/raster.cpp` | ours | CPU C ABI; builds on Linux [measured, PORT.md]; enough for the chase-clip renders |

### 3.4 Generative fixer

| Model | Licence line | Speed | Notes |
|---|---|---|---|
| `nvidia/difix_ref` (default today) | "**This model is ready for research and development/non-commercial use only.** The use of the model and code is governed by the NVIDIA License. Additional Information: LICENSE.md · stabilityai/sd-turbo" [S21]. SD-Turbo is under the **Stability AI Community License** ("research, non-commercial, and limited commercial uses") [S22] | A40 fp16 batch 1: **2.2 s per 1472x960 frame, 0.82 s at 1024x576**, 11-15 GB [measured]; paper: 76 ms at 576x1024 on an A100 | Render + one reference photo. Measured on our scenes (genfill/REPORT.md). Fits a ZeroGPU `large` call |
| `nvidia/difix` | Same as above | Similar | No reference: invents paint (precision 0.32 vs 0.54); do not use |
| `nvidia/Fixer` | "**This model is ready for commercial/non-commercial use.** Your use of the model is governed by the NVIDIA Open Model License Agreement." Code Apache-2.0 [S23] | Card: **26.5 ms H100, 50.6 ms A100, 87.8 ms L40** at 576x1024 | Driving domain (intended domain "Advanced Driver Assistance Systems"), render only, fixed 576x1024. The card now describes a linear-attention DiT with a DC-AE; our report recorded Cosmos-Predict2 0.6B in a CUDA container. Never run on our renders |
| `nvidia/ArtiFixer` | Weights: "governed by the NVIDIA License" (a one-way **non-commercial** licence, linked on the card); code Apache-2.0 [S24] | Few-step causal video; 1.3 B and 14 B; 276 GB repo | The temporal-consistency upgrade; Docker on Ampere, Hopper or Blackwell. Jobs only |
| `zfkuang/StreetNVS` | Code Apache-2.0; preprocessing parts research/non-profit; Waymo-trained | 64 GB or more, H100 | Offline experiment only (research/05) |
| Speed-ups | `madebyollin/taesd` (MIT, 2.4 M parameters) | — | Difix's decoder is a LoRA'd VAE with encoder skips, so a tiny VAE drops those skips: expect softer detail; unmeasured. Measured lever: 1024x576 is 2.7x faster than native on the A40. ZeroGPU's ahead-of-time compilation is the other documented lever [S1] |

### 3.5 Sky

No extra model is needed: sky masks come from Mask2Former (or DA3METRIC's Apache sky output) and the panorama is built
from the photos. Generated sky is already handled, and labelled predicted, by the fixer.

### 3.6 Recommendation per stage

| Stage | Default for the non-commercial hosted page | Permissive alternative | What the permissive path loses |
|---|---|---|---|
| Ingest + telemetry | Our PyAV/SEI code | Same | Nothing. Tesla's `teslamotors/dashcam` repo has **no licence file**: reimplement the schema, do not copy its code |
| Calibration, poses, road, atlas | Our Ceres + C++ code | Same (Ceres is BSD) | Nothing |
| Depth, front and rear | DA3-LARGE-1.1 (treated as CC BY-NC) | DA3-BASE (Apache) | Front temporal error about 2x (3.7 → 7.4 % at 4 m), scale bias 4 % → 16 % [measured] |
| Depth, four side cameras | MapAnything (CC BY-NC) | map-anything-apache | Pillar about equal; front 5.8 → 7.7 % at 4 m [measured] |
| Panoptic masks | Mask2Former Swin-L Cityscapes | RF-DETR-Seg (COCO) + DA3METRIC sky | Road/sidewalk classes and Cityscapes' street tuning; unmeasured |
| Vehicle class check | SigLIP 2 zero-shot (Apache) | Same | Unknown until the 570-crop agreement test |
| Splat fit | gsplat (with 3DGUT) on CUDA | Same | Nothing; porting cost only |
| Fixer | Difix-ref at t199 (NC) | NVIDIA Fixer at 576x1024 | Reference conditioning (more invention, as with plain Difix) and native resolution; but 10-25x faster on paper. Unmeasured on our renders |
| Anonymisation | EgoBlur Gen 2 | Same | — |

---

## 4. How others have done it (Spaces checked 2026-11-22 through the Hub API and their `app.py`)

| Space | Upload | GPU time | Viewer | Results |
|---|---|---|---|---|
| `facebook/vggt`, `facebook/map-anything`, `depth-anything/depth-anything-3` | Gradio video or images; video cut to **1 frame per second** on upload; a per-request temp dir | One `@spaces.GPU(duration=120)` call | `gr.Model3D` GLB point cloud | Temp dir, `shutil.rmtree` on clear; nothing kept |
| `hugging-apps/vggt-omega` | Same, plus a **hard cap of 100 frames** with a message | **Dynamic duration** from the frame count | GLB | Temp |
| `kairunwen/InstantSplat` | Image set | Two GPU functions, **150 s and 600 s** | `gr.Model3D` on a splat PLY, plus a rendered orbit video and a PLY download | Temp |
| `naver/MASt3R` | Images | 180 s | GLB | Gradio cache with `delete_cache` every 2 h |
| `notaneimu/ml-sharp-3d-viewer-zerogpu` (Apple SHARP) | One image | One short call | **SuperSplat viewer** served through `gr.set_static_paths` with a per-generation PLY | Per-generation files |
| `tencent/HunyuanWorld-Mirror` | Images, or video sampled at an interval | 120 s | Splats exported as PLY and `.splat` | Temp |

**Copy:** per-request directories; a visible input cap with the reason; durations computed from the input; a real splat
viewer in the page instead of a mesh widget; downloadable result files; cached examples so anonymous visitors see
something without spending quota. **Avoid:** one monolithic long GPU call (it fails for free users and holds the queue);
server-side rasterisers that need CUDA extensions on ZeroGPU [S9]; results kept only in the Gradio cache (lost on
restart and URL-reachable by other users); `gr.Model3D` for scenes of millions of splats.

None of these handles minutes of six-camera footage, resumable multi-step work or user-owned storage. That part is new.

---

## 5. Proposed design

### 5.1 The visitor's path

1. **Land** on the Space (Gradio, ZeroGPU). A gallery of example drives opens straight into the viewer. A box explains
   what footage works (HW3 or HW4, firmware 2025.44.25 or later, driving rather than parked) and that **encrypted clips
   must be decrypted first** (padlock in the car's Dashcam app, or dashcam.tesla.com).
2. **Sign in with Hugging Face** (scopes above). The app creates the private bucket `<user>/reality-splat` if missing.
3. **Drop a SavedClips folder.** In the browser: read the six MP4s, parse SEI (our own JS port of the schema), reject
   encrypted or SEI-less files, identify the camera set from file names and resolutions, show the route and the moving
   seconds, and let the user pick a window within their tier's limit. **Cut the window without re-encoding**, keeping the
   preceding keyframe's GOP and the SEI clock (the boundary issue noted in PORT.md), and upload only that to
   `raw/<drive-id>/` in the user's bucket.
4. **Choose a tier.** Quick runs in the Space on ZeroGPU. Full starts a Job on the user's account, after showing a cost
   estimate and a hard timeout, or tells free users to duplicate the Space or run the Docker image.
5. **Watch progress.** A per-stage list driven by the receipts in the bucket. Closing the tab pauses the quick tier and
   reopening resumes from the last receipt. Full-tier Jobs run without the tab.
6. **Orbit the world.** The viewer opens at the first published window while later windows are still running, as the
   runner already does locally. Buttons: download the scene, delete everything, publish publicly (a dataset repo with a
   card, after blur, trimming and coordinate removal).

### 5.2 Where each stage runs

| Stage | Quick tier | Full tier |
|---|---|---|
| Validation, SEI, window choice, cutting, upload | Browser | Browser |
| Ingest (decode to JPEG), blur | Space CPU; blur on ZeroGPU | Job |
| Rig check (`--rig verify`), poses, road | Space CPU | Job |
| Preview masks + class check | **ZeroGPU calls** | Job |
| Photometry, ground, atlas, movers | Space CPU | Job |
| Production depth (DA3 + MapAnything), removal, teachers | — | Job (GPU) |
| Splat fit (gsplat) | — | Job (GPU) |
| Chase-clip render and fix | — (one optional 4 s clip on ZeroGPU for PRO) | Job (GPU) |
| Publication | The Space writes `scenes/<drive-id>/` to the bucket | The Job writes to the mounted bucket |
| Viewer | Static Space (free) reading the bucket | Same |

**One Job image for everything heavy:** the Docker image built from the repo, with CUDA 12.8 and torch 2.9.1 (the
versions measured on the A40), Ceres built in, gsplat compiled for sm_86 to sm_120, and models fetched at start from
pinned revisions. Flavour: `l40sx1` ($1.80/h, 48 GB) by default, `rtx-pro-6000` ($2.75/h, 96 GB) for long clip runs.

### 5.3 Units of work and resume

- **Unit = one 4 s window of moving time** (the runner's clip chunk), grouped into the runner's 24 s fit windows with
  3 s overlap for depth and fitting. Each unit has a receipt (input hashes, output hashes, producer version), as
  `pipeline/run` already writes, stored in `work/<drive-id>/receipts/` in the bucket.
- **ZeroGPU call sizing** [estimate from measured A40 rates; ZeroGPU speed unmeasured]: masks at about 0.84 s per
  six-camera keyframe → about 60 keyframes per 60 s call, about 5-6 calls per driving minute. Difix at 2.2 s per native
  frame → about 25 frames per call; a 4 s clip is 5 calls. Every call declares its duration from its unit size.
- **Resume:** the orchestrator skips units whose receipt matches, so a quota error, a closed tab or a Space restart costs
  at most one unit. A Job is one process for a whole drive, uses the same receipts and restarts with `attempts=2`.
- The bucket is working storage; Git repos are only for a deliberate public release.

### 5.4 Footage limits

| Visitor | What they may run | Why |
|---|---|---|
| Anonymous | Examples only | 2 GPU-minutes a day and no bucket to write to |
| Signed-in free | Quick tier, **30 s of moving footage a day** | About 2-3 GPU-minutes of masks [estimate] inside a 5-minute quota, leaving room for retries |
| PRO | Quick tier up to **3 min a day** in quota (more on credits); one 4 s generated clip; **full tier via Jobs, up to 5 min of driving per run**, with the cost cap shown before it starts | 40-minute quota; Jobs are cheaper than credits |
| Any user with credits | Full tier via Jobs, same 5-minute cap | Bills their own account |
| Duplicate or Docker | No limit | Their hardware, their bill |

Raw upload cap: 1 minute of six cameras (about 0.5 GB) for quick, 5 minutes (about 2.5 GB) for full [estimate].

### 5.5 Where every file lives, and for how long

| File | Location | Lifetime |
|---|---|---|
| Raw cut clips | `hf://buckets/<user>/reality-splat/raw/<drive-id>/` | Deleted when the run ends (default) or after 7 days (user's choice) |
| Frames, depth, masks, teachers, checkpoints | Space ephemeral disk, or Job disk (380-475 GB on L40S/RTX PRO 6000) | Gone with the process |
| Receipts and manifest | `…/work/<drive-id>/` | Until the drive is deleted |
| Published scene (RSB tiles at 33 B/splat, ground atlas, objects, MP4 clips, `drive.json`, `rig.json`, `map.json`) | `…/scenes/<drive-id>/` (private) | Until the user deletes it |
| Public share (optional) | Dataset repo `<user>/reality-splat-<drive-id>` with a card | User's choice |
| Anonymous examples | Our public dataset | Permanent, blurred, ours |
| Model weights | Hub, pinned revisions; Job disk or Space cache | Per process |
| Camera presets | Public dataset `reality-splat/tesla-rigs` | Versioned |

**Published size** [estimate from measured scenes]: 2.9 M splats for 30 s of driving × 33 bytes ≈ 96 MB of tiles, plus
about 4 MB of atlas per 300 m (43 MB per 3.3 km) and a few MB per 4 s clip: **roughly 0.2-0.5 GB per driving minute**.
A free account's 100 GB of private storage holds a few hundred minutes. The PLY contract files (96 B/splat, 771 MiB for
the 30 s full release) stay in the Job and are not published.

### 5.6 What serves the viewer

The viewer is already a static three.js + Spark page. Host it as a **static Space** (`sdk: static`, free, with a build
step [S48]) that takes `?scene=<url>`:

- Public scenes: `https://huggingface.co/datasets/<user>/<repo>/resolve/main/...` or
  `https://huggingface.co/buckets/<user>/<bucket>/resolve/...`; both answered byte ranges with CORS in our check.
- Private scenes: the Gradio Space embeds the viewer and hands it the OAuth token for `Authorization` headers.
  **Unverified**; the fallback is a short-lived proxy route in the Gradio app.
- The current LAN host (`deploy/serve.ts`) stays for development.

### 5.7 Quick tier and full tier

| | Quick | Full |
|---|---|---|
| Content | Ground splats, photographic ground atlas, automatic mover tracks with class and paint (today's "preview") | Adds production depth, the standing-world v5 fit, mover removal, optional generated chase clips |
| Where | ZeroGPU + Space CPU | One Job |
| State of the code | Runs on the Studio; Linux needs the classifier swap and Linux runner plumbing | Also needs the gsplat port of `v4fit`/`v5fit` (MetalSplat today) |

### 5.8 Time and cost per minute of driving

Assumptions: "driving minute" = one minute of moving footage from six HW4 cameras. CPU stages take the Studio's measured
times (`pipeline/run/README.md`) and are assumed 1-2x slower on cloud vCPUs. GPU stages take the A40's measured rates
(`pipeline/cuda/README.md`). Half an RTX PRO 6000 is assumed to be 1-2x an A40 and a full one 1.5-2.5x, on paper only.
Everything not marked [measured] is an **unmeasured estimate**.

| Tier | Compute per driving minute | Wall time | Cost | Basis |
|---|---|---|---|---|
| Quick, GPU part (masks + class check) | 2-5 GPU-min | — | Free inside the ZeroGPU quota; $0.10-0.25 on a Job | A40 masks 4.19 min per driving minute [measured projection]; Studio 3.8 min [measured] |
| Quick, CPU part | 10-20 CPU-min | — | About $0 in the Space; $0.01 on cpu-upgrade, $0.30-0.60 on cpu-performance | Studio preview 13.6-14.2 min in total, of which about 10 min are CPU stages [measured] |
| **Quick total** | | **15-30 min** | **$0-0.60** | |
| Full geometry (depth, removal, teachers, fit, merge) | 25-80 GPU-min | 0.5-1.5 h | **$0.75-2.40** on L40S or RTX PRO 6000 | Studio 51.9 min per minute [measured]; CUDA fit not ported; A40 mixed-model depth at 8.9 s per keyframe [measured] does not project cleanly |
| Generated clip, per generated minute | Fixer 26-44 GPU-min at native size (10-16 at 1024x576) + render/composite 30-50 CPU-min | 0.5-1.5 h | **$1.50-4** | A40 fixer 66.1 / 24.6 min [measured]; Studio end to end 85.7-102 min [measured] |
| **Full, every second generated** | | **1.5-3 h** | **$2.50-7** | Sum of the two rows above |
| The same full work on ZeroGPU credits | | | 2-4x the Job price | $1 per 10 min for half a card vs $2.75/h for a whole card [S1][S10] |

The biggest assumption is that the CUDA ports of the MetalSplat fit and the runner's depth stage run at least as fast as
the Studio does today. The largest lever is the fixer: 1024x576 is 2.7x faster [measured], and NVIDIA's Fixer card
claims 26-88 ms per frame, which would cut the clip row by an order of magnitude if its quality holds on our renders.

### 5.9 Camera presets (one calibration per Tesla model)

- **Data:** public dataset `reality-splat/tesla-rigs`, one folder per `model-hardware-year` (for example
  `model3-hw4-2024`, `modely-hw4-2025`, `cybertruck-hw4` with seven cameras, `model3-hw3` with four): `rig.json`,
  `masks/*.png`, `provenance.json` (who, how, which validation drives, which runner version).
- **Contribution:** an owner runs `--rig refit` locally (it already needs at least 120 moving seconds and two 12-degree
  turns) and opens a pull request with the rig and the validation numbers only; footage stays private.
- **Verification on upload:** every run checks its preset with the existing compatibility test (pooled residual
  normalised to 1448 px, gross limit 5.36 px on our car) and records the score. A preset is marked **verified** after two
  other cars pass on held-out panels, and **suspect** if a supported gross mismatch reproduces. Body masks and mounting
  differ by model year and trim (refreshed Model 3/Y, S/X), so presets are keyed on camera set and file resolutions as
  well as model.
- **Unknown:** how far one car's rig transfers to another of the same model; we have one car.

### 5.10 The open-source repository

| Item | Proposal |
|---|---|
| Code licence | **Apache-2.0** for our code (patent grant; compatible with gsplat, Brush, DA3-Base, MapAnything-apache, SigLIP 2). A `NOTICE` and `MODEL_LICENSES.md` state that several default models are non-commercial and that the code licence grants nothing for any weights |
| Model manifest | `models.toml`: Hub id, pinned revision, licence id, the quoted licence line, `commercial = true/false`, stage |
| Download script | `rs-models fetch [--commercial-only]` prints each model's licence line and URL before downloading, refuses NC models in commercial mode and switches the pipeline to the permissive set |
| Container | `Dockerfile` (CUDA 12.8, torch 2.9.1, Ceres, gsplat for sm_86 to sm_120); the same image is the Job image and the self-hosting route |
| Space | `space/`: the Gradio app, ZeroGPU wrappers, OAuth, bucket I/O and Job launcher; `viewer/` builds the static Space |
| Data | One blurred, trimmed example drive under CC BY 4.0, owned by Ryan |
| Left out | `scratch/`, the reference frames from Tesla's promo, and the Sketchfab car model (CC BY 4.0, but a Tesla design; see 5.11) |

### 5.11 Naming and look

- **Do not use "Reality View"** or "Tesla Reality View": it is Tesla's FSD v15 feature name since 2026-11-21
  [S33][S34]. "Reality Splat" is close enough to invite confusion; pick a neutral name (for example *Dashsplat*,
  *Drivescape* or *Splatlane*; no trademark search done).
- Use "Tesla" only to describe compatibility ("works with Tesla dashcam footage"); no logos, nameplates or Tesla
  artwork. Tesla reserves its marks and site materials [S49].
- Replace the deliberate imitation of Tesla's screen: the Gotham-like HUD, the "Tesla-blue" ribbon, the Autopilot/FSD
  wording and the Model 3 Highland car. Keep the information (speed, gear, path ribbon, blinkers, drive state) in our own
  layout and colours, with a neutral car body (in the user's paint colour if they like).
- Keep labelling generated and predicted content, as the viewer already does, and say on the page that generated frames
  are not a record of what happened.

### 5.12 Build order (smallest useful release first)

1. **Hygiene (days):** new name; `models.toml` and the licence script; ask the Depth Anything 3 authors about the
   DA3-LARGE-1.1 licence; neutral HUD and car.
2. **Release 0, viewer only:** the static Space plus one example drive in a public dataset. Proves range-request
   streaming from the Hub and gives people something to orbit.
3. **Release 1, self-host:** the Linux runner with the SigLIP 2 class check and the gsplat fit, in the Docker image. It
   writes a scene folder; `hf upload` it to your own bucket and open it in the viewer. Owners with a GPU can use it at once.
4. **Release 2, quick tier on ZeroGPU:** browser validation and cutting, OAuth, the user's bucket, chunked mask calls,
   EgoBlur, 30 s free / 3 min PRO.
5. **Release 3, full tier on Jobs:** OAuth `jobs` scope, cost estimate and cap, bucket volume, receipts; plus the
   Duplicate path.
6. **Release 4, generated clips and presets:** Difix-ref on the hosted page, Fixer in the permissive build, the rig
   preset dataset and its verification.

---

## 6. Risks and unknowns, ranked

1. **Encrypted clips (high, certain).** Encryption is on by default since 2026.20 in most markets, and decryption needs
   the owner's Tesla sign-in [S30][S31]. Every new user meets this. Mitigation: clear instructions and file detection;
   never handle Tesla credentials or keys.
2. **ZeroGPU behaviour for multi-call jobs (high, unknown).** The per-call cap is unpublished and has moved; whether
   quota and GPU access survive a closed tab is unverified; the Space's one shared container must run 10-20 minutes of
   CPU stages per driving minute for every concurrent user. Measure with a private test Space before promising limits.
3. **Linux port incomplete (high, known).** The Apple Vision classifier and the MetalSplat fit have no validated
   replacement, so the full tier does not exist on CUDA yet.
4. **Licences (medium-high).** DA3-LARGE-1.1's conflicting labels; MapAnything, Difix (plus SD-Turbo's community
   licence) and Cityscapes-trained masks are non-commercial; outputs of NC models end up in user scenes. A
   non-commercial page is fine; a commercial fork must switch to the permissive set and lose some quality.
5. **Privacy (medium-high).** Third parties' faces and plates (EgoBlur recall on Tesla cameras unmeasured), home
   locations in GPS, Gradio's URL-reachable cache, and public sharing. Blur before reconstruction, never keep footage in
   the cache, strip coordinates on publish.
6. **Cost and abuse (medium).** Jobs need user credits, which most free users lack; ZeroGPU credits cost 2-4x a Job.
   Show prices before starting and cap timeouts.
7. **Generalisation (medium, unmeasured).** One car, one dawn drive. Other trims, HW3's four cameras, Cybertruck's
   seven, night and rain are untested; presets may often fail verification.
8. **Browser upload path (medium, unverified).** Large resumable uploads to buckets from the browser,
   `contribute-repos` covering buckets, and private scene reads with a token.
9. **Trademark and trade dress (low-medium).** Handled by 5.11 if done before launch.
10. **Platform drift (low-medium).** ZeroGPU hardware, quotas and plan rules changed within 2026 (H200 to RTX PRO 6000;
    a paid plan for compute Spaces). Re-read [S1]-[S4] before each release.

---

## Sources (all accessed 2026-11-22)

- [S1] ZeroGPU docs. https://huggingface.co/docs/hub/spaces-zerogpu
- [S2] Hugging Face pricing (PRO, Spaces, Endpoints, storage). https://huggingface.co/pricing
- [S3] Spaces overview (paid plan, duplication, hardware, visibility). https://huggingface.co/docs/hub/spaces-overview
- [S4] Using GPU Spaces (tiers, billing, sleep, replicas). https://huggingface.co/docs/hub/spaces-gpus
- [S5] Disk usage on Spaces (ephemeral disk, bucket volumes). https://huggingface.co/docs/hub/spaces-storage
- [S6] Hugging Face skills, "How ZeroGPU duration and quota are checked". https://github.com/huggingface/skills/blob/main/skills/huggingface-zerogpu/references/how-quota-works.md
- [S7] Forum, "ZeroGPU illegal duration (300s)" (2025-02), and ACE-Step Space discussion #11 (2026-02). https://discuss.huggingface.co/t/zerogpu-illegal-duration-the-requested-gpu-duration-300s-is-larger-than-the-maximum-allowed/140849 ; https://huggingface.co/spaces/ACE-Step/Ace-Step-v1.5/discussions/11
- [S8] ZeroGPU Blackwell (sm_120) wheels dataset. https://huggingface.co/datasets/multimodalart/zerogpu-blackwell-wheels
- [S9] Spaces requirements and 3D/gsplat notes. https://raw.githubusercontent.com/huggingface/skills/main/skills/huggingface-spaces/references/requirements.md ; https://github.com/PracticalSwan/agent-skills/blob/main/huggingface-spaces/references/3d-gsplat.md
- [S10] Jobs pricing and billing. https://huggingface.co/docs/hub/jobs-pricing
- [S11] huggingface_hub Jobs guide (flavours, timeout, volumes, schedules). https://huggingface.co/docs/huggingface_hub/guides/jobs
- [S12] Webhook guide: process new files in a bucket with Jobs. https://huggingface.co/docs/hub/webhooks-guide-bucket-jobs
- [S13] Sign in with HF for Spaces (scopes, expiry). https://huggingface.co/docs/hub/spaces-oauth
- [S14] Inference Endpoints. https://huggingface.co/docs/inference-endpoints/index
- [S15] Inference Providers. https://huggingface.co/docs/inference-providers/index
- [S16] Hub model API and cards (licence, gating, size) for every model id in section 3, for example https://huggingface.co/api/models/depth-anything/DA3-LARGE-1.1 and https://huggingface.co/depth-anything/DA3-LARGE-1.1/raw/main/README.md
- [S17] Depth Anything 3 README, model table with licences. https://github.com/ByteDance-Seed/Depth-Anything-3
- [S18] Storage Buckets. https://huggingface.co/docs/hub/storage-buckets ; access patterns https://huggingface.co/docs/hub/storage-buckets-access
- [S19] Storage limits. https://huggingface.co/docs/hub/storage-limits
- [S20] Storage pricing. https://huggingface.co/storage
- [S21] Difix-ref and Difix cards. https://huggingface.co/nvidia/difix_ref ; https://huggingface.co/nvidia/difix ; NVIDIA License in https://github.com/nv-tlabs/Difix3D
- [S22] SD-Turbo licence (Stability AI Community License). https://huggingface.co/stabilityai/sd-turbo/blob/main/LICENSE.md
- [S23] NVIDIA Fixer card and repo. https://huggingface.co/nvidia/Fixer ; https://github.com/nv-tlabs/Fixer
- [S24] NVIDIA ArtiFixer card and repo. https://huggingface.co/nvidia/ArtiFixer ; https://github.com/nv-tlabs/ArtiFixer
- [S25] Metadata and `app.py` of the Spaces in section 4, through https://huggingface.co/api/spaces/<id> and https://huggingface.co/spaces/<id>/raw/main/app.py
- [S26] Gradio Blocks reference (`max_file_size`, `delete_cache`). https://www.gradio.app/docs/gradio/blocks
- [S27] Gradio, Security and file access. https://www.gradio.app/guides/file-access
- [S28] huggingface.js Hub package (buckets, uploads). https://github.com/huggingface/huggingface.js/blob/main/packages/hub/README.md
- [S29] huggingface_hub upload guide (Xet, resumable). https://huggingface.co/docs/huggingface_hub/guides/upload
- [S30] Not a Tesla App, "Tesla Enables Dashcam Video Encryption in Update 2026.20" (2026-05-31). https://www.notateslaapp.com/news/4225/tesla-enables-dashcam-clip-encryption-in-update-202620
- [S31] QloudLTD, TeslaCam on-drive format, section 6. https://github.com/QloudLTD/teslacam-format
- [S32] Same repository, sections 2 and 4 (cameras per hardware generation, H.264 MP4, size per minute).
- [S33] Electrek, "Tesla launches FSD v15…" (2026-11-21). https://electrek.co/2026/11/21/tesla-fsd-v15-rollout-25x-safety-reality-view
- [S34] Drive Tesla Canada, "Tesla Reality View…" (2026-11-21). https://driveteslacanada.ca/news/tesla-reality-view/
- [S35] Tesla dashcam tools (SEI schema, firmware and hardware requirements). https://github.com/teslamotors/dashcam ; https://teslamotors.github.io/dashcam
- [S36] Spaces visibility levels (the "Space visibility" section of S3).
- [S37] heise, "Tesla: New dashcam feature displays detailed telemetry data in video" (2025-12-11). https://www.heise.de/en/news/Tesla-New-dashcam-feature-displays-detailed-telemetry-data-in-video-11112252.html
- [S38] Gradio, Sharing your app (OAuth with Hugging Face). https://www.gradio.app/guides/sharing-your-app
- [S39] Hugging Face privacy policy. https://huggingface.co/privacy
- [S40] EgoBlur repository and model details. https://github.com/facebookresearch/EgoBlur ; https://facebookresearch.github.io/projectaria_tools/docs/open_models/egoblur
- [S41] EgoBlur Gen 2 page and FAQ (licence). https://www.projectaria.com/tools/egoblur
- [S42] FreeForward4D, arXiv 2611.24007 (2026-11-18). https://arxiv.org/abs/2611.24007
- [S43] Mask2Former licence (MIT). https://github.com/facebookresearch/Mask2Former
- [S44] RF-DETR licence (Apache-2.0). https://github.com/roboflow/rf-detr
- [S45] gsplat README, licence and releases. https://github.com/nerfstudio-project/gsplat
- [S46] 3dgrut licence. https://github.com/nv-tlabs/3dgrut
- [S47] Brush licence. https://github.com/ArthurBrussee/brush
- [S48] Static HTML Spaces. https://huggingface.co/docs/hub/spaces-sdks-static
- [S49] Tesla legal: intellectual property and trademarks. https://www.tesla.com/legal/additional-resources
- Repo files used: `PLAN.md`, `pipeline/run/README.md`, `pipeline/depth/REPORT.md`, `pipeline/genfill/README.md`,
  `pipeline/genfill/REPORT.md`, `pipeline/stitch/REPORT-v5.md`, `viewer/README.md`, `research/02`, `research/04`,
  `research/05`, and in the `-cuda` worktree `pipeline/cuda/PORT.md` and `README.md`.
