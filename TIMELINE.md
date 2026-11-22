# Timeline

The repository was published with a fresh history, so this file keeps the original one: every commit of the working
repository, in order, with its time (US Eastern) and its message as written. Short hashes are the original
commits'. Reports, the film notes and some code comments cite them; they do not exist in this repository's
history. Commits marked **[NVIDIA]** were made on a separate branch for the port to NVIDIA GPUs, merged at the end.

The work started on the afternoon of Saturday 21 November 2026, shortly after Tesla began rolling out FSD v15 with
its new 3D view of the world around the car ([film/OUTLINE.md](film/OUTLINE.md) has the context). Dead ends are
included: several of them became parts of what worked.

## Saturday 21 November

### Research, calibration, the first ground and the first viewer

| Time | Commit | Message |
|---|---|---|
| 15:45 | `4ce6433` | Baseline: research reports, plan and ingest |
| 15:52 | `6913bc0` | Add rig calibration, drive export and live-map design |
| 15:59 | `f9f7532` | Revise live-map design and viewer contract after two reviews |
| 16:03 | `77c8a3b` | Add shared rig geometry module and first whole-drive trajectories |
| 16:16 | `d0e104c` | Measure attitude against the road and add the bird's-eye mosaic tool |
| 16:25 | `67b67c1` | Build the ground layer as time-stamped splats for the whole drive |
| 16:38 | `e30ee59` | Ground layer v2: sharper live ground, longer reach, wider coverage |
| 16:43 | `3679be9` | Add a stable host for the viewer on anomastudio |
| 16:51 | `01b6a94` | Add the six-camera strip renderer for side-by-side demo videos |
| 16:53 | `e7f92d6` | Add the Reality View viewer |
| 16:56 | `c7ba429` | Plan: state the spatial, historical and temporal reasoning the map should do |
| 17:00 | `6967e30` | Plan: moving objects are tracked rigid models, not per-frame splats |
| 17:02 | `865b267` | Add the first static object layer from depth (85-140 s) |

### First worlds, depth, movers and the first experiments

| Time | Commit | Message |
|---|---|---|
| 17:06 | `866041a` | Add the photometric optimiser for the stitched map (first pass) |
| 17:20 | `83c0eca` | Live world v4: draw everything the cameras see, newest view wins |
| 17:22 | `e8e1ada` | Viewer realism pass, moving objects and predicted layers |
| 17:22 | `c7f10a6` | Research: how Tesla renders Reality View, and our diffusion route |
| 17:35 | `dd8ffd1` | Ground live-v5 roadside profiles and check them from the viewer chase pose |
| 17:50 | `d083540` | Viewer: hue-safe grade, Tesla-matched chase pose, Generative mode |
| 17:55 | `be3b3f2` | Add depth and mask stage with a model bake-off on the Mac |
| 18:02 | `350fd32` | Optimiser v2: dense initialisation, ground frozen, judged from the real chase pose |
| 18:06 | `4e78cca` | Plan: two update rates, slow detailed world and fast light movers |
| 18:07 | `45470a7` | Add the image-based chase view: photographs chosen by viewpoint, as layers |
| 18:14 | `3364230` | Optimiser v3: opaque tree crowns, grass under the trees, supported poles |
| 18:23 | `36515c3` | Diffusion route: StreetNVS running on the Mac, first generated chase clip |
| 18:28 | `9474677` | Add moving objects as tracked splat models |
| 18:30 | `32c10ae` | Viewer: photographic layers that splat scenes can borrow |

### Dead ends turn into teachers

| Time | Commit | Message |
|---|---|---|
| 18:46 | `e8eecda` | Movers v2: online identity, jump gating, per-object shape and texture |
| 18:53 | `844ad5d` | Viewer: photographed sky at its real brightness, cut-out sanity filter, photo preset |
| 19:04 | `0af7e16` | Re-projected photograph chase view, first version (fails the after-image gate) |
| 19:04 | `dc1f413` | Stop tracking large proof videos |
| 19:19 | `a4f6a06` | Re-projection becomes a teacher for the splats, not a display |
| 19:22 | `cfd8398` | Movers v3: unseeded live tracking matches the reviewed baseline, removal masks |
| 19:30 | `09f0703` | Diffusion route round 2: dense conditioning removes inventions, no net gain |
| 19:40 | `f6b7e9a` | Depth and masks for the traffic windows, 2 Hz stopped keyframes, timing |
| 19:46 | `1ac344c` | Optimiser v4: the splat scene learns the chase view from teacher photographs |
| 19:51 | `95effba` | Teacher views for the traffic stretches (286-330 s and 190-215 s) |

### The standing world, the road atlas and the image model

| Time | Commit | Message |
|---|---|---|
| 20:01 | `b3057f7` | Publish v5 standing-world splats across three drive slices |
| 20:11 | `dfc7a94` | Movers attached to the optimised scene across three slices |
| 20:12 | `1eb949e` | Add the ground texture atlas: the road as a sharp texture on a mesh |
| 20:18 | `9493eba` | Add the generative view fixer (Difix-ref on Apple Silicon) and its evaluation |
| 20:35 | `d19b021` | Viewer: textured ground, blank-frame fix, mover shadows and occlusion |
| 20:40 | `72cb278` | Ground atlas v5: joins, verge edges and mover check; attached to the v5 scenes |
| 21:14 | `5350095` | Ground atlas v6: continuous ground out to 40 m beside the road |
| 21:31 | `ec01f28` | Keep the published viewer running across Studio reboots |

## Overnight into Sunday 22 November

### One command, generated clips and the whole drive

| Time | Commit | Message |
|---|---|---|
| 22:28 | `2e87073` | Viewer: scene-lit car, overcast shadows, Generative mode hand-over, matched grade |
| 23:36 | `a522a80` | Add the one-command drive pipeline (preview tier verified on fresh footage) |
| 23:47 | `9a3d01a` | Generated chase clips: every frame repaired, stabilised, hole-free, in the viewer |
| 00:26 | `c9cf385` | Viewer: clean vehicle models, automatic Generative mode, one neutral grade |
| 02:41 | `d054095` | Generated clips for 93-140 s, 190-215 s and 296-330 s, with cleaned input |
| 02:44 | `51b623d` | Ground atlas: the three bars at the red light are real paint |
| 02:48 | `4aba1fc` | Drive pipeline: full-quality tier verified end to end on fresh footage |
| 04:15 | `3b7a42b` | Generated clips: repair the worst seconds (near-camera objects, posts, floating trees) |
| 06:34 | `fe29f83` | Drive pipeline: whole-drive runs, clips for every moving window, v5 fit recipe |

## Sunday 22 November

### Hosting study, the film, and the start of the NVIDIA port

| Time | Commit | Message |
|---|---|---|
| 08:50 | `e47554e` | Add project AGENTS.md: keep basebook free, run heavy work on the Studio |
| 08:52 | `72b2763` | **[NVIDIA]** CUDA branch: device plumbing and first NVIDIA benchmarks (A40) |
| 09:05 | `babb360` | Research: hosting the pipeline on Hugging Face |
| 09:14 | `b1d5c7b` | Journey film source, capture tooling for the viewer, and the film handoff |
| 09:27 | `606a479` | Film: on-camera script v1 for the YouTube video |
| 09:29 | `8b6a45a` | Film: replace the on-camera script with a talking-point outline |
| 09:32 | `867d69a` | Journey film: step-1 cut delivered, state recorded for handoff |
| 09:38 | `48482d2` | **[NVIDIA]** CUDA: profile of the fixer and depth on A40 and L40S |
| 09:42 | `058a443` | Film: CLEAN=1 renders the film with no lettering |

### The NVIDIA port

| Time | Commit | Message |
|---|---|---|
| 10:01 | `2071911` | **[NVIDIA]** CUDA: render-only decode and NVIDIA Fixer measured against Difix-ref |
| 10:53 | `e2ccb0c` | **[NVIDIA]** Movers: portable vehicle classifier (SigLIP 2) beside Apple Vision |
| 11:01 | `ce7da46` | **[NVIDIA]** Stitch: CUDA backend for the splat fit, at parity with Metal |
| 13:50 | `81b0b79` | **[NVIDIA]** CUDA: the whole pipeline runs end to end on a rented NVIDIA GPU |

### Sky experiment and the release branch

| Time | Commit | Message |
|---|---|---|
| 15:25 | `8ca839a` | Environment map experiment: real sky from six cameras, diffusion-filled above |
| 15:35 | `659c68e` | Merge branch 'cuda' into release |

After this the tree was prepared for publication (licences, this file, the README, the naming notes, removal of
Tesla's reference images and a privacy pass) and published as a single commit.
