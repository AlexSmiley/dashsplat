# dashsplat on NVIDIA

## On your machine

The rest of this README is the record of the port as the author ran it: inputs prepared on his Mac Studio
(`anomastudio`, reached with his `ferry` and `pssh` tools), rented RunPod GPUs driven by a guarded controller, and
results checked against his LAN viewer (`10.0.50.51`). To run the pipeline on any x86-64 Linux machine with an
NVIDIA GPU (measured on 48 GB L40S and A40 cards):

1. Start from `nvcr.io/nvidia/pytorch:25.03-py3` (digest below) as root, with this checkout as the working directory.
2. Put a calibrated six-camera rig at `calib/rig.json` and `calib/masks/` (not included in this repository yet) and
   your clips in a folder, for example `clips/`.
3. Run `bash pipeline/cuda/setup-drive.sh` once. It installs system packages with apt, builds the Ceres solvers and
   C++ helpers, builds the viewer and chase tool with Bun, and downloads the pinned weights after printing each
   model's licence (`drive_models.py`).
4. Export `RS_INSTALL=$PWD HF_HOME=$PWD/third_party/hf TORCH_HOME=$PWD/third_party/torch DIFIX_RENDER_ONLY_DECODE=1`
   and run `.venv/bin/python pipeline/run/drive.py run clips --name my-drive --quality full`. The runner starts a
   private viewer on 127.0.0.1 to check its own output; the published scene is `scene/my-drive`.

`run_pod.py`, `drive_worker.sh`, `drive_receipts.py`, `verify_drive.py`, `window_clips.py` and the `pack_*` and
`*_worker.sh` scripts belong to the author's rental and verification setup and assume his job names. Their
variables: `RUNPODCTL`, `RS_JOB_DIR`, `RS_PROFILE`, `RS_SSH_KEY`, `RS_STUDIO_SSH` (SSH command for the machine
holding the private input, default his alias), `RS_STUDIO_STREAM`, and `RS_VIEWER_URL` for `verify_drive.py`
(default `http://127.0.0.1:8090`).

## Task 5: full drive on a clean worker

[CUDA scene](http://10.0.50.51:8090/?scene=hw4-cuda-224-254) · [Studio reference](http://10.0.50.51:8090/?scene=hw4-raw-224-254). Both geometry windows finished. Preview, geometry and eight adjacent chase clips cover all 30 driving seconds. The Studio verified the final release SHA-256 (7551708b05041814d764c93e2b5d04adf51e6feed04ac1a14f7b2888af9e0ced), rebuilt 81 cache tiles and byte-checked all eight served videos.

| Successful stage wall time (s), in order | L40S | Studio |
|---|---:|---:|
| Inventory / ingest / calibration / poses | 2.07 / 134.76 / 101.29 / 57.63 | 1.37 / 75.22 / 33.74 / 19.19 |
| Road / keyframes / masks / photo | 82.20 / 0.87 / 87.31 / 29.51 | 28.95 / 0.78 / 92.86 / 9.19 |
| Ground / atlas / movers / preview publish | 93.61 / 89.08 / 304.28 / 5.24 | 30.41 / 33.19 / 98.50 / 2.23 |
| Production / removal / teacher | 993.06 / 644.32 / 263.23 | 908.53 / 183.17 / 96.55 |
| Optimize + merge / attach | 347.35 + 6.38 / 226.23 | 75.17 combined / 49.40 |
| Chase / publish after geometry and clips | 5,039.02 / 6.58 + 14.65 | 408.47 / 5.51 once at end |

The 13.6-core EPYC 9354 quota and one 46,068 MiB L40S cost $1.112222/h including disk. Per driving minute: preview **32.93 compute min / $0.610**; additional geometry **82.91 / $1.537**; generated clips **168.46 / $3.123**. Each includes its own publish time. Chase alone took 167.97 s per generated second, versus 102.12 on Studio (30 s versus 4 s of output). Total owned rental upper bound including provisioning, repair and cleanup: $2.903; no named Task 5 pods remained. The 30 s rate measurements are not a quote for longer drives.

L40S vs Studio: 1,453,461 vs 1,467,516 ground splats; 390,260 vs 1,439,423 static-object splats; 112 vs 114 atlas tiles. CUDA has 69 vehicle tracks vs 36; 17 vs 2 last under one second. The [CUDA bus projection against source cameras](../../out/cuda/task5/bus-source.jpg) lands on a bank sign/hedge; no bus is evident. Counts are tracker outputs, not distinct vehicles. In the [233.5 s](../../out/cuda/task5/chase-233.5.jpg), [235.0 s](../../out/cuda/task5/chase-235.0.jpg) and [236.5 s](../../out/cuda/task5/chase-236.5.jpg) chase pairs, CUDA preserves the intersection and lane markings but loses roadside structure and the crossing SUV at 236.5 s. These compare published v5/SigLIP with v4/Vision; the strict input cut and clip boundaries also differ. [Per-model licence output](../../out/cuda/task5/model-licences.txt).

### Replay on the private Studio and worker

Use `nvcr.io/nvidia/pytorch:25.03-py3@sha256:54fd644e268346657619a987596e522e813d7abaaf92faafbe815b2b43cff16d` (x86-64, NVIDIA driver, root). The installed worker expects a source checkout, a read-only calibrated six-camera rig at `calib/{rig.json,masks}`, and the privacy-cut MP4s and clock manifest. No pretrained drive scenes or frame caches are inputs.

**On the private Studio:** prepare the input from the original inventory. The destination must not already exist. Re-encode in native YUV using lossless H.264, copy each original physical picture's Tesla SEI, and decode again to verify the pixels and SEI before transfer. Require both SEI time and offset-corrected shutter time in [224,254]. `window-clock.json` carries the original master clock and native frame numbers, not other footage. Never send the raw minute groups.

~~~sh
# From the CUDA checkout, install only the two task helpers in private Studio storage.
ferry cp pipeline/cuda/window_clips.py anomastudio:.cache/rs-task5/window_clips.py
ferry cp pipeline/cuda/verify_drive.py anomastudio:.cache/rs-task5/verify_drive.py
ferry sh anomastudio <<'SH'
cd ~/dev/reality-splat
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD
umask 077
.venv/bin/python "$HOME/.cache/rs-task5/window_clips.py" \
  work/drives/hw4-raw-224-254/1361f01da6e9b73f/full/inventory.json \
  "$HOME/.cache/rs-task5/clips-replay" --window 224 254
SH
~~~

**On the worker**, from the checkout containing the Task 5 changes, after privately transferring the verified directory as `clips/` and the calibrated rig over SSH:

~~~sh
cd /root/rs
export RS_INSTALL=$PWD HF_HOME=$PWD/third_party/hf TORCH_HOME=$PWD/third_party/torch
export DIFIX_RENDER_ONLY_DECODE=1 MAX_JOBS=3
bash pipeline/cuda/setup-drive.sh
.venv/bin/python pipeline/run/drive.py run clips --name hw4-cuda-224-254 \
  --window 224 254 --quality full --clip all
.venv/bin/python pipeline/run/drive.py status --name hw4-cuda-224-254
~~~

Setup builds Ceres calibration and trajectory solvers, the CPU renderers, and the Bun chase tool. It retains image-native PyTorch/CUDA, isolates the legacy Difix environment, and invokes `pipeline/cuda/drive_models.py` to print each model's card licence and upstream terms before downloading its pinned weights. Non-commercial use only; the pinned revisions are listed in the downloader. The actual drive runs offline. The private loopback viewer checks the same cache and scene bytes as the Studio. The progressive receipts use non-overlapping driving coverage rather than summing the lengths of the reconstruction windows.

**Guarded rental:** `RS_PROFILE=5` runs the same command through `drive_worker.sh` and `drive_receipts.py`, recording CFS quota, affinity, GPU memory, per-stage times and release receipts. Retain the same private `RS_JOB_DIR` across retries for the task meter. Stage receipts stream over SSH, preview and full geometry release bytes go to private Studio storage as soon as available, and final publication is separate. Never register a partial release as the completed scene.

The guarded controller needs two archives. Build the **source-only** one on the controller using tracked source names plus the new driver scripts, never a workspace-wide tar:

~~~bash
set -euo pipefail; umask 077
export RS_JOB_DIR="$TMPDIR/opencode/reality-cuda-preflight/task5" RS_PROFILE=5
export RUNPODCTL="$TMPDIR/opencode/reality-cuda-preflight/runpodctl"
mkdir -p "$RS_JOB_DIR"
python3 - <<'PY'
import os,subprocess,tarfile
from pathlib import Path
paths = map(Path,subprocess.check_output(['git','ls-files','pipeline','viewer/src','viewer/index.html','viewer/bun.lock','viewer/package.json','viewer/vite.config.ts','viewer/tsconfig.json','viewer/tools/build-tile-cache.ts','viewer/tools/chase-track.ts','deploy/serve.ts'],text=True).splitlines())
with tarfile.open(Path(os.environ['RS_JOB_DIR'])/'profile.tar','w') as t:
    files={p for p in paths if p.parts[0]!='pipeline' or (not any('report' in v.lower() for v in p.parts[:-1]) and (p.suffix in ('.py','.cpp','.h','.hpp','.cu') or p.name in ('CMakeLists.txt','rig-baseline.json','constraints-depth.txt','constraints-genfill.txt')))}
    files.update(Path('pipeline/cuda')/f for f in ('drive_models.py','drive_worker.sh','drive_receipts.py','setup-drive.sh'))
    for f in sorted(files):t.add(f,recursive=False)
PY
# Studio preparation above assumes the window exporter from this worktree was installed there.
# From the Studio, stage a tar with only the verified clips, clock and static calibration:
ferry sh anomastudio <<'SH'
set -eu; umask 077; cd ~/dev/reality-splat; q=$HOME/.cache/rs-task5
stage=$(mktemp -d "$q/task5-input.XXXXXXXX"); mkdir "$stage/calib"
ln -s "$q/clips-replay" "$stage/clips"; cp -Rp calib/rig.json calib/masks "$stage/calib/"
COPYFILE_DISABLE=1 tar --no-xattrs --no-mac-metadata -chf "$q/replay.tar" -C "$stage" clips calib
SH
export RS_STUDIO_STREAM='cat ~/.cache/rs-task5/replay.tar'
export GPU='NVIDIA L40S' CLOUD=SECURE RS_MAX_GPU_RATE=1.10 RS_CONTAINER_GB=160
python3 -u pipeline/cuda/run_pod.py
~~~

To register a completed run, pass the private Studio directory and SHA from the **final** streamed receipt to `verify_drive.py`. It verifies the tree before rebuilding its RSB cache, checks all eight served MP4s against disk, and extracts the three aligned chase pairs inside the historical scene's four-second overlap.

~~~sh
# On Studio, using the verify_drive.py from this worktree:
cd ~/dev/reality-splat
PYTHONPATH=$PWD/pipeline/run .venv/bin/python "$HOME/.cache/rs-task5/verify_drive.py" \
  "$HOME/.cache/rs-task5/$JOB_NAME/final" "$FINAL_CONTENT_SHA256" "$HOME/.cache/rs-task5/$JOB_NAME/review"
~~~

## Historical A40 component measurements (before Tasks 2–5)


**The native chase fixer alone needs 66.1 compute minutes per minute of 30 fps video.** The A40 completed the component benchmarks, not a new 224–254 s drive reconstruction. A fresh equivalent preview needs Apple's mover classifier and the full fit needs MetalSplat's differentiable renderer. No CUDA scene, generated MP4 or viewer cache entry was produced.

## What was measured

One secure-cloud A40 at a time (two sequential jobs), 46,068 MiB visible, driver 595.112.07, CUDA 12.8 and PyTorch 2.9.1. $0.590/h GPU + $0.008/h storage. Balance [redacted] → [redacted]: **$0.1969951251 spent** including setup, transfer and compilation. Final rate $0/h; pods [], serverless endpoints [], network volumes []. No account limit changed.

| Stage, 30 driving seconds | Studio wall (s) | A40 result |
|---|---:|---|
| Inventory / ingest / calibration / poses / road / keys | 1.37 / 75.22 / 33.74 / 19.18 / 28.95 / 0.78 | Not run |
| Preview masks / photo / ground / atlas | 92.86 / 9.19 / 30.41 / 33.19 | Masks: 0.843 s/six-camera keyframe |
| Movers / preview publish | 98.50 / 2.23 | 570 independent Apple Vision classifications on Studio; no validated Linux equivalent |
| **Preview** | **425.61; 14.19 compute min/driving min** | **Total unmeasured** |
| Production / removal / teachers | 908.53 / 183.17 / 96.55 | Different production benchmark, below |
| Optimize & merge / attach | 75.17 / 49.40 | Color-only gsplat surrogate; no equivalent fit |
| Chase / publish full | 408.46 / 5.51 | Image fixer only, below |
| **Historical preview + full** | **2152.39; 71.75 compute min/driving min** | **Total unmeasured** |

Historical full produced 121 video frames (233–237 s), not 30 s of video. From actual Studio keyframe counts (149 six-camera sets in preview), the A40 preview **masks alone** project to 149 × 0.843 × 2 = 251 s or **4.19 compute min / $0.042 per driving minute**. This extrapolates from a warm 12-keyframe sample; it excludes all other stages. With a full 30 fps clip covering every driving second, the **fixer alone** projects to 1,800 / 0.454 = 3,967 s or **66.1 min / $0.659 per driving minute** at native size; **24.6 min / $0.246** at 1024×576. Neither number includes CPU raster/composite, clip encoding, reconstruction, or startup; neither is a measured preview/full bill.

**Depth and masks:** 12 consecutive six-camera keyframes beginning at 233.143 s, second warm pass: masks 0.843 s + DA3 front/rear 2.758 s + MapAnything side cameras 5.317 s = **8.918 s/keyframe**, approximately $0.00148/keyframe at the observed rental rate. Includes JPEG decode, temporal windows, rectification, inference, resampling and writes. The Studio comparison (0.63 s masks + 2.21 s depth) used the same benchmark configuration around 300 s on a shared machine. The historical **drive runner**, by contrast, passed DA3-Large explicitly on **all six cameras**, and its stage also includes key selection and verification. Do not project the mixed-model depth result onto that stage.

| Difix-ref, batch 1 | FP16 eager fps; peak allocated/reserved GB | FP32 eager fps; peak allocated/reserved GB | FP16 compiled UNet fps; peak allocated/reserved GB |
|---|---:|---:|---:|
| 1472×960 | 0.454; 11.25/15.03 | 0.178; 22.48/28.53 | 0.456; 11.25/15.02 |
| 1024×576 | 1.217; 6.22/7.86 | 0.557; 12.41/14.40 | 1.246; 6.22/7.85 |

One step, timestep 199, fixed seed, loaded PIL inputs, five CUDA-synchronized wall samples after warmup. Timed batch 1 repeats the first pair; three distinct pairs were processed for visual review. Studio native FP16 median of 120 post-first fix calls: 1.404 s (0.712 fps). Compiling the A40 UNet changed medians by <3%. Only batch 1 is validated: the upstream VAE failed at batch 2 (two latents, four skip maps); 4 and 8 were not timed. No TensorRT results.

**Splat surrogate:** 350.97 iter/s, five warmup and 50 timed iterations on three real fitted tiles, 162,460 splats, 736×480 chase camera. Antialiased gsplat forward, RGB loss, color-only backward/Adam. First CUDA extension build (350.46 s) is excluded. The historical fit was v4, two chunks × 1,600 iterations; the requested v5 path changes crop sampling, parameters and losses. Neither fit can be timed by multiplying iterations by the surrogate rate.

**Review:** three private JPEG sheets, render | raw CUDA Difix | final Studio composite: [233.000](../../out/cuda/compare-0000.jpg), [234.333](../../out/cuda/compare-0040.jpg), [235.667](../../out/cuda/compare-0080.jpg). Road texture and foreground edges differ; the Studio composite also protects road pixels and carries prior frames. These sheets check scene content, not numerical equivalence. c296 was outside the authorized upload interval. None of its images were sent.

[Timings and memory samples](results.json) · [Inventory, licenses, limitations and source revisions](PORT.md).

## Replay the component worker

Run from the **cuda** worktree at **fe29f83**, including its uncommitted device changes. The Studio proof is a read-only data source. Do not package the raw bracketing minutes: a leading GOP starts before the authorized start. Commands below are for the basebook macOS shell. Start with an empty account and retain the **original job-start balance** across retries. The two measurements used the image tag in PORT.md; the packaged controller pins its resolved digest. Component code/weights and recorded Python dependency versions are pinned. Torch Hub DINOv2 main was unpinned during measurement; replay pins the SHA found after teardown. The consolidated, pinned replay was syntax-checked, not billed for a third run.

~~~sh
set -e -o pipefail
umask 077
export RS_JOB_DIR=$(mktemp -d "$TMPDIR/opencode/rs-a40.XXXXXXXX")
export RUNPODCTL="$RS_JOB_DIR/runpodctl"
curl -fsSL https://github.com/runpod/runpodctl/releases/download/v3.1.1/runpodctl-darwin-arm64 -o "$RUNPODCTL"
chmod 700 "$RUNPODCTL"
export RS_START_BALANCE=$("$RUNPODCTL" user | jq -er '.clientBalance')
"$RUNPODCTL" pod list; "$RUNPODCTL" serverless list; "$RUNPODCTL" network-volume list
p="$RS_JOB_DIR/prepared"; mkdir -p "$p"
{ printf '%s\n' 'set -o pipefail' 'cd ~/dev/reality-splat' \
    "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv/bin/python - <<'PY' | base64";
  cat pipeline/cuda/pack_window.py; echo PY; } | ferry sh anomastudio | base64 -D | tar xzf - -C "$p"
mkdir -p "$p"/pipeline/{cuda,depth,masks,common,genfill,livemap}
for dir in depth masks common; do cp pipeline/$dir/*.py "$p/pipeline/$dir/"; done
cp pipeline/genfill/difix.py "$p/pipeline/genfill/"
cp pipeline/livemap/raster.cpp "$p/pipeline/livemap/"
cp pipeline/cuda/{setup.sh,bench_difix.py,bench_splat.py,constraints-*.txt} "$p/pipeline/cuda/"
COPYFILE_DISABLE=1 tar --no-xattrs --no-mac-metadata -czf "$RS_JOB_DIR/job.tar.gz" -C "$p" .
python3 pipeline/cuda/run_pod.py
"$RUNPODCTL" pod list; "$RUNPODCTL" serverless list; "$RUNPODCTL" network-volume list
"$RUNPODCTL" user | jq '{clientBalance,currentSpendPerHr}'
~~~

The packer's 235 input files were independently matched by SHA-256 against the timed jobs' source inputs. It exports only 206 required camera JPEGs, in-window clock/pose data, calibration, the three chase inputs and three tiles. All camera shutters were 225.949–234.404 s. Media travels to the pod over direct SSH only. The controller creates one ephemeral 60 GB container with port 22, no network volume; it installs weights in separate component environments, retrieves the result archive over SSH, and deletes the unique pod. Boot deadline, in-container absolute and idle timers, and a detached controller-side guard run independently of the benchmark. The guard starts before creation and uses a $34 spend threshold under the $35 stop point.

The first job ran the following after setup; upstream batch 2 interrupted both eager sweeps after native resolution, compile produced the native sample before timeout, and gsplat lacked NumPy. The second job enabled only missing cases (small-only and skip-depth) and installed NumPy in the gsplat environment.

~~~sh
envs/genfill/bin/python pipeline/cuda/bench_difix.py --dtype float16
envs/genfill/bin/python pipeline/cuda/bench_difix.py --dtype float32
timeout 240 envs/genfill/bin/python pipeline/cuda/bench_difix.py --dtype float16 --compile
envs/depth/bin/python pipeline/depth/bench_e2e.py --t0 233 --k 12 --configs production
# Second ephemeral A40:
envs/genfill/bin/python pipeline/cuda/bench_difix.py --dtype float16 --small-only
envs/genfill/bin/python pipeline/cuda/bench_difix.py --dtype float32 --small-only
timeout 300 envs/genfill/bin/python pipeline/cuda/bench_difix.py --dtype float16 --compile --small-only
envs/splat/bin/python pipeline/cuda/bench_splat.py
~~~

Next measurement: run the existing DA3-Large **all-camera** benchmark on the identical 12 keyframes, splitting host work from synchronized model forward. It would provide a relevant component estimate for the historical 908.53 s production stage.
