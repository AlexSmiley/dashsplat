# Chase-view video diffusion

This experiment runs the released [StreetNVS](https://github.com/StreetNVS/StreetNVS) driving-video model on the live-v5 viewer track. The delivered conditioning pairs Gaussian road geometry with projected photographic cutouts from the rigid-layer IBR bank. [REPORT.md](REPORT.md) contains results and the paired clip. The car and HUD are absent. No Waymo data, CUDA point renderer, rental, or external video-generation service is used.

**Round two:** [dense stitched-opt-v3 experiment, cost and verdict](round2/REPORT.md). The original live-v5 viewer entry was removed; its round-one files remain archived.

## Reproduce on anomastudio

Run the following from `$RS` (the Studio's Reality Splat root). The project environment is used only to render and evaluate; the model has its own environment. Dependencies: the prepared `work/ibr/popup-v2/` bank and its pose exports, the live-v5 and stitched ground PLYs, its built CPU rasteriser and panorama, calibrated rig and poses, drive frames, keyframe depths/masks and photometry; ffmpeg on PATH. The model files occupy approximately 27 GiB of disk space.

```bash
# 1. Install the exact upstream revision and public weights (only once).
git clone https://github.com/StreetNVS/StreetNVS third_party/StreetNVS
git -C third_party/StreetNVS checkout fd002eafa62589bd84d44f87e4fc8461fcec3354
uv venv --python .venv/bin/python envs/genvideo
uv pip install --python envs/genvideo/bin/python 'torch==2.15.0' 'torchvision==0.29.2' -e third_party/StreetNVS
envs/genvideo/bin/python third_party/StreetNVS/scripts/download_models.py --output third_party/StreetNVS/models

# 2. Render 150 frames on the master clock, 105.0 through 119.9 seconds, and export 81/49-frame packages. The supplied track is the literal output of
# (on basebook): cd viewer && bun tools/chase-track.ts <live-v5-dir> 105 119.9 10 512 320 > ../pipeline/genvideo/report/chase-track.json
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python -m pipeline.genvideo.prepare --scene live-v5

# 3. Run the two native-length windows sequentially (first: 105.0; second: 111.9 s).
export PYTORCH_ENABLE_MPS_FALLBACK=1
GV=work/genvideo
PY=envs/genvideo/bin/python
$PY -m pipeline.genvideo.mps --window 000-81 --size 512 320 --steps 20 --cfg 5 --tiled --checkpoint-steps --output $GV/track-000-81.mp4
$PY -m pipeline.genvideo.mps --window 069-81 --size 512 320 --steps 20 --cfg 5 --tiled --checkpoint-steps --previous $GV/track-000-81.mp4 --output $GV/track-069-81.mp4

# 4. Blend 12 overlap frames, export naked generation, A/B, eight-frame sheets, and metrics.
.venv/bin/python -m pipeline.genvideo.evaluate $GV/track-000-81.mp4 $GV/track-069-81.mp4
.venv/bin/python -m pipeline.genvideo.photos $GV/generated.mp4
```

The old-pose native-resolution baseline is archived on Studio in `work/genvideo/v4-baseline/`. The reduced bring-up command is `$PY -m pipeline.genvideo.mps --window 000-49 --size 320 192 --steps 4 --cfg 1 --output $GV/smoke.mp4`. The CLI has optional FP16 and resident-block modes; the profiled runs use BF16 and streamed blocks. On the shared Mac use `bash pipeline/genvideo/run_windows.sh` instead of steps 3–4: it runs serially after 90 s of normal memory pressure with steady swapout count, and restarts only explicit pressure stops. Its input-hashed checkpoints retain latent state after every denoising step. Check `memory_pressure -Q` and the **change** in `vm_stat` swapouts before increasing resolution or enabling resident mode.

## What goes into the model

- **Target poses:** use the live-v5 drive JSON and the viewer’s pure chase-track exporter; convert its camera-to-world quaternion by right-multiplying `diag(1,−1,−1)` for OpenCV. No Python approximation of the orbit/smoothing is used. Base distance 16.2 m, elevation 16.9°, local tilt +5.3°, no pan, pivot 1.45 m forward and 0.8 m up. Track and output are 512×320, 50° vertical FOV (343.12 px focal length), 10 Hz. Row-time-calibrated physical photographs use the refined ego trajectory.
- **RGB, mask and depth:** render layered IBR from those exact poses in the live-v5 coordinate frame and retain pure v5 splat RGB separately. Stitched Gaussian ground is aligned by the IBR module. Only observed ground with alpha >0.86 and identified near photographic cutouts with alpha >0.65 get finite depth. Projected sky, apron, the assumed 92 m far cylinder and movers without exported opacity are **not** treated as observed depth. Downsample weighted axial depth and supported colour with area filtering; keep pixels with >50% support, and invalidate the union of the viewer and refined-rig ego footprints. Upstream expects the inverse validity (white holes) and sequence-normalized JET depth. The full bare IBR render is used as the first target anchor and the side-by-side comparator. The second window anchors to the first generated overlap frame. `--splat-only` selects the time-aware Gaussian rasteriser instead.
- **References:** select the nearest physical exposure in each camera (corrected timestamps); rank lenses by the fraction of reconstructed chase surface samples landing inside their pinhole view. Use the top five at each window start, shared rig undistortion and ground-v2 photometry. Ranking is a frustum-overlap heuristic, not occlusion-tested. Full pose tracks are included; StreetNVS only reads frame zero of each reference video. The calibrated ray module computes each reference's own K rather than broadcasting the first lens across the rig.
- **Where:** `work/genvideo/raw/` holds the exact pose track, metric z, ego footprint, reverse-opacity, control colour and complete view. `work/genvideo/inputs/{000-81,069-81,000-49}/` are self-contained upstream-format inputs; FFV1 ensures no compression bleed in the model's mask or depth. Unblended native-length windows, per-stage timings and final A/B are in `work/genvideo/`. The text contexts are cached by prompt; clear the cache if the UMT5 checkpoint changes.

## Port and measurements

The original loader hard-codes CUDA and temporarily creates a full FP32 network. `mps.py` builds on the meta device and strictly assigns the published parameters (including rank-1024 adapters). CPU UMT5 uses mmap-backed weights and only unmasked tokens; its output is zero-padded like the upstream embedding. The VAE and DiT run sequentially; each DiT block is brought onto MPS for a forward pass, then reassigned to file-backed CPU weights. Full-key attention is sliced **only in the query dimension**. Wan's unsupported double-precision rotary and timestep calculations use FP32. The ray-relative attention, RGB/mask/depth streams, first-frame anchor, LoRA, CFG and flow scheduler are retained. This is not a distilled or interpolated model.

The per-window JSON reports runtime starting before input decoding and ending after the MP4 is closed, excluding process startup and upstream asset downloads. Metal driver memory is sampled every 250 ms and process peak RSS is the Darwin OS high-water mark. They overlap on unified memory and must not be added. Evaluation details and limitations accompany the results. Upstream model and training-data terms remain applicable; the point preprocessing components in StreetNVS's NOTICE are not used.

## Viewer delivery

`evaluate.py` writes H.264/yuv420p at 10 fps, IDR interval five frames, no audio. Install its `generated.mp4` at `scene/live-v5/generated/streetnvs-105-120.mp4`. Add the record in [report/generated.json](report/generated.json) to `scene/live-v5/generated.json`, preserving other entries. From basebook run `cd viewer && bun run cache:studio`. At 10 Hz the 150-frame interval has **t1=119.9**, the last frame, not the exclusive end 120.0.
