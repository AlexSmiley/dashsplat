# GPU stage profile: Difix and six-camera DA3

Private 224–254 s component pack. All model runs and compilation were on the GPU workers. Two modes use the supported NVIDIA checkpoints: **Difix-ref** (two-view UNet) and **Difix** (single view). Omitting the reference from the two-view checkpoint is not a valid comparison. All images use timestep 199, a CPU generator seeded with zero, and FP16. No outputs were published.

## Stock stage timings

Means across three distinct pairs, milliseconds. Every stage boundary drains CUDA. PNG loading and resizing are measured once per pair and excluded from stock total. Pre = how PIL/NumPy is normalized before transfer; latent prep = remaining transfers, sampling and scale outside VAE encode; D2H = transfer plus NumPy layout conversion; Other includes scheduler, Python and 8-bit conversion. The text-encoder forward is inside the prompt column. Stage means and uninstrumented throughput medians need not match.

| Card, pixels | Mode | Load | Pre | Prompt | VAE encode | Latent prep | UNet | VAE decode | Post | D2H | Other | Total |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L40S, 1024×576 | single | 41.4 | 4.9 | 11.4 | 49.6 | 2.2 | 28.1 | 140.4 | 0.2 | 1.0 | 14.3 | 252.1 |
| L40S, 1024×576 | ref | 83.1 | 10.7 | 13.5 | 96.0 | 5.3 | 45.1 | 273.1 | 0.3 | 1.0 | 8.6 | 453.7 |
| L40S, 1472×960 | single | 24.4 | 7.1 | 11.7 | 163.6 | 4.1 | 56.6 | 439.2 | 0.4 | 1.3 | 12.9 | 697.0 |
| L40S, 1472×960 | ref | 53.8 | 23.5 | 12.7 | 308.5 | 9.7 | 139.8 | 817.5 | 0.5 | 7.4 | 23.7 | 1343.4 |
| A40, 1024×576 | single | 52.8 | 5.1 | 20.7 | 97.1 | 2.5 | 45.4 | 282.4 | 0.3 | 3.9 | 8.9 | 466.3 |
| A40, 1024×576 | ref | 106.5 | 91.6 | 14.1 | 179.6 | 4.6 | 90.2 | 500.1 | 0.2 | 0.7 | 9.9 | 891.0 |
| A40, 1472×960 | single | 33.3 | 24.9 | 14.0 | 263.4 | 21.5 | 116.8 | 732.1 | 0.4 | 2.3 | 25.7 | 1201.1 |
| A40, 1472×960 | ref | 73.1 | 22.5 | 12.2 | 502.3 | 33.4 | 275.9 | 1260.5 | 0.3 | 3.5 | 24.8 | 2135.4 |

The ref encoder already receives a **single batch of two** (render + reference); there are no sequential per-view encoder times to split. The single-view checkpoint gives a direct one-image encode cost. Ref decode takes one latent, broadcasts against two sets of skip activations, produces two RGB images, then discards the second. The single-view model outputs one. This is observed shape information, not a proposed change to conditioning.

## Throughput, cost and live budget

Final tested configuration with **all three RGB images byte-identical** to that checkpoint’s stock baseline; byte identity does not claim quality equivalence between the two checkpoints. Six uninstrumented timed calls, rotating the three frames twice, one warmup. Includes normalization, device work, transfer and CPU quantization; excludes PNG read/resize, rendering, streaming and startup. Cost assumes 1,800 fixed frames per minute of driving and continuous billing at the observed account rates ($0.598/h A40; $0.794/h L40S). The L40S startup rate read zero due to API lag; the account later reported $0.794/h. N = ceil(30/fps) for a 30 Hz display; these are asynchronous fresh-image budgets, not measured live playback or acceptable staleness.

| Card, pixels | Mode | Exact configuration | ms | FPS | Fixer $/driving min | Every Nth display frame |
|---|---|---|---:|---:|---:|---:|
| L40S, 1024×576 | single | cudnn_benchmark | 237.3 | 4.21 | $0.094 | 8 |
| L40S, 1024×576 | ref | cudnn_benchmark | 435.4 | 2.30 | $0.173 | 14 |
| L40S, 1472×960 | single | cudnn_benchmark | 687.7 | 1.45 | $0.273 | 21 |
| L40S, 1472×960 | ref | cudnn_benchmark | 1339.4 | 0.75 | $0.532 | 41 |
| A40, 1024×576 | single | cudnn_benchmark | 456.0 | 2.19 | $0.136 | 14 |
| A40, 1024×576 | ref | cudnn_benchmark | 798.7 | 1.25 | $0.239 | 24 |
| A40, 1472×960 | single | cudnn_benchmark | 1159.5 | 0.86 | $0.347 | 35 |
| A40, 1472×960 | ref | cudnn_benchmark | 2099.8 | 0.48 | $0.628 | 63 |

## Host and device observations

GPU active percentage is nvidia-smi sampled at 200 ms across the uninstrumented stock loop, not SM occupancy. cpu.max is quota/period (e.g. 765000/100000 = 7.65 CPU equivalents).

| Card, pixels | Mode | Host / affinity CPUs | cpu.max | Torch threads | GPU active | Encode / decoded batch |
|---|---|---:|---|---:|---:|---:|
| L40S, 1024×576 | single | 112 / 112 | None | 56 | 85.6% | 1 / 1 |
| L40S, 1024×576 | ref | 112 / 112 | None | 56 | 90.8% | 2 / 2 |
| L40S, 1472×960 | single | 112 / 112 | None | 56 | 93.7% | 1 / 1 |
| L40S, 1472×960 | ref | 112 / 112 | None | 56 | 94.0% | 2 / 2 |
| A40, 1024×576 | single | 96 / 96 | 765000 100000 | 48 | 90.7% | 1 / 1 |
| A40, 1024×576 | ref | 96 / 96 | 765000 100000 | 48 | 95.5% | 2 / 2 |
| A40, 1472×960 | single | 96 / 96 | 765000 100000 | 48 | 95.4% | 1 / 1 |
| A40, 1472×960 | ref | 96 / 96 | 765000 100000 | 48 | 93.0% | 2 / 2 |

## DA3 and Mask2Former: seconds per six-camera keyframe

| Card | Model | CPU + file I/O | Transfers + device prep | GPU forward | GPU post | Wall |
|---|---|---:|---:|---:|---:|---:|
| L40S | mask | 0.111 | 0.010 | 0.129 | 0.010 | 0.261 |
| L40S | depth | 1.713 | 0.028 | 0.784 | 0.000 | 2.525 |
| A40 | mask | 0.174 | 0.007 | 0.202 | 0.012 | 0.395 |
| A40 | depth | 5.126 | 0.023 | 1.250 | 0.000 | 6.399 |

One warm keyframe, then 12 timed, all six authorized cameras. Existing production readers and writers preserve per-camera tracking, rectification, original grid resampling, masks, scale checks and file writes. CPU + I/O is wall time outside synchronized device sections, including setup and thread waits; not CPU-core seconds or a disk-only estimate. Mask forward is FP16 with six images per batch and FP32 logits/post. DA3 uses FP16 backbone autocast and the SDK’s intentional FP32 pose/depth heads. Five independent five-view sequences share a batch; the rear nine-view fisheye sequence runs separately at another height. Combining these sequences along the view axis would change cross-camera conditioning. Native multi-camera batching is not achieved by padding the rear sequence with invented views.

## Hygiene ladder and RGB comparison

Diffs are max/mean absolute 8-bit channel levels, in frame order 233.000 / 234.333 / 235.667. Each trial builds on prior trials that met the predeclared small-drift gate (per-frame max ≤1 and mean ≤0.02); failed layout or input trials were reverted before continuing. The throughput summary above is stricter: **zero changed channel values**. Distinguish an ablation with a small FP16 rounding difference from unchanged pixels. Input optimization includes H2D on every timed call. SDPA, VAE FP16 and batched VAE encoding were already active; their rows are remeasurements, not three independent gains. Processor names, actual ATen flash/efficient kernels, cold-compile times and all raw samples are in [profile-results.json](profile-results.json).

| Card, pixels | Mode | Trial | ms | FPS | Max/mean, three frames | GPU active |
|---|---|---|---:|---:|---|---:|
| L40S, 1024×576 | single | stock | 246.6 | 4.06 | 0/0.0000; 0/0.0000; 0/0.0000 | 85.6% |
| L40S, 1024×576 | single | cached_prompt | 233.3 | 4.29 | 0/0.0000; 0/0.0000; 0/0.0000 | 90.0% |
| L40S, 1024×576 | single | pinned_gpu_input | 214.3 | 4.67 | 5/0.0448; 3/0.0447; 2/0.0357 | 93.0% |
| L40S, 1024×576 | single | channels_last | 252.0 | 3.97 | 3/0.0394; 4/0.0410; 3/0.0322 | 91.6% |
| L40S, 1024×576 | single | sdpa | 235.5 | 4.25 | 0/0.0000; 0/0.0000; 0/0.0000 | 88.4% |
| L40S, 1024×576 | single | fp16_verified | 232.6 | 4.30 | 0/0.0000; 0/0.0000; 0/0.0000 | 89.6% |
| L40S, 1024×576 | single | vae_single | 240.2 | 4.16 | 0/0.0000; 0/0.0000; 0/0.0000 | 87.7% |
| L40S, 1024×576 | single | cudnn_benchmark | 237.3 | 4.21 | 0/0.0000; 0/0.0000; 0/0.0000 | 90.6% |
| L40S, 1024×576 | single | compile_vae_decode | 183.5 | 5.45 | 2/0.0244; 2/0.0249; 2/0.0234 | 88.2% |
| L40S, 1024×576 | ref | stock | 452.1 | 2.21 | 0/0.0000; 0/0.0000; 0/0.0000 | 90.8% |
| L40S, 1024×576 | ref | cached_prompt | 437.2 | 2.29 | 0/0.0000; 0/0.0000; 0/0.0000 | 93.9% |
| L40S, 1024×576 | ref | pinned_gpu_input | 424.6 | 2.35 | 7/0.0475; 5/0.0467; 5/0.0410 | 95.7% |
| L40S, 1024×576 | ref | channels_last | 488.2 | 2.05 | 6/0.0413; 5/0.0385; 4/0.0334 | 92.6% |
| L40S, 1024×576 | ref | sdpa | 440.5 | 2.27 | 0/0.0000; 0/0.0000; 0/0.0000 | 92.8% |
| L40S, 1024×576 | ref | fp16_verified | 438.8 | 2.28 | 0/0.0000; 0/0.0000; 0/0.0000 | 93.6% |
| L40S, 1024×576 | ref | vae_pair_already_batched | 440.1 | 2.27 | 0/0.0000; 0/0.0000; 0/0.0000 | 92.9% |
| L40S, 1024×576 | ref | cudnn_benchmark | 435.4 | 2.30 | 0/0.0000; 0/0.0000; 0/0.0000 | 94.2% |
| L40S, 1024×576 | ref | compile_vae_decode | 368.9 | 2.71 | 1/0.0208; 1/0.0215; 1/0.0205 | 91.7% |
| L40S, 1472×960 | single | stock | 701.3 | 1.43 | 0/0.0000; 0/0.0000; 0/0.0000 | 93.7% |
| L40S, 1472×960 | single | cached_prompt | 691.7 | 1.45 | 0/0.0000; 0/0.0000; 0/0.0000 | 94.5% |
| L40S, 1472×960 | single | pinned_gpu_input | 551.9 | 1.81 | 7/0.0476; 5/0.0456; 8/0.0424 | 95.6% |
| L40S, 1472×960 | single | channels_last | 821.7 | 1.22 | 6/0.0451; 4/0.0422; 6/0.0511 | 95.6% |
| L40S, 1472×960 | single | sdpa | 690.1 | 1.45 | 0/0.0000; 0/0.0000; 0/0.0000 | 95.2% |
| L40S, 1472×960 | single | fp16_verified | 688.3 | 1.45 | 0/0.0000; 0/0.0000; 0/0.0000 | 96.0% |
| L40S, 1472×960 | single | vae_single | 688.7 | 1.45 | 0/0.0000; 0/0.0000; 0/0.0000 | 95.7% |
| L40S, 1472×960 | single | cudnn_benchmark | 687.7 | 1.45 | 0/0.0000; 0/0.0000; 0/0.0000 | 95.7% |
| L40S, 1472×960 | single | compile_vae_decode | 509.7 | 1.96 | 2/0.0254; 2/0.0246; 3/0.0234 | 93.5% |
| L40S, 1472×960 | ref | stock | 1342.6 | 0.74 | 0/0.0000; 0/0.0000; 0/0.0000 | 94.0% |
| L40S, 1472×960 | ref | cached_prompt | 1331.6 | 0.75 | 0/0.0000; 0/0.0000; 0/0.0000 | 95.4% |
| L40S, 1472×960 | ref | pinned_gpu_input | 1240.3 | 0.81 | 5/0.0351; 9/0.0383; 6/0.0330 | 98.0% |
| L40S, 1472×960 | ref | channels_last | 1595.4 | 0.63 | 6/0.0346; 6/0.0349; 5/0.0330 | 96.0% |
| L40S, 1472×960 | ref | sdpa | 1333.3 | 0.75 | 0/0.0000; 0/0.0000; 0/0.0000 | 95.4% |
| L40S, 1472×960 | ref | fp16_verified | 1333.6 | 0.75 | 0/0.0000; 0/0.0000; 0/0.0000 | 95.5% |
| L40S, 1472×960 | ref | vae_pair_already_batched | 1336.3 | 0.75 | 0/0.0000; 0/0.0000; 0/0.0000 | 95.8% |
| L40S, 1472×960 | ref | cudnn_benchmark | 1339.4 | 0.75 | 0/0.0000; 0/0.0000; 0/0.0000 | 95.8% |
| L40S, 1472×960 | ref | compile_vae_decode | 1026.8 | 0.97 | 1/0.0198; 1/0.0202; 1/0.0179 | 93.8% |
| A40, 1024×576 | single | stock | 453.1 | 2.21 | 0/0.0000; 0/0.0000; 0/0.0000 | 90.7% |
| A40, 1024×576 | single | cached_prompt | 452.6 | 2.21 | 0/0.0000; 0/0.0000; 0/0.0000 | 93.8% |
| A40, 1024×576 | single | pinned_gpu_input | 366.1 | 2.73 | 4/0.0448; 5/0.0447; 3/0.0370 | 94.0% |
| A40, 1024×576 | single | channels_last | 530.6 | 1.88 | 6/0.0445; 8/0.0418; 3/0.0328 | 93.4% |
| A40, 1024×576 | single | sdpa | 454.6 | 2.20 | 0/0.0000; 0/0.0000; 0/0.0000 | 89.9% |
| A40, 1024×576 | single | fp16_verified | 446.3 | 2.24 | 0/0.0000; 0/0.0000; 0/0.0000 | 94.8% |
| A40, 1024×576 | single | vae_single | 443.0 | 2.26 | 0/0.0000; 0/0.0000; 0/0.0000 | 93.8% |
| A40, 1024×576 | single | cudnn_benchmark | 456.0 | 2.19 | 0/0.0000; 0/0.0000; 0/0.0000 | 89.3% |
| A40, 1024×576 | single | compile_vae_decode | 324.1 | 3.09 | 2/0.0247; 2/0.0250; 2/0.0229 | 90.6% |
| A40, 1024×576 | ref | stock | 812.1 | 1.23 | 0/0.0000; 0/0.0000; 0/0.0000 | 95.5% |
| A40, 1024×576 | ref | cached_prompt | 795.1 | 1.26 | 0/0.0000; 0/0.0000; 0/0.0000 | 94.0% |
| A40, 1024×576 | ref | pinned_gpu_input | 759.9 | 1.32 | 7/0.0470; 5/0.0445; 5/0.0453 | 98.1% |
| A40, 1024×576 | ref | channels_last | 998.3 | 1.00 | 8/0.0426; 3/0.0400; 5/0.0328 | 97.8% |
| A40, 1024×576 | ref | sdpa | 795.8 | 1.26 | 0/0.0000; 0/0.0000; 0/0.0000 | 99.5% |
| A40, 1024×576 | ref | fp16_verified | 796.7 | 1.26 | 0/0.0000; 0/0.0000; 0/0.0000 | 99.0% |
| A40, 1024×576 | ref | vae_pair_already_batched | 798.0 | 1.25 | 0/0.0000; 0/0.0000; 0/0.0000 | 99.1% |
| A40, 1024×576 | ref | cudnn_benchmark | 798.7 | 1.25 | 0/0.0000; 0/0.0000; 0/0.0000 | 99.0% |
| A40, 1024×576 | ref | compile_vae_decode | 649.8 | 1.54 | 1/0.0208; 2/0.0225; 1/0.0190 | 94.1% |
| A40, 1472×960 | single | stock | 1177.3 | 0.85 | 0/0.0000; 0/0.0000; 0/0.0000 | 95.4% |
| A40, 1472×960 | single | cached_prompt | 1196.1 | 0.84 | 0/0.0000; 0/0.0000; 0/0.0000 | 96.6% |
| A40, 1472×960 | single | pinned_gpu_input | 926.8 | 1.08 | 5/0.0468; 7/0.0480; 13/0.0549 | 96.3% |
| A40, 1472×960 | single | channels_last | 1442.7 | 0.69 | 3/0.0382; 4/0.0381; 5/0.0326 | 93.8% |
| A40, 1472×960 | single | sdpa | 1185.6 | 0.84 | 0/0.0000; 0/0.0000; 0/0.0000 | 97.0% |
| A40, 1472×960 | single | fp16_verified | 1195.2 | 0.84 | 0/0.0000; 0/0.0000; 0/0.0000 | 92.1% |
| A40, 1472×960 | single | vae_single | 1147.2 | 0.87 | 0/0.0000; 0/0.0000; 0/0.0000 | 95.8% |
| A40, 1472×960 | single | cudnn_benchmark | 1159.5 | 0.86 | 0/0.0000; 0/0.0000; 0/0.0000 | 95.2% |
| A40, 1472×960 | single | compile_vae_decode | 874.6 | 1.14 | 2/0.0245; 2/0.0249; 3/0.0228 | 88.0% |
| A40, 1472×960 | ref | stock | 2184.2 | 0.46 | 0/0.0000; 0/0.0000; 0/0.0000 | 93.0% |
| A40, 1472×960 | ref | cached_prompt | 2103.0 | 0.48 | 0/0.0000; 0/0.0000; 0/0.0000 | 95.1% |
| A40, 1472×960 | ref | pinned_gpu_input | 1991.3 | 0.50 | 6/0.0373; 4/0.0362; 7/0.0334 | 99.8% |
| A40, 1472×960 | ref | channels_last | 2802.9 | 0.36 | 8/0.0331; 3/0.0308; 5/0.0272 | 94.9% |
| A40, 1472×960 | ref | sdpa | 2149.8 | 0.47 | 0/0.0000; 0/0.0000; 0/0.0000 | 94.9% |
| A40, 1472×960 | ref | fp16_verified | 2099.7 | 0.48 | 0/0.0000; 0/0.0000; 0/0.0000 | 97.2% |
| A40, 1472×960 | ref | vae_pair_already_batched | 2152.8 | 0.46 | 0/0.0000; 0/0.0000; 0/0.0000 | 93.7% |
| A40, 1472×960 | ref | cudnn_benchmark | 2099.8 | 0.48 | 0/0.0000; 0/0.0000; 0/0.0000 | 93.5% |
| A40, 1472×960 | ref | compile_vae_decode | 1769.1 | 0.57 | 1/0.0189; 1/0.0194; 1/0.0178 | 90.6% |

## Interpretation and next change

The [paper, §4](https://arxiv.org/html/2503.01774) reports 76 ms additional render time on A100. That passage does not specify a timed implementation, reference mode, dimensions, transfer boundary or arithmetic settings. An A100/A40 hardware multiplier cannot be inferred. Single-view measurements in the tables are the closer comparison, but still do not reproduce the paper measurement.

**First change to try:** decode only the kept render by supplying only its skip-activation rows to the decoder, leaving both observations in the encoder and UNet. If this made two-image decode cost the observed single-image decode cost, use stock total minus the observed ref/single decode difference as a projection, not a measured gain. Validate pixels at all three poses and clip boundaries before production. A distilled smaller decoder could reduce decode further, but requires training/quality evaluation; if decode alone became 4× faster, overall speedup would be 1/(1−0.75×decode share). Distilling the UNet alone cannot save more than its measured fraction even if it became free. Neither swap nor distillation was performed.

On the L40S at 576p, eager ref takes 96 ms to encode, 45 ms in the UNet and 273 ms to decode. Removing only the discarded decode row projects to about **1.4×** total throughput (A40 **1.3×**), using the supported single-view decoder as a proxy. A hypothetical 4× faster decoder projects to **1.7–1.8×** overall. Even a free UNet yields just **1.11–1.15×**. These are optimistic scenarios, not benchmark results. For a 30 Hz live UI, skipping fixes buys more: every 14th frame at 576p on L40S, or every 8th using the different single-view checkpoint. Neither makes the intervening frames newly fixed.

Compiled decoder near-match: reference throughput was 1.54 / 0.57 fps on A40 and 2.71 / 0.97 fps on L40S (576p / native). Mean differences were approximately 0.018–0.023 RGB levels across the three frames, maxima 1–2. Not used for the byte-exact cost or live recommendations. Pinned GPU input and channels-last also changed a small number of values; channels-last was not a reliable speedup. A bit-exact pinned layout has not been demonstrated.

The $0.34/h 4090 community listing was not usable through direct SSH under the boot deadline. Secure 4090 and an early L40S attempt were interrupted; their guards terminated them, and neither yielded a benchmark archive. The final L40S community worker was the cheapest accessible eligible pool at selection ($0.79/h GPU). The A40 benchmark was never repeated. No account limit changes; cumulative spend includes cold starts and unsuccessful allocations.

## Accounting

{'pods': [], 'endpoints': [], 'network_volumes': [], 'ending_balance': '[redacted]', 'round_spend': '0.4226716342', 'cumulative_spend': '0.6196667593', 'spend_per_hour': 0, 'checked_at': '2026-11-22T14:37:39.270951+00:00'}

Reproduction: stage the same authorized camera metadata, JPEGs and three render/reference PNG pairs as round 1. Use the two profile scripts and constraints, with profile_worker.sh staged as pipeline/cuda/setup.sh; run the existing controller with RS_PROFILE=2, a private uncompressed profile.tar, original RS_START_BALANCE=60.4255764849 and the selected GPU/CLOUD/rate ceiling. The dead-man and idle timers are installed at pod creation. No commit or production port.
