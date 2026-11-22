# Task 3 retry: faster repair versus keeping the picture

**Decision:** use the render-only Difix-ref decoder when a one-level RGB difference is acceptable. Keep stock if exact bytes are required. NVIDIA Fixer is faster, but is not a visually equivalent replacement on these chase renders.

## Measured on RTX 6000 Ada (48 GB), not L40S

Neither L40S nor L40 was advertised as available at allocation. The 48 GB Ada workstation card was the closest available option; community allocation failed and one secure pod ran the experiment. Rates below use its $1.09/GPU-hour price, excluding disk, startup, encoding, rendering, and streaming. A driving minute is 1,800 generated frames. No new L40S performance is inferred.

| Model / configuration | Pixels | Median ms | FPS | GPU $/driving minute | Fresh every N frames at 30 Hz | Max / mean RGB difference |
|---|---|---:|---:|---:|---:|---:|
| Difix-ref stock | 1024×576 | 404.9 | 2.47 | 0.221 | 13 | 0 / 0.0000 |
| Difix-ref kept | 1024×576 | 291.0 | 3.44 | 0.159 | 9 | 1 / 0.0083 |
| Difix-ref stock | 1472×960 | 1317.8 | 0.76 | 0.718 | 40 | 0 / 0.0000 |
| Difix-ref kept | 1472×960 | 986.8 | 1.01 | 0.538 | 30 | 1 / 0.0081 |
| NVIDIA Fixer eager | 1024×576 | 261.4 | 3.83 | 0.142 | 8 | — |
| NVIDIA Fixer eager | 1472×960 | 235.3 | 4.25 | 0.128 | 8 | — |
| NVIDIA Fixer compiled | 1024×576 | 133.4 | 7.49 | 0.073 | 5 | 3 / 0.1487 |
| NVIDIA Fixer compiled | 1472×960 | 234.2 | 4.27 | 0.128 | 8 | 4 / 0.1381 |
| Difix-ref kept_compile | 1024×576 | 297.5 | 3.36 | 0.162 | 9 | 2 / 0.0212 |
| Difix-ref kept_compile | 1472×960 | 814.6 | 1.23 | 0.444 | 25 | 1 / 0.0186 |

Difix differences are against its stock two-view decoder on the same device and seed; compiled Fixer differences are against eager Fixer, **not** against the Studio. Means are across three frames and all 8-bit RGB channels. Both Difix inputs and UNet conditioning stay the same. At the decode boundary, one render latent meets either two sets of skips or only the render skips. The RGB output batch falls from two to one. Batch-dependent FP16 kernel selection/accumulation is consistent with the nonzero drift; the specific kernel was not traced. Neither slicing nor tiling was active.

Each number is a median of nine synchronized batch-one calls after one warmup, rotating the same three authorized frames; pre-sized PIL input through 8-bit CPU RGB. No offline PNG reading or initial resize in the clock. Fixer is NVIDIA’s pinned 0.6B Cosmos release in BF16, timestep 250; Difix is FP16, timestep 199, cached prompt, cuDNN benchmarking. Native Fixer used a true 1472×960 tensor, not a 576p upscale. The published Fixer card advertises 1024×576. Nine calls are not a sustained serving benchmark: eager 576p latency fell markedly during the sample. Compiled Fixer’s first timed call after warmup still took 1.7 s at both sizes. The worst of the remaining eight calls was 192 ms / 284 ms. At 30 Hz, use every **6th / 9th** frame for that observed jitter, after warmup, with asynchronous reprojection between completed fixes. The theoretical median budget is 5th / 8th. Reprojection quality, its compute cost and live playback were not tested.

## What the pictures show

On the three intersection frames, the Studio composite makes the signal mast, roadside signs and individual trees legible. Fixer preserves major arrows and stop lines but sometimes fades cars almost into the pavement, loses dark signal supports, and extends soft green hedge-like texture across background information. On the 3-second sequence it turns irregular foliage into dense, more uniform trees, supplies a smooth grass strip/sidewalk and cleaner roadway, and loses physical texture. This is plausible imagery, not verified reconstruction. The Studio composite is not ground truth either: it incorporates reference imagery, temporally carried repair and road compositing.

Eager Fixer was used for both sets of triptychs and both clips. Over 89 adjacent transitions, RGB mean absolute difference (plain / Studio / Fixer) was **4.67 / 5.09 / 3.29** at 1024×576 and **4.90 / 5.36 / 3.14** at 1472×960. Fixer changed less frame to frame but also suppressed detail: this motion-inclusive number does not establish superior temporal reconstruction or absence of flicker. Compiling Fixer changed the three eager outputs by up to 3 / 4 channel levels; it was not used to render the sequence.

## Smaller decoder options

- Tiling was already disabled in every Difix run. Disabling it again is a no-op.
- Half-resolution decode and upsample is not a switch: downsample the render latent and every matching encoder skip at its level, then upsample RGB. Small objects and road texture need separate validation. Running the whole pipeline at a smaller size is not this experiment.
- Channel pruning needs matched convolution, normalization, residual and skip-projection dimensions plus new weights and validation. Distillation needs a compact student with compatible skips (or a deliberately different interface), paired teacher targets, permitted training data and temporal/detail validation. Neither was trained or benchmarked.

[NVIDIA’s model card](https://huggingface.co/nvidia/Fixer/blob/ca20a25b55baa15dfc557db4c9b99168da8c2e14/README.md): “This model is ready for commercial/non-commercial use.” “Your use of the model is governed by the NVIDIA Open Model License Agreement.”

## Evidence and cleanup

[Streamed measurements and checksums](round3-retry-results.json). Media are in the git-ignored private [review folder](../../scratch/round3-fixer-retry/). All 26 media files matched their streamed SHA-256; both MP4s decoded as 90 frames, 30 fps, 3 seconds. Numeric rows also matched the independent post-run archive. The balance guard now retries unknown balance for up to 180 s; the earlier known-spend margin, parent-exit deletion and absolute caps are unchanged. The transfer pack was deleted. Retry spend $0.1540331963; cumulative $0.9522575185; all three resource lists empty, $0/h. No commits.
