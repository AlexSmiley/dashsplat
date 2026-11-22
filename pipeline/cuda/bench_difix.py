"""Warm, synchronized Difix-ref inference on three independently selected in-window pairs."""

import argparse
import gc
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from pipeline.genfill.difix import Difix

p = argparse.ArgumentParser()
p.add_argument("--dtype", choices=["float16", "float32"], required=True)
p.add_argument("--small-only", action="store_true")
p.add_argument("--compile", action="store_true")
a = p.parse_args()
root = Path(os.environ["RS"]) / "bench"
out = root / "results"
out.mkdir(exist_ok=True)
m = Difix(device="cuda", dtype=getattr(torch, a.dtype))
if a.compile:
    m.pipe.unet = torch.compile(m.pipe.unet, fullgraph=False)
frames = sorted((root / "render").glob("*.png"))
for w, h in ((1472, 960), (1024, 576)):
    if a.small_only and w != 1024:
        continue
    pairs = [
        (
            Image.open(f).convert("RGB").resize((w, h)),
            Image.open(root / "ref" / f.name).convert("RGB").resize((w, h)),
        )
        for f in frames
    ]

    def run(ids):
        with torch.inference_mode():
            y = m.pipe(
                ["remove degradation"] * len(ids),
                image=[pairs[i][0] for i in ids],
                ref_image=[pairs[i][1] for i in ids],
                width=w,
                height=h,
                num_inference_steps=1,
                timesteps=[199],
                guidance_scale=0.0,
                generator=torch.Generator("cpu").manual_seed(0),
                output_type="np",
            ).images
        if len(y) != len(ids) or not np.isfinite(y).all():
            raise ValueError("invalid pair count or non-finite pixels")
        return np.clip(y * 255 + 0.5, 0, 255).astype("uint8")

    for batch in (1, 2, 4, 8):
        if a.compile and batch > 1:
            break
        ids = [i % len(pairs) for i in range(batch)]
        base = dict(size=[w, h], dtype=a.dtype, compile=a.compile, batch=batch)
        try:
            warm = run(ids)
            if batch > 1:
                delta = [
                    float(np.abs(im.astype(float) - run([idx])[0]).mean())
                    for im, idx in zip(warm, ids)
                ]
                if max(delta) > 2:
                    raise ValueError(f"batch failed single-pair MAE check: {delta}")
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            times = []
            for _ in range(5):
                t = time.perf_counter()
                run(ids)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - t)
            base.update(
                fps=batch / statistics.median(times),
                times_s=times,
                allocated_GB=torch.cuda.max_memory_allocated() / 1e9,
                reserved_GB=torch.cuda.max_memory_reserved() / 1e9,
            )
            if batch == 1 and w == 1472 and a.dtype == "float16" and not a.compile:
                for i, f in enumerate(frames):
                    ims = [
                        pairs[i][0],
                        Image.fromarray(run([i])[0]),
                        Image.open(root / "studio" / f.name),
                    ]
                    canvas = Image.new("RGB", (3 * 736, 504))
                    draw = ImageDraw.Draw(canvas)
                    for j, (im, label) in enumerate(
                        zip(
                            ims,
                            ["Render", "CUDA: raw Difix", "Studio: final composite"],
                        )
                    ):
                        canvas.paste(im.resize((736, 480)), (j * 736, 24))
                        draw.text((j * 736 + 5, 6), label)
                    canvas.save(out / ("compare-" + f.stem + ".jpg"), quality=90)
        except (torch.OutOfMemoryError, ValueError, RuntimeError) as e:
            if (
                isinstance(e, RuntimeError)
                and not isinstance(e, torch.OutOfMemoryError)
                and "non-singleton dimension 0" not in str(e)
            ):
                raise
            base.update(error=str(e).splitlines()[0])
            gc.collect()
            torch.cuda.empty_cache()
        with (out / "difix.jsonl").open("a") as f:
            f.write(json.dumps(base) + "\n")
        print(base, flush=True)
        if "error" in base:
            break
