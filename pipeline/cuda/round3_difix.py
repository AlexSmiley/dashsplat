"""Only alter skip batch at the actual VAE decode boundary; no change to two-view conditioning."""

import json, os, statistics, time, argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from pipeline.genfill.difix import Difix, PROMPT

ap = argparse.ArgumentParser()
ap.add_argument("--compiled", action="store_true")
args = ap.parse_args()
root = Path(os.environ["RS"]) / "bench"
dest = root / "results"
dest.mkdir(exist_ok=True)
m = Difix(ref=True, device="cuda", dtype=torch.float16).pipe


@torch.inference_mode()
def run():
    m.set_progress_bar_config(disable=True)
    emb = m.encode_prompt([PROMPT], torch.device("cuda"), 1, False)[0]
    torch.backends.cudnn.benchmark = True
    orig = m.vae.decode
    active = False
    encoder_batches = []
    decoder_batches = []

    def decode(z, *a, **k):
        skip = m.vae.decoder.incoming_skip_acts
        assert z.shape[0] == 1 and all(x.shape[0] == 2 for x in skip)
        encoder_batches.append(skip[0].shape[0])
        m.vae.decoder.incoming_skip_acts = [x[:1] for x in skip] if active else skip
        try:
            y = orig(z, *a, **k)
            decoder_batches.append(y[0].shape[0])
            assert y[0].shape[0] == (1 if active else 2)
            return y
        finally:
            m.vae.decoder.incoming_skip_acts = skip

    m.vae.decode = decode
    for w, h in [(1024, 576), (1472, 960)]:
        pairs = [
            [
                Image.open(root / tag / f"{n:04}.png").convert("RGB").resize((w, h))
                for tag in ["render", "ref"]
            ]
            for n in (0, 40, 80)
        ]

        def one(i):
            a, b = pairs[i]
            y = m(
                prompt_embeds=emb,
                image=a,
                ref_image=b,
                num_inference_steps=1,
                timesteps=[199],
                guidance_scale=0.0,
                generator=torch.Generator("cpu").manual_seed(0),
                width=w,
                height=h,
                output_type="np",
            ).images
            assert len(y) == 1 and np.isfinite(y).all()
            return np.clip(y[0] * 255 + 0.5, 0, 255).astype("uint8")

        active = False
        baselines = [one(i) for i in range(3)]
        orig_forward = m.vae.decoder.forward
        for case in ["kept_compile"] if args.compiled else ["stock", "kept"]:
            active = case != "stock"
            if args.compiled:
                m.vae.decoder.forward = torch.compile(orig_forward, dynamic=False)
            one(0)
            times = []
            diffs = []
            for rep in range(3):
                for i in range(3):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    v = one(i)
                    torch.cuda.synchronize()
                    times.append(time.perf_counter() - start)
                    if not rep:
                        d = np.abs(v.astype(np.int16) - baselines[i])
                        diffs.append(
                            dict(
                                max=int(d.max()),
                                mean=float(d.mean()),
                                changed_frac=float((d > 0).mean()),
                            )
                        )
            row = dict(
                model="Difix-ref",
                case=case,
                size=[w, h],
                fps=1 / statistics.median(times),
                samples_s=times,
                diffs=diffs,
                encode_batch=encoder_batches[-1],
                decode_batch=decoder_batches[-1],
                tiling=m.vae.use_tiling,
                slicing=m.vae.use_slicing,
            )
            print(row, flush=True)
            with (dest / "difix.jsonl").open("a") as f:
                f.write(json.dumps(row) + chr(10))
            if "RS_RECEIPT_FD" in os.environ:
                os.write(
                    int(os.environ["RS_RECEIPT_FD"]),
                    (json.dumps({"receipt": "difix", **row}) + chr(10)).encode(),
                )
        m.vae.decoder.forward = orig_forward


run()
