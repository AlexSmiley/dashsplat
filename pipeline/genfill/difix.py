"""Difix (NVIDIA Difix3D+, SD-Turbo single-step fixer) on Apple MPS, optionally reference-guided.

Weights: nvidia/difix_ref (render + reference photo) and nvidia/difix (render only), NVIDIA
licence, research and non-commercial use. Code: third_party/Difix3D/src/pipeline_difix.py.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

RS = Path(os.environ.get("RS", Path.home() / "dev/reality-splat")).resolve()
os.environ.setdefault("HF_HOME", str(RS / "third_party/hf"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(RS / "third_party/Difix3D/src"))

PROMPT = "remove degradation"


class Difix:
    def __init__(self, ref=True, device=None, dtype=torch.float16):
        from pipeline_difix import DifixPipeline

        self.name = "nvidia/difix_ref" if ref else "nvidia/difix"
        self.ref = ref
        self.device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.dtype = dtype
        self.pipe = DifixPipeline.from_pretrained(self.name, trust_remote_code=True, torch_dtype=dtype)
        self.pipe.set_progress_bar_config(disable=True)
        self.pipe.to(self.device)
        if ref and self.device == "cuda" and os.getenv("DIFIX_RENDER_ONLY_DECODE") == "1":
            decode = self.pipe.vae.decode
            decoder = self.pipe.vae.decoder

            def render_only(z, *a, **kw):
                skip = decoder.incoming_skip_acts
                if z.shape[0] != 1 or any(x.shape[0] != 2 for x in skip):
                    raise ValueError("expected one conditioned render/reference pair")
                decoder.incoming_skip_acts = [x[:1] for x in skip]
                try:
                    return decode(z, *a, **kw)
                finally:
                    decoder.incoming_skip_acts = skip

            self.pipe.vae.decode = render_only

    @torch.no_grad()
    def __call__(self, image, ref=None, timestep=199, seed=0):
        """image, ref: uint8 HxWx3 RGB, sides multiples of 64. Returns uint8 RGB of the same size."""
        from PIL import Image

        h, w = image.shape[:2]
        g = torch.Generator("cpu").manual_seed(seed)
        out = self.pipe(
            PROMPT,
            image=Image.fromarray(image),
            ref_image=Image.fromarray(ref) if self.ref and ref is not None else None,
            height=h,
            width=w,
            num_inference_steps=1,
            timesteps=[int(timestep)],
            guidance_scale=0.0,
            generator=g,
            output_type="np",
        ).images[0]
        return np.clip(out * 255 + 0.5, 0, 255).astype("B")


def sync(dev):
    if dev == "cuda":
        torch.cuda.synchronize()
    elif dev == "mps":
        torch.mps.synchronize()


def bench(size=(1024, 576), reps=6):
    import cv2 as cv

    root = RS / "third_party/Difix3D/assets"
    a = cv.imread(str(root / "example_input.png"))[:, :, ::-1]
    b = cv.imread(str(root / "example_ref.png"))[:, :, ::-1]
    a, b = [np.ascontiguousarray(cv.resize(x, size, interpolation=cv.INTER_AREA)) for x in (a, b)]
    out = {}
    for ref in (True, False):
        for dtype in (torch.float16, torch.float32):
            torch.mps.empty_cache()
            m = Difix(ref, dtype=dtype)
            times = []
            for i in range(reps + 2):
                sync(m.device)
                t0 = time.perf_counter()
                y = m(a, b)
                sync(m.device)
                times.append(time.perf_counter() - t0)
            key = f"{m.name} {str(dtype)[6:]} {size[0]}x{size[1]}"
            out[key] = dict(
                s_per_image=float(np.median(times[2:])),
                first=float(times[0]),
                driver_gb=torch.mps.driver_allocated_memory() / 1e9,
                tensor_gb=torch.mps.current_allocated_memory() / 1e9,
                nan=bool(np.isnan(y.astype("f")).any()),
                mean=float(y.mean()),
            )
            print(key, out[key], flush=True)
            cv.imwrite(str(RS / "work/genfill" / f"bench-{'ref' if ref else 'noref'}-{str(dtype)[6:]}.jpg"), y[:, :, ::-1])
            del m
    return out


if __name__ == "__main__":
    import json

    res = bench()
    (RS / "work/genfill/bench.json").write_text(json.dumps(res, indent=1))
