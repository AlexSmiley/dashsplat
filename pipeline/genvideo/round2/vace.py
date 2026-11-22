"""Alternative: released VACE-Wan2.1 1.3B, same 81 dense views and opaque mask, no output compositing."""

import gc
import json
import os
import resource
import signal
import subprocess
import threading
import time
from pathlib import Path

import torch
from diffsynth.models import wan_video_dit as wan
from diffsynth.models.wan_video_vace import VaceWanModel
from diffsynth.models.wan_video_vae import WanVideoVAE
from diffsynth.pipelines import wan_video as pipe_lib
from diffsynth.utils.data import save_video
from diffsynth.utils.state_dict_converters.wan_video_dit import (
    WanVideoDiTStateDictConverter,
)
from diffsynth.utils.state_dict_converters.wan_video_vae import (
    WanVideoVAEStateDictConverter,
)
from PIL import Image
from safetensors.torch import load_file

from pipeline.genvideo.mps import MODELS, MiB, rope, sinusoidal, sliced, text_emb


@torch.inference_mode()
def run():
    p = Path("work/genvideo/round2")
    w = p / "vace-weights"
    raw = p / "raw"
    native = os.getenv("VACE_NATIVE") == "1"
    wpx, hpx = (832, 480) if native else (416, 240)
    stem = "vace-native" if native else "vace"
    dtype = torch.bfloat16
    device = "mps"
    torch.set_num_threads(4)
    torch.mps.set_per_process_memory_fraction(0.20)
    start = time.monotonic()
    peak = []
    done = threading.Event()
    levels = []

    def watch():
        while not done.is_set():
            level = int(
                subprocess.check_output(
                    ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"]
                )
            )
            levels.append(level)
            peak.append(torch.mps.driver_allocated_memory())
            if level >= 4 or levels[-3:] == [2, 2, 2]:
                print("Stopping: shared Mac under sustained pressure", flush=True)
                os.kill(os.getpid(), signal.SIGINT)
                return
            done.wait(2)

    threading.Thread(target=watch, daemon=True).start()
    wan.rope_apply = rope
    wan.flash_attention = sliced
    pipe_lib.sinusoidal_embedding_1d = sinusoidal
    pipe = pipe_lib.WanVideoPipeline(device=device, torch_dtype=dtype)

    def vae():
        v = WanVideoVAE()
        sd = torch.load(w / "Wan2.1_VAE.pth", map_location="cpu", weights_only=True)
        v.load_state_dict(WanVideoVAEStateDictConverter(sd), strict=True, assign=True)
        return v.eval().to(device, dtype)

    pipe.vae = vae()
    ims = [
        Image.open(raw / f"{i:03d}-view.png")
        .convert("RGB")
        .resize((wpx, hpx), Image.Resampling.BILINEAR)
        for i in range(81)
    ]
    mask = [
        Image.open(raw / f"{i:03d}-hole.png")
        .convert("RGB")
        .resize((wpx, hpx), Image.Resampling.NEAREST)
        for i in range(81)
    ]
    options = {"tiled": True, "tile_size": (24, 32), "tile_stride": (16, 24)}
    u = pipe_lib.WanVideoUnit_VACE().process(
        pipe, ims, mask, None, 1.0, hpx, wpx, 81, **options
    )
    u["vace_context"] = u["vace_context"].to(device, dtype)
    clean = pipe.vae.encode(pipe.preprocess_video(ims), device=device, **options).to(
        device, dtype
    )
    pipe.vae = None
    gc.collect()
    torch.mps.empty_cache()
    with torch.device("meta"):
        dit = wan.WanModel(
            has_image_input=False,
            patch_size=(1, 2, 2),
            in_dim=16,
            dim=1536,
            ffn_dim=8960,
            freq_dim=256,
            text_dim=4096,
            out_dim=16,
            num_heads=12,
            num_layers=30,
            eps=1e-6,
        )
        adapter = VaceWanModel()
    dit.freqs = tuple(f.to(torch.complex64) for f in wan.precompute_freqs_cis_3d(128))
    sd = load_file(w / "diffusion_pytorch_model.safetensors", device="cpu")
    dit.load_state_dict(WanVideoDiTStateDictConverter(sd), strict=True, assign=True)
    adapter.load_state_dict(
        {k: v for k, v in sd.items() if k.startswith("vace")}, strict=True, assign=True
    )
    del sd
    dit = dit.eval().to(device, dtype)
    adapter = adapter.eval().to(device, dtype)
    ctx = text_emb(
        "Photographic daylight, asphalt, grass, tree canopy and sky.",
        MODELS / "Wan2.2-TI2V-5B",
        dtype,
    )
    sigma = 0.45
    steps = 12
    pipe.scheduler.set_timesteps(
        steps, denoising_strength=sigma / (5 - 4 * sigma), shift=5
    )
    assert abs(float(pipe.scheduler.sigmas[0]) - sigma) < 1e-5
    noise = torch.randn(clean.shape, generator=torch.Generator().manual_seed(42)).to(
        device, dtype
    )
    lat = pipe.scheduler.add_noise(clean, noise, pipe.scheduler.timesteps[0])
    for i, t in enumerate(pipe.scheduler.timesteps):
        pred = pipe_lib.model_fn_wan_video(
            dit,
            vace=adapter,
            latents=lat,
            timestep=t[None].to(device, dtype),
            context=ctx,
            **u,
        )
        lat = pipe.scheduler.step(pred, t, lat)
        assert torch.isfinite(lat).all()
        print("denoise", i + 1, round(time.monotonic() - start, 1), flush=True)
    del dit, adapter, u
    gc.collect()
    torch.mps.empty_cache()
    pipe.vae = vae()
    decoded = pipe.vae.decode(lat, device=device, **options)
    assert torch.isfinite(decoded).all()
    ims = pipe.vae_output_to_video(decoded)
    assert len(ims) == 81
    save_video(ims, str(p / (stem + ".mp4")), fps=10, quality=9)
    done.set()
    (p / (stem + ".json")).write_text(
        json.dumps(
            {
                "wall_s": time.monotonic() - start,
                "peak_driver_MiB": max(peak) / MiB,
                "peak_process_rss_MiB": resource.getrusage(
                    resource.RUSAGE_SELF
                ).ru_maxrss
                / MiB,
                "sigma": sigma,
                "steps": steps,
                "seed": 42,
                "opaque_protection": "VACE latent context only; no pixel compositing",
                "resolution": [wpx, hpx],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    run()
