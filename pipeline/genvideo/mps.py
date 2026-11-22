"""StreetNVS on Apple MPS. Keep the published weights, CFG, calibrated ray attention and scheduler."""

import argparse
import gc
import hashlib
import json
import os
import resource
import signal
import subprocess
import threading
import time
from pathlib import Path

import cv2 as cv
import torch
from diffsynth.models import wan_video_dit as wan
from diffsynth.models.wan_video_text_encoder import HuggingfaceTokenizer, WanTextEncoder
from diffsynth.models.wan_video_vae import WanVideoVAE38
from diffsynth.utils.data import save_video
from peft import LoraConfig, inject_adapter_in_model
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file
from streetnvs import model as geom
from streetnvs import pipeline as street
from streetnvs import pipeline_units as units
from streetnvs.data import load_sample

MiB = 1024**2
QUERY_CHUNK = 512
RS = Path.cwd()
WORK = RS / "work/genvideo"
MODELS = RS / "third_party/StreetNVS/models"


def sinusoidal(dim, position):
    base = 10000 ** (
        -torch.arange(dim // 2, device=position.device, dtype=torch.float32)
        / (dim // 2)
    )
    a = position.float()[:, None] * base
    return torch.cat((a.cos(), a.sin()), 1).to(position.dtype)


def rope(x, freqs, num_heads):
    # Real-valued formulation avoids both double and complex128 on Metal.
    q = x.float().unflatten(-1, (num_heads, -1)).unflatten(-1, (-1, 2))
    a, b = q.unbind(-1)
    c, s = freqs.real, freqs.imag
    return torch.stack((a * c - b * s, a * s + b * c), -1).flatten(2).to(x.dtype)


def sliced(q, k, v, num_heads, compatibility_mode=False):
    def heads(x):
        return x.unflatten(-1, (num_heads, -1)).transpose(1, 2).contiguous()

    q, k, v = map(heads, (q, k, v))
    out = []
    for part in q.split(QUERY_CHUNK, dim=2):
        out.append(torch.nn.functional.scaled_dot_product_attention(part, k, v))
    return torch.cat(out, 2).transpose(1, 2).flatten(2)


def install():
    # Upstream reference camera preparation broadcasts the FIRST camera's K to all references.
    # Keep the other cameras' distinct FOVs; the upstream release uses Waymo's near-uniform intrinsics.
    orig = units.prepare_camera_dit_input

    def cameras(**kw):
        if kw["is_video"]:
            return orig(**kw)
        items = []
        for i in range(len(kw["camera_Ks"])):
            items.append(
                orig(
                    **(
                        kw
                        | {
                            k: kw[k][i : i + 1]
                            for k in ("camera_Ks", "camera_w2cs", "frame_ids")
                        }
                    )
                )
            )
        return {
            k: torch.cat([it[k] for it in items], 0 if k == "frame_ids" else 1)
            for k in items[0]
        }

    units.prepare_camera_dit_input = cameras
    street.sinusoidal_embedding_1d = sinusoidal
    wan.rope_apply = rope
    wan.flash_attention = sliced
    geom.flash_attention = sliced


def text_emb(prompt, model_dir, dtype):
    key = hashlib.sha256(prompt.encode()).hexdigest()[:16]
    p = WORK / f"umt5-{key}.pt"
    if not p.exists():
        with torch.device("meta"):
            enc = WanTextEncoder()
        sd = torch.load(
            model_dir / "models_t5_umt5-xxl-enc-bf16.pth",
            mmap=True,
            map_location="cpu",
            weights_only=True,
        )
        enc.load_state_dict(sd, assign=True)
        enc.eval()
        tok = HuggingfaceTokenizer(
            name=str(model_dir / "google/umt5-xxl"), seq_len=512, clean="whitespace"
        )
        ids, mask = tok(prompt, return_mask=True, add_special_tokens=True)
        n = int(mask.sum())
        ctx = enc(ids[:, :n], mask[:, :n])
        ctx = torch.nn.functional.pad(ctx, (0, 0, 0, 512 - n))
        assert torch.isfinite(ctx).all()
        torch.save(ctx, p)
        del sd, enc
        gc.collect()
    return torch.load(p, weights_only=True).to("mps", dtype=dtype)


def get_vae(base, dtype):
    # Keep small non-parameter mean/std constants on CPU rather than meta.
    with torch.device("meta"):
        vae = WanVideoVAE38()
    with torch.device("cpu"):
        # Recover just the unregistered constants; avoid a second full VAE.
        from unittest.mock import patch

        with patch(
            "diffsynth.models.wan_video_vae.VideoVAE38_",
            return_value=torch.nn.Identity(),
        ):
            constants = WanVideoVAE38()
    vae.scale = constants.scale
    vae.mean, vae.std = constants.mean, constants.std
    state = torch.load(
        base / "Wan2.2_VAE.pth", weights_only=True, mmap=True, map_location="cpu"
    )
    state = vae.state_dict_converter().from_civitai(state)
    vae.load_state_dict(state, assign=True)
    return vae.eval().to("mps", dtype=dtype)


def get_dit(path, w, h, dtype, resident):
    with safe_open(str(path), framework="pt") as f:
        assert (f.metadata() or {}).get("streetnvs_format") == "2"
    with torch.device("meta"):
        dit = geom.StreetNVSModel(h, w)
        dit = inject_adapter_in_model(
            LoraConfig(r=1024, lora_alpha=1024, target_modules=["ffn.0", "ffn.2"]), dit
        )
    dit.freqs = tuple(f.to(torch.complex64) for f in wan.precompute_freqs_cis_3d(128))
    state = load_file(str(path), device="cpu")
    dit.load_state_dict(state, strict=True, assign=True)
    dit.eval()
    for name, mod in dit.named_children():
        if name != "geo_dit_blocks" or resident:
            mod.to("mps", dtype=dtype)
            continue
        for i, b in enumerate(mod):
            prefix = f"{name}.{i}."
            cpu = {
                k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)
            }

            def on(m, args):
                m.to("mps", dtype=dtype)

            b.register_forward_pre_hook(on)

            def off(m, args, out, sd=cpu):
                torch.mps.synchronize()
                m.load_state_dict(sd, assign=True)
                return out

            b.register_forward_hook(off)
    return dit


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--window", default="000-81")
    p.add_argument("--size", type=int, nargs=2, default=[448, 256])
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--cfg", type=float, default=3.5)
    p.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--previous",
        type=Path,
        help="Previous 81-frame window; use its overlapping frame 69 as I2V anchor",
    )
    p.add_argument("--resident", action="store_true")
    p.add_argument("--tiled", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--checkpoint-steps",
        action="store_true",
        help="Save denoising progress on the shared Studio for a memory-pressure restart",
    )
    p.add_argument("--inputs", type=Path, default=WORK / "inputs")
    p.add_argument("--refs", type=int, default=5)
    p.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        help="Actual flow-noise fraction at the start; below 1 initializes from the RGB-control latent",
    )
    p.add_argument("--cond-scale", type=float, default=1.0)
    p.add_argument(
        "--codec-out",
        type=Path,
        help="Also decode the unnoised RGB latent (separately timed diagnostic)",
    )
    p.add_argument("--query-chunk", type=int, default=512)
    p.add_argument("--cache-controls", action="store_true")
    a = p.parse_args()
    global QUERY_CHUNK
    QUERY_CHUNK = a.query_chunk
    assert 0 < a.sigma <= 1 and a.cond_scale > 0
    assert torch.backends.mps.is_available()
    torch.set_num_threads(4)
    torch.mps.set_per_process_memory_fraction(0.27 if a.resident else 0.13)
    w, h = a.size
    assert not (w % 32 or h % 32)
    n = int(a.window.split("-")[1])
    base = MODELS / "Wan2.2-TI2V-5B"
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[a.dtype]
    install()
    start = time.monotonic()
    stages = {}
    driver = []
    done = threading.Event()
    pressures = []

    def monitor():
        next_check = 0
        while not done.is_set():
            now = time.monotonic()
            if now >= next_check:
                next_check = now + 5
                level = int(
                    subprocess.check_output(
                        ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"]
                    ).strip()
                )
                vm = subprocess.check_output(["vm_stat"], text=True)
                swaps = int(
                    next(
                        l.split(":")[1].strip().rstrip(".")
                        for l in vm.splitlines()
                        if l.startswith("Swapouts:")
                    )
                )
                pressures.append((level, swaps))
                if level >= 4 or (
                    len(pressures) > 2
                    and level >= 2
                    and pressures[-2][0] >= 2
                    and swaps > pressures[-3][1]
                ):
                    print(
                        "Stopping: shared Mac under sustained paging pressure",
                        flush=True,
                    )
                    os.kill(os.getpid(), signal.SIGINT)
                    return
            driver.append(torch.mps.driver_allocated_memory())
            done.wait(0.25)

    threading.Thread(target=monitor, daemon=True).start()

    def stamp(s):
        torch.mps.synchronize()
        stages[s] = time.monotonic() - start
        print(
            s,
            round(stages[s], 2),
            "s",
            round(torch.mps.driver_allocated_memory() / MiB),
            "MiB driver",
            flush=True,
        )

    sample, _ = load_sample(
        a.inputs / a.window, "0", "chase", 0, h, w, [str(i) for i in range(a.refs)], n
    )
    if a.previous:
        cap = cv.VideoCapture(str(a.previous))
        cap.set(cv.CAP_PROP_POS_FRAMES, 69)
        ok, im = cap.read()
        cap.release()
        assert ok and a.window == "069-81"
        sample["input_image"] = Image.fromarray(
            cv.cvtColor(im, cv.COLOR_BGR2RGB)
        ).resize((w, h))
    ctx = text_emb(sample["prompt"], base, dtype)
    neg = text_emb("", base, dtype) if a.cfg != 1 else ctx
    stamp("text")
    pipe = street.StreetNVSPipeline(device="mps", torch_dtype=dtype)
    pipe.image_height, pipe.image_width = h, w
    cache_path = None
    if a.cache_controls:
        files = sorted(p for p in (a.inputs / a.window).iterdir() if p.is_file()) + (
            [a.previous] if a.previous else []
        )
        key = hashlib.sha256(
            str([w, h, a.refs, a.seed, a.dtype, a.tiled, torch.__version__]).encode()
        )
        for path in files:
            key.update(hashlib.sha256(path.read_bytes()).digest())
        cache_path = a.output.parent / ("controls-" + key.hexdigest()[:24] + ".pt")
    cached = cache_path is not None and cache_path.exists()
    if not cached:
        pipe.vae = get_vae(base, dtype)
    stamp("vae load")
    # Use released units; skip prompt unit after exact mmap-backed T5 embedding above.
    shared = dict(
        **sample,
        height=h,
        width=w,
        num_frames=n,
        seed=a.seed,
        rand_device="cpu",
        cfg_scale=a.cfg,
        tiled=a.tiled,
        tile_size=(12, 16),
        tile_stride=(8, 12),
        cond_scale=a.cond_scale,
    )
    pos = {"context": ctx}
    neg = {"context": neg}
    if cached:
        restored = torch.load(cache_path, map_location="cpu", weights_only=True)
        shared.update({k: v.to("mps") for k, v in restored.items()})
        stamp("cached controls")
    else:
        for unit in pipe.units[:-1]:
            if type(unit).__name__.endswith("PromptEmbedder"):
                continue
            shared, pos, neg = pipe.unit_runner(unit, pipe, shared, pos, neg)
            stamp(type(unit).__name__)
        if cache_path:
            tmp = cache_path.with_suffix(".tmp")
            torch.save(
                {
                    k: shared[k].cpu()
                    for k in (
                        "latents",
                        "first_frame_latents",
                        "camref_latents",
                        "cond_context",
                    )
                },
                tmp,
            )
            tmp.replace(cache_path)
    pipe.vae = None
    gc.collect()
    torch.mps.empty_cache()
    pipe.geo_dit = get_dit(
        MODELS / "StreetNVS/streetnvs.safetensors", w, h, dtype, a.resident
    )
    stamp("dit load")
    shared, pos, neg = pipe.unit_runner(pipe.units[-1], pipe, shared, pos, neg)
    strength = a.sigma / (5 - 4 * a.sigma)
    pipe.scheduler.set_timesteps(a.steps, denoising_strength=strength, shift=5.0)
    assert abs(float(pipe.scheduler.sigmas[0]) - a.sigma) < 1e-4
    nr = len(sample["camref_images"])
    cwidth = shared["latents"].shape[1]
    # Unit_COND stores [RGB, hole-mask, depth] in contiguous latent channels.
    clean = shared["cond_context"][:, :cwidth, nr:].contiguous()
    if a.sigma < 1:
        lat = shared["latents"]
        lat[:, :, nr:] = pipe.scheduler.add_noise(
            clean, lat[:, :, nr:], pipe.scheduler.timesteps[0]
        )
        lat[:, :, nr : nr + 1] = shared["first_frame_latents"]
    stamp("controls ready")
    progress = a.output.with_suffix(".partial.pt")
    meta = [str(a.window), w, h, a.seed, a.steps, a.cfg, a.dtype, a.tiled]
    if a.checkpoint_steps:
        inputs = a.inputs / a.window
        paths = sorted(q for q in inputs.iterdir() if q.is_file()) + (
            [a.previous] if a.previous else []
        )
        if a.sigma != 1 or a.cond_scale != 1 or a.refs != 5:
            meta += [a.sigma, a.cond_scale, a.refs]
        meta += [hashlib.sha256(q.read_bytes()).hexdigest() for q in paths]
    begin = 0
    prior_time = 0.0
    if a.checkpoint_steps and progress.exists():
        resume = torch.load(progress, map_location="cpu", weights_only=True)
        assert resume["meta"] == meta
        begin = resume["next"]
        prior_time = resume["seconds"]
        shared["latents"] = resume["latents"].to("mps")
        print("restored step", begin, flush=True)
    lat = shared["latents"]
    for i, t in enumerate(pipe.scheduler.timesteps):
        if i < begin:
            continue
        tv = t.unsqueeze(0).to("mps", dtype=dtype)
        pred = street.model_fn_wan_video(pipe.geo_dit, **shared, **pos, timestep=tv)
        if a.cfg != 1:
            pn = street.model_fn_wan_video(pipe.geo_dit, **shared, **neg, timestep=tv)
            pred = pn + a.cfg * (pred - pn)
        lat = pipe.scheduler.step(pred, t, shared["latents"])
        lat[:, :, :nr] = shared["camref_latents"]
        lat[:, :, nr : nr + 1] = shared["first_frame_latents"]
        shared["latents"] = lat
        assert torch.isfinite(lat).all(), i
        if a.query_chunk < 512:
            torch.mps.empty_cache()
        stamp(f"denoise {i + 1}")
        if a.checkpoint_steps:
            tmp = progress.with_suffix(".tmp")
            torch.save(
                {
                    "meta": meta,
                    "next": i + 1,
                    "seconds": prior_time + time.monotonic() - start,
                    "latents": lat.cpu(),
                },
                tmp,
            )
            tmp.replace(progress)
    lat = lat[:, :, nr:]
    pipe.geo_dit = None
    del shared
    gc.collect()
    torch.mps.empty_cache()
    pipe.vae = get_vae(base, dtype)
    out = pipe.vae.decode(
        lat, device="mps", tiled=a.tiled, tile_size=(12, 16), tile_stride=(8, 12)
    )
    assert torch.isfinite(out).all()
    imgs = pipe.vae_output_to_video(out)
    assert len(imgs) == n
    a.output.parent.mkdir(parents=True, exist_ok=True)
    save_video(imgs, str(a.output), fps=10, quality=9)
    stamp("saved")
    done.set()
    a.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "arguments": vars(a),
                "stages": stages,
                "wall_s": stages["saved"] + prior_time,
                "resumed_at_step": begin,
                "frames": n,
                "peak_driver_MiB": max(driver) / MiB,
                "system_vm_pressure_max": max(p[0] for p in pressures),
                "system_swapout_delta_pages": pressures[-1][1] - pressures[0][1],
                "peak_process_rss_MiB": resource.getrusage(
                    resource.RUSAGE_SELF
                ).ru_maxrss
                / MiB,
                "torch": torch.__version__,
            },
            indent=2,
            default=str,
        )
    )

    if a.codec_out:
        clock = time.monotonic()
        auto = pipe.vae.decode(
            clean, device="mps", tiled=a.tiled, tile_size=(12, 16), tile_stride=(8, 12)
        )
        save_video(pipe.vae_output_to_video(auto), str(a.codec_out), fps=10, quality=9)
        print(
            "RGB VAE reconstruction diagnostic",
            round(time.monotonic() - clock, 2),
            "s extra",
            flush=True,
        )


if __name__ == "__main__":
    with torch.inference_mode():
        main()
