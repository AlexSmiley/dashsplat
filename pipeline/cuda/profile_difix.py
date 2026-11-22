"""Real-frame, one-step ablation. Benchmark instrumentation only; no weight or pipeline edits."""

import argparse, contextlib, functools, json, os, statistics, subprocess, threading, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from pipeline.genfill.difix import Difix, PROMPT

args = argparse.ArgumentParser()
args.add_argument("--width", type=int, required=True)
args.add_argument("--ref", type=int, required=True)
args.add_argument("--compile-only", action="store_true")
a = args.parse_args()
root = Path(os.environ["RS"]) / "bench"
results = root / "results"
results.mkdir(exist_ok=True)
w = a.width
h = {1472: 960, 1024: 576}[w]
ref = bool(a.ref)
dest = results / f"difix-{w}-{int(ref)}.jsonl"


def emit(**fields):
    with dest.open("a") as f:
        f.write(json.dumps(fields) + chr(10))
    print(fields.get("case"), fields.get("ms"), flush=True)


class Meter:
    def __init__(self):
        self.active = False
        self.parts = Counter()
        self.shapes = {}
        self.hooks = []

    def wrap(self, obj, name, key):
        original = getattr(obj, name)

        @functools.wraps(original)
        def run(*args, **kw):
            if not self.active:
                return original(*args, **kw)
            torch.cuda.synchronize()
            t = time.perf_counter()
            y = original(*args, **kw)
            torch.cuda.synchronize()
            self.parts[key] += 1000 * (time.perf_counter() - t)
            if key == "vae_encode":
                self.shapes["encoder_input"] = [list(args[0].shape), str(args[0].dtype)]
            if key == "vae_decode":
                self.shapes["decoded"] = [list(y[0].shape), str(y[0].dtype)]
            return y

        setattr(obj, name, run)
        self.hooks.append((obj, name, original))

    def remove(self):
        for o, n, f in self.hooks:
            setattr(o, n, f)


def param_types(module):
    return dict(Counter(str(p.dtype) for p in module.parameters()))


def procs(module):
    return dict(Counter(type(p).__name__ for p in module.attn_processors.values()))


def clock():
    torch.cuda.synchronize()
    return time.perf_counter()


def loadpair(f):
    dirs = ["render", "ref"] if ref else ["render"]
    out = []
    for d in dirs:
        with Image.open(root / d / f.name) as im:
            out.append(im.convert("RGB").resize((w, h)))
    return out


smis = []
proc = subprocess.Popen(
    [
        "nvidia-smi",
        "--query-gpu=utilization.gpu",
        "--format=csv,noheader,nounits",
        "-lms",
        "200",
    ],
    stdout=subprocess.PIPE,
    text=True,
)


def collect():
    for line in proc.stdout:
        if line.strip().isdigit():
            smis.append((time.perf_counter(), int(line)))


threading.Thread(target=collect, daemon=True).start()


def main():
    pipe = Difix(ref=ref, device="cuda", dtype=torch.float16).pipe
    files = sorted((root / "render").glob("*.png"))
    assert len(files) == 3
    pairs = []
    loads = []
    for f in files:
        t = time.perf_counter()
        pairs.append(loadpair(f))
        loads.append(1000 * (time.perf_counter() - t))
    host = [
        [torch.from_numpy(np.array(im, copy=True)).pin_memory() for im in pair]
        for pair in pairs
    ]
    m = Meter()
    knobs = {"prompt": False, "tensor": False, "layout": False}
    cached = None

    def one(i):
        pair = host[i] if knobs["tensor"] else pairs[i]
        if knobs["tensor"]:
            pair = [
                (
                    u.to("cuda", non_blocking=True).permute(2, 0, 1)[None].float()
                    / 255
                    * 2
                    - 1
                ).to(torch.float16)
                for u in pair
            ]
            if knobs["layout"]:
                pair = [u.contiguous(memory_format=torch.channels_last) for u in pair]
        kw = {"prompt_embeds": cached} if knobs["prompt"] else {"prompt": [PROMPT]}
        y = pipe(
            **kw,
            image=pair[0],
            ref_image=pair[1] if ref else None,
            width=w,
            height=h,
            num_inference_steps=1,
            timesteps=[199],
            guidance_scale=0.0,
            generator=torch.Generator("cpu").manual_seed(0),
            output_type="np",
        ).images
        if len(y) != 1 or not np.isfinite(y).all():
            raise ValueError("invalid image")
        return np.clip(y[0] * 255 + 0.5, 0, 255).astype(np.uint8)

    baseline = [one(i) for i in range(3)]

    def measure(name):
        one(0)
        begin = clock()
        times = []
        diffs = []
        for rep in range(2):
            for i in range(3):
                t = clock()
                y = one(i)
                times.append(1000 * (clock() - t))
                if rep == 0:
                    d = np.abs(y.astype(np.int16) - baseline[i])
                    diffs.append([int(d.max()), float(d.mean())])
        end = clock()
        util = [v for t, v in smis if begin <= t <= end]
        row = dict(
            case=name,
            ms=statistics.median(times),
            samples_ms=times,
            diff_max_mean=diffs,
            util_pct=statistics.mean(util) if util else None,
            util_n=len(util),
        )
        emit(**row)
        return row

    if not a.compile_only:
        emit(
            case="hardware",
            gpu=torch.cuda.get_device_name(),
            host_cpus=os.cpu_count(),
            affinity_cpus=len(os.sched_getaffinity(0)),
            cpu_quota=Path("/sys/fs/cgroup/cpu.max").read_text().strip()
            if Path("/sys/fs/cgroup/cpu.max").exists()
            else None,
            torch_cpu_threads=torch.get_num_threads(),
            disk_decode_resize_ms=loads,
            unet_dtype=param_types(pipe.unet),
            vae_dtype=param_types(pipe.vae),
            text_dtype=param_types(pipe.text_encoder),
            unet_attention=procs(pipe.unet),
            vae_attention=procs(pipe.vae),
        )
        for o, n, k in [
            (pipe, "encode_prompt", "prompt"),
            (pipe.text_encoder, "forward", "text_model"),
            (pipe.image_processor, "preprocess", "pre"),
            (pipe, "prepare_latents", "prepare"),
            (pipe.vae, "encode", "vae_encode"),
            (pipe.unet, "forward", "unet"),
            (pipe.vae, "decode", "vae_decode"),
            (pipe.image_processor, "postprocess", "post"),
            (pipe.image_processor, "pt_to_numpy", "transfer"),
        ]:
            m.wrap(o, n, k)
        traces = []
        m.active = True
        for i in range(3):
            m.parts.clear()
            t = clock()
            one(i)
            total = 1000 * (clock() - t)
            r = dict(m.parts)
            r["prepare"] -= r["vae_encode"]
            r["post"] -= r["transfer"]
            r["text_forward_ms"] = r.pop("text_model")
            r["other"] = total - sum(v for k, v in r.items() if k != "text_forward_ms")
            r["total"] = total
            traces.append(r)
        m.active = False
        m.remove()
        emit(case="profile", samples=traces, shapes=m.shapes)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        ) as prof:
            one(0)
        emit(
            case="attention_dispatch",
            ops={
                k.key: k.count
                for k in prof.key_averages()
                if "attention" in k.key or "scaled_dot" in k.key
            },
        )
        measure("stock")
    oldrows = [json.loads(line) for line in dest.read_text().splitlines()]

    def acceptable(r):
        return (
            max(x[0] for x in r["diff_max_mean"]) <= 1
            and max(x[1] for x in r["diff_max_mean"]) <= 0.02
        )

    def trial(name, apply, undo):
        if a.compile_only:
            row = next(r for r in oldrows if r.get("case") == name)
            if acceptable(row):
                apply()
            return
        apply()
        row = measure(name)
        if not acceptable(row):
            undo()

    cached = pipe.encode_prompt([PROMPT], torch.device("cuda"), 1, False)[0]
    trial(
        "cached_prompt",
        lambda: knobs.update(prompt=True),
        lambda: knobs.update(prompt=False),
    )
    orig_pre = pipe.image_processor.preprocess

    def tensors(on):
        knobs["tensor"] = on
        pipe.image_processor.preprocess = (lambda x, *a, **k: x) if on else orig_pre

    trial("pinned_gpu_input", lambda: tensors(True), lambda: tensors(False))

    def layout(on):
        knobs["layout"] = on
        fmt = torch.channels_last if on else torch.contiguous_format
        for mod in (pipe.unet, pipe.vae):
            mod.to(memory_format=fmt)

    trial("channels_last", lambda: layout(True), lambda: layout(False))
    # Preserve the custom multi-view transformer; swap only standard attention processors if necessary.
    from diffusers.models.attention_processor import AttnProcessor, AttnProcessor2_0

    before = [(mod, dict(mod.attn_processors)) for mod in (pipe.unet, pipe.vae)]

    def sdpa():
        for mod, pr in before:
            mod.set_attn_processor(
                {
                    k: AttnProcessor2_0() if type(v) == AttnProcessor else v
                    for k, v in pr.items()
                }
            )

    def restore():
        for mod, pr in before:
            mod.set_attn_processor(dict(pr))

    trial("sdpa", sdpa, restore)
    if not a.compile_only:
        assert all(
            p.dtype == torch.float16
            for mod in [pipe.vae, pipe.unet, pipe.text_encoder]
            for p in mod.parameters()
        )
        measure("fp16_verified")
        measure("vae_pair_already_batched" if ref else "vae_single")
    trial(
        "cudnn_benchmark",
        lambda: setattr(torch.backends.cudnn, "benchmark", True),
        lambda: setattr(torch.backends.cudnn, "benchmark", False),
    )
    if a.compile_only:
        stages = next(r["samples"] for r in oldrows if r["case"] == "profile")
        key = max(
            ["vae_encode", "vae_decode", "unet"],
            key=lambda k: sum(t[k] for t in stages),
        )
        mod = {
            "vae_encode": pipe.vae.encoder,
            "vae_decode": pipe.vae.decoder,
            "unet": pipe.unet,
        }[key]
        orig = mod.forward
        mod.forward = torch.compile(orig, dynamic=False)
        start = clock()
        one(0)
        emit(case="compile_cold", module=key, cold_ms=(clock() - start) * 1000)
        measure("compile_" + key)


try:
    with torch.inference_mode():
        main()
finally:
    proc.terminate()
    proc.wait()
