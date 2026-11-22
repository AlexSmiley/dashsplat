"""NVIDIA's published no-reference inference. Benchmark real inputs + make private, frame-aligned evidence."""

import argparse, hashlib, json, os, shutil, statistics, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

rs = Path(os.environ["RS"])
root = rs / "bench"
out = root / "results"
out.mkdir(exist_ok=True)
sys.path.insert(0, str(rs / "third_party/Fixer/src"))
from inference_pretrained_model import (
    load_and_compile_model,
    preprocess_image,
    postprocess_output,
    model_inference,
)

ap = argparse.ArgumentParser()
ap.add_argument("--width", type=int, required=True)
ap.add_argument("--compile", action="store_true")
ap.add_argument(
    "--stage", choices=["probe", "speed", "three", "sequence"], default="speed"
)
ap.add_argument("--upscale", action="store_true")
a = ap.parse_args()
w = a.width
h = {1024: 576, 1472: 960}[w]
size = (w, h)
model_size = (1024, 576) if a.upscale else size
ckpt = "/work/models/pretrained/pretrained_fixer.pkl"
dtype = torch.bfloat16
device = torch.device("cuda")
if a.stage != "three":
    model = load_and_compile_model(ckpt, 250, False, 1, device, dtype, compile=False)
    model.eval()
    model.set_eval()
    torch.backends.cudnn.benchmark = False
    if a.compile:
        model = torch.compile(model, dynamic=False)


def image(path):
    with Image.open(path) as im:
        return im.convert("RGB")


def emit(name, **r):
    with (out / name).open("a") as f:
        f.write(json.dumps(r) + chr(10))
    print(r, flush=True)
    if "RS_RECEIPT_FD" in os.environ:
        os.write(
            int(os.environ["RS_RECEIPT_FD"]),
            (json.dumps({"receipt": name, **r}) + chr(10)).encode(),
        )


@torch.inference_mode()
def one(im):
    if a.upscale:
        im = im.resize(model_size, Image.Resampling.BILINEAR)
    x = preprocess_image(im, device, dtype)
    y = model_inference(model, 1, model_size[1], model_size[0], dtype, device, x=x)
    assert tuple(y.shape) == (1, 3, model_size[1], model_size[0])
    if not torch.isfinite(y).all():
        raise ValueError("nonfinite Fixer image")
    return postprocess_output(y, size)


def publish(path):
    if "RS_RECEIPT_FD" not in os.environ:
        return
    with os.fdopen(os.dup(int(os.environ["RS_RECEIPT_FD"])), "wb") as wire:
        with path.open("rb") as f:
            digest = hashlib.sha256()
            for block in iter(lambda: f.read(65536), b""):
                digest.update(block)
            sha = digest.hexdigest()
            f.seek(0)
            wire.write(
                (
                    json.dumps(
                        {
                            "media": path.name,
                            "sha256": sha,
                            "bytes": path.stat().st_size,
                        }
                    )
                    + chr(10)
                ).encode()
            )
            shutil.copyfileobj(f, wire)


def triptych(ims, cell, labels):
    width, height = cell
    canvas = Image.new("RGB", (width * 3, height + 32), (22, 22, 22))
    d = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    for j, (im, label) in enumerate(zip(ims, labels)):
        canvas.paste(im.resize(cell, Image.Resampling.LANCZOS), (j * width, 32))
        d.text((j * width + 7, 6), label, font=font, fill="white")
    return canvas


fixed_samples = {}
frames = [0, 40, 80]
small = [
    image(root / "render" / f"{n:04}.png").resize(size, Image.Resampling.BILINEAR)
    for n in frames
]
case = ("compiled" if a.compile else "eager") + ("-576p-upscale" if a.upscale else "")
cache = root / "fixer-baseline"
cache.mkdir(exist_ok=True)
if a.stage == "probe":
    z = one(small[0])
    assert z.size == size
    print("actual Fixer inference passed", size, flush=True)
    sys.exit(0)
if a.stage == "speed":
    torch.cuda.synchronize()
    tc = time.perf_counter()
    one(small[0])
    torch.cuda.synchronize()
    warm = time.perf_counter() - tc
    times = []
    for rep in range(3):
        for n, im in zip(frames, small):
            torch.cuda.synchronize()
            t = time.perf_counter()
            z = one(im)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t)
            if not rep:
                fixed_samples[n] = z
    delta = []
    for n, z in fixed_samples.items():
        path = cache / f"{n}-{w}-{int(a.upscale)}.png"
        if not a.compile:
            z.save(path)
        elif path.is_file():
            d = np.abs(
                np.asarray(z, dtype=np.int16) - np.asarray(image(path), dtype=np.int16)
            )
            delta.append(dict(max=int(d.max()), mean=float(d.mean())))
    emit(
        "fixer.jsonl",
        compiled_vs_eager=delta,
        case=case,
        size=size,
        model_size=model_size,
        timestep=250,
        fps=1 / statistics.median(times),
        samples_s=times,
        warm_s=warm,
        torch=torch.__version__,
        gpu=torch.cuda.get_device_name(),
        dtype=str(dtype),
    )
    sys.exit(0)
media = out / "review"
media.mkdir(exist_ok=True)
label = f"NVIDIA Fixer | {w} x {h}"
labels = ["Plain render", "Studio Difix-ref composite", label]
seq = root / "sequence"
rows = json.loads((seq / "frames.json").read_text())["frames"]
assert len(rows) == 90
if a.stage == "three":
    for n, plain in zip(frames, small):
        z = image(cache / f"{n}-{w}-{int(a.upscale)}.png")
        studio = image(root / "studio" / f"{n:04}.png").resize(
            size, Image.Resampling.BILINEAR
        )
        plate = triptych([plain, studio, z], size, labels)
        plate.save(media / f"frame-{n:04}-{w}.jpg", quality=94, subsampling=0)
        publish(media / f"frame-{n:04}-{w}.jpg")
        plate.thumbnail((1500, 650))
        plate.save(media / f"inspect-{n:04}-{w}.jpg", quality=94)
        publish(media / f"inspect-{n:04}-{w}.jpg")
    sys.exit(0)
# Write panel images directly to an encoder on the pod; metrics use pre-codec RGB at a common grid.
cell = (640, 360) if w == 1024 else (640, 416)
ff = [
    "ffmpeg",
    "-loglevel",
    "error",
    "-y",
    "-f",
    "rawvideo",
    "-pix_fmt",
    "rgb24",
    "-s",
    f"{3 * cell[0]}x{cell[1] + 32}",
    "-r",
    "30",
    "-i",
    "-",
    "-an",
    "-c:v",
    "libx264",
    "-threads",
    "3",
    "-preset",
    "veryfast",
    "-crf",
    "16",
    "-pix_fmt",
    "yuv420p",
    "-movflags",
    "+faststart",
    str(media / f"c296-{w}.mp4"),
]
proc = subprocess.Popen(ff, stdin=subprocess.PIPE)
prev = None
diff = [[], [], []]
strip = Image.new("RGB", (1050, 5 * (230 if w == 1472 else 208)))
refids = set()
try:
    for i, r in enumerate(rows):
        n = r["n"]
        name = f"{n:04}.png"
        p = image(seq / "render" / name).resize(size, Image.Resampling.BILINEAR)
        studio = image(seq / "studio" / name).resize(size, Image.Resampling.BILINEAR)
        fixed = one(p)
        ims = [p, studio, fixed]
        ar = [np.asarray(im, dtype=np.int16) for im in ims]
        if prev is not None:
            for k in range(3):
                diff[k].append(float(np.abs(ar[k] - prev[k]).mean()))
        prev = ar
        plate = triptych(ims, cell, labels)
        proc.stdin.write(plate.tobytes())
        if i in [0, 22, 44, 66, 89]:
            band = triptych(ims, (350, 198 if w == 1024 else 228), labels)
            band = band.resize((1050, 208 if w == 1024 else 230))
            strip.paste(band, (0, [0, 22, 44, 66, 89].index(i) * band.height))
            triptych(ims, size, labels).save(
                media / f"seq-{n:04}-{w}.jpg", quality=94, subsampling=0
            )
            publish(media / f"seq-{n:04}-{w}.jpg")
        refids.add(r["ref"])
finally:
    proc.stdin.close()
    assert proc.wait() == 0
strip.save(media / f"sequence-strip-{w}.jpg", quality=93)
emit(
    "temporal.jsonl",
    size=size,
    model_size=model_size,
    case=case,
    frames=len(rows),
    fps=30,
    start=rows[0]["t"],
    last=rows[-1]["t"],
    refs=sorted(refids),
    mean_adjacent_rgb={
        k: float(np.mean(v)) for k, v in zip(["render", "studio", "fixer"], diff)
    },
    per_transition=diff,
)

publish(media / f"sequence-strip-{w}.jpg")
publish(media / f"c296-{w}.mp4")
