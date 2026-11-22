"""Fill the sky no camera saw (zenith, the joins between cameras, holes behind crowns and wires) by
diffusion inpainting on overlapping perspective tiles of the observed map (build.py).

Tiles: a ring of 8 at 22 deg up (90 deg field, 45 deg apart, so each overlaps its neighbours by half and
the 360 deg wrap is just another overlap), then one looking straight up (100 deg). Tiles run in order,
each conditioned on the real sky and on the fills before it (a later tile replaces an earlier fill only
where the earlier tile covered it poorly, near its edge, and blends where it covered it well). Every tile is inpainted with a per-pixel
strength (differential diffusion: a pixel is held to the noised original until the noise level falls
below its strength): `hole` in holes, which start from the surrounding real colours carried in by a
push-pull fill (from a grey hole both models only paint a flat wash of an invented colour), `edge` on
real pixels touching a hole, falling to 0 over `band` px. Holes of the map are pre-filled the same way
before tiling (prefill), the model's fine detail in holes is capped and its thin strokes removed (guard:
it wrote text and drew dotted wires), and the broad colour comes from the push-pull start (lf_lock).
Below the horizon the tile holds the horizon colour (never shown: the viewer reads the map from
0.9 deg up). Fixed seed.

Model (see bakeoff): Kandinsky 2.2 decoder-inpaint, conditioned on the CLIP image embedding of
patches of the observed sky (no text). Licence Apache-2.0. Weights in third_party/hf.

Writes <out>/sky.jpg (the environment map, 4096x2048), provenance.png (255 observed, 170 real pixel
harmonised at low strength next to a fill, 85 filled, 0 below the horizon), environment.json (the viewer's
descriptor), fill.json, strokes.jpg (thin dark strokes left, the ghost-wire check), tiles/.

bakeoff: held-out test on observed sky (two tiles; the top of the observed band and a 4 deg vertical
join removed): Kandinsky 2.2 (Apache-2.0) vs SDXL inpainting 0.1 (OpenRAIL++), errors vs the truth.
"""

import argparse
import json
import os
import time
from pathlib import Path

import cv2 as cv
import numpy as np

RS = Path(os.environ.get("RS", Path.home() / "dev/reality-splat")).resolve()
os.environ.setdefault("HF_HOME", str(RS / "third_party/hf"))
W, H = 4096, 2048
HORIZON_ROW = H // 2
KAND = "kandinsky-community/kandinsky-2-2-decoder-inpaint"
KAND_PRIOR = "kandinsky-community/kandinsky-2-2-prior"
SDXL = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
PROMPT = "overcast dawn sky, soft grey and blue-grey clouds, photograph"
NEGATIVE = "trees, poles, wires, power lines, buildings, birds, aircraft, sun, text, frame, border, vignette"
cv.setNumThreads(2)


def basis(yaw, pitch):
    """right, down, forward (world) of a camera looking at azimuth yaw, elevation pitch (degrees)."""
    a, p = np.radians(yaw), np.radians(pitch)
    f = np.array([np.cos(p) * np.cos(a), np.cos(p) * np.sin(a), np.sin(p)])
    r = np.array([np.sin(a), -np.cos(a), 0.0])
    return r, np.cross(f, r), f


def tile_dirs(yaw, pitch, fov, size):
    r, d, f = basis(yaw, pitch)
    foc = size / 2 / np.tan(np.radians(fov) / 2)
    y, x = np.mgrid[:size, :size] + 0.5
    v = ((x - size / 2) / foc)[..., None] * r + ((y - size / 2) / foc)[..., None] * d + f
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def equi_xy(d):
    u = (0.5 - np.arctan2(d[..., 1], d[..., 0]) / (2 * np.pi)) % 1.0
    v = 0.5 + np.arcsin(np.clip(d[..., 2], -1, 1)) / np.pi
    return (u * W - 0.5).astype("f"), np.clip((1 - v) * H - 0.5, 0, H - 1).astype("f")


def to_tile(img, yaw, pitch, fov, size, interp=cv.INTER_LINEAR):
    x, y = equi_xy(tile_dirs(yaw, pitch, fov, size))
    return cv.remap(img, x, y, interp, borderMode=cv.BORDER_WRAP)


def map_dirs(rows):
    u = (np.arange(W) + 0.5) / W
    v = 1 - (np.asarray(rows) + 0.5) / H
    az = (0.5 - u)[None, :] * 2 * np.pi
    el = (v - 0.5)[:, None] * np.pi
    return np.stack(np.broadcast_arrays(np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)), -1)


def from_tile(yaw, pitch, fov, size, rows):
    """Tile pixel coordinates of map pixels in `rows`, and a weight: 1 at the centre, falling smoothly to 0
    at 95 % of the half-width (the tile's own borders are never written back; overlaps feather)."""
    r, d, f = basis(yaw, pitch)
    D = map_dirs(rows)
    z = D @ f
    foc = size / 2 / np.tan(np.radians(fov) / 2)
    zz = np.where(z > 1e-3, z, 1e-3)
    x = foc * (D @ r) / zz + size / 2 - 0.5
    y = foc * (D @ d) / zz + size / 2 - 0.5
    e = np.maximum(np.abs(x + 0.5 - size / 2), np.abs(y + 0.5 - size / 2)) / (size / 2)
    w = np.clip((0.95 - e) / 0.95, 0, 1) ** 2 * (z > 1e-3)
    return x.astype("f"), y.astype("f"), w.astype("f")


def strength(known, band, edge, hole):
    """`hole` in holes; on known pixels `edge` where they touch a hole, falling to 0 at `band` px."""
    dist = cv.distanceTransform(known.astype("B"), cv.DIST_L2, 5)
    s = np.where(known, edge * np.clip(1 - dist / band, 0, 1), hole)
    return s.astype("f")


def pushpull(img, valid):
    """The real sky's colours carried smoothly into the holes (mask-weighted Gaussian pyramid: coarse levels
    fill where fine ones have no support). The diffusion starts from this, so it keeps the surrounding
    colours and adds cloud structure, instead of inventing a colour from a grey hole."""
    I = img.astype("f") * valid[..., None]
    Wt = valid.astype("f")
    pyr = [(I, Wt)]
    while min(I.shape[:2]) > 3:
        I, Wt = cv.pyrDown(I), cv.pyrDown(Wt)
        pyr.append((I, Wt))
    I, Wt = pyr[-1]
    mean = img[valid].mean(0) if valid.any() else np.full(3, 128.0)
    out = np.where((Wt > 1e-3)[..., None], I / np.maximum(Wt, 1e-6)[..., None], mean)
    for I, Wt in reversed(pyr[:-1]):
        up = cv.resize(out, (I.shape[1], I.shape[0]), interpolation=cv.INTER_LINEAR)
        a = np.clip(Wt * 3, 0, 1)[..., None]
        out = a * (I / np.maximum(Wt, 1e-6)[..., None]) + (1 - a) * up
    return np.where(valid[..., None], img, out)


class Kandinsky:
    name = "kandinsky-2.2-decoder-inpaint"
    licence = "Apache-2.0"
    size = 768

    def __init__(self):
        import torch
        from diffusers import KandinskyV22InpaintPipeline
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

        self.torch = torch
        self.dev = "mps"
        self.pipe = KandinskyV22InpaintPipeline.from_pretrained(KAND, torch_dtype=torch.float16).to(self.dev)
        self.pipe.set_progress_bar_config(disable=True)
        self.enc = CLIPVisionModelWithProjection.from_pretrained(
            KAND_PRIOR, subfolder="image_encoder", torch_dtype=torch.float16).to(self.dev)
        self.proc = CLIPImageProcessor.from_pretrained(KAND_PRIOR, subfolder="image_processor")
        n = self.enc.config.image_size
        with torch.no_grad():
            self.zero = self.enc(torch.zeros(1, 3, n, n, dtype=torch.float16, device=self.dev))["image_embeds"]
        self.emb = None

    def condition(self, patches):
        """CLIP image embedding of real sky patches (uint8 RGB), averaged: the model's only prompt."""
        t = self.torch
        px = self.proc(images=list(patches), return_tensors="pt").pixel_values.to(self.dev, t.float16)
        with t.no_grad():
            e = self.enc(px)["image_embeds"].float()
        m = e.mean(0, keepdim=True)
        self.emb = (m / m.norm() * e.norm(dim=-1).mean()).half()

    def __call__(self, img, s, seed=0, steps=30, guidance=4.0):
        """img uint8 RGB size x size; s per-pixel strength. Returns float RGB."""
        t = self.torch
        from PIL import Image

        size = img.shape[0]
        lat = cv.resize(s, (size // 8, size // 8), interpolation=cv.INTER_AREA)
        s_lat = t.from_numpy(lat)[None, None].to(self.dev, t.float16)
        T = float(self.pipe.scheduler.config.num_train_timesteps)
        st = {}

        def cb(pipe, i, ts, kw):
            mi = kw["mask_image"]
            steps_ = pipe.scheduler.timesteps
            level = float(steps_[i + 2]) / T if i + 2 < len(steps_) else 0.0
            keep = (s_lat < level).to(mi.dtype) if i + 2 < len(steps_) else (s_lat <= 0).to(mi.dtype)
            keep = keep.repeat(mi.shape[0], 1, 1, 1)
            return dict(mask_image=keep, masked_image=st["img"].repeat(mi.shape[0], 1, 1, 1) * keep)

        with t.no_grad():
            x = t.from_numpy(img.astype("f") / 127.5 - 1).permute(2, 0, 1)[None].to(self.dev, t.float16)
            st["img"] = self.pipe.movq.encode(x)["latents"]
        g = t.Generator("cpu").manual_seed(seed)
        out = self.pipe(image=Image.fromarray(img), mask_image=(s >= 0.999).astype("f"), image_embeds=self.emb,
                        negative_image_embeds=self.zero, height=size, width=size, num_inference_steps=steps,
                        guidance_scale=guidance, generator=g, output_type="np", callback_on_step_end=cb,
                        callback_on_step_end_tensor_inputs=["mask_image", "masked_image"]).images[0]
        return out * 255.0


class SDXL_:
    name = "stable-diffusion-xl-1.0-inpainting-0.1"
    licence = "CreativeML OpenRAIL++-M"
    size = 1024

    def __init__(self):
        import torch
        from diffusers import StableDiffusionXLInpaintPipeline

        self.torch = torch
        self.pipe = StableDiffusionXLInpaintPipeline.from_pretrained(SDXL, torch_dtype=torch.float16,
                                                                     variant="fp16").to("mps")
        self.pipe.set_progress_bar_config(disable=True)

    def condition(self, patches):
        pass

    def __call__(self, img, s, seed=0, steps=30, guidance=7.0):
        from PIL import Image

        # no per-pixel strength here: holes at their strength from the same push-pull start, binary mask
        g = self.torch.Generator("cpu").manual_seed(seed)
        hole = s >= 0.5
        m = Image.fromarray((hole * 255).astype("B"))
        out = self.pipe(prompt=PROMPT, negative_prompt=NEGATIVE, image=Image.fromarray(img), mask_image=m,
                        height=img.shape[0], width=img.shape[1], strength=float(s[hole].max()), num_inference_steps=steps,
                        guidance_scale=guidance, generator=g, output_type="np").images[0]
        return out * 255.0


def sky_patches(obs, known, n=6, fov=14.0, size=224):
    """Square patches of the real sky, at least 97 % observed (the rest set to the patch's mean), for the
    image-embedding condition."""
    out = []
    for yaw in np.arange(-180, 180, 10.0):
        for pitch in (10.0, 14.0, 17.0):
            k = to_tile(known.astype("f"), yaw, pitch, fov, size, cv.INTER_NEAREST) > 0.5
            if k.mean() > 0.97:
                im = to_tile(obs, yaw, pitch, fov, size)
                out.append((float(k.mean()), np.where(k[..., None], im, im[k].mean(0))))
    rng = np.random.default_rng(0)
    pick = rng.choice(len(out), min(n, len(out)), replace=False) if out else []
    return [np.clip(out[i][1], 0, 255).astype("B") for i in pick]


def horizon_fill(obs, known):
    """Per-column horizon colour (median of observed sky 1-6 deg up, smoothed around the circle),
    painted below the horizon and returned."""
    r0, r1 = int((0.5 - 6 / 180) * H), int((0.5 - 1 / 180) * H)
    band, kb = obs[r0:r1].astype("f"), known[r0:r1]
    col = np.full((W, 3), np.nan, "f")
    for x in range(0, W, 16):
        m = kb[:, x:x + 16]
        if m.sum() > 20:
            col[x:x + 16] = np.median(band[:, x:x + 16][m], 0)
    good = ~np.isnan(col[:, 0])
    xs = np.arange(W)
    for c in range(3):
        col[:, c] = np.interp(xs, xs[good], col[good, c], period=W)
    col = cv.GaussianBlur(np.tile(col[None], (3, 1, 1)), (0, 0), 60)[1]
    return col


def load_observed(src):
    obs = cv.imread(str(src / "observed.png"))[:, :, ::-1].astype("f")
    known = cv.imread(str(src / "observed-mask.png"), 0) > 0
    return obs, known


def held_out(known, yaw, pitch, fov, size):
    """Bake-off holes inside observed sky: everything above 13 deg, and a 4 deg wide vertical join."""
    D = tile_dirs(yaw, pitch, fov, size)
    el = np.degrees(np.arcsin(D[..., 2]))
    az = np.degrees(np.arctan2(D[..., 1], D[..., 0]))
    join = (np.abs(((az - yaw + 180) % 360) - 180) < 2.0) & (el > 0)
    return (el > 13.0) | join


def label(im, s):
    out = cv.copyMakeBorder(im, 26, 0, 0, 0, cv.BORDER_CONSTANT, value=(23, 26, 29))
    cv.putText(out, s, (8, 18), cv.FONT_HERSHEY_SIMPLEX, 0.52, (225, 231, 234), 1, cv.LINE_AA)
    return out


def prefill(obs, known):
    """Holes of the equirectangular map pre-filled by push-pull (wrapped across 360 deg), so that sampling a
    tile never mixes the black of a hole into the real pixels beside it: such dark rims came back from the
    model as dotted wires and specks."""
    pad = 256
    wr = lambda a: np.concatenate([a[:, -pad:], a, a[:, :pad]], 1)
    return pushpull(wr(obs), wr(known))[:, pad:-pad]


def guard(G, s, amp=3.0):
    """A sky fill has no hard detail: where the model generated (strength >= 0.5), fine detail is limited to
    +-amp levels and thin dark or bright strokes (text, wire-like lines, specks, all of which it drew) are
    replaced by the median around them."""
    G = np.clip(G, 0, 255).astype("f")
    lo = cv.GaussianBlur(G, (0, 0), 1.5)
    out = lo + np.clip(G - lo, -amp, amp)
    L = cv.cvtColor(G.astype("B"), cv.COLOR_RGB2GRAY).astype("f")
    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 11))
    thin = (cv.morphologyEx(L, cv.MORPH_BLACKHAT, k) > 4) | (cv.morphologyEx(L, cv.MORPH_TOPHAT, k) > 4)
    thin = cv.dilate(thin.astype("B"), np.ones((5, 5), "B")) > 0
    med = cv.GaussianBlur(cv.medianBlur(G.astype("B"), 15).astype("f"), (0, 0), 2)
    out = np.where(thin[..., None], med, out)
    return np.where((s >= 0.5)[..., None], out, G)


def lf_lock(G, base, s, sigma, amp=10.0):
    """Keep the model's cloud structure but take the broad colour (scales above sigma px) from the push-pull
    start, which only knows the real sky: both models drift in colour over a large hole. In holes the
    structure is held within +-amp levels (it once drew a dark blob)."""
    lf = lambda a: cv.GaussianBlur(a.astype("f"), (0, 0), sigma)
    dev = G - lf(G)
    dev = np.where((s >= 0.5)[..., None], np.clip(dev, -amp, amp), dev)
    return np.where((s > 0)[..., None], dev + lf(base), base)


class PushPull:
    """No model: the push-pull start alone (the floor the diffusion has to beat)."""
    name = "push-pull only"
    licence = "-"
    size = 768

    def condition(self, patches):
        pass

    def __call__(self, img, s, seed=0):
        return img.astype("f")


def bakeoff(src, out, tiles=((-90.0, 10.0), (-135.0, 10.0)), fov=60.0, hole_s=0.95, lock=40.0):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    obs, known = load_observed(src)
    patches = sky_patches(obs, known)
    pre = prefill(obs, known)
    res, shown = {}, []
    for cls in (PushPull, Kandinsky, SDXL_):
        tic = time.monotonic()
        model = cls()
        load = time.monotonic() - tic
        model.condition(patches)
        res[cls.name] = dict(licence=cls.licence, load_s=round(load, 1), hole_strength=hole_s, lock_px=lock, tiles=[])
        for k, (yaw, pitch) in enumerate(tiles):
            size = cls.size
            img = to_tile(pre, yaw, pitch, fov, size)
            kn = to_tile(known.astype("f"), yaw, pitch, fov, size, cv.INTER_NEAREST) > 0.5
            hold = held_out(kn, yaw, pitch, fov, size) & kn
            ctx = kn & ~hold
            s = np.where(ctx, 0.0, hole_s).astype("f")
            inp = pushpull(img, ctx)
            t0 = time.monotonic()
            G = model(np.clip(inp, 0, 255).astype("B"), s, seed=0)
            dt = time.monotonic() - t0
            G = lf_lock(guard(G, s), inp, s, lock * size / 768)
            # judge at 768
            r = lambda a, i=cv.INTER_AREA: cv.resize(a, (768, 768), interpolation=i)
            G7, T7, Hm = r(G.astype("f")), r(img.astype("f")), r(hold.astype("B"), cv.INTER_NEAREST) > 0
            lf = lambda a: cv.GaussianBlur(a, (0, 0), 6)
            C7 = r(ctx.astype("B"), cv.INTER_NEAREST) > 0
            # the seam between the fill and the real sky it adjoins (not its edge against never-seen sky)
            ring = (cv.dilate(Hm.astype("B"), np.ones((5, 5), "B")) > 0) & ~(cv.erode(Hm.astype("B"), np.ones((5, 5), "B")) > 0)
            ring &= (Hm & (cv.dilate(C7.astype("B"), np.ones((5, 5), "B")) > 0)) | C7
            gm = lambda a: np.hypot(cv.Sobel(a.mean(-1), cv.CV_32F, 1, 0), cv.Sobel(a.mean(-1), cv.CV_32F, 0, 1))
            Gc = np.where(Hm[..., None], G7, T7)
            m = dict(yaw=yaw, pitch=pitch, seconds=round(dt, 1),
                     lowfreq_abs=round(float(np.abs(lf(G7) - lf(T7))[Hm].mean()), 2),
                     texture_ratio=round(float((G7 - lf(G7))[Hm].std() / max((T7 - lf(T7))[Hm].std(), 1e-3)), 3),
                     structure_ratio=round(float((lf(G7) - cv.GaussianBlur(G7, (0, 0), 40))[Hm].std()
                                                 / max((lf(T7) - cv.GaussianBlur(T7, (0, 0), 40))[Hm].std(), 1e-3)), 3),
                     seam_ratio=round(float(gm(Gc)[ring].mean() / max(gm(T7)[ring].mean(), 1e-3)), 3))
            res[cls.name]["tiles"].append(m)
            print(cls.name, m, flush=True)
            if k == 0:
                truth = np.clip(T7, 0, 255).astype("B")
                holed = np.where(C7[..., None], truth, 0).astype("B")
                shown.append(label(np.clip(np.where(C7[..., None], T7, G7), 0, 255).astype("B"), f"{cls.name} {dt:.0f} s"))
        del model
        import gc
        import torch

        gc.collect()
        torch.mps.empty_cache()
    sheet = np.hstack([label(truth, "truth (black: never seen)"), label(holed, "given to the model")] + shown)
    cv.imwrite(str(out / "bakeoff.jpg"), cv.resize(sheet, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)[:, :, ::-1],
               [cv.IMWRITE_JPEG_QUALITY, 88])
    (out / "bakeoff.json").write_text(json.dumps(res, indent=1))


RING = [(yaw, 22.0, 90.0) for yaw in range(0, 360, 45)]
ZENITH = (0.0, 90.0, 100.0)


def fill(src, out, band=12.0, edge=0.3, hole=0.95, lock=40.0, seed=0, steps=30):
    out = Path(out)
    (out / "tiles").mkdir(parents=True, exist_ok=True)
    obs, known = load_observed(src)
    nrows = HORIZON_ROW + 4
    hz = horizon_fill(obs, known)
    M = prefill(obs, known)
    M[HORIZON_ROW:] = hz[None]
    K = known.copy()
    K[HORIZON_ROW:] = True
    prov = np.where(known, 255, 0).astype("B")
    tic = time.monotonic()
    model = Kandinsky()
    model.condition(sky_patches(obs, known))
    load = time.monotonic() - tic
    # start where the most sky is real and go round, so the last tile closes the circle between two fills
    score = [to_tile(K.astype("f"), y, p, f, 128, cv.INTER_NEAREST).mean() for y, p, f in RING]
    k0 = int(np.argmax(score))
    order = RING[k0:] + RING[:k0] + [ZENITH]
    log = []
    rows = np.arange(nrows)
    # holes: each tile's fill replaces what earlier tiles left there, inside its central area (a mean of
    # several fills is a featureless haze that makes the real sky stand out as hard-edged patches); a hole
    # becomes context for later tiles once a tile has covered it well. Real pixels keep their value plus the
    # weighted mean of the tiles' low-strength changes, which taper to nothing (no resampling).
    R = obs[:nrows].astype("f")
    hole0 = ~known[:nrows]
    accD = np.zeros((nrows, W, 3), "f")
    accH = np.zeros((nrows, W), "f")
    wacc = np.zeros((nrows, W), "f")
    for j, (yaw, pitch, fov) in enumerate(order):
        size = model.size
        img = to_tile(M, yaw, pitch, fov, size)
        kn = to_tile(K.astype("f"), yaw, pitch, fov, size, cv.INTER_NEAREST) > 0.5
        s = strength(kn, band, edge, hole)
        if (~kn).mean() < 1e-4:
            continue
        img = pushpull(img, kn)
        t0 = time.monotonic()
        G = model(np.clip(img, 0, 255).astype("B"), s, seed=seed + j, steps=steps)
        dt = time.monotonic() - t0
        G = lf_lock(guard(G, s), img, s, lock)
        # the change tapers to nothing across the band, so a harmonised patch has no edge
        ws = np.where(kn, np.clip(s / edge, 0, 1), 1.0).astype("f")
        G = (img + ws[..., None] * (G - img)).astype("f")
        cv.imwrite(str(out / "tiles" / f"{j}-in.jpg"), np.clip(np.where(kn[..., None], img, 0), 0, 255).astype("B")[:, :, ::-1])
        cv.imwrite(str(out / "tiles" / f"{j}-out.jpg"), np.clip(G, 0, 255).astype("B")[:, :, ::-1])
        x, y, w = from_tile(yaw, pitch, fov, size, rows)
        w = w * (rows < HORIZON_ROW)[:, None]
        rm = lambda a, bv=0: cv.remap(a, x, y, cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT, borderValue=bv)
        Gm = rm(G)
        accD += w[..., None] * rm((G - img).astype("f"))
        accH += w * rm(ws)
        wacc += w
        Mn = M[:nrows]
        # an earlier tile's well-covered fill (already context here, strength low) is blended, not replaced
        take = np.where(hole0 & ~K[:nrows], (w > 0).astype("f"), w)
        Mn[:] = np.where(hole0[..., None], Mn + (take * hole0)[..., None] * (Gm - Mn),
                         R + np.where((wacc > 1e-4)[..., None], accD / np.maximum(wacc, 1e-4)[..., None], 0))
        K[:nrows] |= hole0 & (w > 0.5)
        log.append(dict(tile=j, yaw=yaw, pitch=pitch, fov=fov, seconds=round(dt, 1),
                        hole_share=round(float((~kn).mean()), 3)))
        print(log[-1], flush=True)
    P = prov[:nrows]
    P[hole0 & (wacc > 1e-4)] = 85
    P[~hole0 & (accH / np.maximum(wacc, 1e-4) > 0.05)] = 170
    left = ~K[:HORIZON_ROW] & (prov[:HORIZON_ROW] != 85)
    if left.any():  # slivers no tile reached: nearest filled colour
        sub = np.clip(M[:HORIZON_ROW], 0, 255).astype("B")
        M[:HORIZON_ROW] = cv.inpaint(sub, left.astype("B"), 5, cv.INPAINT_TELEA).astype("f")
        prov[:HORIZON_ROW][left] = 85
    prov[HORIZON_ROW:] = 0
    env = np.clip(M + 0.5, 0, 255).astype("B")
    cv.imwrite(str(out / "sky.jpg"), env[:, :, ::-1], [cv.IMWRITE_JPEG_QUALITY, 92])
    cv.imwrite(str(out / "provenance.png"), prov)
    el = np.degrees((0.5 - (np.arange(HORIZON_ROW) + 0.5) / H) * np.pi)
    sa = np.cos(np.radians(el))[:, None] * np.ones((1, W))
    share = {k: round(float((sa * (prov[:HORIZON_ROW] == v)).sum() / sa.sum()), 4)
             for k, v in (("observed", 255), ("harmonised", 170), ("filled", 85))}
    rows_c = (el > 0) & (el < 13.4)  # what the default chase view can see of the sky
    share_chase = {k: round(float((sa[rows_c] * (prov[:HORIZON_ROW][rows_c] == v)).sum() / sa[rows_c].sum()), 4)
                   for k, v in (("observed", 255), ("harmonised", 170), ("filled", 85))}
    # ghost poles, wires, specks: thin dark strokes (black-hat over 6 levels, 11 px) per million pixels
    L = cv.cvtColor(env[:HORIZON_ROW], cv.COLOR_RGB2GRAY).astype("f")
    bh = cv.morphologyEx(L, cv.MORPH_BLACKHAT, cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 11))) > 6
    pv = prov[:HORIZON_ROW]
    strokes = {k: round(float(bh[pv == v].mean() * 1e6), 1) for k, v in (("observed", 255), ("harmonised", 170), ("filled", 85))}
    vis = np.clip(env[:HORIZON_ROW].astype("f") * 0.6, 0, 255).astype("B")
    vis[bh] = (255, 40, 40)
    cv.imwrite(str(out / "strokes.jpg"), cv.resize(vis, (2048, 512), interpolation=cv.INTER_AREA)[:, :, ::-1])
    doc = dict(model=model.name, licence=model.licence, seed=seed, steps=steps, guidance=4.0, band_px=band,
               thin_dark_strokes_per_million_px=strokes,
               edge_strength=edge, hole_strength=hole, colour_lock_px=lock, tile_size=model.size, load_s=round(load, 1), tiles=log,
               seconds=round(time.monotonic() - tic, 1), share_upper_hemisphere=share, share_0_13deg=share_chase)
    (out / "fill.json").write_text(json.dumps(doc, indent=1))
    # the viewer's descriptor: the haze colour is the real sky just above the horizon (1-6 deg, observed only)
    r0, r1 = int((0.5 - 6 / 180) * H), int((0.5 - 1 / 180) * H)
    horizon = np.median(obs[r0:r1][known[r0:r1]], 0)
    b = json.loads((Path(src) / "build.json").read_text())
    (out / "environment.json").write_text(json.dumps(dict(
        format="environment-map", version=1, map="sky.jpg", provenance="provenance.png",
        projection="equirectangular; u = 0.5 - atan2(y, x) / 2pi, v = 0.5 + asin(z) / pi, row (1 - v) H; east at the "
                   "centre column, up +z; sky at infinity; below the horizon the horizon colour",
        horizon=[int(round(float(c))) for c in horizon], band=0.06, stretch=[b["t0"], b["t1"]], window=b["window"],
        photometry=b["photometry"], provenance_codes={"255": "observed", "170": "observed, harmonised at low strength",
                                                     "85": "filled", "0": "below the horizon"},
        share_upper_hemisphere=share, share_0_13deg=share_chase, fill_model=model.name, fill_licence=model.licence,
        far_band=None), indent=1))
    print(json.dumps(dict(share=share, chase=share_chase, seconds=doc["seconds"], horizon=horizon.round(1).tolist(),
                          strokes=strokes)), flush=True)


def locked(stage, fn, *a, **k):
    """Run under the Studio's reconstruction lock (pipeline/run): wait for the whole-drive run's heavy
    stage to finish, and hold it off while this runs."""
    from pipeline.run.transactions import identity, lock

    with lock(RS / "work/locks/reconstruction.lock", True, identity(), drive="envmap-300-316", stage=stage):
        print("lock acquired", time.ctime(), flush=True)
        return fn(*a, **k)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["bakeoff", "fill"])
    ap.add_argument("--src", default="work/envmap/300-316")
    ap.add_argument("--out", default="work/envmap/300-316")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nolock", action="store_true", help="the caller already holds the reconstruction lock")
    a = ap.parse_args()
    run = (lambda stage, fn, *x, **k: fn(*x, **k)) if a.nolock else locked
    if a.cmd == "bakeoff":
        run("inpaint-bakeoff", bakeoff, RS / a.src, RS / a.out / "bakeoff")
    else:
        run("inpaint-fill", fill, RS / a.src, RS / a.out, seed=a.seed)
