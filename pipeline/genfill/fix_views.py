"""Repair rendered novel views with Difix and record how much each pixel was changed.

Input: a render set (see views.py): views.json with per-view pose, K and ranked reference
photos, render/<id>.png and render/<id>.npz (alpha, z). Output folder:
  fixed/<id>.png     repaired RGB, same size as the render
  change/<id>.png    uint8, 255 x "changed": how much the fixer altered each region
                     (CIELAB distance after a 1.5 px blur, 4 -> 0, 20 -> 1; 1 where the render
                     has no surface, alpha < 0.5)
  class/<id>.png     0 kept (changed < 0.25), 1 repaired, 2 fixer-only (no rendered surface below
                     the horizon, changed >= 0.75, or road paint that the render does not have
                     within 3 px), 3 sky (no rendered surface, ray above the horizon). Class 2
                     content must be labelled predicted; class 3 is not splat supervision.
                     consistency.py later demotes repaired pixels that disagree with neighbours.
  fixed.json         settings and per-view statistics; rewritten as views complete
Resumable: views whose outputs exist are skipped (--relabel recomputes change/class from existing
fixed images without the model). One model per process.

  envs/genfill/bin/python -m pipeline.genfill.fix_views SET --out OUT [--timestep 199] [--blend 1]
"""

import argparse
import json
import os
import time
from pathlib import Path

import cv2 as cv
import numpy as np

RS = Path(os.environ.get("RS", Path.home() / "dev/reality-splat")).resolve()


def lab(rgb):
    return cv.cvtColor(rgb, cv.COLOR_RGB2LAB).astype("f") * [100 / 255, 1, 1] - [0, 128, 128]


def changed(render, fixed, alpha, lo=4.0, hi=20.0, sigma=1.5):
    a = cv.GaussianBlur(lab(render), (0, 0), sigma)
    b = cv.GaussianBlur(lab(fixed), (0, 0), sigma)
    de = np.linalg.norm(a - b, axis=-1)
    x = np.clip((de - lo) / (hi - lo), 0, 1)
    c = x * x * (3 - 2 * x)
    return np.where(alpha < 0.5, 1.0, c).astype("f"), de


def ground(view, z, drive, max_range=45.0):
    """Rendered pixels on the road surface: within 0.25 m of the car's height, within max_range m."""
    T, K = np.array(view["T_world_cam"]), np.array(view["K"])
    h, w = z.shape
    y, x = np.mgrid[:h, :w]
    ray = np.stack([(x + 0.5 - K[0, 2]) / K[0, 0], (y + 0.5 - K[1, 2]) / K[1, 1], np.ones((h, w))], -1)
    X = (ray * z[..., None]) @ T[:3, :3].T + T[:3, 3]
    car = np.array([np.interp(view["t"], drive["t"], drive[k]) for k in ("x", "y", "z")])
    near = np.linalg.norm(X[..., :2] - car[:2], axis=-1) < max_range
    return (z > 0.3) & near & (np.abs(X[..., 2] - car[2]) < 0.25)


def paint(rgb, mask):
    """Bright thin road paint: white top-hat of CIELAB lightness."""
    L = cv.cvtColor(rgb, cv.COLOR_RGB2LAB)[:, :, 0].astype("f")
    return ((L - cv.morphologyEx(L, cv.MORPH_OPEN, np.ones((15, 15), "B"))) > 22) & mask


def classes(change, alpha, view=None, render=None, fixed=None, z=None, drive=None):
    c = np.where((alpha < 0.5) | (change >= 0.75), 2, np.where(change >= 0.25, 1, 0)).astype("B")
    if view is not None and drive is not None and z is not None:
        g = ground(view, z, drive)
        near = cv.dilate(paint(render, g).astype("B"), np.ones((7, 7), "B")) > 0
        c[paint(fixed, g) & ~near] = 2  # new road paint is never repair
    if view is not None:
        T, K = np.array(view["T_world_cam"]), np.array(view["K"])
        h, w = alpha.shape
        y = np.arange(h) + 0.5
        x = np.arange(w) + 0.5
        dz = (T[2, 0] * (x[None] - K[0, 2]) / K[0, 0] + T[2, 1] * (y[:, None] - K[1, 2]) / K[1, 1] + T[2, 2])
        c[(alpha < 0.5) & (dz > 0)] = 3
    return c


def read_rgb(p):
    im = cv.imread(str(p), cv.IMREAD_COLOR)
    assert im is not None, p
    return np.ascontiguousarray(im[:, :, ::-1])


def reference(set_dir, view, size):
    """First ranked reference photo, from the set folder or rebuilt from the original frames."""
    for r in view.get("refs", [])[:1]:
        p = set_dir / r["path"] if r.get("path") else None
        if p is not None and p.exists():
            im = read_rgb(p)
        else:
            from pipeline.genfill.views import photo

            im, _ = photo(r["c"], r["n"], tuple(size))
        if im.shape[1::-1] != tuple(size):
            im = cv.resize(im, tuple(size), interpolation=cv.INTER_AREA)
        return im, r
    return None, None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("set", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", choices=["ref", "noref"], default="ref")
    ap.add_argument("--timestep", type=int, default=199)
    ap.add_argument("--blend", type=float, default=1.0, help="output = render + blend * (fix - render)")
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--ids", default="", help="comma-separated ids or substrings; empty = all")
    ap.add_argument("--every", type=int, default=1, help="take every n-th view (after --ids)")
    ap.add_argument("--relabel", action="store_true", help="recompute change/class of existing fixes")
    a = ap.parse_args()

    set_dir = (RS / a.set).resolve()
    out = (RS / a.out).resolve()
    meta = json.loads((set_dir / "views.json").read_text())
    views = meta["views"]
    if a.ids:
        keys = a.ids.split(",")
        views = [v for v in views if any(k in v["id"] for k in keys)]
    views = views[:: a.every]
    for d in ("fixed", "change", "class"):
        (out / d).mkdir(parents=True, exist_ok=True)
    man = out / "fixed.json"
    done = json.loads(man.read_text())["views"] if man.exists() else {}
    settings = dict(model="nvidia/difix_ref" if a.model == "ref" else "nvidia/difix", timestep=a.timestep,
                    blend=a.blend, dtype="float32" if a.fp32 else "float16", set=str(set_dir),
                    prompt="remove degradation", change_lab_lo_hi=[4, 20], change_blur_px=1.5)
    drive = json.loads((Path(meta["scene"]) / "drive.json").read_text()) if meta.get("scene") else None
    if a.relabel:
        for v in views:
            vid = v["id"]
            if not (out / "fixed" / f"{vid}.png").exists():
                continue
            render, fix = read_rgb(set_dir / "render" / f"{vid}.png"), read_rgb(out / "fixed" / f"{vid}.png")
            aux = np.load(set_dir / "render" / f"{vid}.npz")
            alpha, z = aux["alpha"].astype("f"), aux["z"].astype("f")
            ch, de = changed(render, fix, alpha)
            cl = classes(ch, alpha, v, render, fix, z, drive)
            cv.imwrite(str(out / "change" / f"{vid}.png"), (ch * 255 + 0.5).astype("B"))
            cv.imwrite(str(out / "class" / f"{vid}.png"), cl)
            if vid in done:
                done[vid].update(kept=float((cl == 0).mean()), repaired=float((cl == 1).mean()),
                                 fixer_only=float((cl == 2).mean()), sky=float((cl == 3).mean()))
        man.write_text(json.dumps(dict(settings=json.loads(man.read_text())["settings"], views=done), indent=1))
        print("relabelled", len(done), flush=True)
        return
    import torch
    from pipeline.genfill.difix import Difix

    model = Difix(a.model == "ref", dtype=torch.float32 if a.fp32 else torch.float16)
    tic = time.monotonic()
    n = 0
    for v in views:
        vid = v["id"]
        if vid in done and all((out / d / f"{vid}.png").exists() for d in ("fixed", "change", "class")):
            continue
        render = read_rgb(set_dir / "render" / f"{vid}.png")
        aux = np.load(set_dir / "render" / f"{vid}.npz")
        alpha, z = aux["alpha"].astype("f"), aux["z"].astype("f")
        h, w = render.shape[:2]
        assert h % 64 == 0 and w % 64 == 0, (vid, w, h)
        ref, r = reference(set_dir, v, (w, h)) if a.model == "ref" else (None, None)
        t0 = time.perf_counter()
        fix = model(render, ref, timestep=a.timestep, seed=0)
        dt = time.perf_counter() - t0
        if a.blend != 1:
            fix = np.clip(render + a.blend * (fix.astype("f") - render) + 0.5, 0, 255).astype("B")
        ch, de = changed(render, fix, alpha)
        cl = classes(ch, alpha, v, render, fix, z, drive)
        cv.imwrite(str(out / "fixed" / f"{vid}.png"), fix[:, :, ::-1])
        cv.imwrite(str(out / "change" / f"{vid}.png"), (ch * 255 + 0.5).astype("B"))
        cv.imwrite(str(out / "class" / f"{vid}.png"), cl)
        cov = alpha >= 0.5
        done[vid] = dict(ref=None if r is None else {k: r[k] for k in ("c", "n", "score")}, seconds=dt,
                         mean_change_covered=float(ch[cov].mean()) if cov.any() else None,
                         median_dE_covered=float(np.median(de[cov])) if cov.any() else None,
                         kept=float((cl == 0).mean()), repaired=float((cl == 1).mean()),
                         fixer_only=float((cl == 2).mean()), sky=float((cl == 3).mean()))
        n += 1
        if n % 20 == 0:
            man.write_text(json.dumps(dict(settings=settings, views=done), indent=1))
            el = time.monotonic() - tic
            print(f"{n} views, {el / n:.2f} s/view, driver {torch.mps.driver_allocated_memory() / 1e9:.1f} GB",
                  flush=True)
    settings["peak_driver_gb"] = torch.mps.driver_allocated_memory() / 1e9 if torch.backends.mps.is_available() else None
    man.write_text(json.dumps(dict(settings=settings, views=done), indent=1))
    print("done", n, "new views", round(time.monotonic() - tic, 1), "s", flush=True)


if __name__ == "__main__":
    main()
