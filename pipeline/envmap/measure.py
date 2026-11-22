"""Frame-to-frame change of two generated clips over the same frames, inside the sky and at crown edges.

Each frame's previous frame is moved onto it through the render's depth and the two camera poses (sky at
infinity: a pure rotation), as in clip.py flicker; what is left is change the camera motion does not
explain. Masks come from the environment run's render (same splats, same poses as the old run): sky =
where the sky canvas shows through fully (skyw >= 0.98) above the horizon, at least 4 px from any splat;
crown edges = the 1-6 px on the sky side and the 1-3 px on the splat side of the splat/sky boundary
above the horizon (crowns, poles, wires, roofs). Mean |difference| in RGB levels, inside the crop.
"""

import argparse
import json
from pathlib import Path

import cv2 as cv
import numpy as np

from pipeline.genfill.clip import RS, J, geometry, rgb, warp


def masks(aux, below):
    sw = aux["skyw"].astype("f") / 255
    up = ~below
    skyb = (sw >= 0.5).astype("B")
    d_sky = cv.distanceTransform(skyb, cv.DIST_L2, 5)
    d_obj = cv.distanceTransform(1 - skyb, cv.DIST_L2, 5)
    sky = (sw >= 0.98) & (d_sky >= 4) & up
    edge = up & (((skyb == 1) & (d_sky <= 6)) | ((skyb == 0) & (d_obj <= 3)))
    return sky, edge


def measure(old_dir, old_tag, new_dir, new_tag, n0, n1, out):
    old_dir, new_dir = RS / old_dir, RS / new_dir
    meta = J(new_dir / "frames.json")
    K = np.array(meta["K"])
    x0, y0, w, h = meta["crop"]
    drive = J(Path(meta["scene"]) / "drive.json")
    res = {"old": dict(sky=[], edge=[]), "new": dict(sky=[], edge=[])}
    prev = {}
    shares = []
    for fr in meta["frames"][n0:n1]:
        n = fr["n"]
        aux = np.load(new_dir / "render" / f"{n:04d}.npz")
        T, z, _, below, _ = geometry(fr, aux["z"].astype("f"), K, drive)
        sky, edge = masks(aux, below)
        crop = np.zeros_like(sky)
        crop[y0:y0 + h, x0:x0 + w] = True
        sky, edge = sky & crop, edge & crop
        shares.append((float(sky[crop].mean()), float(edge[crop].mean())))
        for k, (d, tag) in (("old", (old_dir, old_tag)), ("new", (new_dir, new_tag))):
            im = rgb(d / tag / f"{n:04d}.png")
            if k in prev:
                W, vis = warp(prev[k][0], prev[k][1], prev[k][2], T, z, K)
                e = np.abs(W.astype("f") - im.astype("f")).mean(-1)
                for name, m in (("sky", sky), ("edge", edge)):
                    mm = m & vis
                    if mm.sum() > 200:
                        res[k][name].append(float(e[mm].mean()))
            prev[k] = (im, T, z)
    summary = {k: {m: dict(mean=round(float(np.mean(v)), 3), p95=round(float(np.percentile(v, 95)), 3))
                   for m, v in r.items()} for k, r in res.items()}
    summary["frames"] = [n0, n1]
    summary["mask_share_of_frame"] = dict(sky=round(float(np.mean([s[0] for s in shares])), 4),
                                          edge=round(float(np.mean([s[1] for s in shares])), 4))
    Path(out).write_text(json.dumps(dict(summary=summary, per_frame=res), indent=1))
    print(json.dumps(summary), flush=True)


def stills(old_dir, old_tag, new_dir, new_tag, frames, out_dir):
    """Per frame: the whole frame old | new (half size), and the crown-edge window with the most edge
    pixels, old | new at 2x."""
    old_dir, new_dir, out_dir = RS / old_dir, RS / new_dir, Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = J(new_dir / "frames.json")
    K = np.array(meta["K"])
    x0, y0, w, h = meta["crop"]
    drive = J(Path(meta["scene"]) / "drive.json")
    whole, close = [], []
    lab = lambda im, s: cv.putText(cv.copyMakeBorder(im, 24, 0, 0, 0, cv.BORDER_CONSTANT, value=(23, 26, 29)), s, (8, 17),
                                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (225, 231, 234), 1, cv.LINE_AA)
    for n in frames:
        fr = meta["frames"][n]
        aux = np.load(new_dir / "render" / f"{n:04d}.npz")
        _, _, _, below, _ = geometry(fr, aux["z"].astype("f"), K, drive)
        _, edge = masks(aux, below)
        A = rgb(old_dir / old_tag / f"{n:04d}.png")[y0:y0 + h, x0:x0 + w]
        B = rgb(new_dir / new_tag / f"{n:04d}.png")[y0:y0 + h, x0:x0 + w]
        e = edge[y0:y0 + h, x0:x0 + w].astype("f")
        half = lambda im: cv.resize(im, (w // 2, h // 2), interpolation=cv.INTER_AREA)
        whole.append(np.hstack([lab(half(A), f"published {fr['t']:.1f} s"), lab(half(B), "environment map")]))
        dens = cv.boxFilter(e, -1, (240, 150), normalize=False)
        dens[:75] = dens[-75:] = 0
        dens[:, :120] = dens[:, -120:] = 0
        cy, cx = np.unravel_index(np.argmax(dens), dens.shape)
        cut = lambda im: cv.resize(im[cy - 75:cy + 75, cx - 120:cx + 120], (480, 300), interpolation=cv.INTER_NEAREST)
        close.append(np.hstack([lab(cut(A), f"published {fr['t']:.1f} s, 2x"), lab(cut(B), "environment map, 2x")]))
    cv.imwrite(str(out_dir / "frames-old-new.jpg"), np.vstack(whole)[:, :, ::-1], [cv.IMWRITE_JPEG_QUALITY, 85])
    cv.imwrite(str(out_dir / "crown-edges-old-new.jpg"), np.vstack(close)[:, :, ::-1], [cv.IMWRITE_JPEG_QUALITY, 90])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", default="work/genfill/clip/c296")
    ap.add_argument("--old-tag", default="b")
    ap.add_argument("--new", default="work/genfill/clip/c296-env1")
    ap.add_argument("--new-tag", default="b")
    ap.add_argument("--n0", type=int, default=136)
    ap.add_argument("--n1", type=int, default=585)
    ap.add_argument("--out", default="work/envmap/300-316/flicker.json")
    ap.add_argument("--stills", default=None, help="frames n,n,n: write stills to work/envmap/300-316/stills")
    a = ap.parse_args()
    if a.stills:
        stills(a.old, a.old_tag, a.new, a.new_tag, [int(x) for x in a.stills.split(",")], RS / "work/envmap/300-316/stills")
        raise SystemExit
    measure(a.old, a.old_tag, a.new, a.new_tag, a.n0, a.n1, RS / a.out)
