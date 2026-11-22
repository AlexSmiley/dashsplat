"""Bird's-eye road patches, overlapped tiling with feathered blending, paint-displacement metrics.

Only numpy and OpenCV, so it runs in both envs/genfill (Difix) and envs/genfill-edit (Real-ESRGAN).
"""

import json
import os
from pathlib import Path

import cv2 as cv
import numpy as np

RS = Path(os.environ.get("RS", Path.home() / "dev/reality-splat")).resolve()
WORK = RS / "work/genfill/ground"
SIZE = 1536  # 30.72 m at 2 cm per pixel

# mosaic folder, top-left column/row of the SIZE x SIZE crop, what is in it
PATCHES = {
    "arrows": (RS / "road/mosaics/offline/park", 1150, 150, "turn arrows and stop bar, 93-97 s"),
    "bike": (WORK / "mosaic/bike", 2750, 100, "bike-lane symbol and arrows, 112-115 s (new offline window 106-118 s)"),
    "only": (RS / "road/mosaics/offline/turn", 1050, 950, "ONLY lettering and arrow, 318-320 s"),
}


def load_patch(name):
    """RGB uint8 crop of best.jpg and the bool road mask of the same crop."""
    d, c, r, _ = PATCHES[name]
    img = cv.imread(str(d / "best.jpg"))[r : r + SIZE, c : c + SIZE, ::-1]
    mask = cv.imread(str(d / "road-mask.png"), cv.IMREAD_GRAYSCALE)[r : r + SIZE, c : c + SIZE] > 0
    return np.ascontiguousarray(img), mask


def starts(n, tile, overlap):
    if n <= tile:
        return [0]
    step = tile - overlap
    s = list(range(0, n - tile, step)) + [n - tile]
    return sorted(set(s))


def feather(tile, overlap):
    r = np.minimum(np.arange(tile) + 0.5, tile - np.arange(tile) - 0.5) / overlap
    w = np.clip(r, 1e-3, 1.0)
    return np.outer(w, w)


def tiled(img, fn, tile=768, overlap=128, scale=1):
    """Run fn(tile_rgb_uint8, (row, col)) -> uint8 of size tile*scale over overlapped tiles, feather-blend."""
    h, w = img.shape[:2]
    acc = np.zeros((h * scale, w * scale, 3), np.float64)
    wsum = np.zeros((h * scale, w * scale, 1), np.float64)
    wt = feather(tile * scale, overlap * scale)[..., None]
    for y in starts(h, tile, overlap):
        for x in starts(w, tile, overlap):
            out = fn(np.ascontiguousarray(img[y : y + tile, x : x + tile]), (y, x)).astype(np.float64)
            ys, xs = y * scale, x * scale
            acc[ys : ys + tile * scale, xs : xs + tile * scale] += out * wt
            wsum[ys : ys + tile * scale, xs : xs + tile * scale] += wt
    return np.clip(acc / wsum + 0.5, 0, 255).astype(np.uint8)


def lightness(rgb):
    return cv.cvtColor(rgb, cv.COLOR_RGB2LAB)[..., 0].astype(np.float32) * (100 / 255)


def paint(rgb, mask, thresh=8.0):
    """Bright road paint: white top-hat on CIELAB L* (31 px = 62 cm ellipse), > thresh L*, inside the road mask."""
    L = lightness(rgb)
    th = cv.morphologyEx(L, cv.MORPH_TOPHAT, cv.getStructuringElement(cv.MORPH_ELLIPSE, (31, 31)))
    p = ((th > thresh) & mask).astype(np.uint8)
    p = cv.morphologyEx(p, cv.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return p > 0, th


def within(a, b, r):
    """Fraction of a's pixels with a b pixel within r (Chebyshev disk)."""
    if a.sum() == 0:
        return float("nan")
    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return float((a & (cv.dilate(b.astype(np.uint8), k) > 0)).sum() / a.sum())


def shifts(th0, th1, p0, block=256):
    """Phase correlation of top-hat (paint) signal per block with >= 1 % paint; returns list of (dy, dx)."""
    out = []
    win = cv.createHanningWindow((block, block), cv.CV_32F)
    for y in range(0, th0.shape[0] - block + 1, block):
        for x in range(0, th0.shape[1] - block + 1, block):
            if p0[y : y + block, x : x + block].mean() < 0.01:
                continue
            (dx, dy), _ = cv.phaseCorrelate(
                th0[y : y + block, x : x + block].astype(np.float32), th1[y : y + block, x : x + block].astype(np.float32), win
            )
            out.append((float(dy), float(dx)))
    return out


def grad_energy(rgb, mask):
    L = lightness(rgb)
    g = np.hypot(cv.Sobel(L, cv.CV_32F, 1, 0), cv.Sobel(L, cv.CV_32F, 0, 1))
    return float(g[mask].mean())


def metrics(before, after, mask):
    p0, th0 = paint(before, mask)
    p1, th1 = paint(after, mask)
    sh = shifts(th0, th1, p0)
    mag = [np.hypot(*s) for s in sh]
    d = np.abs(after.astype(np.int16) - before.astype(np.int16)).max(-1)
    dm = d[mask]
    dL = np.abs(lightness(after) - lightness(before))
    return dict(
        paint_px_before=int(p0.sum()),
        paint_px_after=int(p1.sum()),
        recall_1px=within(p0, p1, 1),
        precision_1px=within(p1, p0, 1),
        recall_3px=within(p0, p1, 3),
        precision_3px=within(p1, p0, 3),
        shift_blocks=len(sh),
        shift_mean_px=float(np.mean(mag)) if mag else float("nan"),
        shift_max_px=float(np.max(mag)) if mag else float("nan"),
        shift_mean_dydx=[float(np.mean([s[0] for s in sh])), float(np.mean([s[1] for s in sh]))] if sh else None,
        road_mean_abs_rgb=float(dm.mean()),
        road_p99_abs_rgb=float(np.percentile(dm, 99)),
        road_max_abs_rgb=int(dm.max()),
        road_frac_gt10=float((dm > 10).mean()),
        road_frac_gt25=float((dm > 25).mean()),
        road_mean_abs_L=float(dL[mask].mean()),
        road_mean_signed_dLab=[
            float(v)
            for v in (cv.cvtColor(after, cv.COLOR_RGB2LAB).astype(np.float32) - cv.cvtColor(before, cv.COLOR_RGB2LAB).astype(np.float32))[mask].mean(0)
            * np.array([100 / 255, 1, 1])
        ],
        paint_mean_abs_L=float(dL[p0].mean()) if p0.any() else float("nan"),
        grad_ratio=grad_energy(after, mask) / grad_energy(before, mask),
        all_mean_abs_rgb=float(d.mean()),
        all_max_abs_rgb=int(d.max()),
    )


def save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1))
