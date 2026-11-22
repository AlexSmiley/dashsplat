"""Silhouette proportion fitting, contour registration and material priors."""

from pathlib import Path
import cv2, numpy as np
from .shape import silhouette
from .inventory import W
from pipeline.common.rig import load_rig, Trajectory


def fit_profile(data):
    from .model import projection

    s, rows, track, size, _, _ = data
    cls = s["class"]
    root = W / "extra" if s.get("input") == "extra" else Path(".")
    tr = Trajectory("poses/ego.npz")
    cs = load_rig()
    rr = [
        r
        for r in rows
        if r["key"]
        and not r.get("held")
        and min(np.diff(np.reshape(r["bb"], (2, 2)), axis=0).ravel()) >= 14
        and min(r["bb"][:2]) > 2
        and r["bb"][2] < 720
        and r["bb"][3] < 466
    ]
    rr = (
        [
            rr[i]
            for i in np.unique(
                np.linspace(0, len(rr) - 1, min(11, len(rr))).astype(int)
            )
        ]
        if rr
        else []
    )
    obs = []
    for r in rr:
        mask = (
            cv2.imread(str(root / "masks/dyn" / r["c"] / f"{r['n']:06}.png"), -1)
            == r["id"]
        )
        obs.append(
            (
                r,
                cv2.resize(
                    mask.astype("uint8"), (362, 235), interpolation=cv2.INTER_NEAREST
                ),
            )
        )
    if not obs:
        return size, 0.0, {}
    scores = []
    for lm in [0.93, 1, 1.07]:
        for hm in [0.93, 1, 1.07]:
            for cabin in [-0.14, 0, 0.14] if cls == "car" else [0.0]:
                sz = size * np.array([lm, 1, hm])
                p = silhouette(cls, sz, cabin)
                vs = []
                for r, m in obs:
                    uv, ok, *_ = projection(cs[r["c"]], tr, r["t"], track, p, p * 0)
                    u = uv[ok] * 0.5
                    if len(u) < 5:
                        continue
                    # Translation only for this shape score; retain aspect and roofline.
                    shift = (np.array(r["bb"][:2]) + r["bb"][2:]) * 0.25 - (
                        u.min(0) + u.max(0)
                    ) * 0.5
                    u = u + shift
                    poly = cv2.convexHull(np.rint(u).astype("int32"))
                    im = np.zeros_like(m)
                    cv2.fillConvexPoly(im, poly, 1)
                    vs.append((m & im).sum() / max(1, (m | im).sum()))
                score = (
                    np.median(vs)
                    - 0.026 * (abs(lm - 1) + abs(hm - 1)) / 0.07
                    - 0.006 * abs(cabin) / 0.14
                    if vs
                    else -1
                )
                scores.append((score, lm, hm, cabin))
    score, lm, hm, cabin = max(scores)
    return (
        size * [lm, 1, hm],
        cabin,
        dict(
            length_scale=lm,
            height_scale=hm,
            cabin_fraction=cabin,
            shape_frames=len(obs),
            fit_score=round(float(score), 3),
        ),
    )


def register(xy, visible, mask):
    # Piecewise vertical alignment from projected and observed contour edges.
    # Ignore columns split by poles; no horizontal stretching across occlusion.
    x = np.clip(xy[:, 0].astype(int), 0, 723)
    y = xy[:, 1]
    v = visible & (y >= 0) & (y < 469)
    lo = np.full(724, 999.0)
    hi = np.full(724, -999.0)
    np.minimum.at(lo, x[v], y[v])
    np.maximum.at(hi, x[v], y[v])
    bot = 468 - np.argmax(mask[::-1], 0)
    top = np.argmax(mask, 0)
    mh = bot - top
    ph = hi - lo
    good = (
        mask.any(0)
        & (ph > 8)
        & (mh > ph * 0.82)
        & (mh < ph * 1.19)
        & (abs(top - lo) < np.maximum(2, ph * 0.13))
        & (abs(bot - hi) < np.maximum(2, ph * 0.13))
    )
    if good.sum() < 5:
        return xy
    ker = np.ones(7)
    den = np.convolve(good.astype(float), ker, "same")
    a = np.convolve(np.where(good, top - lo, 0), ker, "same") / np.maximum(1, den)
    b = np.convolve(np.where(good, bot - hi, 0), ker, "same") / np.maximum(1, den)
    f = np.clip((y - lo[x]) / np.maximum(1, ph[x]), 0, 1)
    out = xy.copy()
    out[:, 1] += (a[x] * (1 - f) + b[x] * f) * np.minimum(1, den[x] / 4)
    return out


def glass_mask(p, n, mat, cls, size, cabin):
    x, y, z = (p / size).T
    wheel = np.isin(mat, ["tyre", "rim", "hub"])
    side = (abs(y) > 0.32) & ~wheel
    glass = (
        (mat == "window")
        & ((x - cabin * 0.3) > -0.36)
        & ((x - cabin * 0.3) < 0.22)
        & (z > 0.57)
    )
    glass |= (abs(n[:, 0]) > 0.24) & (z > 0.68) & (abs(x) < 0.42) & ~wheel
    if cls in ("bus", "truck"):
        glass = side & (z > 0.57) & (z < 0.86)
        glass &= np.mod((x + 0.49) * 10, 1) > 0.06
        glass |= (x > 0.473) & (z > 0.53) & (z < 0.83)
    return glass if cls in ("car", "bus", "truck") else np.zeros(len(p), bool)


def material(p, n, mat, cls, size, paint, roof, cabin):
    x, y, z = (p / size).T
    sg = abs(y)
    base = np.tile(paint, (len(p), 1)).astype(float)
    wheel = np.isin(mat, ["tyre", "rim", "hub"])
    base[mat == "tyre"] = [28, 30, 31]
    base[mat == "hub"] = [73, 77, 82]
    spoke = np.cos(
        np.arctan2((z - 0.205) * size[2], (x - np.where(x < 0, -0.315, 0.31)) * size[0])
        * 7
    )
    base[mat == "rim"] = np.where(
        spoke[mat == "rim", None] > 0.2, [160, 166, 168], [75, 80, 82]
    )
    if cls not in ("car", "bus", "truck"):
        base[mat == "head"] = [114, 92, 79]
        return np.clip(base, 0, 255)
    top = (n[:, 2] > 0.74) & (z > 0.81) & ~wheel
    base[top] = np.clip(0.65 * roof + 0.35 * paint, 0, 255) * (
        1.0 + 0.035 * np.cos(y[top, None] * 11)
    )
    glass = glass_mask(p, n, mat, cls, size, cabin)
    tone = np.clip((z - 0.57) * 1.6, 0, 1)
    reflection = 0.5 + 0.5 * np.cos(x * 9 + y * 4)
    base[glass] = np.stack(
        [
            22 + 26 * tone + 8 * reflection,
            33 + 30 * tone + 8 * reflection,
            40 + 33 * tone + 8 * reflection,
        ],
        1,
    )[glass]
    lamp = (abs(x) > 0.46) & (sg > 0.26) & (z > 0.31) & (z < 0.405) & ~wheel
    base[lamp & (x > 0)] = [204, 211, 205]
    base[lamp & (x < 0)] = [135, 28, 32]
    grille = (x > 0.495) & (sg < 0.25) & (z > 0.22) & (z < 0.34)
    base[grille] = [41, 44, 45]
    return np.clip(base, 0, 255)
