"""World-fixed stripe and verge-transfer probes; visualize native derivative flags without fitting their photo coordinates."""

from scipy.ndimage import gaussian_filter1d
from .proof import *
from . import refine
from .v5 import SLICES


def along(at, lo, hi):
    tr = inputs()[0]
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(tr.p[:, :2], axis=0), axis=1))]
    a, b = np.interp([lo, hi], tr.t, arc)
    t = np.interp(np.arange(a, b, 0.25), arc, tr.t)
    T0 = tr.T_world_ego(t)
    xy = T0[:, :2, 3]
    f = T0[:, :2, 0]
    f = f / np.linalg.norm(f, axis=1)[:, None]
    side = np.stack([f[:, 1], -f[:, 0]], 1)
    p = xy[:, None, :] + np.arange(-1.2, 1.201, 0.12)[None, :, None] * side[:, None, :]
    rgba = at.lookup(p)
    valid = rgba[:, :, 3] >= 250
    rgb = rgba[:, :, :3].astype("f")
    rgb[~valid] = np.nan
    prof = np.nanmedian(rgb, axis=1)
    ok = np.isfinite(prof).all(1)
    i = np.arange(len(prof))
    profile = np.column_stack([np.interp(i, i[ok], prof[ok, c]) for c in range(3)])
    y = profile @ np.array([0.299, 0.587, 0.114])
    hp = np.abs(y - gaussian_filter1d(y, 6))
    dy = np.abs(np.diff(y))
    verges = []
    areas = Counter()
    for k in at.idx:
        # Nearest indexed route station; evaluate only this slice's corridor, not a square world bbox.
        if np.min(np.linalg.norm(xy - (np.array(k) + 0.5) * M, axis=1)) > 20:
            continue
        tex = at.data(k, 0)
        v = at.idx[k]
        g = v["gutter"]
        s = v["pixels"]
        aa = cv.resize(
            tex[g : g + s, g : g + s], (N, N), interpolation=cv.INTER_AREA
        ).astype("f")
        alpha = aa[:, :, 3] / 255
        im = aa[:, :, :3] / np.maximum(alpha[:, :, None], 0.01)
        guide = refine.guide(k)
        vg = guide[:, :, :3][:, :, ::-1]
        vm = guide[:, :, 3] > 0.015
        edge = (alpha > 0.3) & (alpha < 0.85) & vm
        lum = cv.cvtColor(im.astype("f"), cv.COLOR_RGB2GRAY)
        edge &= abs(lum - cv.GaussianBlur(lum, (0, 0), 4)) < 5
        diff = np.mean(abs(im - vg), axis=2)
        verges.extend(diff[edge][::7])
        areas["soft_cells"] += int(((alpha > 0.01) & (alpha < 0.98)).sum())
        areas["opaque_cells"] += int((alpha >= 0.98).sum())
    return dict(
        stations=len(xy),
        measured=int(ok.sum()),
        stripe_median=float(np.median(hp[ok])),
        stripe_p90=float(np.percentile(hp[ok], 90)),
        adjacent_p95=float(np.percentile(dy[ok[:-1] & ok[1:]], 95)),
        verge_samples=len(verges),
        verge_mae=float(np.mean(verges)),
        verge_median=float(np.median(verges)),
        **areas,
    )


def mapplate(lo, hi, before, after):
    from .proof import label, save

    obs = json.loads(Path(before).read_text())
    result = json.loads(Path(after).read_text())
    ks = np.array(
        [list(map(int, k.split("_"))) for k in set(obs["tiles"]) | set(result["tiles"])]
    )
    base = ks.mean(0)
    axis = np.linalg.svd(ks - base, full_matrices=False)[2].T
    coord = (ks - base) @ axis
    w, h = 1120, 250
    scale = min(
        (w - 65) / max(8, np.ptp(coord[:, 0]) + 1.5),
        (h - 55) / max(5, np.ptp(coord[:, 1]) + 1.5),
    )
    off = np.array([w / 2, h / 2]) - scale * (coord.min(0) + coord.max(0)) / 2
    arr = []
    for d, text in [(obs, "Committed atlas"), (result, "V5 atlas")]:
        im = np.full((h, w, 3), 28, "B")
        for k, (hit, total) in d["tiles"].items():
            p = np.array(list(map(int, k.split("_"))))
            pts = (
                np.array([[0, 0], [1, 0], [1, 1], [0, 1]]) + p - base - 0.5
            ) @ axis * scale + off
            rgb = (
                (56, 82, 92)
                if not hit
                else (132, 81, 40)
                if hit / max(1, total) < 0.025
                else (161, 55, 46)
            )
            cv.fillConvexPoly(im, np.rint(pts).astype("i"), rgb)
            cv.polylines(
                im, [np.rint(pts).astype("i")], True, (32, 36, 40), 1, cv.LINE_AA
            )
        for x, y, *rest in d["anchors"]:
            pt = np.rint(
                ((np.array([x, y]) / M - 0.5 - base) @ axis) * scale + off
            ).astype(int)
            cv.circle(im, tuple(pt), 1, (255, 220, 126), -1)
        arr.append(
            label(
                im,
                f"{text} | {lo}-{hi} s | sampled 10.24 m tiles; gold = flagged physical edge",
            )
        )
    save(f"flags-{lo}-{hi}.jpg", np.vstack(arr))


if __name__ == "__main__":
    from collections import Counter
    import sys

    refine.GUIDE = Path("scene/stitched-opt-v5")
    at = Atlas()
    dump(RPT / "surface.json", {f"{lo}-{hi}": along(at, lo, hi) for lo, hi in SLICES})
    if len(sys.argv) > 1:
        for lo, hi in SLICES:
            mapplate(
                lo,
                hi,
                Path(sys.argv[1]) / f"tiles-{lo}-{hi}.json",
                RPT / f"tiles-{lo}-{hi}.json",
            )
