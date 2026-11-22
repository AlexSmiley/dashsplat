"""Fit source RGB gains to one robust world-direction dawn radiance field; retain one cloud epoch's detail."""

from .horizon_inputs import *
from concurrent.futures import ThreadPoolExecutor
from scipy.ndimage import gaussian_filter, distance_transform_edt

SKYEL = [-12.0, 82.0]
BACKEL = [-7.0, 27.0]
BW, BH = 4096, 448
SW, SH = 1536, 400
OUT = W / "horizon-r4"
OUT.mkdir(exist_ok=True)


def sphere(w, h, els, unit=True):
    y, x = np.mgrid[:h, :w]
    az = ((x + 0.5) / w - 0.5) * 2 * np.pi
    el = np.deg2rad(els[1] - (els[1] - els[0]) * (y + 0.5) / h)
    v = np.stack([np.broadcast_to(np.cos(az), el.shape), np.sin(az), np.tan(el)], 2)
    return (v * np.cos(el)[:, :, None] if unit else v).astype("f")


def mapping(c, e, grid, origin=None, Rfix=None):
    R = Rotation.from_quat(e["pose"][3:]).as_matrix()
    O = np.array(e["pose"][:3])
    v = grid if origin is None else grid + origin - O
    v = v @ (R if Rfix is None else Rfix @ R)
    C = rig[c]
    p, ok = C.project(v)
    p = np.clip(
        np.nan_to_num(p), [-3 * C.width, -3 * C.height], [4 * C.width, 4 * C.height]
    )
    uv = (p * [B / C.width, H / C.height] - 0.5).astype("f")
    edge = np.minimum.reduce(
        [
            uv[:, :, 0] + 0.5,
            B - 0.5 - uv[:, :, 0],
            uv[:, :, 1] + 0.5,
            H - 0.5 - uv[:, :, 1],
        ]
    )
    return uv, np.clip(edge / 55, 0, 1).astype("f") * ok


def pe(c, n, small=False):
    return (
        photo("live").apply(c, n, picture(c, n, small)[:, :, ::-1])[:, :, ::-1].copy()
    )


def blur(im, s):
    return gaussian_filter(
        im,
        (s, s * 1.6) + ((0,) if im.ndim == 3 else ()),
        mode=("nearest", "wrap") + (("nearest",) if im.ndim == 3 else ()),
    )


def extrapolate(im, w):
    # Multi-scale normalized convolution only in unseen directions; preserve observed radiance.
    v = np.tile((im.sum((0, 1)) / max(1, w.sum())).reshape(1, 1, 3), (*w.shape, 1))
    for s in [18, 6, 2, 0.65]:
        den = blur(w, s)
        num = blur(im, s)
        a = np.clip(den / 0.12, 0, 1)[:, :, None]
        v += a * (num / np.maximum(den[:, :, None], 0.0001) - v)
    return v


def solve():
    ls = load(W / "knots.json")
    allobs = [(c, e) for s in ls for c, e in s.items()]
    low = sphere(128, 36, SKYEL)
    lc = np.zeros((len(allobs), 36, 128, 3), "B")
    lw = np.zeros(lc.shape[:-1], "B")

    def obs(i):
        c, e = allobs[i]
        p, edge = mapping(c, e, low)
        n = e["n"]
        sk = cv.erode(masks(c, n)[0].astype("B"), np.ones((5, 5), "B"))
        im = cv.remap(pe(c, n, True), p, None, cv.INTER_LINEAR)
        w = (
            cv.remap(sk, p, None, cv.INTER_NEAREST)
            * edge
            * (im.min(2) > 12)
            * (im.max(2) < 246)
        )
        lc[i] = im
        lw[i] = np.uint8(w * 255 + 0.5)

    with ThreadPoolExecutor(6) as pool:
        list(pool.map(obs, range(len(allobs))))
    g = np.ones((len(allobs), 3), "f")
    ref = np.array([c == "front" for c, e in allobs])
    fine = []
    for iteration in range(5):
        sums = np.zeros(low.shape, "f")
        ws = np.zeros(low.shape[:2], "f")
        for i in range(len(g)):
            w = lw[i] / 255.0
            val = lc[i] * g[i]
            if iteration:
                w *= np.minimum(1, 17 / np.maximum(1, np.abs(val - field).mean(2)))
            sums += val * w[:, :, None]
            ws += w
        support = np.clip((ws - 6) / 24, 0, 1).astype("f")
        avg = sums / np.maximum(1, ws[:, :, None])
        field = extrapolate(avg * support[:, :, None], support)
        field = blur(field, 1.1)
        new = []
        for i in range(len(g)):
            ok = lw[i] > 55
            new.append(
                np.median(
                    np.log(np.maximum(10, field[ok]) / np.maximum(10, lc[i][ok])), 0
                )
                if ok.sum() >= 12
                else [np.nan] * 3
            )
        l = np.array(new)
        anchor = np.nanmedian(l[ref], 0)
        l -= anchor
        for c in CAMS:
            idx = [i for i, (cc, e) in enumerate(allobs) if cc == c]
            valid = np.isfinite(l[idx, 0])
            if valid.any():
                for k in range(3):
                    l[idx, k] = np.interp(
                        idx, np.array(idx)[valid], l[np.array(idx)[valid], k]
                    )
            else:
                l[idx] = 0
        g = np.exp(np.clip(l, -0.48, 0.48)).astype("f")
    gains = {f"{c}:{e['n']}": v.tolist() for (c, e), v in zip(allobs, g)}
    dump(W / "gains.json", gains)
    # A common infinity texture is intentional: clouds do not move with a source switch. Detail is taken
    # from a tightly spaced time epoch first. Later photographs compete only for remaining spherical gaps.
    coarse = cv.resize(field, (SW, SH), interpolation=cv.INTER_CUBIC)
    hi = sphere(SW // 3, SH // 2, SKYEL)
    detail = np.zeros_like(coarse)
    weight = np.zeros((SH, SW), "f")
    epoch = min(range(len(ls)), key=lambda k: abs(ls[k]["front"]["t"] - 112))
    indices = [epoch] + [
        k for k in np.linspace(0, len(ls) - 1, 64, dtype=int) if k != epoch
    ]
    for j, k in enumerate(indices):
        for c, e in ls[k].items():
            uv, edge = mapping(c, e, hi)
            p = cv.resize(uv, (SW, SH))
            ed = cv.resize(edge, (SW, SH))
            n = e["n"]
            sk = cv.erode(masks(c, n)[0].astype("B"), np.ones((5, 5), "B")).astype("f")
            w = cv.remap(sk, p, None, cv.INTER_LINEAR) * ed
            im = (
                cv.remap(pe(c, n), p * 2 + 0.5, None, cv.INTER_LINEAR)
                * gains[f"{c}:{n}"]
            )
            lb = cv.GaussianBlur(w, (0, 0), 11)
            smooth = cv.GaussianBlur(
                (im * w[:, :, None]).astype("f"), (0, 0), 11
            ) / np.maximum(0.0001, lb[:, :, None])
            hp = np.clip(im - smooth, -38, 38)
            co = 0.95 if j == 0 else 0.33
            v = w * co * (1 - np.minimum(1, weight))
            detail += v[:, :, None] * hp
            weight += v
    sky = np.uint8(
        np.clip(coarse + detail / np.maximum(0.6, weight[:, :, None]) + 0.5, 0, 255)
    )
    # Reject isolated windscreen dirt and tiny mask leaks without flattening cloud gradients.
    med = cv.medianBlur(sky, 7)
    sky = np.maximum(sky, np.maximum(0, med.astype("i") - 8).astype("B"))
    assert cv.imwrite(str(OUT / "sky.jpg"), sky, [cv.IMWRITE_JPEG_QUALITY, 94])
    res = []
    for i in range(len(g)):
        ok = lw[i] > 55
        if ok.sum() >= 12:
            res.append(np.median(abs(lc[i][ok] * g[i] - field[ok]), 0))
    dump(
        W / "sky-audit.json",
        dict(
            observations=len(g),
            calibrated=len(res),
            medianSourceResidualBGR=np.median(res, 0).tolist(),
            detailCoverage=float((weight > 0.35).mean()),
            skyEpoch=ls[epoch]["front"]["t"],
        ),
    )
    print("sky", len(res), "source fits complete", flush=True)


if __name__ == "__main__":
    solve()
