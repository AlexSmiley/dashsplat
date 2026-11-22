"""Bake a continuous, world-aligned photographic sky and finite-radius curtain per source knot."""

from .v3_input import *
from scipy.ndimage import gaussian_filter1d, median_filter, distance_transform_edt
from scipy.spatial.transform import Rotation
from concurrent.futures import ThreadPoolExecutor

SW, SH = 1024, 180
BW, BH = 1536, 168
SKY_EL = [-9, 64]
BACK_EL = [-7, 27]


def sphere(w, h, el):
    y, x = np.mgrid[:h, :w]
    a = ((x + 0.5) / w - 0.5) * 2 * np.pi
    b = np.deg2rad(el[1] - (el[1] - el[0]) * (y + 0.5) / h)
    return np.stack(
        [np.cos(b) * np.cos(a), np.cos(b) * np.sin(a), np.sin(b)], -1
    ).astype("f")


SK = sphere(SW, SH, SKY_EL)
BA = sphere(BW, BH, BACK_EL)
BA /= np.linalg.norm(BA[:, :, :2], axis=2)[:, :, None]


@lru_cache(None)
def gains(c):
    if c not in C3:
        return None
    ls = json.load(open(DST / (c + "-final.json")))["base"]
    return np.array([a["t"] for a in ls]), np.array([a["gain"] for a in ls])


def gain(c, t):
    r = gains(c if c in C3 else "front")
    return np.array([np.interp(t, r[0], r[1][:, i]) for i in range(3)], "f")


def smear(im, m):
    # Cyclic support; fill small disocclusions locally and larger holes with low-frequency observed colour.
    from scipy.ndimage import gaussian_filter

    v = im / np.maximum(1e-5, m[:, :, None])
    base = np.broadcast_to(im.sum((0, 1)) / max(1.0, m.sum()), v.shape).copy()
    for sx, sy, conf in [(34, 12, 0.12), (12, 5, 0.16), (3, 2, 0.24)]:
        den = gaussian_filter(m, (sy, sx), mode=("nearest", "wrap"))
        num = gaussian_filter(im, (sy, sx, 0), mode=("nearest", "wrap", "nearest"))
        a = np.clip(den / conf, 0, 1)[:, :, None]
        base += a * (num / np.maximum(1e-5, den[:, :, None]) - base)
    a = np.clip(m / 0.45, 0, 1)[:, :, None]
    return base + a * (v - base)


def pano(arg):
    k, src = arg
    origin = np.array(src["front"]["rowPoses"])[:, :3].mean(0)
    data = []
    dist = []
    for c, e in src.items():
        rgb = cv.resize(cv.imread(e["rgb"]), (B, H), interpolation=cv.INTER_AREA)[
            :, :, ::-1
        ]
        cat = cv.imread(e["mask"], -1)
        z = cv.imread(e["z"], -1) * 0.01
        r = rays(c)
        rr = (
            z * np.linalg.norm(r[:, :, :2], axis=2) / np.maximum(0.06, r[:, :, 2])
        )  # world-horizontal range, not camera-horizontal radius
        rot, oo = rowmat(e, np.array([0.5]))
        rot = rot[0]
        oo = oo[0]
        rr = (
            z
            * np.linalg.norm((r @ rot.T)[:, :, :2], axis=2)
            / np.maximum(0.06, r[:, :, 2])
        )
        far = (cat == 1) & (rr > 45) & (rr < 260)
        sem0 = sem(c, e)
        if c in C3 and sem0.exists():
            sf = cv.imread(str(sem0), 0)
            far &= ~np.isin(sf, [0, 1, 9, 10, 11, 12, 13, 14, 15, 17, 18])
        dist.extend(rr[far][::160])
        data.append((c, e, rot, oo, rgb, cat, far, rr))
    radius = float(np.clip(np.median(dist) if len(dist) else 76, 48, 125))
    bg = np.zeros((BH, BW, 3), "f")
    bgw = np.zeros((BH, BW), "f")
    sk = np.zeros((SH, SW, 3), "f")
    skw = np.zeros((SH, SW), "f")
    ws = BA * radius + origin
    fac = 1
    for c, e, R, O, rgb, cat, far, rr in data:
        # Broad C1 lens taper; camera gains referenced to front without a display grade.
        for xyz, obs, s, w, finite in [
            (SK, cat == 3, sk, skw, False),
            (ws, far, bg, bgw, True),
        ]:
            vec = (xyz - O if finite else xyz) @ R
            C = rig[c]
            uv, ok = C.project(vec)
            uv = (
                np.nan_to_num(uv).astype("f") * np.float32([B / C.width, H / C.height])
                - 0.5
            )
            edge = np.minimum.reduce(
                [uv[:, :, 0], B - 1 - uv[:, :, 0], uv[:, :, 1], H - 1 - uv[:, :, 1]]
            )
            ww = np.clip(edge / (45 if finite else 74), 0, 1)
            ww *= ww * (3 - 2 * ww) * ok
            mask = cv.remap(obs.astype("f"), uv, None, cv.INTER_LINEAR)
            ww *= mask
            col = cv.remap(rgb, uv, None, cv.INTER_LINEAR).astype("f") * gain(c, e["t"])
            s += col * ww[:, :, None]
            w += ww
    sky = smear(sk, skw)
    seen = bgw > 0.06
    observed = bg / np.maximum(0.001, bgw[:, :, None])
    rows = np.arange(BH)[:, None]
    gplane = src["front"]["groundPlane"]
    floor = -(origin[:2] @ gplane[:2] + gplane[3]) / gplane[2]
    end = int(
        np.clip(
            (BACK_EL[1] - np.rad2deg(np.arctan2(floor - origin[2] - 0.65, radius)))
            / (BACK_EL[1] - BACK_EL[0])
            * BH,
            0,
            BH - 1,
        )
    )  # Avoid isolated wire/pole pixels turning a high gantry into a solid wall.
    density = cv.boxFilter(seen.astype("f"), -1, (7, 5))
    solid = seen & (density > 0.48)
    top = np.argmax(solid, axis=0).astype("f")
    ok = solid.sum(0) >= 3
    ii = np.flatnonzero(ok)
    fallback = max(0, end - 6)
    if len(ii) > 1:
        top = np.interp(np.arange(BW), np.r_[ii - BW, ii, ii + BW], np.tile(top[ii], 3))
    else:
        top = np.full(BW, fallback)
    top = gaussian_filter1d(median_filter(top, size=9, mode="wrap"), 1.5, mode="wrap")
    top = np.minimum(top, end - 3)
    alpha = np.clip((rows - top + 0.5) / 2.5, 0, 1) * np.clip(
        (end + 3 - rows) / 3, 0, 1
    )
    alpha = np.maximum(
        alpha, np.minimum(1, bgw * 2)
    )  # Preserve real isolated branches above the skyline.
    # Extend photo texture down to measured ground, not a hazy gap; for unsupported columns use local angular fill.
    bg = smear(bg, bgw)
    use = seen.any(0)
    ridge = np.maximum(0, np.where(seen, rows, -1).max(0))
    ix = np.arange(BW)
    last = bg[ridge, ix]
    mix = np.clip((rows - ridge) / 6, 0, 1) * use
    bg = bg * (1 - mix[:, :, None]) + last[None] * mix[:, :, None]
    stem = f"pano/{k:04}"
    (BUILD / "pano").mkdir(parents=True, exist_ok=True)
    (W / "confidence").mkdir(exist_ok=True)
    paths = {}
    for key, suf, img in [
        ("sky", "-s.jpg", sky),
        ("rgb", "-b.jpg", bg),
        ("matte", "-m.png", alpha * 255),
    ]:
        path = stem + suf
        im = np.uint8(np.clip(img + 0.5, 0, 255))
        im = im[:, :, ::-1] if im.ndim == 3 else im
        assert cv.imwrite(
            str(BUILD / path),
            im,
            [cv.IMWRITE_JPEG_QUALITY, 88]
            if suf.endswith("jpg")
            else [cv.IMWRITE_PNG_COMPRESSION, 5],
        )
        paths[key] = path
    cv.imwrite(
        str(W / "confidence" / f"{k:04}.png"), np.uint8(np.clip(bgw, 0, 1) * 255)
    )
    return dict(
        id=k, t=src["front"]["t"], origin=origin.tolist(), radius=radius, **paths
    )


def build():
    src = getsets()
    BUILD.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(4) as pool:
        ps = list(pool.map(pano, enumerate(src)))
    dump(W / "panos.json", ps)
    print("panoramas", len(ps), flush=True)


if __name__ == "__main__":
    build()
