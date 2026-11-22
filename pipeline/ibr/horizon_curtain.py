"""Project real photographs, fit bounded SO(3) corrections on static feature tracks, then choose one donor per sector.
The colour fusion is a selection, never an average of disagreeing silhouettes. All intermediate masks are evidence.
"""

from .horizon_sky import *
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.optimize import least_squares
import math

cv.setNumThreads(1)
GRID = sphere(BW // 4, BH // 4, BACKEL, False)
YY = np.arange(BH)[:, None]
DEGPX = (BACKEL[1] - BACKEL[0]) / BH


@lru_cache(None)
def rows():
    return load(W / "knots.json")


@lru_cache(None)
def gains():
    return load(W / "gains.json")


def rng(c, e):
    n = e["n"]
    p = RS / "depth" / c / f"{n:06}.npy"
    if not p.is_file():
        return None
    r, valid = lattice(c)
    R = Rotation.from_quat(e["pose"][3:]).as_matrix()
    z = np.load(p).astype("f")
    return np.where(
        valid & (z > 0.15),
        z * np.linalg.norm((r @ R.T)[:, :, :2], axis=2) / np.maximum(0.04, r[:, :, 2]),
        0,
    ).astype("f")


@lru_cache(None)
def origins():
    return np.array([s["front"]["pose"][:3] for s in rows()])


@lru_cache(None)
def radii():
    frozen = W / "radii.npy"
    if frozen.is_file():
        return np.load(frozen)

    def radius(s):
        ds = []
        for c, e in s.items():
            r = rng(c, e)
            if r is None:
                continue
            sk, rd, dy = masks(c, e["n"])
            v = r[~(sk | rd | dy) & (r >= 45) & (r < 250)]
            ds.extend(v[::100])
        return float(np.clip(np.median(ds), 58, 150)) if len(ds) > 35 else 90.0

    with ThreadPoolExecutor(4) as pool:
        a = np.array(list(pool.map(radius, rows())))
    out = gaussian_filter1d(a, 2.4, mode="nearest")
    temp = W / f"radii-{os.getpid()}.npy"
    np.save(temp, out)
    try:
        os.link(temp, frozen)
    except FileExistsError:
        pass
    temp.unlink()
    return np.load(frozen)


def rotation(ref, donor, overlap):
    # Lucas-Kanade correspondences in panorama pixel space; fit all three angular DOFs, not a deforming mesh.
    p = cv.goodFeaturesToTrack(
        donor, 240, 0.015, 5, mask=overlap.astype("B"), blockSize=5
    )
    if p is None or len(p) < 9:
        return np.eye(3), dict(n=0, accepted=False)
    q, ok, _ = cv.calcOpticalFlowPyrLK(
        donor, ref, p, None, winSize=(19, 19), maxLevel=2
    )
    pp, back, _ = cv.calcOpticalFlowPyrLK(
        ref, donor, q, None, winSize=(19, 19), maxLevel=2
    )
    p = p[:, 0] * 2
    q = q[:, 0] * 2
    good = (
        (ok[:, 0] * back[:, 0] > 0)
        & (np.linalg.norm(pp[:, 0] * 2 - p, axis=1) < 1.6)
        & (np.linalg.norm(q - p, axis=1) < 20)
    )
    p = p[good]
    q = q[good]
    if len(p) < 9:
        return np.eye(3), dict(n=len(p), accepted=False)
    a = (p[:, 0] / BW - 0.5) * 2 * np.pi
    b = np.deg2rad(BACKEL[1] - (p[:, 1] / BH) * (BACKEL[1] - BACKEL[0]))
    fx = BW / (2 * np.pi)
    fy = BH / np.deg2rad(BACKEL[1] - BACKEL[0])
    J = np.stack(
        (
            fx
            * np.stack(
                [-np.cos(a) * np.tan(b), -np.sin(a) * np.tan(b), np.ones(len(a))], 1
            ),
            fy * np.stack([-np.sin(a), np.cos(a), np.zeros(len(a))], 1),
        ),
        1,
    )
    delta = q - p
    fit = least_squares(
        lambda v: (J @ v - delta).ravel(),
        np.zeros(3),
        loss="soft_l1",
        f_scale=0.9,
        max_nfev=18,
    )
    v = fit.x
    e = np.linalg.norm(J @ v - delta, axis=1)
    keep = e < 2.0
    if keep.sum() >= 8:
        v = np.linalg.lstsq(J[keep].reshape(-1, 3), delta[keep].ravel(), rcond=1.0e-3)[
            0
        ]
        e = np.linalg.norm(J @ v - delta, axis=1)
    pre = float(np.median(np.linalg.norm(delta, axis=1)))
    post = float(np.median(e))
    deg = float(np.rad2deg(np.linalg.norm(v)))
    accepted = (
        len(p) >= 9
        and deg < 1.65
        and (e < 2).sum() >= 8
        and post < pre + 0.05
        and post < 2.4
    )
    return (Rotation.from_rotvec(v).as_matrix() if accepted else np.eye(3)), dict(
        n=len(p),
        pre=pre,
        post=post,
        deg=deg,
        accepted=bool(accepted),
        within2=float(np.mean(e < 2)),
    )


def wrapuv(v):
    a = np.arctan2(v[:, :, 1], v[:, :, 0])
    a = np.unwrap(np.unwrap(a, axis=1), axis=0)
    e = np.rad2deg(np.arctan2(v[:, :, 2], np.linalg.norm(v[:, :, :2], axis=2)))
    return np.stack(
        [
            (0.5 + a / (2 * np.pi)) * BW - 0.5,
            (BACKEL[1] - e) / (BACKEL[1] - BACKEL[0]) * BH - 0.5,
        ],
        2,
    ).astype("f")


def remap(im, uv, nearest=False):
    return cv.remap(
        im,
        uv,
        None,
        cv.INTER_NEAREST if nearest else cv.INTER_LINEAR,
        borderMode=cv.BORDER_WRAP,
    )


def profile(sk, rd, bad, z, edge, im, start, stop):
    valid = edge > 0.035
    static = valid & (sk < 0.48) & (rd < 0.50) & (bad < 0.45)
    density = cv.boxFilter(static.astype("f"), -1, (11, 11))
    deeper = density[np.minimum(np.arange(BH) + 8, BH - 1)]
    above = np.cumsum(((sk > 0.62) & valid).astype("i"), axis=0)
    support = (density > 0.56) & (deeper > 0.64) & static & (above >= 4)
    top = np.argmax(support, axis=0)
    has = support.any(0)
    top = median_filter(top, 3, mode="wrap")
    el = BACKEL[1] - (YY + 0.5) * DEGPX
    # Snap the semantic contour to a source-image gradient within its label uncertainty (≤3 curtain texels).
    lum = cv.cvtColor(im, cv.COLOR_BGR2GRAY).astype("f")
    gy = np.abs(cv.Sobel(lum, -1, 0, 1)) * 0.125
    xx = np.arange(BW)
    dy = np.arange(-3, 4)[:, None]
    yidx = np.clip(top + dy, 0, BH - 1)
    top += dy[:, 0][np.argmax(gy[yidx, xx] - 0.95 * np.abs(dy), axis=0)]
    top = np.clip(top, 0, BH - 1)
    feed = static & (YY >= top) & (YY < top + 24)
    # Ground contact gives a conservative skyline-range estimate where shared metric depth is absent.
    foot = np.argmax((rd > 0.52) & valid & (YY > top), axis=0)
    foot = np.where(foot > top, foot, BH - 1)
    pel = np.deg2rad(BACKEL[1] - (foot + 0.5) * DEGPX)
    proxy = np.clip(1.55 / np.maximum(0.007, -np.tan(pel)), 8, 220)
    proxy = median_filter(proxy, 15, mode="wrap")
    if z is not None:
        measurable = feed & (z > 2) & (z < 350)
        far = (z >= start) & (z <= stop)
        frac = (measurable & far).sum(0) / np.maximum(1, measurable.sum(0))
        frac = np.where(
            measurable.sum(0) >= 3,
            frac,
            0.50 * np.clip((proxy - start + 10) / 20, 0, 1),
        )
        interior = static & ((z >= start * 0.87) | (z <= 1)) & (z <= stop + 8)
    else:
        frac = 0.66 * np.clip((proxy - start * 0.70) / (start * 0.55), 0, 1)
        interior = static
    # Ignore false ceiling (completely occluded directions) and lens borders. Score a contiguous column, not pixels.
    conf = np.where(
        has,
        (0.06 + 0.94 * frac)
        * edge[top, xx]
        * np.clip((15 - (BACKEL[1] - top * DEGPX)) / 6, 0.08, 1),
        0,
    )
    conf = gaussian_filter1d(conf, 2, mode="wrap")
    usable = interior & (YY >= top - 2)
    last = (
        np.where(usable & (YY <= top + 110), YY, -1).max(0).clip(0, BH - 1)
    )  # Do not drag nearby vehicles/ground up into the horizon: continue the last distant measured row below its foot.
    # A streetlight/wire above the actual treeline must not drag an opaque vertical sheet of sky behind it.
    # Distance-transform the observed semantic edge (including holes between branches) to a 1.5 px feather.
    solid = (sk < 0.48) & (bad < 0.6) & valid
    land = (YY >= top - 5) & solid
    if z is not None:
        land |= (
            solid
            & (z >= start)
            & (z < stop)
            & (YY < top)
            & (YY >= top - 25)
            & (density > 0.20)
        )
    real = cv.distanceTransform(land.astype("B"), cv.DIST_L2, 3) - cv.distanceTransform(
        (~land).astype("B"), cv.DIST_L2, 3
    )
    m = np.clip(0.5 + real / 2.8, 0, 1)
    # Use an observed ground colour field below the last distant sample, not a vertically extruded tree or lane marking.
    gw = (rd > 0.65) & valid & (bad < 0.4)
    base = np.median(im[gw][::31], 0) if gw.sum() > 90 else np.array([97, 104, 90])
    den = cv.GaussianBlur(gw.astype("f"), (0, 0), 18)
    road = cv.GaussianBlur(im.astype("f") * gw[:, :, None], (0, 0), 18) / np.maximum(
        0.001, den[:, :, None]
    )
    road = base + (road - base) * np.clip(den / 0.30, 0, 1)[:, :, None]
    # Aggressive near-depth veto only below the actual horizon. Avoid copying holes or current-time cars into a curtain.
    fill = np.clip((YY - last + 1) / 12, 0, 1)
    fill = np.maximum(fill, np.clip(bad, 0, 1) * (YY >= top))
    m = np.where((YY > last + 3) & (YY >= top) & has[None], np.maximum(m, fill), m)
    packed = np.uint8(np.clip(im + (road - im) * fill[:, :, None] + 0.5, 0, 255))
    alpha = np.uint8(np.clip(m * 255 + 0.5, 0, 255))
    return packed, alpha, conf.astype("f"), top, interior


def photochart(c, e, i, Rfix, start, stop):
    radius = radii()[i] if stop > 200 else (start + stop) * 0.5
    uv, edge = mapping(c, e, GRID * radius, origins()[i], Rfix)
    p = cv.resize(uv, (BW, BH))
    ed = cv.resize(edge, (BW, BH))
    n = e["n"]
    im = cv.remap(pe(c, n), 2 * p + 0.5, None, cv.INTER_LINEAR)
    im = np.uint8(np.clip(im * np.array(gains()[f"{c}:{n}"], "f") + 0.5, 0, 255))
    sk, rd, dy = masks(c, n)
    sk = cv.remap(sk.astype("f"), p, None, cv.INTER_LINEAR)
    rd = cv.remap(rd.astype("f"), p, None, cv.INTER_LINEAR)
    dy = cv.remap(dy.astype("f"), p, None, cv.INTER_LINEAR)
    z = rng(c, e)
    z = cv.remap(z, p, None, cv.INTER_NEAREST) if z is not None else None
    return profile(sk, rd, dy, z, ed, im, start, stop)


def primary(i, start=45, stop=320):
    folder = W / f"primary-{start}-{stop}"
    folder.mkdir(exist_ok=True)
    stem = folder / f"{i:04}"
    if stem.with_suffix(".npz").exists():
        return
    order = [
        "front",
        "left_pillar",
        "right_pillar",
        "back",
        "left_repeater",
        "right_repeater",
    ]
    best = np.zeros(BW, "f")
    colour = np.zeros((BH, BW, 3), "B")
    alpha = np.zeros((BH, BW), "B")
    valid = np.zeros_like(alpha)
    cam = np.full(BW, -1, "b")
    diag = []
    for ci, c in enumerate(order):
        e = rows()[i][c]
        im, ma, conf, top, ok = photochart(c, e, i, None, start, stop)
        ov = (valid > 0) & ok & (ma > 180) & (alpha > 180)
        R, q = rotation(
            cv.resize(cv.cvtColor(colour, cv.COLOR_BGR2GRAY), (BW // 2, BH // 2)),
            cv.resize(cv.cvtColor(im, cv.COLOR_BGR2GRAY), (BW // 2, BH // 2)),
            ov[::2, ::2],
        )
        diag.append(q)
        if q["accepted"]:
            im, ma, conf, top, ok = photochart(c, e, i, R, start, stop)
        conf *= 1.22 if c == "front" else 0.98 if "repeater" in c else 1.0
        take = (
            conf > best
        )  # single colour/matte/footprint donor for the entire longitude sector
        colour[:, take] = im[:, take]
        alpha[:, take] = ma[:, take]
        valid[:, take] = ok[:, take] * 255
        best[take] = conf[take]
        cam[take] = ci
    assert cv.imwrite(
        str(stem.with_suffix(".jpg")), colour, [cv.IMWRITE_JPEG_QUALITY, 95]
    )
    assert cv.imwrite(
        str(stem.with_suffix(".png")),
        np.dstack([alpha, valid, np.zeros_like(alpha)]),
        [cv.IMWRITE_PNG_COMPRESSION, 3],
    )
    dump(stem.with_suffix(".json"), diag)
    np.savez_compressed(stem.with_suffix(".npz"), conf=best, cam=cam)


@lru_cache(20)
def read(i, start, stop):
    st = W / f"primary-{start}-{stop}" / f"{i:04}"
    a = cv.imread(str(st.with_suffix(".jpg")))
    ma = cv.imread(str(st.with_suffix(".png")))
    j = np.load(st.with_suffix(".npz"))
    return a, ma, j["conf"], j["cam"]


def reproject(k, i, R, start, stop):
    r = radii()[i] if stop > 200 else 0.5 * (start + stop)
    v = (
        GRID * r + origins()[i] - origins()[k]
    )  # rotation maps the contributing source rays into target world; inverse is applied for lookup.
    uv = wrapuv(v @ R)
    uv = cv.resize(uv, (BW, BH))
    uv[:, :, 1] = uv[:, :, 1].clip(0, BH - 1)
    a, ma, w, cam = read(k, start, stop)
    col = remap(a, uv)
    mm = remap(ma, uv)
    ix = np.rint(uv[BH // 2, :, 0]).astype(int) % BW
    return col, mm, w[ix], cam[ix]


def braid(i, start=45, stop=320, save=None):
    a, ma, best, lead = read(i, start, stop)
    best = best.copy()
    rgb = a.copy()
    al = ma[:, :, 0].copy()
    stats = []
    donor = np.full(BW, i, "i")
    t = rows()[i]["front"]["t"]
    candidate = sorted(
        (
            k
            for k in range(max(0, i - 5), min(len(rows()), i + 6))
            if i != k
            and np.linalg.norm(origins()[k] - origins()[i]) < 22
            and abs(rows()[k]["front"]["t"] - t) < 8
        ),
        key=lambda k: abs(k - i),
    )
    ref = cv.resize(cv.cvtColor(a, cv.COLOR_BGR2GRAY), (BW // 2, BH // 2))
    for k in candidate:
        col, mm, w, cams = reproject(k, i, np.eye(3), start, stop)
        other = cv.resize(cv.cvtColor(col, cv.COLOR_BGR2GRAY), (BW // 2, BH // 2))
        ov = (
            (ma[::2, ::2, 1] > 192)
            & (mm[::2, ::2, 1] > 192)
            & (ma[::2, ::2, 0] > 192)
            & (mm[::2, ::2, 0] > 192)
        )
        dist = np.linalg.norm(origins()[k] - origins()[i])
        for ci in np.unique(cams):
            if ci < 0:
                continue
            cm = cams == ci
            R, q = rotation(ref, other, ov & cm[None, ::2])
            stats.append(q)
            im, m, conf, cam = (
                reproject(k, i, R, start, stop) if q["accepted"] else (col, mm, w, cams)
            )
            score = (
                conf
                * np.exp(-dist / 27.0)
                * (0.94 if q["accepted"] else 0.68 if q["n"] < 9 else 0.52)
            )
            take = (
                (cam == ci)
                & (
                    gaussian_filter1d(score, 4, mode="wrap")
                    > gaussian_filter1d(best, 4, mode="wrap") + 0.018
                )
                & (score > 0.015)
            )
            rgb[:, take] = im[:, take]
            al[:, take] = m[:, take, 0]
            best[take] = score[take]
            donor[take] = k
    if stop > 200 and np.mean(best < 0.19) > 0.035:
        # A nearby crown can block several consecutive frames. Look around it using geographically nearby
        # cameras farther along the street, but only inside weak sectors; never replace a good measurement.
        neighbors = sorted(
            set(np.clip(i + np.array([-16, -11, -8, 8, 11, 16]), 0, len(rows()) - 1)),
            key=lambda k: abs(i - k),
        )
        for k in neighbors:
            if k == i or np.linalg.norm(origins()[k] - origins()[i]) > 85:
                continue
            st = W / f"primary-{start}-{stop}" / f"{k:04}.npz"
            if not st.exists():
                continue
            col, mm, w, cams = reproject(k, i, np.eye(3), start, stop)
            gray = cv.resize(cv.cvtColor(col, cv.COLOR_BGR2GRAY), (BW // 2, BH // 2))
            over = (
                (ma[::2, ::2, 1] > 192)
                & (mm[::2, ::2, 1] > 192)
                & (ma[::2, ::2, 0] > 192)
                & (mm[::2, ::2, 0] > 192)
            )
            for ci in np.unique(cams[(best < 0.19) & (w > 0.35)]):
                if ci < 0:
                    continue
                R, fit = rotation(ref, gray, over & (cams[None, ::2] == ci))
                stats.append(fit)
                im, m, conf, cam = (
                    reproject(k, i, R, start, stop)
                    if fit["accepted"]
                    else (col, mm, w, cams)
                )
                score = 0.62 * conf
                take = (
                    (cam == ci) & (best < 0.19) & (score > best + 0.06) & (conf > 0.35)
                )
                rgb[:, take] = im[:, take]
                al[:, take] = m[:, take, 0]
                best[take] = score[take]
                donor[take] = k
    # Fill only genuinely unobserved sectors with contiguous real profiles, cyclically. No RGB interpolation between silhouettes.
    supported = best > 0.105
    idx = np.flatnonzero(supported)
    u = np.arange(BW)
    filled = int((~supported).sum())
    if stop < 200:
        # An optional mid band is an observed surface only. Never replicate its occluded/absent sectors.
        floor = (BACKEL[1] - np.rad2deg(np.arctan2(-1.8, 0.5 * (start + stop)))) / DEGPX
        al = np.uint8(
            al
            * np.clip((best - 0.19) / 0.40, 0, 1)[None]
            * np.clip((floor + 3 - YY) / 3, 0, 1)
        )
    elif len(idx):
        # Unsupported angles get a low-frequency continuation bounded by *both* measured silhouettes.
        # No repeating a pole, single scanline or opaque mirror-image tree through a missing sector.
        xp = np.r_[idx - BW, idx, idx + BW]
        u = np.arange(BW)
        j = np.searchsorted(xp, u)
        L = xp[j - 1] % BW
        R = xp[j] % BW
        gap = xp[j] - xp[j - 1]
        f = ((u - xp[j - 1]) / np.maximum(1, gap)).astype("f")
        bad = ~supported
        xb = u[bad]
        contour = cv.boxFilter((al > 160).astype("f"), -1, (5, 9)) > 0.54
        top = np.argmax(contour, axis=0)
        smooth = median_filter(top, 31, mode="wrap")
        curve = np.interp(u, xp, np.tile(smooth[idx], 3))
        # For long gaps bend gently towards the adjacent robust treeline; only synthesise an edge where no exposure exists.
        curve = gaussian_filter1d(curve, 2.0, mode="wrap")
        # A single pole or high crown beside a long blind sector is not a distant ridge.
        baseline = np.percentile(smooth[supported], 70)
        f0 = np.clip(np.minimum(u - xp[j - 1], xp[j] - u) / 38.0, 0, 1)
        curve += f0 * np.maximum(0, baseline - 15 - curve)
        near = np.minimum(u - xp[j - 1], xp[j] - u)
        m = np.clip(0.5 + (YY - curve) / 1.6, 0, 1)
        low = cv.GaussianBlur(rgb, (0, 0), 4)
        y = np.arange(BH)[:, None]
        lv = low[
            np.clip(np.rint(y - curve[bad] + top[L[bad]]).astype(int), 0, BH - 1),
            L[bad],
        ]
        rv = low[
            np.clip(np.rint(y - curve[bad] + top[R[bad]]).astype(int), 0, BH - 1),
            R[bad],
        ]
        # Match heights before interpolating low frequencies. This mixes no competing measured structures.
        rgb[:, bad] = np.uint8(lv + (rv.astype("f") - lv) * f[bad][None, :, None])
        al[:, bad] = np.uint8(255 * m[:, bad])
    else:
        # Fully unobserved (e.g. covered parking): don't claim an outdoor silhouette.
        al.fill(0)
    folder = save or OUT
    folder.mkdir(parents=True, exist_ok=True)
    name = f"{i:04}"
    assert cv.imwrite(str(folder / (name + ".jpg")), rgb, [cv.IMWRITE_JPEG_QUALITY, 94])
    assert cv.imwrite(
        str(folder / (name + ".png")), al, [cv.IMWRITE_PNG_COMPRESSION, 4]
    )
    return dict(
        id=i,
        filledColumns=filled,
        neighbourColumns=int((donor != i).sum()),
        rotation=stats,
    )


def anchors():
    ls = rows()
    r = [0]
    for i in range(1, len(ls) - 1):
        e = ls[i]["front"]
        last = ls[r[-1]]["front"]
        if (
            e["s"] - last["s"] >= (6.5 if priority(e["t"]) == 0 else 10.0)
            or e["t"] - last["t"] >= 3.9
        ):
            r.append(i)
    return r + [len(ls) - 1]


def build():
    a = anchors()
    diag = []
    with ThreadPoolExecutor(4) as pool:
        for g in range(4):
            sites = [i for i in a if priority(rows()[i]["front"]["t"]) == g]
            needed = sorted(
                {
                    j
                    for i in sites
                    for j in range(max(0, i - 16), min(len(rows()), i + 17))
                }
            )
            list(pool.map(primary, needed))
            group = list(pool.map(braid, sites))
            diag.extend(group)
            dump(W / f"curtains-group-{g}.json", group)
            print("completed region", g, len(group), flush=True)
    dump(W / "curtains.json", diag)
    print("curtains", len(a), "finished", flush=True)


if __name__ == "__main__":
    build()
