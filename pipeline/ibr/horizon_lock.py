"""Bake coherent photographic sectors onto stationary neighbourhood cylinders. Paired RGB/matte ownership is solved on a closed longitude ring; no disagreeing imagery is averaged."""

from .horizon_curtain import *
from functools import lru_cache
from scipy.ndimage import maximum_filter1d

LOCK = W / "stations-final"
LOCK.mkdir(exist_ok=True)


@lru_cache(None)
def stations():
    a = anchors()
    ss = np.array([r["front"]["s"] for r in rows()])
    reps = [a[0]]
    for i in a[1:]:
        if ss[i] - ss[reps[-1]] >= 27 or (
            rows()[i]["front"]["t"] - rows()[reps[-1]]["front"]["t"] > 19
            and ss[i] - ss[reps[-1]] > 5
        ):
            reps.append(i)
    if ss[a[-1]] - ss[reps[-1]] > 8:
        reps.append(a[-1])
    return {
        i: min(
            reps,
            key=lambda j: (
                abs(ss[i] - ss[j])
                + 0.003 * abs(rows()[i]["front"]["t"] - rows()[j]["front"]["t"])
            ),
        )
        for i in a
    }


@lru_cache(64)
def source(c, k):
    e = rows()[k][c]
    n = e["n"]
    sk, rd, dy = masks(c, n)
    return sk.astype("B"), rd.astype("B"), dy.astype("B"), rng(c, e), pe(c, n, True)


def chart(i, k, c, R=None, high=False):
    e = rows()[k][c]
    sk, rd, dy, depth, small = source(c, k)
    uv, ed = mapping(c, e, GRID * radii()[i], origins()[i], R)
    p = cv.resize(uv, (BW, BH)) if high else uv
    ed = cv.resize(ed, (BW, BH)) if high else ed
    rm = lambda im, nn=False: cv.remap(
        im, p, None, cv.INTER_NEAREST if nn else cv.INTER_LINEAR
    )
    sky = rm(sk.astype("f"))
    road = rm(rd.astype("f"))
    bad = rm(dy.astype("f"))
    valid = ed > 0.025
    solid = (sky < 0.50) & (road < 0.60) & (bad < 0.55) & valid
    if high:
        raw = cv.imread(str(RS / "data/drive/frames" / c / f"{e['n']:06}.jpg"))
        im = photo("live").apply(c, e["n"], raw[:, :, ::-1])[:, :, ::-1]
        ratio = im.shape[1] / B
        im = cv.remap(im, (p + 0.5) * ratio - 0.5, None, cv.INTER_LINEAR)
    else:
        im = rm(small)
    im = np.uint8(np.clip(im.astype("f") * gains()[f"{c}:{e['n']}"] + 0.5, 0, 255))
    h, w = sky.shape
    y = np.arange(h)[:, None]
    el = BACKEL[1] - (y + 0.5) / h * (BACKEL[1] - BACKEL[0])
    mass = cv.boxFilter(solid.astype("f"), -1, (9, 9) if high else (3, 3))
    possible = (
        (mass > 0.53)
        & solid
        & (np.cumsum((sky > 0.64) & valid, axis=0) >= max(2, h // 100))
    )
    top = np.argmax(possible, axis=0)
    xx = np.arange(w)
    count = possible.any(0)
    near = solid & (y >= top) & (y < top + max(6, h // 12))
    if depth is not None:
        z = rm(depth, True)
        den = (near & (z > 1)).sum(0)
        far = (near & (z >= 45) & (z < 320)).sum(0) / np.maximum(1, den)
        far = np.where(den >= 2, far, 0.35)
    else:
        foot = np.argmax((road > 0.65) & valid & (y > top), axis=0)
        foot = np.where(foot > top, foot, h - 1)
        angle = np.deg2rad(BACKEL[1] - (foot + 0.5) / h * (BACKEL[1] - BACKEL[0]))
        far = np.clip((1.55 / np.maximum(0.007, -np.tan(angle)) - 24) / 38, 0.03, 1)
    # Source quality is evaluated at a broad patch of sky/land and at its foot, not on a single mask pixel.
    zero = int((BACKEL[1] - 0.0) / (BACKEL[1] - BACKEL[0]) * h)
    avail = ed[zero] * np.maximum(0.1, ed[np.clip(top, 0, h - 1), xx])
    score = (
        0.25
        + 0.67 * gaussian_filter1d(far, 13 * w / BW, mode="wrap")
        + 0.25 * (c == "front")
        + 0.19 * np.clip((top / h - 0.45) * 3, 0, 1)
        - 0.008 * np.linalg.norm(origins()[k] - origins()[i])
    )
    score = np.where(count, score, 0.03)
    score += 0.4 * avail
    score -= np.where(ed[zero] > 0.04, 0, 10)
    score -= 1.5 * np.clip((0.54 - top / h) * 3, 0, 1) * count
    if not high:
        return im, top * 4, score, near, avail
    # Fit the edge to the original-resolution luminance in a narrow band around the observed mask. Preserve holes.
    land = (sky < 0.5) & (bad < 0.5) & valid
    m = land.astype("f")
    lum = cv.cvtColor(im, cv.COLOR_BGR2GRAY).astype("f")
    g = cv.GaussianBlur(m, (0, 0), 2.4)
    means = []
    for v in [m, 1 - m]:
        means.append(
            cv.GaussianBlur(lum * v, (0, 0), 2.4)
            / np.maximum(0.0001, cv.GaussianBlur(v, (0, 0), 2.4))
        )
    edge = (g > 0.06) & (g < 0.94) & (abs(means[0] - means[1]) > 9)
    fraction = np.clip(
        (lum - means[1])
        / np.where(abs(means[0] - means[1]) > 1, means[0] - means[1], 1),
        0,
        1,
    )
    land = np.where(edge, 0.67 * fraction + 0.33 * g > 0.5, land) & valid
    # Lens borders can be classified as land before the first visible sky pixel. A valid skyline
    # has real sky above it locally; allow an uninterrupted facade if its neighbouring rays have sky.
    above = np.cumsum((sky > 0.72) & (ed > 0.10), axis=0) >= 4
    land &= (
        above | (cv.dilate(above.astype("B"), np.ones((1, 31), "B")) > 0) | (el < 0.65)
    )
    if depth is not None:
        z = rm(depth, True)
        close = cv.GaussianBlur(((z > 1) & (z < 39) & land).astype("f"), (0, 0), 2)
        far = cv.GaussianBlur(((z >= 45) & land).astype("f"), (0, 0), 2)
        land &= ~((close > 0.36) & (far < 0.16) & (el > 1.1))
    # Never extrude the lowest observed row. Below the footprint use only a smooth road colour from this photograph.
    gw = (road > 0.65) & valid
    floor = (
        np.median(im[gw][::67], axis=0) if gw.sum() > 400 else np.array([101, 104, 96])
    )
    floor = np.array([105.0, 106.0, 97.0]) + 0.12 * (
        floor - np.array([105.0, 106.0, 97.0])
    )
    ground = np.clip((-el - 0.9) / 2.2, 0, 1)
    ground = ground * ground * (3 - 2 * ground)
    fill = np.maximum(ground, bad * (el < 0))
    col = np.uint8(np.clip(im + (floor - im) * fill[:, :, None] + 0.5, 0, 255))
    solid = land | ((ground > 0.9) & valid)
    dist = cv.distanceTransform(
        solid.astype("B"), cv.DIST_L2, 3
    ) - cv.distanceTransform((~solid).astype("B"), cv.DIST_L2, 3)
    al = np.clip(0.5 + dist / 2.8, 0, 1)
    al = np.maximum(al, ground * valid)
    return col, np.uint8(255 * al + 0.5)


def solve(i):
    path = LOCK / f"{i:04}.json"
    if path.is_file():
        return load(path)
    ss = np.array([r["front"]["s"] for r in rows()])
    ks = list(
        dict.fromkeys(
            [i, *[int(np.argmin(abs(ss - (ss[i] + v)))) for v in [-16, 16, -30, 30]]]
        )
    )
    cams = [
        "front",
        "left_pillar",
        "right_pillar",
        "back",
        "left_repeater",
        "right_repeater",
    ]
    entries = [(k, c) for k in ks for c in cams]
    ims = []
    tops = []
    qs = []
    eligible = []
    mats = []
    fits = []
    for k, c in entries:
        im, y, q, v, av = chart(i, k, c)
        R = np.eye(3)
        gray = cv.resize(cv.cvtColor(im, cv.COLOR_BGR2GRAY), (BW // 2, BH // 2))
        if ims:
            union = np.zeros(v.shape, bool)
            ref = np.zeros(v.shape, "B")
            best = np.zeros(v.shape[1], "f")
            for col, mask, qq in zip(ims[:6], eligible[:6], qs[:6]):
                choose = mask & ((qq > best)[None] | ~union)
                ref[choose] = cv.cvtColor(col, cv.COLOR_BGR2GRAY)[choose]
                union |= mask
                best = np.maximum(best, np.where(mask.any(0), qq, 0))
            R, fit = rotation(
                cv.resize(ref, (BW // 2, BH // 2)),
                gray,
                cv.resize(
                    (v & union).astype("B"),
                    (BW // 2, BH // 2),
                    interpolation=cv.INTER_NEAREST,
                )
                > 0,
            )
            fits.append(fit)
            if fit["accepted"]:
                im, y, q, v, av = chart(i, k, c, R)
        ims.append(im)
        tops.append(y)
        qs.append(q)
        eligible.append(v)
        mats.append(R)
    Q = np.array(qs)
    Y = np.array(tops)
    N, L = Q.shape
    good = Q > 0.55
    # Prefer a measured distant silhouette supported by more than one observation. A near crown standing
    # above other clear sightings is unlikely to be the distant skyline even if its label was vegetation.
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        cons = np.nanpercentile(np.where(good, Y, np.nan), 63, axis=0)
    cons = np.nan_to_num(cons, nan=BH * 0.75)
    Q -= np.minimum(1.0, np.maximum(0, cons - Y - 8) * 0.018) * good
    # Closed-chain Viterbi with a colour and outline cost at each seam. Holding ownership has zero seam cost.
    cost = -Q
    cols = np.array(
        [im[np.clip(y // 4 + 2, 0, BH // 4 - 1), np.arange(L)] for im, y in zip(ims, Y)]
    ).astype("f")
    srt = np.sort(Q, axis=0)
    start = int(np.argmax(srt[-1] - srt[-2]))
    ids = (start + np.arange(L)) % L
    j0 = int(np.argmax(Q[:, start]))
    hist = np.zeros((L, N), "i")
    dp = np.full(N, 1e8)
    dp[j0] = cost[j0, start]
    di = np.diag_indices(N)
    for b, x in enumerate(ids[1:], 1):
        switch = (
            17.0
            + 0.75 * abs(Y[:, x, None] - Y[None, :, x])
            + np.abs(cols[:, None, x] - cols[None, :, x]).mean(2) * 0.18
        )
        switch[di] = 0
        v = dp[:, None] + switch
        hist[b] = np.argmin(v, axis=0)
        dp = v[hist[b], np.arange(N)] + cost[:, x]
    lab = np.empty(L, "i")
    j = j0
    for b in range(L - 1, -1, -1):
        lab[ids[b]] = j
        j = hist[b, j]
    owner = np.repeat(lab, 4)
    out = np.zeros((BH, BW, 3), "B")
    alpha = np.zeros((BH, BW), "B")
    for j in np.unique(lab):
        k, c = entries[j]
        im, ma = chart(i, k, c, mats[j], True)
        on = owner == j
        out[:, on] = im[:, on]
        alpha[:, on] = ma[:, on]
    assert cv.imwrite(str(LOCK / f"{i:04}.jpg"), out, [cv.IMWRITE_JPEG_QUALITY, 95])
    assert cv.imwrite(str(LOCK / f"{i:04}.png"), alpha, [cv.IMWRITE_PNG_COMPRESSION, 4])
    stats = dict(
        i=i,
        owners=int(len(np.unique(lab))),
        seams=int(np.sum(lab != np.roll(lab, 1))),
        unseenCols=int(np.sum(Q.max(0) < -5) * 4),
        rotation=fits,
    )
    dump(path, stats)
    return stats


def bake():
    groups = stations()
    order = list(
        dict.fromkeys(
            groups[i]
            for i in sorted(groups, key=lambda k: priority(rows()[k]["front"]["t"]))
        )
    )
    with ThreadPoolExecutor(3) as p:
        r = list(p.map(solve, order))
    dump(W / "stations.json", r)
    print(
        "coherent stations",
        len(r),
        "median seams",
        np.median([v["seams"] for v in r]),
        flush=True,
    )


if __name__ == "__main__":
    if sys.argv[-1] == "pilot":
        for i in {stations()[k] for k in [65, 67]}:
            print(solve(i), flush=True)
    else:
        bake()
