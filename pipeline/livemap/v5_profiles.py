"""Causal grounded profiles: only a connected vertical sequence with a measured foot can be born."""

from .v4_packet import *
from scipy.ndimage import binary_propagation

V5 = RS / "maps/live-v5"
OUT = RS / "scene/live-v5"
V5.mkdir(parents=True, exist_ok=True)
KINDS = ["verge", "wall", "foliage", "pole", "far", "instance"]


def surface(t, world, release):
    T = traj("live").T_world_ego(t)
    n, h = traj("live").ground_plane(t)
    nw = T[:3, :3] @ n
    # Local pitch extrapolation at 100 m makes another floating sheet. Use the
    # measured v2 height where it had already been released, a bounded tangent elsewhere.
    plane = (
        T[2, 3]
        - h / max(0.8, nw[2])
        - np.clip((world[:, :2] - T[:2, 3]) @ nw[:2] / max(0.8, nw[2]), -0.45, 0.45)
    )
    born, z = ground("live", world)
    return np.where((born <= release) & np.isfinite(z), z, plane)


def packet(c, e, root, now):
    key = f"{e['n']:06d}"
    C = rig[c]
    ray, ok = lattice(c)
    z = np.load(root / "depth" / c / (key + ".npy")).astype("f")
    raw = z / np.maximum(0.01, ray[:, :, 2])
    co = coefficients(c)
    j = co[:, 0].searchsorted(now, "right") - 1
    a, b = co[j, 1:] if j >= 0 else (1.0, 0.0)
    dep = (raw / np.maximum(0.6, a + b * np.minimum(raw, 40))).astype("f")
    sky, road, ins = [
        cv.imread(str(root / "masks" / k / c / (key + ".png")), cv.IMREAD_UNCHANGED)
        for k in ["sky", "road", "dyn"]
    ]
    base = ok & (z > 0) & (sky == 0)
    D = cv.distanceTransform(((sky == 0) & ok).astype("B"), cv.DIST_L2, 5)
    lo = cv.erode(np.where(base, dep, INF).astype("f"), None)
    hi = cv.dilate(np.where(base, dep, 0).astype("f"), None)
    ed = base & ((hi - lo) > 0.07 * np.maximum(dep, 0.1))
    clean = base & (cv.dilate(ed.astype("B"), np.ones((3, 3), "B")) == 0)
    im = photo("live").apply(
        c,
        e["n"],
        cv.resize(
            cv.imread(str(RS / "data/drive/frames" / c / (key + ".jpg")))[:, :, ::-1],
            (B, H),
            interpolation=cv.INTER_AREA,
        ),
    )
    cw, _, _ = rows("live", c, e["t"])
    origin = cw[H // 2, :3, 3]
    K = np.r_[
        C.params[:4] * [B / C.width, H / C.height, B / C.width, H / C.height],
        C.params[4:],
    ]
    extra = (
        np.ascontiguousarray(cw[np.linspace(0, H - 1, 33).round().astype(int)], "d"),
        np.ascontiguousarray(K, "d"),
        np.ascontiguousarray(
            np.where(base, dep, np.where((D == 0) & ok & (sky > 0), -1, 0)), "f"
        ),
        np.ascontiguousarray(ins, "H"),
        np.ascontiguousarray(im, "B"),
    )
    valid = clean & ((road == 0) | (ins > 0)) & (D > 2.5)
    yy, xx = np.where(valid[::2, ::2])
    yy *= 2
    xx *= 2
    world = cw[yy, :3, 3] + np.einsum(
        "nij,nj->ni", cw[yy, :3, :3], ray[yy, xx] * dep[yy, xx, None]
    )
    v = world - origin
    r = np.hypot(v[:, 0], v[:, 1])
    floor = surface(e["t"], world, now)
    h = world[:, 2] - floor
    ids = ins[yy, xx]
    pad = D[yy, xx]
    keep = (
        (r > 1.5)
        & (r < 145)
        & (h > -0.28)
        & (h < 15)
        & (pad > np.where(h > 2, np.minimum(18, 5 + 0.09 * r), 3.0))
    )
    yy, xx, world, r, h, ids, pad = [v[keep] for v in (yy, xx, world, r, h, ids, pad)]
    empty = lambda: (
        np.empty((0, 21), "f"),
        np.empty((0, 16), "f"),
        np.empty(0, "i"),
        np.empty(0, "B"),
        extra,
    )
    if not len(r):
        return empty()
    # Weight colour inward, excluding depth jumps and sky. Fall back to an
    # eroded interior measurement only, not the colour of the outside silhouette.
    weight = np.clip((D - 4) / 7, 0, 1) * clean
    weight = weight.astype("f")
    blur = lambda x: cv.GaussianBlur(x, (9, 9), 2.0)
    den = blur(weight)
    col = blur(im.astype("f") * weight[:, :, None]) / np.maximum(den[:, :, None], 0.015)
    validcol = den[yy, xx] > 0.15
    yy, xx, world, r, h, ids, pad = [
        v[validcol] for v in (yy, xx, world, r, h, ids, pad)
    ]
    rgb = col[yy, xx]
    if not len(r):
        return empty()
    # Pool adjacent azimuths around an observed foot. Unlike independent pencils,
    # an overhanging crown can connect *sideways* to a trunk. Its patch and trunk
    # share a playback lifetime; no floating crown is left when a base retires.
    az = np.arctan2(world[:, 1] - origin[1], world[:, 0] - origin[0])
    na = 504
    width = 8
    arc = 2 * np.pi / na
    ia = np.clip(((az + np.pi) / arc).astype("i"), 0, na - 1)
    sector = ia // width
    ir = np.floor(np.log1p(r / 3) / 0.15).astype("i")
    rings = 64
    sec = na // width
    key = (ids.astype("i8") * rings + ir) * sec + sector
    # Absolute altitude close to the measured road/plane is the root witness.
    foot = h < np.minimum(0.68, 0.20 + 0.009 * r)
    keys, g = np.unique(key, return_inverse=True)
    cnt = np.bincount(g, weights=foot, minlength=len(keys))
    sums = np.bincount(g, weights=r * foot, minlength=len(keys))
    roots = keys[cnt >= 4]
    rr = sums[cnt >= 4] / cnt[cnt >= 4]
    if not len(roots):
        return empty()
    # Reconcile the noisy range of upper pixels to a *measured* nearby root, never
    # to an unrelated silhouette or to a fabricated ground intersection in the sky.
    assign = np.full(len(r), -1, "i")
    score = np.full(len(r), 1.0)
    typ = ids.astype("i8") * rings
    for shift in [0, -1, 1]:
        kk = (typ + ir + shift) * sec + sector
        j = np.searchsorted(roots, kk).clip(0, len(roots) - 1)
        cost = abs(r - rr[j]) / np.maximum(2.0, rr[j])
        hit = (roots[j] == kk) & (cost < score) & (cost < np.where(foot, 0.13, 0.245))
        assign[hit] = j[hit]
        score[hit] = cost[hit]
    take = assign >= 0
    if not np.any(take):
        return empty()
    g = assign[take]
    x = ia[take] % width
    dz = np.clip(0.0125 * rr, 0.12, 0.84)
    height = h[take] * rr[g] / r[take]
    nb = 96
    b = np.floor(np.maximum(0, height) / dz[g]).astype("i").clip(0, nb - 1)
    q = (g * width + x) * nb + b
    size = len(roots) * width * nb
    shape = (len(roots), width, nb)
    count = np.bincount(q, minlength=size).reshape(shape)
    seen = count > 0
    col = np.stack(
        [
            np.bincount(q, weights=rgb[take, k], minlength=size).reshape(shape)
            / np.maximum(1, count)
            for k in range(3)
        ],
        3,
    ).astype("f")

    # Fill isolated grid-sized sampling gaps between measured cells, not sky or
    # large holes. Grow only connected components touching the measured base.
    def shift(v, dx, dy):
        a = np.roll(v, (dx, dy), (1, 2))
        if dx > 0:
            a[:, :dx] = False
        if dx < 0:
            a[:, dx:] = False
        if dy > 0:
            a[:, :, :dy] = False
        if dy < 0:
            a[:, :, dy:] = False
        return a

    eligible = (
        seen
        | (shift(seen, -1, 0) & shift(seen, 1, 0))
        | (shift(seen, 0, -1) & shift(seen, 0, 1))
    )
    height_cap = np.where(rr >= 43, 9.0, 12.0)
    eligible &= (
        np.arange(nb)[None, None, :] * dz[:, None, None] < height_cap[:, None, None]
    )

    seed = np.zeros_like(seen)
    seed[:, :, 0] = eligible[:, :, 0]
    seed[:, :, 1] = eligible[:, :, 1] & (dz[:, None] < 0.5)
    structure = np.zeros((3, 3, 3), "B")
    structure[1] = 1
    reach = binary_propagation(seed, structure=structure, mask=eligible)
    g0, x0, b0 = np.where(reach & seen)
    top = np.max(np.where(reach, np.arange(nb), 0), axis=(1, 2))
    ht = (top + 0.5) * dz
    instance = (roots // (sec * rings)).astype("i")
    coverage = (reach[:, :, 2:].any(2)).sum(1)
    pole = (coverage <= 2) & (ht > 1.2) & (instance == 0)
    present = (count * reach).sum((1, 2))
    ratio = lambda v: (
        np.bincount(g, weights=v, minlength=len(roots))
        / np.maximum(1, np.bincount(g, minlength=len(roots)))
    )
    green = ratio(
        (rgb[take, 1] > 1.06 * rgb[take, 0]) & (rgb[take, 1] > 1.01 * rgb[take, 2])
    )
    rough = ratio(abs(im[yy[take], xx[take], 1] - rgb[take, 1]) > 5.5)
    kind = np.select(
        [instance > 0, rr > 43, pole, ht < 0.52, (green > 0.19) | (rough > 0.21)],
        [5, 4, 3, 0, 2],
        default=1,
    ).astype("B")
    ok = (present >= 5) & ~(pole & (rr > 32))
    counter = np.sum(
        reach, axis=1
    )  # Unsupported high thin crossarms have no crown width or physical base.
    dense = counter >= 3
    high_limit = np.max(np.where(dense, np.arange(nb), 0), axis=1) + 2
    keep = (
        ok[g0]
        & ((b0 <= high_limit[g0]) | (kind[g0] == 3) | (b0 * dz[g0] < 1.1))
        & ((kind[g0] != 0) | (b0 == 0))
    )
    g0, x0, b0 = g0[keep], x0[keep], b0[keep]
    if not len(g0):
        return empty()
    # Put the bottom on the surface even if the first measured height cell was
    # the second one. Do not independently lift the top or extend it over sky.
    baselevel = np.min(np.where(reach, np.arange(nb), nb), axis=(1, 2))
    theta = ((roots[g0] % sec) * width + x0 + 0.5) * arc - np.pi
    heading = np.c_[np.cos(theta), np.sin(theta)]
    p = origin[:2] + heading * rr[g0, None]
    xyz = np.column_stack((p, np.zeros(len(p))))
    gz = surface(e["t"], xyz, now)
    xyz[:, 2] = gz + (b0 - baselevel[g0] + 0.46) * dz[g0]
    normal = np.c_[-heading, np.zeros(len(p))]
    u = np.c_[-heading[:, 1], heading[:, 0], np.zeros(len(p))]
    km = kind[g0]
    sx = np.clip(0.74 * arc * rr[g0], 0.055, 0.84)
    sy = 0.70 * dz[g0]
    sn = np.maximum(0.012, 0.15 * np.minimum(sx, sy))
    fol = km == 2
    ball = np.clip(np.sqrt(sx * sy), 0.068, 0.42)
    sx[fol] = ball[fol]
    sy[fol] = ball[fol] * 1.06
    sn[fol] = ball[fol] * 0.86
    narrow = km == 3
    sx[narrow] = np.clip(sx[narrow] * 0.7, 0.04, 0.13)
    sn[narrow] = sx[narrow] * 0.8
    flat = km == 0
    xyz[flat, 2] = gz[flat] + 0.04
    normal[flat] = [0, 0, 1]
    sn[flat] = 0.04
    sy[flat] = np.minimum(0.34, sx[flat] * 1.33)
    colour = col[g0, x0, b0].copy()
    # Suppress impossible cyan/magenta leaf edges in *viewer-balanced* space.
    # Keep the measured luma, texture and valid hue instead of painting a tree.
    balanced = colour * wb()
    lum = balanced @ np.array([0.2126, 0.7152, 0.0722])
    fringe = fol & (
        (balanced[:, 2] > 1.12 * balanced[:, 1])
        | (balanced[:, 0] > 1.28 * balanced[:, 1])
    )
    colour[fringe] = (
        0.23 * balanced[fringe] + 0.77 * lum[fringe, None] * [0.94, 1.025, 0.90]
    ) / wb()
    alpha = np.select(
        [km == 4, fol, narrow, flat],
        [np.clip(0.70 - 0.004 * (rr[g0] - 43), 0.25, 0.70), 0.73, 0.6, 0.46],
        default=0.81,
    )
    m = np.column_stack(
        (xyz, normal, u, sx, sy, sn, np.clip(colour, 0, 255), alpha)
    ).astype("f")
    chosen = np.unique(g0)
    ki = kind[
        chosen
    ]  # fusion representative chosen from a real occupied column and height
    pos = np.searchsorted(g0, chosen)
    end = np.searchsorted(g0, chosen, "right")
    mid = (pos + end) // 2
    rep = xyz[mid]
    uv, pok, vec, _ = projected(c, rep, cw)
    uv *= np.array([B / C.width, H / C.height])
    sz = np.maximum(
        0.10, 0.40 * arc * rr[chosen] * np.maximum(2, coverage[chosen])
    )  # axial confirmation footprint is bounded so close patches keep detail
    obs = np.column_stack(
        (
            rep,
            normal[mid],
            u[mid],
            np.minimum(1.5, sz),
            np.minimum(1.1, np.maximum(dz[chosen], 0.24 * ht[chosen])),
            np.full(len(mid), 0.06),
            colour[mid],
            uv,
            np.linalg.norm(vec, axis=1),
            np.full(len(mid), 4),
            instance[chosen],
            np.full(len(mid), INF),
        )
    )
    good = pok & (uv[:, 0] >= 0) & (uv[:, 1] >= 0) & (uv[:, 0] < B) & (uv[:, 1] < H)
    owner = np.searchsorted(chosen, g0)
    mg = good[owner]
    return (
        np.ascontiguousarray(obs[good], "f"),
        m[mg],
        (np.cumsum(good) - 1)[owner[mg]].astype("i"),
        ki[good],
        extra,
    )


@lru_cache(None)
def wb():
    d = json.load(open(RS / "scene/live-v4/rsb/index.json"))
    return np.asarray(d["grade"]["gains"])
