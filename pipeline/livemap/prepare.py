"""Physical ray range, row poses, tangent covariance and ground-layer ownership."""

from .geometry import *
import os
from pipeline.road.scene import reader, INF


@lru_cache(maxsize=110)
def ground_tile(mode, tx, ty):
    p = RS / "scene" / f"road-{mode}-v2/tiles" / f"{tx}_{ty}.ply"
    first = np.full(220**2, INF, "f")
    z = np.full(220**2, np.nan, "f")
    if p.exists():
        a = reader(p)
        ij = np.floor((a[:, :2].astype("d") - [50 * tx, 50 * ty]) * 4.4).astype(int)
        cell = ij[:, 1] * 220 + ij[:, 0]
        np.minimum.at(first, cell, a[:, 17])
        jj = a[:, 17] == first[cell]
        z[cell[jj]] = a[jj, 2]
    return first, z


def ground(mode, world):
    tiles = np.floor(world[:, :2] / 50).astype(int)
    cell = np.floor((world[:, :2] - tiles * 50) * 4.4).astype(int)
    cell = cell[:, 0] + 220 * cell[:, 1]
    birth = np.full(len(cell), INF, "f")
    zz = np.full(len(cell), np.nan, "f")
    key = tiles[:, 0] * 10000 + tiles[:, 1]
    for j in np.unique(key):
        ix = key == j
        g, h = ground_tile(mode, *tiles[np.flatnonzero(ix)[0]])
        birth[ix] = g[cell[ix]]
        zz[ix] = h[cell[ix]]
    if mode == "live" and "LIVEMAP_CUTOFF" in os.environ:
        later = birth > float(os.environ["LIVEMAP_CUTOFF"])
        birth[later] = INF
        zz[later] = np.nan
    return birth, zz


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)


def packet(mode, c, e, policy):
    C = rig[c]
    rays, ok = lattice(c)
    dist, safe, ro, _ = maps(mode, c, e["n"])
    cw, nc, h = rows(mode, c, e["t"])
    a, b = policy["a"], policy["b"]
    dist = dist / np.maximum(0.5, a + b * dist)
    cap = policy["max_range"]
    safe &= (
        (rays[:, :, 2] > np.cos(np.deg2rad(policy["max_angle"])))
        & (dist >= 1.4)
        & (dist <= cap)
    )
    allowed = np.zeros_like(safe)
    for lo, hi in policy["bands_admitted"]:
        allowed |= (dist >= lo) & (dist < hi)
    safe &= allowed
    mind = cv.erode(np.where(safe, dist, 0).astype("f"), np.ones((3, 3), "B"))
    orig = cv.imread(str(RS / "data/drive/frames" / c / f"{e['n']:06d}.jpg"))
    assert orig is not None
    small = cv.resize(orig, (B, H), interpolation=cv.INTER_AREA)
    lum = cv.cvtColor(small, cv.COLOR_BGR2GRAY)
    edges = cv.dilate(
        (
            (cv.dilate(lum, None).astype(int) - cv.erode(lum, None).astype(int)) > 27
        ).astype("B"),
        np.ones((5, 5), "B"),
    )
    mind[edges != 0] = 0
    y, x = np.mgrid[4 : H - 4 : 2, 4 : B - 4 : 2]
    y = y.ravel()
    x = x.ravel()
    d = dist[y, x]
    rr = rays[y, x]
    ce = rr * d[:, None]
    ego = ce @ C.T_ego_cam[:3, :3].T + C.T_ego_cam[:3, 3]
    wanted = np.clip(0.068 + 0.004 * d + 0.005 * abs(ego[:, 1]), 0.07, 0.36)
    fx = C.params[0] * B / C.width
    step = np.clip(
        2 ** np.floor(np.log2(np.maximum(2, wanted * fx / np.maximum(d, 1.0)))), 2, 32
    ).astype(int)
    ix = safe[y, x] & (x % step == 0) & (y % step == 0)
    x, y, step, rr, d, ce = [v[ix] for v in (x, y, step, rr, d, ce)]
    dx = rays[y, x + 2] * dist[y, x + 2, None] - rays[y, x - 2] * dist[y, x - 2, None]
    dy = rays[y + 2, x] * dist[y + 2, x, None] - rays[y - 2, x] * dist[y - 2, x, None]
    n = unit(np.cross(dx, dy))
    n = np.where((n * rr).sum(1)[:, None] > 0, -n, n)
    cos = np.maximum(0.1, -(rr * n).sum(1))
    pixx = (rays[y, x + 1] - rays[y, x - 1]) * 0.5
    pixy = (rays[y + 1, x] - rays[y - 1, x]) * 0.5

    def tangent(delta):
        raw = (
            (delta + rr * ((delta * n).sum(1) / cos)[:, None])
            * d[:, None]
            * step[:, None]
        )
        raw -= n * (raw * n).sum(1)[:, None]
        cap = 3.5 * np.linalg.norm(delta, axis=1) * d * step
        return unit(raw) * np.minimum(cap, np.linalg.norm(raw, axis=1))[:, None]

    tx = tangent(pixx)
    ty = tangent(pixy)
    aa = (tx * tx).sum(1)
    bb = (tx * ty).sum(1)
    cc = (ty * ty).sum(1)
    angle = 0.5 * np.arctan2(2 * bb, aa - cc)
    u = unit(tx * np.cos(angle)[:, None] + ty * np.sin(angle)[:, None])
    l1 = np.sqrt(np.maximum(0.0001, (aa + cc + np.hypot(aa - cc, 2 * bb)) * 0.5))
    l2 = np.sqrt(np.maximum(0.0001, (aa + cc - np.hypot(aa - cc, 2 * bb)) * 0.5))
    world = cw[y, :3, 3] + np.einsum("nij,nj->ni", cw[y, :3, :3], ce)
    height = (nc[y] * ce).sum(1) + h[y]
    birth, z = ground(mode, world)
    now = ready(mode, c, e)
    owned = birth <= now
    nw = np.einsum("nij,nj->ni", cw[y, :3, :3], n)
    height = np.where(owned, world[:, 2] - z, height)
    rgb = small[y, x, ::-1].astype(float)
    green = (rgb[:, 1] > rgb[:, 0] * 1.07) & (rgb[:, 1] > rgb[:, 2] * 1.06)
    rough = abs(dist[y + 3, x] + dist[y - 3, x] - 2 * d) + abs(
        dist[y, x + 3] + dist[y, x - 3] - 2 * d
    )
    foliage = (green & (height > 0.45)) | ((rough > 0.022 * d) & (height > 0.65))
    terrain = (ro[y, x] | ((height < 0.34) & (abs(nw[:, 2]) > 0.65))) & ~foliage
    good = (height > np.where(owned, 0.17, 0.07)) | terrain
    good &= ~(terrain & owned)
    good &= (
        (height > -0.55)
        & (np.linalg.norm(n, axis=1) > 0.95)
        & (cos > 0.16)
        & (l2 > 0.018)
        & np.isfinite(world).all(1)
    )
    good &= ~(terrain & ((x % (2 * step) != 0) | (y % (2 * step) != 0)))
    sigma = np.stack(
        (
            np.clip(l1 * 0.70, 0.032, 0.62),
            np.clip(l2 * 0.70, 0.028, 0.39),
            np.where(
                foliage, np.clip(0.47 * l2, 0.055, 0.24), np.maximum(0.008, 0.075 * l2)
            ),
        ),
        1,
    )
    sigma[terrain, :2] = np.minimum(0.68, sigma[terrain, :2] * 1.75)
    uv = np.stack(((x + 0.5) * C.width / B, (y + 0.5) * C.height / H), 1)
    pix = orig[np.floor(uv[:, 1]).astype(int), np.floor(uv[:, 0]).astype(int), ::-1]
    color = photo(mode).sample(c, e["n"], uv, pix)
    uw = np.einsum(
        "nij,nj->ni", cw[y, :3, :3], u
    )  # Do not place even the core of coarse terrain on owned v2 road. Claim
    # times are propagated to expiration, never to an earlier live birth.
    ids = np.flatnonzero(good & terrain)
    side = np.cross(nw[ids], uw[ids])
    ex = np.sqrt(
        (uw[ids, :2] * sigma[ids, 0:1]) ** 2 + (side[:, :2] * sigma[ids, 1:2]) ** 2
    )
    lo = np.floor((world[ids, :2] - ex) * 4.4).astype(int)
    hi = np.floor((world[ids, :2] + ex) * 4.4).astype(int)
    span = hi - lo
    # Enumerate the conservative XY bounding box of the oriented one-sigma core.
    for i in range(span[:, 0].max(initial=-1) + 1):
        for j in range(span[:, 1].max(initial=-1) + 1):
            take = (span[:, 0] >= i) & (span[:, 1] >= j)
            sel = ids[take]
            xy = (lo[take] + [i + 0.5, j + 0.5]) / 4.4
            g0, _ = ground(mode, np.column_stack((xy, world[sel, 2])))
            birth[sel] = np.minimum(birth[sel], g0)
    good[ids] &= birth[ids] > now
    conditional = np.zeros(len(d), bool)
    for lo, hi in policy.get("conditional", []):
        conditional |= (d >= lo) & (d < hi)
    data = np.column_stack(
        (
            world,
            nw,
            uw,
            sigma,
            color,
            x,
            y,
            d,
            cos,
            terrain,
            foliage,
            birth,
            conditional,
        )
    )
    q = np.array(
        [
            *C.params[:4] * [B / C.width, H / C.height, B / C.width, H / C.height],
            *C.params[4:],
        ],
        "d",
    )  # Same corrected colour code used for independent corroboration of flat walls.
    appearance = photo(mode).apply(c, e["n"], small[:, :, ::-1])
    return (
        np.ascontiguousarray(data[good], "f"),
        np.ascontiguousarray(cw[np.linspace(0, H - 1, 33).round().astype(int)], "d"),
        np.ascontiguousarray(np.where(safe, dist, 0), "f"),
        np.ascontiguousarray(mind, "f"),
        np.ascontiguousarray(appearance, "B"),
        q,
        now,
    )


@lru_cache(None)
def sweep(mode):
    tr = traj(mode)
    ix = np.where((tr.t >= 80) & (tr.t <= 140))[0][::3]
    T = tr.poses[ix]
    release = tr.t[ix] if mode == "stitched" else release_table()[1][ix]
    valid = release <= float(os.environ.get("LIVEMAP_CUTOFF", np.inf))
    T = T[valid]
    release = release[valid]
    tiles = np.floor(T[:, :2, 3] / 8).astype(int)
    key = tiles[:, 0] * 100000 + tiles[:, 1]
    lookup = {k: np.flatnonzero(key == k) for k in np.unique(key)}
    return T, release, lookup


def body_clearance(mode, xyz):
    # The occupied car hull is independent metric free-space evidence. Return
    # the first instrument release proving this location could not hold a surfel.
    T, release, lookup = sweep(mode)
    dead = np.full(len(xyz), INF, "f")
    cells = np.floor(xyz[:, :2] / 8).astype(int)
    keys = cells[:, 0] * 100000 + cells[:, 1]
    for k in np.unique(keys):
        ids = np.flatnonzero(keys == k)
        p = xyz[ids]
        for off in [x * 100000 + y for x in (-1, 0, 1) for y in (-1, 0, 1)]:
            for j in lookup.get(k + off, []):
                loc = (p - T[j, :3, 3]) @ T[j, :3, :3]
                hit = (
                    (abs(loc[:, 1]) < 0.98)
                    & (loc[:, 0] > -0.86)
                    & (loc[:, 0] < 3.64)
                    & (loc[:, 2] > 0.19)
                    & (loc[:, 2] < 1.6)
                )
                dead[ids[hit]] = np.minimum(dead[ids[hit]], release[j])
    return dead
