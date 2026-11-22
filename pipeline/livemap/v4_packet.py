"""Sample the full observed cone. Erosion only removes uncertain seeds, never sets a range gate."""

from .v4_inputs import *
from .prepare import ground, unit, INF


@lru_cache(None)
def coefficients(c):
    pol = json.loads((W / "live" / (c + "-causal-policy.json")).read_text())
    ls = sorted(
        (ready("live", c, e), pol[str(e["n"])]["a"], pol[str(e["n"])]["b"])
        for e in frames()[c]
        if str(e["n"]) in pol
    )
    return np.array(ls)


def observe(c, e, root, now):
    n = e["n"]
    key = f"{n:06d}"
    C = rig[c]
    z = np.load(root / "depth" / c / (key + ".npy")).astype("f")
    ray, ok = lattice(c)
    raw = z / np.maximum(ray[:, :, 2], 0.01)
    coef = coefficients(c)
    ii = coef[:, 0].searchsorted(now, "right") - 1
    a, b = coef[ii, 1:] if ii >= 0 else [1.0, 0.0]
    d = raw / np.maximum(0.60, a + b * np.minimum(raw, 40))
    sky, inst, road = [
        cv.imread(str(root / "masks" / k / c / (key + ".png")), cv.IMREAD_UNCHANGED)
        for k in ["sky", "dyn", "road"]
    ]
    assert inst.dtype == np.uint16 and z.shape == (H, B)
    base = (z > 0) & ok & (sky == 0)
    # The producer already erodes its depth silhouettes. A missing sample is
    # not a second, zero-range surface. Only erode across two measured ranges.
    hi = cv.dilate(np.where(base, d, 0).astype("f"), None)
    lo = cv.erode(np.where(base, d, INF).astype("f"), None)
    edge = base & ((hi - lo) > 0.04 * np.maximum(d, 0.1))
    safe = base & (cv.dilate(edge.astype("B"), np.ones((5, 5), "B")) == 0)
    cw, nc, h = rows("live", c, e["t"])
    orig = cv.imread(str(RS / "data/drive/frames" / c / (key + ".jpg")))[:, :, ::-1]
    im = photo("live").apply(c, n, cv.resize(orig, (B, H), interpolation=cv.INTER_AREA))
    # One representative per angular cell, range octave and instance; small valid
    # islands survive when the regular centre falls on a rejected silhouette.
    iy, ix = np.where(safe)
    val = d[iy, ix]
    step = np.select([val < 15, val < 42, val < 85], [3, 4, 4], default=5)
    octave = np.floor(np.log2(np.maximum(val, 0.1)) * 2).astype("i8")
    cells = ((iy // step * B + ix // step) * 12 + step) * 65536 + inst[iy, ix].astype(
        "i8"
    )
    cells = cells * 80 + octave
    err = ((iy % step + 0.5) / step - 0.5) ** 2 + ((ix % step + 0.5) / step - 0.5) ** 2
    order = np.lexsort((err, cells))
    u = (
        order[np.r_[True, cells[order][1:] != cells[order][:-1]]]
        if len(order)
        else order
    )
    y, x, ss = iy[u], ix[u], step[u]
    ids = inst[y, x]
    r = d[y, x]
    rr = ray[y, x]
    v = rr * r[:, None]
    objroad = (road[y, x] > 0) & (ids == 0)
    den = np.sum(nc[y] * rr, axis=1)
    inter = -h[y] / np.minimum(den, -0.0001)
    snap = objroad & (inter > 1) & (inter < 35) & (den < -0.04)
    r = np.where(snap, inter, r)
    v = rr * r[:, None]
    world = cw[y, :3, 3] + np.einsum("nij,nj->ni", cw[y, :3, :3], v)
    claims, gz = ground("live", world)
    low = (abs(world[:, 2] - gz) < 0.11) & (ids == 0)
    owned = (claims <= now) & (objroad | low)
    expiry = np.where(objroad, claims, INF)
    xp = np.minimum(B - 1, x + 1)
    xm = np.maximum(0, x - 1)
    yp = np.minimum(H - 1, y + 1)
    ym = np.maximum(0, y - 1)

    def neighbour(yy, xx):
        return ray[yy, xx] * np.where(base[yy, xx], d[yy, xx], d[y, x])[:, None]

    dx = neighbour(y, xp) - neighbour(y, xm)
    dy = neighbour(yp, x) - neighbour(ym, x)
    normal = unit(np.cross(dx, dy))
    normal = np.where(np.sum(normal * rr, 1)[:, None] > 0, -normal, normal)
    cos = -np.sum(normal * rr, 1)
    normal = np.where((cos < 0.18)[:, None], -rr, normal)
    normal[snap] = nc[y[snap]]
    cos = np.maximum(0.18, -np.sum(normal * rr, 1))

    def tangent(dr):
        t = dr + rr * (np.sum(normal * dr, 1) / cos)[:, None]
        return (
            unit(t)
            * (
                np.minimum(np.linalg.norm(t, axis=1), np.linalg.norm(dr, axis=1) * 2.25)
                * ss
                * r
            )[:, None]
        )

    tx = tangent((ray[y, xp] - ray[y, xm]) * 0.5)
    ty = tangent((ray[yp, x] - ray[ym, x]) * 0.5)
    aa = np.sum(tx * tx, 1)
    bb = np.sum(tx * ty, 1)
    cc = np.sum(ty * ty, 1)
    rot = 0.5 * np.arctan2(2 * bb, aa - cc)
    uvec = unit(tx * np.cos(rot)[:, None] + ty * np.sin(rot)[:, None])
    l1 = np.sqrt(np.maximum(0.0001, (aa + cc + np.hypot(aa - cc, 2 * bb)) * 0.5))
    l2 = np.sqrt(np.maximum(0.0001, (aa + cc - np.hypot(aa - cc, 2 * bb)) * 0.5))
    pix = im[y, x].astype("f")
    green = (pix[:, 1] > 1.06 * pix[:, 0]) & (pix[:, 1] > 1.04 * pix[:, 2])
    rough = (
        abs(d[y, xp] + d[y, xm] - 2 * r) + abs(d[yp, x] + d[ym, x] - 2 * r)
    ) > 0.008 * r
    thick = (green | rough) & ~objroad & (ids == 0)
    sig = np.column_stack(
        (0.88 * l1, 0.88 * l2, np.maximum(0.009, l2 * np.where(thick, 0.52, 0.12)))
    )
    nw = np.einsum("nij,nj->ni", cw[y, :3, :3], normal)
    uw = np.einsum("nij,nj->ni", cw[y, :3, :3], uvec)
    obs = np.column_stack(
        (world, nw, uw, sig, pix, x + 0.5, y + 0.5, r, ss, ids, expiry)
    )
    valid = ~owned & (np.linalg.norm(uw, axis=1) > 0.99) & np.isfinite(obs).all(1)
    meta = []
    jp = root / "masks/dyn" / c / (key + ".json")
    classes = (
        {int(v["id"]): v for v in json.loads(jp.read_text())} if jp.exists() else {}
    )
    for ident in np.unique(inst[inst > 0]):
        my, mx = np.where(base & (inst == ident))
        cent = None
        med = None
        if len(mx):
            pt = cw[my, :3, 3] + np.einsum(
                "nij,nj->ni", cw[my, :3, :3], ray[my, mx] * d[my, mx, None]
            )
            cent = np.median(pt, axis=0).round(4).tolist()
            med = float(np.median(d[my, mx]))
        meta.append(
            dict(
                id=int(ident),
                key=f"{c}:{ident}",
                label=classes.get(int(ident), {}).get("class"),
                bbox=classes.get(int(ident), {}).get("bbox"),
                pixels=int((inst == ident).sum()),
                depth_pixels=len(mx),
                seeds=int(np.sum(valid & (ids == ident))),
                centroid_world=cent,
                range_m=med,
            )
        )
    k = np.r_[
        C.params[:4] * [B / C.width, H / C.height, B / C.width, H / C.height],
        C.params[4:],
    ]
    return (
        np.ascontiguousarray(obs[valid], "f"),
        np.ascontiguousarray(cw[np.linspace(0, H - 1, 33).round().astype(int)], "d"),
        np.ascontiguousarray(k, "d"),
        np.ascontiguousarray(
            np.where(
                base, d, np.where(ok & (cv.erode(sky, np.ones((5, 5), "B")) > 0), -1, 0)
            ),
            "f",
        ),
        np.ascontiguousarray(inst, "H"),
        np.ascontiguousarray(im, "B"),
        meta,
    )
