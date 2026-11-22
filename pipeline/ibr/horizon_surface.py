"""Rasterize fixed, measured photo surfaces into the existing v1 cylinder representation. The renderer still consumes only JPEG+PNG. No metric depth is required to use the registered fallback."""

from .horizon_lock import *
from scipy.ndimage import distance_transform_edt
import ctypes as ct
from pipeline.livemap.geometry import rows as row_transforms

SURF = W / "surface-r4"
SURF.mkdir(exist_ok=True)


@lru_cache(None)
def native():
    lib = ct.CDLL(str(W / "horizon_surface.dylib"))
    P, I = ct.c_void_p, ct.c_int
    lib.paint.argtypes = [P, I, I, P, P, I, I, I, I, ct.c_float, P, P, P, I, I]
    return lib


@lru_cache(None)
def donors():
    s = np.array([x["front"]["s"] for x in rows()])
    idx = [0]
    for k in range(1, len(s)):
        if s[k] - s[idx[-1]] > 12.5:
            idx.append(k)
    return np.array(idx), s


@lru_cache(72)
def mesh(k, c):
    e = rows()[k][c]
    n = e["n"]
    file = RS / "depth" / c / f"{n:06}.npy"
    if not file.exists():
        return None
    z = np.load(file).astype("f")
    sk, rd, dy = masks(c, n)
    ray, valid = lattice(c)
    range_ = rng(c, e)
    good = (range_ > 44) & (range_ < 290) & (~(sk | rd | dy)) & valid
    if good.sum() < 60:
        return None
    # Nearest-depth padding at the real photographic edge supports clipped boundary triangles; RGB and alpha
    # still come from their true subpixel source coordinates, not the dilated depth footprint.
    dist, nearest = distance_transform_edt(~good, return_indices=True)
    step = 3
    u = np.arange(0, B, step)
    v = np.arange(0, H, step)
    V, U = np.meshgrid(v, u, indexing="ij")
    ix = nearest[:, V, U]
    d = z[ix[0], ix[1]]
    r = ray[V, U]
    local = r * d[:, :, None] / np.maximum(0.06, r[:, :, 2:3])
    cw = row_transforms("live", c, e["t"])[0][v]
    world = np.einsum("yij,yxj->yxi", cw[:, :3, :3], local) + cw[:, :3, 3][:, None]
    rr = np.where(dist[V, U] <= 4.8, range_[ix[0], ix[1]], -1)
    uv = np.stack([U, V], -1)
    ma = (~(sk | rd | dy) & valid & (dist < 3.1)).astype("f")
    raw = cv.imread(str(RS / "data/drive/frames" / c / f"{n:06}.jpg"))
    im = photo("live").apply(c, n, raw[:, :, ::-1])[:, :, ::-1]
    im = np.ascontiguousarray(np.uint8(np.clip(im * gains()[f"{c}:{n}"] + 0.5, 0, 255)))
    return world, uv, rr, ma, im


@lru_cache(12)
def fallback(i):
    im = cv.imread(str(LOCK / f"{i:04}.jpg"))
    al = cv.imread(str(LOCK / f"{i:04}.png"), 0)
    assert im is not None and al is not None
    return im, al


def clean(m):
    n, label, stats, _ = cv.connectedComponentsWithStats((m > 120).astype("B"), 8)
    keep = (stats[:, cv.CC_STAT_AREA] > 75) & (
        stats[:, cv.CC_STAT_TOP] + stats[:, cv.CC_STAT_HEIGHT] > BH * 0.68
    )
    keep[0] = False
    return np.where(keep[label], m, 0).astype("B")


def build_one(i):
    mark = SURF / f"{i:04}.json"
    if mark.is_file():
        return load(mark)
    k = stations()[i]
    org = origins()[i]
    off = org - origins()[k]
    b = GRID[:, :, :2] @ off[:2]
    disc = b * b + radii()[k] ** 2 - off[:2] @ off[:2]
    uv = cv.resize(
        wrapuv(off + GRID * (-b + np.sqrt(np.maximum(0, disc)))[:, :, None]), (BW, BH)
    )
    uv[:, :, 1] = uv[:, :, 1].clip(0, BH - 1)
    fic, fia = fallback(k)
    bg = remap(fic, uv)
    ba = clean(remap(fia, uv))
    rgb = np.zeros_like(bg)
    a = np.zeros_like(ba)
    depth = np.full((BH, BW), 1e6, "f")
    ix, ss = donors()
    near = ix[abs(ss[ix] - ss[i]) <= 49]
    used = 0
    for j in near:
        for c in CAMS:
            obj = mesh(int(j), c)
            if obj is None:
                continue
            xyz, puv, rr, ma, im = obj
            rel = xyz - org
            rad = np.hypot(rel[:, :, 0], rel[:, :, 1])
            az = (0.5 + np.arctan2(rel[:, :, 1], rel[:, :, 0]) / (2 * np.pi)) * BW - 0.5
            el = (BACKEL[1] - np.rad2deg(np.arctan2(rel[:, :, 2], rad))) / DEGPX - 0.5
            vert = np.ascontiguousarray(np.dstack([az, el, rad, puv, rr]), "f")
            native().paint(
                vert.ctypes.data,
                *rr.shape,
                im.ctypes.data,
                ma.ctypes.data,
                *ma.shape,
                *im.shape[:2],
                0 if c == "front" else 3,
                depth.ctypes.data,
                rgb.ctypes.data,
                a.ctypes.data,
                BH,
                BW,
            )
            used += 1
    if used:
        # A measured outline owns its whole column through the ground. Only use the fallback where observations
        # cannot establish an outline. Suppress isolated fragments, retaining branching edges attached to land.
        tight = cv.morphologyEx(a, cv.MORPH_CLOSE, np.ones((3, 3), "B"))
        cnt, labels, cs, _ = cv.connectedComponentsWithStats((tight > 0).astype("B"), 8)
        keep = (cs[:, 4] > 85) & (cs[:, 1] + cs[:, 3] > BH * 0.66)
        keep[0] = False
        occupied = keep[labels]
        height = occupied.sum(0)
        cover = height > 7
        cover = median_filter(cover.astype("B"), 13, mode="wrap") > 0
        occ = occupied & a.astype(bool)
        if occ.any():
            dist, nn = distance_transform_edt(~occ, return_indices=True)
            nearby = rgb[nn[0], nn[1]]
            top = np.argmax(occupied, axis=0)
            last = np.where(occupied, YY, -1).max(0)
            body = (YY >= top) & cover[None]
            ground = np.array([105, 106, 97], "f")
            decay = np.exp(-dist[:, :, None] / 10.0)
            color = ground + (nearby.astype("f") - ground) * decay
            color[occ] = rgb[occ]
            floor = (YY > last) & body
            solid = (occupied | floor) & body
            sd = cv.distanceTransform(
                solid.astype("B"), cv.DIST_L2, 3
            ) - cv.distanceTransform((~solid).astype("B"), cv.DIST_L2, 3)
            ma = np.uint8(np.clip(0.5 + sd / 2.8, 0, 1) * 255 + 0.5)
            bg[:, cover] = np.uint8(color[:, cover] + 0.5)
            ba[:, cover] = ma[:, cover]
    else:
        cover = np.zeros(BW, bool)
    assert cv.imwrite(str(SURF / f"{i:04}.jpg"), bg, [cv.IMWRITE_JPEG_QUALITY, 95])
    assert cv.imwrite(str(SURF / f"{i:04}.png"), ba, [cv.IMWRITE_PNG_COMPRESSION, 4])
    rec = dict(
        i=i,
        meshSources=used,
        measuredColumns=int(cover.sum()),
        measuredPixels=int((a > 0).sum()),
    )
    dump(mark, rec)
    return rec


def build(ids):
    with ThreadPoolExecutor(3) as p:
        info = list(p.map(build_one, ids))
    return info


if __name__ == "__main__":
    a = anchors()
    ids = sorted(a, key=lambda i: priority(rows()[i]["front"]["t"]))
    print("fixed geometry", len(ids), flush=True)
    q = build(ids)
    dump(W / "surfaces.json", q)
    print("depth-backed knots", sum(e["meshSources"] > 0 for e in q), flush=True)
