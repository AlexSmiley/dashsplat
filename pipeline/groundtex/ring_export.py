"""Compose v5, independently supported outer photos and low-frequency class continuation; publish matching lossless provenance."""

from .outer import *
from .proof import pose
from .v5 import actual_guard
import io
from scipy.ndimage import distance_transform_edt

TERRAIN = os.getenv("GT_TERRAIN") == "1"
ROLL = {(-2, 3), (-2, 4), (-48, 16)}
v4obs = None


@lru_cache(64)
def ctrl(id):
    with np.load(W / "ctrl" / f"{id}.npz") as z:
        return (
            z["box"][:2].astype(float) * L + L / 2,
            z["d"].astype("f"),
            z["gain"].astype("f"),
        )


def rem(a, p, linear=True):
    return cv.remap(
        a,
        p.astype("f"),
        None,
        cv.INTER_LINEAR if linear else cv.INTER_NEAREST,
        borderMode=cv.BORDER_REPLICATE,
    )


def static_col():
    g = Grid(4)
    for e in read(OBJ / "map.json")["tiles"]:
        if e["layer"] != "ground":
            continue
        a = reader(OBJ / e["file"])
        v = a[(a[:, 5] > 0.90) & (a[:, 10] < np.log(0.50)) & (a[:, 11] < np.log(0.45))]
        rgb = (0.5 + 0.28209479 * v[:, 6:9]) * 255
        green = (rgb[:, 1] > rgb[:, 2] * 1.14) & (rgb[:, 1] > rgb[:, 0] * 1.015)
        v = v[green]
        rgb = rgb[green]
        key = np.floor(v[:, :2] / L).astype(int)
        ks = key // U
        enc = (ks[:, 0] + 3000) * 7000 + ks[:, 1] + 3000
        ix = np.argsort(enc)
        bounds = np.r_[0, np.flatnonzero(np.diff(enc[ix])) + 1, len(ix)]
        for l, h in zip(bounds[:-1], bounds[1:]):
            if l == h:
                continue
            ii = ix[l:h]
            k = tuple(ks[ii[0]])
            xy = key[ii] - np.array(k) * U
            idx = xy[:, 1] * U + xy[:, 0]
            s = (
                np.stack(
                    [
                        np.bincount(idx, weights=rgb[ii, c], minlength=U * U)
                        for c in range(3)
                    ]
                    + [np.bincount(idx, minlength=U * U)],
                    -1,
                )
                .reshape(U, U, 4)
                .astype("f")
            )
            if k in g.a:
                g.a[k] += s
            else:
                g.a[k] = s
    return g


def prepare():
    g = Grid(10)
    ids = Grid(1, "i4")
    obj = Grid(1)
    dist = Grid(1)
    grass = static_col()
    G = Grid(3)
    tone = Grid(3)
    ks = pathkeys()
    for k in ks:
        with np.load(
            W / ("terrain-map" if TERRAIN else "map") / f"{k[0]}_{k[1]}.npz"
        ) as d:
            i = d["id"]
            rgb = d["rgb"][:, :, ::-1].astype("f")
            prior = d["prior"].astype("f")
            alpha = prior[:, :, 3]
            col = np.where(
                (alpha > 0.78)[:, :, None],
                prior[:, :, :3][:, :, ::-1] / alpha[:, :, None].clip(0.01),
                rgb,
            )
            seen = (i > 0) | (alpha > 0.78)
            green = (
                (col[:, :, 1] > col[:, :, 2] * 1.10 + 1)
                & (col[:, :, 1] > col[:, :, 0] * 1.001)
                & (d["road"] < d["votes"] * 0.7)
            )
            luma = col.mean(2)
            mark = (
                (abs(luma - cv.GaussianBlur(luma, (0, 0), 2.5)) > 11)
                | (luma > 188)
                | (luma < 27)
            )
            if TERRAIN:
                green = np.where((i > 0) & (alpha < 0.78), d["label"] == 9, green)
            concrete = (~green) & (luma > 124) & (d["road"] < d["votes"] * 0.72)
            static = grass.a.get(k)
            gain = np.ones_like(rgb)
            if static is not None:
                splat = static[:, :, :3] / static[:, :, 3:4].clip(0.01)
                supported = (static[:, :, 3] > 0) & green & seen
                gain = np.where(
                    supported[:, :, None], np.clip(splat / col.clip(15), 0.76, 1.24), 1
                )
                gain = np.dstack(
                    [np.log(gain) * supported[:, :, None], supported.astype("f")]
                )
            else:
                gain = np.zeros((U, U, 4), "f")
            arr = np.zeros((U, U, 10), "f")
            c = np.where(green, 1, np.where(concrete, 2, 0))
            arr[:, :, :3] = col
            arr[:, :, 3] = seen & ~mark
            arr[:, :, 4] = c
            arr[:, :, 5:9] = gain
            arr[:, :, 9] = seen
            g.a[k] = arr
            ids.a[k] = i[:, :, None]
            obj.a[k] = d["object"].astype("f")[:, :, None]
            dist.a[k] = d["dist"].astype("f")[:, :, None]
    # Infer only a slowly varying class colour. Long-distance extensions have no copied photographic edges or marks.
    for ix, k in enumerate(ks):
        pad = 112
        v = g.pad(k, pad)
        sl = np.s_[pad:-pad, pad:-pad]
        seen = v[:, :, 3] > 0.5
        nearcol = []
        basecolors = [[91, 96, 97], [76, 84, 60], [123, 121, 114]]
        ds = []
        for c in range(3):
            ok = seen & (v[:, :, 4] == c)
            wt = ok.astype("f")
            ww = cv.GaussianBlur(wt, (0, 0), 5)
            numer = cv.GaussianBlur(v[:, :, :3] * wt[:, :, None], (0, 0), 5)
            stable = ww > 0.10
            if stable.any():
                d, ij = distance_transform_edt(~stable, return_indices=True)
                reference = np.where(
                    stable[:, :, None], numer / ww[:, :, None].clip(0.01), 0
                )
                vv = reference[ij[0], ij[1]]
                d = (
                    d * L
                )  # Limit propagated brightness/chroma to a quiet class tone rather than copying any signage.
                vv = cv.GaussianBlur(vv, (0, 0), 7)
                ds.append(d[sl])
                nearcol.append(vv[sl])
            else:
                ds.append(np.full((U, U), 1000))
                nearcol.append(np.tile(basecolors[c], (U, U, 1)))
        ds = np.stack(
            ds, -1
        )  # grass/asphalt/concrete are class hypotheses, not a semantic ground-truth mask.
        wt = np.exp(-(ds - ds.min(2, keepdims=True)) / 1.65)
        wt /= wt.sum(2, keepdims=True)
        fill = np.sum(np.stack(nearcol, 2) * wt[:, :, :, None], axis=2)
        G.a[k] = fill.astype("f")
        gain = cv.GaussianBlur(v[:, :, 5:9], (0, 0), 4.5)[sl]
        tone.a[k] = np.exp(gain[:, :, :3] / (gain[:, :, 3:] + 0.04)).astype("f")
    return ids, obj, dist, G, tone


@lru_cache(24)
def oldstate(k):
    return np.load(Path("work/groundtex-final/state/state") / f"{k[0]}_{k[1]}.npy")


def rollback(k, rgba, xy):
    if k not in ROLL or k not in prev():
        return np.zeros(rgba.shape[:2], bool)
    # Restore the native V4 front mapping only within the jointly supported, paint-bearing interior; not the bollard, mover masks or kerb repairs.
    t = prev()[k]
    st = (xy - np.array(k) * M) / t["texelSize"] + t["gutter"] - 0.5
    old = rem(np.asarray(Image.open(V4 / t["texture"]).convert("RGBA")), st)
    source = oldstate(k)
    ij = np.floor((xy - np.array(k) * M) / C).astype(int).clip(0, N - 1)
    ii = source["id"][ij[:, :, 1], ij[:, :, 0]]
    frame = np.array([False] + [e["cam"] == "front" for e in v4obs])
    g = (rgba[:, :, 3] >= 250) & (old[:, :, 3] >= 250) & frame[ii]
    lum = cv.cvtColor(old[:, :, :3], cv.COLOR_RGB2GRAY).astype("f")
    bright = lum - cv.GaussianBlur(lum, (0, 0), 15) > 8
    bright = (
        cv.dilate(
            bright.astype("B"), cv.getStructuringElement(cv.MORPH_ELLIPSE, (79, 79))
        )
        > 0
    )
    g &= bright
    for id in np.unique(ii[g]):
        yy, xx = np.nonzero(g & (ii == id))
        x, y = xx.min(), yy.min()
        w, h = int(np.ptp(xx)) + 1, int(np.ptp(yy)) + 1
        r = v4obs[id - 1]
        uv = uvgrid(r, xy[y, x] - 0.5 * t["texelSize"], (w, h), t["texelSize"], 4)
        accept = actual_guard(r, uv, yy - y, xx - x, k)
        g[yy[~accept], xx[~accept]] = False
    rgba[g, :3] = old[g, :3]
    return g


def asset(out, kind, k, im, ext):
    bio = io.BytesIO()
    img = Image.fromarray(im)
    img.save(
        bio, "WEBP", quality=94, method=3, exact=True
    ) if ext == "webp" else img.save(bio, "PNG", compress_level=5)
    b = bio.getvalue()
    rel = f"{kind}/{k[0]}_{k[1]}_{hashlib.sha256(b).hexdigest()[:12]}.{ext}"
    p = out / rel
    if not p.exists():
        p.write_bytes(b)
    return rel, len(b)


def coherent_apron(xy, col, ap, pcode, rows):
    # Only observed, mutually supported pavement can change owner. Preserve labels, opacity and the junction paint.
    from .terrain_select import goodmask

    q, fw, rt = pose(91.4)
    u = xy - q
    across = u @ rt
    along = u @ fw
    target = (
        (across < -0.95)
        & (across > -7.8)
        & (abs(along) < 6.8)
        & (pcode == 255)
        & (ap > 245)
    )
    if not target.any():
        return np.zeros(ap.shape, bool)
    ys, xs = np.nonzero(target)
    sl = np.s_[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    a = xy[sl]
    ok = target[sl].copy()
    ims = []
    for n in [3333, 3507]:
        id, r = next(
            (i, r)
            for i, r in enumerate(rows, 1)
            if r["cam"] == "left_repeater" and r["n"] == n
        )
        o, delta, gain = ctrl(id)
        mp = (a - o) / (4 * L)
        u = a + rem(delta, mp)
        keys = np.floor(u / M).astype("i")
        zz = np.zeros(a.shape[:2], "f")
        for kk in np.unique(keys.reshape(-1, 2), axis=0):
            m = (keys == kk).all(2)
            s = u[m] / M - kk
            zz[m] = htile(tuple(kk), s[:, 0], s[:, 1])
        uv, v, zc, ds = project(r, u, zz)
        v, _, _ = groundvalid(r, uv, v, zc, ds, zz, u)
        m, lab = goodmask(id)
        cam = inputs()[1][r["cam"]]
        v &= (sam(m, uv, cam, True) > 0) & (sam(lab, uv, cam, True) <= 1)
        ok &= v
        ims.append(
            np.float32(photo(r, uv, False)[:, :, ::-1]) * rem(gain, mp)[:, :, ::-1]
        )
    ok &= cv.GaussianBlur(abs(ims[0] - ims[1]).mean(2), (0, 0), 5) < 31
    f = col[sl].astype("f")
    rgb = ims[0]
    border = cv.dilate((~ok).astype("B"), np.ones((17, 17), "B")).astype(bool) & ok
    diff = (f - rgb).clip(-26, 26)
    ww = border.astype("f")
    g = cv.GaussianBlur(np.dstack([diff * ww[:, :, None], ww]), (0, 0), 18)
    offset = g[:, :, :3] / (g[:, :, 3:] + 0.004)
    rgb = np.uint8(np.clip(rgb + offset + 0.5, 0, 255))
    col[sl][ok] = rgb[ok]
    out = np.zeros(ap.shape, bool)
    out[sl] = ok
    return out


def render(k, stack, rows):
    ids, ob, dist, G, tone = stack
    prior = base().get(k)
    cell = prior["texelSize"] if prior else 0.04
    pix = round(M / cell)
    g = 8
    size = pix + 2 * g
    y, x = np.mgrid[:size, :size].astype("f")
    xy = np.array(k) * M + np.stack([(x + 0.5 - g) * cell, (y + 0.5 - g) * cell], -1)
    uv = (xy - np.array(k) * M) / L + 3.5
    lookup = lambda grid, lin: rem(grid.pad(k, 4), uv, lin)
    # An absent neighbour is outside the band, not distance zero. Evaluate its gutter from the same trajectory so a transparent outside edge cannot turn opaque again.
    dpad = dist.pad(k, 4)
    missing = dpad[:, :, 0] == 0
    if missing.any():
        iy, ix = np.nonzero(missing)
        pts = (np.stack([ix, iy], -1) - 4 + 0.5) * L + np.array(k) * M
        dpad[iy, ix, 0] = height_model()[0].query(pts)[0]
    d = rem(dpad, uv)
    outer = np.uint8(np.rint(255 * smooth((44 - d) / 4)))
    obstacle = lookup(ob, True)
    ap = np.uint8(np.rint(outer * (1 - smooth((obstacle - 0.13) / 0.67))))
    col = lookup(G, True).clip(0, 255).astype("B")
    ident = lookup(ids, False)
    gain = lookup(tone, True)
    pcode = np.where(ap > 0, 127, 0).astype("B")
    old = None
    oldpick = np.zeros((size, size), bool)
    reverted = 0
    if prior:
        old = np.asarray(Image.open(B / prior["texture"]).convert("RGBA")).copy()
        reverted = int(rollback(k, old, xy).sum())
        oldpick = (
            old[:, :, 3] >= 128
        )  # Keep a single source through photographed boundaries; replaced soft support is opaque only if the ring has a witness or an explicit prediction.
    nearentrance = np.linalg.norm((np.array(k) + 0.5) * M - pose(91.4)[0]) < 22
    if nearentrance and prior:
        # On the occluded apron use the winning depth-confirmed observation beyond the driven core, but preserve confirmed paint.
        q, fw, rt = pose(91.4)
        v = xy - q
        select = (
            ((v @ rt) < -1.35) & ((v @ rt) > -6.0) & (abs(v @ fw) < 6.0) & (ident > 0)
        )
        oldpick[select] = False
    need = (ap > 0) & ~oldpick & (ident > 0)
    filled_guard = 0
    validphotos = 0
    native_classes = Counter()
    for id in np.unique(ident[need]):
        yy, xx = np.nonzero(need & (ident == id))
        x0, y0 = int(xx.min()), int(yy.min())
        ww, hh = int(np.ptp(xx)) + 1, int(np.ptp(yy)) + 1
        r = rows[id - 1]
        sl = np.s_[y0 : y0 + hh, x0 : x0 + ww]
        u = xy[sl]
        origin, delta, tg = ctrl(id)
        lattice = (u - origin) / (4 * L)
        shift = rem(delta, lattice)
        u = (
            u + shift
        )  # exactly the same world mesh projection as the evidence pass, with a local cross-view correction
        sx = u[:, :, 0] / M - k[0]
        sy = (
            u[:, :, 1] / M - k[1]
        )  # slight gutter/warp extrapolation at a shared edge uses adjacent surveyed triangles
        zz = htile(k, sx, sy)
        ik = np.floor(u / M).astype("i")
        off = (ik[:, :, 0] != k[0]) | (ik[:, :, 1] != k[1])
        if off.any():
            for kk in np.unique(ik[off], axis=0):
                oo = (ik == kk).all(2)
                zz[oo] = htile(
                    tuple(kk), u[:, :, 0][oo] / M - kk[0], u[:, :, 1][oo] / M - kk[1]
                )
        # Interpolate rolling-shutter UV on a fine physical lattice to bound CPU/memory. Test each admitted native texel against all masks.
        step = max(1, int(0.08 / cell))
        Y, X = np.mgrid[:hh, :ww].astype("f")
        u0, valid0, zc, ds = project(r, u[::step, ::step], zz[::step, ::step])
        mp = np.stack([X, Y], -1) / step
        uu = rem(u0, mp)
        vc = rem(valid0.astype("B"), mp, False) > 0
        zc = rem(zc, mp)
        ds = rem(ds, mp)
        v, sem, _ = groundvalid(r, uu, vc, zc, ds, zz, u)
        if TERRAIN:
            from .terrain_select import goodmask

            valid_class, native_sem = goodmask(id)
            v &= sam(valid_class, uu, inputs()[1][r["cam"]], True) > 0
        accepted = v[yy - y0, xx - x0]
        ys, xs = yy[accepted], xx[accepted]
        if len(ys):
            if TERRAIN:
                codes = sam(native_sem, uu, inputs()[1][r["cam"]], True)[
                    ys - y0, xs - x0
                ]
                assert np.all(np.isin(codes, [0, 1, 9]))
                native_classes.update(
                    {
                        str(int(c)): int(n)
                        for c, n in zip(*np.unique(codes, return_counts=True))
                    }
                )
            im = photo(r, uu, False)[:, :, ::-1]
            local = rem(tg, lattice)[:, :, ::-1]
            col[ys, xs] = np.uint8(
                (im[ys - y0, xs - x0] * local[ys - y0, xs - x0] * gain[ys, xs]).clip(
                    0, 255
                )
            )
            pcode[ys, xs] = 255
            validphotos += len(ys)
        filled_guard += int((~accepted).sum())
    if old is not None:
        col[oldpick] = old[oldpick, :3]
        ap[oldpick] = np.maximum(old[oldpick, 3], ap[oldpick])
        pcode[oldpick] = 255
    if TERRAIN and nearentrance and prior:
        apron = coherent_apron(xy, col, ap, pcode, rows)
        oldpick[apron] = False
    if TERRAIN and old is not None and oldpick.any() and not oldpick.all():
        # Fit only the low-frequency colour step along ownership boundaries. Never interpolate two photographs' high-frequency RGB.
        inside = oldpick
        f = col.astype("f")
        res = np.zeros(f.shape[:2] + (4,), "f")
        for dy, dx in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nb = np.roll(inside, (dy, dx), (0, 1))
            eligible = ~inside & nb & (ap > 230)
            eligible[:2] = eligible[-2:] = False
            eligible[:, :2] = eligible[:, -2:] = False
            delta = np.roll(f, (dy, dx), (0, 1)) - f
            similar = (abs(delta).mean(2) < 50) & eligible
            res[:, :, :3] += np.clip(delta, -35, 35) * similar[:, :, None]
            res[:, :, 3] += similar
        n0 = max(20, round(size * cell / 0.04))
        small = cv.resize(res, (n0, n0), interpolation=cv.INTER_AREA)
        v = cv.GaussianBlur(small, (0, 0), 4.2)
        delta = v[:, :, :3] / (v[:, :, 3:] + 0.004)
        delta = cv.resize(delta, (size, size), interpolation=cv.INTER_LINEAR)
        sd = cv.distanceTransform((~inside).astype("B"), cv.DIST_L2, 5) * cell
        strength = np.exp(-sd * sd / 0.26) * (~inside)
        col = np.uint8(np.clip(f + delta * strength[:, :, None] + 0.5, 0, 255))
    ap[outer == 0] = 0
    pcode[ap == 0] = 0
    col[ap == 0] = 0
    if not np.any(ap[g:-g, g:-g]):
        return None
    rgb, len1 = asset(OUT, "tiles", k, np.dstack([col, ap]), "webp")
    mask, len2 = asset(OUT, "provenance", k, pcode, "png")
    t = dict(
        id=f"{k[0]}_{k[1]}",
        bounds=np.round([*(np.array(k) * M), *((np.array(k) + 1) * M)], 5).tolist(),
        pixels=pix,
        texelSize=cell,
        gutter=g,
        texture=rgb,
        bytes=len1,
        provenance=mask,
        provenanceBytes=len2,
        grid=prior["grid"]
        if prior
        else dict(cols=9, rows=9, z=mesh(k).astype(float).round(5).ravel().tolist()),
    )
    inner = np.s_[g:-g, g:-g]
    a = ap[inner]
    prov = pcode[inner]
    old0 = oldpick[inner]
    rr = dict(
        id=t["id"],
        present_m2=float((a > 0).sum() * cell**2),
        legacy_m2=float(((prov == 255) & old0).sum() * cell**2),
        new_observed_m2=float(((prov == 255) & ~old0).sum() * cell**2),
        predicted_m2=float((prov == 127).sum() * cell**2),
        core40_predicted_m2=float(((prov == 127) & (d[inner] <= 40)).sum() * cell**2),
        core40_present_m2=float(((prov > 0) & (d[inner] <= 40)).sum() * cell**2),
        object_cut_m2=float(
            ((a == 0) & (d[inner] <= 40) & (obstacle[inner] > 0.7)).sum() * cell**2
        ),
        unexplained_core40_m2=float(
            ((a == 0) & (d[inner] <= 40) & (obstacle[inner] <= 0.7)).sum() * cell**2
        ),
        reverted_texels=reverted,
        guard_rejected=filled_guard,
    )
    for key in ["0", "1", "9"]:
        rr["admitted_class_" + key] = native_classes[key]
    return t, rr


def build(workers):
    global v4obs
    v4obs = read("work/groundtex-final/state/observations.json")
    rows = read(W / "outer-rows.json")
    (OUT / "tiles").mkdir(parents=True, exist_ok=True)
    (OUT / "provenance").mkdir(exist_ok=True)
    print("class propagation", flush=True)
    stack = prepare()
    results = []
    area = []
    print("native export", flush=True)
    with ThreadPoolExecutor(workers) as ex:
        for j, z in enumerate(ex.map(lambda k: render(k, stack, rows), pathkeys())):
            if z:
                results.append(z[0])
                area.append(z[1])
            if j % 100 == 0:
                print("tile", j, flush=True)
    meta = read(B / "ground.json")
    meta["tiles"] = results
    meta["extension"] = {
        "revision": 6,
        "groundBandMetres": 40,
        "fadeEndMetres": 44,
        "outerTexelSize": 0.04,
        "support": "native-depth+multiview+removal-masks",
    }
    meta["provenance"] = {
        "kind": "per-tile-mask",
        "tileField": "provenance",
        "encoding": "png-gray8",
        "values": SCHEM,
        "orientation": "same raster and gutters as texture",
        "predicted": "low-frequency adjacent surface class; no photographic features",
    }
    dump(OUT / "ground.json", meta)
    dump(W / "tile-areas.json", area)
    print("export ready", len(results), flush=True)


if __name__ == "__main__":
    import sys

    build(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
