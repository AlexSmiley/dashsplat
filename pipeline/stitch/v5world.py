"""Native photographic witnesses for smaller, more opaque object footprints. Road never enters the fit."""

from .v4teacher import *
from .v2seed import unit
from .v3terrain import surfels

W5 = Path(os.getenv("STITCH_V5_WORK", RS / "work/stitch-v5" / f"{START:g}-{END:g}"))
BENCHMARK = START == 85 and "STITCH_V5_META" not in os.environ
W5.mkdir(parents=True, exist_ok=True)
route = traj.T_world_ego(np.arange(START, END + 0.01, 0.1))[:, :2, 3]
rt = cKDTree(route)


@lru_cache(None)
def semantic(c, n):
    extra = W5 / "sem" / c / f"{n:06}.png"
    if extra.exists():
        return extra
    for d in ["popup-v3/sem", "popup-v2/sem", "horizon-l/semantic"]:
        p = RS / "work/ibr" / d / c / f"{n:06}.png"
        if p.is_file():
            return p
    return None


def labels(c, n):
    sky, dyn, road = masks(c, n)
    sem = semantic(c, n)
    s = cv.imread(str(sem), 0) if sem else np.full(sky.shape, 255, "B")
    s[sky != 0] = 10
    s[road != 0] = 0
    m = RS / "maps/movers-v1/remove" / c / f"{n:06}.png"
    rem = cv.imread(str(m), -1) if m.is_file() else dyn
    if rem.ndim == 3:
        rem = rem.max(2)
    rem = cv.resize((rem != 0).astype("B"), (724, 469), interpolation=0)
    s[(rem != 0) | ~rays(c)[1]] = 250
    return s


@lru_cache(None)
def physical_schedule():
    d = schedule()
    if BENCHMARK:
        return d
    for c in CAMS:
        index = J(RS / "depth" / c / "index.json")
        held = {e["n"] for e in d[c] if e["hold"]}
        for e in d[c]:
            entry = index.get(str(e["n"]), {})
            f = RS / "depth" / c / f"{e['n']:06}.npy"
            if (
                not e["depth"]
                and not e["hold"]
                and entry.get("posed")
                and entry.get("model") == "mapanything-nomask"
                and entry.get("window")
                and not held.intersection(entry["window"])
                and f.exists()
            ):
                e["depth"] = str(f)
                e["metric_side"] = True
    return d


def depth(e):
    if e.get("metric_side"):
        z = np.load(e["depth"]).astype("f")
        ray, ok = rays(e["c"])
        r = z / np.maximum(0.045, ray[:, :, 2])
        return np.where(ok & (r > 1) & (r < 200) & (ray[:, :, 2] > 0.04), r, 0).astype(
            "f"
        )
    rr = (
        physical_range(e["c"], np.load(e["depth"]).astype("f"))[0]
        if e["depth"]
        else np.zeros((469, 724), "f")
    )
    mask_root = os.getenv("STITCH_V5_RANGE_MASKS")
    if mask_root and e["depth"]:
        mask = cv.imread(str(Path(mask_root) / e["c"] / f"{e['n']:06}.png"), -1)
        if mask is None or mask.shape != rr.shape:
            raise ValueError(f"Missing audited range mask: {e['c']} {e['n']}")
        return np.where(mask == 0, rr, 0)
    return rr


def frames(c, spacing):
    last = -1e10
    for e in physical_schedule()[c]:
        if e["hold"] or not e["depth"] or e["odo"] - last < spacing:
            continue
        last = e["odo"]
        yield e


def tilefile(dst, rec, A):
    if not len(A):
        return None
    f = dst / rec["file"]
    f.parent.mkdir(exist_ok=True, parents=True)
    with f.open("wb") as fd:
        fd.write(binary_header(len(A)))
        np.ascontiguousarray(A, "<f4").tofile(fd)
    return dict(
        rec, count=len(A), bbox=np.r_[A[:, :3].min(0), A[:, :3].max(0)].tolist()
    )


def assemble(dst, src, objects):
    dst.mkdir(parents=True, exist_ok=False)
    (dst / "tiles").mkdir()
    m = J(src / "map.json")
    v3 = RS / "scene/stitched-opt-v3"
    gsrc = v3 if BENCHMARK else GROUND
    gl = [t for t in J(gsrc / "map.json")["tiles"] if t["layer"] == "ground"]
    tiles = []
    for t in gl:
        b = np.array(t["bbox"])
        r = np.linalg.norm(b[3:5] - b[:2]) / 2
        ctr = (b[3:5] + b[:2]) / 2
        if not BENCHMARK and rt.query(ctr)[0] > 85 + r:
            continue
        os.link(gsrc / t["file"], dst / t["file"])
        tiles.append(t)
    for t, A in objects:
        rec = tilefile(dst, t, A)
        if rec:
            tiles.append(rec)
    for f in ["rig.json", "drive.json", "sky.jpg"]:
        shutil.copy2(Path(os.getenv("STITCH_V5_META", RS / "scene/stitched-opt-v4")) / f, dst / f)
    savej(dst / "map.json", dict(m, tiles=tiles, sky="sky.jpg", slice_s=[START, END]))


def group(A, prefix):
    ij = np.floor(A[:, :2] / 32).astype(int)
    key = ij[:, 0] * 65536 + ij[:, 1]
    o = np.argsort(key)
    b = np.r_[0, np.flatnonzero(np.diff(key[o])) + 1, len(A)]
    out = []
    for lo, hi in zip(b[:-1], b[1:]):
        q = A[o[lo:hi]]
        i, j = ij[o[lo]]
        name = f"{prefix}_{i}_{j}"
        out.append(
            (
                dict(
                    id=name,
                    kind="static",
                    layer="objects",
                    file=f"tiles/{name}.ply",
                    t_min=0,
                    t_max=361.42,
                ),
                q,
            )
        )
    return out


def traffic_seed(dstname="seed"):
    from . import v2data

    v2data.tiles_ground = lambda: [
        t for t in J(GROUND / "map.json")["tiles"] if t["layer"] == "ground"
    ]
    vol = Volume()
    buf = []
    ks = []
    scores = []
    for c in CAMS:
        ray, _ = rays(c)
        for e in frames(c, 2.2):
            n = e["n"]
            lab = labels(c, n)
            rr = depth(e)
            y, x = np.mgrid[2 + n % 4 : 466 : 4, 2 + (n // 4) % 4 : 720 : 4]
            d = rr[y, x]
            sel = ~np.isin(lab[y, x], [0, 1, 9, 10, 250]) & (d > 1) & (d < 155)
            x, y, d = x[sel], y[sel], d[sel]
            M = poses(c, e["t"])[(y * 8 // 469)]
            u = np.einsum("nij,nj->ni", M[:, :3, :3], ray[y, x])
            P = M[:, :3, 3] + u * d[:, None]
            g, h, _ = vol.lookup(P)
            sel = (P[:, 2] > np.maximum(g + 0.12, h + 0.07)) & (P[:, 2] < g + 38)
            P, u, x, y, d = P[sel], u[sel], x[sel], y[sel], d[sel]
            lat = rt.query(P[:, :2])[0]
            pitch = np.clip(0.085 + 0.006 * lat, 0.085, 0.57)
            level = np.digitize(pitch, [0.13, 0.21, 0.35, 0.48])
            pitch = np.array([0.105, 0.17, 0.28, 0.4, 0.56])[level]
            ij = np.floor(P / pitch[:, None]).astype("int64") + 65536
            key = ((level * 131072 + ij[:, 0]) * 131072 + ij[:, 1]) * 131072 + ij[:, 2]
            a = surfels(
                P,
                rgb(c, n)[y, x],
                np.c_[pitch * 0.83, pitch * 0.83, pitch * 0.3],
                0.67,
                e["t"],
            )
            norm = -u
            tx = unit(
                np.cross(
                    np.where(abs(norm[:, 2:3]) > 0.88, [0, 1.0, 0], [0, 0, 1.0]), norm
                )
            )
            a[:, 3:6] = norm
            a[:, 13:17] = Rotation.from_matrix(
                np.stack([tx, np.cross(norm, tx), norm], 2)
            ).as_quat()[:, [3, 0, 1, 2]]
            buf.append(a)
            ks.append(key)
            scores.append(1 / (1 + d * 0.03))
        print("seed", c, sum(map(len, buf)), flush=True)
    k = np.concatenate(ks)
    o = np.lexsort((-np.concatenate(scores), k))
    i = o[np.r_[True, np.diff(k[o]) != 0]]
    a = np.concatenate(buf)[i]
    assemble(W5 / dstname, GROUND, group(a, "traffic"))
    print("unique seed", len(a), flush=True)


def prepare(src):
    vol = Volume()
    M = J(src / "map.json")
    bags = []
    parent = []
    tiles = []
    for t in M["tiles"]:
        if t["layer"] != "objects":
            continue
        a = reader(src / t["file"])
        if not BENCHMARK:
            # Independently posed depth points that meet in a coarse voxel support one surface. Keep its closest-to-centroid witness rather than layering noisy depth sheets.
            lat0 = rt.query(a[:, :2])[0]
            levels = np.digitize(lat0, [14, 34, 65])
            pitch = np.array([0.18, 0.29, 0.46, 0.76])[levels]
            ijk = np.floor(a[:, :3] / pitch[:, None]).astype("i8") + 65536
            keys = ((levels * 131072 + ijk[:, 0]) * 131072 + ijk[:, 1]) * 131072 + ijk[
                :, 2
            ]
            _, grp, nobs = np.unique(keys, return_inverse=True, return_counts=True)
            mu = np.stack([np.bincount(grp, weights=v) / nobs for v in a[:, :3].T], 1)
            dist = np.linalg.norm(a[:, :3] - mu[grp], axis=1)
            order = np.lexsort((dist, grp))
            first = order[np.r_[True, np.diff(grp[order]) != 0]]
            near = lat0[first] < 20
            take = first[(nobs[grp[first]] >= 2) | near]
            a = a[take]
            if not len(a):
                continue
        g, h, owned = vol.lookup(a[:, :3])
        sz = np.exp(a[:, 10:12])
        lat = rt.query(a[:, :2])[0]
        height = a[:, 2] - g
        genuine = (height > 0.23) & (a[:, 2] > h + 0.05)
        split = (
            genuine
            & (sz.max(1) > 0.057 + 0.0008 * np.minimum(100, lat))
            & (a[:, 9] > -2.25)
        )
        keep = ~((height < 0.10) & owned.astype(bool) & (abs(a[:, 5]) > 0.76))
        split &= keep
        b = np.repeat(a[split], 4, 0)
        R = Rotation.from_quat(a[split][:, [14, 15, 16, 13]]).as_matrix()
        hx = R[:, :, 0] * sz[split, :1] * 0.67
        hy = R[:, :, 1] * sz[split, 1:] * 0.67
        b[:, :3] += np.stack([-hx - hy, -hx + hy, hx - hy, hx + hy], 1).reshape(-1, 3)
        b[:, 10:12] += np.log(0.61)
        b[:, 12] += np.log(0.66)
        A = np.ascontiguousarray(np.concatenate([a[keep & ~split], b]), "f")
        bags.append(A)
        tiles.append(t)
        parent.append(len(a[keep & ~split]))
    return bags, tiles, parent


def native(src, dstname="native"):
    A, tiles, ns = prepare(src)
    S = [np.zeros((len(a), 18), "f") for a in A]
    ctr = np.array([a[:, :3].mean(0) for a in A])
    rad = np.array(
        [np.linalg.norm(a[:, :3] - ctr[i], axis=1).max() for i, a in enumerate(A)]
    )
    lib = W5 / "native.dylib"
    subprocess.run(
        [
            "c++",
            "-O3",
            "-std=c++17",
            *(["-shared", "-fPIC"] if sys.platform.startswith("linux") else ["-dynamiclib"]),
            Path(__file__).with_name("v5sample.cpp"),
            "-o",
            lib,
        ],
        check=True,
    )
    fn = ct.CDLL(str(lib)).observe
    P = ct.c_void_p
    I = ct.c_int
    fn.argtypes = [P, I, P, P, P, I, I, P, P, P, ct.c_float]
    stats = {}
    with ThreadPoolExecutor(6) as ex:
        for c in CAMS:
            C = rig[c]
            K = np.ascontiguousarray(C.params, "d")
            num = 0
            for e in frames(c, 1.8 if c == "front" else 3.6):
                n = e["n"]
                T = poses(c, e["t"])
                delta = ctr - T[4, :3, 3]
                z = delta @ T[4, :3, 2]
                ids = np.flatnonzero(
                    (
                        np.linalg.norm(delta, axis=1)
                        < (205 if c == "front" else 100) + rad
                    )
                    & (z > -rad)
                )
                lab = labels(c, n)
                rr = depth(e)
                im = cv.imread(str(RS / "data/drive/frames" / c / f"{n:06}.jpg"))
                im = np.ascontiguousarray(photo.apply(c, n, im[:, :, ::-1]))
                hh, ww = im.shape[:2]
                assert (ww, hh) == (C.width, C.height)

                def one(i):
                    fn(
                        A[i].ctypes.data,
                        len(A[i]),
                        S[i].ctypes.data,
                        T.ctypes.data,
                        K.ctypes.data,
                        ww,
                        hh,
                        im.ctypes.data,
                        lab.ctypes.data,
                        rr.ctypes.data,
                        1.6 if c == "front" else 1.0,
                    )

                list(ex.map(one, ids))
                num += 1
            stats[c] = num
            print("native", c, num, flush=True)
    vol = Volume()
    parts = []
    cnt = np.zeros(6, int)
    for t, a, s, num in zip(tiles, A, S, ns):
        g, h, o = vol.lookup(a[:, :3])
        lat = rt.query(a[:, :2])[0]
        wt = s[:, 2]
        alpha = 1 / (1 + np.exp(-a[:, 9]))
        avg = s[:, 4:7] / np.maximum(0.01, wt[:, None])
        std = np.sqrt(np.maximum(0, s[:, 3] / np.maximum(0.01, wt) - avg.mean(1) ** 2))
        best = np.clip(
            s[:, 8:11],
            avg - np.minimum(52, 9 + 1.8 * std[:, None]),
            avg + np.minimum(52, 9 + 1.8 * std[:, None]),
        )
        reliable = (s[:, 14] >= 2) & (wt > 1.25)
        colour = (a[:, 6:9] * C0 + 0.5) * 255
        frac = np.where(reliable, 0.84, np.minimum(0.22, wt * 0.13))
        a[:, 6:9] = (
            np.clip(colour + frac[:, None] * (best - colour), 0, 255) / 255 - 0.5
        ) / C0
        high = a[:, 2] > g + 0.60
        fol = s[:, 11] > 0.35 * wt
        sky = (s[:, 0] > np.maximum(2.5, 1.3 * wt)) & high
        neg = (
            (s[:, 1] > np.maximum(4, 1.5 * wt)) & (a[:, 2] < g + 0.38) & o.astype(bool)
        )
        protect = (s[:, 13] > 0.30 * wt) & reliable
        # Signed two-well prior on physically observed elevated footprints; shafts are never pruned on a sky mask alone.
        logit = a[:, 9].copy()
        logit += np.where(high & reliable, 1.08 * np.tanh((alpha - 0.32) * 5) + 0.56, 0)
        logit -= np.where(high & (wt < 0.5) & (alpha < 0.33), 1.25, 0)
        logit -= np.where(sky & ~protect, 2.5, 0)
        a[:, 9] = np.clip(logit, -7, 5)
        al = 1 / (1 + np.exp(-a[:, 9]))
        keep = ~(neg | ((al < 0.14) & high & ~protect))
        cap = np.clip(0.105 + 0.007 * lat, 0.105, 1.10)
        floor = np.clip(0.012 + 0.0007 * lat, 0.012, 0.11)
        a[:, 10:12] = np.clip(a[:, 10:12], np.log(floor[:, None]), np.log(cap[:, None]))
        a[:, 12] = np.minimum(a[:, 12], np.log(cap * 0.45))
        parts.append((t, a[keep]))
        cnt += [
            len(a),
            sum(keep),
            sum(reliable),
            sum(~keep),
            sum(al[keep] > 0.8),
            sum(fol & reliable),
        ]
    assemble(W5 / dstname, src, parts)
    savej(W5 / (dstname + ".json"), dict(views=stats, counters=cnt.tolist()))
    print("counts", cnt, flush=True)


if __name__ == "__main__":
    if sys.argv[1].startswith("seed"):
        traffic_seed(sys.argv[1])
    else:
        native(Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "native")
