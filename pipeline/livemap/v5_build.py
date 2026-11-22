"""Build grounded profile lifetimes, a narrow road corridor and viewer-compatible temporal chunks."""

from .v5_profiles import *
from .v4_build import ct, time, ThreadPoolExecutor, deque, Rotation, link, binary_header
from .v4_render import reader
from scipy.ndimage import gaussian_filter


def make_sky(plan, root, dst):
    values = []
    dirs = []
    for c, es in plan.items():
        ray, ok = lattice(c)
        for e in es[::8]:
            key = f"{e['n']:06d}"
            mask = cv.erode(
                cv.imread(str(root / "masks/sky" / c / (key + ".png")), 0),
                np.ones((17, 17), "B"),
            )
            y, x = np.where(((mask > 0) & ok)[::8, ::8])
            y *= 8
            x *= 8
            if not len(y):
                continue
            img = photo("live").apply(
                c,
                e["n"],
                cv.resize(
                    cv.imread(str(RS / "data/drive/frames" / c / (key + ".jpg")))[
                        :, :, ::-1
                    ],
                    (B, H),
                    interpolation=cv.INTER_AREA,
                ),
            )
            dr = np.einsum(
                "nij,nj->ni", rows("live", c, e["t"])[0][y, :3, :3], ray[y, x]
            )
            dirs.append(dr)
            values.append(img[y, x])
    v = np.vstack(values).astype("f")
    dr = np.vstack(dirs)
    Y = v @ [0.2126, 0.7152, 0.0722]
    target = np.clip(np.percentile(Y, 92) * 1.20, 184, 208)
    ang = (0.5 - np.arctan2(dr[:, 1], dr[:, 0]) / (2 * np.pi)) % 1
    vert = 0.5 - np.arcsin(np.clip(dr[:, 2], -1, 1)) / np.pi
    u = np.floor(ang * 32).astype(int)
    vbin = np.clip((vert * 16).astype(int), 0, 15)
    grid = np.zeros((16, 32))
    wt = grid.copy()
    for i in range(16):
        for j in range(32):
            ls = Y[(vbin == i) & (u == j)]
            if len(ls) > 10:
                grid[i, j] = np.percentile(ls, 67)
                wt[i, j] = min(len(ls), 600)
    blur = lambda a: gaussian_filter(a, (1.7, 2.6), mode=("nearest", "wrap"))
    smooth = blur(grid * wt) / np.maximum(1, blur(wt))
    base = np.percentile(Y, 67)
    delta = np.clip((smooth - base) * 0.22, -8, 8) * np.clip(blur(wt) / 120, 0, 1)
    h, w = 512, 1024
    y, x = np.mgrid[:h, :w]
    dev = cv.remap(
        delta.astype("f"),
        ((x + 0.5) * 32 / w - 0.5).astype("f"),
        np.clip((y + 0.5) * 16 / h - 0.5, 0, 15).astype("f"),
        cv.INTER_CUBIC,
        borderMode=cv.BORDER_WRAP,
    )
    up = np.maximum(0, np.cos((y + 0.5) * np.pi / h))
    lum = target + 11 * (1 - up) - 6 * up + dev
    rgb = np.stack([lum + 3 * (1 - up) - 2 * up, lum, lum + 4 * up], 2)
    cv.imwrite(
        str(dst / "sky.jpg"),
        np.uint8(np.clip(rgb / wb(), 0, 255))[:, :, ::-1],
        [cv.IMWRITE_JPEG_QUALITY, 94],
    )
    return dict(
        samples=len(Y),
        source_p50=float(np.median(Y)),
        source_p92=float(np.percentile(Y, 92)),
        output_p10_p50_p90=np.percentile(lum[:256], [10, 50, 90]).tolist(),
    )


def time_keys(a):
    b, d = a[:, 17], a[:, 18]
    finite = d < INF * 0.9
    short = finite & (d - b < 8)
    death = np.where(short, np.floor(d / 1.5), np.floor(d / 5))
    return np.where(
        ~finite,
        np.floor(b / 8),
        np.where(short, 100000 + death, 200000 + np.floor(b / 8) * 1024 + death),
    ).astype("i8")


def store(dst, tiles, name, a, layer, total):
    if not len(a):
        return
    file = "tiles/" + name + ".ply"
    tmp = (dst / file).with_suffix(".partial")
    assert np.isfinite(a).all() and np.all(a[:, 17] < a[:, 18])
    with open(tmp, "wb") as f:
        f.write(binary_header(len(a)))
        a.astype("<f4", copy=False).tofile(f)
    os.replace(tmp, dst / file)
    tiles.append(
        dict(
            id=name,
            file=file,
            count=len(a),
            layer=layer,
            kind="transient" if np.all(a[:, 18] < INF * 0.9) else "static",
            bbox=np.r_[a[:, :3].min(0), a[:, :3].max(0)].tolist(),
            t_min=float(a[:, 17].min()),
            t_max=float(min(total, a[:, 18].max())),
        )
    )


def road(dst, tiles, total):
    gr = RS / "scene/road-live-v2"
    gm = json.load(open(gr / "map.json"))
    dr = json.load(open(gr / "drive.json"))
    S = {k: np.asarray(dr[k]) for k in ["t", "x", "y", "yaw"]}
    S["yaw"] = np.unwrap(S["yaw"])
    orig = count = 0
    for item in gm["tiles"]:
        a = reader(gr / item["file"])
        orig += len(a)
        b = a[:, 17]
        xx, yy = a[:, 0], a[:, 1]
        lat = np.full(len(a), 1000.0)
        for look in [0, 0.4, 1.0, 2.0, 4.0, 7.0]:
            t = b - look
            x, y, yaw = [np.interp(t, S["t"], S[k]) for k in ["x", "y", "yaw"]]
            dx = xx - x
            dy = yy - y
            f = dx * np.cos(yaw) + dy * np.sin(yaw)
            ls = abs(-dx * np.sin(yaw) + dy * np.cos(yaw))
            allow = abs(f) < (38 if look == 0 else 7)
            lat = np.minimum(lat, np.where(allow, ls, 1000))
        sel = lat < 13.5
        a = np.array(a[sel])
        lat = lat[sel]
        count += len(a)
        mix = np.clip((13.5 - lat) / 4, 0.001, 1)
        alpha = (1 / (1 + np.exp(-a[:, 9]))) * mix * mix * (3 - 2 * mix)
        a[:, 9] = np.log(np.maximum(1e-5, alpha) / np.maximum(1e-5, 1 - alpha))
        k = time_keys(a)
        for t in np.unique(k):
            store(dst, tiles, f"g_{item['id']}_{t}", a[k == t], "ground", total)
    return dict(before=orig, after=count, full_width_m=19, outer_width_m=27)


def build(scope="slice"):
    root = DV if scope == "slice" else V4 / "whole-inputs"
    p = json.load(open(V4 / (scope + "-plan.json")))
    dst = Path(os.environ.get("V5_DEST", V5 / ("candidate-" + scope)))
    (dst / "tiles").mkdir(parents=True, exist_ok=True)
    records = sorted(
        (
            (ready("live", c, e), NAMES.index(c), c, e)
            for c, es in p.items()
            for e in es
        ),
        key=lambda r: (r[0], r[3]["t"], r[1]),
    )
    desc = json.loads((root / "depth/keyframes.json").read_text())["depth"]
    for c, es in p.items():
        source = desc["cameras"][c].get("trajectory", desc.get("trajectory", ""))
        assert (
            Path(source).name == "ego_causal.npz"
            and (root / ("complete-" + c)).is_file()
        ), c
        complete(root, c, es)
    gr = RS / "scene/road-live-v2"
    gm = json.load(open(gr / "map.json"))
    total = gm["t_end"]
    tiles = []
    for f in ["drive.json", "rig.json"]:
        link(gr / f, dst / f)
    gs = road(dst, tiles, total)
    lib = ct.CDLL(str(V4 / "now.dylib"))
    P = ct.c_void_p
    I = ct.c_int
    F = ct.c_float
    lib.now_init.restype = P
    lib.now_take.argtypes = [P, P, I, P, P, P, P, P, F, I, I, P]
    lib.now_save.argtypes = [P, ct.c_char_p, ct.c_char_p, P, F]
    lib.now_save.restype = ct.c_long
    lib.now_free.argtypes = [P]
    engine = lib.now_init()
    pack = []
    offset = 0
    pts = []
    owners = []
    kinds = []
    start = time.monotonic()

    def prep(rr):
        t, ci, c, e = rr
        return packet(c, e, root, t)

    with ThreadPoolExecutor(5) as pool:
        q = deque(pool.submit(prep, r) for r in records[:8])
        for i, (t, ci, c, e) in enumerate(records):
            reps, m, o, k, ex = q.popleft().result()
            if i + 8 < len(records):
                q.append(pool.submit(prep, records[i + 8]))
            head = np.ascontiguousarray(traj("live").T_world_ego(e["t"])[:3, 0], "f")
            lib.now_take(
                engine,
                reps.ctypes.data,
                len(reps),
                *[a.ctypes.data for a in ex],
                t,
                ci,
                e["ordinal"],
                head.ctypes.data,
            )
            pack.append((ci, e["ordinal"], np.ascontiguousarray(reps[:, :3]), offset))
            pts.append(m)
            owners.append(o + offset)
            kinds.append(k)
            offset += len(k)
            if i % 300 == 0:
                print(scope, i, "of", len(records), "profiles", offset, flush=True)
    cnt = np.zeros(7, "q")
    raw = V5 / (scope + ".bin")
    tags = V5 / (scope + ".tags")
    ng = lib.now_save(
        engine, os.fsencode(raw), os.fsencode(tags), cnt.ctypes.data, records[-1][0]
    )
    lib.now_free(engine)
    group = np.memmap(raw, "f", "r", shape=(ng, 22))
    meta = np.memmap(tags, "i", "r", shape=(ng, 3))
    members = np.concatenate(pts)
    owner = np.concatenate(owners)
    kind = np.concatenate(kinds)
    assert np.all(np.diff(owner) >= 0)
    del pts, owners
    decoded = np.empty(ng, "i")
    pair = meta[:, 0] * 100000 + meta[:, 1]
    ri = np.argsort(pair)
    pairs = pair[ri]
    for ci, ordinal, reps, off in pack:
        a, b = np.searchsorted(
            pairs, [ci * 100000 + ordinal, ci * 100000 + ordinal + 1]
        )
        idx = ri[a:b]
        if not len(idx):
            continue
        code = reps.view("V12").ravel()
        ix = np.argsort(code)
        vals = np.ascontiguousarray(group[idx, :3]).view("V12").ravel()
        j = ix[np.searchsorted(code[ix], vals)]
        assert np.array_equal(reps[j], group[idx, :3])
        decoded[idx] = off + j
    st = np.searchsorted(owner, decoded)
    en = np.searchsorted(owner, decoded, "right")
    n = en - st
    gidx = np.repeat(np.arange(ng), n)
    mj = np.repeat(st - np.r_[0, np.cumsum(n)[:-1]], n) + np.arange(len(gidx))
    m = members[mj]
    labels = kind[decoded[gidx]]
    tm = group[gidx, 15:]
    mt = meta[gidx]
    normal = m[:, 3:6]
    u = m[:, 6:9]
    rot = Rotation.from_matrix(np.stack((u, np.cross(normal, u), normal), 2)).as_quat()[
        :, [3, 0, 1, 2]
    ]
    alpha = m[:, 15]
    A = np.column_stack(
        (
            m[:, :6],
            (m[:, 12:15] / 255 - 0.5) / 0.28209479,
            np.log(alpha / (1 - alpha)),
            np.log(m[:, 9:12]),
            rot,
            tm,
        )
    ).astype("f")
    assert len(A) < 60000000
    xy = np.floor(A[:, :2] / 42).astype("i8")
    tk = time_keys(A)
    key = (xy[:, 0] + 10000) * 20000 + xy[:, 1] + 10000 + tk * 500000000
    order = np.lexsort((mt[:, 2], mt[:, 1], mt[:, 0], key))
    _, starts = np.unique(key[order], return_index=True)
    runs = {}
    for b, h in zip(starts, np.r_[starts[1:], len(order)]):
        ix = order[b:h]
        j = ix[0]
        name = f"o_{xy[j, 0]}_{xy[j, 1]}_{tk[j]}"
        store(dst, tiles, name, A[ix], "objects", total)
        md = mt[ix]
        ii = np.r_[0, 1 + np.flatnonzero(np.any(md[1:] != md[:-1], 1))]
        for l, h0 in zip(ii, np.r_[ii[1:], len(md)]):
            c, n, iid = md[l]
            if iid:
                runs.setdefault((int(c), int(n), int(iid)), []).append(
                    dict(tile=name, first=int(l), count=int(h0 - l))
                )
    side = (
        json.load(
            open(
                V4 / ("instances.json" if scope == "slice" else "whole-instances.json")
            )
        )
        if scope == "slice" or (V4 / "whole-instances.json").exists()
        else {"frames": []}
    )
    for row in side["frames"]:
        for inst in row["instances"]:
            r = runs.pop((NAMES.index(row["cam"]), row["ordinal"], inst["id"]), [])
            inst["tile_runs"] = r
            inst["seeds"] = sum(z["count"] for z in r)
    (V5 / (scope + "-instances.json")).write_text(
        json.dumps(
            {
                "frames": side["frames"],
                "additional_runs": [(k, v) for k, v in runs.items()],
            },
            separators=(",", ":"),
        )
    )
    sky = make_sky(p, root, dst)
    gm.update(tiles=tiles, sky="sky.jpg")
    (dst / "map.json").write_text(json.dumps(gm, separators=(",", ":")))
    info = dict(
        scope=scope,
        ground=gs,
        objects=len(A),
        profiles=ng,
        source_profiles=offset,
        frames=len(records),
        shapes=dict(zip(KINDS, np.bincount(labels, minlength=6).tolist())),
        sky=sky,
        tiles=len(tiles),
        object_tiles=sum(t["layer"] == "objects" for t in tiles),
        seconds=time.monotonic() - start,
        finite_lifetime=int((A[:, 18] < INF * 0.9).sum()),
    )
    (V5 / (scope + "-summary.json")).write_text(json.dumps(info, indent=2))
    print(info, flush=True)


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "slice")
