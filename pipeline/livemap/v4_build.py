"""Write immutable now+memory records, original ground links and per-instance transient ranges."""

from .v4_sky import *
from .build import link, binary_header, Rotation, ct, time, ThreadPoolExecutor, deque


def build(scope="slice", cams=CAMS):
    p = json.loads((V4 / (scope + "-plan.json")).read_text())
    root = DV if scope == "slice" else V4 / "whole-inputs"
    tag = os.environ.get("V4_TAG") or (
        scope if cams == CAMS else scope + "-" + "-".join(cams)
    )
    dst = (
        Path(os.environ.get("V4_DEST", RS / "scene/live-v4"))
        if cams == CAMS
        else V4 / ("preview-" + tag)
    )
    desc = json.loads((root / "depth/keyframes.json").read_text())["depth"]
    for c in cams:
        assert (
            Path(desc["cameras"][c].get("trajectory", desc.get("trajectory", ""))).name
            == "ego_causal.npz"
        ), c
    p = {c: p[c] for c in cams}
    for c, es in p.items():
        assert (root / ("complete-" + c)).exists()
        complete(root, c, es)
    records = sorted(
        (
            (ready("live", c, e), NAMES.index(c), c, e)
            for c, es in p.items()
            for e in es
        ),
        key=lambda r: (r[0], r[3]["t"], r[1]),
    )
    (dst / "tiles").mkdir(parents=True, exist_ok=True)
    groot = RS / "scene/road-live-v2"
    gm = json.loads((groot / "map.json").read_text())
    tiles = list(gm["tiles"])
    for f in ["rig.json", "drive.json", *[t["file"] for t in tiles]]:
        link(groot / f, dst / f)
    lp = ct.CDLL(str(V4 / "now.dylib"))
    P = ct.c_void_p
    I = ct.c_int
    F = ct.c_float
    lp.now_init.restype = P
    lp.now_free.argtypes = [P]
    lp.now_take.argtypes = [P, P, I, P, P, P, P, P, F, I, I, P]
    lp.now_save.argtypes = [P, ct.c_char_p, ct.c_char_p, P, F]
    lp.now_save.restype = ct.c_long
    engine = lp.now_init()
    counts = np.zeros(7, "q")
    raw = V4 / (tag + ".bin")
    tags = V4 / (tag + ".tags")
    t0 = time.monotonic()
    seedcount = 0
    log = []

    def prep(r):
        t, j, c, e = r
        return observe(c, e, root, t)

    try:
        with ThreadPoolExecutor(3) as pool:
            q = deque(pool.submit(prep, r) for r in records[:5])
            for j, (t, ci, c, e) in enumerate(records):
                obs, cw, K, dep, ids, col, m = q.popleft().result()
                if j + 5 < len(records):
                    q.append(pool.submit(prep, records[j + 5]))
                head = np.ascontiguousarray(
                    traj("live").T_world_ego(e["t"])[:3, 0], "f"
                )
                lp.now_take(
                    engine,
                    obs.ctypes.data,
                    len(obs),
                    cw.ctypes.data,
                    K.ctypes.data,
                    dep.ctypes.data,
                    ids.ctypes.data,
                    col.ctypes.data,
                    t,
                    ci,
                    e["ordinal"],
                    head.ctypes.data,
                )
                seedcount += len(obs)
                log.append(
                    dict(
                        cam=c,
                        frame=e["n"],
                        ordinal=e["ordinal"],
                        time=e["t"],
                        released=t,
                        instances=m,
                    )
                )
                if j % 240 == 0:
                    print(tag, j, "/", len(records), "seeds", seedcount, flush=True)
        elapsed = time.monotonic() - t0
        num = lp.now_save(
            engine,
            os.fsencode(raw),
            os.fsencode(tags),
            counts.ctypes.data,
            records[-1][0],
        )
        assert num > 0
        if scope == "whole":
            assert num <= 60000000, (
                f"whole-drive budget exceeded: {num} before publication"
            )
    finally:
        lp.now_free(engine)
    aa = np.memmap(raw, "f", "r", shape=(num, 22))
    meta = np.memmap(tags, "i", "r", shape=(num, 3))
    dynamic = meta[:, 2] != 0
    sz = np.where(dynamic, 50, 25)
    xy = np.floor(aa[:, :2] / sz[:, None]).astype("i8")
    key = (xy[:, 0] + 100000) * 200000 + (xy[:, 1] + 100000)
    key += np.where(
        dynamic, (1 + np.floor(aa[:, 15] / 2).astype("i8")) * 50000000000, 0
    )
    order = np.lexsort((meta[:, 2], meta[:, 1], meta[:, 0], key))
    ks, st = np.unique(key[order], return_index=True)
    end = np.r_[st[1:], num]
    obslog = {
        (NAMES.index(e["cam"]), e["ordinal"], m["id"]): m
        for e in log
        for m in e["instances"]
    }
    totalbytes = 0
    for b, h in zip(st, end):
        ix = order[b:h]
        a = aa[ix]
        md = meta[ix]
        x, y = xy[ix[0]]
        dm = dynamic[ix[0]]
        name = f"mov_{int(a[0, 15] // 2)}_{x}_{y}" if dm else f"obj_{x}_{y}"
        n = unit(a[:, 3:6])
        u = unit(a[:, 6:9] - n * np.sum(n * a[:, 6:9], 1)[:, None])
        v = np.cross(n, u)
        quat = Rotation.from_matrix(np.stack((u, v, n), 2)).as_quat()[:, [3, 0, 1, 2]]
        tm = a[:, 15:]
        assert np.all(tm[:, 0] < tm[:, 1]) and np.all(tm[:, 2] <= tm[:, 1])
        A = np.column_stack(
            (
                a[:, :3],
                n,
                (a[:, 12:15] / 255 - 0.5) / 0.28209479,
                np.full(len(ix), 2.45),
                np.log(a[:, 9:12]),
                quat,
                tm,
            )
        ).astype("<f4")
        assert np.isfinite(A).all()
        f = "tiles/" + name + ".ply"
        tmp = (dst / f).with_suffix(".partial")
        with open(tmp, "wb") as fh:
            fh.write(binary_header(len(A)))
            A.tofile(fh)
        os.replace(tmp, dst / f)
        totalbytes += (dst / f).stat().st_size
        tiles.append(
            dict(
                id=name,
                kind="transient" if dm else "static",
                layer="objects",
                file=f,
                count=len(A),
                bbox=np.r_[A[:, :3].min(0), A[:, :3].max(0)].tolist(),
                t_min=float(tm[:, 0].min()),
                t_max=float(min(gm["t_end"], tm[:, 1].max())),
            )
        )
        if dm:
            starts = np.r_[0, 1 + np.flatnonzero(np.any(md[:-1] != md[1:], 1))]
            for lo, hi in zip(starts, np.r_[starts[1:], len(md)]):
                obslog[tuple(md[lo])].setdefault("tile_runs", []).append(
                    dict(tile=name, first=int(lo), count=int(hi - lo))
                )
    claims = sum(
        sum(r["count"] for r in m.get("tile_runs", [])) for m in obslog.values()
    )
    assert claims == int(dynamic.sum())
    instfile = V4 / (
        (
            (tag + "-instances.json" if tag != scope else "instances.json")
            if scope == "slice"
            else "whole-instances.json"
        )
        if cams == CAMS
        else tag + "-instances.json"
    )
    instfile.write_text(
        json.dumps(
            dict(
                scope=scope,
                id_namespace="camera:mask_id; tile_runs tag zero-based contiguous transient PLY rows",
                centroid="coordinate-wise median in ENU metres over all valid instance depth pixels; null = no valid depth",
                frames=log,
            ),
            separators=(",", ":"),
        )
    )
    pan = sky(p, root, dst / "sky.jpg")
    gm.update(tiles=tiles, sky="sky.jpg")
    (dst / "map.partial").write_text(json.dumps(gm, separators=(",", ":")))
    os.replace(dst / "map.partial", dst / "map.json")
    current = {t["file"] for t in tiles}
    for old in (dst / "tiles").glob("*.ply"):
        if "tiles/" + old.name not in current:
            bak = V4 / "previous-tiles"
            bak.mkdir(exist_ok=True)
            os.replace(old, bak / old.name)
    tl = np.load(RS / "data/drive/timeline.npz")
    odo = np.r_[0, np.cumsum(np.hypot(np.diff(tl["x"]), np.diff(tl["y"])))]
    travel = np.interp([records[0][3]["t"], records[-1][3]["t"]], tl["t"], odo)
    factor = odo[-1] / max(0.1, np.diff(travel)[0])
    time_factor = tl["t"][-1] / (records[-1][0] - records[0][0])
    summary = dict(
        scope=scope,
        captures={c: len(es) for c, es in p.items()},
        last_release=records[-1][0],
        first_release=records[0][0],
        records=num,
        movers=claims,
        static=num - claims,
        bytes=totalbytes,
        seeds=seedcount,
        fps=len(records) / elapsed,
        seconds=elapsed,
        events=dict(
            zip(
                [
                    "see_through",
                    "sheet_replaced",
                    "finer",
                    "mover_redraw",
                    "memory_decimated",
                    "continuations",
                    "confirmations",
                ],
                counts.tolist(),
            )
        ),
        whole_distance=int(num * factor),
        whole_time=int(num * time_factor),
        panorama=pan,
    )
    (V4 / (tag + "-summary.json")).write_text(json.dumps(summary, indent=2))
    print(summary, flush=True)


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "slice", sys.argv[2:] or CAMS)
