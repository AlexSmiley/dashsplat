"""Combine frozen v2 ground with tiled DC surfels; keep original camera/drive contracts."""

from pathlib import Path
from .prepare import *
import ctypes as ct, os, time, argparse
from concurrent.futures import ThreadPoolExecutor
from collections import deque, Counter
from scipy.spatial.transform import Rotation
from pipeline.road.scene import binary_header


def scene(mode, stage):
    return (
        W / "stages" / f"{mode}-prefix"
        if "LIVEMAP_CUTOFF" in os.environ
        else (
            RS / "scene" / f"{mode}-v3"
            if stage == "final"
            else W / "stages" / f"{mode}-{stage}"
        )
    )


def link(src, dst):
    if not dst.exists():
        os.link(src, dst)


def build(mode, stage):
    tstart = time.monotonic()
    groundroot = RS / "scene" / f"road-{mode}-v2"
    dst = scene(mode, stage)
    (dst / "tiles").mkdir(parents=True, exist_ok=True)
    gm = json.loads((groundroot / "map.json").read_text())
    tiles = list(gm["tiles"])
    for tile in tiles:
        link(groundroot / tile["file"], dst / tile["file"])
    for fn in ["rig.json", "drive.json"]:
        link(groundroot / fn, dst / fn)
    names = {"front": ["front"], "sides": CAMS[:-1], "back": CAMS, "final": CAMS}[stage]
    end = float(os.environ.get("LIVEMAP_CUTOFF", 110 if stage == "front" else 140))
    plan = frames()
    desc = json.loads((W / mode / "depth/keyframes.json").read_text())["depth"]
    expected = RS / "poses" / ("ego_causal.npz" if mode == "live" else "ego.npz")
    origin = Path(desc["trajectory"])
    assert os.path.samefile(
        origin if origin.is_absolute() else W / mode / origin, expected
    )
    assert (
        desc["model"] == "da3-large-1.1"
        and desc["pose_conditioned"]
        and desc["scale_factor"] == 1
        and desc["scale_correction"] == "none"
        and desc["window"] == 5
        and desc["window_multiview"] == 3
        and desc["quantity"].startswith("z:")
    )
    records = []
    policies = {
        c: json.loads(
            (
                W
                / mode
                / (
                    c
                    + (
                        "-prefix-policy.json"
                        if "LIVEMAP_CUTOFF" in os.environ
                        else "-causal-policy.json"
                        if mode == "live"
                        else "-policy.json"
                    )
                )
            ).read_text()
        )
        for c in names
    }

    def policy(c, e):
        return policies[c][str(e["n"])]

    for c in names:
        idx = json.loads((W / mode / "depth" / c / "index.json").read_text())
        hold = {e["n"] for e in plan[c] if e["hold"]}
        rows0 = [e for e in plan[c] if e["train"] and 85 <= e["t_key"] <= end]
        for e in rows0:
            path = W / mode / "depth" / c / f"{e['n']:06d}.npy"
            key = str(e["n"])
            if not path.exists() or key not in idx:
                if stage == "final":
                    raise ValueError(f"missing {mode} {c} {key}")
                continue
            assert idx[key]["window"] == e["window"] and not (hold & set(e["window"]))
            assert (
                idx[key]["posed"]
                and idx[key]["model"] == desc["model"]
                and idx[key]["scale_applied"] == 1
                and abs(idx[key]["t"] - e["t"]) < 0.00001
            )
            stamp = ready(mode, c, e)
            if "LIVEMAP_CUTOFF" not in os.environ or stamp <= end:
                records.append((stamp, c, e))
    records.sort(key=lambda v: v[0])
    assert records
    lib = ct.CDLL(str(W / "fusion.dylib"))
    P = ct.c_void_p
    I = ct.c_int
    F = ct.c_float
    lib.init.argtypes = [I]
    lib.init.restype = P
    lib.destroy.argtypes = [P]
    lib.take.argtypes = [P, P, I, P, P, P, P, P, F, F, F, I, I, F]
    lib.output.argtypes = [P, ct.c_char_p, ct.c_char_p, P]
    lib.output.restype = ct.c_long
    g = lib.init(mode == "live")
    outtag = "prefix" if "LIVEMAP_CUTOFF" in os.environ else stage
    raw = W / mode / f"{outtag}.vertices"
    side = W / mode / f"{outtag}.extra"
    counts = np.zeros(8, np.int64)
    camframes = Counter()
    ts = time.monotonic()
    cand = 0
    native = 0.0

    def prep(r):
        t, c, e = r
        return packet(mode, c, e, policy(c, e))

    try:
        with ThreadPoolExecutor(3) as pool:
            queue = deque(pool.submit(prep, r) for r in records[:4])
            for j, (t, c, e) in enumerate(records):
                data, cw, dep, mn, app, K, now = queue.popleft().result()
                if j + 4 < len(records):
                    queue.append(pool.submit(prep, records[j + 4]))
                q = time.monotonic()
                lib.take(
                    g,
                    data.ctypes.data,
                    len(data),
                    cw.ctypes.data,
                    K.ctypes.data,
                    dep.ctypes.data,
                    mn.ctypes.data,
                    app.ctypes.data,
                    now,
                    e["t"] + rig[c].time_offset_s,
                    rig[c].rs_readout_s or 0,
                    NAMES.index(c),
                    e["ordinal"],
                    policy(c, e)["max_range"],
                )
                native += time.monotonic() - q
                cand += len(data)
                camframes[c] += 1
        wall = time.monotonic() - ts
        n = lib.output(g, os.fsencode(raw), os.fsencode(side), counts.ctypes.data)
    finally:
        lib.destroy(g)
    assert n > 0
    aa = np.memmap(raw, dtype="f", mode="r", shape=(n, 23))
    xyz = aa[:, :3]
    idx = np.floor(xyz[:, :2] / 25).astype("i")
    key = idx[:, 0] * 100000 + idx[:, 1]
    order = np.argsort(key, kind="stable")
    keys, starts = np.unique(key[order], return_index=True)
    ends = np.r_[starts[1:], n]
    totalbytes = 0
    export_ids = np.full(n, -1, np.int32)
    for k, b, e in zip(keys, starts, ends):
        a = aa[order[b:e]].copy()
        limit = body_clearance(mode, a[:, :3])
        body = limit < INF
        mask = (
            (~body) if mode == "stitched" else (np.minimum(a[:, 17], limit) > a[:, 16])
        )
        a = a[mask]
        limit = limit[mask]
        if not len(a):
            continue
        a[:, 17] = np.minimum(a[:, 17], np.where(limit < INF, limit + 0.25, INF))
        a[:, 18] = np.minimum(a[:, 18], a[:, 17])
        for j in (19, 21):
            absent = a[:, j] >= a[:, 17]
            a[absent, j : j + 2] = INF
            a[~absent, j + 1] = np.minimum(a[~absent, j + 1], a[~absent, 17])
        export_ids[order[b:e][mask]] = np.arange(len(a))
        nx = unit(a[:, 3:6])
        uu = unit(a[:, 6:9] - nx * np.sum(nx * a[:, 6:9], axis=1)[:, None])
        vv = np.cross(nx, uu)
        q = Rotation.from_matrix(np.stack((uu, vv, nx), axis=2)).as_quat()
        x, y = idx[order[b]]
        tag = f"obj_{x}_{y}"
        A = np.ascontiguousarray(
            np.column_stack(
                (
                    a[:, :3],
                    nx,
                    (a[:, 12:15] / 255 - 0.5) / 0.28209479,
                    np.log(a[:, 15] / (1 - a[:, 15])),
                    np.log(a[:, 9:12]),
                    q[:, [3, 0, 1, 2]],
                    a[:, 16:23],
                )
            ),
            "<f4",
        )
        assert np.isfinite(A).all()
        file = "tiles/" + tag + ".ply"
        p = dst / file
        tmp = p.with_suffix(".partial")
        with open(tmp, "wb") as f:
            f.write(binary_header(len(A)))
            A.tofile(f)
        os.replace(tmp, p)
        totalbytes += p.stat().st_size
        tiles.append(
            dict(
                id=tag,
                kind="static",
                layer="objects",
                file=file,
                count=len(A),
                bbox=np.r_[A[:, :3].min(0), A[:, :3].max(0)].tolist(),
                t_min=float(A[:, 17].min()),
                t_max=gm["t_end"],
            )
        )
    # Extra original visibility runs (stitched) mapped to the actual tile and record index.
    extra = np.fromfile(side, dtype=[("record", "<i4"), ("a", "<f4"), ("b", "<f4")])
    if len(extra):
        extra = extra[export_ids[extra["record"]] >= 0]
        ix = extra["record"]
        pos = export_ids[ix]
        np.savez(
            W / mode / f"{stage}-extra-runs.npz",
            tile_xy=idx[ix],
            index=pos,
            start=extra["a"],
            end=extra["b"],
        )
    drive = np.load(RS / "data/drive/timeline.npz")
    t = drive["t"]
    s = np.r_[
        0,
        np.cumsum(
            np.maximum(0, (drive["speed"][:-1] + drive["speed"][1:]) / 2) * np.diff(t)
        ),
    ]
    km = np.interp([85, end], t, s)
    factor = float(s[-1] / np.diff(km)[0])
    n0 = n
    metadata = np.memmap(str(raw) + ".meta", np.uint8, mode="r", shape=(n, 4))[
        export_ids >= 0
    ]
    lands = np.flatnonzero(export_ids >= 0)[metadata[:, 1] > 0]
    np.savez(
        W / mode / (outtag + "-terrain.npz"),
        tile_xy=idx[lands],
        index=export_ids[lands],
    )
    n = len(metadata)
    ms = dict(
        trajectory=str(expected.relative_to(RS)),
        release_range=[records[0][0], records[-1][0]],
        release_lag_max=max(stamp - e["t"] for stamp, _, e in records),
        source_records={
            c: int((metadata[:, 0] == NAMES.index(c)).sum()) for c in names
        },
        terrain=int(metadata[:, 1].sum()),
        volumetric=int(metadata[:, 2].sum()),
        provisional=int((metadata[:, 3] & 1).sum()),
        hull_pruned=n0 - n,
        mode=mode,
        stage=stage,
        interval=[85, end],
        actual=[
            min(e["t_key"] for _, _, e in records),
            max(e["t_key"] for _, _, e in records),
        ],
        frames=dict(camframes),
        candidates=cand,
        splats=n,
        tiles=len(tiles)
        - len(json.loads((groundroot / "map.json").read_text())["tiles"]),
        object_ply_bytes=totalbytes,
        ground_ply_bytes=sum((dst / x["file"]).stat().st_size for x in gm["tiles"]),
        fusion_seconds=wall,
        fps=len(records) / wall,
        native_seconds=native,
        build_seconds=time.monotonic() - tstart,
        travel_factor=factor,
        projected_whole_drive=n * factor,
        events=dict(
            zip(
                (
                    "promotions",
                    "confirms",
                    "see_through",
                    "false_retired",
                    "proposals",
                    "superseded",
                    "continuations",
                    "terrain_records",
                ),
                map(int, counts),
            )
        ),
        extra_runs=len(extra),
    )  # Retain previous generated tiles outside the published set in staging.
    keep = {a["file"] for a in tiles}
    backup = W / mode / "previous"
    backup.mkdir(exist_ok=True)
    for old in (dst / "tiles").glob("obj_*.ply"):
        if "tiles/" + old.name not in keep:
            os.replace(old, backup / (stage + "-" + old.name))
    gm.update(tiles=tiles)
    (dst / "map.partial").write_text(json.dumps(gm, separators=(",", ":")))
    os.replace(dst / "map.partial", dst / "map.json")
    (W / mode / (outtag + "-summary.json")).write_text(json.dumps(ms, indent=2))
    print(ms, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode")
    p.add_argument("stage")
    a = p.parse_args()
    build(a.mode, a.stage)
