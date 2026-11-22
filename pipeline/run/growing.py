"""Materialize a whole-drive map and atomically publish geometry or one more clip."""

import os, sys, subprocess, time, urllib.request, hashlib
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from stages import read, cache, HOME
from transactions import atomic_json, content_signature


def clone(src, dst):
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    from transactions import clone as clone_tree
    clone_tree(src, dst)


def models(src, lo, hi, tag, dst):
    from pipeline.road.scene import reader

    bag = []
    for obj in read(src / "objects.json")["objects"]:
        t = np.array(obj["track"]["t"])
        sel = (t >= lo) & (t < hi)
        if sel.sum() < 2:
            continue
        o = dict(obj)
        tr = {
            k: np.array(v)[sel].tolist()
            for k, v in o["track"].items()
            if isinstance(v, list) and len(v) == len(t)
        }
        L, W, H = o["size"]
        if not (
            (0.2 if o["class"] in ("person", "bicycle", "motorcycle") else 0.8) < L < 25
            and 0.35 < W < 4
            and 0.4 < H < 5
            and np.isfinite([L, W, H]).all()
        ):
            raise ValueError(f"invalid physical box: {o['id']}")
        pts = reader(src / o["file"])
        z = pts[:, 2] - pts[:, 2].min()
        side = pts[
            (z >= 0.35 * H)
            & (z <= 0.62 * H)
            & (abs(pts[:, 1]) >= 0.3 * W)
            & (pts[:, 9] >= np.log(0.4 / 0.6))
        ]
        o["paint_samples"] = len(side)
        if len(side) >= 50 and o["class"] not in ("person", "bicycle"):
            rgb = np.clip(0.5 + 0.28209479 * np.median(side[:, 6:9], axis=0), 0, 1)
            o["paint_rgb"] = rgb.tolist()
            # Use a measured name only for unambiguous paint. Otherwise the viewer
            # derives its continuously graded colour directly from the same scan.
            if max(rgb) - min(rgb) < 0.12:
                lum = np.mean(rgb)
                color = (
                    "white"
                    if lum > 0.76
                    else "black"
                    if lum < 0.19
                    else "silver"
                    if lum > 0.53
                    else "grey"
                )
            else:
                idx = np.argsort(rgb)
                color = (
                    ["red", "green", "blue"][idx[-1]]
                    if rgb[idx[-1]] - rgb[idx[-2]] > 0.16
                    else None
                )
            if color:
                o["name"] = color + " " + o["class"]
        o.update(
            id=f"{tag}-{o['id']}", track=tr, t_first=tr["t"][0], t_last=tr["t"][-1]
        )
        for f in ["file", "predicted_file"]:
            if o.get(f):
                dest = "objects/" + tag + "-" + Path(o[f]).name
                if not (dst / dest).exists():
                    os.link(src / o[f], dst / dest)
                o[f] = dest
        bag.append(o)
    return bag


def box_clear(data, objects):
    # Compare each standing splat at its photographic source time (column 19),
    # not to the swept volume of a car at unrelated times. Keep the road itself.
    alive = np.ones(len(data), bool)
    stamp = data[:, 19]
    xyz = data[:, :3]
    lo = xyz.min(0)
    hi = xyz.max(0)
    for o in objects:
        tr = o["track"]
        ts = np.asarray(tr["t"])
        path = np.column_stack([tr[k] for k in ("x", "y", "z")])
        L, W, H = o["size"]
        pad = np.array([max(L, W), max(L, W), H])
        if (path.max(0) + pad < lo).any() or (path.min(0) - pad > hi).any():
            continue
        ii = np.flatnonzero(alive & (stamp >= ts[0]) & (stamp <= ts[-1]))
        tt = stamp[ii]
        if not len(ii):
            continue
        delta = xyz[ii] - np.column_stack(
            [np.interp(tt, ts, path[:, j]) for j in range(3)]
        )
        yaw = np.interp(tt, ts, np.unwrap(tr["yaw"]))
        c, s = np.cos(yaw), np.sin(yaw)
        x, y, z = delta.T
        hit = (
            (abs(c * x + s * y) < L / 2)
            & (abs(c * y - s * x) < W / 2)
            & (z > 0.15)
            & (z < H)
        )
        alive[ii[hit]] = False
    return alive


def observed_atlas(path):
    import cv2 as cv

    j = read(path / "ground.json")
    for t in j["tiles"]:
        if t.get("provenance"):
            continue
        # Our base atlas only samples actual photos, and zeros texels without a
        # source label; it has no inpainted layer. Label opaque witnesses only.
        rgba = cv.imread(str(path / t["texture"]), -1)
        alpha = (
            rgba[:, :, 3]
            if rgba.shape[2] == 4
            else np.full(rgba.shape[:2], 255, "uint8")
        )
        lab = (alpha >= 250).astype("uint8") * 255
        f = t["texture"] + ".source.png"
        assert cv.imwrite(str(path / f), lab)
        t["provenance"] = f
    atomic_json(path / "ground.json", j)


def merge(req, out):
    from pipeline.road.scene import reader, binary_header

    r = Path(req["root"])
    src = r / "scene/road-stitched-v2"
    wins = req["windows"]
    out.mkdir()
    (out / "tiles").mkdir()
    (out / "objects").mkdir()
    meta = read(src / "map.json")
    tiles = []
    banks = {}
    trees = []
    for f in ["rig.json", "drive.json"]:
        clone(src / f, out / f)
    clone(Path(wins[-1]["path"]) / "sky.jpg", out / "sky.jpg")
    clone(r / "work/run/atlas", out / "ground")
    observed_atlas(out / "ground")
    d = read(out / "drive.json")
    t = np.array(d["t"])
    xy = np.c_[d["x"], d["y"]]
    movers = []
    prev = -float("inf")
    preview = r / "work/run"
    for i, w in enumerate(wins):
        a, b = w["window"]
        p = Path(w["path"])
        trees.append(cKDTree(xy[(t >= a) & (t <= b)]))
        movers += models(preview, prev, a, f"p{i}", out)
        movers += models(p, max(prev, a), b + 1e-8, f"w{i}", out)
        prev = b + 1e-8
        for tile in read(p / "map.json")["tiles"]:
            if tile["layer"] == "objects":
                banks.setdefault(tile["id"], []).append((i, p, tile))
    movers += models(preview, prev, float("inf"), "preview", out)
    atomic_json(out / "objects.json", dict(objects=movers))
    for e in meta["tiles"]:
        if e["layer"] == "ground":
            os.link(src / e["file"], out / e["file"])
            tiles.append(e)
    removed = 0
    for tid, entries in banks.items():
        e = entries[0][2]
        dst = out / e["file"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        chunks = []
        for k, (idx, p, en) in enumerate(entries):
            arr = reader(p / en["file"])
            keep = box_clear(arr, movers)
            removed += int((~keep).sum())
            arr = arr[keep]
            if len(entries) > 1 and len(arr):
                owner = np.stack(
                    [trees[i].query(arr[:, :2])[0] for i, _, _ in entries]
                ).argmin(0)
                arr = arr[owner == k]
            chunks.append(arr)
        arr = np.concatenate(chunks)
        if not len(arr):
            continue
        with dst.open("wb") as f:
            f.write(binary_header(len(arr)))
            arr.astype("<f4", copy=False).tofile(f)
        tiles.append(
            dict(
                e,
                count=len(arr),
                bbox=np.r_[arr[:, :3].min(0), arr[:, :3].max(0)].tolist(),
                t_min=float(arr[:, 17].min()),
            )
        )
    atomic_json(
        out / "map.json",
        dict(meta, sky="sky.jpg", tiles=tiles, windows=[w["window"] for w in wins]),
    )
    atomic_json(
        out / "vehicle-audit.json",
        dict(
            tracks=len(movers),
            named=sum(bool(o.get("name")) for o in movers),
            with_scan_paint=sum(o["paint_samples"] >= 50 for o in movers),
            source_time_box_centres_removed=removed,
            rule="objects only; capture time within tracked oriented body, 0.15m above contact to measured roof",
        ),
    )
    assert len(set(e["id"] for e in tiles)) == len(tiles)


def publish(request):
    name = os.environ["RUN_NAME"]
    parent = Path(os.environ["RUN_RELEASES"])
    public = Path(request["public"])
    src = Path(request["scene"])
    dest = HOME / "scene" / name
    last = dest.resolve(strict=True)
    if not dest.is_symlink() or last.parent != parent:
        raise ValueError("expected this drive’s verified preview or earlier release")
    cliponly = (last / "geometry.json").is_file() and read(last / "geometry.json")[
        "source"
    ] == str(src)
    out = parent / f"progress-{time.time_ns()}"
    clone(last if cliponly else src, out)
    atomic_json(out / "geometry.json", dict(source=str(src)))
    entries = []
    for clip in request["clips"]:
        e = clip["entry"]
        p = Path(clip["scratch"]) / "bundle" / e["file"]
        entries.append(e)
        (out / "generated").mkdir(exist_ok=True)
        if not (out / e["file"]).exists():
            os.link(p, out / e["file"])
    if entries:
        atomic_json(out / "generated.json", dict(clips=entries))
    elif (out / "generated.json").exists():
        (out / "generated.json").unlink()
    cache(out)
    idx = read(out / "rsb/index.json")
    tiles = read(out / "map.json")["tiles"]
    assert idx["generated"] == bool(entries) and idx["ground"] and idx["objects"]
    assert len(idx["tiles"]) == len(tiles) and all(
        idx["tiles"][t["id"]]["count"] == t["count"]
        and (out / "rsb" / idx["tiles"][t["id"]]["file"]).stat().st_size
        == idx["tiles"][t["id"]]["bytes"]
        for t in tiles
    )
    tmp = dest.with_name("." + name + "-next")
    tmp.unlink(missing_ok=True)
    tmp.symlink_to(out)
    tmp.replace(dest)
    host = os.environ.get("RS_VIEWER_URL", "http://127.0.0.1:8090")
    try:
        fs = [
            "map.json",
            "rsb/index.json",
            "objects.json",
            "ground/ground.json",
            "rsb/" + idx["tiles"][tiles[-1]["id"]]["file"],
        ] + (["generated.json", entries[-1]["file"]] if entries else [])
        for f in fs:
            with (
                urllib.request.urlopen(
                    host + "/scenes/" + name + "/" + f, timeout=45
                ) as resp,
                (out / f).open("rb") as local,
            ):
                assert (
                    hashlib.file_digest(resp, "sha256").digest()
                    == hashlib.file_digest(local, "sha256").digest()
                ), f
    except BaseException:
        tmp.symlink_to(last)
        tmp.replace(dest)
        raise
    receipt = dict(
        directory=str(out),
        name=name,
        url=host + "/?scene=" + name + "&mode=stitched",
        content_sha256=content_signature(out),
        windows=read(out / "map.json")["windows"],
        clips=entries,
        tier="progressive-full",
    )
    atomic_json("publication.json", receipt)
    atomic_json(public / "work/run/published-full.json", receipt)


if __name__ == "__main__":
    if sys.argv[1] == "clip":
        from clips import render

        render(
            Path(sys.argv[5]),
            Path(sys.argv[2]),
            float(sys.argv[3]),
            float(sys.argv[4]),
            None if sys.argv[6] == "-" else sys.argv[6],
        )
    elif sys.argv[1] == "merge":
        req = read("request.json")
        os.environ["RS"] = req["root"]
        tmp = Path(f".scene-{time.time_ns()}")
        merge(req, tmp)
        dst = Path("scene")
        if dst.exists():
            dst.rename(f".superseded-{time.time_ns()}")
        tmp.rename(dst)
    elif sys.argv[1] == "publish":
        publish(read("request.json"))
