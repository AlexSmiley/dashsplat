"""Attach independently built movers to a specific static scene, retaining its sky and cache."""

import argparse, json, os, pickle
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from .inventory import W
from .publish import source_tiles

R = W / os.environ.get("MOVERS_RUN", "round2")
R.mkdir(exist_ok=True)


def link(a, b):
    b.parent.mkdir(parents=True, exist_ok=True)
    temp = b.with_suffix(".next")
    temp.unlink(missing_ok=True)
    if b.exists() and os.path.samefile(a, b):
        return
    os.link(a, temp)
    os.replace(temp, b)


def road_check(src, o):
    tr = o["track"]
    xy = np.c_[tr["x"], tr["y"]]
    size = o["size"]
    yaw = np.array(tr["yaw"])
    wheels = np.array(
        [[a * size[0], b * size[1]] for a in [-0.315, 0.31] for b in [-0.49, 0.49]]
    )
    cs, sn = np.cos(yaw), np.sin(yaw)
    qx = xy[:, 0, None] + cs[:, None] * wheels[:, 0] - sn[:, None] * wheels[:, 1]
    qy = xy[:, 1, None] + sn[:, None] * wheels[:, 0] + cs[:, None] * wheels[:, 1]
    Q = np.stack([qx, qy], -1)
    lo = Q.min((0, 1)) - 3
    hi = Q.max((0, 1)) + 3
    pts = []
    for v in json.load(open(src / "map.json"))["tiles"]:
        bb = v["bbox"]
        if (
            v.get("layer") != "ground"
            or np.any(np.array(bb[:2]) > hi)
            or np.any(np.array(bb[3:5]) < lo)
        ):
            continue
        header, blob = (src / v["file"]).read_bytes().split(b"end_header\n", 1)
        stride = sum(x.startswith(b"property float ") for x in header.splitlines())
        p = np.frombuffer(blob, "<f4").reshape(-1, stride)[:, :3]
        pts.append(p[((p[:, :2] >= lo) & (p[:, :2] <= hi)).all(1)])
    p = np.concatenate(pts) if pts else np.empty((0, 3))
    res = dict(samples=len(xy), supported_wheels=0, total_wheels=4 * len(xy))
    if len(p) < 8:
        return res
    dist, ix = cKDTree(p[:, :2]).query(Q, k=8)
    ok = dist[:, :, -1] < 1.0
    road = np.median(p[ix, 2], axis=-1)
    res["supported_wheels"] = int(ok.sum())
    z = np.array(tr["z"])
    if ok.any():
        dz = road - z[:, None]
        valid = ok.sum(1) >= 2
        if valid.any():
            corr = np.array([np.median(v[o]) for v, o in zip(dz[valid], ok[valid])])
            dt = np.array(tr["t"])
            adj = np.interp(dt, dt[valid], corr)
            close = np.min(abs(dt[:, None] - dt[valid]), axis=1)
            adj *= np.clip(1 - close / 0.4, 0, 1)
            adj = np.convolve(np.pad(adj, (3, 3), mode="edge"), np.ones(7) / 7, "valid")
            z += np.clip(adj, -0.35, 0.35)
            tr["z"] = np.round(z, 6).tolist()
        resid = (z[:, None] - road)[ok]
        res.update(
            before_abs_p50_m=float(np.median(abs(dz[ok]))),
            after_signed_p50_m=float(np.median(resid)),
            after_abs_p90_m=float(np.percentile(abs(resid), 90)),
            nearest_p50_m=float(np.median(dist[:, :, 0])),
        )
    return res


def bus_height(o):
    """Fit a smooth height correction from visible near-side tyre / mask contacts.
    Uses fit exposures only; do not move the bus laterally to meet a tree-split mask.
    """
    import cv2
    from pipeline.common.rig import load_rig, Trajectory
    from .model import projection

    _, rows, *_ = pickle.load(
        open(
            (R / f"fit-{o['id']}.pickle")
            if R.name.startswith("photo-demo")
            else W / f"stitched-{o['id']}.pickle",
            "rb",
        )
    )
    times = []
    dz = []
    cs = load_rig()
    ego = Trajectory("poses/ego.npz")
    sz = np.array(o["size"])
    w = np.array(
        [[a * sz[0], b * sz[1], 0.0] for a in [-0.315, 0.31] for b in [-0.49, 0.49]]
    )
    pts = np.vstack([w, w + [0, 0, 0.15]])
    tr = o["track"]
    for r in rows:
        if (
            not r["key"]
            or r.get("held")
            or r["c"] not in ("left_pillar", "left_repeater")
            or r["bb"][2] - r["bb"][0] < 95
        ):
            continue
        c = r["c"]
        uv, ok, _, dep = projection(cs[c], ego, r["t"], tr, pts, pts * 0)
        mask = cv2.imread(f"masks/dyn/{c}/{r['n']:06}.png", -1) == r["id"]
        side = 0 if np.mean(dep[[0, 2]]) < np.mean(dep[[1, 3]]) else 1
        for j in [side, side + 2]:
            x = int(uv[j, 0])
            coef = (uv[j + 4, 1] - uv[j, 1]) / 0.15
            if x < 6 or x > 717 or coef > -0.8:
                continue
            ys = np.where(mask[:, x - 2 : x + 3].any(1))[0]
            if not len(ys) or abs((ys[-1] + 0.5) - uv[j, 1]) > 8:
                continue
            delta = (ys[-1] + 0.5 - uv[j, 1]) / coef
            dz.append(np.clip(delta, -0.45, 0.45))
            times.append(r["rt"])
    tt = np.array(tr["t"])
    times = np.array(times)
    dz = np.array(dz)
    off = []
    for t in tt:
        nearby = abs(times - t) < 0.48
        off.append(np.median(dz[nearby]) if nearby.any() else 0.0)
    off = np.convolve(np.pad(off, (8, 8), mode="edge"), np.ones(17) / 17, "valid")
    tr["z"] = np.round(np.array(tr["z"]) + off, 6).tolist()
    return dict(fit_wheel_contacts=len(dz), max_vertical_adjust_m=float(max(abs(off))))


def attach(parent, source, dest, windows=None, assets=None):
    src = Path("scene") / parent
    dst = Path("scene") / dest
    dst.mkdir(exist_ok=True)
    m = json.load(open(src / "map.json"))
    start, end = m.get("slice_s", [0, 1e9])
    objs = []
    audit = {}
    for o in json.load(open(source / "objects.json"))["objects"]:
        if (
            o["t_last"] < start
            or o["t_first"] > end
            or (
                windows
                and not any(o["t_first"] <= b and o["t_last"] >= a for a, b in windows)
            )
        ):
            continue
        for key in ("file", "predicted_file"):
            if o.get(key):
                link((assets or source) / o[key], dst / o[key])
        support = (
            bus_height(o)
            if parent.startswith("stitched") and o["class"] == "bus"
            else {}
        )
        audit[str(o["id"])] = {
            **support,
            **(road_check(src, o) if parent.startswith("stitched") else {}),
        }
        objs.append(o)
    # Link the immutable base map without bringing in other ground-revision tiles.
    source_tiles("copy", src, dst, [])
    (dst / "objects.json").write_text(
        json.dumps({"objects": objs}, separators=(",", ":"))
    )
    (R / (dest + "-contact.json")).write_text(json.dumps(audit, indent=2))
    print(dest, len(objs), "objects", len(m["tiles"]), "tiles", m["sky"], flush=True)


def targets():
    for p in sorted(Path("scene").glob("stitched-opt-*")):
        if not (p / "map.json").exists() or "movers" in p.name:
            continue
        spans = json.load(open(p / "map.json")).get("slice_s")
        if p.name == "stitched-opt-v3" or (
            spans
            and any(spans[0] < b and spans[1] > a for a, b in [(190, 215), (286, 330)])
        ):
            yield p.name


def run(upgrade=False):
    src = Path("scene/stitched-opt-v2-movers-v1")
    assets = R / "hindsight" if upgrade else None
    if upgrade:
        src = assets
    for parent in targets():
        attach(parent, src, parent + "-movers-v1", assets=assets)
    if upgrade:
        attach(
            "live-v5",
            src,
            "live-v5-traffic-hindsight",
            [(190, 215), (286, 330)],
            assets,
        )


def follow_later(minutes):
    """On Studio, wait for two complete, cached optimized windows (bounded)."""
    import select, subprocess, time

    deadline = time.monotonic() + 60 * minutes
    done = {}
    watched = {}
    notify = select.kqueue()
    sky = R / "hindsight"
    cache = Path("viewer-cache-tool/tools/build-tile-cache.ts").resolve()
    windows = [(190, 215), (286, 330)]

    def watch(p):
        if p in watched or not p.is_dir():
            return
        fd = os.open(p, os.O_RDONLY)
        notify.control(
            [
                select.kevent(
                    fd,
                    filter=select.KQ_FILTER_VNODE,
                    flags=select.KQ_EV_ADD | select.KQ_EV_CLEAR,
                    fflags=select.KQ_NOTE_WRITE | select.KQ_NOTE_RENAME,
                )
            ],
            0,
            0,
        )
        watched[p] = fd

    try:
        while time.monotonic() < deadline and len(done) < 2:
            watch(Path("scene"))
            for src in sorted(Path("scene").glob("stitched-opt-*")):
                if "movers" in src.name:
                    continue
                watch(src)
                watch(src / "rsb")
                try:
                    m = json.load(open(src / "map.json"))
                    span = m.get("slice_s", [])
                    todo = [
                        (a, b)
                        for a, b in windows
                        if str(a) not in done
                        and len(span) == 2
                        and span[0] <= a
                        and span[1] >= b
                    ]
                    if not todo:
                        continue
                    idx = json.load(open(src / "rsb/index.json"))
                    ready = bool(m["tiles"]) and all(
                        v["id"] in idx["tiles"]
                        and (src / v["file"]).is_file()
                        and (src / "rsb" / idx["tiles"][v["id"]]["file"]).is_file()
                        for v in m["tiles"]
                    )
                    if not ready or (m.get("sky") and not (src / m["sky"]).is_file()):
                        continue
                except (OSError, ValueError, KeyError):
                    continue
                dest = src.name + (
                    "-movers-photo-v1"
                    if R.name.startswith("photo-demo")
                    else "-movers-v1"
                )
                attach(src.name, sky, dest, assets=sky)
                p = src.parent / dest
                subprocess.run(["bun", str(cache), str(p.resolve())], check=True)
                assert json.load(open(p / "rsb/index.json"))["objects"]
                oo = json.load(open(p / "objects.json"))["objects"]
                assert all(
                    (p / o[k]).is_file() for o in oo for k in ["file", "predicted_file"]
                )
                for a, b in todo:
                    done[str(a)] = dict(
                        parent=src.name,
                        scene=dest,
                        objects=sum(o["t_first"] <= b and o["t_last"] >= a for o in oo),
                        tiles=len(m["tiles"]),
                    )
                (R / "late-parents.json").write_text(json.dumps(done, indent=2))
                print(done, flush=True)
            if len(done) < 2:
                notify.control([], 1, max(0.01, deadline - time.monotonic()))
    finally:
        for fd in watched.values():
            os.close(fd)
        notify.close()
    return done


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--upgrade", action="store_true")
    p.add_argument("--follow-later", type=int, metavar="MINUTES")
    a = p.parse_args()
    if a.follow_later:
        follow_later(a.follow_later)
    else:
        run(a.upgrade)
