"""Camera events in readout order. Mask IDs identify pixels, not world identities."""

import json, pickle
from pathlib import Path
import cv2, numpy as np
from scipy.spatial.transform import Rotation
from pipeline.common.rig import Trajectory, frame_table, load_rig
from .observations import read
from .inventory import W, SIZE, WIN
from .demo import R

cv2.setNumThreads(1)


def iou(a, b):
    a, b = np.asarray(a), np.asarray(b)
    i = np.prod(np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])))
    return i / max(1, np.prod(a[2:] - a[:2]) + np.prod(b[2:] - b[:2]) - i)


def held_schedule():
    # Only this benchmark adapter reads last round's labels. It removes pixels;
    # the discovery/filter process is passed no subjects, truth IDs or held boxes.
    from collections import defaultdict

    out = defaultdict(list)
    for p in W.glob("stitched-*.pickle"):
        s, rows, *_ = pickle.load(open(p, "rb"))
        for v in rows:
            if v.get("held"):
                out[(v["c"], v["n"])].append(np.array(v["bb"]))
    assert sum(map(len, out.values())) == 136
    return out


def build(holdout=False, windows=WIN):
    cs = load_rig()
    tr = Trajectory("poses/ego_causal.npz")
    rdy = tr.t + np.load("work/road/causal-lag.npy")
    maskout = held_schedule() if holdout else {}
    events = []
    counts = {}
    sources = {}
    people = []
    for c, cam in cs.items():
        table = frame_table(cam)
        if R.name.startswith("photo-"):
            table = {v["n"]: v for v in table}
        paths = {}
        for root in (W / "extra", Path(".")):
            for p in (root / "masks/dyn" / c).glob("[0-9]*.json"):
                n = int(p.stem)
                if R.name.startswith("photo-") and (
                    n not in table or not p.with_suffix(".png").is_file()
                ):
                    continue
                t = table[n]["t"]
                if any(a <= t <= b for a, b in windows) and (
                    root == Path(".") or 190 <= t <= 215
                ):
                    paths[n] = (root, p)
        sources[c] = {str(n): str(root) for n, (root, _) in paths.items()}
        for n, (root, p) in sorted(paths.items()):
            tf = table[n]["t"]
            tm = tf + cam.time_offset_s
            now = max(tf, tm + abs(cam.rs_readout_s) / 2)
            entities = json.loads(p.read_text())
            counts[(c, int(next(a for a, b in windows if a <= tf <= b)))] = (
                counts.get((c, int(next(a for a, b in windows if a <= tf <= b))), 0) + 1
            )
            if not entities:
                continue
            m = cv2.imread(str(p.with_suffix(".png")), -1)
            im = read(c, n)
            lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
            # Latest released samples only. Extrapolate previous ego displacement to
            # this shutter rather than waiting ~0.3 s for a new refined ego fit.
            k = np.searchsorted(tr.t, now, "right") - 1
            while max(rdy[k - 1 : k + 1]) > now:
                k -= 1
            vel = (tr.p[k] - tr.p[k - 1]) / tr.dt[k - 1]
            omega = (
                Rotation.from_matrix(tr.R[k] @ tr.R[k - 1].T).as_rotvec() / tr.dt[k - 1]
            )
            e = tr.poses[k].copy()
            dt = tm - tr.t[k]
            e[:3, 3] += dt * vel
            e[:3, :3] = Rotation.from_rotvec(dt * omega).as_matrix() @ e[:3, :3]
            C = e @ cam.T_ego_cam
            ne, h = tr.ground_plane(tr.t[k])
            nw = e[:3, :3] @ ne
            plane = np.r_[
                -nw[:2] / nw[2], e[2, 3] + nw[:2] @ e[:2, 3] / nw[2] - h / nw[2]
            ]
            obs = []
            for ent in entities:
                cls = ent["class"]
                bb = np.array(ent["bbox"], float) + 0.5
                area = ent["area"]
                l, t, r, b = ent["bbox"]
                if (
                    cls in ("person", "rider", "motorcycle", "bicycle")
                    and 190 <= tf <= 215
                ):
                    people.append(
                        [c, n, tf, ent["id"], cls, area, ent["bbox"], str(root)]
                    )
                if (
                    cls not in SIZE
                    or area < 30
                    or r - l < 4
                    or b - t < 5
                    or any(iou(bb, v) > 0.42 for v in maskout.get((c, n), []))
                ):
                    continue
                sub = m[t:b, l:r] == ent["id"]
                yy, xx = np.where(sub)
                if len(yy) < 25:
                    continue
                yy = yy + t + 0.5
                xx = xx + l + 0.5
                bottom = np.percentile(yy, 98.8)
                foot = [np.median(xx[yy >= np.percentile(yy, 95)]), bottom]
                uv = np.array(
                    [foot, [(l + r) / 2, t + 0.5], [(l + r) / 2, (b + t) / 2]]
                ) * [cam.width / 724, cam.height / 469]
                rays = cam.unproject(uv) @ C[:3, :3].T
                O = C[:3, 3]
                direction = rays[2, :2] / max(1e-9, np.linalg.norm(rays[2, :2]))
                floor = -(nw @ (O - e[:3, 3]) + h) / min(-0.001, nw @ rays[0])
                rg = floor * np.linalg.norm(rays[0, :2])
                elev = rays[:, 2] / np.maximum(
                    0.03, np.linalg.norm(rays[:, :2], axis=1)
                )
                rh = SIZE[cls][2] / max(0.008, elev[1] - elev[0])
                valid = 2 < rg < 145
                dh = np.clip(rh, 2, 140)
                w = 0.68 if valid and rg < 38 else 0.34 if valid else 0
                range_ = w * rg + (1 - w) * dh + 0.75
                if range_ > 120 or range_ < 2.3 or (r - l) > 600 or (b - t) > 360:
                    continue
                color = np.median(lab[t:b, l:r][sub][:: max(1, area // 100)], axis=0)
                xy = O[:2] + direction * range_
                rec = dict(
                    c=c,
                    n=n,
                    t=tf,
                    rt=float(cam.row_time(tf, foot[1] * cam.height / 469)),
                    release=now,
                    id=ent["id"],
                    cls=cls,
                    score=ent["score"],
                    bb=bb,
                    foot=foot,
                    depth=0,
                    key=True,
                    area=area,
                    xy=xy,
                    dir=direction,
                    range=range_,
                    root=str(root),
                    color=color,
                    plane=plane,
                    C=C,
                    velocity=vel + np.cross(omega, O - e[:3, 3]),
                    omega=omega,
                    ego_yaw=np.arctan2(e[1, 0], e[0, 0]),
                    pose_k=int(k),
                )
                obs.append(rec)
            if obs:
                events.append((now, c, n, obs))
    events.sort(key=lambda v: (v[0], v[1], v[2]))
    pickle.dump(events, open(R / "packets.pickle", "wb"))
    if R.name.startswith("photo-full"):
        (R / "mask-sources.next").write_text(json.dumps(sources, separators=(",", ":")))
        (R / "mask-sources.next").replace(R / "mask-sources.json")
    json.dump(people, open(R / "junction-people.json", "w"))
    json.dump({str(k): v for k, v in counts.items()}, open(R / "inputs.json", "w"))
    print(
        len(events),
        sum(len(e[3]) for e in events),
        len(people),
        "people-like junction detections",
        flush=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--holdout",
        action="store_true",
        help="remove the frozen evaluation masks; omit on unlabelled drives",
    )
    parser.add_argument(
        "--all", action="store_true", help="ingest all available segmented shutters"
    )
    args = parser.parse_args()
    build(args.holdout, [(0, 86400)] if args.all else WIN)
