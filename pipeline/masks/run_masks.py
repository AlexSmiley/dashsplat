#!/usr/bin/env python3
"""Sky, road and movable-object instance masks for the keyframes in $RS/depth/keyframes.json.

Model: Mask2Former Swin-L trained on Cityscapes panoptic (facebook/mask2former-swin-large-cityscapes-
panoptic), run on MPS on each original (distorted) camera image resized to the output grid.

Outputs, on the same grid as the depth maps (default 724x469; recorded in keyframes.json):
  masks/sky/<cam>/<n:06d>.png    uint8, 255 = sky
  masks/road/<cam>/<n:06d>.png   uint8, 255 = road surface (extra product, used for depth scale checks)
  masks/dyn/<cam>/<n:06d>.png    uint16, instance track id of movable objects, 0 = background
  masks/dyn/<cam>/<n:06d>.json   [{id, class, score, area, bbox [x0,y0,x1,y1], moving, speed_mps}]
  masks/dyn/<cam>/tracks.json    per track: class, frames, world positions, speed, moving (--motion)
Pixels outside the calibration mask (car body, vignette, image-circle border) are 0 in every mask.

Track ids are per camera, kept across keyframes by greedy mask-IoU matching (classes merged into
vehicle / person / two-wheeler groups). --motion (run after the depth tool) lifts each instance to
the world with its depth and the trajectory and flags tracks moving faster than --moving-mps.

Usage:
  run_masks.py --cams front                # segmentation + tracking (resumable)
  run_masks.py --cams front --motion       # moving/parked flags from depth + ego motion
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "depth"))
from rscommon import (RS, CAMS, setup_env, load_rig, load_image, trajectory, traj_name, crig,  # noqa: E402
                      atomic_imwrite, atomic_write_text, wait_for_memory, time_filter)

setup_env()
import cv2  # noqa: E402
import numpy as np  # noqa: E402

GROUP = {11: "person", 12: "person", 13: "vehicle", 14: "vehicle", 15: "vehicle", 16: "vehicle",
         17: "two-wheeler", 18: "two-wheeler"}
KF_PATH = RS / "depth" / "keyframes.json"


def out_path(kind, cam, n, ext="png"):
    return RS / "masks" / kind / cam / f"{n:06d}.{ext}"


def world_mask(cam, size):
    """True where the whole output pixel sees the world (calibration mask, area-downsampled >= 0.5)."""
    m = cv2.resize(cam.mask.astype(np.float32), size, interpolation=cv2.INTER_AREA)
    return m >= 0.5


def iou_match(prev, cur):
    """Greedy IoU matching of instance dicts within class groups. Returns {cur_idx: prev_idx}."""
    pairs = []
    for i, c in enumerate(cur):
        for j, p in enumerate(prev):
            if GROUP[c["label"]] != GROUP[p["label"]]:
                continue
            inter = np.logical_and(c["mask"], p["mask"]).sum()
            if inter == 0:
                continue
            iou = inter / np.logical_or(c["mask"], p["mask"]).sum()
            if iou > 0.15:
                pairs.append((iou, i, j))
    pairs.sort(reverse=True)
    used_i, used_j, out = set(), set(), {}
    for iou, i, j in pairs:
        if i in used_i or j in used_j:
            continue
        out[i] = j
        used_i.add(i)
        used_j.add(j)
    return out


def segment_camera(cam_name, frames, size, seg, overwrite=False):
    from seg import SKY, ROAD
    rig = load_rig()
    cam = rig[cam_name]
    wm = world_mask(cam, size)
    for kind in ("sky", "road", "dyn"):
        (RS / "masks" / kind / cam_name).mkdir(parents=True, exist_ok=True)
    state_p = RS / "masks" / "dyn" / cam_name / "state.json"
    state = json.loads(state_p.read_text()) if state_p.exists() and not overwrite else {"next_id": 1}
    prev = None
    stale = True
    done = 0
    t0 = time.time()
    from seg import MOVABLE
    for idx, e in enumerate(frames):
        n = e["n"]
        if not overwrite and out_path("dyn", cam_name, n, "json").exists():
            stale = True
            continue
        if stale:
            # resume: continue tracks from the previous keyframe's saved instances, if any
            prev = None
            if idx > 0 and out_path("dyn", cam_name, frames[idx - 1]["n"], "json").exists():
                pn = frames[idx - 1]["n"]
                ids = cv2.imread(str(out_path("dyn", cam_name, pn)), cv2.IMREAD_UNCHANGED)
                info = json.loads(out_path("dyn", cam_name, pn, "json").read_text())
                prev = [dict(mask=ids == d["id"], label=d["label"], id=d["id"]) for d in info]
            stale = False
        im = load_image(cam_name, n, size)
        o = seg([im])[0]
        sem = o["semantic"]
        sky = (sem == SKY) & wm
        road = (sem == ROAD) & wm
        cur = [dict(mask=ins["mask"] & wm, label=ins["label"], score=ins["score"]) for ins in o["instances"]]
        cur = [c for c in cur if c["mask"].sum() >= 20]
        match = iou_match(prev or [], cur)
        ids = np.zeros(sem.shape, np.uint16)
        info = []
        for i, c in enumerate(cur):
            if i in match:
                c["id"] = prev[match[i]]["id"]
            else:
                c["id"] = state["next_id"]
                state["next_id"] += 1
            ids[c["mask"]] = c["id"]
            ys, xs = np.nonzero(c["mask"])
            info.append(dict(id=int(c["id"]), label=int(c["label"]), **{"class": MOVABLE[c["label"]]},
                             score=round(float(c["score"]), 3), area=int(c["mask"].sum()),
                             bbox=[int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
                             moving=None, speed_mps=None))
        atomic_imwrite(out_path("sky", cam_name, n), sky.astype(np.uint8) * 255)
        atomic_imwrite(out_path("road", cam_name, n), road.astype(np.uint8) * 255)
        atomic_imwrite(out_path("dyn", cam_name, n), ids)
        atomic_write_text(state_p, json.dumps(state))
        atomic_write_text(out_path("dyn", cam_name, n, "json"), json.dumps(info))  # last: marks the frame done
        prev = cur
        done += 1
        if done % 50 == 0:
            print(f"{cam_name}: {done} frames, {(time.time() - t0) / done:.3f} s/frame", flush=True)
    print(f"{cam_name}: segmented {done} new frames", flush=True)


def prune_camera(cam_name, frames):
    """Remove instances whose class is no longer in seg.MOVABLE from existing outputs."""
    from seg import MOVABLE
    keep = set(MOVABLE.values())
    removed = 0
    for e in frames:
        pj = out_path("dyn", cam_name, e["n"], "json")
        if not pj.exists():
            continue
        info = json.loads(pj.read_text())
        drop = [d for d in info if d["class"] not in keep]
        if not drop:
            continue
        ids = cv2.imread(str(out_path("dyn", cam_name, e["n"])), cv2.IMREAD_UNCHANGED)
        for d in drop:
            ids[ids == d["id"]] = 0
        atomic_imwrite(out_path("dyn", cam_name, e["n"]), ids)
        atomic_write_text(pj, json.dumps([d for d in info if d["class"] in keep]))
        removed += len(drop)
    print(f"{cam_name}: pruned {removed} instances", flush=True)


def motion_camera(cam_name, frames, size, moving_mps=2.0, traj=None):
    """Lift instances to the world with depth + trajectory; flag moving tracks."""
    rig = load_rig()
    cam = rig[cam_name]
    W, H = size
    sx, sy = cam.width / W, cam.height / H
    jj, ii = np.meshgrid((np.arange(W) + 0.5) * sx, (np.arange(H) + 0.5) * sy)
    rays = cam.unproject(np.stack([jj, ii], -1))
    rays = np.nan_to_num(rays)
    tracks = {}
    per_frame = {}
    # per-frame instance observations are cached (keyed on the depth, mask and trajectory files)
    cache_p = RS / "work" / "depth" / "cache" / f"motion_obs_{cam_name}.json"
    traj_key = f"{traj.source}:{Path(str(traj.source)).stat().st_mtime_ns}"
    cache = json.loads(cache_p.read_text()) if cache_p.exists() else {}
    if cache.get("traj") != traj_key:
        cache = {"traj": traj_key, "frames": {}}
    for e in frames:
        n = e["n"]
        pj = out_path("dyn", cam_name, n, "json")
        pd = RS / "depth" / cam_name / f"{n:06d}.npy"
        if not pj.exists() or not pd.exists():
            continue
        info = json.loads(pj.read_text())
        if not info:
            continue
        stamp = f"{pd.stat().st_mtime_ns}:{out_path('dyn', cam_name, n).stat().st_mtime_ns}"
        hit = cache["frames"].get(str(n))
        if hit and hit["stamp"] == stamp:
            per_frame[n] = info
            for tid, cls, Xw, r in hit["obs"]:
                tracks.setdefault(tid, dict(cls=cls, obs=[]))["obs"].append((e["t"], n, Xw, r))
            continue
        frame_obs = []
        ids = cv2.imread(str(out_path("dyn", cam_name, n)), cv2.IMREAD_UNCHANGED)
        z = np.load(pd).astype(np.float32)
        T = crig.T_world_cam(cam, e["t"], None, traj)
        per_frame[n] = info
        for d in info:
            m = cv2.erode((ids == d["id"]).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & (z > 0)
            if m.sum() < 10:
                continue
            r = rays[m]
            X = r / np.maximum(r[:, 2:3], 1e-6) * z[m][:, None]
            # nearest 50% of the object's pixels: its visible surface, robust to background leaks
            zz = z[m]
            keep = zz <= np.percentile(zz, 50)
            Xc = np.median(X[keep], 0)
            Xw = T[:3, :3] @ Xc + T[:3, 3]
            tracks.setdefault(d["id"], dict(cls=d["class"], obs=[]))["obs"].append(
                (e["t"], n, Xw.tolist(), float(np.median(zz))))
            frame_obs.append([d["id"], d["class"], Xw.tolist(), float(np.median(zz))])
        cache["frames"][str(n)] = dict(stamp=stamp, obs=frame_obs)
    cache_p.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(cache_p, json.dumps(cache))
    for tr in tracks.values():
        tr["obs"].sort()
    summary = {}
    for tid, tr in tracks.items():
        summary[str(tid)] = fit_track(tr["obs"], tr["cls"])
    objects = link_tracks(summary, tracks)
    decided = {"moving": 0, "static": 0, "undetermined": 0}
    obs_decided = 0
    obs_total = 0
    obj_summary = {}
    tl = np.load(RS / "data/drive/timeline.npz")
    odo_t = tl["t"]
    odo_v = np.concatenate([[0], np.cumsum(np.hypot(np.diff(tl["x"]), np.diff(tl["y"])))])
    for oid, members in objects.items():
        obs = sorted(o for m in members for o in tracks[int(m)]["obs"])
        ego = float(np.interp(obs[-1][0], odo_t, odo_v) - np.interp(obs[0][0], odo_t, odo_v))
        fit = fit_object(obs, moving_mps, ego)
        obj_summary[str(oid)] = dict(**{"class": summary[members[0]]["class"]}, tracks=[int(m) for m in members],
                                     n_obs=len(obs), t_first=obs[0][0], t_last=obs[-1][0], **fit)
        key = "undetermined" if fit["moving"] is None else ("moving" if fit["moving"] else "static")
        decided[key] += 1
        obs_total += len(obs)
        obs_decided += len(obs) if fit["moving"] is not None else 0
        for m in members:
            summary[m].update(object_id=oid, moving=fit["moving"], speed_mps=fit["speed_mps"])
    for n, info in per_frame.items():
        for d in info:
            s = summary.get(str(d["id"]))
            if s:
                d["object_id"] = s.get("object_id")
                d["moving"] = s["moving"]
                d["speed_mps"] = None if s["speed_mps"] is None else round(s["speed_mps"], 2)
        atomic_write_text(out_path("dyn", cam_name, n, "json"), json.dumps(info))
    atomic_write_text(RS / "masks" / "dyn" / cam_name / "tracks.json", json.dumps(
        dict(trajectory=traj_name(traj), moving_mps=moving_mps,
             rule="tracks linked into objects across gaps <= 1.5 s by world position; an object is decided when "
                  "observed >= 1 s with >= 3 observations: moving if fitted speed > moving_mps and its displacement "
                  "beats the noise (max of 1 m, 2x fit scatter, 5% of range), static if it stays within the noise "
                  "while the ego moved >= 2 m or over >= 3 s; otherwise undetermined",
             counts=dict(objects=len(objects), **decided, observations=obs_total, observations_decided=obs_decided),
             objects=obj_summary, tracks=summary)))
    print(f"{cam_name}: {len(summary)} tracks -> {len(objects)} objects: {decided['moving']} moving, "
          f"{decided['static']} static, {decided['undetermined']} undetermined; "
          f"{obs_decided}/{obs_total} instance observations decided", flush=True)
    return dict(tracks=len(summary), objects=len(objects), **decided, observations=obs_total,
                observations_decided=obs_decided)


def fit_track(obs, cls):
    ts = np.array([o[0] for o in obs])
    P = np.array([o[2] for o in obs])[:, :2]
    v = np.zeros(2)
    if len(obs) >= 2 and ts[-1] - ts[0] > 0.05:
        A = np.stack([ts - ts.mean(), np.ones_like(ts)], 1)
        v = np.linalg.lstsq(A, P, rcond=None)[0][0]
    return {"class": cls, "n_obs": len(obs), "t_first": float(ts[0]), "t_last": float(ts[-1]),
            "p_first": P[0].tolist(), "p_last": P[-1].tolist(), "v": v.tolist(),
            "range": float(np.median([o[3] for o in obs])), "speed_mps": None, "moving": None,
            "positions": [[round(o[0], 3), o[1]] + [round(x, 2) for x in o[2]] for o in obs]}


def link_tracks(summary, tracks, max_gap=1.5):
    """Union fragments of one object: a track that starts within max_gap s after another ends, same class
    group, where the first track's extrapolated position lands within the noise of the second's start."""
    from seg import MOVABLE
    group = {name: GROUP[k] for k, name in MOVABLE.items()}
    parent = {k: k for k in summary}

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    import bisect
    ids = sorted(summary, key=lambda k: summary[k]["t_first"])
    ends = sorted(summary, key=lambda k: summary[k]["t_last"])
    end_t = [summary[k]["t_last"] for k in ends]
    for b in ids:
        sb = summary[b]
        best, bd = None, None
        lo = bisect.bisect_left(end_t, sb["t_first"] - max_gap)
        hi = bisect.bisect_left(end_t, sb["t_first"])
        for a in ends[lo:hi]:
            sa = summary[a]
            gap = sb["t_first"] - sa["t_last"]
            if gap <= 0 or gap > max_gap or a == b or group[sa["class"]] != group[sb["class"]]:
                continue
            if find(a) == find(b):
                continue
            pred = np.array(sa["p_last"]) + np.array(sa["v"]) * gap
            d = float(np.linalg.norm(pred - np.array(sb["p_first"])))
            tol = max(1.5, 0.08 * max(sa["range"], sb["range"]))
            if d < tol and (bd is None or d < bd):
                best, bd = a, d
        if best is not None:
            parent[find(b)] = find(best)
    objs = {}
    for k in summary:
        objs.setdefault(find(k), []).append(k)
    return {i + 1: sorted(v, key=lambda k: summary[k]["t_first"]) for i, v in enumerate(objs.values())}


def fit_object(obs, moving_mps, ego_m=0.0):
    ts = np.array([o[0] for o in obs])
    P = np.array([o[2] for o in obs])[:, :2]
    rng = np.array([o[3] for o in obs])
    span = float(ts[-1] - ts[0])
    if len(obs) < 3 or span < 1.0:
        return dict(speed_mps=None, moving=None, span_s=round(span, 3))
    A = np.stack([ts - ts.mean(), np.ones_like(ts)], 1)
    coef, *_ = np.linalg.lstsq(A, P, rcond=None)
    speed = float(np.linalg.norm(coef[0]))
    sig = float(np.sqrt(((P - A @ coef) ** 2).sum(1).mean()))
    disp = speed * span
    noise = max(1.0, 2 * sig, 0.05 * float(np.median(rng)))
    if speed > 35.0:
        moving = None  # implausible for road users here: depth noise dominates
    elif speed > moving_mps and disp > noise:
        moving = True
    elif disp < noise and (ego_m >= 2.0 or span >= 3.0):
        # static needs evidence from changing viewpoints (ego moved >= 2 m) or a long look
        moving = False
    else:
        moving = None
    return dict(speed_mps=speed, moving=moving, span_s=round(span, 3), scatter_m=round(sig, 2),
                ego_travel_m=round(ego_m, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cams", default="front")
    ap.add_argument("--size", default="724x469")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--motion", action="store_true")
    ap.add_argument("--prune", action="store_true", help="drop instances of classes no longer in seg.MOVABLE")
    ap.add_argument("--moving-mps", type=float, default=2.0)
    ap.add_argument("--trajectory", default=None)
    ap.add_argument("--t0", type=float, default=None, help="segment only keyframes with key time >= t0")
    ap.add_argument("--t1", type=float, default=None, help="segment only keyframes with key time <= t1")
    ap.add_argument("--min-free", type=float, default=20, help="wait while system free memory %% is below this")
    args = ap.parse_args()
    kf = json.loads(KF_PATH.read_text())
    size = tuple(map(int, args.size.split("x")))
    cams = CAMS if args.cams == "all" else args.cams.split(",")
    if args.prune:
        for c in cams:
            prune_camera(c, kf["frames"][c])
        return
    if args.motion:
        traj = trajectory(args.trajectory)
        for c in cams:
            motion_camera(c, kf["frames"][c], size, args.moving_mps, traj)
        return
    from seg import Segmenter, REPO, LICENCE
    wait_for_memory(args.min_free, what="loading Mask2Former")
    seg = Segmenter()
    for c in cams:
        segment_camera(c, time_filter(kf["frames"][c], args.t0, args.t1), size, seg, args.overwrite)
    meta_p = RS / "masks" / "masks.json"
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    meta.update(model=REPO, licence=LICENCE, grid=dict(width=size[0], height=size[1],
                note="original camera image grid resized; same grid as depth/<cam>/*.npy"),
                sky="255 = sky", road="255 = road", dyn="uint16 track id per camera, 0 = background",
                classes="person, rider, car, truck, bus, motorcycle, bicycle (Cityscapes; train dropped)",
                motion="masks/dyn/<cam>/tracks.json: counts, objects (linked tracks) with moving true/false/null")
    atomic_write_text(meta_p, json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
