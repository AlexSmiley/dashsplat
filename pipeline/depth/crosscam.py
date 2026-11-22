#!/usr/bin/env python3
"""Cross-camera and cross-time consistency of the delivered depth maps ($RS/depth/<cam>/*.npy).

For a camera pair (A at time ta, B at time tb) every valid static pixel of A (not sky, not a movable
object) is unprojected with the rig model, moved to the world with A's row-time pose, projected into
B (row-time pose refined once) and its predicted z compared with B's depth at that pixel.
Reported: median AbsRel, fraction within 5 %, median ratio (A-implied / B), number of pixels.

Usage: crosscam.py [--t0 80 --t1 140 --every 2.0] [--out $RS/work/depth/report/crosscam.json]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rscommon import RS, setup_env, load_rig, trajectory, traj_name, crig  # noqa: E402

setup_env()
import cv2  # noqa: E402
import numpy as np  # noqa: E402

KF = RS / "depth" / "keyframes.json"


class Grid:
    def __init__(self, cam, size):
        self.cam, self.size = cam, size
        W, H = size
        self.sx, self.sy = cam.width / W, cam.height / H
        jj, ii = np.meshgrid((np.arange(W) + 0.5) * self.sx, (np.arange(H) + 0.5) * self.sy)
        self.uv = np.stack([jj, ii], -1)
        r = cam.unproject(self.uv)
        self.ok = np.all(np.isfinite(r), -1)
        self.rays = np.nan_to_num(r)


def load_frame(cam, n, size):
    z = np.load(RS / "depth" / cam / f"{n:06d}.npy").astype(np.float32)
    bad = np.zeros(z.shape, bool)
    dp = RS / "masks" / "dyn" / cam / f"{n:06d}.png"
    if dp.exists():
        bad |= cv2.dilate((cv2.imread(str(dp), cv2.IMREAD_UNCHANGED) > 0).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    return z, bad


def warp_pair(gA, zA, badA, tA, gB, zB, badB, tB, traj, zmax=40.0):
    m = (zA > 0) & (zA < zmax) & ~badA & gA.ok
    ii, jj = np.nonzero(m)
    if len(ii) < 200:
        return None
    sel = np.random.default_rng(0).choice(len(ii), min(len(ii), 40000), replace=False)
    ii, jj = ii[sel], jj[sel]
    r = gA.rays[ii, jj]
    Xa = r / r[:, 2:3] * zA[ii, jj][:, None]
    v_nat = gA.uv[ii, jj, 1]
    TA = crig.T_world_cam(gA.cam, tA, v_nat, traj)  # per-point row time (N,4,4)
    Xw = np.einsum("nij,nj->ni", TA[:, :3, :3], Xa) + TA[:, :3, 3]
    TB = crig.T_world_cam(gB.cam, tB, None, traj)
    for _ in range(2):
        if TB.ndim == 2:
            Rb, tb = TB[:3, :3], TB[:3, 3]
            Xb = (Xw - tb) @ Rb
        else:
            Xb = np.einsum("nji,nj->ni", TB[:, :3, :3], Xw - TB[:, :3, 3])
        uv, ok = gB.cam.project(Xb)
        TB = crig.T_world_cam(gB.cam, tB, np.where(ok, uv[:, 1], gB.cam.height / 2), traj)
    jb = np.floor(uv[:, 0] / gB.sx).astype(int)
    ib = np.floor(uv[:, 1] / gB.sy).astype(int)
    W, H = gB.size
    ok &= (jb >= 0) & (jb < W) & (ib >= 0) & (ib < H) & (Xb[:, 2] > 0.3)
    jb, ib, zp = jb[ok], ib[ok], Xb[ok, 2]
    db = zB[ib, jb]
    good = (db > 0) & ~badB[ib, jb]
    zp, db = zp[good], db[good]
    if len(zp) < 200:
        return None
    rel = np.abs(zp - db) / db
    return dict(absrel=float(np.median(rel)), inlier5=float((rel < 0.05).mean()),
                ratio=float(np.median(zp / db)), n=int(len(zp)))


def nearest(frames, t):
    ts = np.array([f["t"] for f in frames])
    i = int(np.argmin(np.abs(ts - t)))
    return frames[i], abs(ts[i] - t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t0", type=float, default=80)
    ap.add_argument("--t1", type=float, default=140)
    ap.add_argument("--every", type=float, default=2.0)
    ap.add_argument("--out", default=str(RS / "work/depth/report/crosscam.json"))
    args = ap.parse_args()
    kf = json.loads(KF.read_text())
    W, H = kf["depth"]["cameras"]["front"]["grid"]["width"], kf["depth"]["cameras"]["front"]["grid"]["height"]
    rig = load_rig()
    traj = trajectory()
    have = {c: [f for f in fr if (RS / "depth" / c / f"{f['n']:06d}.npy").exists()] for c, fr in kf["frames"].items()}
    grids = {c: Grid(rig[c], (W, H)) for c in have if have[c]}
    tl = np.load(RS / "data/drive/timeline.npz")
    odo = np.concatenate([[0], np.cumsum(np.hypot(np.diff(tl["x"]), np.diff(tl["y"])))])

    def t_after(t, metres):
        o = np.interp(t, tl["t"], odo) + metres
        return float(np.interp(o, odo, tl["t"])) if o <= odo[-1] else None

    pairs = [("front", "front", 1.0), ("front", "front", 4.0),
             ("left_pillar", "left_repeater", 0.0), ("right_pillar", "right_repeater", 0.0),
             ("left_repeater", "back", 0.0), ("right_repeater", "back", 0.0),
             ("front", "left_pillar", 12.0), ("front", "right_pillar", 12.0),
             ("left_pillar", "left_repeater", 4.0), ("right_pillar", "right_repeater", 4.0),
             ("left_pillar", "left_pillar", 1.0), ("back", "back", -1.0)]
    res = {}
    for a, b, dist in pairs:
        if a not in grids or b not in grids:
            continue
        key = f"{a}->{b}@{dist:+.0f}m"
        rows = []
        for t in np.arange(args.t0, args.t1, args.every):
            fa, da = nearest(have[a], t)
            if da > 0.5:
                continue
            tb = fa["t"] if dist == 0 else t_after(fa["t"], dist) if dist > 0 else None
            if dist < 0:
                tb = t_after(fa["t"], 0) if False else None
                # back camera: compare with the frame from 1 m earlier (the scene recedes)
                o = np.interp(fa["t"], tl["t"], odo) + dist
                tb = float(np.interp(o, odo, tl["t"])) if o > 0 else None
            if tb is None:
                continue
            fb, db_ = nearest(have[b], tb)
            if db_ > 0.1 or (a == b and fb["n"] == fa["n"]):
                continue
            za, ba = load_frame(a, fa["n"], (W, H))
            zb, bb = load_frame(b, fb["n"], (W, H))
            r = warp_pair(grids[a], za, ba, fa["t"], grids[b], zb, bb, fb["t"], traj)
            if r:
                r.update(ta=fa["t"], tb=fb["t"])
                rows.append(r)
        if rows:
            res[key] = dict(n_pairs=len(rows), absrel=float(np.median([r["absrel"] for r in rows])),
                            inlier5=float(np.median([r["inlier5"] for r in rows])),
                            ratio=float(np.median([r["ratio"] for r in rows])),
                            ratio_iqr=float(np.subtract(*np.percentile([r["ratio"] for r in rows], [75, 25]))),
                            rows=rows)
            print(f"{key:40s} pairs {len(rows):3d}  AbsRel {100 * res[key]['absrel']:.1f}%  "
                  f"within5% {100 * res[key]['inlier5']:.0f}%  ratio {res[key]['ratio']:.3f}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(dict(trajectory=traj_name(traj), depth=kf["depth"].get("model"), pairs=res), indent=1))


if __name__ == "__main__":
    main()
