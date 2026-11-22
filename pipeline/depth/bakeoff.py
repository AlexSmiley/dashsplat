#!/usr/bin/env python3
"""Depth-model bake-off on real frames.

For each test moment it takes a causal sequence of keyframes (12 frames, 1 m apart while moving,
0.25 s apart while stopped), rectifies them to a pinhole view, and runs every model configuration.
Each frame's depth comes from that frame's own causal window (the frame plus up to 4 earlier
keyframes 1 m apart) for multi-view configurations.

Checks per configuration:
  road    camera height above a RANSAC road plane fitted to the predicted depth (Mask2Former road
          pixels) divided by the calibrated mounting height. 1.0 = right metric scale.
  tri     SIFT matches between keyframes ~3 m apart, two-view relative pose with the baseline set
          from the trajectory, triangulated depth vs predicted depth (scale ratio and AbsRel after a
          median rescale).
  cons    frame a's depth moved into frame b with that relative pose and compared with b's depth on
          static, non-sky, non-edge pixels (1 m and 4 m apart), raw and after a per-frame road-plane
          scale correction.

Usage:
  bakeoff.py --cam front --configs all --out $RS/work/depth/bakeoff
Outputs per camera: metrics.json, summary.md, depth stacks (npz) per config and moment, renders.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "masks"))
from rscommon import (RS, setup_env, load_rig, trajectory, traj_name, T_world_cam, frame_table,
                      frame_at_time, load_image, default_views, inv, edge_mask, colorize_depth)

setup_env()
import cv2  # noqa: E402
import numpy as np  # noqa: E402

import evalgeo as G  # noqa: E402

MOMENTS = [86, 101, 110, 125, 140, 163, 192, 222, 245, 270, 289, 320, 335]
# name: (model, input width, window size, use poses)
CONFIGS = {
    "da3metric@518": ("da3metric-large", 518, 1, False),
    "da3metric@1036": ("da3metric-large", 1036, 1, False),
    "da3nested-mono@518": ("da3nested-1.1", 518, 1, False),
    "da3nested-win5@518": ("da3nested-1.1", 518, 5, False),
    "da3nested-win5-posed@518": ("da3nested-1.1", 518, 5, True),
    "da3large-win5-posed@518": ("da3-large-1.1", 518, 5, True),
    "da3large-win5-posed@1036": ("da3-large-1.1", 1036, 5, True),
    "da3base-win5-posed@518": ("da3-base", 518, 5, True),
    "da3small-win5-posed@518": ("da3-small", 518, 5, True),
    "mapany-apache-mono@518": ("mapanything-apache", 518, 1, False),
    "mapany-apache-win5-posed@518": ("mapanything-apache", 518, 5, True),
    "mapany-nc-win5-posed@518": ("mapanything", 518, 5, True),
    "mapany-nc-nomask-win5-posed@518": ("mapanything-nomask", 518, 5, True),
    "moge2-vitl@518": ("moge2-vitl", 518, 1, False),
    "unidepth2-l@518": ("unidepth-v2-l", 518, 1, False),
}
FEAT_W = 1036
SEQ_LEN = 12
SPACING = 1.0
WIN_SPACING = 1.0


def bakeoff_view(cam, w):
    vs = default_views(cam, w)
    return vs[1] if len(vs) == 3 else vs[0]


class Drive:
    def __init__(self, cam, traj):
        self.cam, self.traj = cam, traj
        self.ns, self.ts = frame_table(cam.name)
        self.odo = np.array([traj_odo(traj, t) for t in self.ts])

    def row(self, n):
        return int(np.searchsorted(self.ns, n))

    def back_keyframes(self, n, spacing, count):
        """n and up to count-1 earlier frames, each >= spacing metres behind the previous pick.
        Frame numbers have gaps at clip boundaries, so walk table rows, not frame numbers."""
        i = self.row(n)
        out = [int(self.ns[i])]
        o = self.odo[i]
        while len(out) < count and i > 0:
            i -= 1
            if o - self.odo[i] >= spacing:
                out.append(int(self.ns[i]))
                o = self.odo[i]
        return out

    def sequence(self, t, length=SEQ_LEN, spacing=SPACING):
        n = frame_at_time(self.cam.name, t)
        kf = self.back_keyframes(n, spacing, length)
        if len(kf) < length or self.odo[n] - self.odo[kf[-1]] > 3 * spacing * length:
            pass
        # stopped (or barely moving): use time-spaced frames instead
        if self.odo[n] - self.odo[max(n - 72, 0)] < 1.0:
            kf = [n - 9 * k for k in range(length)]
            return kf[::-1], "time"
        return kf[::-1], "distance"


_ODO = {}


def traj_odo(traj, t):
    key = id(traj)
    if key not in _ODO:
        tl = np.load(RS / "data/drive/timeline.npz")
        d = np.hypot(np.diff(tl["x"]), np.diff(tl["y"]))
        _ODO[key] = (tl["t"], np.concatenate([[0], np.cumsum(d)]))
    tt, oo = _ODO[key]
    return float(np.interp(t, tt, oo))


def prepare(cam, traj, view, moments, cache_dir):
    """Images, segmentation and pair geometry for all moments (model independent)."""
    from seg import Segmenter, SKY, ROAD
    seg = Segmenter()
    drive = Drive(cam, traj)
    Rv = np.eye(4)
    Rv[:3, :3] = view.R
    K1 = view.K
    data = {}
    up_ego = np.array([0, 0, 1.0])
    for t in moments:
        seq, mode = drive.sequence(t)
        need = set(seq)
        wins = {k: drive.back_keyframes(k, WIN_SPACING, 5) for k in seq}
        for w in wins.values():
            need |= set(w)
        imgs = {k: view.rectify(load_image(cam.name, k)) for k in sorted(need)}
        Twv = {k: T_world_cam(cam, k, traj) @ Rv for k in sorted(need)}
        segs = {}
        for k in seq:
            im = cv2.resize(imgs[k], (FEAT_W // 2, imgs[k].shape[0] // 2), interpolation=cv2.INTER_AREA)
            o = seg([im])[0]
            dyn = np.zeros(o["semantic"].shape, bool)
            for ins in o["instances"]:
                dyn |= ins["mask"]
            segs[k] = dict(sky=o["semantic"] == SKY, road=o["semantic"] == ROAD, dyn=dyn,
                           ninst=len(o["instances"]))
        # pair geometry on the FEAT_W rectified images
        pairs = {}
        for gap in (1, 3, 4):
            for a_i in range(len(seq) - gap):
                a, b = seq[a_i], seq[a_i + gap]
                T_ba = inv(Twv[b]) @ Twv[a]
                base = float(np.linalg.norm(T_ba[:3, 3]))
                ma = view.valid & ~cv2.resize(segs[a]["sky"].astype(np.uint8), view.size, interpolation=cv2.INTER_NEAREST).astype(bool)
                mb = view.valid & ~cv2.resize(segs[b]["sky"].astype(np.uint8), view.size, interpolation=cv2.INTER_NEAREST).astype(bool)
                da = cv2.dilate(segs[a]["dyn"].astype(np.uint8), np.ones((7, 7), np.uint8))
                db = cv2.dilate(segs[b]["dyn"].astype(np.uint8), np.ones((7, 7), np.uint8))
                ma &= ~cv2.resize(da, view.size, interpolation=cv2.INTER_NEAREST).astype(bool)
                mb &= ~cv2.resize(db, view.size, interpolation=cv2.INTER_NEAREST).astype(bool)
                rec = dict(a=a, b=b, gap=gap, base=base, src="prior", R=T_ba[:3, :3], t=T_ba[:3, 3])
                if base > 0.3:
                    pa, pb = G.match_features(imgs[a], imgs[b], ma, mb)
                    rp = G.relative_pose(pa, pb, K1, base, T_ba)
                    if rp is not None and rp[3] is not None and rp[3] < 15:
                        R, tt, inl, ang = rp
                        rec.update(src="vision", R=R, t=tt, ang=ang, nmatch=int(inl.sum()))
                        if gap == 3:
                            z, pix, ok = G.triangulate(pa[inl], pb[inl], K1, R, tt)
                            rec["tri"] = (z[ok], pix[ok])
                pairs[(a, b)] = rec
        nv = sum(1 for p in pairs.values() if p["src"] == "vision")
        print(f"t={t} {mode} seq {seq[0]}..{seq[-1]} pairs {len(pairs)} vision {nv} "
              f"tri {[len(p['tri'][0]) for p in pairs.values() if 'tri' in p]}", flush=True)
        data[t] = dict(seq=seq, mode=mode, wins=wins, imgs=imgs, Twv=Twv, segs=segs, pairs=pairs)
    del seg
    return data


def resize_mask(m, size):
    return cv2.resize(m.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)


def run_config(name, model, view, cam, traj, data, out_dir):
    mname, w, win, posed = CONFIGS[name]
    K1 = view.K
    h = int(round(view.size[1] * w / view.size[0]))
    s = w / view.size[0]
    K = K1.copy()
    K[:2] *= s
    up_cam = view.R.T @ cam.T_ego_cam[:3, :3].T @ np.array([0, 0, 1.0])
    h_cam = float(cam.T_ego_cam[2, 3])
    res = {}
    times = []
    for t, D in data.items():
        imgs_w = {k: (D["imgs"][k] if w == view.size[0] else
                      cv2.resize(D["imgs"][k], (w, h), interpolation=cv2.INTER_AREA)) for k in D["imgs"]}
        depth = {}
        for k in D["seq"]:
            ids = D["wins"][k][:win]
            ims = [imgs_w[i] for i in ids]
            Ks = [K] * len(ids)
            T = [D["Twv"][i] for i in ids] if posed and len(ids) > 1 else None
            t0 = time.time()
            if model.kind == "multi" and T is None and win > 1:
                out = model.predict(ims, Ks, None, use_pose=False)
            else:
                out = model.predict(ims, Ks, T)
            times.append((time.time() - t0, len(ids)))
            depth[k] = out[0]["depth"]
        # masks on this grid
        valid, scale = {}, {}
        road_fit = []
        for k in D["seq"]:
            sg = D["segs"][k]
            dk = depth[k]
            v = (dk > 0) & (dk < 80) & resize_mask(view.valid, (w, h)) & ~resize_mask(sg["sky"], (w, h)) \
                & ~resize_mask(cv2.dilate(sg["dyn"].astype(np.uint8), np.ones((5, 5), np.uint8)), (w, h)) & ~edge_mask(dk)
            valid[k] = v
            rf = G.fit_road_plane(dk, K, resize_mask(sg["road"], (w, h)) & v, up_cam, h_cam)
            scale[k] = rf["scale"] if rf else np.nan
            road_fit.append(1 / rf["scale"] if rf else np.nan)
        tri_scale, tri_rel = [], []
        cons = {1: [], 4: []}
        cons_corr = {1: [], 4: []}
        for (a, b), p in D["pairs"].items():
            if "tri" in p and len(p["tri"][0]) >= 15:
                z, pix = p["tri"]
                zp = G.sample(depth[b], pix, K1, K)
                ok = zp > 0
                if ok.sum() >= 15:
                    r = zp[ok] / z[ok]
                    sc = np.median(r)
                    tri_scale.append(sc)
                    tri_rel.append(float(np.median(np.abs(zp[ok] / sc - z[ok]) / z[ok])))
            if p["gap"] in cons:
                c = G.warp_consistency(depth[a], depth[b], K, p["R"], p["t"], valid[a], valid[b])
                if c:
                    cons[p["gap"]].append(c["absrel_med"])
                sa, sb = scale[a], scale[b]
                if np.isfinite(sa) and np.isfinite(sb):
                    c2 = G.warp_consistency(depth[a] * sa, depth[b] * sb, K, p["R"], p["t"], valid[a], valid[b])
                    if c2:
                        cons_corr[p["gap"]].append(c2["absrel_med"])
        # "align to already-fused geometry": anchor the first frame (road plane, else raw), then scale
        # every next frame to the previous aligned frame warped into it (what a live fuser can do)
        seq = D["seq"]
        chain = {seq[0]: depth[seq[0]] * (scale[seq[0]] if np.isfinite(scale[seq[0]]) else 1.0)}
        for a, b in zip(seq[:-1], seq[1:]):
            p = D["pairs"][(a, b)]
            c = G.warp_consistency(chain[a], depth[b], K, p["R"], p["t"], valid[a], valid[b])
            chain[b] = depth[b] * (c["scale_med"] if c else 1.0)
        ch_tri, ch_cons = [], []
        for (a, b), p in D["pairs"].items():
            if "tri" in p and len(p["tri"][0]) >= 15:
                z, pix = p["tri"]
                zp = G.sample(chain[b], pix, K1, K)
                ok = zp > 0
                if ok.sum() >= 15:
                    ch_tri.append(float(np.median(zp[ok] / z[ok])))
            if p["gap"] == 4:
                c = G.warp_consistency(chain[a], chain[b], K, p["R"], p["t"], valid[a], valid[b])
                if c:
                    ch_cons.append(c["absrel_med"])
        rf = np.array(road_fit, float)
        res[t] = dict(
            chain_tri_scale=float(np.median(ch_tri)) if ch_tri else None,
            chain_cons4=float(np.median(ch_cons)) if ch_cons else None,
            road_h_ratio=float(np.nanmedian(rf)) if np.isfinite(rf).any() else None,
            road_h_ratio_std=float(np.nanstd(rf)) if np.isfinite(rf).any() else None,
            tri_scale=float(np.median(tri_scale)) if tri_scale else None,
            tri_absrel_aligned=float(np.median(tri_rel)) if tri_rel else None,
            cons1=float(np.median(cons[1])) if cons[1] else None,
            cons4=float(np.median(cons[4])) if cons[4] else None,
            cons1_corr=float(np.median(cons_corr[1])) if cons_corr[1] else None,
            cons4_corr=float(np.median(cons_corr[4])) if cons_corr[4] else None,
            mode=D["mode"])
        np.savez_compressed(out_dir / f"{name}_t{t}.npz", seq=np.array(D["seq"]),
                            depth=np.stack([depth[k] for k in D["seq"]]).astype(np.float16),
                            scale=np.array([scale[k] for k in D["seq"]]), K=K)
        print(name, t, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res[t].items()}, flush=True)
    per_frame = [dt / n for dt, n in times[2:]] if len(times) > 3 else [dt / n for dt, n in times]
    per_call = [dt for dt, n in times[2:]] if len(times) > 3 else [dt for dt, n in times]
    return dict(moments=res, s_per_call=float(np.median(per_call)), s_per_image=float(np.median(per_frame)),
                window=win, width=w, height=h, model=mname, licence=model.licence, posed=posed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", default="front")
    ap.add_argument("--configs", default="all")
    ap.add_argument("--moments", default=",".join(map(str, MOMENTS)))
    ap.add_argument("--out", default=str(RS / "work/depth/bakeoff"))
    ap.add_argument("--trajectory", default=None)
    args = ap.parse_args()
    import models
    rig = load_rig()
    cam = rig[args.cam]
    traj = trajectory(args.trajectory)
    view = bakeoff_view(cam, FEAT_W)
    out = Path(args.out) / args.cam
    out.mkdir(parents=True, exist_ok=True)
    moments = [float(x) if "." in x else int(x) for x in args.moments.split(",")]
    cfgs = list(CONFIGS) if args.configs == "all" else args.configs.split(",")
    t0 = time.time()
    data = prepare(cam, traj, view, moments, out)
    print(f"prepared in {time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(out / "prep.npz", **{f"img_{t}_{k}": D["imgs"][k] for t, D in data.items() for k in D["seq"]})
    meta = dict(cam=args.cam, view=dict(K=view.K.tolist(), R=view.R.tolist(), size=view.size),
                trajectory=traj_name(traj), moments={str(t): dict(seq=D["seq"], mode=D["mode"],
                Twv=[D["Twv"][k].tolist() for k in D["seq"]],
                pairs_vision=sum(1 for p in D["pairs"].values() if p["src"] == "vision"),
                n_pairs=len(D["pairs"])) for t, D in data.items()})
    (out / "prep_meta.json").write_text(json.dumps(meta))
    mpath = out / "metrics.json"
    metrics = json.loads(mpath.read_text()) if mpath.exists() else {}
    by_model = {}
    for c in cfgs:
        by_model.setdefault(CONFIGS[c][0], []).append(c)
    for mname, cs in by_model.items():
        model = models.load_model(mname)
        for c in cs:
            metrics[c] = run_config(c, model, view, cam, traj, data, out)
            metrics[c]["trajectory"] = traj_name(traj)
            mpath.write_text(json.dumps(metrics, indent=1))
        del model
        import torch
        torch.mps.empty_cache()
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
