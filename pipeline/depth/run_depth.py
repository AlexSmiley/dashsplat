#!/usr/bin/env python3
"""Metric depth for the keyframes in $RS/depth/keyframes.json.

For each keyframe of each camera: rectify the original image to one or more virtual pinhole views
(one for the front, pillar and repeater lenses; three yawed views for the 180-degree rear fisheye),
run the depth model on the frame's causal window (the frame plus earlier keyframes of the same camera
at least --win-spacing metres apart, conditioned on their poses: rig mount x trajectory), convert
each view's z-depth to range along the ray, resample range onto the original camera grid (nearest,
flying pixels removed first), and convert to z along the physical camera axis.

Output: $RS/depth/<cam>/<n:06d>.npy float16 metres along the camera z axis, 0 = invalid, on the
original (distorted) image grid resized to --size (default 724x469). Unproject a pixel (j, i) with
the rig.json model scaled to that grid: centre (j+0.5, i+0.5) * (W/w, H/h) in native pixels ->
common.rig Camera.unproject -> unit ray r -> point = r * z / r_z. Pixels whose ray is more than 88
degrees off the axis, sky (masks/sky), the calibration mask and depth edges are 0.
Per-frame details (window, scale checks) go to $RS/depth/<cam>/index.json; the global description
(quantity, grid, model, licence, trajectory) to the "depth" section of keyframes.json.

Usage:
  run_depth.py --cams front [--model da3-large-1.1 --width 518 --window 5] [--overwrite]
"""
import argparse
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rscommon import (RS, CAMS, setup_env, load_rig, load_image, trajectory, traj_name, T_world_cam,  # noqa: E402
                      default_views, views_to_original, edge_mask, frame_table, wait_for_memory, atomic_save_npy,
                      atomic_write_text, time_filter)

setup_env()
import cv2  # noqa: E402
import numpy as np  # noqa: E402

KF_PATH = RS / "depth" / "keyframes.json"
MAX_RAY_DEG = 88.0
# --model auto: the bake-off winner per lens (REPORT.md)
AUTO_MODEL = {"front": "da3-large-1.1", "back": "da3-large-1.1",
              "left_pillar": "mapanything-nomask", "right_pillar": "mapanything-nomask",
              "left_repeater": "mapanything-nomask", "right_repeater": "mapanything-nomask"}


class Odometer:
    """Planar odometer for every frame of a camera (frame numbers have gaps: walk table rows)."""

    def __init__(self, cam_name):
        self.ns, ts = frame_table(cam_name)
        tl = np.load(RS / "data/drive/timeline.npz")
        d = np.hypot(np.diff(tl["x"]), np.diff(tl["y"]))
        self.odo = np.interp(ts, tl["t"], np.concatenate([[0], np.cumsum(d)]))

    def window(self, n, spacing, size):
        """Causal window: frame n plus up to size-1 earlier frames, each >= spacing m behind the last."""
        i = int(np.searchsorted(self.ns, n))
        out = [int(self.ns[i])]
        o = self.odo[i]
        while len(out) < size and i > 0:
            i -= 1
            if o - self.odo[i] >= spacing:
                out.append(int(self.ns[i]))
                o = self.odo[i]
        return out, float(self.odo[np.searchsorted(self.ns, n)] - o)


def road_scale(z, rays, road, up_cam, h_cam):
    """Road-plane check on the original grid: returns (height ratio, plane normal) or (None, None)."""
    ok = road & (z > 0.5) & (z < 30)
    if ok.sum() < 300:
        return None, None
    r = rays[ok]
    P = r / r[:, 2:3] * z[ok][:, None]
    # reuse the pinhole RANSAC by packing points as a fake 1-row depth map is awkward; inline here
    rng = np.random.default_rng(0)
    if len(P) > 20000:
        P = P[rng.choice(len(P), 20000, replace=False)]
    best, cnt = None, 0
    for _ in range(300):
        s = P[rng.choice(len(P), 3, replace=False)]
        nrm = np.cross(s[1] - s[0], s[2] - s[0])
        if np.linalg.norm(nrm) < 1e-9:
            continue
        nrm /= np.linalg.norm(nrm)
        if nrm @ up_cam < 0:
            nrm = -nrm
        if nrm @ up_cam < np.cos(np.deg2rad(12)):
            continue
        d = -nrm @ s[0]
        c = int((np.abs(P @ nrm + d) / max(abs(d), 1e-3) < 0.04).sum())
        if c > cnt:
            best, cnt = (nrm, d), c
    if best is None or cnt < 200:
        return None, None
    nrm, d = best
    Q = P[np.abs(P @ nrm + d) / max(abs(d), 1e-3) < 0.04]
    c0 = Q.mean(0)
    nrm = np.linalg.svd(Q - c0, full_matrices=False)[2][2]
    nrm = nrm if nrm @ up_cam > 0 else -nrm
    return float(abs(nrm @ c0) / h_cam), nrm


class RectCache:
    def __init__(self, views, cam_name, maxlen=16):
        self.views, self.cam, self.d, self.maxlen = views, cam_name, OrderedDict(), maxlen

    def __call__(self, n):
        if n in self.d:
            self.d.move_to_end(n)
            return self.d[n]
        img = load_image(self.cam, n)
        r = [v.rectify(img) for v in self.views]
        self.d[n] = r
        if len(self.d) > self.maxlen:
            self.d.popitem(last=False)
        return r


def process_camera(cam_name, frames, model, args, traj, fallback):
    rig = load_rig()
    odo = Odometer(cam_name)
    cam = rig[cam_name]
    size = tuple(map(int, args.size.split("x")))
    views = default_views(cam, args.width)
    rect = RectCache(views, cam_name)
    W, H = size
    sx, sy = cam.width / W, cam.height / H
    jj, ii = np.meshgrid((np.arange(W) + 0.5) * sx, (np.arange(H) + 0.5) * sy)
    rays = np.nan_to_num(cam.unproject(np.stack([jj, ii], -1)))
    up_cam = cam.T_ego_cam[:3, :3].T @ np.array([0, 0, 1.0])
    h_cam = float(cam.T_ego_cam[2, 3])
    out_dir = RS / "depth" / cam_name
    out_dir.mkdir(parents=True, exist_ok=True)
    idx_p = out_dir / "index.json"
    index = json.loads(idx_p.read_text()) if idx_p.exists() else {}
    Rv = []
    for v in views:
        M = np.eye(4)
        M[:3, :3] = v.R
        Rv.append(M)
    done, t_start = 0, time.time()
    for i, e in enumerate(frames):
        n = e["n"]
        f = out_dir / f"{n:06d}.npy"
        if f.exists() and str(n) in index and not args.overwrite:
            continue
        if args.limit and done >= args.limit:
            break
        if args.every > 1 and i % args.every:
            continue
        if done and done % 50 == 0:
            wait_for_memory(args.min_free, what=f"{cam_name} frame {n}")
        wsize = args.window if len(views) == 1 else args.window_multiview
        ns, span = odo.window(n, args.win_spacing, wsize if model.kind == "multi" else 1)
        posed = args.posed and model.kind == "multi" and len(ns) > 1
        use = model
        if model.kind == "multi" and args.posed and (len(ns) < 2 or span < 0.3):
            # no baseline behind this frame (start of drive): metric monocular fallback
            use, ns, posed = fallback(), ns[:1], False
        ranges = []
        if len(views) > 1 and use.kind == "multi":
            # several virtual views of one lens (rear fisheye): one joint call so the views agree
            ims, Ks, T = [], [], []
            for vi, v in enumerate(views):
                ims += [rect(m)[vi] for m in ns]
                Ks += [v.K] * len(ns)
                T += [T_world_cam(cam, m, traj) @ Rv[vi] for m in ns]
            out = use.predict(ims, Ks, T if posed else None, **({} if posed else {"use_pose": False}))
            per_view = [out[vi * len(ns)] for vi in range(len(views))]
        else:
            per_view = []
            for vi, v in enumerate(views):
                ims = [rect(m)[vi] for m in ns]
                Ks = [v.K] * len(ns)
                T = [T_world_cam(cam, m, traj) @ Rv[vi] for m in ns] if posed else None
                if use.kind == "multi" and not posed and len(ns) > 1:
                    per_view.append(use.predict(ims, Ks, None, use_pose=False)[0])
                else:
                    per_view.append(use.predict(ims, Ks, T)[0])
        for vi, v in enumerate(views):
            d = per_view[vi]["depth"].astype(np.float32)
            d[edge_mask(d, args.edge)] = 0
            d[~v.valid] = 0
            ranges.append(d * v.ray_norm)
        z, _ = views_to_original(views, ranges, size, np.cos(np.deg2rad(MAX_RAY_DEG)))
        sky_p = RS / "masks" / "sky" / cam_name / f"{n:06d}.png"
        road_p = RS / "masks" / "road" / cam_name / f"{n:06d}.png"
        if sky_p.exists():
            z[cv2.imread(str(sky_p), cv2.IMREAD_GRAYSCALE) > 0] = 0
        ratio = None
        if road_p.exists():
            road = cv2.imread(str(road_p), cv2.IMREAD_GRAYSCALE) > 0
            ratio, _ = road_scale(z, rays, road, up_cam, h_cam)
        scale = args.cam_factors.get(cam_name, args.scale_factor)
        if args.scale == "road" and ratio is not None and 0.5 < ratio < 2.0:
            scale = 1.0 / ratio
        z *= scale
        z[(z > args.max_depth)] = 0
        atomic_save_npy(f, z.astype(np.float16))
        index[str(n)] = dict(t=e["t"], kind=e.get("kind", "distance"), window=ns, window_span_m=round(span, 3),
                             posed=posed, model=use.name,
                             road_h_ratio=None if ratio is None else round(ratio * scale, 4), scale_applied=scale)
        done += 1
        if done % 10 == 0:
            atomic_write_text(idx_p, json.dumps(index))  # index after the depth files it lists
        if done % 50 == 0:
            print(f"{cam_name}: {done} frames, {(time.time() - t_start) / done:.3f} s/frame", flush=True)
    atomic_write_text(idx_p, json.dumps(index))
    el = time.time() - t_start
    print(f"{cam_name}: {done} new depth maps in {el:.0f} s ({el / max(done, 1):.3f} s/frame)", flush=True)
    return dict(views=[dict(name=v.name, size=list(v.size), K=np.round(v.K, 4).tolist(),
                            R_virtual_to_cam=np.round(v.R, 6).tolist()) for v in views],
                grid=dict(width=W, height=H, native_width=cam.width, native_height=cam.height))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cams", default="front")
    ap.add_argument("--model", default="auto",
                    help="auto = DA3-LARGE-1.1 for front/back, MapAnything (no internal mask) for pillars/repeaters")
    ap.add_argument("--width", type=int, default=518, help="model input width (multiple of 14)")
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--window-multiview", type=int, default=3,
                    help="window for lenses split into several views (rear fisheye: 3 views x this many frames)")
    ap.add_argument("--win-spacing", type=float, default=1.0)
    ap.add_argument("--no-pose", dest="posed", action="store_false")
    ap.add_argument("--scale", choices=["none", "road"], default="none",
                    help="road: per-frame scale from the road plane (not recommended for posed DA3)")
    ap.add_argument("--scale-factor", type=float, default=1.0, help="global depth multiplier")
    ap.add_argument("--scale-factors", default="", help="per-camera multipliers, e.g. front=0.96,left_pillar=1.05")
    ap.add_argument("--size", default="724x469")
    ap.add_argument("--edge", type=float, default=0.06, help="log-depth jump marking flying pixels")
    ap.add_argument("--max-depth", type=float, default=150.0)
    ap.add_argument("--trajectory", default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="stop after this many new frames per camera (testing)")
    ap.add_argument("--every", type=int, default=1, help="only every k-th keyframe (testing)")
    ap.add_argument("--t0", type=float, default=None, help="only keyframes with key time >= t0")
    ap.add_argument("--t1", type=float, default=None, help="only keyframes with key time <= t1")
    ap.add_argument("--min-free", type=float, default=20, help="wait while system free memory %% is below this")
    args = ap.parse_args()
    args.cam_factors = {k: float(v) for k, v in (s.split("=") for s in args.scale_factors.split(",") if s)}
    import models
    kf = json.loads(KF_PATH.read_text())
    cams = CAMS if args.cams == "all" else args.cams.split(",")
    traj = trajectory(args.trajectory)
    loaded = {}
    licences = set()

    def get_model(name):
        # one model in memory at a time (shared machine): drop others before loading
        if name not in loaded:
            for k in list(loaded):
                del loaded[k]
            import gc
            import torch
            gc.collect()
            models.empty_cache()
            wait_for_memory(args.min_free, what=f"loading {name}")
            loaded[name] = models.load_model(name)
            licences.add(loaded[name].licence)
        return loaded[name]

    fb = {}

    def fallback():
        if "m" not in fb:
            fb["m"] = models.load_model("da3metric-large")
        return fb["m"]

    info = {}
    # cameras grouped by model so each model is loaded once
    order = sorted(cams, key=lambda c: (AUTO_MODEL[c] if args.model == "auto" else args.model) != AUTO_MODEL.get(cams[0]))
    for c in order:
        mname = AUTO_MODEL[c] if args.model == "auto" else args.model
        frames = time_filter(kf["frames"][c], args.t0, args.t1)
        if not frames:
            continue
        model = get_model(mname)
        info[c] = process_camera(c, frames, model, args, traj, fallback)
        info[c].update(model=mname, licence=model.licence, input_width=args.width,
                       window=args.window if len(info[c]["views"]) == 1 else args.window_multiview,
                       win_spacing_m=args.win_spacing, pose_conditioned=args.posed,
                       scale_factor=args.cam_factors.get(c, args.scale_factor), trajectory=traj_name(traj))
    kf = json.loads(KF_PATH.read_text())
    dep = kf.get("depth", {})
    dep.update(
        quantity="z: metres along the physical camera's optical (z) axis, float16, 0 = invalid",
        internal="models predict z in virtual pinhole views; converted to range along the calibrated ray, "
                 "resampled (nearest) onto the original image grid, then z = range * ray_z",
        grid_note="original (distorted) camera image resized to width x height; intrinsics = rig.json params "
                  "with fx, cx scaled by width/native_width and fy, cy by height/native_height; distortion "
                  "unchanged; COLMAP pixel centres (j+0.5, i+0.5)",
        max_ray_angle_deg=MAX_RAY_DEG,
        invalid="0 where: sky (masks/sky), outside calib mask, ray > max angle, depth edges, no prediction",
        models_note="settings can differ per camera: depth.cameras.<cam> holds the model and settings used for "
                    "that camera; the top-level model/window/... fields describe the most recent run",
        model=args.model if args.model != "auto" else "auto: " + json.dumps({k: AUTO_MODEL[k] for k in cams}),
        licence=sorted(licences), input_width=args.width, window=args.window,
        window_multiview=args.window_multiview, win_spacing_m=args.win_spacing, pose_conditioned=args.posed,
        scale_correction=args.scale, scale_factor=args.scale_factor, trajectory=traj_name(traj),
        fallback_model="da3metric-large (only when no baseline behind a frame)", rig="calib/rig.json")
    for c, v in info.items():
        dep.setdefault("cameras", {}).setdefault(c, {}).update(v)
    kf["depth"] = dep
    atomic_write_text(KF_PATH, json.dumps(kf, indent=1))


if __name__ == "__main__":
    main()
