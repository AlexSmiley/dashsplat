#!/usr/bin/env python3
"""Chase-view renders of the delivered depth: all cameras' keyframes over the last --span metres fused
into one coloured point cloud and seen from 5 m behind / 3 m above the car (the product's view).

Usage: render_prod.py --times 95,110,125,138 [--span 25] [--cams all] [--dyn tint|drop]
Writes $RS/work/depth/report/prod_chase_t<t>.jpg (and a top-down view).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rscommon import RS, CAMS, setup_env, load_rig, trajectory, load_image, crig, edge_mask  # noqa: E402
from render import chase_pose, render_points, draw_car, look_at, label  # noqa: E402

setup_env()
import cv2  # noqa: E402
import numpy as np  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--times", default="95,110,125,138")
    ap.add_argument("--span", type=float, default=25.0)
    ap.add_argument("--cams", default="all")
    ap.add_argument("--dyn", default="tint")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--zmax", type=float, default=50.0)
    ap.add_argument("--out", default=str(RS / "work/depth/report"))
    ap.add_argument("--tag", default="prod")
    args = ap.parse_args()
    kf = json.loads((RS / "depth/keyframes.json").read_text())
    rig = load_rig()
    traj = trajectory()
    cams = CAMS if args.cams == "all" else args.cams.split(",")
    grid = kf["depth"]["cameras"]["front"]["grid"]
    W, H = grid["width"], grid["height"]
    rays = {}
    for c in cams:
        cam = rig[c]
        jj, ii = np.meshgrid((np.arange(W) + 0.5) * cam.width / W, (np.arange(H) + 0.5) * cam.height / H)
        uv = np.stack([jj, ii], -1)
        rays[c] = (np.nan_to_num(cam.unproject(uv)), uv[..., 1])
    out = Path(args.out)
    for t in map(float, args.times.split(",")):
        o_now = None
        Xs, Cs = [], []
        for c in cams:
            fr = [f for f in kf["frames"][c] if f["t"] <= t + 0.02 and (RS / "depth" / c / f"{f['n']:06d}.npy").exists()]
            if not fr:
                continue
            o_now = fr[-1]["odo"]
            use = [f for f in fr if o_now - f["odo"] <= args.span]
            # thin to ~1 m spacing to keep the cloud readable
            keep, last = [], None
            for f in reversed(use):
                if last is None or last - f["odo"] >= 1.0:
                    keep.append(f)
                    last = f["odo"]
            cam = rig[c]
            r, vrow = rays[c]
            for f in keep:
                z = np.load(RS / "depth" / c / f"{f['n']:06d}.npy").astype(np.float32)
                ok = (z > 0) & (z < args.zmax) & ~edge_mask(z)
                sub = np.zeros_like(ok)
                sub[::args.stride, ::args.stride] = True
                ok &= sub
                dyn = cv2.imread(str(RS / "masks/dyn" / c / f"{f['n']:06d}.png"), cv2.IMREAD_UNCHANGED)
                dyn = dyn > 0 if dyn is not None else np.zeros_like(ok)
                if args.dyn == "drop":
                    ok &= ~dyn
                ii, jj = np.nonzero(ok)
                rr = r[ii, jj]
                Xc = rr / np.maximum(rr[:, 2:3], 1e-6) * z[ii, jj][:, None]
                T = crig.T_world_cam(cam, f["t"], vrow[ii, jj], traj)
                Xs.append(np.einsum("nij,nj->ni", T[:, :3, :3], Xc) + T[:, :3, 3])
                im = load_image(c, f["n"], (W, H))
                col = im[ii, jj].astype(np.float32)
                if args.dyn == "tint":
                    d = dyn[ii, jj]
                    col[d] = 0.4 * col[d] + 0.6 * np.array([255, 60, 60])
                Cs.append(col.astype(np.uint8))
        if not Xs:
            continue
        X, C = np.concatenate(Xs), np.concatenate(Cs)
        T_we = traj.T_world_ego(t)
        Tc = chase_pose(T_we)
        img, f = render_points(X, C, Tc, size=(1280, 720), hfov=75, splat=2)
        draw_car(img, Tc, T_we, f)
        label(img, f"t={t:.1f}s  {len(cams)} cams, last {args.span:.0f} m fused, {len(X) / 1e6:.1f} M points", 0.7)
        cv2.imwrite(str(out / f"{args.tag}_chase_t{t:.0f}.jpg"), img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])
        top = look_at((T_we @ np.array([4, 0, 35, 1.0]))[:3], (T_we @ np.array([4, 0, 0, 1.0]))[:3],
                      up=T_we[:3, 0])
        img2, f2 = render_points(X, C, top, size=(1280, 720), hfov=70, splat=2)
        draw_car(img2, top, T_we, f2)
        label(img2, f"t={t:.1f}s top-down (35 m up)", 0.7)
        cv2.imwrite(str(out / f"{args.tag}_top_t{t:.0f}.jpg"), img2[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])
        print("rendered", t, len(X), flush=True)


if __name__ == "__main__":
    main()
