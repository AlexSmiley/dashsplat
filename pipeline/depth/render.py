#!/usr/bin/env python3
"""Render bake-off results: colourised depth grids and fused multi-frame point clouds seen from the
raised chase viewpoint (5 m behind, 3 m above the rear axle, looking forward and down).

Usage: render.py --cam front --moments 125,163 --configs all [--in $RS/work/depth/bakeoff]
Writes $RS/work/depth/report/<cam>_chase_t<t>.jpg and <cam>_depth_t<t>.jpg.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "masks"))
from rscommon import RS, setup_env, trajectory, frame_time, load_rig, edge_mask, colorize_depth  # noqa: E402

setup_env()
import cv2  # noqa: E402
import numpy as np  # noqa: E402


def look_at(eye, target, up=(0, 0, 1.0)):
    """Camera-to-world (OpenCV axes) for a camera at eye looking at target."""
    eye, target, up = map(lambda a: np.asarray(a, float), (eye, target, up))
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.stack([x, y, z], 1)
    T[:3, 3] = eye
    return T


def chase_pose(T_we, back=5.0, up=3.0, ahead=12.0):
    eye = T_we @ np.array([-back, 0, up, 1.0])
    tgt = T_we @ np.array([ahead, 0, 0, 1.0])
    return look_at(eye[:3], tgt[:3])


def render_points(X, C, T_wc, size=(960, 540), hfov=75.0, splat=2, bg=(18, 18, 22)):
    w, h = size
    f = (w / 2) / np.tan(np.deg2rad(hfov / 2))
    R, t = T_wc[:3, :3], T_wc[:3, 3]
    Y = (X - t) @ R
    z = Y[:, 2]
    k = z > 0.3
    Y, C, z = Y[k], C[k], z[k]
    u = (f * Y[:, 0] / z + w / 2).astype(np.int64)
    v = (f * Y[:, 1] / z + h / 2).astype(np.int64)
    img = np.empty((h, w, 3), np.uint8)
    img[:] = bg
    zb = np.full((h, w), np.inf)
    order = np.argsort(-z)
    u, v, z, C = u[order], v[order], z[order], C[order]
    for du in range(splat):
        for dv in range(splat):
            uu, vv = u + du, v + dv
            ok = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
            img[vv[ok], uu[ok]] = C[ok]  # far to near: nearer points overwrite
            zb[vv[ok], uu[ok]] = np.minimum(zb[vv[ok], uu[ok]], z[ok])
    return img, f


def draw_car(img, T_wc, T_we, f, color=(230, 230, 235)):
    w, h = img.shape[1], img.shape[0]
    # Model 3 footprint: rear axle origin, ~0.9 m rear overhang, 3.8 m ahead, 1.85 m wide, 1.44 m tall
    xs, ys = (-0.95, 3.75), (-0.93, 0.93)
    pts = []
    for zc in (0.0, 1.0):
        for x, y in [(xs[0], ys[0]), (xs[1], ys[0]), (xs[1], ys[1]), (xs[0], ys[1])]:
            pts.append(T_we @ np.array([x, y, zc, 1.0]))
    P = np.array(pts)[:, :3]
    Y = (P - T_wc[:3, 3]) @ T_wc[:3, :3]
    uv = np.stack([f * Y[:, 0] / Y[:, 2] + w / 2, f * Y[:, 1] / Y[:, 2] + h / 2], 1).astype(np.int32)
    over = img.copy()
    cv2.fillPoly(over, [uv[4:8]], color)
    cv2.addWeighted(over, 0.55, img, 0.45, 0, img)
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]:
        cv2.line(img, tuple(uv[a]), tuple(uv[b]), (250, 250, 250), 1, cv2.LINE_AA)


def fuse(depths, images, Ks, Twc, masks, zmax=60.0, stride=1):
    Xs, Cs = [], []
    for d, im, K, T, m in zip(depths, images, Ks, Twc, masks):
        d = d.astype(np.float32)
        ok = (d > 0) & (d < zmax) & ~edge_mask(d) & m
        if stride > 1:
            sub = np.zeros_like(ok)
            sub[::stride, ::stride] = True
            ok &= sub
        ii, jj = np.nonzero(ok)
        z = d[ii, jj]
        Xc = np.stack([(jj + 0.5 - K[0, 2]) / K[0, 0] * z, (ii + 0.5 - K[1, 2]) / K[1, 1] * z, z], 1)
        Xs.append(Xc @ T[:3, :3].T + T[:3, 3])
        Cs.append(im[ii, jj])
    return np.concatenate(Xs), np.concatenate(Cs)


def label(img, text, scale=0.7):
    cv2.rectangle(img, (0, 0), (img.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def grid(tiles, cols):
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    return np.concatenate([np.concatenate(tiles[i:i + cols], 1) for i in range(0, len(tiles), cols)], 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", default="front")
    ap.add_argument("--moments", default="125,163")
    ap.add_argument("--configs", default="all")
    ap.add_argument("--inp", default=str(RS / "work/depth/bakeoff"))
    ap.add_argument("--out", default=str(RS / "work/depth/report"))
    ap.add_argument("--corrected", action="store_true", help="apply per-frame road-plane scale")
    ap.add_argument("--tile", default="640x360")
    args = ap.parse_args()
    from seg import Segmenter, SKY
    seg = Segmenter()
    inp = Path(args.inp) / args.cam
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((inp / "prep_meta.json").read_text())
    metrics = json.loads((inp / "metrics.json").read_text())
    prep = np.load(inp / "prep.npz")
    cfgs = list(metrics) if args.configs == "all" else args.configs.split(",")
    traj = trajectory()
    cam = load_rig()[args.cam]
    tw, th = map(int, args.tile.split("x"))
    for t in args.moments.split(","):
        mm = meta["moments"][t]
        seq = mm["seq"]
        Twv = [np.array(T) for T in mm["Twv"]]
        imgs = [prep[f"img_{t}_{k}"] for k in seq]
        nonsky = []
        for im in imgs:
            small = cv2.resize(im, (im.shape[1] // 2, im.shape[0] // 2), interpolation=cv2.INTER_AREA)
            o = seg([small])[0]
            bad = o["semantic"] == SKY
            for ins in o["instances"]:
                bad |= ins["mask"]
            nonsky.append(~bad)
        t_last = frame_time(args.cam, seq[-1]) + cam.time_offset_s
        T_we = traj.T_world_ego(t_last)
        T_chase = chase_pose(T_we)
        chase_tiles, depth_tiles = [], []
        for c in cfgs:
            f = inp / f"{c}_t{t}.npz"
            if not f.exists():
                continue
            z = np.load(f)
            D = z["depth"].astype(np.float32)
            K = z["K"]
            if args.corrected:
                s = np.where(np.isfinite(z["scale"]), z["scale"], np.nanmedian(z["scale"]))
                D = D * s[:, None, None]
            h, w = D.shape[1:]
            ims = [cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA) for im in imgs]
            ms = [cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool) for m in nonsky]
            X, C = fuse(D, ims, [K] * len(D), Twv, ms, stride=1 if w <= 600 else 2)
            img, fpx = render_points(X, C, T_chase)
            draw_car(img, T_chase, T_we, fpx)
            mt = metrics[c]["moments"].get(t) or metrics[c]["moments"].get(str(t)) or {}
            txt = f"{c}  road {mt.get('road_h_ratio') or float('nan'):.2f} tri {mt.get('tri_scale') or float('nan'):.2f}"
            chase_tiles.append(label(cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA), txt, 0.5))
            dt = colorize_depth(D[-1])
            dt = cv2.resize(dt, (tw, int(tw * h / w)), interpolation=cv2.INTER_NEAREST)
            depth_tiles.append(label(dt, c, 0.5))
        suffix = "_corr" if args.corrected else ""
        rgb = cv2.resize(imgs[-1], (tw, int(tw * imgs[-1].shape[0] / imgs[-1].shape[1])), interpolation=cv2.INTER_AREA)
        depth_tiles.insert(0, label(rgb.copy(), f"{args.cam} t={t} frame {seq[-1]}", 0.5))
        cv2.imwrite(str(out / f"{args.cam}_chase_t{t}{suffix}.jpg"), grid(chase_tiles, 4)[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])
        cv2.imwrite(str(out / f"{args.cam}_depth_t{t}.jpg"), grid(depth_tiles, 4)[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])
        print("rendered", t, len(chase_tiles), flush=True)


if __name__ == "__main__":
    main()
