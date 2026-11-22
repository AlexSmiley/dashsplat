"""Native StreetNVS controls: stitched Gaussians over IBR sky/backdrop, no cards or movers."""

import csv
import ctypes as ct
import json
import subprocess
from pathlib import Path

import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation

from pipeline.common.rig import T_world_cam, Trajectory, frame_table
from pipeline.genfill.views import BG, RS, K_vfov, Scene, design, photo, rig
from pipeline.genvideo.prepare import encode
from pipeline.ibr.layer_player import Player

ROOT = RS / "work/genvideo/round2"
SCENE = RS / "scene/stitched-opt-v3"
SIZE = (832, 480)
W, H = SIZE
K = K_vfov(50, SIZE)
TRAJ = Trajectory(RS / "poses/ego.npz")
Y, X = np.mgrid[:H, :W]
RAYS = np.stack(
    [(X + 0.5 - K[0, 2]) / K[0, 0], (Y + 0.5 - K[1, 2]) / K[1, 1], np.ones_like(X)], 2
).astype("f")
RN = np.linalg.norm(RAYS, axis=2)


def state(d, t):
    return np.array(
        [
            np.interp(t, d["t"], np.unwrap(d[k]) if k == "yaw" else d[k])
            for k in ("x", "y", "z", "yaw")
        ]
    )


def yaw(a):
    return Rotation.from_euler("z", a).as_matrix()


def photosphere(pl, t, C):
    wr = RAYS @ C[:3, :3].astype("f").T
    az = np.arctan2(wr[:, :, 1], wr[:, :, 0])
    el = np.rad2deg(np.arctan2(wr[:, :, 2], np.linalg.norm(wr[:, :, :2], axis=2)))
    O = C[:3, 3]
    weights, _ = pl.selection(t, C)
    sky = np.zeros((*Y.shape, 3), "f")
    far = np.zeros((*Y.shape, 4), "f")
    lo, hi = pl.doc["backdropElevation"]
    slo, shi = pl.doc["skyElevation"]

    def sample(im, a, e, lo, hi):
        h, w = im.shape[:2]
        return cv.remap(
            im,
            ((a / (2 * np.pi) + 0.5) * w - 0.5).astype("f"),
            np.clip((hi - e) / (hi - lo) * h - 0.5, 0, h - 1).astype("f"),
            cv.INTER_LINEAR,
            borderMode=cv.BORDER_WRAP,
        )

    for k, weight in weights.items():
        p = pl.pan[k]
        sky += weight * sample(pl.rgb(p["sky"]), az, el, slo, shi)
        origin = np.array(p["origin"])
        delta = O[:2] - origin[:2]
        den = (wr[:, :, :2] ** 2).sum(2)
        b = wr[:, :, :2] @ delta
        disc = b * b - den * (delta @ delta - p["radius"] ** 2)
        z = (-b + np.sqrt(disc.clip(0))) / den
        world = O - origin + wr * z[:, :, None]
        a = np.arctan2(world[:, :, 1], world[:, :, 0])
        e = np.rad2deg(np.arctan2(world[:, :, 2], p["radius"]))
        ok = (disc > 0) & (z > 0.3) & (e >= lo) & (e <= hi)
        far += sample(pl.rgba(p["rgb"], p["matte"]), a, e, lo, hi) * (
            weight * ok[:, :, None]
        )
    fa = far[:, :, 3] / 255
    return far[:, :, :3] + sky * (1 - fa[:, :, None]), fa, (el >= 0) & (el <= shi)


def main():
    raw = ROOT / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    track = json.loads((Path(__file__).parent / "report/chase-track.json").read_text())
    assert (track["width"], track["height"], track["vfov"], track["fps"]) == (
        *SIZE,
        50,
        10,
    )
    ts = np.array([f["t"] for f in track["frames"]])
    poses = np.repeat(np.eye(4)[None], len(ts), axis=0)
    poses[:, :3, 3] = [f["pos"] for f in track["frames"]]
    poses[:, :3, :3] = Rotation.from_quat(
        [f["quat"] for f in track["frames"]]
    ).as_matrix() @ np.diag([1, -1, -1])
    np.savez(raw / "track.npz", c2w=poses, K=K, t=ts)
    view = Scene(SCENE, margin=225)
    wide = ct.CDLL(str(RS / "work/stitch/report/v2/wide.dylib")).raster
    wide.argtypes = view.rast.argtypes
    wide.restype = None
    pl = Player(RS / "scene/ibr-v1")
    dr = json.loads((SCENE / "drive.json").read_text())
    other = json.loads((pl.root / "drive.json").read_text())
    stats = []
    bounds = []
    intr = np.r_[K[0, 0], K[1, 1], W / 2, H / 2, np.zeros(4)]
    fp = np.array(
        [[-1.01, -0.93, 0], [3.71, -0.93, 0], [3.71, 0.93, 0], [-1.01, 0.93, 0]]
    )
    for i, (t, C) in enumerate(zip(ts, poses)):
        st = state(dr, t)
        ot = state(other, t)
        R = yaw(ot[3] - st[3])
        P = C.copy()
        P[:3, :3] = R @ C[:3, :3]
        P[:3, 3] = ot[:3] + R @ (C[:3, 3] - st[:3])
        bg, fa, sky = photosphere(pl, t, P)
        pts = view.local(C[:3, 3], 225)
        rw = np.ascontiguousarray(np.repeat(C[None], 33, 0), "d")
        buf = np.zeros((H, W, 6), "f")
        buf[:, :, :3] = BG
        wide(
            pts.ctypes.data,
            len(pts),
            t,
            2,
            rw.ctypes.data,
            intr.ctypes.data,
            0,
            W,
            H,
            buf.ctypes.data,
        )
        a = buf[:, :, 3].clip(0, 1)
        col = buf[:, :, :3] + (bg - BG) * (1 - a[:, :, None])
        q = (fp @ yaw(st[3]).T + st[:3] - C[:3, 3]) @ C[:3, :3]
        uv = q @ K.T
        ego = (
            cv.fillConvexPoly(
                np.zeros((H, W), "B"),
                np.rint(uv[:, :2] / uv[:, 2:] - 0.5).astype("i"),
                255,
            )
            > 0
        )
        opaque = (a >= 0.8) & ~ego
        z = np.where(opaque, buf[:, :, 4] / a.clip(0.001) / RN, 0)
        appearance = (opaque | ((fa >= 0.8) | sky) & (a < 0.8)) & ~ego
        # Mask encodes measured geometry only. Full RGB is separately shown, supplied as RGB control, and encoded for video-to-video.
        for tag, img in [
            ("view", col[:, :, ::-1]),
            ("rgb", col[:, :, ::-1]),
            ("alpha", a * 255),
            ("appearance", appearance * 255),
            ("hole", (~opaque) * 255),
            ("ego", ego * 255),
        ]:
            cv.imwrite(
                str(raw / f"{i:03d}-{tag}.png"), np.clip(img, 0, 255).astype("B")
            )
        np.save(raw / f"{i:03d}-z.npy", z.astype("e"))
        bounds.append([z[z > 0].min(), z.max()])
        stats.append(
            {
                "t": t,
                "opaque": opaque.mean(),
                "photo_added": (appearance & ~opaque).mean(),
                "unknown": (~appearance).mean(),
                "ego": ego.mean(),
            }
        )
        if i % 25 == 0:
            print("frame", i, stats[-1], flush=True)
    lo, hi = np.min(bounds, 0)[0], np.max(bounds, 0)[1]
    depthinfo = {"depth_min": lo, "depth_max": hi, "normalization_mode": "per_sequence"}
    for i in range(len(ts)):
        z = np.load(raw / f"{i:03d}-z.npy").astype("f")
        im = cv.applyColorMap(
            ((z - lo) / (hi - lo) * 255).clip(0, 255).astype("B"), cv.COLORMAP_JET
        )
        im[z == 0] = 0
        cv.imwrite(str(raw / f"{i:03d}-depth.png"), im)
    (ROOT / "coverage.json").write_text(json.dumps(stats, indent=1))
    export(poses, ts, depthinfo)


def export(poses, ts, depthinfo):
    raw = ROOT / "raw"
    refs = {}
    db = {}
    for cam, lens in rig.items():
        rows = frame_table(lens)
        sel = [r for r in rows if 100 < r["t"] < 128]
        ft = np.array([r["t"] for r in sel])
        db[cam] = (sel, T_world_cam(lens, ft, trajectory=TRAJ))
    for start in [0, 69]:
        p = ROOT / "inputs" / f"{start:03d}-81"
        p.mkdir(parents=True, exist_ok=True)
        count = 81
        times = ts[start : start + count]
        camera = {}
        rows = []
        refmeta = []

        def register(cid, ts_, k, camera=camera, rows=rows, count=count, **fields):
            camera[cid] = {
                "c2w_per_frame": ts_.tolist(),
                "intrinsics": k.tolist(),
                "image_width": W,
                "image_height": H,
            }
            rows.append(
                dict(
                    scene_id="0",
                    camera_id=cid,
                    start_frame=0,
                    end_frame=count,
                    prompt="",
                    **fields,
                )
            )

        cand = []
        front = []
        for offset in [0, 25, 45, 70]:
            target = poses[start + offset]
            z = np.load(raw / f"{start + offset:03d}-z.npy")[::16, ::16].astype("f")
            xyz = (RAYS[::16, ::16] * z[:, :, None])[z > 0] @ target[:3, :3].T + target[
                :3, 3
            ]
            for cam, (es, ps) in db.items():
                ds = np.linalg.norm(ps[:, :2, 3] - target[:2, 3], axis=1)
                ang = np.arccos(np.clip(ps[:, :3, 2] @ target[:3, 2], -1, 1))
                dt = np.abs([e["t"] - times[offset] for e in es])
                nearest = np.argmin(ds + 0.001 * dt + 1e5 * (dt > 5))
                k = design(cam, SIZE)[0]
                for idx in np.unique(np.r_[np.where(dt < 4)[0][::8], nearest]):
                    pose = ps[idx]
                    q = (xyz - pose[:3, 3]) @ pose[:3, :3]
                    uv = q @ k.T
                    uv = uv[:, :2] / uv[:, 2:].clip(0.001)
                    share = float(
                        np.mean(
                            (q[:, 2] > 0)
                            & (uv[:, 0] >= 0)
                            & (uv[:, 0] < W)
                            & (uv[:, 1] >= 0)
                            & (uv[:, 1] < H)
                        )
                    )
                    v = (cam, offset, share, es[idx], pose, k, ds[idx], ang[idx])
                    score = share - 0.011 * ds[idx] - 0.002 * np.rad2deg(ang[idx])
                    if idx == nearest and cam == "front" and offset in [0, 45]:
                        front.append(v)
                    cand.append((score, v))
        chosen = front.copy()
        for _, v in sorted(cand, key=lambda x: -x[0]):
            if v[2] < 0.1 or any(
                v[0] == u[0] and abs(v[3]["t"] - u[3]["t"]) < 1.3 for u in chosen
            ):
                continue
            chosen.append(v)
            if len(chosen) == 6:
                break
        assert len(chosen) == 6
        register(
            "chase",
            poses[start : start + count],
            K,
            cond_video="rgb.mkv",
            cond_video_mask="hole.mkv",
            cond_video_depth="depth.mkv",
            cond_reference_image="anchor.png",
        )
        (p / "anchor.png").write_bytes((raw / f"{start:03d}-view.png").read_bytes())
        for key in ["rgb", "hole", "depth"]:
            encode(raw / ("%03d-" + key + ".png"), p / (key + ".mkv"), start, count)
        for j, (cam, offset, share, e, T, k, dist, ang) in enumerate(chosen):
            img, _ = photo(cam, e["n"], SIZE)
            cv.imwrite(str(p / f"ref{j}.png"), img[:, :, ::-1])
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    str(p / f"ref{j}.png"),
                    "-frames:v",
                    "1",
                    "-c:v",
                    "ffv1",
                    str(p / f"ref{j}.mkv"),
                ],
                check=True,
            )
            register(str(j), np.repeat(T[None], count, 0), k, video=f"ref{j}.mkv")
            refmeta.append(
                {
                    "cam": cam,
                    "n": e["n"],
                    "photo_t": e["t"],
                    "target_t": times[offset],
                    "horizontal_distance": dist,
                    "height_difference": poses[start + offset, 2, 3] - T[2, 3],
                    "angle_deg": np.rad2deg(ang),
                    "overlap": share,
                }
            )
        (p / "camera_data.json").write_text(json.dumps({"0": camera}))
        (p / "depth_normalization.json").write_text(
            json.dumps(depthinfo, default=float)
        )
        refs[p.name] = refmeta
        with (p / "metadata_cond.csv").open("w") as f:
            w = csv.DictWriter(
                f, fieldnames=list(dict.fromkeys(k for row in rows for k in row))
            )
            w.writeheader()
            w.writerows(rows)
    (ROOT / "references.json").write_text(json.dumps(refs, indent=1))
    print("dense controls ready", flush=True)


if __name__ == "__main__":
    main()
