"""Render drive-clock frames in the viewer's un-orbited chase camera; export StreetNVS samples."""

import argparse
import csv
import json
import subprocess
from pathlib import Path

import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation

from pipeline.common.rig import T_world_cam, Trajectory, frame_table
from pipeline.genfill.views import RS, design, photo, rig
from pipeline.livemap.v4_render import View

traj = Trajectory(RS / "poses/ego.npz")

W, H = 512, 320
F = H / (2 * np.tan(np.deg2rad(25)))
K = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1]])
ROOT = RS / "work/genvideo"
SIZE = W, H
NAMES = [
    "front",
    "left_pillar",
    "right_pillar",
    "left_repeater",
    "right_repeater",
    "back",
]
TIMES = 105 + np.arange(150) / 10


def chase(track):
    doc = json.loads(Path(track).read_text())
    assert (doc["pose"], doc["vfov"], doc["width"], doc["height"], doc["fps"]) == (
        "chase",
        50,
        W,
        H,
        10,
    )
    frames = doc["frames"]
    assert np.allclose([f["t"] for f in frames], TIMES, rtol=0, atol=1e-5)
    T = np.repeat(np.eye(4)[None], len(frames), axis=0)
    T[:, :3, 3] = [f["pos"] for f in frames]
    T[:, :3, :3] = Rotation.from_quat(
        [f["quat"] for f in frames]
    ).as_matrix() @ np.diag([1, -1, -1])
    return T


def render(scene, track, layered):
    raw = ROOT / "raw"
    raw.mkdir(exist_ok=True, parents=True)
    view = View(RS / "scene" / scene)
    poses = chase(track)
    if layered:
        from pipeline.ibr import popup_render as pr

        pr.box = lambda col, *args: (
            col
        )  # omit prototype ego overlay; viewer supplies its own
        popup = pr.Popup()
        assert scene == "live-v5" and W / H == pr.WID / pr.HEI
        for t, T in zip(TIMES, poses):
            assert np.allclose(popup.cam.pose(t)[0], T, atol=2e-6, rtol=0)
            popup.cam.poses[(0, round(t * 30))] = T

    assert len(poses) == len(TIMES)
    np.savez(
        raw / "track.npz",
        t=TIMES,
        c2w=poses,
        K=K,
        scene=scene,
        width=W,
        height=H,
        layered=layered,
    )
    yy, xx = np.mgrid[:H, :W]
    rays = np.stack(
        [(xx + 0.5 - W / 2) / F, (yy + 0.5 - H / 2) / F, np.ones((H, W))], -1
    ).astype("f")
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    sky = view.sky
    ph, pw = sky.shape[:2]
    lens = np.r_[F, F, W / 2, H / 2, [0.0] * 4].astype("d")
    # Plan-view footprint of Model 3, relative to rear-axle origin, not its silhouette.
    footprint = np.array(
        [[-1.01, -0.93, 0], [3.71, -0.93, 0], [3.71, 0.93, 0], [-1.01, 0.93, 0]]
    )
    bounds = []
    for i, (t, C) in enumerate(zip(TIMES, poses)):
        E = traj.T_world_ego(t)
        pts = view.points(t, E[:3, 3])
        rw = np.repeat(C[None], 33, axis=0).astype("d")
        buf = np.zeros((H, W, 6), "f")
        view.lib.raster(
            pts.ctypes.data,
            len(pts),
            t,
            2,
            rw.ctypes.data,
            lens.ctypes.data,
            0,
            W,
            H,
            buf.ctypes.data,
        )
        wr = rays @ C[:3, :3].astype("f").T
        bg = cv.remap(
            sky,
            (
                (0.5 - np.arctan2(wr[..., 1], wr[..., 0]) / (2 * np.pi)) * pw - 0.5
            ).astype("f"),
            np.clip((0.5 - np.arcsin(wr[..., 2]) / np.pi) * ph - 0.5, 0, ph - 1).astype(
                "f"
            ),
            cv.INTER_LINEAR,
            borderMode=cv.BORDER_WRAP,
        )
        a = buf[..., 3]
        colour = buf[..., :3] + bg * (1 - a[..., None])
        support = a > 0.12
        z = np.where(support, buf[..., 4] * rays[..., 2] / a.clip(0.001), 0)
        rgb = buf[..., :3]
        if layered:
            cv.imwrite(
                str(raw / f"{i:03d}-splat.png"),
                np.clip(colour[..., ::-1], 0, 255).astype("B"),
            )
            full, d = popup.draw(t)
            assert np.allclose(d["T"], C, rtol=0, atol=1e-9)
            # Do not promote extended sky/apron or the fixed far cylinder to measured RGB-D.
            ids = d["ids"]
            ground = (ids == 0) & (d["ground"] > 0.86)
            near = (ids > 0) & (ids < 900000) & (d["alpha"] > 0.65)
            # The moving sprite's alpha is not exported. Its depth cannot validate a static surface.
            mask = (ground | near) & (d["depth"] > 0) & (d["depth"] < 150)
            full_a = (
                np.where(ground, d["ground"], np.where(near, d["alpha"], 0)).clip(0, 1)
                * mask
            )
            a = cv.resize(full_a, SIZE, interpolation=cv.INTER_AREA)
            support = a > 0.5
            colour = cv.resize(full, SIZE, interpolation=cv.INTER_AREA)
            rgb = cv.resize(
                full.astype("f") * full_a[..., None], SIZE, interpolation=cv.INTER_AREA
            )
            z = cv.resize(
                np.where(mask, d["depth"] * full_a, 0),
                SIZE,
                interpolation=cv.INTER_AREA,
            ) / a.clip(0.001)
            z[~support] = 0
        cp = (footprint @ E[:3, :3].T + E[:3, 3] - C[:3, 3]) @ C[:3, :3]
        uv = cp @ K.T
        ego = (
            cv.fillConvexPoly(
                np.zeros((H, W), "B"),
                np.rint(uv[:, :2] / uv[:, 2:] - 0.5).astype("i"),
                255,
            )
            > 0
        )
        support[ego] = False
        z[ego] = 0

        # Point RGB against black (premultiplied), full render separately for A/B and seed.
        def image(s, im, i=i):
            assert cv.imwrite(
                str(raw / f"{i:03d}-{s}.png"), np.clip(im, 0, 255).astype("B")
            )

        image("view", colour[..., ::-1])
        image("rgb", np.where(support[..., None], rgb, 0)[..., ::-1])
        image("hole", 255 * (1 - np.where(support, a, 0)))
        image("ego", ego * 255)
        np.save(raw / f"{i:03d}-z.npy", z.astype("e"))
        bounds.append([z[support].min(), z.max()])
        if i % 10 == 0:
            print("render", i, float(support.mean()), flush=True)
    mask_viewer_ego(scene, raw, poses, footprint)


def mask_viewer_ego(scene, raw, poses, footprint):
    # The offline rig and the displayed live trajectory can differ by over one metre.
    # Exclude both footprints; the renderer draws neither car.
    d = json.loads((RS / "scene" / scene / "drive.json").read_text())
    bounds = []
    for i, (t, C) in enumerate(zip(TIMES, poses)):
        x, y, z, a = [
            np.interp(t, d["t"], np.unwrap(d[k]) if k == "yaw" else d[k])
            for k in ("x", "y", "z", "yaw")
        ]
        R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        cp = (footprint @ R.T + [x, y, z] - C[:3, 3]) @ C[:3, :3]
        uv = cp @ K.T
        ego = cv.imread(str(raw / f"{i:03d}-ego.png"), 0)
        cv.fillConvexPoly(ego, np.rint(uv[:, :2] / uv[:, 2:] - 0.5).astype("i"), 255)
        mask = ego > 0
        rgb = cv.imread(str(raw / f"{i:03d}-rgb.png"))
        rgb[mask] = 0
        hole = cv.imread(str(raw / f"{i:03d}-hole.png"), 0)
        hole[mask] = 255
        for key, img in [("rgb", rgb), ("hole", hole), ("ego", ego)]:
            cv.imwrite(str(raw / f"{i:03d}-{key}.png"), img)
        path = raw / f"{i:03d}-z.npy"
        dep = np.load(path)
        dep[mask] = 0
        np.save(path, dep)
        bounds.append([dep[dep > 0].min(), dep.max()])
    lo, hi = float(np.min(bounds, axis=0)[0]), float(np.max(bounds, axis=0)[1])
    for i in range(len(TIMES)):
        z = np.load(raw / f"{i:03d}-z.npy").astype("f")
        im = cv.applyColorMap(
            np.clip((z - lo) / (hi - lo) * 255, 0, 255).astype("B"), cv.COLORMAP_JET
        )
        im[z == 0] = 0
        cv.imwrite(str(raw / f"{i:03d}-depth.png"), im)
    (raw / "depth.json").write_text(
        json.dumps(
            {"depth_min": lo, "depth_max": hi, "normalization_mode": "per_sequence"}
        )
    )


def encode(src, dst, start, count):
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            "10",
            "-start_number",
            str(start),
            "-i",
            str(src),
            "-frames:v",
            str(count),
            "-c:v",
            "ffv1",
            "-threads",
            "2",
            str(dst),
        ],
        check=True,
    )


def export():
    raw = ROOT / "raw"
    d = ROOT / "inputs"
    d.mkdir(exist_ok=True)
    poses = np.load(raw / "track.npz")["c2w"]
    rows = []
    ref_meta = {}
    cams = {}
    for n in NAMES:
        table = frame_table(rig[n])
        times = np.array([r["t"] for r in table])
        idx = np.abs(TIMES[:, None] - (times + rig[n].time_offset_s)[None, :]).argmin(
            axis=1
        )
        ts = times[idx]
        ns = [table[j]["n"] for j in idx]
        T = T_world_cam(rig[n], ts, trajectory=traj)
        cams[n] = (T, ns, design(n, SIZE)[0])

    def info(T, k):
        return {
            "c2w_per_frame": T.tolist(),
            "intrinsics": k.tolist(),
            "image_width": W,
            "image_height": H,
        }

    fields = [
        "video",
        "prompt",
        "cond_video",
        "cond_video_mask",
        "cond_video_depth",
        "cond_reference_image",
        "scene_id",
        "camera_id",
        "start_frame",
        "end_frame",
    ]
    for first, count in [(0, 81), (69, 81), (0, 49)]:
        # Each clip is standalone, including its correct first anchor and reference exposures.
        p = d / f"{first:03d}-{count}"
        p.mkdir(exist_ok=True)
        rows = []
        cd = {"chase": info(poses[first : first + count], K)}

        def row(cid, count=count):
            return {
                "scene_id": "0",
                "camera_id": cid,
                "start_frame": 0,
                "end_frame": count,
                "prompt": "An elevated forward moving view of a suburban Florida boulevard at dawn, asphalt, painted lanes, trees and roadside poles, natural overcast light.",
            }

        entry = row("chase")
        entry["cond_reference_image"] = "anchor.png"
        cv.imwrite(str(p / "anchor.png"), cv.imread(str(raw / f"{first:03d}-view.png")))
        for tag, field in [
            ("rgb", "cond_video"),
            ("hole", "cond_video_mask"),
            ("depth", "cond_video_depth"),
        ]:
            encode(raw / ("%03d-" + tag + ".png"), p / (tag + ".mkv"), first, count)
            entry[field] = tag + ".mkv"
        rows.append(entry)
        C = poses[first]
        dep = np.load(raw / f"{first:03d}-z.npy")[::10, ::10].astype("d")
        yy, xx = np.mgrid[:H:10, :W:10]
        pts = (
            np.stack(
                [(xx + 0.5 - W / 2) * dep / F, (yy + 0.5 - H / 2) * dep / F, dep], -1
            )[dep > 0]
            @ C[:3, :3].T
            + C[:3, 3]
        )
        scored = []
        for n, (T, ns, k) in cams.items():
            q = (pts - T[first, :3, 3]) @ T[first, :3, :3]
            uv = q @ k.T
            uv = uv[:, :2] / np.maximum(q[:, 2:], 0.0001)
            inside = (
                (q[:, 2] > 0)
                & (uv[:, 0] >= 0)
                & (uv[:, 0] < W)
                & (uv[:, 1] >= 0)
                & (uv[:, 1] < H)
            )
            scored.append((float(inside.mean()), n))
        for cid, (score, n) in enumerate(sorted(scored, reverse=True)[:5]):
            T, ns, k = cams[n]
            im, _ = photo(n, ns[first], SIZE)
            cv.imwrite(str(p / f"ref{cid}.png"), im[:, :, ::-1])
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    str(p / f"ref{cid}.png"),
                    "-frames:v",
                    "1",
                    "-c:v",
                    "ffv1",
                    str(p / f"ref{cid}.mkv"),
                ],
                check=True,
            )
            cd[str(cid)] = info(T[first : first + count], k)
            rows.append(dict(row(cid), video=f"ref{cid}.mkv"))
            ref_meta.setdefault(p.name, []).append(
                {"camera": n, "n": int(ns[first]), "view_overlap": score}
            )
        (p / "camera_data.json").write_text(json.dumps({"0": cd}))
        (p / "depth_normalization.json").write_text((raw / "depth.json").read_text())
        with (p / "metadata_cond.csv").open("w") as f:
            writer = csv.DictWriter(f, fields)
            writer.writeheader()
            writer.writerows(rows)
    (d / "reference-selection.json").write_text(json.dumps(ref_meta, indent=2))
    print("exported", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="live-v5")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument(
        "--track", type=Path, default=Path(__file__).parent / "report/chase-track.json"
    )
    parser.add_argument("--splat-only", action="store_true")
    args = parser.parse_args()
    if not args.export_only:
        render(args.scene, args.track, not args.splat_only)
    export()
