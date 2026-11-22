"""Same-surface colour comparison in real rig exposures, screened by physical camera depth."""

import json
import sys
from pathlib import Path

import cv2 as cv
import numpy as np

from pipeline.common.rig import T_world_cam, Trajectory
from pipeline.genfill.views import RS, design, photo, rig

from .evaluate import frames

traj = Trajectory(RS / "poses/ego.npz")


def run(clip, root=RS / "work/genvideo"):
    root = Path(root)
    tr = np.load(root / "raw/track.npz")
    images = frames(clip)
    h, w = images[0].shape[:2]
    w0, h0 = (int(tr["K"][0, 2] * 2), int(tr["K"][1, 2] * 2))
    db = RS / "maps/live-v4/inputs/depth"
    meta = json.loads((db / "keyframes.json").read_text())
    assert meta["depth"]["quantity"].startswith("z")
    rows = []
    yy, xx = np.mgrid[4:h:8, 4:w:8]
    k = np.diag([w / w0, h / h0, 1]) @ tr["K"]
    rays = np.stack([xx + 0.5, yy + 0.5, np.ones_like(xx)], -1) @ np.linalg.inv(k).T
    for i in np.linspace(5, len(images) - 5, 10).round().astype(int):
        t = tr["t"][i]
        c = tr["c2w"][i]
        z = cv.resize(
            np.load(root / "raw" / f"{i:03d}-z.npy").astype("f"),
            (w, h),
            interpolation=cv.INTER_NEAREST,
        )[yy, xx]
        a = cv.resize(
            cv.imread(str(root / "raw" / f"{i:03d}-hole.png"), 0),
            (w, h),
            interpolation=cv.INTER_AREA,
        )[yy, xx]
        good = (a < 64) & (z > 1) & (z < 60)
        points = (rays[good] * z[good, None]) @ c[:3, :3].T + c[:3, 3]
        where = yy[good], xx[good]
        src = cv.resize(
            cv.imread(str(root / "raw" / f"{i:03d}-view.png")),
            (w, h),
            interpolation=cv.INTER_AREA,
        )[where][:, ::-1].astype("f")
        splat = src
        spath = root / "raw" / f"{i:03d}-splat.png"
        if spath.exists():
            splat = cv.resize(
                cv.imread(str(spath)), (w, h), interpolation=cv.INTER_AREA
            )[where][:, ::-1].astype("f")
        gen = images[i][where][:, ::-1].astype("f")
        for cam, lens in rig.items():
            e = min(
                meta["frames"][cam], key=lambda f: abs(f["t"] + lens.time_offset_s - t)
            )
            n = e["n"]
            ft = e["t"]
            s = e["t"] + lens.time_offset_s - t
            if abs(s) > 0.18:
                continue
            depth = np.load(db / cam / f"{n:06d}.npy").astype("f")
            dh, dw = depth.shape
            # Pixel coordinates and row-dependent pose on the original rolling-shutter sensor.
            T = T_world_cam(lens, ft, trajectory=traj)
            q = (points - T[:3, 3]) @ T[:3, :3]
            uv, valid = lens.project(q)
            for _ in range(2):
                T = T_world_cam(lens, ft, np.clip(uv[:, 1], 0, lens.height), traj)
                q = np.einsum("ni,nij->nj", points - T[:, :3, 3], T[:, :3, :3])
                uv, valid = lens.project(q)

            def ix(sz, uv=uv, lens=lens):
                return np.clip(
                    (uv[:, 1] * sz[0] / lens.height).astype(int), 0, sz[0] - 1
                ), np.clip((uv[:, 0] * sz[1] / lens.width).astype(int), 0, sz[1] - 1)

            obs_z = depth[ix((dh, dw))]
            ok = (
                valid
                & (q[:, 2] > 0)
                & (obs_z > 0)
                & (abs(obs_z - q[:, 2]) < np.maximum(0.5, 0.05 * q[:, 2]))
            )
            for kind in ("sky", "dyn"):
                mask = cv.imread(str(RS / "masks" / kind / cam / f"{n:06d}.png"), -1)
                if mask is None:
                    ok[:] = False
                    break
                ok &= mask[ix(mask.shape)] == 0
            kref = design(cam, (w0, h0))[0]
            pp = q @ kref.T
            uvrect = (pp[:, :2] / pp[:, 2:].clip(0.001) - 0.5).astype("f")
            ok &= (
                (uvrect[:, 0] > 1)
                & (uvrect[:, 0] < w0 - 2)
                & (uvrect[:, 1] > 1)
                & (uvrect[:, 1] < h0 - 2)
            )
            if ok.sum() < 12:
                continue
            img, _ = photo(cam, n, (w0, h0))
            xy = uvrect[ok, None, :]
            sample = cv.remap(img, xy, None, cv.INTER_LINEAR)[:, 0].astype("f")
            rows.append(
                {
                    "frame": int(i),
                    "cam": cam,
                    "exposure": n,
                    "offset_s": float(s),
                    "count": int(ok.sum()),
                    "conditioning_mae": float(abs(src[ok] - sample).mean()),
                    "splat_mae": float(abs(splat[ok] - sample).mean()),
                    "generated_mae": float(abs(gen[ok] - sample).mean()),
                }
            )
    total = sum(a["count"] for a in rows)
    assert total
    summary = dict(
        samples=total,
        exposures=len(rows),
        **{
            k: sum(a[k] * a["count"] for a in rows) / total
            for k in ("conditioning_mae", "splat_mae", "generated_mae")
        },
    )
    (clip.parent / "photo-metrics.json").write_text(
        json.dumps({"summary": summary, "exposures": rows}, indent=2)
    )
    print(summary)


if __name__ == "__main__":
    run(Path(sys.argv[1]), *[Path(p) for p in sys.argv[2:]])
