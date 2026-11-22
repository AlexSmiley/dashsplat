"""How far the environment map can reach: if content beyond the viewer's 150 m cutoff were painted
into the map (frozen at infinity, as seen from the chase camera at `ref`), how many pixels would it
slide against where it really is, over the stretch, in the published 1440x900 chase view (vfov 50)?

For each distance D and each direction in the horizon band (0-3 deg up, every azimuth the chase view
shows at that moment), the angle between the frozen direction (from the camera at ref) and the true
one (from the camera at t), in pixels at the view's focal length. Poses: the clip's chase track.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

RS = Path(os.environ.get("RS", Path.home() / "dev/reality-splat")).resolve()
FOCAL = 450 / np.tan(np.radians(25))  # 1440x900 at vfov 50
HALF_H = np.degrees(np.arctan(720 / FOCAL))


def reach(frames_json, t0, t1, ref, out, threshold=1.0):
    fr = [f for f in json.loads(Path(frames_json).read_text())["frames"] if t0 <= f["t"] <= t1]
    C = np.array([f["T_world_cam"] for f in fr])[:, :3, 3]
    fwd = np.array([f["T_world_cam"] for f in fr])[:, :3, 2]
    yaw = np.degrees(np.arctan2(fwd[:, 1], fwd[:, 0]))
    t = np.array([f["t"] for f in fr])
    k = int(np.argmin(np.abs(t - ref)))
    c0 = C[k]
    travel = float(np.linalg.norm(np.diff(C[:, :2], axis=0), axis=1).sum())
    res = dict(stretch=[t0, t1], ref=float(t[k]), camera_travel_m=round(travel, 1),
               max_offset_from_ref_m=round(float(np.linalg.norm(C - c0, axis=1).max()), 1),
               yaw_range_deg=[round(float(yaw.min()), 1), round(float(yaw.max()), 1)], focal_px=round(FOCAL, 1),
               threshold_px=threshold, distances={})
    az = np.radians(np.arange(-180, 180, 0.5))
    for D in (150, 300, 600, 1000, 3000, 10000, 30000):
        worst = np.zeros(len(az))
        seen = np.zeros(len(az), bool)
        for el in np.radians([0.0, 1.5, 3.0]):
            d0 = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.full_like(az, np.sin(el))], 1)
            P = c0 + D * d0
            for i in range(len(fr)):
                v = P - C[i]
                v /= np.linalg.norm(v, axis=1, keepdims=True)
                ang = np.arccos(np.clip((v * d0).sum(1), -1, 1))
                rel = (np.degrees(np.arctan2(v[:, 1], v[:, 0])) - yaw[i] + 180) % 360 - 180
                vis = np.abs(rel) < HALF_H
                worst = np.where(vis, np.maximum(worst, ang * FOCAL), worst)
                seen |= vis
        w = worst[seen]
        res["distances"][str(D)] = dict(max_px=round(float(w.max()), 1), median_px=round(float(np.median(w)), 1),
                                        share_under_threshold=round(float((w < threshold).mean()), 4))
        print(D, res["distances"][str(D)], flush=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "distances"}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", default="work/genfill/clip/c296/frames.json")
    ap.add_argument("--t0", type=float, default=300.0)
    ap.add_argument("--t1", type=float, default=316.0)
    ap.add_argument("--ref", type=float, default=308.0)
    ap.add_argument("--threshold", type=float, default=1.0)
    ap.add_argument("--out", default="work/envmap/300-316/reach.json")
    a = ap.parse_args()
    reach(RS / a.frames, a.t0, a.t1, a.ref, RS / a.out, a.threshold)
