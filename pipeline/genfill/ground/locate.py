"""Draw the offline ego track with time labels on a mosaic preview to pick patch centres."""

import sys
from pathlib import Path

import cv2 as cv
import numpy as np

from pipeline.common.rig import Trajectory

RS = Path.home() / "dev/reality-splat"


def ego_px(mosaic, times, tr):
    z = np.load(Path(mosaic) / "grid.npz")
    p = tr.T_world_ego(np.asarray(times, float))[:, :2, 3]
    return (p - z["origin"]) @ z["basis"] / float(z["cell"])  # column, row


if __name__ == "__main__":
    mosaic, t0, t1, out = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
    tr = Trajectory(str(RS / "poses/ego.npz"))
    img = cv.imread(str(Path(mosaic) / "best.jpg"))
    ts = np.arange(t0, t1 + 1e-6, 1.0)
    cr = ego_px(mosaic, ts, tr)
    s = 1600 / max(img.shape[:2])
    small = cv.resize(img, None, fx=s, fy=s, interpolation=cv.INTER_AREA)
    for t, (c, r) in zip(ts, cr):
        print(f"{t:.1f} col {c:.0f} row {r:.0f}")
        q = (int(c * s), int(r * s))
        cv.circle(small, q, 3, (0, 0, 255), -1)
        cv.putText(small, f"{t:.0f}", (q[0] + 4, q[1] - 4), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    cv.imwrite(out, small)
