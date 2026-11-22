"""Retire a coasting duplicate only after concurrent multi-camera evidence.

No label or future end time is used to pick a winner. Viewer fade is scheduled
forward from the decision: its 0.45 s ramp must not anticipate a later shutter.
"""

from collections import defaultdict
from itertools import combinations
import numpy as np
from .online import box
from .packets import iou


def retire(objs, data):
    ds = {d[0]["id"]: d for d in data}
    oo = {o["id"]: o for o in objs}
    times = {i: np.array([r["release"] for r in ds[i][1]]) for i in oo}
    proof = defaultdict(list)
    dropped = {}
    limits = {i: o["t_last"] for i, o in oo.items()}

    def pos(i, t):
        tr = ds[i][2]
        return np.array([np.interp(t, tr["t"], tr[k]) for k in ("x", "y", "yaw")])

    for tick in range(
        int(min(o["t_first"] for o in objs) * 12), int(max(limits.values()) * 12) + 1
    ):
        now = tick / 12
        active = [
            i
            for i, o in oo.items()
            if o["t_first"] <= now <= limits[i] and i not in dropped
        ]
        rows = {}
        pts = {}
        vel = {}
        for i in active:
            k = np.searchsorted(times[i], now, "right") - 1
            if k < 0:
                continue
            rows[i] = ds[i][1][k]
            pts[i] = pos(i, now)
            vel[i] = (pts[i][:2] - pos(i, now - 0.14)[:2]) / 0.14
        for i, j in combinations(rows, 2):
            if i in dropped or j in dropped or oo[i]["class"] != oo[j]["class"]:
                continue
            gap = np.linalg.norm(pts[i][:2] - pts[j][:2])
            dv = np.linalg.norm(vel[i] - vel[j])
            if gap > 3.9 or dv > 7:
                continue
            r, q = rows[i], rows[j]
            key = (i, j)
            a, b = now - r["release"], now - q["release"]
            ts = proof[key]
            ts[:] = [v for v in ts if now - v[0] < 1.2]
            if r["c"] == q["c"]:
                continue
            col = np.mean(abs(r["color"] - q["color"]) * [0.36, 1, 1])
            if (
                max(a, b) < 0.105
                and gap < 1.8
                and dv < 5.5
                and col < 15
                and abs(
                    np.arctan2(
                        np.sin(pts[i][2] - pts[j][2]), np.cos(pts[i][2] - pts[j][2])
                    )
                )
                < 0.42
            ):
                ratios = []
                for v in [r, q]:
                    for uid in [i, j]:
                        p = pos(uid, v["rt"])
                        ratios.append(
                            iou(box(v, p[:2], p[2], oo[uid]["class"]), v["bb"])
                        )
                frames = (r["n"], q["n"])
                if min(ratios) > 0.46 and (not ts or ts[-1][1] != frames):
                    ts.append((now, frames))
            if len(ts) < 3 or ts[-1][0] - ts[0][0] < 0.15:
                continue
            loser, winner = (i, j) if a > b else (j, i)
            if max(a, b) < 0.105 or min(a, b) > 0.09 or max(a, b) - min(a, b) < 0.085:
                continue
            end = min(limits[loser], now + 0.45)
            dropped[loser] = dict(
                surviving_id=winner,
                decision_s=now,
                t_last=end,
                matched_exposure_pairs=len(ts),
            )
            limits[loser] = end
            oo[loser]["t_last"] = end
    return dropped
