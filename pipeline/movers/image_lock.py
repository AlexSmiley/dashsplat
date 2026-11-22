"""Causal steering from the newest silhouette. A long range-biased fit must not
steer through an incoming camera's clear side view. Pose is still speed bounded.
"""

import numpy as np
from .online import box, cap, localize
from .packets import iou


def steer(data):
    sp, rows, tr, sz, stats, scan = data
    release = np.array([r["release"] for r in rows])
    epochs = stats["class_epochs"]
    tick = 1 / 36
    j = 0
    hist = {}
    xy = None
    vel = None
    yaw = None
    anchor = None
    last = -1e4
    lock = None
    last_x = None
    last_cls = None
    scan_out = {k: [] for k in scan}
    si = 0
    for i, td in enumerate(tr["t"]):
        t = td - tick
        raw = np.array([tr["x"][i], tr["y"][i]])
        psi = tr["yaw"][i]
        rv = cap((raw - last_x) / tick, 39.0) if last_x is not None else np.zeros(2)
        last_x = raw.copy()
        cls = next(
            (cl for at, cl in reversed(epochs) if at <= t + 1.0e-7), epochs[0][1]
        )
        if cls != last_cls:
            hist = {}
            last_cls = cls
        while j < len(rows) and release[j] <= t + 1.0e-7:
            r = rows[j]
            j += 1
            if t - r["rt"] > 0.45:
                continue
            u = r["dir"]
            b = r["bb"]
            edge = min(b[0], 724 - b[2])
            key = (r["c"], r["root"], r["id"])
            prev = hist.get(key)
            angles = [psi, r["ego_yaw"]]
            if prev and r["rt"] > prev[0] + 0.012:
                angles.append(prev[4])
            cand = []
            for a in angles:
                p = localize(r, a, cls, r["xy"])
                q = iou(b, box(r, p, a, cls))
                cand.append((q, p, a))
            q, p, a = max(cand, key=lambda v: v[0])
            span = max(0.02, r["rt"] - prev[0]) if prev else 0.02
            if prev and span < 0.55:
                # Alpha-beta in world space, including camera motion. Resist the
                # longitudinal range noise while following lateral bearing promptly.
                pred = prev[1] + prev[2] * span
                delta = p - pred
                dl = delta @ u
                dlat = delta - dl * u
                v = cap(
                    prev[2] + cap((0.32 * dlat + 0.10 * dl * u) / max(0.09, span), 4.2),
                    35.0,
                )
                p = pred + 0.76 * dlat + np.clip(0.30 + span, 0.33, 0.8) * dl * u
            else:
                v = rv.copy() if xy is not None else np.zeros(2)
            hist[key] = (r["rt"], p, v, q, a, r, edge)
        candidates = []
        for key, (rt, p, v, q, a, r, edge) in hist.items():
            age = t - rt
            if age > 0.26 or q < 0.29:
                continue
            pos = p + v * age
            score = (
                q
                * np.exp(-age * 5.0)
                * (0.6 if edge < 2 else 1.0)
                * min(1.0, np.sqrt(r["area"] / 230))
            )
            candidates.append((score, key, pos, v, a, r))
        target = raw
        feed = rv
        ang = psi
        if candidates:
            strength, key, pos, v, a, r = max(candidates, key=lambda q: q[0])
            quality = iou(box(r, raw - rv * (td - r["rt"]), psi, cls), r["bb"])
            if strength > 0.18 and (quality < 0.57 or (key == lock and quality < 0.71)):
                lock = key
                target = pos
                feed = v
                ang = a
                last = t
                anchor = (pos, v, a)
            else:
                lock = None
        elif lock is not None and t - last < 0.52:
            target = anchor[0] + anchor[1] * (t - last)
            feed = anchor[1]
            ang = anchor[2]
        else:
            lock = None
        if xy is None:
            xy = target.copy()
            vel = feed.copy()
            yaw = ang
        desired = cap(feed + (target - xy) * 12, 39.0)
        vel += cap(desired - vel, 5.5)
        xy += vel * tick
        if lock is not None:
            # A box constrains axis but not front/back. Orient it along the
            # supported direction of travel, not toward a noisy range offset.
            if (
                np.linalg.norm(feed) > 2
                and np.dot(feed, [np.cos(ang), np.sin(ang)]) < 0
            ):
                ang += np.pi
        yaw += np.clip(np.arctan2(np.sin(ang - yaw), np.cos(ang - yaw)), -0.13, 0.13)
        tr["x"][i], tr["y"][i], tr["yaw"][i] = *xy, yaw
        while si < len(scan["t"]) and scan["t"][si] <= t + 1e-7:
            for k, b in zip(
                ("t", "x", "y", "vx", "vy", "z"), (t, *xy, *vel, tr["z"][i])
            ):
                scan_out[k].append(b)
            si += 1
    return sp, rows, tr, sz, stats, {k: np.asarray(v) for k, v in scan_out.items()}
