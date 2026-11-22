"""Maintain camera-local detector continuity and join tentative handovers online.

A camera/mask token refers to a stream of pixels; it is never used across views.
Two separate tokens visible in one exposure cannot belong to the same mover.
"""

from collections import Counter
import pickle, numpy as np
from scipy.optimize import linear_sum_assignment
from .online import cap, group
from .packets import iou
from .demo import R


def robust_class(rr):
    vote = Counter()
    for r in rr:
        vote[r["cls"]] += (
            np.sqrt(r["area"]) * r["score"] * (2.8 if r["cls"] == "bus" else 1)
        )
    return max(vote, key=vote.get)


def centre(r, cls):
    x = r["xy"]
    rg = r["range"]
    if cls == "bus" and r["cls"] == "car":
        x = x + r["dir"] * min(0.42 * rg, 20)
    return x


def anchor(rr, t, cls):
    last = rr[-1]
    recent = [r for r in rr[-22:] if last["rt"] - r["rt"] < 0.70]
    pos = np.array([centre(r, cls) for r in recent])
    ts = np.array([r["rt"] for r in recent])
    u = ts - ts.mean()
    vel = cap((u @ (pos - pos.mean(0))) / max(0.018, u @ u), 33.0)
    x = np.median(pos + (t - ts)[:, None] * vel, axis=0)
    return x, vel


def cost(h, rr, visible, photo=False):
    o = rr[-1]
    c = o["c"]
    t = o["rt"]
    source = h["local"]
    cl = robust_class(rr)
    oldcl = robust_class(h["rows"][-24:])
    bus = "bus" in (oldcl, cl)
    cls = "bus" if bus else cl
    if group(cl) != group(oldcl):
        return 99.0
    prev = source.get(c, [])
    age = t - h["rows"][-1]["rt"]
    if age > 1.3:
        return 99.0
    if prev and (c, prev[-1]["root"], prev[-1]["id"]) in visible:
        return 99.0
    u = o["dir"]
    rg = o["range"]
    color = np.median([r["color"] for r in rr], axis=0)
    scores = []
    for cam, rows in source.items():
        last = rows[-1]
        dt = t - last["rt"]
        if photo and cam != c:
            from .parallax import cross_cost

            scores.append(cross_cost(rows, rr))
        if dt > 1.3 or dt < -0.08:
            continue
        pred, vel = anchor(rows, t, cls)
        pos, v2 = anchor(rr, t, cls)
        delta = pos - pred
        along = abs(delta @ u)
        across = abs(delta @ [-u[1], u[0]])
        col = np.mean(abs(last["color"] - color) * [0.32, 1, 1])
        overlap = iou(last["bb"], o["bb"]) if cam == c else 0.0
        gap = max(0, dt)
        if cam == c:
            if gap > 0.68 or (overlap < 0.09 and across > 1.1 + 0.018 * rg):
                continue
            if last["cls"] != o["cls"] and bus and overlap < 0.3:
                continue
        if col > (18 if bus else 15):
            continue
        lat = (3.4 if bus else 1.55) + 0.023 * rg + min(4.0, gap * 5)
        ax = (7.0 if bus else 3.2) + 0.18 * rg + min(8, gap * 9)
        if across > lat * 1.7 or along > ax * 1.9:
            continue
        s = (
            0.70 * across / lat
            + 0.52 * along / ax
            + 0.46 * col / 14
            + 0.35 * gap
            - 0.60 * overlap
        )
        if bus and cl != oldcl:
            s += 0.21
        if len(rr) >= 3 and t - rr[0]["rt"] > 0.17 and len(rows) >= 4:
            s += min(0.55, np.linalg.norm(v2 - vel) / 39)
        scores.append(s)
    return min(scores, default=99.0)


def associate(events, photo=False):
    tracks = []
    active = []
    binding = {}
    pending = {}
    uid = 0
    for release, c, n, obs in events:
        active = [h for h in active if release - h["rows"][-1]["release"] < 1.40]
        alive = {h["id"] for h in active}
        current = {(c, o["root"], o["id"]) for o in obs}
        used = set()
        fresh = []
        for o in obs:
            k = (c, o["root"], o["id"])
            h = binding.get(k)
            if (
                h is not None
                and h["id"] in alive
                and release - h["local"][c][-1]["release"] < 0.8
                and h["local"][c][-1]["id"] == o["id"]
                and h["id"] not in used
            ):
                h["rows"].append(o)
                h["local"][c].append(o)
                used.add(h["id"])
                continue
            rr = pending.get(k, [])
            rr = [r for r in rr if release - r["release"] < 0.68] + [o]
            pending[k] = rr
            if len(rr) >= 2 and (
                release - rr[0]["release"] >= 0.045 or o["area"] > 1000
            ):
                fresh.append((k, rr))
        avail = [h for h in active if h["id"] not in used]
        scores = np.array(
            [[cost(h, rr, current, photo) for k, rr in fresh] for h in avail]
        )
        assigned = {}
        if scores.size:
            for i, j in zip(*linear_sum_assignment(scores)):
                if scores[i, j] < 1.27:
                    assigned[j] = avail[i]
        for j, (k, rr) in enumerate(fresh):
            h = assigned.get(j)
            if h is None:
                uid += 1
                h = dict(id=uid, rows=[], local={}, cls=None, confirm=None, rejected=0)
                active.append(h)
                tracks.append(h)
            # Buffered observations become available at the resolving exposure,
            # so a late handover can never rewrite an earlier released position.
            rr = [dict(r, release=release) for r in rr]
            h["rows"].extend(rr)
            h["local"][c] = rr.copy()
            binding[k] = h
            del pending[k]
        for h in active:
            if h["confirm"] is not None or h["rows"][-1]["release"] != release:
                continue
            rr = h["rows"]
            span = release - rr[0]["release"]
            good = sum(r["score"] > 0.955 and r["area"] >= 100 for r in rr[-12:])
            early = (
                photo
                and len(rr) >= 3
                and span >= 0.07
                and sum(r["score"] > 0.86 and r["area"] >= 55 for r in rr[-5:]) >= 2
            )
            if early or (len(rr) >= 3 and span >= 0.08 and good >= 3):
                h["confirm"] = release
                h["cls"] = robust_class(rr)
    return [{k: v for k, v in h.items() if k != "local"} for h in tracks]


def run():
    events = pickle.load(open(R / "packets.pickle", "rb"))
    tracks = associate(events, R.name.startswith("photo-"))
    pickle.dump(tracks, open(R / "identities.pickle", "wb"))
    print(
        "automatic component handover:",
        len(tracks),
        "hypotheses",
        sum(h["confirm"] is not None for h in tracks),
        "candidates",
        flush=True,
    )


if __name__ == "__main__":
    run()
