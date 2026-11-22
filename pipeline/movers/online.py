"""Online instance discovery, association, robust silhouette updates and 36 Hz motion."""

import pickle
from collections import Counter
from functools import lru_cache
import numpy as np
from scipy.optimize import linear_sum_assignment
from pipeline.common.rig import load_rig
from .inventory import SIZE
from .shape import silhouette
from .packets import iou
from .demo import R


@lru_cache(None)
def outline(cls):
    return silhouette(cls, SIZE[cls])


@lru_cache(None)
def camera(c):
    return load_rig()[c]


def cap(v, maxlen):
    return v * min(1, maxlen / max(1e-8, np.linalg.norm(v)))


def group(c):
    return (
        "vehicle"
        if c in ("car", "bus", "truck")
        else "human"
        if c == "person"
        else "cycle"
    )


def box(o, xy, yaw, cls):
    p = outline(cls)
    cs, sn = np.cos(yaw), np.sin(yaw)
    w = (
        np.c_[p[:, 0] * cs - p[:, 1] * sn, p[:, 0] * sn + p[:, 1] * cs, p[:, 2]]
        + np.r_[xy, o["plane"][:2] @ xy + o["plane"][2]]
    )
    cam = camera(o["c"])
    uv, ok = cam.project((w - o["C"][:3, 3]) @ o["C"][:3, :3])
    uv *= np.array([724 / cam.width, 469 / cam.height])
    return (
        np.clip(np.r_[uv.min(0), uv.max(0)], [0, 0, 0, 0], [724, 469, 724, 469])
        if ok.any()
        else np.zeros(4)
    )


def localize(o, yaw, cls, guess):
    # One image constrains bearing strongly, range weakly. Fit centre and footprint
    # dimensions on the road using a bounded update along the sightline.
    b = o["bb"]
    wh = np.maximum(4, b[2:] - b[:2])
    u = o["dir"]
    J = np.array([u, [-u[1], u[0]]])
    x = guess.copy()
    ds = max(1, o["range"] * 0.06)
    ds2 = 0.32

    def residual(q):
        v = box(o, q, yaw, cls)
        vw = np.maximum(3, v[2:] - v[:2])
        return np.r_[
            ((v[:2] + v[2:] - b[:2] - b[2:]) * 0.5) / np.maximum(2, wh * 0.09),
            np.log(vw / wh) / [0.30, 0.17],
            (u @ (q - o["xy"])) / (2.8 + 0.14 * o["range"]),
        ]

    for _ in range(3):
        v = residual(x)
        jac = np.stack(
            [(residual(x + ds * J[0]) - v) / ds, (residual(x + ds2 * J[1]) - v) / ds2],
            1,
        )
        w = 1 / np.maximum(1, abs(v) / 2)
        step = np.linalg.solve(
            jac.T @ (w[:, None] * jac) + np.eye(2) * 0.009, -jac.T @ (v * w)
        )
        x += np.clip(step, [-ds * 2, -2], [ds * 2, 2]) @ J
    return x


class Candidate:
    def __init__(self, uid, o):
        self.uid = uid
        self.cls = o["cls"]
        self.rows = []
        self.hist = []
        self.scans = []
        self.start = o["release"]
        self.last = self.start
        self.xy = o["xy"].copy()
        self.vel = np.zeros(2)
        self.yaw = o["ego_yaw"]
        self.bycam = {}
        self.color = o["color"].copy()
        self.confirm = None
        self.birth_cls = None
        self.votes = Counter()
        self.rejected = 0
        self.initial = self.xy.copy()
        self.plane = o["plane"]
        self.accept(o, True)

    def predict(self, t):
        return self.xy + self.vel * (t - self.last)

    def score(self, o):
        if group(o["cls"]) != group(self.cls):
            return 99
        age = o["release"] - self.last
        pred = self.predict(o["rt"])
        delta = o["xy"] - pred
        u = o["dir"]
        rg = o["range"]
        along = abs(delta @ u)
        across = abs(delta @ [-u[1], u[0]])
        c = o["c"]
        col = np.mean(abs(self.color - o["color"]) * [0.4, 1, 1])
        prev = self.bycam.get(c)
        same = prev is not None and o["release"] - prev["release"] < 0.65
        if same and col < 19:
            overlap = iou(prev["bb"], o["bb"])
            dtp = o["release"] - prev["release"]
            bc = (o["bb"][:2] + o["bb"][2:]) * 0.5
            bp = (prev["bb"][:2] + prev["bb"][2:]) * 0.5
            wh = np.maximum(8.0, prev["bb"][2:] - prev["bb"][:2])
            pixdist = np.linalg.norm((bc - bp) / wh)
            if overlap > max(0.17, 0.52 - dtp) and pixdist < 0.66:
                return -0.7 - overlap + 0.018 * col + 0.14 * dtp
        if not same and (age > 0.72 and o["range"] > 36):
            return 99
        if self.cls == "bus" and o["cls"] == "car":
            along *= 0.55
        if (
            across > 3.8 + 0.05 * rg + min(7, age * 10)
            or along > 9 + 0.42 * rg + age * 19
            or col > 34
        ):
            return 99
        overlap = iou(prev["bb"], o["bb"]) if same else 0
        pix = iou(box(o, pred, self.yaw, self.cls), o["bb"])
        score = (
            0.60 * across / (1.2 + 0.023 * rg + age * 4)
            + 0.28 * along / (3 + 0.17 * rg + age * 5)
            + 0.36 * col / 12
            + 0.65 * (1 - pix)
            + 0.28 * age
            - 1.2 * overlap
        )
        if same and overlap < 0.02 and pix < 0.08 and age < 0.07:
            return 99
        if not same and (col > 17 or (pix < 0.12 and across > 1.7 + 0.025 * rg)):
            return 99
        return score

    def accept(self, o, first=False):
        t = o["release"]
        dt = max(1 / 36, t - self.last)
        pred = self.predict(t)
        self.votes[o["cls"]] += o["area"] ** 0.5 * (2 if o["cls"] == "bus" else 1)
        cls = max(self.votes, key=self.votes.get)
        yaw = self.yaw
        if cls == "bus" and o["cls"] == "car":
            rawfix = o["xy"] + o["dir"] * min(12, 0.20 * o["range"])
            o = dict(o, xy=rawfix)
        raw = o["xy"]
        self.hist = [q for q in self.hist if q[0] >= t - 0.85]
        hist = self.hist + [(t, raw)]
        if len(hist) >= 3 and t - hist[0][0] > 0.10:
            ts = np.array([x[0] for x in hist])
            v = np.array([x[1] for x in hist])
            a = ts - ts.mean()
            slope = a @ (v - v.mean(0)) / max(0.001, a @ a)
            slope = cap(slope, 33)
            if np.linalg.norm(slope) > 2 and (
                self.confirm is None or np.linalg.norm(self.vel) < 3
            ):
                yaw = np.arctan2(slope[1], slope[0])
        else:
            slope = self.vel
        target = localize(o, yaw, cls, raw if first else pred)
        delta = target - pred
        if (
            not first
            and self.confirm is not None
            and (
                abs(delta @ o["dir"]) > 6 + 0.19 * o["range"] + dt * 10
                or abs(delta @ [-o["dir"][1], o["dir"][0]])
                > 3 + 0.025 * o["range"] + dt * 8
            )
        ):
            self.rejected += 1
            return False
        self.hist = hist
        self.rows.append(o)
        self.cls = cls
        self.color = 0.82 * self.color + 0.18 * o["color"]
        self.bycam[o["c"]] = o
        if first:
            self.xy = target
            self.initial = target.copy()
        else:
            gain = np.clip(dt / 0.16, 0.22, 0.82)
            self.xy = pred + gain * cap(delta, 1.4 + dt * 20)
            dv = cap(slope - self.vel, dt * 30 + 0.4)
            self.vel = cap(self.vel + dv, 34)
        if np.linalg.norm(self.vel) > 1.4:
            self.yaw = np.arctan2(self.vel[1], self.vel[0])
        self.last = t
        self.plane = o["plane"]
        age = t - self.start
        travel = np.linalg.norm(self.xy - self.initial)
        if (
            self.confirm is None
            and len(self.rows) >= 3
            and age >= 0.1
            and travel > max(0.40, 0.012 * o["range"])
            and np.linalg.norm(self.vel) > 1.7
            and sum(r["score"] > 0.965 and r["area"] > 140 for r in self.rows[-8:]) >= 3
        ):
            self.confirm = t
            self.birth_cls = self.cls
        return True


def associate(events):
    live = []
    alltracks = []
    uid = 0
    for t, c, n, obs in events:
        live = [
            h for h in live if t - h.last < (1.35 if h.confirm is not None else 0.65)
        ]
        cost = np.full((len(live), len(obs)), 99.0)
        for j, o in enumerate(obs):
            for i, h in enumerate(live):
                cost[i, j] = h.score(o)
        used = set()
        if cost.size:
            for i, j in zip(*linear_sum_assignment(cost)):
                if cost[i, j] < 1.67 and live[i].accept(obs[j]):
                    used.add(j)
        for j, o in enumerate(obs):
            if j not in used:
                uid += 1
                h = Candidate(uid, o)
                live.append(h)
                alltracks.append(h)
    return alltracks


def run():
    events = pickle.load(open(R / "packets.pickle", "rb"))
    hs = associate(events)
    payload = [
        dict(
            id=h.uid,
            cls=h.birth_cls or h.cls,
            confirm=h.confirm,
            rows=h.rows,
            rejected=h.rejected,
        )
        for h in hs
    ]
    pickle.dump(payload, open(R / "identities.pickle", "wb"))
    print(
        len(hs),
        "hypotheses;",
        sum(h.confirm is not None for h in hs),
        "motion votes;",
        sum(h.rejected for h in hs),
        "outlier rejections",
        flush=True,
    )


if __name__ == "__main__":
    run()
