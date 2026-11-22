"""Released footprint displacement, static-null fit and tri-state decisions.

Range is weak along a monocular sightline: use an anisotropic error model and
require a metre-scale result over >=1s, not flow of the box relative to ego.
Scores describe strength of evidence, not calibrated probabilities.
"""

from collections import defaultdict
from functools import lru_cache
from pathlib import Path
import pickle, cv2, numpy as np
from .demo import R


@lru_cache(32)
def road(root, c, n):
    p = Path(root) / "masks/road" / c / f"{n:06}.png"
    return cv2.imread(str(p), 0) if p.exists() else None


def context(r):
    v = np.array([np.cos(r["ego_yaw"]), np.sin(r["ego_yaw"])])
    d = r["xy"] - r["C"][:2, 3]
    f = v @ d
    l = abs(v[0] * d[1] - v[1] * d[0])
    gr = road(r["root"], r["c"], r["n"])
    x, y = map(int, r["foot"])
    sl = (
        gr[min(468, y + 1) : min(469, y + 9), max(0, x - 4) : min(724, x + 5)]
        if gr is not None
        else None
    )
    onroad = float(np.mean(sl > 0)) if sl is not None and sl.size else None
    return (
        "lane-ahead"
        if f > 2 and l < 2.8 and (onroad is None or onroad > 0.35)
        else "roadside"
        if onroad is not None and onroad < 0.25
        else "unresolved"
    )


def measure(rr):
    if len(rr) < 4 or rr[-1]["rt"] - rr[0]["rt"] < 1.0:
        return None
    keep = np.unique(np.linspace(0, len(rr) - 1, min(16, len(rr))).astype(int))
    rr = [rr[i] for i in keep]
    t = np.array([r["rt"] for r in rr])
    dt = t - t.mean()
    pt = np.array([r["xy"] for r in rr])
    pt0 = pt.mean(0)
    rg = np.array([r["range"] for r in rr])
    d = np.array([r["dir"] for r in rr])
    O = np.array([r["C"][:2, 3] for r in rr])
    A = np.stack(
        [
            d / np.maximum(1.2, rg[:, None] * 0.17),
            d[:, ::-1] * [-1, 1] / np.maximum(0.50, rg[:, None] * 0.025),
        ],
        axis=1,
    )
    target = np.einsum("nij,nj->ni", A, pt - pt0).ravel()
    A = A.reshape(-1, 2)
    B = np.c_[A, A * np.repeat(dt, 2)[:, None]]

    def solve(X):
        w = np.ones(len(target))
        for _ in range(3):
            b = np.linalg.lstsq(X * w[:, None], target * w, rcond=None)[0]
            res = X @ b - target
            w = 1 / np.sqrt(np.maximum(1, abs(res) / 1.3))
        return b, np.median(np.linalg.norm(res.reshape(-1, 2), axis=1))

    fixed, es = solve(A)
    moving, em = solve(B)
    vel = moving[2:]
    speed = np.linalg.norm(vel)
    disp = speed * np.ptp(t)
    advance = O[-1] - O[0]
    rel = (pt[-1] - O[-1]) - (pt[0] - O[0])
    matched = (
        np.linalg.norm(advance) > 4
        and np.linalg.norm(rel) < 0.30 * np.linalg.norm(advance)
        and abs(vel @ (advance / np.linalg.norm(advance)) - speed) < 2.2
    )
    move = disp > 2.2 and 1.6 < speed < 37 and es > max(0.95, em * 1.9) and em < 2.7
    if matched and disp > 3 and es > max(0.65, 1.45 * em):
        move = True
    park = speed < 1.25 and disp < 1.85 and es < 1.1
    return dict(
        state="moving" if move else "parked" if park else "unknown",
        confidence=round(
            float(
                np.clip(0.52 + 0.12 * (es - em) + 0.05 * min(speed, 4), 0.53, 0.97)
                if move
                else np.clip(0.94 - 0.16 * es - 0.105 * speed, 0.52, 0.96)
                if park
                else 0.5
            ),
            3,
        ),
        baseline_s=round(float(np.ptp(t)), 3),
        displacement_m=round(float(disp), 2),
        world_velocity=np.round(vel, 2).tolist(),
        stationary_residual=round(float(es), 2),
        moving_residual=round(float(em), 2),
        matched_speed=bool(matched),
    )


def classify(h):
    hist = defaultdict(list)
    last = {}
    votes = {}
    out = []
    for r in h["rows"]:
        c = r["c"]
        k = c, r["root"], r["id"]
        t = r["release"]
        rr = hist[k]
        rr[:] = [a for a in rr if r["rt"] - a["rt"] <= 2.3]
        rr.append(r)
        ctx = context(r)
        if t - last.get(k, -999) > 0.18:
            last[k] = t
            v = measure(rr)
            votes[k] = (t, v)
        v = votes.get(k, (0, None))[1]
        vv = dict(v) if v else dict(state="unknown", confidence=0.5, baseline_s=0.0)
        other = [
            v
            for ck, (tm, v) in votes.items()
            if ck[0] != c and t - tm < 0.38 and v and v["state"] != "unknown"
        ]
        agree = [
            z
            for z in other
            if vv["state"] == z["state"]
            and np.linalg.norm(np.subtract(z["world_velocity"], vv["world_velocity"]))
            < 4.0
        ]
        if agree:
            vv["confidence"] = min(0.99, vv["confidence"] + 0.08)
        if any(z["state"] != vv["state"] for z in other) and vv["state"] != "unknown":
            vv.update(state="unknown", confidence=0.5)
        vv.update(
            c=c,
            n=r["n"],
            id=r["id"],
            root=r["root"],
            t=r["t"],
            release=t,
            context=ctx,
            camera_agreement=len(agree),
        )
        out.append(vv)
    return out


def run():
    hs = pickle.load(open(R / "identities.pickle", "rb"))
    out = {h["id"]: classify(h) for h in hs}
    pickle.dump(out, open(R / "footprint.pickle", "wb"))
    print("classified", len(hs), sum(map(len, out.values())), flush=True)


if __name__ == "__main__":
    run()
