"""Compare entire contact tracks and cross-camera stationary landmarks.

For delayed removal, use fit results on the photographed past segment. Each
verdict includes the earliest time its contributing observations are released;
live consumers must NOT apply it to earlier frames.
"""

from collections import defaultdict, Counter
from pathlib import Path
import pickle, numpy as np, cv2
from .online import camera
from .demo import R
from .filter import PacketFactors
from .handoff import robust_class


def fit(rr, cls):
    f = PacketFactors(rr, cls)
    ii = np.unique(np.linspace(0, len(rr) - 1, min(24, len(rr))).astype(int))
    tm = np.mean(f.t[ii])
    x, s, em = f.fit(ii, tm)
    fixed, _, es = f.fit(ii, tm, x, sz=s, stationary=True)
    vel = x[2:]
    return dict(
        x=fixed[:2],
        v=vel,
        tm=tm,
        em=em,
        es=es,
        speed=np.linalg.norm(vel),
        baseline_s=np.ptp(f.t),
        t0=min(f.release),
        at=max(f.release),
    )


def depth_check(rr):
    points = []
    times = []
    ranges = []
    cams = []
    for ix in np.unique(np.linspace(0, len(rr) - 1, min(14, len(rr))).astype(int)):
        r = rr[ix]
        c = r["c"]
        cam = camera(c)
        p = Path("depth") / c / f"{r['n']:06}.npy"
        if not p.is_file():
            continue
        m = (
            cv2.imread(str(Path(r["root"]) / "masks/dyn" / c / f"{r['n']:06}.png"), -1)
            == r["id"]
        )
        y, x = np.where(m)
        if y.size < 12:
            continue
        dep = np.load(p)[y, x].astype(float)
        good = (
            (dep > 1)
            & (dep < 140)
            & (y >= np.percentile(y, 55))
            & (y < np.percentile(y, 93))
        )
        if good.sum() < 12:
            continue
        g = np.flatnonzero(good)[:: max(1, good.sum() // 50)]
        uv = np.stack([x[g] + 0.5, y[g] + 0.5], -1) * [
            cam.width / 724,
            cam.height / 469,
        ]
        ray = cam.unproject(uv)
        P = (ray * (dep[g] / np.maximum(0.12, ray[:, 2]))[:, None]) @ r["C"][
            :3, :3
        ].T + r["C"][:3, 3]
        xy = np.nanmedian(P[:, :2], 0)
        rg = np.linalg.norm(xy - r["C"][:2, 3])
        if not 0.5 < rg / r["range"] < 1.8:
            continue
        points.append(xy)
        times.append(r["rt"])
        ranges.append(rg)
        cams.append(c)
    if len(times) < 4 or np.ptp(times) < 1.0:
        return None
    t = np.array(times)
    p = np.array(points)
    v, xy = np.linalg.lstsq(np.c_[t - t.mean(), np.ones(len(t))], p, rcond=None)[0]
    err = np.median(np.linalg.norm(p - xy - (t - t.mean())[:, None] * v, axis=1))
    return dict(
        speed=float(np.linalg.norm(v)),
        velocity=v.tolist(),
        baseline=float(np.ptp(t)),
        scatter=float(err),
        reliable=bool(err < max(0.75, 0.06 * np.median(ranges))),
    )


def run():
    hs = pickle.load(open(R / "identities.pickle", "rb"))
    groups = defaultdict(list)
    data = []
    for h in hs:
        for r in h["rows"]:
            groups[(r["root"], r["c"], r["id"])].append(r)
    for k, allr in groups.items():
        seq = sorted(allr, key=lambda r: r["rt"])
        chunks = []
        for r in seq:
            if (
                not chunks
                or r["rt"] - chunks[-1][-1]["rt"] > 0.68
                or r["rt"] - chunks[-1][0]["rt"] > 5.8
            ):
                chunks.append([])
            chunks[-1].append(r)
        for rr in chunks:
            if len(rr) < 5 or rr[-1]["rt"] - rr[0]["rt"] < 0.46:
                continue
            cl = robust_class(rr)
            f = fit(rr, cl)
            f.update(
                key=k,
                rows=rr,
                cls=cl,
                color=np.median([r["color"] for r in rr], 0),
                state="unknown",
                confidence=0.50,
                source="single-null",
            )
            f["peers"] = [k]
            if f["baseline_s"] >= 1 and f["speed"] < 1.4 and f["es"] < 1.25:
                f.update(state="parked", confidence=0.79)
            if (
                f["baseline_s"] >= 1
                and 2.1 < f["speed"] < 37
                and f["es"] > max(0.9, 1.5 * f["em"])
                and f["em"] < 2.35
            ):
                f.update(state="moving", confidence=0.83)
            f["depth"] = depth_check(rr)
            data.append(f)
    events = []
    family = {}
    for j, b in enumerate(sorted(data, key=lambda v: v["at"])):
        rr = b["rows"]
        prior = [
            a
            for a in data
            if a["key"][1] != b["key"][1]
            and a["at"] <= b["at"]
            and a["at"] >= b["t0"] - 0.55
            and a["t0"] <= b["at"]
            and np.linalg.norm(a["x"] - b["x"]) < 12
            and np.mean(abs(a["color"] - b["color"]) * [0.22, 1, 1]) < 9
            and a["cls"] == b["cls"]
        ]
        prior.sort(key=lambda a: np.linalg.norm(a["x"] - b["x"]))
        peers = {b["key"][1]}
        join = [b]
        for a in prior:
            if a["key"][1] not in peers:
                join.append(a)
                peers.add(a["key"][1])
            if len(join) == 3:
                break
        for a in join[:]:
            for g in family.get(a["key"], []):
                if (
                    g["key"][1] not in peers
                    and b["at"] - g["t0"] < 9
                    and np.linalg.norm(g["x"] - b["x"]) < 14
                    and np.mean(abs(g["color"] - b["color"]) * [0.22, 1, 1]) < 9
                ):
                    join.append(g)
                    peers.add(g["key"][1])
        if len(join) > 1:
            # Do not treat depth jumps between cameras as object displacement.
            # Re-fit all released silhouettes to ONE still footprint and one
            # translating rigid body, with the same free dimensions.
            z = fit(
                sorted(sum((g["rows"] for g in join), []), key=lambda r: r["rt"]),
                b["cls"],
            )
            speed = z["speed"]
            state = "unknown"
            conf = 0.50
            if (
                z["baseline_s"] >= 1
                and speed < 1.85
                and z["es"] < 2.5
                and z["es"] < max(1.15, 1.32 * z["em"])
            ):
                state = "parked"
                conf = 0.81
            elif (
                z["baseline_s"] >= 1
                and 2.5 < speed < 38
                and z["es"] > max(1.0, 1.48 * z["em"])
                and z["em"] < 2.55
            ):
                state = "moving"
                conf = 0.85
            elif (
                any(g["state"] == "moving" for g in join)
                and z["es"] <= 1.25 * z["em"]
                and speed < 3
            ):
                conf = 0.63
            if state != "unknown" or conf > 0.5:
                for g in join:
                    family[g["key"]] = join.copy()
                for g in join:
                    events.append(
                        dict(
                            key=g["key"],
                            lo=g["t0"] - 0.10,
                            hi=g["rows"][-1]["release"] + 0.10,
                            at=b["at"],
                            state=state,
                            confidence=conf,
                            source="joint-null",
                            cameras=sorted(peers),
                            speed_mps=float(speed),
                            baseline_s=float(z["baseline_s"]),
                            depth_support=[g["depth"] for g in join if g["depth"]],
                        )
                    )
        if b["state"] != "unknown":
            events.append(
                dict(
                    key=b["key"],
                    lo=b["t0"] - 0.1,
                    hi=b["at"] + 0.1,
                    at=b["at"],
                    state=b["state"],
                    confidence=b["confidence"],
                    source="single-null",
                    cameras=[b["key"][1]],
                    speed_mps=float(b["speed"]),
                    baseline_s=float(b["baseline_s"]),
                    depth_support=[b["depth"]] if b["depth"] else [],
                )
            )
    for e in events:
        reliable = [d for d in e["depth_support"] if d["reliable"]]
        opposite = [
            d
            for d in reliable
            if (e["state"] == "parked" and d["speed"] > 3.5)
            or (e["state"] == "moving" and d["speed"] < 1.25)
        ]
        if opposite and len(opposite) == len(reliable):
            if e["source"] == "single-null":
                e.update(state="unknown", confidence=0.5)
            else:
                e["confidence"] = max(0.52, e["confidence"] - 0.12)
        elif reliable and e["state"] != "unknown":
            e["confidence"] = min(0.96, e["confidence"] + 0.04)
    pickle.dump(events, open(R / "landmarks.pickle", "wb"))
    print(
        "independent stationary/translation fits",
        len(data),
        Counter(e["source"] for e in events),
        flush=True,
    )


if __name__ == "__main__":
    run()
