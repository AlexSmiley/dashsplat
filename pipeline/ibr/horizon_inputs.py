"""Full-drive source knots. Run the existing mask producer only for missing sampled exposures."""

from pathlib import Path
import json, sys, os
import numpy as np, cv2 as cv
from pipeline.livemap.geometry import RS, rig, CAMS, lattice, photo, traj, B, H
from .pack import dump, timed, pose
from scipy.spatial.transform import Rotation
from functools import lru_cache

W = RS / "work/ibr/horizon-l"
SCENE = RS / "scene/ibr-v1"
W.mkdir(parents=True, exist_ok=True)
CAMS = list(CAMS)
C3 = ["front", "left_pillar", "right_pillar"]
drive = json.loads((SCENE / "drive.json").read_text())
ts = np.array(drive["t"])
arc = np.r_[0, np.cumsum(np.hypot(np.diff(drive["x"]), np.diff(drive["y"])))]
END = drive["duration"]


def load(f):
    return json.loads(Path(f).read_text())


def nearest(es, t):
    i = np.searchsorted(es, t)
    i = min(len(es) - 1, i)
    return i - 1 if i > 0 and abs(es[i - 1] - t) < abs(es[i] - t) else i


def make():
    raw = {c: load(RS / "data/drive/cams" / f"{c}.json") for c in CAMS}
    key = load(RS / "depth/keyframes.json")["frames"]
    rts = {c: np.array([e["t"] for e in raw[c]]) for c in CAMS}
    kts = {c: np.array([e["t"] for e in key[c]]) for c in CAMS}
    front = rts["front"]
    s = np.interp(front, ts, arc)
    rows = []
    last = -1
    for i, t in enumerate(front):
        special = 79 <= t <= 141 or 185 <= t <= 216 or 280 <= t <= 331
        spacing = 3.3 if special else 5.5
        if (
            last >= 0
            and s[i] - s[last] < spacing
            and t - front[last] < (1.6 if special else 3.0)
            and i != len(s) - 1
        ):
            continue
        last = i
        sources = {}
        for c in CAMS:
            j = nearest(kts[c], t)
            e = key[c][j] if abs(kts[c][j] - t) < 0.075 else raw[c][nearest(rts[c], t)]
            et = e["t"]
            n = e["n"]
            T = (
                traj("live").T_world_ego(rig[c].row_time(et, rig[c].height * 0.5))
                @ rig[c].T_ego_cam
            )
            sources[c] = dict(
                n=n,
                t=et,
                s=float(np.interp(et, ts, arc)),
                pose=pose(T),
                rowPoses=timed(c, et),
            )
        if rows and sources["front"]["n"] == rows[-1]["front"]["n"]:
            continue
        rows.append(sources)
    dump(W / "knots.json", rows)
    print("source knots", len(rows), flush=True)
    return rows


def maskpath(c, n, k):
    return RS / "masks" / k / c / f"{n:06}.png"


def private(c, n):
    return W / "semantic" / c / f"{n:06}.png"


def picture(c, n, small=False):
    im = cv.imread(
        str(RS / "data/drive/frames" / c / f"{n:06}.jpg"),
        cv.IMREAD_REDUCED_COLOR_2 if rig[c].width > 1500 else cv.IMREAD_COLOR,
    )
    assert im is not None, (c, n)
    return cv.resize(im, (B, H), interpolation=cv.INTER_AREA) if small else im


def masks(c, n):
    p = private(c, n)
    v = lattice(c)[1]
    if p.exists():
        a = cv.imread(str(p), 0)
        return (
            (a == 10) & v,
            np.isin(a, [0, 1, 9]) & v,
            np.isin(a, [11, 12, 13, 14, 15, 17, 18]) | ~v,
        )
    imgs = [cv.imread(str(maskpath(c, n, k)), -1) for k in ["sky", "road", "dyn"]]
    assert all(a is not None for a in imgs), (c, n)
    return (imgs[0] > 0) & v, (imgs[1] > 0) & v, (imgs[2] > 0) | ~v


def priority(t):
    return (
        0 if 78 <= t <= 142 else 1 if 279 <= t <= 332 else 2 if 183 <= t <= 217 else 3
    )


def inputs():
    from pipeline.masks.seg import Segmenter

    rows = make()
    model = None
    stats = {}
    for group in range(4):
        need = []
        shared = 0
        for s in rows:
            if priority(s["front"]["t"]) != group:
                continue
            for c, e in s.items():
                n = e["n"]
                out = private(c, n)
                out.parent.mkdir(parents=True, exist_ok=True)
                if all(maskpath(c, n, k).exists() for k in ["sky", "road", "dyn"]):
                    shared += 1
                    continue
                if not out.exists():
                    need.append((c, n))
        if need and model is None:
            model = Segmenter()
        for i in range(0, len(need), 4):
            batch = need[i : i + 4]
            ms = model([picture(c, n, True)[:, :, ::-1] for c, n in batch])
            for (c, n), o in zip(batch, ms):
                assert cv.imwrite(str(private(c, n)), o["semantic"])
        stats[group] = dict(shared=shared, segmented=len(need))
        print("mask region", group, stats[group], flush=True)
        dump(W / f"ready-{group}.json", stats[group])
    dump(W / "inputs-audit.json", stats)


if __name__ == "__main__":
    inputs()
