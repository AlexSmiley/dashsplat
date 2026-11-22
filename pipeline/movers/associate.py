"""Extend reviewed seed tracks by silhouette overlap, physical velocity and paint appearance."""

import json, cv2, numpy as np
from collections import defaultdict
from pathlib import Path
from .inventory import W
from .tracking import Factors
from .observations import read


def proposals(s, allrows, blocked):
    extra = s.get("input") == "extra"
    root = W / "extra" if extra else Path(".")
    ids = {(c, i) for c, ii in s["fragments"].items() for i in ii}
    by = defaultdict(list)
    for x in allrows:
        if (
            s["window"][0] <= x[1] <= s["window"][1]
            and x[5] > 60
            and (
                (x[4] in ("car", "bus", "truck"))
                == (s["class"] in ("car", "bus", "truck"))
            )
        ):
            by[(x[0], x[3])].append(x)
    rows = sorted([r for id in ids for r in by[id]], key=lambda r: r[1])
    if len(rows) < 5:
        return []

    def obs(x):
        c, t, n, i, cls, area = x[:6]
        bb = np.array(x[6:10], float)
        return dict(
            c=c,
            t=t,
            rt=t,
            release=t,
            bb=bb.tolist(),
            foot=[(bb[0] + bb[2]) / 2, bb[3]],
            id=i,
            n=n,
            key=True,
            depth=0,
        )

    def color(x):
        c, t, n, i, cls, area = x[:6]
        mask = cv2.imread(str(root / "masks/dyn" / c / f"{n:06}.png"), -1) == i
        im = cv2.cvtColor(read(c, n), cv2.COLOR_BGR2LAB)
        pix = im[mask]
        return (
            np.percentile(pix, [30, 58, 82], axis=0) if len(pix) else np.zeros((3, 3))
        )

    f = Factors([obs(x) for x in rows], s["class"], "stitched")
    ii = np.arange(f.n)
    ref = np.mean(f.t)
    pose, sc, err = f.fit(ii, ref, sz=np.ones(3))
    refcols = np.array(
        [color(rows[int(k * (len(rows) - 1))]) for k in [0.28, 0.56, 0.82]]
    )
    joins = []
    for id, rr in by.items():
        if id in blocked or id in ids or len(rr) < 3:
            continue
        ta = np.array([x[1] for x in rr])
        dt = min(abs(ta[:, None] - f.t).ravel())
        if dt > 1.45 or (
            id[0] in s["fragments"]
            and any(any(abs(r[1] - ta) < 0.016) for r in rows if r[0] == id[0])
        ):
            continue
        centroid = np.array([x[10:12] for x in rr])
        pred = pose[:2] + (ta - ref)[:, None] * pose[2:]
        d = np.linalg.norm(pred - centroid, axis=1)
        rg = np.median([x[13] for x in rr])
        if np.median(d) > max(7.5, 0.24 * rg):
            continue
        rtest = [
            rr[j]
            for j in np.unique(np.linspace(0, len(rr) - 1, min(8, len(rr))).astype(int))
        ]
        g = Factors([obs(x) for x in rtest], s["class"], "stitched")
        xy = pose[:2] + (g.t - ref)[:, None] * pose[2:]
        bs, _ = g.edges(np.arange(g.n), xy, pose[2:], np.arctan2(*pose[2:][::-1]), sc)
        bs = np.clip(bs, [0, 0, 0, 0], [724, 469, 724, 469])
        m = g.bb
        inter = np.prod(
            np.maximum(
                0, np.minimum(bs[:, 2:], m[:, 2:]) - np.maximum(bs[:, :2], m[:, :2])
            ),
            axis=1,
        )
        iou = inter / np.maximum(
            1,
            np.prod(bs[:, 2:] - bs[:, :2], axis=1)
            + np.prod(m[:, 2:] - m[:, :2], axis=1)
            - inter,
        )
        cs = color(rr[len(rr) // 2])
        cd = np.min((abs(refcols - cs) * [0.55, 1.0, 1.0]).mean(axis=(1, 2)))
        if np.median(iou) > 0.36 and cd < 12 and np.sum(iou > 0.3) >= 3:
            joins.append(
                dict(
                    cam=id[0],
                    id=id[1],
                    iou=round(float(np.median(iou)), 2),
                    color=round(float(cd), 2),
                    t=[round(ta[0], 2), round(ta[-1], 2)],
                )
            )
    return joins


def run():
    specs = json.loads(Path(__file__).with_name("subjects.json").read_text())
    blocked = defaultdict(set)
    log = {}
    for s in specs:
        for c, ii in s["fragments"].items():
            blocked[s.get("input")].update((c, i) for i in ii)
    for prefix in ["", "extra-"]:
        rows = json.load(open(W / (prefix + "inventory.json")))
        for s in specs:
            if (s.get("input") == "extra") != bool(prefix):
                continue
            joins = proposals(s, rows, blocked[s.get("input")]) if s["id"] > 2 else []
            log[s["name"]] = joins
            for v in joins:
                s["fragments"].setdefault(v["cam"], []).append(v["id"])
                blocked[s.get("input")].add((v["cam"], v["id"]))
            print(s["name"], joins, flush=True)
    json.dump(specs, open(W / "subjects.json", "w"), indent=1)
    json.dump(log, open(W / "associations.json", "w"), indent=1)


if __name__ == "__main__":
    run()
