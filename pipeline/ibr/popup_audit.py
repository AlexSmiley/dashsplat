"""Private proof audit: immutable tracked planes, load demand and actual ground/contact witnesses."""

from .popup_proof import *
from collections import Counter


def files(e):
    paths = [
        e[k]
        for k in ["sky", "rgb", "alpha"]
        if e.get(k) and (k == "sky" or e["layers"])
    ]
    h = e.get("horizon")
    return paths + ([h["rgb"], h["alpha"]] if h else [])


def audit(r):
    sizes = {}
    records = {}
    total = Counter()
    for c, d in r.bank.items():
        seen = defaultdict(list)
        anchors = Counter()
        n = 0
        for e in d["base"]:
            for a in e["layers"]:
                seen[a["id"]].append(a["plane"])
                n += 1
                anchors[a["anchor"]] += 1
        assert all(np.allclose(x, x[0], atol=1e-8) for x in seen.values())
        assert all(
            len(set(a["id"] for a in e["layers"])) == len(e["layers"])
            for e in d["movers"]
        )
        records[c] = dict(
            bases=len(d["base"]),
            moverExposures=len(d["movers"]),
            quads=n,
            contacts=dict(anchors),
            singleImageIdentities=sum(len(x) == 1 for x in seen.values()),
            identities=len(seen),
        )
        for kind in ["base", "movers"]:
            for e in d[kind]:
                for f in files(e):
                    if f not in sizes:
                        sizes[f] = (DST / f).stat().st_size
                        total[kind] += sizes[f]
    requests = {}
    for start, end in [(93, 104), (105, 120)]:
        seen = set()
        demand = []
        front = []
        for i in range((end - start) * 30):
            t = start + i / 30
            keys = set()
            for c in LENSES:
                for e, w in r.pair(c, t, 0):
                    if w > 0.0001:
                        keys.update(files(e))
                e = r.bank[c]["movers"][np.argmin(abs(r.mtimes[c] - t))]
                if abs(e["t"] - t) < 0.3:
                    keys.update(files(e))
            demand.append(sum(sizes[f] for f in keys - seen))
            seen.update(keys)
        demand = np.array(demand)
        requests[str(start)] = dict(
            uniqueBytes=int(demand.sum()),
            meanMBs=demand.sum() / 1e6 / (end - start),
            peakRollingMB=float(
                np.convolve(demand, np.ones(30, dtype=int), "valid").max() / 1e6
            ),
            uniqueFiles=len(seen),
        )
    report = dict(
        records=records,
        assetBytes=dict(total),
        bytesManifest=sum((DST / (c + "-final.json")).stat().st_size for c in LENSES),
        requests=requests,
    )
    dump(OUT / "bank.json", report)
    return report


def witnesses(r):
    rows = []
    recs = []
    for t in [99 + 8 / 30, 102 + 25 / 30, 112, 135]:
        im, d = r.draw(t)
        T = d["T"]
        st = r.cam.raw[round(t * 30)]
        wr = RAYS @ T[:3, :3].astype("f").T
        ground, _, ga = r.road.layer(
            t, T, r.road.points(t), np.full_like(wr, 155), wr, st
        )
        gg = np.uint8(np.clip(ground, 0, 255))
        scores = []
        for ci, c in enumerate(LENSES):
            e, w = max(r.pair(c, t, 0), key=lambda ew: ew[1])
            for a in e["layers"]:
                if a["kind"] not in [2, 5, 7, 8] or a["anchor"] != "contact":
                    continue
                id = (ci + 1) * 100000 + a["id"]
                y, x = np.where(d["ids"] == id)
                if len(y) < 90:
                    continue
                lower = y >= np.percentile(y, 95)
                cx = int(np.median(x[lower]))
                cy = int(np.median(y[lower]))
                support = float(
                    ga[
                        max(0, cy - 4) : min(HEI, cy + 9),
                        max(0, cx - 5) : min(WID, cx + 6),
                    ].mean()
                )
                score = np.sqrt(len(y)) * (0.15 + support) * (0.7 if cy < 365 else 1)
                scores.append((score, id, a["kind"], cx, cy, support))
        scores.sort(reverse=True)
        info = []
        m = im.copy()
        g = gg.copy()
        for _, idd, kind, x, y, supp in scores[:3]:
            name = {2: "facade", 5: "pole", 7: "sign", 8: "tree"}[kind]
            info.append(dict(id=idd, category=name, uv=[x, y], groundOpacity=supp))
            for pic in [m, g]:
                cv.drawMarker(pic, (x, y), (255, 187, 34), cv.MARKER_CROSS, 18, 2)
                cv.putText(
                    pic,
                    name,
                    (min(x + 8, WID - 90), max(18, y - 7)),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.56,
                    (255, 187, 34),
                    1,
                    cv.LINE_AA,
                )
        rows.append(
            np.hstack(
                [
                    label(cv.resize(pic, (680, 425), interpolation=cv.INTER_AREA), name)
                    for pic, name in [
                        (m, f"{t:.2f} s | visible lower edges"),
                        (g, "same pose | stitched ground alone"),
                    ]
                ]
            )
        )
        recs.append(dict(t=t, contacts=info))
    jpg(OUT / "ground-contact.jpg", np.vstack(rows))
    dump(OUT / "ground-contact.json", recs)


def pose_audit(r):
    from .pack import traj
    from .proof import tagged

    old = Old()
    old.cam = r.cam
    old.fb.cam = r.cam
    out = {}
    angles = {}
    central = {}
    stop = {}
    traffic = {}
    for l, u in [(93, 104), (105, 120)]:
        hit = 0
        both = 0
        gap = None
        gaps = []
        for i in range((u - l) * 30):
            t = l + i / 30
            T = r.cam.pose(t)[0]
            pair = r.pair("front", t, 0)
            dir = T[:3, 2]
            chk = [
                bool(rig["front"].project(dir @ rowmat(e, np.array([0.5]))[0][0])[1])
                for e, w in pair
            ]
            hit += any(chk)
            both += all(chk)
            if not any(chk) and gap is None:
                gap = t
            if any(chk) and gap is not None:
                gaps.append([gap, t])
                gap = None
        if gap is not None:
            gaps.append([gap, u])
        central[str(l)] = dict(
            frames=(u - l) * 30,
            centerInEither=hit,
            centerInBoth=both,
            neitherSpans=gaps,
        )
    for t in [100, 112, 128, 135]:
        T = r.cam.pose(t)[0]
        pair = r.pair("front", t, 0)
        ax = sum(rowmat(e, np.array([0.5]))[0][0, :2, 2] * w for e, w in pair)
        angles[str(t)] = dict(
            headingDifferenceDeg=float(
                np.rad2deg(np.arccos(np.clip(norm(T[:2, 2]) @ norm(ax), -1, 1)))
            ),
            base=[e["t"] for e, w in pair],
            weights=[w for e, w in pair],
            arcLagM=float(r.odom[round(t * 30)] - sum(e["s"] * w for e, w in pair)),
        )
    dr = json.load(open(BASE / "drive.json"))
    j = json.load(open(RS / "depth/keyframes.json"))
    fs = json.load(open(RS / "data/drive/cams/front.json"))
    arc = np.r_[0, np.cumsum(np.hypot(np.diff(dr["x"]), np.diff(dr["y"])))]
    for t in [81.5, 200.0]:
        T, st, _ = r.cam.pose(t)
        tt = np.array([e["t"] for e in fs if t - 65 <= e["t"] <= t])
        o = (traj("live").T_world_ego(tt) @ rig["front"].T_ego_cam)[:, :2, 3]
        ds = np.linalg.norm(o - T[:2, 3], axis=1)
        ix = ds.argmin()
        stop[str(t)] = dict(
            speed=float(st[4]),
            frontPhoto=float(tt[ix]),
            xyGapM=float(ds[ix]),
            pathLagM=float(
                np.interp(t, dr["t"], arc) - np.interp(tt[ix], dr["t"], arc)
            ),
            depthCount=sum(
                (RS / "depth" / c / f"{e['n']:06}.npy").exists()
                for c, ls in j["frames"].items()
                for e in ls
                if abs(e["t"] - t) <= 2
            ),
        )
    for t in [102 + 25 / 30, 131 + 23 / 30]:
        a, b, v, k, T = tagged(old, t, 0)
        front = (old.best[:, :, 3] // 100000) == 0
        traffic[str(t)] = dict(
            allInstancePixels=int((k >= 8).sum()),
            crossingFrontTrack46Pixels=int(
                ((k >= 8) & (((k - 8) // 3) == 46) & front).sum()
            ),
        )
    out = dict(centre=central, stills=angles, stops=stop, meshTraffic=traffic)
    dump(OUT / "pose-audit.json", out)
    print(out)


if __name__ == "__main__":
    r = Popup()
    print(audit(r))
    witnesses(r)
    pose_audit(r)
