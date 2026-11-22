"""Inspect the actually published static field at each stationary footprint."""

from collections import defaultdict
from pathlib import Path
import json, pickle, cv2 as cv, numpy as np
from scipy.optimize import least_squares
from .demo import W
from .tracking import Factors
from .model import projection
from .shape import silhouette
from .online import camera
from .packets import iou
from .handoff import robust_class
from pipeline.common.rig import Trajectory, T_world_cam

PARENT = Path("scene/stitched-opt-v5")
DST = PARENT.with_name(PARENT.name + "-movers-v1")
BANK = W / "photo-demo-native"
OUT = W / "round4"
OUT.mkdir(exist_ok=True)
SPAN = [(85, 140), (190, 215), (286, 330)]
EGO = Trajectory("poses/ego.npz")
cv.setNumThreads(1)


def load():
    packets = pickle.load(open(W / "photo-full-native/packets.pickle", "rb"))
    q = {(r["c"], r["n"], r["id"], r["root"]): r for _, _, _, rr in packets for r in rr}
    vs = json.load(open("maps/movers-v1/decisions.json"))
    out = defaultdict(list)
    for c, n, i, uid, state, conf, ctx, remove, why, at, root in vs["rows"]:
        r = q.get((c, n, i, root))
        if r and uid is not None and any(a <= r["rt"] <= b for a, b in SPAN):
            out[state, uid].append(
                {k: v for k, v in r.items() if k != "C"}
                | dict(evidence=why, strength=conf)
            )
    return out


def pts(path):
    with open(path, "rb") as f:
        hdr = []
        while (l := f.readline()) != b"end_header" + bytes([10]):
            if not l:
                raise ValueError(path)
            hdr.append(l)
        return np.fromfile(f, "<f4").reshape(
            -1, sum(l.startswith(b"property float ") for l in hdr)
        )


def tiles(xy, size, layer):
    vv = []
    for v in json.load(open(PARENT / "map.json"))["tiles"]:
        if v.get("layer") != layer:
            continue
        b = np.array(v["bbox"])
        rad = max(size[:2]) + 2
        if np.any(b[:2] > xy + rad) or np.any(b[3:5] < xy - rad):
            continue
        a = pts(PARENT / v["file"])
        v = a[np.max(abs(a[:, :2] - xy), 1) <= rad]
        vv.append(v)
    return np.concatenate(vv) if vv else np.empty((0, 24))


def bodyfit(rr):
    rr = sorted(rr, key=lambda r: r["rt"])
    cls = robust_class(rr)
    f = Factors(rr, cls, "stitched")
    idx = np.unique(np.linspace(0, len(rr) - 1, min(28, len(rr))).astype(int))
    sz = f.size.copy()
    base = np.median(f.g[idx, :2], 0)
    ego = rr[len(rr) // 2]["ego_yaw"]
    best = None

    def fun(x, raw=False):
        pos = np.tile(x[:2], (len(idx), 1))
        pred, _ = f.edges(idx, pos, np.zeros(2), x[2], x[3:])
        pred = np.clip(pred, [0, 0, 0, 0], [724, 469, 724, 469])
        wh = f.bb[idx, 2:] - f.bb[idx, :2]
        res = (pred - f.bb[idx]) / np.tile(
            np.maximum([2.0, 1.5], wh * [0.10, 0.085]), 2
        )
        return (
            res
            if raw
            else np.r_[
                res.ravel(), np.sqrt(len(idx)) * (x[3:] - 1) / [0.16, 0.12, 0.13]
            ]
        )

    for ang in [ego, ego + np.pi / 2, ego + np.pi, ego - np.pi / 2]:
        z = least_squares(
            fun,
            np.r_[base, ang, [1, 1, 1]],
            loss="soft_l1",
            bounds=(
                [-np.inf, -np.inf, -20, 0.70, 0.76, 0.7],
                [np.inf, np.inf, 20, 1.4, 1.28, 1.3],
            ),
            max_nfev=55,
        )
        cost = np.mean(np.sqrt(1 + fun(z.x) ** 2) - 1)
        if best is None or cost < best[0]:
            best = (cost, z.x)
    _, x = best
    xy = x[:2]
    yaw = x[2]
    size = sz * x[3:]
    floor = float(np.median(f.height(idx, xy)))
    g = tiles(xy, size, "ground")
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    near = g[
        (np.linalg.norm(g[:, :2] - xy, axis=1) < 2.2) & (abs(g[:, 2] - floor) < 0.55)
    ]
    floor = float(np.median(near[:, 2])) if len(near) > 15 else floor
    t0, t1 = next(z for z in SPAN if z[0] <= rr[len(rr) // 2]["rt"] <= z[1])
    track = dict(
        t=np.array([t0, t1]),
        x=np.repeat(xy[0], 2),
        y=np.repeat(xy[1], 2),
        z=np.repeat(floor, 2),
        yaw=np.repeat(yaw, 2),
        state=np.zeros(2, int),
    )
    return cls, track, size, rot


def crop(im, b):
    x, y, r, bt = b
    w = max(14, r - x) * 1.4
    h = max(10, bt - y) * 1.5
    cx = (x + r) / 2
    cy = (y + bt) / 2
    w = max(w, h * 1.6)
    h = w / 1.6
    x0 = max(0, int(cx - w / 2))
    y0 = max(0, int(cy - h / 2))
    return cv.resize(
        im[y0 : min(469, round(cy + h / 2) + 1), x0 : min(724, round(cx + w / 2) + 1)],
        (192, 120),
    )


def physical_tokens():
    tokens = defaultdict(list)
    for _, c, _, rows in pickle.load(
        open(W / "photo-full-native/packets.pickle", "rb")
    ):
        for r in rows:
            tokens[c, r["id"]].append(r)
    return tokens


def analyse():
    groups = load()
    tok = physical_tokens()
    groups["parked", 10401] = tok["left_pillar", 798] + tok["left_repeater", 1474]
    records = {}
    sheet = []
    fits = {}
    for (state, uid), rr in groups.items():
        if state != "parked":
            continue
        cls, tr, sz, rot = bodyfit(rr)
        center = np.array([tr[k][0] for k in ["x", "y", "z"]])
        static = tiles(center[:2], sz, "objects")
        local = (static[:, :3] - center) @ rot
        hit = (
            (np.abs(local[:, :2]) < sz[:2] * 0.55).all(1)
            & (local[:, 2] > 0.16)
            & (local[:, 2] < sz[2] * 1.12)
        )
        static = static[hit]
        local = local[hit]
        pred = silhouette(cls, sz)
        q = []
        cols = []
        allcams = set()
        rr = sorted(rr, key=lambda r: -r["area"])
        rr0 = []
        for r in rr:
            if r["c"] not in allcams or all(abs(v["t"] - r["t"]) > 0.45 for v in rr0):
                rr0.append(r)
                allcams.add(r["c"])
            if len(rr0) >= 6:
                break
        for r in rr0:
            cam = camera(r["c"])
            m = (
                cv.imread(
                    str(Path(r["root"]) / "masks/dyn" / r["c"] / f"{r['n']:06}.png"), -1
                )
                == r["id"]
            )
            im = cv.resize(
                cv.imread(f"data/drive/frames/{r['c']}/{r['n']:06}.jpg"), (724, 469)
            )
            uv, ok, *_ = projection(cam, EGO, r["t"], tr, pred, pred * 0)
            b = np.clip(np.r_[uv.min(0), uv.max(0)], [0, 0, 0, 0], [724, 469, 724, 469])
            fit = iou(b, r["bb"])
            geom = np.zeros(m.shape, np.uint8)
            field = np.zeros_like(im) + 32
            if len(static):
                C = T_world_cam(cam, r["t"], trajectory=EGO)
                P = (static[:, :3] - C[:3, 3]) @ C[:3, :3]
                uv, ok = cam.project(P)
                uv *= [724 / cam.width, 469 / cam.height]
                col = np.uint8(np.clip(0.5 + 0.28209479 * static[:, 6:9], 0, 1) * 255)[
                    :, ::-1
                ]
                rad = np.clip(
                    np.exp(static[:, 10:13]).max(1)
                    * cam.params[0]
                    * (724 / cam.width)
                    / np.maximum(1, P[:, 2])
                    * 1.45,
                    1,
                    7,
                ).astype(int)
                for i in np.argsort(-P[:, 2]):
                    if not ok[i]:
                        continue
                    x, y = np.int32(np.round(uv[i]))
                    cv.circle(geom, (x, y), rad[i], 255, -1)
                    cv.circle(field, (x, y), rad[i], col[i].tolist(), -1)
            coverage = float(np.mean(geom[m] > 0)) if m.any() else 0.0
            q.append((round(float(fit), 3), round(coverage, 3)))
            bb = r["bb"]
            view = crop(im, bb)
            contour, _ = cv.findContours(
                m.astype("uint8"), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
            )
            cv.drawContours(field, contour, -1, (100, 150, 70), 1)
            cols.append(np.hstack([view, crop(field, bb)]))
        if not cols:
            continue
        banner = np.zeros((28, 768, 3), np.uint8)
        cv.putText(
            banner,
            f"{uid} {min(r['t'] for r in rr):.1f}s  native | v5 at footprint   box IoU / silhouette coverage: {q[:2]}",
            (5, 20),
            0,
            0.49,
            (240, 240, 240),
            1,
        )
        sheet.append(np.vstack([banner, np.hstack((cols + [cols[0]])[:2])]))
        fits[uid] = (
            dict(id=uid, name=f"parked-{uid}", **{"class": cls}),
            rr,
            tr,
            sz,
            {},
            tr,
        )
        records[uid] = dict(
            cameras=sorted(allcams),
            times=[min(r["t"] for r in rr), max(r["t"] for r in rr)],
            center=center.tolist(),
            yaw=float(tr["yaw"][0]),
            size=sz.tolist(),
            footprint_splats=len(static),
            tests=q,
            median_iou=float(np.median([a for a, b in q])),
            median_coverage=float(np.median([b for a, b in q])),
            tokens=sorted({(r["c"], r["id"]) for r in rr}),
        )
        print(uid, len(rr), records[uid]["center"], q, len(static), flush=True)
    for i in range(0, len(sheet), 6):
        cv.imwrite(
            str(OUT / f"parked-sheet-{i // 6}.jpg"),
            np.vstack(sheet[i : i + 6]),
            [cv.IMWRITE_JPEG_QUALITY, 88],
        )
    (OUT / "parked.json").write_text(json.dumps(records, indent=2))
    pickle.dump(fits, open(OUT / "parked.pickle", "wb"))


if __name__ == "__main__":
    analyse()
