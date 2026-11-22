"""Rigid, full-lifetime *hindsight* splats for the three v5 slices. No screen plates."""

from .standing import OUT, BANK, DST, PARENT, EGO, load, bodyfit, physical_tokens
from .demo import road_check
from .tracking import Factors, make_track
from .model import ply, projection, INF
from .shape import samples
from .appearance import material
from .photographic import calibration, metadata
from .inventory import W
from .online import camera
from .packets import iou
from .upgrade import smooth_corrections
from collections import defaultdict
from pathlib import Path
from scipy.spatial import cKDTree
import cv2 as cv, numpy as np, json, pickle

# Decisions from projected static field plus native-camera track review, not merely proximity.
PARK = {
    10401: "far white sedan behind the blue SUV queue",
    4: "silver sedan",
    7: "burgundy sedan",
    492: "distant blue vehicle",
    587: "grey Tesla at the light",
    593: "white Tesla across the intersection",
    935: "red SUV at the roadside",
    972: "blue SUV at the corner",
    1091: "silver SUV stationary at end of observation",
}
KEEP = {
    47: "same silver sedan as 4 across the island",
    497: "background behind trunks; a car is not established",
    498: "occluded sliver at a concrete support; class and heading indeterminate",
    502: "three distant dark fragments split by a post; class not established",
    504: "hedge and pale facade; no reliable car outline",
    589: "static splats already describe the bonnet, roof, flank and wheels",
    770: "roadside board overlaps the mask; vehicle extent not established",
    789: "same blue SUV as 972",
    976: "pickup in the moving queue; poor fixed fit; do not freeze",
    1040: "merged near and far white cars; near car creeps forward (17); far car separately fitted using two clean mask IDs (10401)",
}
EXTRA = {
    562: "blue crossing off-roader",
    709: "white sedan ahead",
    773: "dark SUV ahead",
    787: "silver SUV ahead",
    1062: "dark SUV after the turn",
    1074: "white sedan after the turn",
}


def tokenrows(rr):
    # Prefer a broad complete witness on each face, not a mosaic of the sharpest pixels.
    groups = defaultdict(list)
    for r in rr:
        if r["key"]:
            groups[r["c"]].append(r)
    for c, g in groups.items():
        sel = []
        for r in sorted(
            g, key=lambda q: -(q["bb"][2] - q["bb"][0]) * (q["bb"][3] - q["bb"][1])
        ):
            if all(abs(r["t"] - q["t"]) > 0.11 for q in sel):
                sel.append(r)
            if len(sel) == 14:
                break
        yield from sel


def texture(spec, rr, tr, sz, uid):
    cls = spec["class"]
    pitch = 0.071 if cls == "bus" else 0.053
    P, N, S, Q, M, patch = samples(cls, sz, pitch)
    L = len(P)
    camdonor = np.full(L, -1)
    rgb = np.zeros((L, 3))
    scores = np.zeros(L)
    views = []
    paint = []
    roof = []
    wheel = np.isin(M, ["tyre", "rim", "hub"])
    top = (N[:, 2] > 0.65) & (P[:, 2] > 0.73 * sz[2])
    side = np.argmax(abs(N), axis=1)
    surfs = side * 2 + (N[np.arange(L), side] > 0)
    pcal = calibration(False)
    for r in tokenrows(rr):
        c, n = r["c"], r["n"]
        cam = camera(c)
        uv, ok, face, z = projection(cam, EGO, r["t"], tr, P, N)
        v = ok & (z > 0.3)
        if v.sum() < 24:
            continue
        b = np.array(r["bb"])
        ext = np.r_[uv[v].min(0), uv[v].max(0)]
        bb = np.clip(ext, [0, 0, 0, 0], [724, 469, 724, 469])
        fit = iou(bb, b)
        if fit < 0.24:
            continue
        wh = np.maximum(1, bb[2:] - bb[:2])
        bw = b[2:] - b[:2]
        tight = (
            (bb[:2] > 1)
            & (bb[2:] < [722, 467])
            & (b[:2] > 1)
            & (b[2:] < [722, 467])
            & (bw / wh > 0.72)
            & (bw / wh < 1.4)
        )
        xy = (
            (uv - (bb[:2] + bb[2:]) * 0.5)
            * np.where(tight, np.clip(bw / wh, 0.84, 1.18), 1)
            + (bb[:2] + bb[2:]) * 0.5
            + np.where(tight, (b[:2] + b[2:] - bb[:2] - bb[2:]) * 0.5, 0)
        )
        u = np.clip(xy[:, 0].astype(int), 0, 723)
        y = np.clip(xy[:, 1].astype(int), 0, 468)
        root = Path(r.get("root", W / "extra" if spec.get("input") == "extra" else "."))
        mask = (
            cv.imread(str(root / "masks/dyn" / c / f"{n:06}.png"), -1) == r["id"]
        ).astype("B")
        inner = cv.erode(mask, np.ones((3, 3), "B")) if mask.sum() > 120 else mask
        valid = (
            v
            & (face > 0.22)
            & (xy[:, 0] > 0)
            & (xy[:, 0] < 724)
            & (xy[:, 1] > 0)
            & (xy[:, 1] < 469)
        )
        depth = np.full((469, 724), 1e5, "f")
        np.minimum.at(depth, (y[valid], u[valid]), z[valid].astype("f"))
        visible = (
            valid
            & (inner[y, u] > 0)
            & (
                z
                < cv.erode(depth, np.ones((3, 3), "B"))[y, u]
                + 0.13
                + np.maximum(z, 1) * 0.013
            )
        )
        if visible.sum() < 25:
            continue
        # Read the native photograph in precisely the stitch teacher's corrected RGB convention;
        # the viewer, not this producer, applies the Natural grade and haze.
        im = cv.imread(f"data/drive/frames/{c}/{n:06}.jpg")[:, :, ::-1]
        im = pcal.apply(c, n, im)
        im = cv.GaussianBlur(im, (5, 5), 0.95)
        col = im[
            np.clip(np.int32(xy[:, 1] * im.shape[0] / 469), 0, im.shape[0] - 1),
            np.clip(np.int32(xy[:, 0] * im.shape[1] / 724), 0, im.shape[1] - 1),
        ]
        j = len(views)
        views.append((c, n, r["id"], float(r["rt"])))
        # One score per face/frame suppresses checkerboarding and stretched glancing samples.
        for f in range(6):
            part = surfs == f
            hit = part & visible
            if hit.sum() < 10:
                continue
            quality = float(
                np.median(face[hit]) ** 1.7
                * np.sqrt(hit.sum() / part.sum())
                * np.median(cam.params[0] / np.maximum(z[hit], 1))
                * fit**1.3
            )
            use = hit & (scores < quality)
            camdonor[use] = j
            scores[use] = quality
            rgb[use] = col[use]
    obs = camdonor >= 0
    if not obs.any():
        raise ValueError(("no registered photograph", uid))
    body = obs & ~wheel & ~top & (P[:, 2] < sz[2] * 0.57) & (P[:, 2] > sz[2] * 0.25)
    paint = np.median(rgb[body if body.sum() > 40 else obs], axis=0)
    roof = np.median(rgb[obs & top], axis=0) if (obs & top).sum() > 35 else paint.copy()
    prior = material(P, N, M, cls, sz, paint, roof, 0)
    # Tyres slide through the image-plane contact contour even in good body fits. Keep their
    # measured samples only if tyre-dark; use the rubber prior instead of baking asphalt.
    obs &= ~((M == "tyre") & (rgb.max(1) > 88))
    rgb[~obs] = prior[~obs]
    mirror = P * [1, -1, 1]
    _, nn = cKDTree(P).query(mirror)
    other = (~obs) & obs[nn] & (abs(P[:, 1]) > sz[1] * 0.3) & ~wheel
    rgb[other] = 0.92 * rgb[nn[other]] + 0.08 * prior[other]
    rgb[M == "tyre"] = np.minimum(rgb[M == "tyre"], [53, 55, 56])
    tt = np.full((L, 7), INF)
    tt[:, 0] = tr["t"][0] - 0.5
    tt[:, 2] = tr["t"][-1]
    A = np.c_[
        P,
        N,
        (rgb / 255 - 0.5) / 0.28209479,
        np.where(obs, 2.7, 2.25),
        np.log(S * np.array([1.09, 1.09, 1.0])),
        Q,
        tt,
    ]
    f = f"objects/h4-{uid}"
    ply(DST / (f + ".ply"), A[obs])
    ply(DST / (f + ".pred.ply"), A[~obs])
    cv.setNumThreads(1)
    return f, dict(
        measured=int(obs.sum()),
        inferred=int((~obs).sum()),
        paint_rgb=paint.tolist(),
        views=[
            dict(cam=c, n=n, mask_id=i, t=t, cells=int((obs & (camdonor == j)).sum()))
            for j, (c, n, i, t) in enumerate(views)
            if (obs & (camdonor == j)).any()
        ],
    )


def lane_track(tr, stopped=False):
    from scipy.optimize import isotonic_regression
    from scipy.ndimage import gaussian_filter1d

    xy = np.stack([tr["x"], tr["y"]], axis=1)
    tt = np.asarray(tr["t"])
    direction = xy[-1] - xy[0]
    direction /= np.linalg.norm(direction)
    left = np.array([-direction[1], direction[0]])
    at = xy.mean(0)
    a = isotonic_regression((xy - at) @ direction).x
    b = (xy - at) @ left
    if stopped:
        hold = tt < 288.15
        a[hold] = np.median(a[hold])
        b[hold] = np.median(b[hold])
        a = np.maximum.accumulate(a)
    a = gaussian_filter1d(a, 10, mode="nearest")
    b = gaussian_filter1d(b, 24, mode="nearest")
    fit = at + a[:, None] * direction + b[:, None] * left
    slope = np.clip(np.gradient(b) / np.maximum(0.06, np.gradient(a)), -0.24, 0.24)
    yaw = np.arctan2(direction[1], direction[0]) + np.arctan(slope)
    return dict(tr, x=fit[:, 0], y=fit[:, 1], yaw=yaw)


def run():
    from .publish import source_tiles

    DST.mkdir(exist_ok=True)
    source_tiles("copy", PARENT, DST, [])
    groups = load()
    tokens = physical_tokens()
    fits = pickle.load(open(OUT / "parked.pickle", "rb"))
    oset = []
    audit = {}
    decision = json.load(open(OUT / "parked.json"))
    missing = []
    for o in json.load(open(BANK / "hindsight/objects.json"))["objects"]:
        uid = o["id"]
        s, rr, tr, sz, *_ = pickle.load(open(BANK / f"fit-{uid}.pickle", "rb"))
        if uid == 9:
            tr = lane_track(tr, True)
        if uid == 17:
            rr = sorted(
                sum(
                    (
                        tokens[p]
                        for p in [
                            ("front", 1709),
                            ("front", 1742),
                            ("left_pillar", 755),
                            ("left_repeater", 1466),
                        ]
                    ),
                    [],
                ),
                key=lambda r: r["rt"],
            )
            f = Factors(rr, "car", "stitched")
            i = np.unique(np.linspace(0, f.n - 1, 40).astype(int))
            mid = f.t[i].mean()
            g, scale, _ = f.fit(i, mid)
            tt = np.arange(f.t.min(), f.t.max() + 1e-6, 1 / 36)
            xy = g[:2] + (tt - mid)[:, None] * g[2:]
            ix = np.searchsorted(f.t, tt).clip(0, f.n - 1)
            tr = dict(
                t=tt,
                x=xy[:, 0],
                y=xy[:, 1],
                z=f.height(ix, xy),
                yaw=np.full(len(tt), np.arctan2(g[3], g[2])),
                state=np.zeros(len(tt), int),
            )
            sz = f.size * scale
        oset.append((uid, s, rr, tr, sz, False))
    for u, name in EXTRA.items():
        rr = sorted(groups["moving", u], key=lambda r: r["release"])
        if u != 562:
            q = rr[len(rr) // 2]
            full = tokens[q["c"], q["id"]]
            chunks = np.split(
                np.arange(len(full)),
                np.where(np.diff([r["t"] for r in full]) > 0.5)[0] + 1,
            )
            ch = max(chunks, key=len)
            rr = [full[i] for i in ch]
        f = Factors(rr, "car", "stitched")
        v = make_track(f, "stitched")
        if not v:
            missing.append(name)
            continue
        tr = smooth_corrections(v[0])
        tr = lane_track(tr) if u != 562 else tr
        oset.append(
            (20000 + u, {"name": name, "class": "car"}, rr, tr, f.size * v[1], False)
        )
    for u, d in decision.items():
        i = int(u)
        d["review"] = PARK.get(i, KEEP.get(i))
        assert d["review"]
        d["added_as"] = 10000 + i if i in PARK else None
    for i, name in PARK.items():
        s, rr, tr, sz, *_ = fits[i]
        extra = {4: 47, 972: 789}.get(i)
        if extra is not None:
            rr = rr + fits[extra][1]
        if i == 593:
            rr = [r for r in rr if r["id"] == 603]
            _, tr, sz, _ = bodyfit(rr)
        if i in [
            587,
            593,
            1091,
        ]:  # stopped in a traffic lane rather than permanently parked
            ta = min(q["rt"] for q in rr)
            tb = max(q["rt"] for q in rr)
            tr = dict(tr, t=np.array([ta - 0.48, tb + 0.48]))
        oset.append((10000 + i, dict(s, name=name), rr, tr, sz, True))
    manifest = []
    for uid, s, rr, tr, sz, parked in oset:
        t0, t1 = tr["t"][0], tr["t"][-1]
        f, info = texture(s, rr, tr, sz, uid)
        o = dict(
            id=uid,
            name=s["name"],
            **{"class": s["class"]},
            size=np.round(sz, 5).tolist(),
            file=f + ".ply",
            predicted_file=f + ".pred.ply",
            t_first=float(t0),
            t_last=float(t1),
            track={k: np.asarray(v).round(6).tolist() for k, v in tr.items()},
        )
        ground = road_check(PARENT, o)
        o["contact_shadow"] = metadata((s, rr, tr, sz, {}, tr), [], t0, t1)[
            "contact_shadow"
        ]
        o["photometry"] = dict(
            space="ungraded corrected camera RGB, SH DC",
            source="stitched-ground-v2",
            viewer_grade_baked=False,
        )
        o["variant"] = "hindsight"
        if parked:
            assert all(
                np.ptp(o["track"][k]) < 1e-6 for k in ["x", "y", "z", "yaw", "state"]
            ) and not any(o["track"]["state"])
        audit[uid] = dict(name=s["name"], parked=parked, ground=ground, **info)
        manifest.append(o)
        print(uid, s["name"], info["measured"], info["inferred"], flush=True)
    (OUT / "world-traffic.json").write_text(
        json.dumps(
            dict(objects=audit, parked=decision, unsupported_moving=missing), indent=2
        )
    )
    p = DST / "objects.next"
    p.write_text(json.dumps(dict(objects=manifest), separators=(",", ":")))
    p.replace(DST / "objects.json")
    print("published", len(manifest))


if __name__ == "__main__":
    run()
