"""Side-by-side playback, source/PLY sharpness audit and independent camera footprints."""

import json
from concurrent.futures import ThreadPoolExecutor
import cv2 as cv
import numpy as np
from .render import Scene, TIMES, label, save, RS
from pipeline.photo.sample import CAMS, NAMES, frame, read_mask
from pipeline.common.rig import Trajectory
from .geometry import ground_to_pixels

OUT = RS / "pipeline/road/report/v2"
WORK = RS / "maps/ground-v2"
DEBUG = WORK / "debug"
PALETTE = [
    (255, 112, 78),
    (124, 188, 247),
    (244, 205, 72),
    (244, 132, 201),
    (147, 208, 103),
    (201, 145, 255),
]


def masks():
    snap = json.loads((WORK / "mask-index.json").read_text())
    return {
        (i, z): {int(n): p for n, p in paths.items()}
        for i, v in enumerate(snap)
        for z, paths in v.items()
    }


def footprint(cam, i, k, s, world, origin, fwd, left, index, tr):
    """Independent public rig evaluation. No visibility bits or native-fuser API."""
    xy = world.reshape(-1, 2)
    dv = xy - origin
    lat = dv @ left
    lon = dv @ fwd
    maxr = 82 if i == 0 else 30
    tf = s["tc"][k, i]
    uv, ok, _ = ground_to_pixels(cam, tr, tf, xy, maxr)
    ok = (
        ok.reshape(-1)
        & (abs(lat) <= 20)
        & ~((abs(lat) < 1.05) & (lon > -0.98) & (lon < 3.75))
    )
    ok &= (lon >= 0) if i == 0 else True
    scale = 1 if i == 0 else 2
    w = (cam.width + scale - 1) // scale
    h = (cam.height + scale - 1) // scale
    b = cv.erode(
        cv.resize(cam.mask.astype("uint8"), (w, h), interpolation=cv.INTER_NEAREST),
        np.ones((3, 3), "uint8"),
    )
    bad, _ = read_mask(index, i, s["ns"][k, i], (w, h))
    if bad is not None:
        b &= 1 - bad
    x = np.clip(np.floor(uv[:, 0] * w / cam.width - 0.5).astype(int), 0, w - 1)
    y = np.clip(np.floor(uv[:, 1] * h / cam.height - 0.5).astype(int), 0, h - 1)
    return (ok & (b[y, x] != 0)).reshape(world.shape[:-1])


def memory(v, target, s, tr, index):
    k = int(np.argmin(abs(s["t"] - target)))
    t = float(s["t"][k])
    p = v.pose(t)
    f = np.array([np.cos(p[3]), np.sin(p[3])])
    left = np.array([-f[1], f[0]])
    right = -left
    w, h = 760, 590
    span = 28.0
    pic, _ = v.draw(t, "top", span=span, mode="clock", size=(w, h))
    xx, yy = np.meshgrid(np.arange(w) + 0.5 - w / 2, np.arange(h) + 0.5 - h / 2)
    world = p[:2] + (xx[:, :, None] * right - yy[:, :, None] * f) * span / w
    masklist = []
    for i, c in enumerate(CAMS):
        cm = footprint(c, i, k, s, world[::2, ::2], p[:2], f, left, index, tr)
        cm = cv.resize(cm.astype("uint8"), (w, h), interpolation=cv.INTER_NEAREST)
        masklist.append(cm)
        cnt, _ = cv.findContours(cm, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(pic, cnt, -1, PALETTE[i], 1, cv.LINE_AA)

    def pixel(x, y):
        return (int(w / 2 - y * w / span), int(h / 2 - x * w / span))

    cv.polylines(
        pic,
        [
            np.array(
                [
                    pixel(x, y)
                    for x, y in [
                        (-0.98, -1.05),
                        (3.75, -1.05),
                        (3.75, 1.05),
                        (-0.98, 1.05),
                    ]
                ]
            )
        ],
        True,
        (245, 245, 245),
        2,
        cv.LINE_AA,
    )
    a, _ = v.local(t, p, 14)
    dv = a[:, :2] - p[:2]
    u = (dv @ right * w / span + w / 2).astype(int)
    vv = (-dv @ f * w / span + h / 2).astype(int)
    valid = (u >= 0) & (u < w) & (vv >= 0) & (vv < h)
    under = (abs(dv @ left) < 0.65) & (dv @ f > -0.45) & (dv @ f < 2.9)
    last = np.minimum(t, a[:, 19])
    for j in [20, 22]:
        last = np.where(
            t >= a[:, j], np.maximum(last, np.minimum(t, a[:, j + 1])), last
        )
    age = t - last
    union = np.maximum.reduce(masklist).astype("uint8")
    grow = cv.dilate(union, np.ones((15, 15), "uint8"))
    erode = cv.erode(union, np.ones((15, 15), "uint8"))
    ii = np.where(valid)[0]
    outside = ii[grow[vv[ii], u[ii]] == 0]
    inside = ii[erode[vv[ii], u[ii]] != 0]
    gap = outside[~under[outside]]
    result = dict(
        time=t,
        under_n=int(under.sum()),
        under_age50=float(np.median(age[under])) if under.any() else None,
        under_remembered=float((age[under] > 0.04).mean()) if under.any() else None,
        interior_n=len(inside),
        interior_fresh=float((age[inside] < 0.04).mean()) if len(inside) else None,
        gap_n=len(gap),
        gap_remembered=float((age[gap] >= 0.04).mean()) if len(gap) else None,
    )
    band = np.full((70, w, 3), 26, "uint8")
    for i, n in enumerate(NAMES):
        cv.putText(band, n, (8 + i * 124, 22), 0, 0.44, PALETTE[i], 1, cv.LINE_AA)
    cv.putText(
        band,
        "Age: green <40 ms   amber <250 ms   rose <1 s   violet >1 s   dark >25 s",
        (8, 49),
        0,
        0.50,
        (218, 222, 224),
        1,
        cv.LINE_AA,
    )
    save(
        OUT / f"memory-{target:.1f}.jpg",
        np.vstack(
            (
                label(
                    pic,
                    f"Live v2 | {t:.3f} s | camera outlines from native calibration, white = car",
                ),
                band,
            )
        ),
    )
    return result


def detail(scenes, times, s, tr, index, tag):
    # Same road-relative 6–15 m grid in the native frame and in each export.
    w = h = 360
    span = 9.0
    xx, yy = np.meshgrid(np.arange(w) + 0.5 - w / 2, np.arange(h) + 0.5 - h / 2)
    rank = -1
    for t in times:
        k = np.argmin(abs(s["t"] - t))
        t = float(s["t"][k])
        p = scenes[1].pose(t)
        fw = np.array([np.cos(p[3]), np.sin(p[3])])
        right = np.array([fw[1], -fw[0]])
        centre = p[:2] + fw * 10.5
        world = centre + (xx[:, :, None] * right - yy[:, :, None] * fw) * span / w
        c = CAMS[0]
        uv, ok, _ = ground_to_pixels(c, tr, s["tc"][k, 0], world, 24.0)
        ma, _ = read_mask(index, 0, s["ns"][k, 0], (c.width, c.height))
        im = frame(0, s["ns"][k, 0], 1)[:, :, ::-1]
        src = cv.remap(
            im, *(uv[:, :, i].astype("f") - 0.5 for i in [0, 1]), cv.INTER_LINEAR
        )
        ok &= c.mask_at(uv)
        uvInt = np.clip(
            np.rint(uv - 0.5).astype(int), [0, 0], [c.width - 1, c.height - 1]
        )
        if ma is not None:
            ok &= ma[uvInt[:, :, 1], uvInt[:, :, 0]] == 0
        inside = cv.erode(ok.astype("uint8"), np.ones((17, 17), "uint8")) != 0
        grey = cv.cvtColor(src, cv.COLOR_RGB2GRAY).astype("f")
        hi = grey - cv.GaussianBlur(grey, (0, 0), 10)
        score = np.sum((hi > 15) & inside)
        if score > rank:
            best = (t, src, ok, inside)
            rank = score
    t, src, ok, inside = best
    src[~ok] = [20, 24, 29]
    pics = [label(src, f"Front frame | 6-15 m | {t:.2f} s")]
    high = lambda im: (lambda grey: grey - cv.GaussianBlur(grey, (0, 0), 10))(
        cv.cvtColor(im, cv.COLOR_RGB2GRAY).astype("f")
    )
    ref = high(src)
    metrics = {}
    for name, v in zip(["Live v1", "Live v2", "Stitched v2"], scenes):
        p = v.pose(t)
        fw = np.array([np.cos(p[3]), np.sin(p[3]), 0])
        im, _ = v.draw(t, "top", centre=p[:3] + 10.5 * fw, span=span, size=(w, h))
        pics.append(label(im, name + " | 6-15 m"))
        hit = im.max(2) > 37
        surf = cv.erode(hit.astype("uint8"), np.ones((9, 9), "uint8")) != 0
        val = high(im)
        ix = inside & surf
        paint = (val > 12) & surf
        dil = cv.dilate(paint.astype("uint8"), np.ones((7, 7), "uint8")) != 0
        stroke = inside & (ref > 15)
        metrics[name] = dict(
            coverage=float((inside & surf).sum() / inside.sum()),
            stroke_recall=float((dil & stroke).sum() / max(stroke.sum(), 1)),
            contrast_correlation=float(np.corrcoef(val[ix], ref[ix])[0, 1])
            if ix.sum() > 10
            else None,
        )
    save(OUT / ("detail-" + tag + ".jpg"), np.hstack(pics))
    return dict(time=t, source_stroke_pixels=int(stroke.sum()), models=metrics)


def proof():
    OUT.mkdir(exist_ok=True, parents=True)
    DEBUG.mkdir(exist_ok=True, parents=True)
    sc = [Scene("live", 1), Scene("live", 2), Scene("stitched", 2)]
    s = dict(np.load(WORK / "schedule.npz"))
    tr = Trajectory(RS / "poses/ego_causal.npz")
    index = masks()
    times = sorted(
        TIMES
        + [(318.0, "only-approach"), (320.0, "only-departure"), (321.0, "cross-turn")]
    )
    metrics = {}

    def compare_tone(t):
        # Same 25 m nadir framing, reject paint/void/boundaries in all scenes.
        ims = [v.draw(t, "top", span=20, size=(420, 340))[0] for v in sc]
        ims = [im[:, 100:320] for im in ims]
        mask = np.ones(ims[0].shape[:2], np.uint8)
        for im in ims:
            grey = cv.cvtColor(im, cv.COLOR_RGB2GRAY)
            mask &= (
                (grey > 53)
                & (grey < 174)
                & (np.abs(grey.astype(float) - cv.GaussianBlur(grey, (0, 0), 5)) < 7)
            ).astype(np.uint8)
        mask = cv.erode(mask, np.ones((5, 5), np.uint8)) != 0
        labs = [
            cv.cvtColor(
                cv.GaussianBlur(im, (0, 0), 7).astype("f") / 255, cv.COLOR_RGB2LAB
            )
            for im in ims
        ]
        lightdiff = [(a[..., 0] - labs[-1][..., 0])[mask] for a in labs[:-1]]
        return dict(
            pixels=int(mask.sum()),
            p90_brightness_residual=[
                float(np.percentile(abs(delta - np.median(delta)), 90))
                for delta in lightdiff
            ],
            rgb=[np.median(im[mask], axis=0).tolist() for im in ims],
            median_deltaE76=[
                float(np.median(np.linalg.norm((a - labs[-1])[mask], axis=1)))
                for a in labs[:-1]
            ],
            median_tone_deltaE76=[
                float(np.linalg.norm(np.median((a - labs[-1])[mask], axis=0)))
                for a in labs[:-1]
            ],
        )

    def one(item):
        t, n = item
        pan = []
        measure = {}
        titles = ["Live v1", "Live v2", "Stitched v2"]
        for view in ["top", "chase"]:
            row = []
            for v, title in zip(sc, titles):
                pic, _ = v.draw(t, view, size=(640, 370))
                row.append(label(pic, f"{title} | {view} | {t:.2f} s"))
            pan.append(np.hstack(row))
        save(OUT / (n + ".jpg"), np.vstack(pan))
        for v, title in zip(sc, titles):
            p = v.pose(t)
            f = np.array([np.cos(p[3]), np.sin(p[3])])
            l = np.array([-f[1], f[0]])
            a, _ = v.local(t, p, 19)
            d = a[:, :2] - p[:2]
            mask = (d @ f >= 6) & (d @ f <= 15) & (abs(d @ l) < 3.5)
            a = a[mask]
            rgb = (0.5 + a[:, 6:9] * 0.28209479) * 255
            measure[title] = dict(
                count=len(a),
                sigma_m=np.median(np.exp(a[:, 10:12]), axis=0).tolist()
                if len(a)
                else None,
                rgb=np.median(rgb, axis=0).tolist() if len(a) else None,
            )
        return n, measure

    metrics["views"] = dict(ThreadPoolExecutor(3).map(one, times))
    metrics["tone"] = {str(t): compare_tone(t) for t in [87.9, 93.7, 101.2, 318.6]}
    metrics["detail"] = [
        detail(sc, times0, s, tr, index, tag)
        for tag, times0 in [
            ("exit", np.arange(91, 94.1, 0.25)),
            ("only-early", np.arange(315.3, 318.0, 0.25)),
            ("only", s["t"][(s["t"] >= 318) & (s["t"] <= 322)][::2]),
            ("only-318", [s["t"][np.searchsorted(s["t"], 318.0)]]),
        ]
    ]
    # Fix position, yaw and scale to the early native-source lettering crop.
    # Track that same piece of asphalt from far approach to the front bumper.
    patch_t = metrics["detail"][1]["time"]
    v = sc[1]
    p = v.pose(patch_t)
    yaw = p[3]
    centre = p[:3] + 10.5 * np.array([np.cos(yaw), np.sin(yaw), 0.0])
    timeline = np.arange(patch_t - 7, patch_t + 1, 0.05)
    poses = np.array([v.pose(t) for t in timeline])
    directions = np.stack((np.cos(poses[:, 3]), np.sin(poses[:, 3])), 1)
    ahead = np.sum((centre[:2] - poses[:, :2]) * directions, axis=1)
    panels = []
    refinements = []
    for distance in (60, 40, 20, 10, 5):
        j = int(np.argmin(abs(ahead - distance)))
        t = float(timeline[j])
        im, _ = v.draw(t, "top", centre=centre, span=9, size=(280, 280), fixed_yaw=yaw)
        a, _ = v.local(t, centre, 4.5)
        live = a[(a[:, 17] + 0.125 <= t) & (a[:, 18] > 0.125 + t)]
        sigma = np.median(np.exp(live[:, 10:12]), 0).tolist() if len(live) else None
        panels.append(label(im, f"{t:.2f} s | {ahead[j]:.1f} m ahead"))
        refinements.append(
            dict(time=t, range=float(ahead[j]), splats=len(live), sigma_m=sigma)
        )
    save(OUT / "approach-refinement.jpg", np.hstack(panels))
    metrics["refinement"] = refinements
    metrics["footprints"] = [memory(sc[1], t, s, tr, index) for t in [93.7, 200.2]]
    (DEBUG / "proof.json").write_text(json.dumps(metrics, indent=2))
    bundle(metrics)
    print("proof", metrics["footprints"], flush=True)


def bundle(metrics):
    builds = {
        n: json.loads((WORK / n / "summary.json").read_text())
        for n in ("live", "stitched")
    }
    validator = json.loads((DEBUG / "verified.json").read_text())
    assert all(validator[n]["validated"] == builds[n]["splats"] for n in builds), (
        "Validate these exports before making the report"
    )
    doc = dict(builds=builds, validator=validator, proof=metrics)
    for k, f in [
        ("prefix_tile", "prefix-tile"),
        ("prefix_fit", "causality"),
        ("pose_difference", "pose-diagnosis"),
    ]:
        path = DEBUG / (f + ".json")
        if path.exists():
            doc[k] = json.loads(path.read_text())
    (OUT / "measurements.json").write_text(json.dumps(doc, indent=2) + chr(10))


if __name__ == "__main__":
    proof()
