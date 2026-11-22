"""Judge rigid layers at the published viewer pose, separating front source events from other frames."""

from .popup_render import *
from .proof import label, encoder, finish

OUT = RS / "pipeline/ibr/report-chase"
OUT.mkdir(parents=True, exist_ok=True)


def temporal(prev, img, d):
    sl = np.s_[4::8, 4::8]
    lum = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    T = d["T"]
    id = d["ids"][sl]
    z = d["depth"][sl]
    cur = (lum, d["depth"].copy(), d["ids"].copy(), T)
    ans = None
    if prev:
        im, zp, ip, P = prev
        w = (RAYS[sl] @ T[:3, :3].T) * z[:, :, None] + T[:3, 3]
        q = (w - P[:3, 3]) @ P[:3, :3]
        uv = q[:, :, :2] / np.maximum(q[:, :, 2, None], 0.1) * FOC + [
            WID / 2 - 0.5,
            HEI / 2 - 0.5,
        ]
        ix = np.round(uv[:, :, 0]).astype(int).clip(0, WID - 1)
        iy = np.round(uv[:, :, 1]).astype(int).clip(0, HEI - 1)
        inside = (
            (ix > 0) & (iy > 0) & (ix < WID - 1) & (iy < HEI - 1) & (q[:, :, 2] > 0.3)
        )
        err = abs(
            cv.remap(im, uv.astype("f"), None, cv.INTER_LINEAR).astype("f") - lum[sl]
        )
        stat = (id > 0) & (id < 2000000)
        same = stat & (id == ip[iy, ix]) & inside
        geom = abs(zp[iy, ix] - q[:, :, 2]) < np.maximum(0.35, 0.055 * z)
        allstatic = inside & (stat | ((ip[iy, ix] > 0) & (ip[iy, ix] < 2000000)))
        ans = [
            [int(m.sum()), float(err[m].sum()), int((err[m] > 20).sum())]
            for m in [same & geom, allstatic, inside & geom & (id < 2000000)]
        ]
    return cur, ans


def grad(a):
    l = cv.cvtColor(a, cv.COLOR_RGB2GRAY).astype("f")
    return cv.magnitude(cv.Sobel(l, -1, 1, 0), cv.Sobel(l, -1, 0, 1)) * 0.125


def stills(r):
    from .proof import tagged, exposure

    oldscene = Old()
    oldscene.cam = r.cam
    oldscene.fb.cam = r.cam
    stats = []
    for t in [100, 112, 128, 135]:
        raw = r.road.points(t)
        ls = []
        for az in [0, -30, 30]:
            a, d = r.draw(t, az, raw)
            old, b, _, _, oldT = tagged(oldscene, t, az, oldscene.fb.points(t))
            assert np.array_equal(oldT, d["T"])
            photo, physical = exposure(oldscene, t)
            im = np.hstack(
                [
                    label(x, name)
                    for x, name in [
                        (b, "v5"),
                        (old, "IBR 1 | depth meshes"),
                        (a, f"IBR 2 | rigid layers | {t}s {az:+d} deg"),
                        (
                            photo,
                            f"physical front | {physical['t']:.3f}s | not the chase pose",
                        ),
                    ]
                ]
            )
            jpg(OUT / f"{t}-{az:+d}.jpg", im)
            ls.append(cv.resize(im, (1728, 278), interpolation=cv.INTER_AREA))
            ids = d["ids"]
            mask = (ids > 0) & (ids < 2000000) & (d["alpha"] > 0.87)
            mask = cv.erode(mask.astype("B"), np.ones((7, 7), "B")) > 0
            metric = [
                float(np.percentile(grad(x)[mask], 90)) if mask.any() else None
                for x in [b, old, a]
            ]
            stats.append(
                dict(
                    t=t,
                    az=az,
                    p90_gradient=metric,
                    static_fraction=float(((ids > 0) & (ids < 2000000)).mean()),
                    ground_fraction=float((d["ground"] > 0.5).mean()),
                    mover_pixels=int((ids >= 2000000).sum()),
                    frontPair=d["frontPair"],
                    frontTime=d["frontTime"],
                    frac=d["frac"],
                )
            )
        jpg(OUT / f"{t}-sheet.jpg", np.vstack(ls))
        print("stills", t, flush=True)
    dump(OUT / "stills.json", stats)


def traffic(r):
    rows = []
    stats = []
    for t in [102 + 25 / 30, 131 + 23 / 30]:
        a, d = r.draw(t)
        c = "front"
        es = bank()[c]["keyframes"]
        e = min(es, key=lambda e: abs(e["t"] - t))
        src = cv.imread(str(STAGE / e["rgb"]))[:, :, ::-1]
        info = json.load(open(RS / "masks/dyn" / c / f"{e['n']:06}.json"))
        overlay = a.copy()
        ids = d["ids"]
        m = ids >= 2000000
        overlay[m] = overlay[m] * 0.43 + [140, 65, 0]
        targets = []
        for inst in info:
            reg = ids == (3000000 + inst["id"])
            targets.append(
                dict(
                    track=inst["id"],
                    cls=inst["class"],
                    moving=inst.get("moving"),
                    texture_pixels=int(reg.sum()),
                    bbox=inst["bbox"],
                )
            )
        rows.append(
            np.hstack(
                [
                    label(cv.resize(pic, (480, 300)), title)
                    for pic, title in [
                        (src, f"front at {e['t']:.3f}"),
                        (a, "rigid chase"),
                        (overlay, "current instances: orange"),
                    ]
                ]
            )
        )
        stats.append(
            dict(
                t=t, frontTime=e["t"], targets=targets, total_mover_pixels=int(m.sum())
            )
        )
    jpg(OUT / "traffic.jpg", np.vstack(rows))
    dump(OUT / "traffic.json", stats)


def movie(r, name):
    start, end = (105, 120) if name == "road" else (93, 104)
    N = (end - start) * 30
    path = OUT / f"{name}-{start}-{end}.mp4"
    enc = encoder(path, WID, HEI, 30, "13500k")
    sheet = []
    prev = None
    lastKey = None
    lastPair = None
    frames = []
    for i in range(N):
        t = start + i / 30
        if i % 15 == 0:
            raw = r.road.points(t + 7 / 30, 0.28)
        img, d = r.draw(t, raw=raw)
        pair = d["frontPair"]
        key = pair[int(d["frac"] >= 0.5)]
        p, err = temporal(prev, img, d)
        frames.append(
            dict(
                t=t,
                dominantSwitch=lastKey is not None and key != lastKey,
                pairBoundary=lastPair is not None and pair != lastPair,
                frontPair=pair,
                blend=d["frac"],
                frontTime=d["frontTime"],
                err=err,
                static_pixels=int(((d["ids"] > 0) & (d["ids"] < 2000000)).sum()),
                mover_pixels=int((d["ids"] >= 2000000).sum()),
            )
        )
        prev = p
        lastKey = key
        lastPair = pair
        enc.stdin.write(img.tobytes())
        if i % 30 == 0:
            sheet.append(
                label(
                    cv.resize(img, (384, 240), interpolation=cv.INTER_AREA),
                    f"{t:.1f} s",
                )
            )
        if i % 90 == 0:
            print(name, i, "/", N, flush=True)
    info = finish(enc, path, N, 30)
    pad = sheet + [np.zeros_like(sheet[0])] * ((-len(sheet)) % 3)
    jpg(
        OUT / (name + "-sequence.jpg"),
        np.vstack([np.hstack(pad[i : i + 3]) for i in range(0, len(pad), 3)]),
    )
    dump(OUT / (name + ".json"), dict(video=info, frames=frames))
    print(info, flush=True)


if __name__ == "__main__":
    r = Popup()
    mode = sys.argv[1]
    if mode == "stills":
        stills(r)
        traffic(r)
        a, d = r.draw(81.5)
        jpg(
            OUT / "stopped-81.5.jpg",
            label(a, "81.5 s | 0 m/s | chase location not traversed"),
        )
    else:
        movie(r, mode)
