"""Check real pseudo-view ingestion and compare teacher to static host tiles at identical poses."""

import argparse
import hashlib
from .teacher import (
    D,
    K,
    RS,
    cv,
    np,
    J,
    savej,
    seq,
    View,
    camera_image,
    resize,
    hi,
    remap,
    stations,
    select,
    schedule,
    OW,
    OH,
    L,
    H,
)
from pipeline.genfill.views import Scene


def review_poses():
    cs = [e for e in stations() if e["kind"] == "chase"]
    return [min(cs, key=lambda e: abs(e["t"] - t)) for t in np.linspace(85, 140, 20)]


def splats():
    scene = Scene("scene/stitched-opt-v3")
    kw = np.array([[K[0], 0, K[2]], [0, K[1], K[3]], [0, 0, 1]])
    out = D / "review/splat"
    out.mkdir(parents=True, exist_ok=True)
    for e in review_poses():
        val, _, _ = scene.render(np.array(e["T_world_cam"]), kw, (OW, OH), e["t"])
        assert cv.imwrite(
            str(out / (e["id"] + ".png")),
            val[:, :, ::-1].clip(0, 255).astype("B"),
            [cv.IMWRITE_PNG_COMPRESSION, 1],
        )
    print("rendered 20 static host comparisons", flush=True)


def measure(e, img, splat, weight, depth):
    conf = weight >= 128
    V = View(e["t"], np.array(e["T_world_cam"], "f"), 50)
    ts, ns, poses, arc = seq("front")
    base, n, _, _, _ = select(e)
    ids = np.flatnonzero(
        (abs(ts - ts[base]) < 1.1)
        & (np.arange(len(ns)) != base)
        & (np.arange(len(ns)) != n)
    )
    held = {q["n"] for q in schedule()["front"] if q["hold"]}
    available = [
        i
        for i in ids
        if (RS / "maps/movers-v1/remove/front" / f"{ns[i]:06}.png").exists()
    ]
    assert available
    # Prefer an acquisition excluded from the stitch fit, within 2 m. Otherwise
    # select an unused front exposure roughly 0.8 m away. Not a chase-height truth image.
    obs = min(
        available,
        key=lambda i: (
            0 if ns[i] in held and abs(arc[i] - arc[base]) < 2.0 else 1,
            abs(abs(arc[i] - arc[base]) - 0.8),
            abs(i - base),
        ),
    )
    z = resize(100 / np.maximum(depth.astype("f"), 1), L, H)
    gt, uv, _, ok = camera_image(V, obs, z)
    real = remap(gt, hi(uv))
    valid = cv.erode(conf.astype("B"), np.ones((17, 17), "B")).astype(bool) & (
        hi(ok.astype("f")) > 0.999
    )
    diff = lambda a: (
        float(abs(a[valid].astype("f") - real[valid]).mean()) if valid.any() else None
    )
    g = lambda a: cv.cvtColor(a, cv.COLOR_BGR2GRAY).astype("f")
    hpf = lambda a: abs(g(a) - cv.GaussianBlur(g(a), (0, 0), 1.3))
    gttex = hpf(real)
    m = valid & (gttex > 3)
    sharp = lambda a: float(hpf(a)[m].mean()) if m.any() else None
    # Split observations by agreement with the analytic ground depth, so a
    # large road surface cannot hide weak evidence for the structures above it.
    isground = abs(100 / np.maximum(depth.astype("f"), 1) - hi(V.gr)) < 0.0004
    regions = {}
    for name, mask in [("ground", valid & isground), ("upright", valid & ~isground)]:
        if mask.sum() < 200:
            continue
        terr = abs(img[mask].astype("f") - real[mask]).mean(1)
        serr = abs(splat[mask].astype("f") - real[mask]).mean(1)
        regions[name] = dict(
            pixels=int(mask.sum()),
            teacher_mae=float(terr.mean()),
            splat_mae=float(serr.mean()),
            teacher_over20=float((terr > 20).mean()),
        )
    # Choose a much smaller native-resolution crop with real detail inside the
    # intersection of confident support and independent visibility; prefer uprights.
    q = cv.boxFilter((valid * np.minimum(gttex, 16)).astype("f"), -1, (320, 150))
    q[:76] = 0
    q[-76:] = 0
    q[:, :161] = 0
    q[:, -161:] = 0
    q[: int(0.59 * OH)] *= 1.6
    cy, cx = np.unravel_index(q.argmax(), q.shape)
    sl = np.s_[cy - 75 : cy + 75, cx - 160 : cx + 160]
    return dict(
        check_frame=int(ns[obs]),
        check_t=float(ts[obs]),
        check_held=int(ns[obs]) in held,
        check_baseline_m=float(np.linalg.norm(poses[obs, :3, 3] - poses[base, :3, 3])),
        check_pixels=int(valid.sum()),
        regions=regions,
        teacher_mae=diff(img),
        splat_mae=diff(splat),
        teacher_detail=sharp(img),
        splat_detail=sharp(splat),
        check_detail=sharp(real),
        crop=[int(cx - 160), int(cy - 75), 320, 150],
    ), sl


def panel(e, img, splat, weight, depth):
    m, sl = measure(e, img, splat, weight, depth)
    views = [img, cv.cvtColor(weight, cv.COLOR_GRAY2BGR), splat]
    w, h = 420, 263
    card = np.full((h + 216, w * 3, 3), 23, "B")
    x, y, cw, ch = m["crop"]
    for j, (a, label) in enumerate(
        zip(views, ["Front teacher", "RGB weight  |  white = 1", "stitched-opt-v3"])
    ):
        small = cv.resize(a, (w, h), interpolation=cv.INTER_AREA)
        cv.rectangle(
            small,
            (int(x * w / OW), int(y * h / OH)),
            (int((x + cw) * w / OW), int((y + ch) * h / OH)),
            (55, 180, 250),
            1,
        )
        card[34 : 34 + h, j * w : (j + 1) * w] = small
        card[h + 43 : h + 193, j * w + 50 : j * w + 370] = a[sl]
        cv.putText(
            card, label, (j * w + 10, 22), 0, 0.53, (238, 238, 238), 1, cv.LINE_AA
        )
    cv.putText(
        card,
        f"{e['t']:.3f} s | {e['id']} | confident {100 * e['confident_fraction']:.1f}% | orange box shown at native size below",
        (10, h + 209),
        0,
        0.47,
        (230, 230, 230),
        1,
        cv.LINE_AA,
    )
    return card, m


def review():
    rows = J(D / "manifest.json")
    rmap = {e["id"]: e for e in rows}
    missing = J(D / "rejected.json")
    records = []
    pages = []
    for task in review_poses():
        e = rmap[task["id"]]
        img = cv.imread(e["rgb"])
        w = cv.imread(e["weight"], 0)
        dep = cv.imread(e["depth"], -1)
        sp = cv.imread(str(D / "review/splat" / (e["id"] + ".png")))
        card, a = panel(e, img, sp, w, dep)
        path = D / "review" / (e["id"] + ".jpg")
        assert (
            cv.imwrite(str(path), card, [cv.IMWRITE_JPEG_QUALITY, 91])
            and path.stat().st_size < 500000
        )
        records.append(
            dict(
                id=e["id"],
                t=e["t"],
                confident_fraction=e["confident_fraction"],
                audit=e["audit"],
                solo_audit=e["solo_audit"],
                shift9_audit=e["shift9_audit"],
                **a,
            )
        )
        pages.append(cv.resize(card, (630, 240)))
    for i in range(4):
        assert cv.imwrite(
            str(D / "review" / f"sheet-{i + 1}.jpg"),
            np.vstack(pages[5 * i : 5 * i + 5]),
            [cv.IMWRITE_JPEG_QUALITY, 86],
        )
    savej(D / "review/comparison.json", records)
    stats = {}
    for kind in sorted({e["kind"] for e in rows}):
        es = [e for e in rows if e["kind"] == kind]
        stats[kind] = dict(
            count=len(es),
            conf_min=min(e["confident_fraction"] for e in es),
            conf_med=float(np.median([e["confident_fraction"] for e in es])),
            conf_max=max(e["confident_fraction"] for e in es),
        )
    edges, twins = np.array([e["audit"] for e in rows]).sum(0)
    worst = max(rows, key=lambda e: e["audit"][1] / max(e["audit"][0], 1))
    cs = [e["confident_fraction"] for e in rows]
    summary = dict(
        manifest_sha256=hashlib.sha256((D / "manifest.json").read_bytes()).hexdigest(),
        stations=sum(e["kind"] == "chase" for e in rows),
        views=len(rows),
        rejected=len(missing),
        stats=stats,
        min_conf=min(cs),
        median_conf=float(np.median(cs)),
        max_conf=max(cs),
        edges=int(edges),
        twins=int(twins),
        positive=np.sum([r["shift9_audit"] for r in records], axis=0).tolist(),
        solo=np.sum([r["solo_audit"] for r in records], axis=0).tolist(),
        worst=dict(id=worst["id"], audit=worst["audit"]),
        witness_win=sum(
            r["teacher_mae"] < r["splat_mae"]
            for r in records
            if r["teacher_mae"] is not None
        ),
        detail_win=sum(
            r["teacher_detail"] > r["splat_detail"]
            for r in records
            if r["teacher_detail"] is not None
        ),
        teacher_mae=float(
            np.median(
                [r["teacher_mae"] for r in records if r["teacher_mae"] is not None]
            )
        ),
        splat_mae=float(
            np.median([r["splat_mae"] for r in records if r["splat_mae"] is not None])
        ),
    )
    savej(D / "review/result.json", summary)
    print(summary, flush=True)


def ingest():
    from pipeline.stitch.v2fit import pseudofile, crop

    rows = J(D / "manifest.json")
    rng = np.random.default_rng(13)
    ids = list(
        dict.fromkeys(
            [r["id"] for r in review_poses()]
            + [
                next(r["id"] for r in rows if r["kind"] == k)
                for k in sorted({e["kind"] for e in rows})
            ]
        )
    )
    es = {e["id"]: e for e in rows}
    held = {e["n"] for e in schedule()["front"] if e["hold"]}
    assert len(es) == len(rows)
    assert all(
        e["base_frame"] not in held and e["neighbor_frame"] not in held for e in rows
    )
    for e in rows:
        for k in ["rgb", "weight", "sky", "depth"]:
            assert __import__("os").path.getsize(e[k]) > 100
    for k in ids:
        e = es[k]
        im, w = pseudofile(e["rgb"], e["weight"])
        sk = cv.imread(e["sky"], -1)
        dep = cv.imread(e["depth"], -1)
        assert (
            im.shape == (OH, OW, 3) and dep.dtype == np.uint16 and sk.dtype == np.uint8
        )
        assert (
            not np.any(w[sk > 0]) and not np.any(dep[w == 0]) and np.all(dep[w > 0] > 0)
        )
        assert np.max(im[(sk == 0) & (w == 0)]) == 0
        assert abs(np.mean(w >= 0.5) - e["confident_fraction"]) < 1e-8
        args = crop(e, rng, True)
        assert args is not None and args[-1]["weak"].max().item() > 0
        assert np.allclose(args[0], e["T_world_cam"])
    savej(
        D / "review/ingest.json",
        dict(
            manifest_entries=len(rows),
            existing_files=len(rows) * 4,
            decoded_and_cropped=len(ids),
            samples=ids,
            source_holds_excluded=True,
            sky_weight_zero=True,
            unobserved_rgb_zero=True,
            masked_depth_zero=True,
        ),
    )
    print("actual v2fit.pseudofile and v2fit.crop passed", len(ids), "poses")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["splats", "review", "ingest"])
    globals()[p.parse_args().action]()
