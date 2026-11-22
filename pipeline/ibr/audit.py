"""Verify the private payload, spatial selection on a real stop, and current-time mover sourcing."""

from .proof import *
from PIL import Image


def audit():
    doc = json.load(open(STAGE / "ibr.json"))
    j = json.load(open(RS / "depth/keyframes.json"))
    rows = {}
    samples = {}
    size = 0
    for c, cs in doc["cameras"].items():
        es = cs["keyframes"]
        assert np.all(np.diff([e["t"] for e in es]) > 0)
        a = np.zeros(3, "int64")
        models = Counter()
        annual = {}
        for e in es:
            assert (
                np.isfinite([e["rowPoses"][0], e["rowPoses"][1]]).all()
                and len(e["groundPlane"]) == 4
            )
            for k, (name, shape) in enumerate(
                [("rgb", (1448, 938)), ("z", (724, 469)), ("mask", (724, 469))]
            ):
                p = STAGE / e[name]
                a[k] += p.stat().st_size
                with Image.open(p) as im:
                    assert im.size == shape and (
                        im.mode == "RGB"
                        if k == 0
                        else im.mode in ["I", "I;16", "I;16B"]
                    )
            models[e["depthModel"]] += 1
            samples[c, e["n"]] = sum(
                (STAGE / e[k]).stat().st_size for k in ["rgb", "z", "mask"]
            )
        for t0, t1 in [(93, 104), (105, 120), (125, 137.25)]:
            annual[f"{t0}-{t1}"] = [
                sum(t0 <= e["t"] <= t1 for e in es),
                sum(t0 <= e["t"] <= t1 for e in j["frames"][c]),
            ]
        rows[c] = dict(
            count=len(es),
            rgb_z_mask_bytes=a.tolist(),
            intervals=annual,
            models=dict(models),
        )
        size += int(a.sum())
    fixed = (
        (STAGE / "ibr.json").stat().st_size
        + (STAGE / "sky.jpg").stat().st_size
        + sum((STAGE / cs["rays"]).stat().st_size for cs in doc["cameras"].values())
    )
    movie_bank = STAGE.parent / "movie-bank.json"
    r = Scene(movie_bank if movie_bank.is_file() else None)
    stream = {}
    for start, end in [(105, 120), (93, 104)]:
        used = set()
        per = []
        loads = []
        for n in range((end - start) * 30):
            t = start + n / 30
            sel = r.select(t, r.cam.pose(t)[0])
            now = {(c, r.frames[c][i]["n"]) for c, i, m in sel}
            bn = sum(samples[k] for k in now - used)
            loads.append(bn)
            used.update(now)
            per.append(len(now))
        stream[f"{start}-{end}"] = dict(
            unique=len(used),
            bytes=sum(loads),
            mean_bps=sum(loads) / (end - start),
            peak_1s_bytes=int(
                np.convolve(loads, np.ones(30, dtype=int), "valid").max()
            ),
            simultaneous_p50_p95=np.percentile(per, [50, 95]).tolist(),
        )
    r = Scene()
    dr = json.load(open(BASE / "drive.json"))
    frames = json.load(open(RS / "data/drive/cams/front.json"))
    stop = {}
    panels = []
    for t in [81.5, 200.0]:
        T, st, _ = r.cam.pose(t)
        cand = [e for e in frames if t - 65 <= e["t"] <= t]
        tt = np.array([e["t"] for e in cand])
        C = traj("live").T_world_ego(tt) @ rig["front"].T_ego_cam
        o = C[:, :3, 3]
        ix = np.linalg.norm(o[:, :2] - T[:2, 3], axis=1).argmin()
        arc = np.r_[0, np.cumsum(np.hypot(np.diff(dr["x"]), np.diff(dr["y"])))]
        s0 = np.interp(t, dr["t"], arc)
        s1 = np.interp(tt[ix], dr["t"], arc)
        depth = sum(
            (RS / "depth" / c / f"{e['n']:06}.npy").exists()
            for c, es in j["frames"].items()
            for e in es
            if t - 2 <= e["t"] <= t + 2
        )
        stop[str(t)] = dict(
            speed=float(st[4]),
            nearest_ground_position_photo_t=float(tt[ix]),
            photo_lag=t - tt[ix],
            path_lag_m=float(s0 - s1),
            ground_xy_gap_m=float(np.linalg.norm(o[ix, :2] - T[:2, 3])),
            production_depths_within_2s=depth,
        )
        if t == 81.5:
            a, b, v, k, P = tagged(r, t, 0)
            jpg(OUT / "stopped-81.5.jpg", a)
            stop[str(t)]["bank_selected_front"] = r.closest.get("front")
            stop[str(t)]["coverage"] = float(v.mean())
    traffic = {}
    for t in [102 + 25 / 30, 131 + 23 / 30]:
        a, b, v, k, T = tagged(r, t, 0)
        p, e = exposure(r, t)
        mask = k >= 8
        confirmed = mask & ((k - 8) % 3 == 2)
        visible = np.unique(r.best[:, :, 3][mask]).astype(int)
        src = {}
        for tag in visible:
            c = CAMS[tag // 100000]
            n = tag % 100000
            frame = next(e for e in r.frames[c] if e["n"] == n)
            src[c] = dict(
                t=frame["t"],
                skew_ms=round((frame["t"] - t) * 1000, 2),
                pixels=int((mask & (r.best[:, :, 3] == tag)).sum()),
            )
        overlay = a.copy()
        overlay[mask] = np.uint8(0.50 * a[mask] + [0, 95, 95])
        overlay[confirmed] = np.uint8(0.50 * a[confirmed] + [127, 55, 0])
        panels.append(
            np.hstack(
                [
                    label(cv.resize(x, (520, 325)), title)
                    for x, title in [
                        (p, f"front at T={t:.2f}"),
                        (a, "IBR chase"),
                        (overlay, "orange: moving | cyan: parked or unknown"),
                    ]
                ]
            )
        )
        traffic[str(t)] = dict(
            pixels=int(mask.sum()),
            confirmed_pixels=int(confirmed.sum()),
            current_sources=src,
        )
    jpg(OUT / "traffic.jpg", np.vstack(panels))
    info = dict(
        cameras=rows,
        payload_bytes=size,
        fixed_bytes=fixed,
        bank_bps=size / (T1 - T0),
        movie_stream=stream,
        stop=stop,
        traffic=traffic,
    )
    dump(OUT / "audit.json", info)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    audit()
