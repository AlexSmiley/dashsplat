"""Independent triangulation and road checks, fit/evaluate on disjoint tracks and captures."""

from .geometry import *
import argparse, os
from collections import defaultdict

BANDS = [(2, 8), (8, 15), (15, 25), (25, 40), (40, 70)]


def landmarks():
    out = W / "bundle-observations-v2.npz"
    if out.exists():
        return dict(np.load(out))
    s = np.load(RS / "work/poses/schedule.npz")
    st = s["tc"]
    v1 = Trajectory(RS / "poses/ego_v1.npz")
    lt = []
    native = np.load(RS / "work/poses/offline-fit.npz")
    lo = float(os.getenv("AUDIT_START", 79.5))
    hi = float(os.getenv("AUDIT_END", 140.5))
    startid = 0
    for test, p in [(False, native), (True, np.load(RS / "work/poses/hold.npz"))]:
        n = p["num" if test else "n"]
        k = p["k"]
        cc = p["c" if test else "cam"]
        uv = p["uv"]
        ids = np.repeat(np.arange(len(n)), n)
        keep = (st[k, cc] > lo - 7.5) & (st[k, cc] < hi + 8.5)
        relevant = np.unique(ids[keep])
        ii = np.isin(ids, relevant)
        k, cc, uv, ids = k[ii], cc[ii], uv[ii], ids[ii]
        nm = n[relevant]
        off = np.r_[0, nm.cumsum()[:-1]]
        d = np.empty((len(k), 3))
        o = d.copy()
        bundle_xyz = d.copy()
        t = st[k, cc]
        for j, c in enumerate(NAMES):
            ix = cc == j
            C = rig[c]
            T = traj("stitched").T_world_ego(C.row_time(t[ix], uv[ix, 1])) @ C.T_ego_cam
            o[ix] = T[:, :3, 3]
            d[ix] = np.einsum("nij,nj->ni", T[:, :3, :3], C.unproject(uv[ix]))
            if not test:
                # BA xyz is in the v1 gauge. Transport the stored landmark through the
                # original observation camera into the v2 vertical/road gauge.
                old = v1.T_world_ego(C.row_time(t[ix], uv[ix, 1])) @ C.T_ego_cam
                loc = np.einsum(
                    "nji,nj->ni", old[:, :3, :3], p["pts"][ids[ix]] - old[:, :3, 3]
                )
                bundle_xyz[ix] = o[ix] + np.einsum("nij,nj->ni", T[:, :3, :3], loc)
        m = np.eye(3) - d[:, :, None] * d[:, None, :]
        A = np.add.reduceat(m, off)
        rhs = np.add.reduceat(np.einsum("nij,nj->ni", m, o), off)
        xyz = np.linalg.solve(A + np.eye(3) * 1e-8, rhs[..., None])[..., 0]
        point = np.repeat(xyz, nm, axis=0) if test else bundle_xyz
        vec = point - o
        dist = np.linalg.norm(vec, axis=1)
        err = np.linalg.norm(np.cross(vec, d), axis=1) / np.maximum(dist, 0.1)
        par = np.sqrt(np.maximum(np.linalg.eigvalsh(A)[:, 0] / nm, 0))
        ang = np.repeat(par, nm)
        ok = (
            (err < 0.0025)
            & (ang > 0.016)
            & (dist > 1.5)
            & (dist < 80)
            & (np.sum(vec * d, 1) > 0)
            & (t > lo)
            & (t < hi)
        )
        wt = np.minimum(4.0, ang * 70 / np.maximum(dist / 12, 0.5))
        lt.append(
            np.column_stack(
                (
                    cc[ok],
                    t[ok],
                    (ids + startid)[ok],
                    point[ok],
                    wt[ok],
                    np.full(ok.sum(), test),
                )
            )
        )
        startid += len(n)
    a = np.vstack(lt)
    np.savez(out, obs=a)
    return dict(obs=a)


def stats(d):
    if len(d) == 0:
        return {"n": 0}
    med = np.median(d)
    return dict(
        n=len(d),
        median=float(med),
        scatter=float(1.4826 * np.median(abs(d - med))),
        p90abs=float(np.percentile(abs(d), 90)),
    )


def collect(mode, c, obs, plan):
    l = []
    road = []
    ray, _ = lattice(c)
    C = rig[c]
    obs = obs[obs[:, 0] == NAMES.index(c)]
    for e in plan:
        n, t = e["n"], e["t"]
        path = W / mode / "depth" / c / f"{n:06d}.npy"
        if not path.exists() or not (e["train"] or e["hold"]):
            continue
        dist, ok, ro, _ = maps(mode, c, n)
        tw, nc, h = rows(mode, c, t)
        y, x = np.where((ok & ro)[5::16, 7::16])
        y = y * 16 + 5
        x = x * 16 + 7
        den = (nc[y] * ray[y, x]).sum(1)
        ref = -h[y] / np.minimum(den, -1.0e-4)
        z = dist[y, x]
        ii = (den < -0.012) & (ref > 2) & (ref < 70)
        r = np.column_stack(
            (
                np.full(ii.sum(), e["ordinal"]),
                z[ii],
                ref[ii],
                (z * den + h[y])[ii],
                den[ii],
                np.full(ii.sum(), e["hold"]),
            )
        )
        road.extend(r[:: max(1, len(r) // 320)])
        near = obs[abs(obs[:, 1] - t) < 0.18]
        if not len(near):
            continue
        near = near[
            np.unique(near[:, 2], return_index=True)[1]
        ]  # Offline anchor coordinates are used only to diagnose the delivered depth, never in the live map.
        cw = tw if mode == "stitched" else rows("stitched", c, t)[0]
        uv, vis, v, _ = projected(c, near[:, 3:6], cw)
        ix = np.clip(uv[:, 0] * B / C.width, 0, B - 1).astype(int)
        iy = np.clip(uv[:, 1] * H / C.height, 0, H - 1).astype(int)
        rr = np.linalg.norm(v, axis=1)
        good = vis & ok[iy, ix] & (rr >= 2) & (rr < 70)
        ang = np.degrees(np.arccos(np.clip(v[:, 2] / rr, -1, 1)))
        l.extend(
            np.column_stack(
                (
                    np.full(len(rr), e["ordinal"]),
                    dist[iy, ix],
                    rr,
                    ang,
                    near[:, 6],
                    near[:, 7],
                    near[:, 2],
                    np.full(len(rr), e["hold"]),
                )
            )[good]
        )
    return np.array(l).reshape(-1, 8), np.array(road).reshape(-1, 6)


def fit(l, r):
    lt = l[(l[:, 5] == 0) & (l[:, 7] == 0) & (l[:, 2] < 40)]
    rr = r[(r[:, 1] < 10) & (r[:, 2] < 10) & (r[:, 2] >= 3) & (r[:, 5] == 0)]
    anchors = []
    for ord in np.unique(rr[:, 0]):
        sub = rr[rr[:, 0] == ord]
        for lo, hi in [(3, 5), (5, 7), (7, 10)]:
            u = sub[(sub[:, 2] >= lo) & (sub[:, 2] < hi)]
            if len(u) > 3:
                anchors.append([np.median(u[:, 1]), np.median(u[:, 1] / u[:, 2]), 3])
    data = np.vstack(
        (
            np.column_stack((lt[:, 1], lt[:, 1] / lt[:, 2], lt[:, 4])),
            np.array(anchors).reshape(-1, 3),
        )
    )
    x, y, w = data.T
    A = np.stack((np.ones(len(x)), x), 1)
    q = np.array([1.0, 0.0])
    for _ in range(8):
        ww = w / np.maximum(1, abs(A @ q - y) / 0.06)
        pr = np.diag([20.0, max(5000, ww.sum() * 12)])
        q = np.linalg.solve(
            A.T @ (ww[:, None] * A) + pr, A.T @ (ww * y) + pr @ np.array([1.0, 0.0])
        )
        q = np.clip(q, [0.83, -0.003], [1.2, 0.003])
    return q


def run(mode, cams):
    o = landmarks()["obs"]
    plan = frames()
    doc = {}
    for c in cams:
        l, r = collect(mode, c, o, plan[c])
        np.savez(W / mode / (c + "-anchors.npz"), landmarks=l, road=r)
        assert len(l) > 10 and len(r) > 10, (c, len(l), len(r))
        q = fit(l[l[:, 6].astype(int) % 5 != 0], r)
        validation = l[
            (l[:, 5] == 0)
            & (l[:, 7] == 0)
            & (l[:, 6].astype(int) % 5 == 0)
            & (l[:, 2] < 40)
        ]
        err = lambda a, v: np.median(
            abs(v[:, 1] / (a[0] + a[1] * v[:, 1]) / v[:, 2] - 1)
        )
        use = bool(
            len(validation) >= 25
            and err(q, validation) < 0.985 * err([1, 0], validation)
        )
        q = q if use else np.array([1.0, 0.0])
        test = l[(l[:, 5] == 1) & (l[:, 7] == 0) & (l[:, 6].astype(int) % 3 != 0)]
        tr = test[:, 2]
        cr = test[:, 1] / (q[0] + q[1] * test[:, 1])
        rim = test[:, 3] >= 65
        valid = {}
        bundle = l[(l[:, 5] == 0) & (l[:, 7] == 0) & (l[:, 6].astype(int) % 5 == 0)]
        ro = r[r[:, 5] == 0]
        for lo, hi in BANDS:
            i = (tr >= lo) & (tr < hi)
            rr = ro[(ro[:, 2] >= lo) & (ro[:, 2] < hi)]
            bu = bundle[(bundle[:, 2] >= lo) & (bundle[:, 2] < hi)]
            valid[f"{lo}-{hi}"] = dict(
                bundle_raw_m=stats(bu[:, 1] - bu[:, 2]),
                bundle_corrected_m=stats(
                    bu[:, 1] / (q[0] + q[1] * bu[:, 1]) - bu[:, 2]
                ),
                bundle_frac=stats(bu[:, 1] / (q[0] + q[1] * bu[:, 1]) / bu[:, 2] - 1),
                landmark_raw_m=stats(test[i, 1] - tr[i]),
                landmark_m=stats(cr[i] - tr[i]),
                landmark_frac=stats(cr[i] / tr[i] - 1),
                road_raw_height_m=stats(rr[:, 3]),
                road_height_m=stats(
                    rr[:, 3]
                    + (rr[:, 1] / (q[0] + q[1] * rr[:, 1]) - rr[:, 1]) * rr[:, 4]
                ),
            )
        cap = 8
        accepted_bands = []
        for lo, hi in BANDS[:4]:
            v = valid[f"{lo}-{hi}"]
            z = (
                v["landmark_frac"]
                if v["landmark_frac"]["n"] >= 15
                else v["bundle_frac"]
            )
            gd = v["road_height_m"]
            enough = z["n"] >= 12
            accept = (enough and z["p90abs"] <= 0.32 and abs(z["median"]) < 0.12) or (
                not enough
                and hi <= 15
                and gd["n"] >= 25
                and abs(gd["median"]) < 0.22
                and gd["scatter"] < 0.24
            )
            if accept:
                accepted_bands.append([lo, hi])
                cap = max(cap, hi)
        if c == "front":
            z = valid["40-70"]["landmark_frac"]
            bb = valid["40-70"]["bundle_frac"]
            if (
                z["n"] >= 30
                and bb["n"] >= 15
                and max(z["p90abs"], bb["p90abs"]) < 0.32
                and abs(z["median"]) < 0.12
            ):
                accepted_bands.append([40, 70])
                cap = 70
        ray, cal = lattice(c)
        angles = np.degrees(np.arccos(np.clip(ray[:, :, 2][cal], -1, 1)))
        rim_start = 65 if c == "back" else float(np.percentile(angles, 85))
        rim = test[:, 3] >= rim_start
        admit = np.zeros(len(tr), bool)
        for lo, hi in accepted_bands:
            admit |= (cr >= lo) & (cr < hi)
        rimstats = stats(cr[rim] / tr[rim] - 1)
        aperture = stats((cr / tr - 1)[rim & admit])
        limit = float(angles.max() + 0.01)
        limit = min(limit, 78) if c == "back" else limit
        if (
            aperture["n"] < 15
            or aperture["p90abs"] > 0.32
            or abs(aperture["median"]) > 0.13
        ):
            limit = min(limit, rim_start)
        doc[c] = dict(
            a=float(q[0]),
            b=float(q[1]),
            accepted=use,
            fit_observations=int((l[:, 5] == 0).sum()),
            unseen_track_ids=len(np.unique(test[:, 6])),
            validation_n=len(validation),
            max_range=cap,
            bands_admitted=accepted_bands,
            max_angle=limit,
            rim_start=rim_start,
            rim=rimstats,
            admitted_rim=aperture,
            bands=valid,
        )
        print(mode, c, len(l), len(r), "fit", q, "gate", cap, limit, flush=True)
        if mode == "stitched":
            # A lens-wide fit cannot repair a single internally inconsistent DA3 window.
            # Compare independent static-track scale in a local time/range neighbourhood;
            # never use a reserved frame or reserved track to qualify a production packet.
            tt = np.array([e["t"] for e in plan[c]])
            ttL = tt[l[:, 0].astype(int)]
            ttR = tt[r[:, 0].astype(int)]
            profiles = {}
            suppressed = 0
            for e in plan[c]:
                ll = l[(l[:, 5] == 0) & (l[:, 7] == 0) & (abs(ttL - e["t"]) < 0.68)]
                rr = r[(r[:, 5] == 0) & (abs(ttR - e["t"]) < 1.2)]
                bands = []
                for lo, hi in accepted_bands:
                    lo1 = lo * 0.82
                    hi1 = hi * 1.25
                    pred = ll[:, 1] / (q[0] + q[1] * ll[:, 1])
                    z = ll[(pred >= lo1) & (pred < hi1)]
                    z = z[np.argsort(abs(tt[z[:, 0].astype(int)] - e["t"]))]
                    z = z[np.unique(z[:, 6], return_index=True)[1]]
                    err = z[:, 1] / (q[0] + q[1] * z[:, 1]) / z[:, 2] - 1
                    isbad = len(z) >= 4 and (
                        abs(np.median(err)) > 0.17 or np.median(abs(err)) > 0.24
                    )
                    near = abs(tt[z[:, 0].astype(int)] - e["t"]) < 0.5
                    gross = err[near]
                    isbad |= len(gross) >= 2 and (
                        np.mean(gross > 0.33) >= 0.66 or np.mean(gross < -0.30) >= 0.66
                    )
                    if hi <= 15 and len(rr) > 10:
                        zz = rr[(rr[:, 2] >= lo) & (rr[:, 2] < hi)]
                        isbad |= (
                            len(zz) >= 12
                            and np.median(
                                abs(zz[:, 1] / (q[0] + q[1] * zz[:, 1]) / zz[:, 2] - 1)
                            )
                            > 0.22
                        )
                    if not isbad:
                        bands.append([lo, hi])
                    else:
                        suppressed += 1
                profiles[str(e["n"])] = {
                    **{k: doc[c][k] for k in ("a", "b", "max_angle", "max_range")},
                    "bands_admitted": bands,
                }
            (W / mode / (c + "-policy.json")).write_text(json.dumps(profiles))
            doc[c]["packet_bands_quarantined"] = suppressed
    path = W / mode / "audit.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    old.update(doc)
    path.write_text(json.dumps(old, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode")
    p.add_argument("cams", nargs="*")
    a = p.parse_args()
    run(a.mode, a.cams or CAMS)
