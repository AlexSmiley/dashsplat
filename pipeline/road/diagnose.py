"""Split-route calibration hypotheses; no writes to the calibrated rig."""

from .fit import *
from .geometry import pixels_to_ground


def run():
    ctx = inputs()
    s, d, t, B, stop = ctx
    tr = Trajectory(WORK / "ego.npz")
    cams = [load_rig(RS / "calib/rig.json")[n] for n in NAMES]
    m = np.load(WORK / "pairs.npy")
    m = m[::2]
    parts = []
    far = []
    axes = []
    for side in [0, 4]:
        k, c = m[:, side : side + 2].astype(int).T
        uv = m[:, side + 2 : side + 4]
        G = np.zeros((len(m), 2))
        J = np.zeros((len(m), 2, 6))
        radius = np.zeros(len(m))
        edge = np.zeros(len(m))
        rr = []
        for ci, cam in enumerate(cams):
            ii = np.flatnonzero(c == ci)
            u = uv[ii]
            tc = s["tc"][k[ii], ci]
            time = cam.row_time(tc, u[:, 1])
            E = tr.T_world_ego(time)
            normal, h = tr.ground_plane(time)
            o = cam.T_ego_cam[:3, 3]
            ray = cam.unproject(u) @ cam.T_ego_cam[:3, :3].T

            def xy(ray, E, normal, h):
                dep = -(normal @ o + h) / np.einsum("ni,ni->n", ray, normal)
                v = ray * dep[:, None] + o
                return np.einsum("nij,nj->ni", E[:, :2, :3], v) + E[:, :2, 3]

            g = xy(ray, E, normal, h)
            G[ii] = g
            radius[ii] = np.linalg.norm(g - E[:, :2, 3], axis=1)
            edge[ii] = np.linalg.norm(
                (u - cam.params[2:4]) / [cam.width / 2, cam.height / 2], axis=1
            )
            for a in range(3):
                direction = np.eye(3)[a]
                J[ii, :, a] = (
                    (xy(ray + np.cross(direction, ray) * 0.0001, E, normal, h) - g)
                    / 0.0001
                    * (np.pi / 180)
                )  # per degree of ego mounting rotation
            tp = time + 0.001
            Em = tr.T_world_ego(tp)
            nn, hh = tr.ground_plane(tp)
            J[ii, :, 3] = xy(ray, Em, nn, hh) - g
            J[ii, :, 4] = J[ii, :, 3] * (u[:, 1:2] / cam.height - 0.5)
            offset = u - cam.params[2:4]
            ray2 = (
                cam.unproject(u + offset * (edge[ii] ** 2 * 0.001)[:, None])
                @ cam.T_ego_cam[:3, :3].T
            )
            J[ii, :, 5] = (xy(ray2, E, normal, h) - g) * 10  # 1% edge radial correction
        parts.append((G, J, c))
        far.append(radius)
        axes.append(edge)
    G, J, c = parts[0]
    g, j, cc = parts[1]
    e = G - g
    ran = np.maximum(*far)
    edge = np.maximum(*axes)
    k = m[:, 0].astype(int)
    ke = m[:, 4].astype(int)
    kt = s["t"][k]
    cs = []
    allmat = []
    N = 6
    for i in range(6):
        allmat.append(J * (c == i)[:, None, None] - j * (cc == i)[:, None, None])
    A = np.concatenate(allmat, axis=2)
    sel = (ran <= 10) & (ran > 0) & np.isfinite(e).all(1)
    sig = np.tile([0.18, 0.18, 0.25, 4, 6, 0.5], 6)
    reg = np.diag(1 / sig**2)
    fold = {}
    out = {}
    for region, u in [
        ("early", kt < 196),
        ("late", kt >= 196),
        ("slow", d["speed"][s["ix"][k]] < 9),
        ("fast", d["speed"][s["ix"][k]] >= 15),
    ]:
        ii = sel & u & (m[:, 10] == 0)
        a = A[ii]
        b = e[ii]
        p = np.zeros(N * 6)
        for it in range(6):
            err = b + np.einsum("nij,j->ni", a, p)
            wt = (0.035**2 + np.linalg.norm(err, axis=1) ** 2) ** -0.8
            aw = a * wt[:, None, None] ** 0.5
            aw = aw.reshape(-1, 36)
            rhs = (b * wt[:, None] ** 0.5).ravel()
            p = np.linalg.solve(aw.T @ aw + reg, -aw.T @ rhs)
        fold[region] = p.reshape(6, 6).tolist()
        ev = sel & ~u & (m[:, 10] == 1)
        pred = e[ev] + np.einsum("nij,j->ni", A[ev], p)
        val = lambda x: np.percentile(
            np.linalg.norm(x, axis=1) * 100, [50, 90]
        ).tolist()
        out[region] = [int(ii.sum()), int(ev.sum()), val(e[ev]), val(pred)]
    edge_report = {}
    for name, a0, b0 in SEG:
        q = sel & (m[:, 10] == 1) & (kt >= a0) & (kt <= b0) & (s["t"][ke] >= a0)
        ee = np.linalg.norm(e, axis=1) * 100
        edge_report[name] = {}
        for key, mask0 in [
            ("centre", edge < 0.7),
            ("edge", edge >= 0.9),
            ("0–4 m", ran < 4),
            ("4–7 m", (ran >= 4) & (ran < 7)),
            ("7–10 m", ran >= 7),
        ]:
            z = ee[q & mask0]
            edge_report[name][key] = (
                [len(z), *np.percentile(z, [50, 90]).tolist()]
                if len(z)
                else [0, None, None]
            )
    # Independent scene check: perturb frozen optics by the split's proposed
    # changes and reproject retained 3D landmarks without refitting them.
    from pipeline.poses.validate import Fit
    from pipeline.common.rig import _exp

    fs = Fit().load()
    X = np.load(WORK / "scene-v2.npz")["X"]
    ids = np.repeat(np.arange(len(X)), fs.n)
    native = {}
    for half in ("early", "late", "slow", "fast"):
        native[half] = []
        for i, cam in enumerate(cams):
            ix = np.flatnonzero(fs.cam == i)[::17]
            pix = fs.uv[ix]
            tf = fs.tc[fs.k[ix], i]
            pp = fold[half][i]
            an = np.deg2rad(pp[:3])
            tt = cam.row_time(tf, pix[:, 1])
            world = X[ids[ix]]
            err = []
            for use in (False, True):
                E = tr.T_world_ego(
                    tt + (pp[3] + pp[4] * (pix[:, 1] / cam.height - 0.5)) * 0.001 * use
                )
                P = (
                    np.einsum("nji,nj->ni", E[:, :3, :3], world - E[:, :3, 3])
                    - cam.T_ego_cam[:3, 3]
                )
                q = P @ _exp(an * use) @ cam.T_ego_cam[:3, :3]
                pred = cam.project(q)[0]
                v = pred - cam.params[2:4]
                rr = np.linalg.norm(v / [cam.width / 2, cam.height / 2], axis=1)
                pred -= v * (0.01 * pp[5] * rr**2 * use)[:, None]
                err.append(float(np.sqrt(np.mean(((pred - pix) ** 2).sum(1)))))
            native[half].append(err)
    # Isolated hypotheses distinguish a coherent single camera defect from a
    # mixture of changes that only looks helpful when fitted jointly.
    import csv

    single = []
    labels = [
        "mount roll deg",
        "mount pitch deg",
        "mount yaw deg",
        "clock ms",
        "readout ms",
        "1pct radial warp",
    ]
    for i, cam in enumerate(cams):
        ix = np.flatnonzero(fs.cam == i)[::23]
        pix = fs.uv[ix]
        tf = fs.tc[fs.k[ix], i]
        tt = cam.row_time(tf, pix[:, 1])
        world = X[ids[ix]]

        def native_px(a, coef):
            p = np.zeros(6)
            p[a] = coef
            E = tr.T_world_ego(
                tt + (p[3] + p[4] * (pix[:, 1] / cam.height - 0.5)) * 0.001
            )
            P = (
                np.einsum("nji,nj->ni", E[:, :3, :3], world - E[:, :3, 3])
                - cam.T_ego_cam[:3, 3]
            )
            q = P @ _exp(np.deg2rad(p[:3])) @ cam.T_ego_cam[:3, :3]
            pred = cam.project(q)[0]
            v = pred - cam.params[2:4]
            pred -= (
                v
                * (
                    0.01
                    * p[5]
                    * np.linalg.norm(v / [cam.width / 2, cam.height / 2], axis=1) ** 2
                )[:, None]
            )
            return float(np.sqrt(np.mean(((pred - pix) ** 2).sum(1))))

        for a, nm in enumerate(labels):
            aa = A[:, :, 6 * i + a]
            seen = sel & ((c == i) | (cc == i))
            r0 = native_px(a, 0)
            for part, train in [("early", kt < 196), ("late", kt >= 196)]:
                ii = seen & train & (m[:, 10] == 0)
                ev = seen & ~train & (m[:, 10] == 1)
                p = 0
                for it0 in range(6):
                    res = e[ii] + aa[ii] * p
                    w = (0.035**2 + np.linalg.norm(res, axis=1) ** 2) ** -0.8
                    p = -np.einsum("ni,n,ni->", aa[ii], w, e[ii]) / (
                        np.einsum("ni,n,ni->", aa[ii], w, aa[ii])
                        + 1 / sig[i * 6 + a] ** 2
                    )
                before = np.median(np.linalg.norm(e[ev], axis=1)) * 100
                after = np.median(np.linalg.norm(e[ev] + aa[ev] * p, axis=1)) * 100
                single.append(
                    [
                        cam.name,
                        nm,
                        part,
                        int(ev.sum()),
                        p,
                        before,
                        after,
                        r0,
                        native_px(a, p),
                    ]
                )
    with open(PRODUCT / "debug/single-parameter-check.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "lens",
                "parameter",
                "trained_on",
                "validation_pairs",
                "proposed_change",
                "before_cm",
                "after_cm",
                "before_feature_RMS",
                "after_feature_RMS",
            ]
        )
        writer.writerows(single)
    json.dump(
        dict(fits=fold, transfer=out, distribution=edge_report, native=native),
        open(WORK / "cal-diagnostic.json", "w"),
        indent=2,
    )
    print("hypothesis transfer", out, flush=True)


if __name__ == "__main__":
    run()
