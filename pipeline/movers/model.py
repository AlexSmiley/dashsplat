"""Colour a dense, closed class loft from the actual masked photographs, in object space."""

from pathlib import Path
from collections import defaultdict
import cv2, numpy as np
from scipy.spatial import cKDTree
from .shape import samples
from .inventory import W
from pipeline.common.rig import load_rig, Trajectory, T_world_cam

FIELDS = "x y z nx ny nz f_dc_0 f_dc_1 f_dc_2 opacity scale_0 scale_1 scale_2 rot_0 rot_1 rot_2 rot_3 t_birth t_death e0 s1 e1 s2 e2".split()
INF = 1e9


def ply(path, a):
    a = np.asarray(a, dtype="<f4").reshape(-1, len(FIELDS))
    assert len(a) and np.isfinite(a).all() and (a[:, 18] > a[:, 17]).all()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(
            (
                "ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % len(a)
                + "".join("property float %s\n" % v for v in FIELDS)
                + "end_header\n"
            ).encode()
        )
        a.tofile(f)


def interp(track, t):
    return np.stack(
        [
            np.interp(t, track["t"], np.unwrap(track[k]) if k == "yaw" else track[k])
            for k in ["x", "y", "z", "yaw"]
        ],
        axis=-1,
    )


def projection(cam, tr, tf, track, p, n, exposure=None):
    t = cam.row_time(tf, cam.height / 2)
    pos = interp(track, t)
    ang = pos[3]
    R = np.array(
        [[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]]
    )
    w = p @ R.T + pos[:3]
    C = exposure["C"] if exposure is not None else T_world_cam(cam, tf, trajectory=tr)
    delta = w - C[:3, 3]
    local = delta @ C[:3, :3]
    uv, ok = cam.project(local)
    rt = cam.row_time(tf, uv[:, 1])
    dt = rt - t
    # Take camera and object twist at this exposure, not at the next keyframe.
    if exposure is not None:
        V, omega = exposure["velocity"], exposure["omega"]
    else:
        V, omega = tr.velocity(t)
        E = tr.T_world_ego(t)
        V = V + np.cross(omega, C[:3, 3] - E[:3, 3])
    past = interp(track, t - 0.04)
    v = (pos[:3] - past[:3]) / 0.04
    turn = np.arctan2(np.sin(pos[3] - past[3]), np.cos(pos[3] - past[3])) / 0.04
    corr = (v + np.cross([0, 0, turn], w - pos[:3]) - V - np.cross(omega, delta)) * dt[
        :, None
    ]
    local = (delta + corr) @ C[:3, :3]
    uv, ok = cam.project(local)
    face = -(n @ R.T * delta).sum(1) / np.maximum(1e-6, np.linalg.norm(delta, axis=1))
    return uv * np.array([724 / cam.width, 469 / cam.height]), ok, face, local[:, 2]


def bake(data, mode, dest, upgrade=False, cabin=0.0, photographic=False):
    from .appearance import material, register, glass_mask

    s, rows, track, size, stats, scan = data
    root = W / "extra" if s.get("input") == "extra" else Path(".")
    cls = s["class"]
    cs = load_rig()
    tr = Trajectory("poses/ego" + ("_causal" if mode == "live" else "") + ".npz")
    pitch = (
        (0.092 if cls in ("bus", "truck") else 0.053)
        if cls not in ("person", "rider")
        else 0.032
    )
    if upgrade:
        pitch *= 0.76
    if photographic:
        pitch *= 0.86
    cp = pitch * 2.4
    fine = samples(cls, size, pitch, cabin)
    coarse = samples(cls, size, cp, cabin)
    nf = len(fine[0])
    nc = len(coarse[0])
    P, N, S, Q, M, patch = [np.concatenate([a, b]) for a, b in zip(coarse, fine)]
    parent = np.zeros(nf, int)
    for j in np.unique(coarse[-1]):
        ii = np.where(coarse[-1] == j)[0]
        kk = np.where(fine[-1] == j)[0]
        parent[kk] = ii[cKDTree(coarse[0][ii]).query(fine[0][kk])[1]]
    mirror = cKDTree(coarse[0]).query(coarse[0] * [1, -1, 1])[1]
    measured = []
    pred = []
    latest = np.full(nc + nf, -1)
    soft = np.full(nc, -1)
    best = np.zeros(nc + nf)
    nrev = np.zeros(nc + nf, int)
    sym = np.zeros(nc)
    default = np.array([136, 145, 147])
    roof = default
    samples_used = 0
    material_prior = None
    if upgrade:
        glass = glass_mask(P, N, M, cls, size, cabin)
        previous_color = np.full((nc + nf, 3), -1.0, dtype=float)
        previous_time = np.full(nc + nf, -1e5)
    last = defaultdict(lambda: -1000)
    first_real = None

    def record(ids, rgb, t, is_pred=False):
        ids = np.asarray(ids)
        nr = len(ids)
        tt = np.full((nr, 7), INF)
        tt[:, 0] = t
        tt[:, 2] = t + (1 / 36 if mode == "live" and not is_pred else 0)
        return np.c_[
            P[ids],
            N[ids],
            (rgb / 255.0 - 0.5) / 0.28209479,
            np.full(nr, 1.8 if is_pred else 2.9),
            np.log(S[ids]),
            Q[ids],
            tt,
        ].tolist()

    def retire(a, ix, t):
        for i in ix:
            if i >= 0:
                a[i][18] = min(a[i][18], t + 0.25)

    def see(a, i, t):
        row = a[i]
        if t < row[18]:
            for k in (5, 3, 1):
                # visibility runs are [birth,e0], [s1,e1], [s2,e2].
                end = 17 + k + 1
                start = 17 + (k if k > 1 else 0)
                if row[start] < INF and t - row[end] < (
                    1e-8 if mode == "live" else 0.43
                ):
                    row[end] = max(row[end], t + (1 / 36 if mode == "live" else 0))
                    return True
            else:
                if row[20] >= INF:
                    row[20] = t
                    row[21] = t + (1 / 36 if mode == "live" else 0)
                elif row[22] >= INF:
                    row[22] = t
                    row[23] = t + (1 / 36 if mode == "live" else 0)
                else:
                    return False
        return True

    for r in rows:
        if r.get("held") or not r["key"] or r["t"] - last[r["c"]] < 0.095:
            continue
        c = r["c"]
        cam = cs[c]
        t = r["release"]
        frame_track = scan
        if mode == "live":
            j = np.searchsorted(scan["t"], t)
            if j == len(scan["t"]) or scan["t"][j] - t > 0.5:
                continue
            t = scan["t"][j]
            dt = np.array([-0.9, 0.0])
            vx, vy = scan["vx"][j], scan["vy"][j]
            frame_track = dict(
                t=t + dt,
                x=scan["x"][j] + vx * dt,
                y=scan["y"][j] + vy * dt,
                z=np.full(2, scan["z"][j]),
                yaw=np.full(2, np.arctan2(vy, vx)),
            )
        elif t < scan["t"][0]:
            continue
        last[c] = r["t"]
        uv, ok, face, z = projection(
            cam, tr, r["t"], frame_track, P, N, r if "C" in r else None
        )
        root = Path(r.get("root", root))
        mask = (
            cv2.imread(str(root / "masks/dyn" / c / f"{r['n']:06}.png"), -1) == r["id"]
        ).astype("uint8")
        mask = cv2.erode(mask, np.ones((3, 3), np.uint8))
        im = cv2.imread(f"data/drive/frames/{c}/{r['n']:06}.jpg")
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        b = np.array(r["bb"])
        ext = np.r_[uv.min(0), uv.max(0)]
        ext = np.clip(ext, [0, 0, 0, 0], [724, 469, 724, 469])
        wh = np.maximum(1, ext[2:] - ext[:2])
        bw = np.maximum(1, b[2:] - b[:2])
        if np.any(
            abs((b[2:] + b[:2] - ext[2:] - ext[:2]) * 0.5) > np.maximum(wh * 0.46, 12)
        ) or np.any(wh < 5):
            continue
        # Limited image registration to align the photographed trim and wheels to the loft;
        # never expand a pole-split mask to the whole vehicle.
        xy = uv.copy()
        tight = (bw / wh > 0.79) & (bw / wh < 1.25) & (b[:2] > 2) & (b[2:] < [721, 466])
        xy = (
            (uv - (ext[:2] + ext[2:]) * 0.5) * np.where(tight, bw / wh, 1)
            + (ext[:2] + ext[2:]) * 0.5
            + np.where(tight, (b[:2] + b[2:] - ext[:2] - ext[2:]) * 0.5, 0)
        )
        if upgrade:
            xy = register(xy, ok & (face > 0.14), mask)
        u, v = (
            np.clip(xy[:, 0].astype(int), 0, 723),
            np.clip(xy[:, 1].astype(int), 0, 468),
        )
        zi = np.full((469, 724), 1e5, np.float32)
        valid = ok & (face > 0.15)
        np.minimum.at(zi, (v[valid], u[valid]), z[valid].astype(np.float32))
        mn = cv2.erode(zi, np.ones((3, 3), np.uint8))
        visible = (
            valid & (mask[v, u] > 0) & (z < mn[v, u] + np.maximum(0.13, z * 0.011))
        )
        if r.get("depth_root") and mode == "stitched":
            dep = np.load(Path(r["depth_root"]) / "depth" / c / f"{r['n']:06}.npy")[
                v, u
            ].astype(float)
            supported = visible & (dep > 0)
            tol = 0.16 if c == "back" else 0.10
            if supported.sum() > 16 and abs(np.median((dep - z)[supported])) < max(
                0.8, np.median(z[supported]) * tol
            ):
                visible &= (dep <= 0) | (np.abs(dep - z) < 0.6 + tol * z)
        if not visible.any():
            continue
        col = im[
            np.clip((xy[:, 1] * cam.height / 469).astype(int), 0, cam.height - 1),
            np.clip((xy[:, 0] * cam.width / 724).astype(int), 0, cam.width - 1),
        ]
        if photographic:
            from .photographic import native_colour

            col = native_colour(im, xy, c, r["n"], t, mode == "live")
        qual = np.maximum(0.02, face) ** 0.7 * cam.params[0] / np.maximum(2, z)
        if upgrade:
            # Winner-takes-colour for each surface sample. Prefer native pixel
            # density, frontal incidence and a locally sharp, registered exposure.
            grey = cv2.resize(
                cv2.cvtColor(im, cv2.COLOR_RGB2GRAY),
                (724, 469),
                interpolation=cv2.INTER_AREA,
            )
            lap = cv2.blur(abs(cv2.Laplacian(grey, cv2.CV_32F)), (9, 9))
            qual = np.maximum(0.015, face) ** 1.15 * cam.params[0] / np.maximum(2, z)
            qual *= np.clip(0.68 + lap[v, u] / 32, 0.72, 1.23)
        if first_real is None:
            first_real = t
            body = (M == "body") & visible
            default = np.median(col[body if body.any() else visible], axis=0)
            top = body & (P[:, 2] > 0.77 * size[2])
            roof = np.median(col[top], axis=0) if top.any() else default
            base = np.tile(default, (nc, 1))
            base[coarse[0][:, 2] > 0.84 * size[2]] = roof
            base[coarse[4] == "window"] = default * 0.45
            base[coarse[4] == "tyre"] = [35, 37, 37]
            base[coarse[4] == "rim"] = [115, 119, 120]
            base[coarse[4] == "hub"] = [85, 89, 89]
            if upgrade:
                material_prior = material(P, N, M, cls, size, default, roof, cabin)
                base = material_prior[:nc]
            soft = np.arange(nc)
            pred.extend(record(np.arange(nc), base, t, True))
        use = visible.copy()
        if upgrade:
            # Through-glass background and glare vary with viewpoint. Do not
            # stamp that colour as paint unless separate exposures corroborate it.
            repeat = (
                (t - previous_time > 0.06)
                & (t - previous_time < 0.85)
                & (np.max(abs(col - previous_color), axis=1) < 28)
            )
            use &= ~glass | (repeat & (face > 0.30))
            if photographic:
                # Keep stable photographed sky streaks on dark glass, with a
                # compressed diffuse base. Lamp and chrome samples stay intact.
                gl = glass & use
                lum = col[gl].max(1) / 255
                col[gl] *= (0.62 + 0.38 * lum**2)[:, None]
            previous_color[visible] = col[visible]
            previous_time[visible] = t
        use[nc:] &= qual[nc:] * pitch > 1.05
        replace = use & (
            (latest < 0)
            | ((qual > best * (1.005 if upgrade else 1.45)) & ((nrev < 3) | upgrade))
        )
        idx = np.flatnonzero(replace)
        retire(measured, latest[idx], t)
        latest[idx] = np.arange(len(measured), len(measured) + len(idx))
        # Ambiguous glass is inferred, so do not relabel it as a measured pixel.
        measured.extend(record(idx, col[idx], t))
        best[idx] = qual[idx]
        nrev[idx] += 1
        for cell in np.flatnonzero(use & ~replace):
            i = latest[cell]
            if not see(measured, i, t) and measured[i][18] > t:
                # Start a fresh record at the fourth disjoint sighting.
                q = measured[i].copy()
                measured[i][18] = t
                q[17:] = [
                    t,
                    INF,
                    t + (1 / 36 if mode == "live" else 0),
                    INF,
                    INF,
                    INF,
                    INF,
                ]
                latest[cell] = len(measured)
                measured.append(q)
        # Retire whole coarse discs only after fine children cover the cell; keep uncovered rims.
        count = np.bincount(parent, minlength=nc)
        fine_seen = np.bincount(parent[latest[nc:] >= 0], minlength=nc)
        covered = (fine_seen >= count * 0.88) & (count > 0)
        retire(measured, latest[:nc][covered], t)
        observed = (latest[:nc] >= 0) | covered
        retire(pred, soft[observed], t)
        source = mirror
        inferred = (
            (~observed)
            & use[source]
            & (qual[source] > 1.55 * sym)
            & (np.abs(P[:nc, 1]) > size[1] * 0.26)
        )
        ids = np.where(inferred)[0]
        retire(pred, soft[ids], t)
        soft[ids] = np.arange(len(pred), len(pred) + len(ids))
        mirrored = (
            0.82 * col[source[ids]] + 0.18 * default
            if not upgrade
            else 0.95 * col[source[ids]] + 0.05 * material_prior[ids]
        )
        pred.extend(record(ids, mirrored, t, True))
        sym[ids] = qual[source[ids]]
        samples_used += 1
    if not measured:
        raise ValueError(f"no photographs for {s['name']} {mode}")
    a = np.array(measured)
    b = np.array(pred)
    ply(dest / f"{s['id']}.ply", a)
    ply(dest / f"{s['id']}.pred.ply", b)
    return dict(
        photographs=samples_used,
        fine_cells=nf,
        real_fine=int(sum(latest[nc:] >= 0)),
        observed_records=len(a),
        predicted_records=len(b),
        first_photo=first_real,
        coarse_cells=nc,
    )
