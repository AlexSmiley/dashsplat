"""Fit suspension and planar corrections from matched road radiance, in metric coordinates."""

from .pairs import *
from scipy.sparse import coo_matrix, csr_matrix, eye, vstack
from scipy.sparse.linalg import spsolve
from scipy.spatial.transform import Rotation as Rot
from pipeline.poses.motion import brackets, assemble, corr_interp
from pipeline.common.rig import _exp, _log, _mv, _T


def accel(t, d, stop):
    a = np.column_stack((-d["ay"], -d["ax"]))
    f = a.copy()
    for i in range(1, len(t)):
        f[i] = (
            f[i - 1]
            if stop[i]
            else f[i - 1] + (a[i] - f[i - 1]) * (1 - np.exp(-(t[i] - t[i - 1]) / 0.20))
        )
    return f


def obs(ctx, m, side, tr):
    s, d, t, B, stop = ctx
    z = m[:, side : side + 4]
    k, c = z[:, :2].astype(int).T
    uv = z[:, 2:4]
    C = [load_rig(RS / "calib/rig.json")[n] for n in NAMES]
    term = np.zeros(len(k))
    ray = np.empty((len(k), 3))
    o = np.stack([cam.T_ego_cam[:3, 3] for cam in C])[c]
    for i, cam in enumerate(C):
        ii = c == i
        term[ii] = cam.row_time(s["tc"][k[ii], i], uv[ii, 1])
        ray[ii] = cam.unproject(uv[ii]) @ cam.T_ego_cam[:3, :3].T
    E = tr.T_world_ego(term)
    az = np.arctan2(E[:, 1, 0], E[:, 0, 0])
    return dict(k=k, c=c, t=term, az=az, xy=E[:, :2, 3], ray=ray, o=o)


def intersect(A, v, J=False):
    # Body → yaw-aligned road tangent rotation Ry(-pitch) Rx(roll), then positive heave.
    p, r, h, x, y, w = v.T
    cp, sp, cr, sr = np.cos(p), np.sin(p), np.cos(r), np.sin(r)
    co, si = np.cos(w + A["az"]), np.sin(w + A["az"])
    rotate = lambda a: np.stack(
        (
            cp * a[:, 0] - sp * (sr * a[:, 1] + cr * a[:, 2]),
            cr * a[:, 1] - sr * a[:, 2],
            sp * a[:, 0] + cp * (sr * a[:, 1] + cr * a[:, 2]),
        ),
        1,
    )
    u = rotate(A["ray"])
    o = rotate(A["o"])
    lam = -(o[:, 2] + h) / np.minimum(-1e-7, u[:, 2])
    G = o + lam[:, None] * u
    px, py = G[:, :2].T
    ground = np.stack((co * px - si * py + x, si * px + co * py + y), 1) + A["xy"]
    if not J:
        return ground
    # dG/dp,dG/dr for a rotated ray hitting the same road plane.
    pitch = np.cross([0, -1.0, 0], G)
    roll = np.cross(np.stack((cp, p * 0, sp), 1), G)
    q = np.stack((pitch, roll, np.zeros_like(G)), 2)
    q[:, :2, :] -= (
        u[:, :2, None]
        / u[:, 2, None, None]
        * (q[:, 2:3, :] + np.array([0, 0, 1])[None, None, :])
    )
    local = q[:, :2]
    j = np.empty((len(v), 2, 6))
    j[:, 0, :3] = co[:, None] * local[:, 0] - si[:, None] * local[:, 1]
    j[:, 1, :3] = si[:, None] * local[:, 0] + co[:, None] * local[:, 1]
    j[:, :, 3:5] = np.eye(2)
    j[:, 0, 5] = -si * px - co * py
    j[:, 1, 5] = co * px - si * py
    return ground, j


def batch(ctx, m):
    s, d, t, B, stop = ctx
    sk, ids, first = knots(s, t, stop)
    tk = s["t"][sk]
    nk = int(ids.max() + 1)
    nt = nk * 6 + 4
    tr = Trajectory(OUT / "ego_v1.npz")
    a, b = obs(ctx, m, 0, tr), obs(ctx, m, 4, tr)
    ac = accel(t, d, stop)[s["ix"][sk[first]]]
    anchors = []
    for A in [a, b]:
        j, u = brackets(tk, A["t"])
        anchors.append((ids[j], ids[j + 1], u))
    # Fixed sparse penalties in physical units. Four global coefficients: two
    # level intercepts and two gains (rad per m/s²). All are learned, not imposed.
    row = []
    col = []
    val = []
    npr = 0

    def penalty(cols, vals):
        nonlocal npr
        row.extend([npr] * len(cols))
        col.extend(cols)
        val.extend(vals)
        npr += 1

    for k in range(nk):
        for q, sigma in enumerate([0.006, 0.007, 0.035, 0.16, 0.16, 0.006]):
            cs = [k * 6 + q]
            vs = [1 / sigma]
            if q < 2:
                cs += [nk * 6 + q, nk * 6 + 2 + q]
                vs += [-1 / sigma, -ac[k, q] / sigma]
            penalty(cs, vs)
    for q in range(4):
        penalty([nk * 6 + q], [1 / ([0.012, 0.012, 0.008, 0.008][q])])
    for j in range(len(tk) - 1):
        k, l = ids[j : j + 2]
        dt = tk[j + 1] - tk[j]
        if k == l:
            continue
        dd = s["distance"][sk[j + 1]] - s["distance"][sk[j]]
        for q, sigma in enumerate(
            [
                0.0017 + 0.003 * min(dt, 0.3),
                0.002 + 0.004 * min(dt, 0.3),
                0.008 + 0.018 * min(dt, 0.3),
                0.009 + 0.009 * dd,
                0.009 + 0.009 * dd,
                0.0009 + 0.0004 * dd,
            ]
        ):
            penalty([k * 6 + q, l * 6 + q], [-1 / sigma, 1 / sigma])
    P = coo_matrix((val, (row, col)), shape=(npr, nt)).tocsr()
    PtP = P.T @ P
    X = np.zeros(nt)
    scale = np.tile(np.arange(6), (len(m), 2, 1))
    ii = np.broadcast_to(np.arange(len(m) * 2).reshape(-1, 2, 1), scale.shape)
    rows = np.tile(ii.ravel(), 4)
    cols = np.concatenate(
        [
            (p[:, None, None] * 6 + scale).ravel()
            for (k, l, u) in anchors
            for p in (k, l)
        ]
    )
    cols = cols.astype(np.int32)
    rows = rows.astype(np.int32)
    noise = 0.023
    w0 = np.clip((m[:, 8] - 0.65) / 0.30, 0.25, 1)
    w0 *= np.where(m[:, 0] == m[:, 4], 0.65, 1)
    for it in range(6):
        F = X[: nk * 6].reshape(-1, 6)
        vv = [F[k] * (1 - u[:, None]) + F[l] * u[:, None] for k, l, u in anchors]
        g, jac = intersect(a, vv[0], True)
        g1, j1 = intersect(b, vv[1], True)
        err = g - g1
        mag = np.linalg.norm(err, axis=1)
        w = (
            w0
            / np.sqrt(1 + (mag / ([0.15, 0.085, 0.055, 0.04, 0.035, 0.035][it])) ** 2)
            / noise
        )
        der = []
        for (_, _, u), jj in zip(anchors, [jac, -j1]):
            der += [
                (jj * ((1 - u) * w)[:, None, None]).ravel(),
                (jj * (u * w)[:, None, None]).ravel(),
            ]
        A = coo_matrix(
            (np.concatenate(der), (rows, cols)), shape=(len(m) * 2, nt)
        ).tocsr()
        H = A.T @ A + PtP
        dx = spsolve(H, -A.T @ (err * w[:, None]).ravel() - PtP @ X)
        X += dx
        print(
            "fit",
            it,
            np.percentile(mag * 100, [50, 90]),
            "gains/intercepts deg",
            np.rad2deg(X[-4:]),
            flush=True,
        )
    return tk, ids, X[:-4].reshape(-1, 6), X[-4:]


def selective(M, n=140000):
    # Deterministic observation subsample after spatial holdout.
    rng = np.random.default_rng(992)
    ix = np.flatnonzero(M[:, 10] == 0)
    rng.shuffle(ix)
    sel = ix[:n]
    return M[np.sort(sel)]


def causal(ctx, M):
    s, d, t, B, stop = ctx
    tr = Trajectory(OUT / "ego_causal_v1.npz")
    a, b = obs(ctx, M, 0, tr), obs(ctx, M, 4, tr)
    sk, ids, first = knots(s, t, stop)
    kt = s["t"]
    acc = accel(t, d, stop)
    keys = np.union1d(sk, np.unique(M[:, [0, 4]]).astype(int))
    X = np.zeros((len(keys), 6))
    coeff = np.zeros((2, 2))
    prec = np.tile(np.diag([8.0, 20.0])[None], (2, 1, 1))
    rhs = np.zeros((2, 2))
    order = np.argsort(a["k"])
    start = np.searchsorted(a["k"][order], keys)
    end = np.searchsorted(a["k"][order], keys, side="right")
    ks = np.zeros(len(kt), int)
    gainhist = []
    acs = acc[s["ix"]]
    last = 0
    for q, k in enumerate(keys):
        kc = s["ix"][k]
        dt = kt[k] - kt[last]
        dist = s["distance"][k] - s["distance"][last]
        ks[k] = q
        pred = X[max(0, q - 1)].copy()
        features = np.column_stack((np.ones(2), acs[k]))
        phys = (features * coeff).sum(1)
        if q and not (stop[kc] and stop[s["ix"][last]]):
            pred[:2] += (phys - pred[:2]) * (1 - np.exp(-dt / 0.32))
            pred[2] *= np.exp(-dt / 0.48)
            pred[3:5] *= np.exp(-dt / 2)
            pred[5] *= np.exp(-dt / 2)
            idx = order[
                start[q] : end[q]
            ]  # source packet correction can interpolate only from already solved keys
            if len(idx):
                A = {key: v[idx] for key, v in a.items()}
                BB = {key: v[idx] for key, v in b.items()}
                past = b["k"][idx] != k
                old = X[ks[b["k"][idx]]]
                v = pred.copy()
                sig = np.array(
                    [
                        0.003,
                        0.0035,
                        0.016,
                        0.012 + 0.013 * dist,
                        0.012 + 0.013 * dist,
                        0.0013 + 0.0006 * dist,
                    ]
                )
                reg = np.diag(1 / sig**2)
                for it in range(4):
                    cur = np.broadcast_to(v, (len(idx), 6))
                    g, j = intersect(A, cur, True)
                    g0, j0 = intersect(BB, np.where(past[:, None], old, cur), True)
                    e = g - g0
                    J = j - np.where(past[:, None, None], 0, j0)
                    mag = np.linalg.norm(e, axis=1)
                    w = (1 / (1 + (mag / 0.045) ** 2)) / 0.03**2
                    H = np.einsum("nij,n,nik->jk", J, w, J, optimize=True) + reg
                    grad = np.einsum("nij,n,ni->j", J, w, e, optimize=True) + reg @ (
                        v - pred
                    )
                    v -= np.linalg.solve(H, grad)
                pred = v
                if len(idx) >= 8:
                    wt = min(len(idx) / 35, 1.0)
                    for axis in range(2):
                        vec = features[axis]
                        prec[axis] += wt * np.outer(vec, vec)
                        rhs[axis] += wt * vec * pred[axis]
                        coeff[axis] = np.linalg.solve(prec[axis], rhs[axis])
        X[q] = pred
        last = k
        gainhist.append(coeff.copy())
    # Bound dependency of suspension interpolation together with the V1 source exposure window.
    cams = [load_rig(RS / "calib/rig.json")[n] for n in NAMES]
    exposure = np.max(
        np.stack(
            [
                s["tc"][:, i] + cam.time_offset_s + 0.5 * abs(cam.rs_readout_s or 0)
                for i, cam in enumerate(cams)
            ]
        ),
        axis=0,
    )
    tk = kt[keys]
    j, u = brackets(tk, t)

    # Upper bound for an exported V1 transform queried at a rolling-shutter time:
    # next master sample, its next V1 visual knot, latest physical exposure there.
    def available(time):
        mi = np.minimum(np.searchsorted(t, time), len(t) - 1)
        q = np.minimum(np.searchsorted(kt, t[mi]), len(kt) - 1)
        begin = np.maximum.accumulate(
            np.where(~stop | ~np.r_[False, stop[:-1]], np.arange(len(t)), 0)
        )
        q[stop[mi]] = np.minimum(
            np.searchsorted(kt, t[begin[mi[stop[mi]]]]), len(kt) - 1
        )
        return np.maximum(t[mi], np.maximum(kt[q], exposure[q]))

    dep = np.maximum.accumulate(np.maximum(exposure[keys], available(exposure[keys])))
    out = corr_interp(tk, X, t)
    wanted = np.where(u > 0, j + 1, j)
    late = dep[wanted] > t + 0.49
    last_available = np.clip(
        np.searchsorted(dep, t + 0.49, "right") - 1, 0, len(keys) - 1
    )
    out[late] = X[last_available[late]]
    latest = np.maximum(available(t), np.where(late, dep[last_available], dep[wanted]))
    np.save(WORK / "causal-lag.npy", np.maximum(0, latest - t))
    return out, np.array(gainhist)


def run():
    ctx = inputs()
    s, d, t, B, stop = ctx
    m0 = np.load(WORK / "pairs.npy")
    m = selective(m0)
    tk, ids, X, g = batch(ctx, m)
    offline = corr_interp(tk, X[ids], t)
    m = m0[m0[:, 10] == 0]
    live, hist = causal(ctx, m)
    np.savez(
        WORK / "spring.npz",
        offline=offline,
        live=live,
        gains=g,
        live_gains=hist,
        tk=tk,
        X=X,
        ids=ids,
    )
    print("springs saved", flush=True)


if __name__ == "__main__":
    run()
