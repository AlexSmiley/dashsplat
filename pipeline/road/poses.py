"""Choose the weakly observable vertical world gauge; retain v1 feature factors.
All live filters read only the prefix. Plane/height conventions are implemented
by the common geometry module, not by assigning a different plane to each lens.
"""

from .fit import *
from pipeline.common.rig import _exp, _log, _T, _mv
from scipy.ndimage import gaussian_filter1d as smooth
import subprocess


def zyx(z, p, r):
    return Rot.from_euler("ZYX", np.stack((z, -p, r), -1)).as_matrix()


def angles(R):
    a = Rot.from_matrix(R).as_euler("ZYX")
    return a[:, 0], -a[:, 1], a[:, 2]


def arena(ctx, F, live):
    s, d, t, B, stop = ctx
    v1 = Trajectory(OUT / ("ego_causal_v1.npz" if live else "ego_v1.npz"))
    T = v1.poses
    az, pitch, roll = angles(T[:, :3, :3])
    p, r, h, dx, dy, daz = F.T
    ds = np.r_[0, np.cumsum(np.linalg.norm(np.diff(B[:, :2], axis=0), axis=1))]
    pr = pitch - p
    rr = roll - r
    z0 = T[:, 2, 3] - h
    if not live:
        grid = np.arange(int(ds[-1]) + 2, dtype=float)
        interp = lambda x: np.interp(grid, ds, x)
        go = (
            lambda x: np.interp(ds, grid, x)
        )  # thickness/curvature priors set the long-wavelength vertical gauge, not road-relative alignment
        roadz = 0.62 * smooth(interp(z0), 31)
        slope = np.gradient(roadz)
        rp0 = smooth(interp(pr), 35)
        ro0 = smooth(interp(rr), 37)
        ang = interp(np.unwrap(az))
        cg, sg = np.cos(ang), np.sin(ang)
        gp = 0.65 * slope + 0.35 * rp0
        gr = 0.34 * ro0  # length-weighted mean ENU gradient explicitly removed
        gx, gy = np.mean(cg * gp - sg * gr), np.mean(sg * gp + cg * gr)
        gp -= cg * gx + sg * gy
        gr -= (
            -sg * gx + cg * gy
        )  # bounded integrated grade. Solve z anchoring + smoothed gradient least squares.
        from scipy.sparse import diags

        I = eye(len(grid), format="csr")
        G = diags(
            [-np.ones(len(grid) - 1), np.ones(len(grid) - 1)],
            [0, 1],
            shape=(len(grid) - 1, len(grid)),
            format="csr",
        )
        Z = spsolve(I / 38**2 + G.T @ G, roadz / 38**2 + G.T @ gp[:-1])
        slope = np.gradient(
            Z
        )  # retain integrated grade and solve the two mean-normal equations with smooth crossfall only
        for _ in range(4):
            lateral = np.column_stack((-sg, cg))
            nxy = (
                np.column_stack((cg, sg)) * (-np.sin(slope) * np.cos(gr))[:, None]
                + lateral * (-np.sin(gr))[:, None]
            )
            gr += (
                lateral
                @ np.linalg.lstsq(lateral.T @ lateral, nxy.sum(0), rcond=None)[0]
            )
        rp = go(slope)
        ro = go(gr)
        H = go(Z - Z[0])
    else:
        rp = np.zeros(len(t))
        ro = np.zeros(len(t))
        H = np.zeros(len(t))
        zprior = 0.0
        slope = 0.0
        bank = 0.0
        gw = np.zeros(2)
        length = 0.0
        for i in range(1, len(t)):
            dd = ds[i] - ds[i - 1]
            sl = 1 - np.exp(-dd / 29)
            sg = 1 - np.exp(-dd / 46)
            zprior += sl * (0.58 * z0[i] - zprior)
            slope += sl * (0.32 * pr[i] - slope)
            bank += sg * (0.34 * rr[i] - bank)
            cg, ss = np.cos(az[i]), np.sin(az[i])
            length += dd
            moment = np.array([cg * slope - ss * bank, ss * slope + cg * bank])
            gw += (moment - gw) * (dd / (120 + length))
            rp[i] = slope - gw @ [cg, ss] + (zprior - H[i - 1]) / 55
            ro[i] = bank - gw @ [-ss, cg]
            H[i] = H[i - 1] + dd * rp[i]
    Rroad = zyx(az + daz, rp, ro)
    Rsus = zyx(p * 0, p, r)
    R = Rroad @ Rsus
    P = T[:, :3, 3].copy()
    P[:, :2] += F[:, 3:5]
    P[:, 2] = H + h / np.maximum(0.99, Rroad[:, 2, 2])
    P -= P[0].copy()
    return _T(R, P), rp, ro, h


def patch_factors(ctx, target, rp, ro, h, gauge, sk, ids):
    from pipeline.poses.motion import sample

    s, d, t, B, stop = ctx
    tr = Trajectory(OUT / "ego_v1.npz")
    m = selective(np.load(WORK / "pairs.npy"))
    record = np.empty(
        len(m),
        np.dtype(
            [("k", np.int32, 4), ("p", np.float64, (2, 16)), ("weight", np.float64)]
        ),
    )
    az = angles(target[:, :3, :3])[0]
    nx = -np.sin(rp) * np.cos(ro)
    ny = -np.sin(ro)
    nw = np.stack(
        (
            np.cos(az) * nx - np.sin(az) * ny,
            np.sin(az) * nx + np.cos(az) * ny,
            np.cos(rp) * np.cos(ro),
        ),
        1,
    )
    plane_d = h - np.einsum("ni,ni->n", nw, target[:, :3, 3] + gauge)
    for i, side in enumerate((0, 4)):
        a = obs(ctx, m, side, tr)
        tt = a["t"]
        j, u = brackets(s["t"][sk], tt)
        record["k"][:, 2 * i : 2 * i + 2] = np.stack((ids[j], ids[j + 1]), 1)
        base = sample(t, B, tt)
        normal = sample(t, nw, tt)
        record["p"][:, i] = np.column_stack(
            (
                u,
                base[:, :2],
                np.cos(base[:, 2]),
                np.sin(base[:, 2]),
                tt * 0,
                a["ray"],
                a["o"],
                normal,
                np.interp(tt, t, plane_d),
            )
        )
    record["weight"] = np.clip((m[:, 8] - 0.4) / 0.55, 0.5, 1.0) / 0.018
    with open(WORK / "patch-factors.bin", "wb") as out:
        np.array([len(m)], "i4").tofile(out)
        record.tofile(out)


def constrain(ctx, target, rp, ro, h):
    s, d, t, B, stop = ctx
    sk, ids, first = knots(s, t, stop)
    ix = s["ix"][sk[first]]
    T = target[ix]
    base = B[
        ix
    ]  # fit-parameter world origin: preserved old world point gauge and calibration anchors
    fit = np.load(PW / "offline-fit.npz")
    v1shift = fit["corr"][0, 3:] + np.r_[base[0, :2], 0]
    rot0 = _exp(np.stack((base[:, 2] * 0, base[:, 2] * 0, -base[:, 2]), -1))
    des = np.column_stack(
        (
            _log(rot0 @ T[:, :3, :3]),
            T[:, :3, 3] + v1shift - np.pad(base[:, :2], ((0, 0), (0, 1))),
        )
    )
    problem = bytearray((PW / "problem.bin").read_bytes())
    nf, np_, no, ne, ns, it, jobs = np.frombuffer(problem, dtype="i", count=7)
    assert nf == len(des) and np_ == len(fit["pts"])
    offset = 7 * 4 + 8 + 6 * 20 * 8
    # Warm-start each landmark in its own local rigid patch, not in the old
    # vertical frame. Every original observed pixel still participates in BA.
    v1 = Trajectory(OUT / "ego_v1.npz")
    middle = np.cumsum(fit["n"]) - fit["n"] // 2 - 1
    anchor = s["ix"][fit["k"][middle]]
    old = v1.poses[anchor]
    new = target[anchor]
    initX = (
        _mv(
            new[:, :3, :3] @ old[:, :3, :3].swapaxes(1, 2),
            fit["pts"] - v1shift - old[:, :3, 3],
        )
        + new[:, :3, 3]
        + v1shift
    )
    np.frombuffer(problem, dtype="d", count=nf * 6, offset=offset)[:] = des.ravel()
    np.frombuffer(problem, dtype="d", count=np_ * 3, offset=offset + nf * (6 + 5) * 8)[
        :
    ] = initX.ravel()
    np.frombuffer(problem, dtype="i", count=7)[5] = 40
    np.frombuffer(problem, dtype="d", count=1, offset=7 * 4)[:] = 4.5
    (WORK / "feature.bin").write_bytes(problem)
    sigma = np.tile([0.0009, 0.0009, 0.0018, 0.025, 0.025, 0.010], (nf, 1))
    sigma[0, 3:] = 1e-5
    patch_factors(ctx, target, rp, ro, h, v1shift, sk, ids)
    np.column_stack((des, sigma)).astype("d").tofile(WORK / "road-prior.bin")
    subprocess.run(
        [
            WORK / "build/trajectory_ba",
            WORK / "feature.bin",
            WORK / "feature-out.bin",
            WORK / "road-prior.bin",
            WORK / "patch-factors.bin",
        ],
        check=True,
    )
    sol = np.fromfile(WORK / "feature-out.bin")
    fr = sol[: nf * 6].reshape(nf, 6)
    X = sol[nf * 6 :].reshape(-1, 3)
    C = corr_interp(s["t"][sk], fr[ids], t)
    P = assemble(B, C)
    shift = P[0, :3, 3].copy()
    P[:, :3, 3] -= shift
    np.savez(WORK / "scene-v2.npz", X=X - shift, corr=fr)
    return P


def write(ctx, live, P, target, rp, ro, F):
    s, d, t, B, stop = ctx
    stem = "ego_causal" if live else "ego"
    v = dict(np.load(OUT / (stem + "_v1.npz")))
    yaw, p, r = angles(P[:, :3, :3])
    cy, sy = np.cos(yaw), np.sin(yaw)
    nx = -np.sin(rp) * np.cos(ro)
    ny = -np.sin(ro)
    nz = np.cos(rp) * np.cos(ro)
    nw = np.stack((cy * nx - sy * ny, sy * nx + cy * ny, nz), 1)
    nbody = _mv(P[:, :3, :3].swapaxes(1, 2), nw)
    sp = np.arcsin(np.clip(nbody[:, 0], -1, 1))
    sr = np.arctan2(nbody[:, 1], nbody[:, 2])
    h = F[:, 2] + (P[:, 2, 3] - target[:, 2, 3]) * nz
    v.update(
        T_world_ego=P,
        road_pitch=rp,
        road_roll=ro,
        susp_pitch=sp.copy(),
        susp_roll=sr.copy(),
        susp_heave=h.copy(),
    )
    for a, b in zip(
        np.flatnonzero(stop & ~np.r_[False, stop[:-1]]),
        np.flatnonzero(stop & ~np.r_[stop[1:], False]),
    ):
        for key in [
            "T_world_ego",
            "road_pitch",
            "road_roll",
            "susp_pitch",
            "susp_roll",
            "susp_heave",
        ]:
            v[key][a : b + 1] = v[key][a]
    m = np.load(WORK / "pairs.npy", mmap_mode="r")
    mt = np.sort(s["t"][m[m[:, 10] == 0, 0].astype(int)])
    nroad = np.searchsorted(mt, t, "right") - np.searchsorted(mt, t - 0.35, "left")
    nroad[stop] = 0
    v["n_road_pairs"] = nroad.astype("i4")
    weak = (nroad < 4) & ~stop
    v["quality_flag"] |= weak.astype("uint8") * 8
    v["sigma_rot"] = np.maximum(v["sigma_rot"], np.where(weak, 0.009, 0.002))
    np.savez(WORK / (stem + ".npz"), **v)


def run():
    ctx = inputs()
    spr = np.load(WORK / "spring.npz")
    for live, key in [(False, "offline"), (True, "live")]:
        F = spr[key]
        T, rp, ro, h = arena(ctx, F, live)
        P = T if live else constrain(ctx, T, rp, ro, h)
        write(ctx, live, P, T, rp, ro, F)


if __name__ == "__main__":
    run()
