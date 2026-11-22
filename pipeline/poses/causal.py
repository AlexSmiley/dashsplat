"""Forward-only multiview odometry with past-ray landmark triangulation."""

from .motion import *
from pipeline.common.rig import _exp, _mv, _T, _log, _skew


def run():
    s, d, t, B, stationary = inputs()
    ix = s["ix"]
    kt = s["t"]
    cams = load_rig(RS / "calib/rig.json")
    C = [cams[n] for n in NAMES]
    data = []
    rays = []
    offsets = []
    elig = []
    cum = 0
    for c in C:
        a = np.load(W / f"lk-{c.name}.npz")
        ids = a["id"] + cum
        number = int(a["total"])
        elig.append(
            (np.arange(number, dtype="uint64") * 9176 + len(elig) * 37 + 17) % 13 != 0
        )
        cum += number
        uv = a["uv"]
        data.append((ids, a["ptr"], uv))
        rays.append(c.unproject(uv) @ c.T_ego_cam[:3, :3].T)
        offsets.append(c.row_time(0, uv[:, 1]))
    eligible = np.concatenate(elig)
    M = np.zeros((cum, 3, 3))
    b = np.zeros((cum, 3))
    count = np.zeros(cum, int)
    ref = np.zeros((cum, 3))
    last = np.zeros(cum, int)
    seed = np.zeros((cum, 3))
    PE = []
    noise = []
    tracks = []
    R = Rs(B[0, 2])
    pos = np.zeros(3)
    prev_v = np.zeros(3)
    prev_w = np.zeros(3)
    gps_offset = np.zeros(2)
    gps = np.stack((d["gps_x"], d["gps_y"]), -1)
    foc = np.array([c.params[:2].mean() for c in C])
    lever = np.array([c.T_ego_cam[:3, 3] for c in C])
    for k, i in enumerate(ix):
        stat = stationary[i]
        j = ix[max(0, k - 1)]
        dt = t[i] - t[j]
        dy = B[i, 2] - B[j, 2]
        dist = np.linalg.norm(B[i, :2] - B[j, :2])
        normal = not (stat and (k == 0 or stationary[j]))
        if k and normal:
            rp = Rs(dy) @ R
            pp = pos.copy()
            pp[:2] += B[i, :2] - B[j, :2]
            pp[2] += np.clip(prev_v[2], -0.7, 0.7) * min(dt, 0.15) * 0.6
            # Very weak causal ENU, level and heading priors; GNSS is fused and noisy.
            gps_offset += (gps[i] - B[i, :2] - gps_offset) * (1 - np.exp(-dt / 7))
        else:
            rp = R.copy()
            pp = pos.copy()
        ids = []
        cc = []
        rr = []
        tau = []
        for ci, c in enumerate(C):
            tid, ptr, uv = data[ci]
            sl = slice(ptr[k], ptr[k + 1])
            ids.append(tid[sl])
            cc.append(np.full(ptr[k + 1] - ptr[k], ci))
            rr.append(rays[ci][sl])
            tau.append(s["tc"][k, ci] - kt[k] + offsets[ci][sl])
        ids = np.concatenate(ids)
        cc = np.concatenate(cc)
        rr = np.concatenate(rr)
        tau = np.concatenate(tau)
        vel = (
            rp.T
            @ np.r_[
                d["speed"][i] * np.cos(B[i, 2]),
                d["speed"][i] * np.sin(B[i, 2]),
                prev_v[2],
            ]
        )
        omega = np.r_[prev_w[:2], dy / dt if dt > 0 and normal else 0]
        dr = _exp(tau[:, None] * omega)
        orig = _mv(dr, lever[cc]) + tau[:, None] * vel
        ray = _mv(dr, rr)
        ok = (count[ids] >= 3) & eligible[ids]
        sid = ids[ok]
        idx = np.flatnonzero(ok)
        cnt = 0
        resid = 4.0
        if len(idx) > 10 and normal and k and dt < 0.65:
            a = M[sid]
            vals = np.linalg.eigvalsh(a)
            ok = vals[:, 0] / count[sid] > 4e-5
            idx = idx[ok]
            aid = ids[idx]
            pts = (
                ref[aid] + np.linalg.solve(M[aid], b[aid, :, None])[..., 0]
                if len(aid)
                else np.zeros((0, 3))
            )
            orig2 = orig[idx]
            ray2 = ray[idx]
            f = foc[cc[idx]]
            R0 = rp.copy()
            p0 = pp.copy()
            delta = np.zeros(6)
            sd = np.array(
                [
                    0.0014,
                    0.0014,
                    0.00115,
                    0.009 + 0.013 * dist,
                    0.008 + 0.011 * dist,
                    0.010 + 0.012 * dist,
                ]
            )
            prec = 1 / sd**2
            for it in range(3):
                E = (pts - pp) @ rp
                L = E - orig2
                depth = (L * ray2).sum(1)
                rng = np.linalg.norm(L, axis=1)
                u = L / np.maximum(rng[:, None], 1e-6)
                e = (u - ray2) * f[:, None]
                err = np.linalg.norm(e, axis=1)
                weight = 1 / (1 + (err / (3.4 + 1.2 * it)) ** 2) ** 1.6
                weight *= 0.10 * ((depth > 1) & (depth < 240) & (err < 35))
                P = np.eye(3) - u[:, :, None] * u[:, None, :]
                J = (
                    P
                    @ np.concatenate(
                        (_skew(E), np.broadcast_to(-np.eye(3), (len(E), 3, 3))), axis=2
                    )
                ) * (f / np.maximum(rng, 1e-6))[:, None, None]
                H = np.einsum("nij,n,nim->jm", J, weight, J, optimize=True) + np.diag(
                    prec
                )
                g = np.einsum("nij,n,ni->j", J, weight, e, optimize=True) + delta * prec
                step = -np.linalg.solve(H, g)
                step = np.clip(step, -3 * sd, 3 * sd)
                rp = rp @ _exp(step[:3])
                pp += rp @ step[3:]
                delta += step
                cnt = int(((err < 5) & (weight > 0)).sum())
                resid = (
                    np.median(err[weight > 0.025]) if np.any(weight > 0.025) else 8.0
                )
        if k and normal:
            # Complementary posterior anchor. The same monocular depth may appear in
            # scores of frames and cannot be counted as independent metric evidence.
            # Apply the absolute reference *after* resection so old landmarks cannot
            # veto the ENU anchor. Fuse only low-frequency GPS minus wheel position.
            pp[:2] += (B[i, :2] + gps_offset - pp[:2]) * (1 - np.exp(-dt / 1.5))
            pp[2] *= np.exp(-dt / 0.8)
            corr = _log(Rs(-B[i, 2]) @ rp)
            rp = Rs(B[i, 2]) @ _exp(corr * np.exp(-dt / np.array([2.3, 2.3, 1.5])))
        if k and normal and dt > 0.001:
            prev_v = 0.55 * prev_v + 0.45 * (pp - pos) / dt
            prev_w = 0.6 * prev_w + 0.4 * _log(R.T @ rp) / dt
            prev_w = np.clip(prev_w, -1, 1)
        else:
            prev_v[:] = 0
            prev_w[:] = 0
        R = rp
        pos = pp
        PE.append(_T(R, pos))
        tracks.append(cnt)
        noise.append(
            [
                max(0.035, 0.8 / np.sqrt(max(1, cnt))) * (1 + min(resid, 8) / 4),
                np.deg2rad(
                    max(0.06, 1.8 / np.sqrt(max(1, cnt))) * (1 + min(resid, 8) / 4)
                ),
            ]
        )
        O = orig @ R.T + pos
        r = ray @ R.T
        zero = count[ids] == 0
        ref[ids[zero]] = O[zero]
        A = np.eye(3) - r[:, :, None] * r[:, None, :]
        M[ids] += A
        b[ids] += _mv(A, O - ref[ids])
        count[ids] += 1
        if k % 1000 == 0:
            print("causal", k, "/", len(ix), cnt, flush=True)
    P = np.array(PE)
    noise = np.array(noise)
    c = np.column_stack(
        (
            _log(Rs(-B[ix, 2]) @ P[:, :3, :3]),
            P[:, :3, 3] - np.pad(B[ix, :2], ((0, 0), (0, 1))),
        )
    )
    ci = corr_interp(kt, c, t)
    m = assemble(B, ci)
    jj, alpha = brackets(kt, t)
    nt = np.where(
        alpha > 0,
        np.minimum(np.array(tracks)[jj], np.array(tracks)[jj + 1]),
        np.array(tracks)[jj],
    )
    nt[stationary] = 0
    sp = np.interp(t, kt, noise[:, 0])
    sr = np.interp(t, kt, noise[:, 1])
    for a, b0 in zip(
        np.flatnonzero(stationary & ~np.r_[False, stationary[:-1]]),
        np.flatnonzero(stationary & ~np.r_[stationary[1:], False]),
    ):
        sp[a : b0 + 1] = sp[a]
        sr[a : b0 + 1] = sr[a]
    export("ego_causal", t, m, sp, sr, nt, stationary)
    np.savez(W / "causal-knots.npz", c=c, n=tracks, noise=noise)
    print("causal saved", flush=True)


if __name__ == "__main__":
    run()
