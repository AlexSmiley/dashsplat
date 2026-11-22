"""Measure untouched patch shifts, native feature residuals and attitude diagnostics."""

from .mosaic import trajectory
from .fit import *
from .geometry import pixels_to_ground
from .poses import angles
from pipeline.poses.validate import Fit, exact
import csv


def raymap(m, ctx, tr, side):
    s, *_ = ctx
    g = np.empty((len(m), 3))
    ran = np.empty(len(m))
    tm = np.empty(len(m))
    cams = load_rig(RS / "calib/rig.json")
    valid = np.zeros(len(m), bool)
    for ci, n in enumerate(NAMES):
        ii = np.flatnonzero(m[:, side + 1] == ci)
        cam = cams[n]
        k = m[ii, side].astype(int)
        v = m[ii, side + 2 : side + 4]
        tf = s["tc"][k, ci]
        tm[ii] = cam.row_time(tf, v[:, 1])
        g[ii], valid[ii] = pixels_to_ground(cam, tr, tf, v, np.inf)
        E = tr.T_world_ego(tm[ii])
        ran[ii] = np.linalg.norm(g[ii, :2] - E[:, :2, 3], axis=1)
    return g, ran, tm, valid


def feature_rms():
    s = Fit().load()
    X = np.load(WORK / "scene-v2.npz")["X"]
    tr = Trajectory(WORK / "ego.npz")
    ids = np.repeat(np.arange(len(X)), s.n)
    out = []
    for j, n in enumerate(NAMES):
        ix = np.flatnonzero(s.cam == j)
        err = []
        for i in range(0, len(ix), 20000):
            ind = ix[i : i + 20000]
            k = s.k[ind]
            c = s.cam[ind]
            uv = s.uv[ind]
            _, o, R, *_ = exact(s, tr, k, c, uv)
            e = s.project(c, X[ids[ind]], R, o, uv)[0]
            err.extend(e)
        e = np.asarray(err)
        out.append([n, len(e), np.sqrt(np.mean(e**2)), np.percentile(e, 95)])
    return out


def run():
    ctx = inputs()
    s, d, t, B, stop = ctx
    C = [load_rig(RS / "calib/rig.json")[n] for n in NAMES]
    allm = np.load(WORK / "pairs.npy")
    m = allm[allm[:, 10] == 1]
    ci = m[:, 1].astype(int)
    cj = m[:, 5].astype(int)
    ka = m[:, 0].astype(int)
    kb = m[:, 4].astype(int)
    rows = []
    summary = {}
    fracs = {}
    detail = {}
    models = [
        "planar",
        "offline-v1",
        "causal-v1",
        "offline-v1/local-level",
        "causal-v1/local-level",
        "offline",
        "causal",
    ]
    base = trajectory("planar")
    ga, ra, ta, oka = raymap(m, ctx, base, 0)
    gb, rb, tb, okb = raymap(m, ctx, base, 4)
    ranges = np.maximum(ra, rb)
    eligible = (
        oka
        & okb
        & (
            (ci != cj)
            | (abs(ta - tb) >= 1)
            | (abs(s["distance"][ka] - s["distance"][kb]) >= 5)
        )
    )
    same = ci == cj
    sg = []
    for seg, a, b in SEG:
        sg.append(
            (
                seg,
                (s["t"][ka] >= a)
                & (s["t"][ka] <= b)
                & (s["t"][kb] >= a)
                & (s["t"][kb] <= b),
            )
        )
    # The pavement mask is fixed using the planar image before reading V2.
    # Neither the tested shift nor the correlation score determines this mask.
    for seg, window in sg:
        bp = PRODUCT / "mosaics" / "planar" / seg
        mask = cv.imread(str(bp / "road-mask.png"), 0)
        gr = np.load(bp / "grid.npz")
        ii = np.flatnonzero(window)
        for world in (ga, gb):
            uv = np.rint(
                (world[ii, :2] - gr["origin"]) @ gr["basis"] / gr["cell"]
            ).astype(int)
            x, y = uv.T
            v = (x >= 0) & (x < mask.shape[1]) & (y >= 0) & (y < mask.shape[0])
            window[ii] &= v & (
                mask[np.clip(y, 0, mask.shape[0] - 1), np.clip(x, 0, mask.shape[1] - 1)]
                > 0
            )
    limiting = []
    for name in models:
        tr = (
            Trajectory(
                WORK / ("ego" + ("_causal" if name == "causal" else "") + ".npz")
            )
            if name in ("causal", "offline")
            else trajectory(name.split("/")[0])
        )
        if name.endswith("/local-level"):

            def horizontal_at_axle(tt):
                return tr.T_world_ego(tt)[..., 2, :3], np.zeros_like(tt)

            tr.ground_plane = horizontal_at_axle
        g0, _, _, v0 = raymap(m, ctx, tr, 0)
        g1, _, _, v1 = raymap(m, ctx, tr, 4)
        ee = np.linalg.norm(g0[:, :2] - g1[:, :2], axis=1) * 100
        if name == "offline":
            sensitivity = []
            for side, raw in [(0, g0), (4, g1)]:
                js = []
                for axis in [0, 1]:
                    mm = m.copy()
                    mm[:, side + 2 + axis] += 0.25
                    alt = raymap(mm, ctx, tr, side)[0]
                    js.append((alt[:, :2] - raw[:, :2]) * 400)
                sensitivity.append(
                    np.linalg.svd(np.nan_to_num(np.stack(js, 2)), compute_uv=False)[
                        :, 0
                    ]
                )
            sensitivity = np.maximum(*sensitivity)
            heading = tr.T_world_ego(ta)
            signed = np.einsum(
                "nji,nj->ni", heading[:, :2, :2], (g0[:, :2] - g1[:, :2]) * 100
            )
            signed[ci > cj] *= -1
        ok = eligible & v0 & v1 & np.isfinite(ee)
        fracs[name] = int(np.sum(eligible & ~ok))
        summary[name] = {}
        detail[name] = {}

        def val(q):
            return (
                [int(q.sum()), *np.percentile(ee[q], [50, 90]).tolist()]
                if q.any()
                else [0, None, None]
            )

        for seg, u in sg:
            q = ok & u & (ranges <= 10)
            summary[name][seg] = [val(q & same), val(q & ~same)]
            detail[name][seg] = []
            for i, n in enumerate(NAMES):
                for j in range(i, 6):
                    pair = ((ci == i) & (cj == j)) | ((ci == j) & (cj == i))
                    detail[name][seg].append([n, NAMES[j], *val(q & pair)])
                    if name == "offline" and (q & pair).any():
                        ii = q & pair
                        limiting.append(
                            [
                                seg,
                                n,
                                NAMES[j],
                                int(ii.sum()),
                                *np.median(signed[ii], axis=0).tolist(),
                                float(np.median(sensitivity[ii])),
                                float(np.median(m[ii, 8])),
                            ]
                        )
                    for lo, hi in [(0, 4), (4, 7), (7, 10), (10, 12), (0, 10)]:
                        rows.append(
                            [
                                name,
                                seg,
                                n,
                                NAMES[j],
                                lo,
                                hi,
                                *val(ok & u & pair & (ranges > lo) & (ranges <= hi)),
                            ]
                        )
    out = PRODUCT / "debug"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "patch-shifts.csv", "w") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "trajectory",
                "segment",
                "camera_a",
                "camera_b",
                "farther_radius_lo_m",
                "hi_m",
                "matches",
                "median_cm",
                "p90_cm",
            ]
        )
        w.writerows(rows)
    with open(out / "pair-diagnostics.csv", "w") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "segment",
                "camera_a",
                "camera_b",
                "n",
                "signed_fwd_median_cm",
                "signed_left_median_cm",
                "larger_radial_ground_scale_cm_per_native_px",
                "median_NCC",
            ]
        )
        w.writerows(limiting)
    spr = np.load(WORK / "spring.npz")
    off = dict(np.load(WORK / "ego.npz"))
    live = dict(np.load(WORK / "ego_causal.npz"))
    moving = ~stop
    ac = accel(t, d, stop)
    stats = {}
    arg = ["susp_pitch", "susp_roll", "susp_heave", "road_pitch", "road_roll"]
    for name, v in [("offline", off), ("causal", live)]:
        factor = np.array([57.2957795, 57.2957795, 100, 57.2957795, 57.2957795])
        amps = np.stack([v[k] * fac for k, fac in zip(arg, factor)], 1)
        Y, p, r = angles(v["T_world_ego"][:, :3, :3])
        stats[name] = dict(
            cor=[np.corrcoef(ac[moving, i], amps[moving, i])[0, 1] for i in range(2)],
            amplitudes=np.percentile(amps, [5, 50, 95], axis=0).tolist(),
            worldp95=np.rad2deg(
                np.percentile(abs(np.stack((p, r))), 95, axis=1)
            ).tolist(),
            height=np.percentile(v["T_world_ego"][:, 2, 3], [0, 100]).tolist(),
        )
    angle = np.stack([off[k] - live[k] for k in arg], 1) * [
        57.29578,
        57.29578,
        100,
        57.29578,
        57.29578,
    ]
    difference = np.percentile(abs(angle[moving]), [50, 90, 95], axis=0)
    rp = [angles(v["T_world_ego"][:, :3, :3])[1:] for v in (off, live)]
    stats["world_angle_diff"] = np.rad2deg(
        np.percentile(np.abs(np.array(rp[0]) - rp[1])[:, moving], [50, 95], axis=1)
    ).tolist()
    data = dict(
        summary=summary,
        detail=detail,
        invalid=fracs,
        feature=feature_rms(),
        attitude=stats,
        delta=difference.tolist(),
        gains=np.rad2deg(spr["gains"][2:]).tolist(),
        live_gain=np.rad2deg(spr["live_gains"][-1, :, 1]).tolist(),
        matches=[len(allm), len(m)],
    )
    json.dump(data, open(WORK / "results.json", "w"), indent=2)
    print(
        "summary",
        summary,
        "features",
        data["feature"],
        "gains",
        data["gains"],
        flush=True,
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(5, 1, figsize=(12, 8), sharex=True, layout="constrained")
    for i, (k, ttl) in enumerate(
        zip(
            arg,
            [
                "Suspension pitch (°), nose up",
                "Suspension roll (°), left up",
                "Heave (cm)",
                "Road grade (°), along heading",
                "Road crossfall (°), left up",
            ],
        )
    ):
        fac = 100 if i == 2 else 180 / np.pi
        ax[i].plot(t, off[k] * fac, label="offline", lw=1)
        ax[i].plot(t, live[k] * fac, label="causal", lw=0.7, alpha=0.7)
        ax[i].set_ylabel(ttl, fontsize=8)
        ax[i].grid(alpha=0.22)
        if i < 2:
            ax[i].plot(
                t,
                (spr["gains"][i] + spr["gains"][i + 2] * ac[:, i]) * fac,
                lw=0.8,
                c="black",
                label="fit × accel + intercept",
                alpha=0.7,
            )
        for _, a, b in SEG:
            ax[i].axvspan(a, b, color="orange", alpha=0.13)
    ax[0].legend(ncol=3)
    ax[-1].set_xlabel("Master time (s)")
    fig.savefig(out / "road-attitude.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    run()
