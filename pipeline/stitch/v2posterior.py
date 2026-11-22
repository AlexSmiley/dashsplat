"""Multiview opacity and thickness update for low-support depth fog; no orbit-space carving."""

from .v2clear import *


def observations(a):
    key = np.floor(a[:, :2] / 12).astype(int)
    code = key[:, 0] * 65536 + key[:, 1]
    order = np.argsort(code)
    b = np.r_[0, np.flatnonzero(np.diff(code[order])) + 1, len(code)]
    cent = (key[order[b[:-1]]] + 0.5) * 12
    bins = [order[i:j].astype("i") for i, j in zip(b[:-1], b[1:])]
    tree = cKDTree(cent)
    stats = np.zeros((len(a), 20), "f")
    dirs = np.zeros(len(a), "uint32")
    L = make_lib()
    vote = ct.CDLL(str(V / "match.dylib")).posterior
    vote.argtypes = (
        [ct.c_void_p, ct.c_void_p, ct.c_int] + [ct.c_void_p] * 9 + [ct.c_int]
    )
    nc = 0
    for lens, c in enumerate(CAMS):
        last = -10
        C = rig[c]
        K = np.ascontiguousarray(
            C.params * np.r_[[724 / C.width, 469 / C.height] * 2, [1] * 4], "d"
        )
        ray, ok = rays(c)
        for e in schedule()[c]:
            if e["hold"] or not e["depth"] or e["odo"] - last < 0.85:
                continue
            last = e["odo"]
            T = poses(c, e["t"])
            reach = 210 if c == "front" else 93
            which = tree.query_ball_point(T[4, :2, 3], reach + 12)
            front = (cent[which] - T[4, :2, 3]) @ T[4, :2, 2]
            which = np.asarray(which)[front > -12]
            ids = (
                np.concatenate([bins[i] for i in which])
                if len(which)
                else np.empty(0, "i")
            )
            lab = imask(c, e["n"])
            r, wt = exposure_range(e)
            ego = traj.T_world_ego(e["t"])
            ng, h = traj.ground_plane(e["t"])
            normal = T[4, :3, :3].T @ ego[:3, :3] @ ng
            den = ray @ normal
            camh = (T[4, :3, 3] - ego[:3, 3]) @ (ego[:3, :3] @ ng) + h
            rg = -camh / np.minimum(-0.01, den)
            ro = (lab == 3) & (den < -0.055) & (rg < 32)
            r[ro] = rg[ro]
            wt[ro] = 1
            im = np.ascontiguousarray(rgb(c, e["n"]))
            sky = cv.erode((lab == 2).astype("B"), np.ones((5, 5), "B"))
            sky_dist = cv.distanceTransform(1 - sky, cv.DIST_L2, 3)
            vote(
                a.ctypes.data,
                ids.ctypes.data,
                len(ids),
                T.ctypes.data,
                K.ctypes.data,
                lab.ctypes.data,
                r.ctypes.data,
                wt.ctypes.data,
                im.ctypes.data,
                stats.ctypes.data,
                dirs.ctypes.data,
                sky_dist.ctypes.data,
                lens,
            )
            nc += 1
    return stats, dirs, nc


def posterior(src, dst, silhouette=True):
    tic = time.monotonic()
    a = np.ascontiguousarray(
        np.concatenate(
            [
                reader(src / t["file"])
                for t in J(src / "map.json")["tiles"]
                if t["layer"] == "objects"
            ]
        ),
        "f",
    )
    stats, dirs, nc = observations(a)
    neg, both, srf, strong, seen, occ = stats[:, :6].T
    bits = np.unpackbits(dirs.view("uint8").reshape(-1, 4), axis=1).sum(1)
    ratio = (both + 0.15) / (seen + neg + 0.35)
    route = traj.T_world_ego(np.arange(START, END, 0.10))[:, :2, 3]
    lat = cKDTree(route).query(a[:, :2])[0]
    near = np.clip((45 - lat) / 32, 0, 1)
    conf = np.clip((ratio - 0.10) / 0.52, 0, 1) * np.clip((both + 0.15) / 2.0, 0, 1)
    good = (bits >= 2) & (both > 1.5) & (ratio > 0.38)
    conf[good] = np.maximum(conf[good], 0.82)
    unseen = (
        neg + seen < 0.12
    )  # preserve standalone rays, but don't pile up metre-thick discs with no sightline agreement
    factor = (0.08 + 0.92 * conf) ** (0.5 + near * 0.75)
    factor[unseen] = np.where(lat[unseen] < 30, 0.22, 0.65)
    factor = np.where((neg > 2.5) & (neg > 2 * both + 1.5), factor * 0.28, factor)
    alpha = 1 / (1 + np.exp(-a[:, 9]))
    alpha = 1 - (1 - alpha) ** factor
    reject = (alpha < 0.021) & (neg + seen > 1.5)
    a[:, 9] = np.log(np.maximum(0.001, alpha) / np.maximum(0.001, 1 - alpha))
    a[:, 12] += np.log(0.32 + 0.68 * conf)
    a[:, 10:12] += np.log(0.76 + 0.24 * conf)[:, None]
    # Dispersion of photographed luminance distinguishes a stable wall from mixed sky/foliage. Restore calibrated colour after the unconstrained RGB update.
    rgbmean = stats[:, 7:10] / np.maximum(0.001, stats[:, 6, None])
    luma = rgbmean.mean(1)
    scatter = np.sqrt(
        np.maximum(0, stats[:, 15] / np.maximum(0.001, stats[:, 6]) - luma**2)
    )
    prior = (0.58 * rgbmean + 0.42 * stats[:, 11:14]) / 255
    old = a[:, 6:9] * 0.28209479 + 0.5
    lo = np.maximum(0, prior - 0.045)
    hi = np.minimum(1, prior + 0.045)
    colour = np.clip(old, lo, hi)
    wcol = np.clip(stats[:, 6] * 2, 0, 0.96) * np.where(scatter < 24, 1, 0.78)
    a[:, 6:9] = (old + wcol[:, None] * (colour - old) - 0.5) / 0.28209479
    # A low-support haze Gaussian must not grow to metres across when its observed pixel footprint is only centimetres. High-confidence coarse surfaces can remain large.
    bound = np.where(
        stats[:, 10] > 0,
        np.clip(stats[:, 14] * 5, 0.07, 1.65),
        np.clip(0.08 + 0.009 * lat, 0.09, 1.35),
    ) * (1 + 0.90 * conf)
    a[:, 10:13] = np.minimum(a[:, 10:13], np.log(bound)[:, None])
    edge = stats[:, 16:19]
    ratio_edge = edge / np.maximum(1, stats[:, 19, None])
    tight = (edge > 1.35) & (ratio_edge > 0.32) & silhouette
    a[:, 10:13] += np.log(np.where(tight, np.maximum(0.32, 1 - 0.83 * ratio_edge), 1))
    scene(dst, a[~reject])
    res = dict(
        total=len(a),
        pruned=int(reject.sum()),
        angular_colour_depth_agree=int(good.sum()),
        unobserved=int(unseen.sum()),
        observations=nc,
        sky_edge_axes_shrunk=int(tight.sum()),
        sky_edge_splats=int(tight.any(1).sum()),
        fitted_colour=int((wcol > 0.2).sum()),
        seconds=time.monotonic() - tic,
        median_optical_thickness_multiplier=float(np.median(factor)),
    )
    savej(V / (dst.name + "-posterior.json"), res)
    print(res, flush=True)


if __name__ == "__main__":
    import sys

    posterior(Path(sys.argv[1]), Path(sys.argv[2]), "--seed" not in sys.argv[3:])
