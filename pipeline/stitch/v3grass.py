"""Complete small holes around measured turf via lateral-view texture agreement on the ground plane."""

from .v3terrain import *
from scipy.ndimage import distance_transform_edt


def complete():
    v = Volume()
    t0 = np.load(W / "verge.npy")
    grounds = tiles_ground()
    src = np.concatenate([reader(GROUND / t["file"])[::3] for t in grounds])
    col = 0.28209479 * src[:, 6:9] + 0.5
    turf = (col[:, 1] > col[:, 0] * 1.13) & (col[:, 1] > col[:, 2] * 1.06)
    q = np.concatenate([t0[:, :2], src[turf, :2]])
    pitch = 0.20
    origin = q.min(0) - 5
    dims = np.ceil((q.max(0) - origin + 5) / pitch).astype(int)
    mask = np.ones(dims[::-1], "B")
    ix, iy = np.floor((q - origin) / pitch).astype(int).T
    mask[iy, ix] = 0
    D = distance_transform_edt(mask)
    iy, ix = np.nonzero(D < 4.2 / pitch)
    xy = origin + (np.c_[ix, iy] + 0.5) * pitch
    g, h, _ = v.lookup(xy)
    route = cKDTree(traj.T_world_ego(np.arange(START, END, 0.14))[:, :2, 3])
    lat = route.query(xy)[0]
    keep = (
        (lat > 2.85) & (lat < 40) & (h < g) & (cKDTree(t0[:, :2]).query(xy)[0] > 0.15)
    )
    xy = xy[keep]
    g = g[
        keep
    ]  # Exclude footprints of all frozen ground, not merely centres at grid resolution.
    src = np.concatenate([reader(GROUND / t["file"]) for t in grounds])
    tree = cKDTree(src[:, :2])
    dd, ids = tree.query(xy, k=4)
    rad = np.exp(src[ids, 10:12]).min(2)
    good = np.min(dd / np.maximum(0.042, rad), axis=1) > 1.42
    P = np.c_[xy[good], g[good] + 0.015]
    del src, tree
    N = len(P)
    wgt = np.zeros(N)
    vs = np.zeros((N, 3))
    sq = np.zeros(N)
    seen = np.zeros(N)
    cnt = np.zeros(N, int)
    mi = np.full(N, 1e9)
    mx = np.full(N, -1e9)
    tm = np.zeros(N)
    az0 = np.zeros((N, 2))
    sep = np.zeros(N)
    last = {}
    tree = cKDTree(P[:, :2])
    nf = 0
    for e in J(V / "train.json"):
        c = e["c"]
        yaw = e["yaw"]
        key = (c, yaw)
        if e["odo"] - last.get(key, -100) < 0.80:
            continue
        last[key] = e["odo"]
        w, h, f, R, _, _, bands, _ = pin(c, yaw)
        M = poses(c, e["t"])
        ids = np.array(tree.query_ball_point(M[4, :2, 3], 32), int)
        if not len(ids):
            continue
        X = (P[ids] - M[4, :3, 3]) @ M[4, :3, :3] @ R
        z = X[:, 2].clip(0.001)
        x = (w / 2 + f * X[:, 0] / z).astype(int).clip(0, w - 1)
        y = (h / 2 + f * X[:, 1] / z).astype(int).clip(0, h - 1)
        T = M[bands[y, x].clip(0, 7)]
        delta = P[ids] - T[:, :3, 3]
        X = np.einsum("ni,nij->nj", delta, T[:, :3, :3] @ R)
        z = X[:, 2].clip(0.001)
        x = (w / 2 + f * X[:, 0] / z).astype(int)
        y = (h / 2 + f * X[:, 1] / z).astype(int)
        s = (
            (z > 1.4)
            & (x >= 1)
            & (x < w - 2)
            & (y >= 1)
            & (y < h - 2)
            & (delta[:, 2] < -0.055 * np.linalg.norm(delta, axis=1))
        )
        ids, x, y, z, delta = [a[s] for a in [ids, x, y, z, delta]]
        if not len(ids):
            continue
        with np.load(e["file"]) as a:
            lab = a["lab"][y, x]
            im = a["rgb"][y, x].astype(float)
            dep = a["z"]
            wt = a["wt"]
            conf = np.ones(len(ids)) * 0.38
        if dep.size:
            d = dep[y, x]
            wt = wt[y, x]
            conf = np.where(
                (wt > 8) & (d > 0), np.exp(-(((d - z) / (0.42 + z * 0.12)) ** 2)), conf
            )
        acceptable = (lab == 1) & (conf > 0.09)
        seen[ids] += lab != 0
        ids = ids[acceptable]
        I = im[acceptable]
        d = delta[acceptable]
        ww = conf[acceptable] / (1 + z[acceptable] / 16)
        nd = unit(d[:, :2])
        first = cnt[ids] == 0
        az0[ids[first]] = nd[first]
        sep[ids] = np.maximum(
            sep[ids], np.arccos(np.clip((az0[ids] * nd).sum(1), -1, 1))
        )
        wgt[ids] += ww
        cnt[ids] += 1
        vs[ids] += ww[:, None] * I
        sq[ids] += ww * I.mean(1) ** 2
        mi[ids] = np.minimum(mi[ids], e["odo"])
        mx[ids] = np.maximum(mx[ids], e["odo"])
        tm[ids] = np.maximum(tm[ids], e["t"])
        nf += 1
    rgb = vs / np.maximum(0.001, wgt[:, None])
    scatter = np.sqrt(np.maximum(0, sq / np.maximum(0.001, wgt) - rgb.mean(1) ** 2))
    green = (rgb[:, 1] > 1.07 * rgb[:, 0]) & (rgb[:, 1] > 1.018 * rgb[:, 2])
    ok = (
        (cnt >= 4)
        & (wgt > 1.65)
        & (mx - mi > 1.55)
        & (sep > 0.095)
        & (scatter < np.where(green, 24, 13))
        & (cnt / np.maximum(1, seen) > 0.40)
        & (green | (scatter < 9))
    )
    extra = surfels(P[ok], rgb[ok], [pitch * 0.88, pitch * 0.88, 0.012], 0.94, tm[ok])
    np.save(W / "grass.npy", np.concatenate([t0, extra]))
    res = dict(
        frames=nf,
        projected=len(P),
        new=len(extra),
        original=len(t0),
        rgb_median=np.median(rgb[ok], 0).tolist(),
        median_photos=float(np.median(cnt[ok])),
    )
    savej(W / "grass.json", res)
    print(res, flush=True)


if __name__ == "__main__":
    complete()
