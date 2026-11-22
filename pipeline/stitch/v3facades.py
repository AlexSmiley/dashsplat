"""Piecewise vertical facade planes seeded by displaced real-image tracks; support texels by independent projected views."""

from .v3terrain import *
from scipy.spatial import ConvexHull


def facade():
    points = np.load(W / "patches.npy")
    v = Volume()
    g = v.lookup(points[:, :3])[0]
    lat = cKDTree(traj.T_world_ego(np.arange(START, END, 0.16))[:, :2, 3]).query(
        points[:, :2]
    )[0]
    a = points[(lat > 15) & (points[:, 2] > g + 0.9) & (abs(points[:, 5]) < 0.72)]
    angle = (np.arctan2(a[:, 4], a[:, 3]) + np.pi) % np.pi
    groups = np.c_[np.floor(a[:, :2] / 12), np.floor(angle / (np.pi / 12))].astype(int)
    _, idx = np.unique(groups, axis=0, return_inverse=True)
    order = np.argsort(idx)
    b = np.r_[0, np.flatnonzero(np.diff(idx[order])) + 1, len(order)]
    parts = []
    planes = 0
    for l, h in zip(b[:-1], b[1:]):
        if h - l < 20:
            continue
        subset = a[order[l:h]]
        X = subset[:, :3]
        N = subset[:, 3:5]
        N = unit(np.where((N @ N[0] < 0)[:, None], -N, N).mean(0))
        bias = X[:, :2] @ N
        res = []
        for _ in range(2):
            if len(bias) < 18:
                break
            bins = np.floor(bias / 0.38).astype(int)
            val, count = np.unique(bins, return_counts=True)
            B = (val[count.argmax()] + 0.5) * 0.38
            ok = abs(bias - B) < 0.51
            Y = X[ok]
            X = X[~ok]
            bias = bias[~ok]
            if len(Y) < 16:
                continue
            center = np.median(Y[:, :2], 0)
            _, _, vh = np.linalg.svd(Y[:, :2] - center, full_matrices=False)
            normal = vh[-1]
            tangent = vh[0]
            uv = np.c_[(Y[:, :2] - center) @ tangent, Y[:, 2]]
            size = np.ptp(uv, axis=0)
            if (
                size[0] < 2.4
                or size[1] < 1.55
                or np.percentile(abs((Y[:, :2] - center) @ normal), 80) > 0.39
            ):
                continue
            hull = ConvexHull(uv)
            step = 0.32
            u, z = np.meshgrid(
                np.arange(uv[:, 0].min(), uv[:, 0].max(), step),
                np.arange(uv[:, 1].min(), uv[:, 1].max(), step),
            )
            q = np.c_[u.ravel(), z.ravel()]
            valid = (q @ hull.equations[:, :2].T + hull.equations[:, 2] < 0).all(1) & (
                cKDTree(uv).query(q)[0] < 1.65
            )
            q = q[valid]
            if not len(q):
                continue
            pos = np.c_[center + q[:, :1] * tangent, q[:, 1]]
            P = surfels(
                pos, np.ones(3) * 90, [0.26, 0.25, 0.065], 0.86, subset[:, 19].max()
            )
            n = np.r_[normal, 0]
            tx = np.r_[-normal[1], normal[0], 0]
            P[:, 3:6] = n
            P[:, 13:17] = Rotation.from_matrix(
                np.stack([tx, [0, 0, 1.0], n], 1)
            ).as_quat()[[3, 0, 1, 2]]
            parts.append(P)
            planes += 1
    A = np.concatenate(parts) if parts else np.empty((0, 24), "f")
    _, j = np.unique(np.floor(A[:, :3] / 0.26).astype(int), axis=0, return_index=True)
    A = A[j]
    P = A[:, :3]
    tree = cKDTree(P[:, :2])
    cnt = np.zeros(len(A), int)
    neg = cnt.copy()
    sq = np.zeros(len(A))
    rgb0 = np.zeros((len(A), 3))
    mins = np.full(len(A), 1e8)
    maxs = -mins
    nearest = np.full(len(A), 500.0)
    anchor = np.zeros_like(rgb0)
    last = {}
    for e in J(V / "train.json"):
        c = e["c"]
        yaw = e["yaw"]
        key = c, yaw
        if e["odo"] - last.get(key, -100) < 2.4:
            continue
        last[key] = e["odo"]
        w, h, f, R, _, _, bands, _ = pin(c, yaw)
        M = poses(c, e["t"])
        ix = np.array(tree.query_ball_point(M[4, :2, 3], 180), int)
        if not len(ix):
            continue
        X = (P[ix] - M[4, :3, 3]) @ M[4, :3, :3] @ R
        z = X[:, 2].clip(0.01)
        x = (w / 2 + f * X[:, 0] / z).round().astype(int).clip(0, w - 1)
        y = (h / 2 + f * X[:, 1] / z).round().astype(int).clip(0, h - 1)
        T = M[bands[y, x].clip(0, 7)]
        X = np.einsum("ni,nij->nj", P[ix] - T[:, :3, 3], T[:, :3, :3] @ R)
        z = X[:, 2].clip(0.01)
        x = (w / 2 + f * X[:, 0] / z).round().astype(int)
        y = (h / 2 + f * X[:, 1] / z).round().astype(int)
        sel = (z > 5) & (x > 1) & (y > 1) & (x < w - 2) & (y < h - 2)
        ix, x, y, z = [j[sel] for j in [ix, x, y, z]]
        if not len(ix):
            continue
        with np.load(e["file"]) as d:
            lab = d["lab"][y, x]
            im = d["rgb"][y, x]
            dep = d["z"]
            wt = d["wt"]
            vis = lab == 1
        if dep.size:
            dr = dep[y, x]
            conf = wt[y, x]
            vis &= (conf < 8) | ((dr * 1.26 + 1.2 > z) & (dr * 0.73 - 1.3 < z))
        neg[ix[lab == 2]] += 1
        ix, z, im = [j[vis] for j in [ix, z, im]]
        cnt[ix] += 1
        rgb0[ix] += im
        sq[ix] += im.mean(1) ** 2
        mins[ix] = np.minimum(mins[ix], e["odo"])
        maxs[ix] = np.maximum(maxs[ix], e["odo"])
        better = z < nearest[ix]
        nearest[ix[better]] = z[better]
        anchor[ix[better]] = im[better]
    mean = rgb0 / np.maximum(1, cnt[:, None])
    scatter = np.sqrt(np.maximum(0, sq / np.maximum(1, cnt) - mean.mean(1) ** 2))
    ok = (
        (cnt >= 5)
        & (maxs - mins > 4)
        & (scatter < 17)
        & (neg < 0.17 * cnt)
        & (nearest < 140)
    )
    color = 0.65 * mean + 0.35 * np.clip(anchor, mean - 22, mean + 22)
    A[:, 6:9] = (color / 255 - 0.5) / 0.28209479
    np.save(W / "walls.npy", A[ok])
    res = dict(
        plane_pieces=planes,
        candidates=len(A),
        accepted=int(ok.sum()),
        median_baseline=float(np.median((maxs - mins)[ok])),
    )
    savej(W / "walls.json", res)
    print(res, flush=True)


if __name__ == "__main__":
    facade()
