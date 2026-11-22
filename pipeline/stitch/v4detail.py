"""Real multiview-supported object detail, measured in chase teacher optical depth. Replace, not stack, overlapping support."""

from .v4teacher import *
from .v2seed import unit
from .v3terrain import surfels
from .v3build import support_bound


def propose():
    train, _ = split()
    vol = Volume()
    rows = []
    src = [
        reader(SRC / t["file"])
        for t in J(SRC / "map.json")["tiles"]
        if t["layer"] == "objects"
    ]
    P = np.concatenate([a[:, :3] for a in src])
    anchor = cKDTree(P)
    rt = cKDTree(traj.T_world_ego(np.arange(START, END, 0.12))[:, :2, 3])
    nframes = 0
    for e in train:
        if e["kind"] != "chase":
            continue
        rgb, w, sk, dm = decode(e)
        y, x = np.mgrid[
            5 + e["station"] % 4 : 895 : 4, 5 + e["base_frame"] % 4 : 1435 : 4
        ]
        z = dm.astype("f") * 0.01
        d = z[y, x]
        ok = (w[y, x] >= 115) & (d > 4) & (d < 185)
        y, x, d = [s[ok] for s in [y, x, d]]
        K = e["K"]
        T = np.array(e["T_world_cam"])

        def pt(x, y, d):
            return (
                T[:3, 3]
                + np.c_[(x + 0.5 - K[2]) / K[0] * d, (y + 0.5 - K[3]) / K[1] * d, d]
                @ T[:3, :3].T
            )

        X = pt(x, y, d)
        g, h, _ = vol.lookup(X)
        ok = (X[:, 2] > np.maximum(h + 0.15, g + 0.45)) & (X[:, 2] < g + 38)
        y, x, d, X = [s[ok] for s in [y, x, d, X]]
        if not len(x):
            continue
        # Metric support must already exist in the real-camera surface cloud. Coarse or absent structures get no invented positions.
        ds = anchor.query(X, distance_upper_bound=2.4)[0]
        r = np.linalg.norm(X - T[:3, 3], axis=1)
        ok = ds < np.minimum(1.6, 0.16 + 0.012 * r)
        X, x, y, d, ds, r = [s[ok] for s in [X, x, y, d, ds, r]]
        n = unit(
            np.cross(
                pt(x + 1, y, z[y, x + 1]) - pt(x - 1, y, z[y, x - 1]),
                pt(x, y + 1, z[y + 1, x]) - pt(x, y - 1, z[y - 1, x]),
            )
        )
        edge = np.maximum.reduce(
            [abs(z[y, x + j] - d) for j in [-2, 2]]
            + [abs(z[y + j, x] - d) for j in [-2, 2]]
        )
        ok = (edge < 0.22 + 0.035 * d) & (np.linalg.norm(n, axis=1) > 0.98)
        X, x, y, d, ds, r, n = [s[ok] for s in [X, x, y, d, ds, r, n]]
        n = np.where(((X - T[:3, 3]) * n).sum(1)[:, None] > 0, -n, n)
        lat = rt.query(X[:, :2])[0]
        lev = np.digitize(lat, [13, 35, 70])
        pitch = np.array([0.073, 0.14, 0.26, 0.53])[lev]
        ij = np.floor(X / pitch[:, None]).astype("i8") + 30000
        key = ((lev * 65536 + ij[:, 0]) * 65536 + ij[:, 1]) * 65536 + ij[:, 2]
        score = w[y, x] / 255 / (1 + e["base_distance_m"] * 0.08) / (1 + r * 0.012)
        o = np.lexsort((-score, key))
        take = o[np.r_[True, np.diff(key[o]) != 0]] if len(o) else o
        rows.append(
            (
                key[take],
                np.c_[
                    X,
                    n,
                    rgb[y, x],
                    pitch,
                    score,
                    np.full(len(x), e["travel_m"]),
                    np.full(len(x), e["base_frame"]),
                    np.full(len(x), e["t"]),
                ][take].astype("f"),
            )
        )
        nframes += 1
    k = np.concatenate([r[0] for r in rows])
    a = np.concatenate([r[1] for r in rows])
    o = np.lexsort((-a[:, 10], k))
    k = k[o]
    a = a[o]
    s = np.r_[0, 1 + np.flatnonzero(np.diff(k))]
    num = np.diff(np.r_[s, len(a)])
    w = a[:, 10]
    tot = np.add.reduceat(w, s)
    mean = np.add.reduceat(a[:, 6:9] * w[:, None], s) / tot[:, None]
    lum = a[:, 6:9].mean(1)
    var = np.sqrt(
        np.maximum(0, np.add.reduceat(lum**2 * w, s) / tot - mean.mean(1) ** 2)
    )
    span = np.maximum.reduceat(a[:, 11], s) - np.minimum.reduceat(a[:, 11], s)
    distinct = np.maximum.reduceat(a[:, 12], s) != np.minimum.reduceat(a[:, 12], s)
    norm = unit(np.add.reduceat(a[:, 3:6] * w[:, None], s))
    coherence = (norm * a[s, 3:6]).sum(1)
    ok = (
        (num >= 3)
        & (span > 2.2)
        & distinct
        & (tot > 0.6)
        & (var < 16)
        & (coherence > 0.6)
    )
    B = a[s[ok]]
    N = B[:, 3:6]
    xx = unit(
        np.cross(np.where((abs(N[:, 2]) > 0.84)[:, None], [0, 1.0, 0], [0, 0, 1.0]), N)
    )
    yy = unit(np.cross(N, xx))
    q = Rotation.from_matrix(np.stack([xx, yy, N], 2)).as_quat()[:, [3, 0, 1, 2]]
    sig = B[:, 9] * 0.82
    A = surfels(
        B[:, :3],
        0.7 * B[:, 6:9] + 0.3 * mean[ok],
        np.c_[sig, sig, np.maximum(0.018, sig * 0.28)],
        0.80,
        B[:, 13],
    )
    A[:, 3:6] = N
    A[:, 13:17] = q
    A = support_bound(A, vol)
    np.save(OUT / "detail.npy", A)
    res = dict(
        observations=len(a),
        voxels=len(s),
        accepted=len(A),
        median_teacher_span_m=float(np.median(span[ok])),
        stations=nframes,
    )
    savej(OUT / "detail.json", res)
    print(res, flush=True)


def merge(src, dst):
    extra = np.load(OUT / "detail.npy")
    tree = cKDTree(extra[:, :3])
    tiles = []
    meta = J(src / "map.json")
    kill = 0
    assign = np.floor(extra[:, :2] / 32).astype(int)
    covered = np.zeros(len(extra), bool)
    for t in meta["tiles"]:
        A = reader(src / t["file"])
        if t["layer"] == "ground":
            tiles.append(A)
            continue
        key = tuple(map(int, t["id"].split("_")[-2:]))
        which = np.flatnonzero((assign == key).all(1))
        covered[which] = True
        dist, idx = tree.query(A[:, :3], distance_upper_bound=1.6)
        over = dist < np.minimum(0.46, np.exp(A[:, 10:12]).min(1) * 0.8 + 0.04)
        i = np.flatnonzero(over)
        angle = abs((A[i, 3:6] * extra[idx[i], 3:6]).sum(1))
        over[i] &= (angle > 0.62) & (dist[i] < np.exp(extra[idx[i], 10]) * 0.89)
        kill += int(over.sum())
        tiles.append(np.concatenate([A[~over], extra[which]]))
    write_scene(src, dst, tiles)
    print("replaced", kill, "with", int(covered.sum()), flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        propose()
    else:
        merge(Path(sys.argv[1]), Path(sys.argv[2]))
