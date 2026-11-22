"""Multi-baseline optical support for previously empty scenery; no learned or painted backdrop."""

from .views import *
from pipeline.road.scene import reader, binary_header
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from concurrent.futures import ThreadPoolExecutor
import shutil, argparse


@lru_cache(36)
def frame(p):
    with np.load(p) as a:
        return a["rgb"], a["mask"], a["z"]


def norm(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-10)


def lens(e, uv):
    c, y = e["c"], e["yaw"]
    K, R, mx, my, band, _ = design(c, y)
    ij = np.rint(uv).astype("i")
    ij = np.clip(ij, [0, 0], [W - 1, H - 1])
    b = band[ij[:, 1], ij[:, 0]]
    T = np.array([world(c, y, e["t"], j) for j in range(BANDS)])[b.clip(0)]
    ray = norm(np.c_[uv + 0.5, np.ones(len(uv))] @ np.linalg.inv(K).T)
    return T[:, :3, 3], np.einsum("nij,nj->ni", T[:, :3, :3], ray), b, ij


def trace(es):
    records = []
    poses = np.stack([world(e["c"], e["yaw"], e["t"], 3)[:3, 3] for e in es])
    dist = np.r_[0, np.cumsum(np.linalg.norm(np.diff(poses, axis=0), axis=1))]
    last = -10
    for j, e in enumerate(es):
        if dist[j] - last < 1.4:
            continue
        bef = np.searchsorted(dist, dist[j] - 1.65, side="right") - 1
        aft = np.searchsorted(dist, dist[j] + 1.65)
        last = dist[j]
        if (
            bef < 0
            or aft >= len(es)
            or max(dist[aft] - dist[j], dist[j] - dist[bef]) > 4.2
        ):
            continue
        ee = [es[bef], e, es[aft]]
        ims = [frame(path(q)) for q in ee]
        gray = [cv.cvtColor(x[0], cv.COLOR_RGB2GRAY) for x in ims]
        c = e["c"]
        C = rig[c]
        K, R, mx, my, band, _ = design(c, e["yaw"])
        rm = cv.imread(
            str(RS / "masks/road" / c / ("%06d.png" % e["n"])), cv.IMREAD_GRAYSCALE
        )
        road = remap(rm, mx, my, C, cv.INTER_NEAREST) > 0
        mask = (
            cv.erode((ims[1][1] == 1).astype("B"), np.ones((5, 5), "B")).astype(bool)
            & ~road
        )
        # A lattice gives local neighbours for metric tangents. Reject weak texture rather than guessing disparity.
        ys, xs = np.mgrid[4 : H - 4 : 3, 4 : W - 4 : 3]
        grid = np.stack([xs, ys], -1)
        s0 = mask[ys, xs] & (cv.cornerMinEigenVal(gray[1], 5)[ys, xs] > 0.00005)
        P = grid[s0].astype("f")
        if len(P) < 10:
            continue
        us = []
        good = np.ones(len(P), bool)
        kw = dict(
            winSize=(17, 17),
            maxLevel=4,
            criteria=(cv.TERM_CRITERIA_COUNT | cv.TERM_CRITERIA_EPS, 24, 0.015),
            minEigThreshold=0.00014,
        )
        for k in [0, 2]:
            q, st, err = cv.calcOpticalFlowPyrLK(
                gray[1], gray[k], P[:, None], None, **kw
            )
            rev, sr, _ = cv.calcOpticalFlowPyrLK(gray[k], gray[1], q, None, **kw)
            q = q[:, 0]
            jj = np.clip(q.round().astype("i"), [0, 0], [W - 1, H - 1])
            ok = (
                (st[:, 0] > 0)
                & (sr[:, 0] > 0)
                & (np.linalg.norm(rev[:, 0] - P, axis=1) < 0.50)
                & (err[:, 0] < 12)
                & (q[:, 0] > 3)
                & (q[:, 0] < W - 3)
                & (q[:, 1] > 3)
                & (q[:, 1] < H - 3)
                & (ims[k][1][jj[:, 1], jj[:, 0]] == 1)
            )
            good &= ok
            us.append(q)
        u = [us[0], P, us[1]]
        v = [lens(q, uu) for q, uu in zip(ee, u)]
        o0, d0, _, _ = v[0]
        o2, d2, _, _ = v[2]
        o, d, b, ij = v[1]
        cross = np.cross(d0, d2)
        sine = np.linalg.norm(cross, axis=1)
        base = o2 - o0
        v0 = np.sum(np.cross(base, d2) * cross, axis=1) / np.maximum(sine**2, 1e-8)
        v2 = np.sum(np.cross(base, d0) * cross, axis=1) / np.maximum(sine**2, 1e-8)
        X = (o0 + d0 * v0[:, None] + o2 + d2 * v2[:, None]) * 0.5
        r = np.linalg.norm(X - o, axis=1)
        pX = norm(X - o)
        err = np.linalg.norm(pX - d, axis=1) * K[0, 0]
        line = np.linalg.norm(o0 + d0 * v0[:, None] - o2 - d2 * v2[:, None], axis=1)
        good &= (
            (err < 0.68)
            & (line / np.maximum(r, 1) * K[0, 0] < 1.0)
            & (sine > 0.045)
            & (r > 2.8)
            & (r < 82)
            & (v0 > 0)
            & (v2 > 0)
            & (r * 0.55 / (K[0, 0] * np.maximum(sine, 0.001)) < 1.5)
        )
        # Where training-only DA3 exists, veto a competing surface with clearly different depth.
        if ims[1][2].size:
            dep = ims[1][2][ij[:, 1], ij[:, 0]].astype("f")
            rayz = d @ (world(c, e["yaw"], e["t"], 3)[:3, :3] @ R.T[:, 2])
            good &= (dep < 1) | (abs(dep - r * rayz) < np.maximum(2.5, 0.28 * dep))
        ego = traj.T_world_ego(e["t"])
        normal, h = traj.ground_plane(e["t"])
        height = ((X - ego[:3, 3]) @ ego[:3, :3]) @ normal + h
        good &= height > 0.40
        # Require adjacent independently matched lattice rays on the same surface in both axes.
        dense = np.full((*s0.shape, 3), np.nan)
        dense[s0] = np.where(good[:, None], X, np.nan)
        dx = np.roll(dense, -1, 1) - np.roll(dense, 1, 1)
        dy = np.roll(dense, -1, 0) - np.roll(dense, 1, 0)
        dx[:, [0, -1]] = np.nan
        dy[[0, -1]] = np.nan
        dx, dy = dx[s0], dy[s0]
        sx, sy = np.linalg.norm(dx, axis=1) * 0.5, np.linalg.norm(dy, axis=1) * 0.5
        pix = 3 * r / K[0, 0]
        good &= (
            (sx < 3.4 * pix + 0.08)
            & (sy < 3.4 * pix + 0.08)
            & (sx > 0.35 * pix)
            & (sy > 0.35 * pix)
        )
        n = norm(np.cross(dx, dy))
        n = np.where(np.sum(n * d, axis=1)[:, None] > 0, -n, n)
        good &= np.isfinite(n).all(1) & (abs(np.sum(n * d, axis=1)) > 0.20)
        xx = norm(dx)
        yy = norm(np.cross(n, xx))
        good &= np.linalg.norm(yy, axis=1) > 0.9
        if not good.any():
            continue
        rgb = ims[1][0][ij[:, 1], ij[:, 0]].astype("f") / 255
        a = np.zeros((good.sum(), 24), "f")
        a[:, :3] = X[good]
        a[:, 3:6] = n[good]
        a[:, 6:9] = (rgb[good] - 0.5) / 0.28209479
        a[:, 9] = 0.7
        a[:, 10:13] = np.log(
            np.column_stack(
                [
                    np.clip(sx[good] * 0.87, 0.045, 0.45),
                    np.clip(sy[good] * 0.87, 0.045, 0.45),
                    np.clip(pix[good] * 0.13, 0.012, 0.08),
                ]
            )
        )
        a[:, 13:17] = Rotation.from_matrix(
            np.stack([xx[good], yy[good], n[good]], 2)
        ).as_quat()[:, [3, 0, 1, 2]]
        a[:, 17] = 0
        a[:, 18:] = 1e9
        a[:, 19] = ee[0]["t"]
        a[:, 20:24] = [e["t"], e["t"], ee[2]["t"], ee[2]["t"]]
        records.append(a)
    return np.concatenate(records) if records else np.empty((0, 24), "f")


def photo_seeds(src, dst):
    if dst.exists():
        raise FileExistsError(f"Use a new seed-scene directory: {dst}")
    tic = time.monotonic()
    es = [e for e in J(CACHE / "manifest.json") if not e["hold"]]
    groups = [
        [e for e in es if e["c"] == c and e["yaw"] == y]
        for c in CAMS
        for y in designs(c)
    ]
    with ThreadPoolExecutor(3) as ex:
        raw = np.concatenate(list(ex.map(trace, groups)))
    # Distinct triplets must agree in world position and photographic colour. A neighbour from the same image is not confirmation.
    raw_n = len(raw)
    tree = cKDTree(raw[:, :3])
    positive = np.zeros(raw_n, "B")
    chronology = raw[:, 19:24].copy()
    for j in range(0, len(raw), 20000):
        a = raw[j : j + 20000]
        d, ix = tree.query(a[:, :3], k=min(18, len(raw)))
        b = raw[ix]
        different = (abs(b[:, :, 20] - a[:, None, 20]) > 0.16) & (
            (b[:, :, 19] != a[:, None, 19]) & (b[:, :, 22] != a[:, None, 22])
        )
        normal = abs((b[:, :, 3:6] * a[:, None, 3:6]).sum(2))
        colour = abs(b[:, :, 6:9] - a[:, None, 6:9]).mean(2) * 0.28209479
        ok = (
            different
            & (d < 0.22 + np.exp(a[:, 10:11]) * 0.70)
            & (normal > 0.68)
            & (colour < 0.085)
        )
        positive[j : j + len(a)] = ok.any(1)
        for i in np.flatnonzero(ok.any(1)):
            seen = np.unique(
                np.r_[a[i, [19, 20, 22]], b[i, ok[i]][:, [19, 20, 22]].ravel()]
            )
            breaks = np.flatnonzero(np.diff(seen) > 0.25) + 1
            runs = np.split(seen, breaks)
            runs = runs if len(runs) <= 3 else [runs[0], runs[1], runs[-1]]
            info = [runs[0][-1]]
            info.extend(z for run in runs[1:] for z in (run[0], run[-1]))
            chronology[j + i] = info + [1e9] * (5 - len(info))
    raw[:, 19:24] = chronology
    raw = raw[positive.astype(bool)]
    meta = J(src / "map.json")
    existing = np.concatenate(
        [
            reader(src / t["file"])[:, :3]
            for t in meta["tiles"]
            if t["layer"] == "objects"
        ]
    )
    dist = cKDTree(existing).query(raw[:, :3])[0]
    raw = raw[dist > np.maximum(0.12, np.exp(raw[:, 10]) * 0.6)]
    voxel = np.floor(raw[:, :3] / 0.11).astype("i")
    idx = np.unique(voxel, axis=0, return_index=True)[1]
    raw = raw[np.sort(idx)]
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "tiles").mkdir(exist_ok=True)
    tiles = []
    for t in meta["tiles"]:
        dest = dst / t["file"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            os.link(src / t["file"], dest)
        tiles.append(t)
    for fn in ["drive.json", "rig.json"]:
        shutil.copyfile(src / fn, dst / fn)
    xy = np.floor(raw[:, :2] / 50).astype(int)
    unique, ids = np.unique(xy, axis=0, return_inverse=True)
    for j, coord in enumerate(unique):
        a = raw[ids == j]
        name = "stereo_%d_%d" % tuple(coord)
        fn = "tiles/" + name + ".ply"
        with (dst / fn).open("wb") as f:
            f.write(binary_header(len(a)))
            a.tofile(f)
        tiles.append(
            dict(
                id=name,
                kind="static",
                layer="objects",
                file=fn,
                count=len(a),
                bbox=np.r_[a[:, :3].min(0), a[:, :3].max(0)].tolist(),
                t_min=0,
                t_max=meta["t_end"],
            )
        )
    savej(dst / "map.json", dict(meta, tiles=tiles))
    res = dict(
        raw=raw_n,
        confirmed=int(positive.sum()),
        new=len(raw),
        seconds=time.monotonic() - tic,
    )
    savej(WORK / "photo-seeds.json", res)
    print(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    photo_seeds(args.source, args.out)
