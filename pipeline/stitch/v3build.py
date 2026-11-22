"""Use real multiview evidence to concentrate crowns, remove near-floor haze and recover coarse distant surfaces."""

from .v3evidence import *
from .v2seed import scene, shutil
from scipy.ndimage import maximum_filter


def support_bound(a, volume):
    from scipy.spatial.transform import Rotation

    floor = np.maximum(volume.h, volume.g + np.where(volume.owned, 0.085, 0.015))
    R = Rotation.from_quat(a[:, [14, 15, 16, 13]]).as_matrix()
    sig = np.exp(a[:, 10:13])
    r = 3 * np.sqrt(((R * sig[:, None, :]) ** 2).sum(2))
    i, j = np.floor((a[:, :2] - volume.origin) / volume.step).astype(int).T
    i = i.clip(0, floor.shape[1] - 1)
    j = j.clip(0, floor.shape[0] - 1)
    factor = np.ones(len(a), "f")
    ok = np.zeros(len(a), bool)
    grids = {}
    for _ in range(7):
        ids = np.flatnonzero(~ok)
        b = np.ceil(r[ids, :2].max(1) * factor[ids] / volume.step).astype(int)
        bs = 2 ** np.ceil(np.log2(np.maximum(1, b))).astype(int)
        for k in np.unique(bs):
            ix = ids[bs == k]
            if k not in grids:
                grids[k] = maximum_filter(floor, size=2 * k + 1, mode="nearest")
            good = a[ix, 2] - r[ix, 2] * factor[ix] > grids[k][j[ix], i[ix]] + 0.001
            ok[ix[good]] = True
            factor[ix[~good]] *= 0.72
    a[:, 10:13] += np.log(factor[:, None])
    return a[ok]


def add_ground(dst, aa):
    meta = J(dst / "map.json")
    ij = np.floor(aa[:, :2] / 32).astype(int)
    key = ij[:, 0] * 65536 + ij[:, 1]
    ord = np.argsort(key)
    bounds = np.r_[0, 1 + np.flatnonzero(np.diff(key[ord])), len(ord)]
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        ids = ord[lo:hi]
        A = aa[ids]
        x, y = ij[ids[0]]
        name = f"verge_{x}_{y}"
        file = "tiles/" + name + ".ply"
        with (dst / file).open("wb") as f:
            f.write(binary_header(len(A)))
            A.astype("<f4").tofile(f)
        meta["tiles"].append(
            dict(
                id=name,
                file=file,
                layer="ground",
                kind="static",
                count=len(A),
                t_min=0,
                t_max=meta["t_end"],
                bbox=np.r_[A[:, :3].min(0), A[:, :3].max(0)].tolist(),
            )
        )
    meta["sky"] = "sky.jpg"
    savej(dst / "map.json", meta)
    shutil.copyfile(RS / "scene/live-v5/sky.jpg", dst / "sky.jpg")


def build(dst, rev):
    a, n = candidates()
    D = np.load(W / "observations.npz")
    s = D["stats"]
    di = D["dirs"]
    assert int(D["n"]) == n and len(s) == len(a)
    bits = np.unpackbits(di.view("B").reshape(-1, 4), axis=1).sum(1)
    vol = Volume()
    floor, hull, own = vol.lookup(a)
    h = a[:, 2] - floor
    lat = cKDTree(traj.T_world_ego(np.arange(START, END, 0.10))[:, :2, 3]).query(
        a[:, :2]
    )[0]
    alpha = 1 / (1 + np.exp(-a[:, 9]))
    prior_alpha = alpha.copy()
    rgb = np.clip(a[:, 6:9] * 0.28209479 + 0.5, 0, 1)
    obs = s[:, 7:10] / np.maximum(0.001, s[:, 6, None]) / 255
    best = s[:, 11:14] / 255
    ratio = s[:, 2] / np.maximum(0.4, s[:, 4] + s[:, 0])
    scatter = np.sqrt(
        np.maximum(0, s[:, 15] / np.maximum(0.001, s[:, 6]) - (255 * obs.mean(1)) ** 2)
    )
    support = (s[:, 2] > 1.3) & (ratio > 0.43) & (bits >= 2)
    solid = (s[:, 2] > 3.6) & (ratio > 0.52) & (s[:, 0] < 0.75 * s[:, 2]) & (bits >= 2)
    green = (obs[:, 1] > 1.035 * obs[:, 0]) & (obs[:, 1] > 0.965 * obs[:, 2])
    crown = (h > 1.35) & (h < 15) & green
    contradicted = (s[:, 0] > 2) & (s[:, 0] > s[:, 2] * 1.3)
    core = crown & solid & ~contradicted
    anchor = cKDTree(a[core, :3])
    gap = anchor.query(a[:, :3], distance_upper_bound=2.5)[0]
    near = crown & (gap < 0.45) & support
    strength = np.clip((ratio - 0.38) / 0.37, 0, 1)
    restore = (
        core | near
    )  # Real exposure with strongest local footprint preserves chroma; brightness cannot stray far from the multi-image mean.
    ref = 0.50 * obs + 0.50 * best
    ref = np.clip(ref, obs - 0.065, obs + 0.065)
    w = np.where(restore, 0.7, np.where(support, 0.43, 0.12)) * np.clip(s[:, 6], 0, 1)
    new = (
        rgb * (1 - w[:, None]) + ref * w[:, None]
    )  # Cull detached bright fringe rather than increasing all opacities. Preserve non-green trunks and deep-shadow cores.
    reject = (
        (h < 1) & (alpha < 0.19) & ~((h > 0.53) & solid & (bits >= 5) & (alpha > 0.13))
    )
    reject |= (h < 1.05) & own.astype(bool) & (alpha < 0.38) & (~solid | (h < 0.26))
    fringe = (
        (crown | ((h > 1) & (h < 16) & (lat < 45)))
        & (~support)
        & (alpha < 0.16)
        & ((gap > 0.58) | contradicted | (s[:, 0] > 2))
    )
    reject |= fringe
    alpha[restore] = np.maximum(
        alpha[restore], (0.43 + 0.34 * strength)[restore]
    )  # Tighten boundary splats even if their centre agrees; the released mass goes into the observed core rather than the sky.
    boundary = crown & (~core) & ((gap > 0.5) | (scatter > 20))
    a[boundary, 10:13] += np.log(0.74)
    loose = (h > 1) & (lat < 30) & (np.exp(a[:, 10:13]).max(1) > 0.38) & (~solid)
    a[loose, 10:13] = np.minimum(a[loose, 10:13], np.log(0.31))
    far = (
        (lat > 32)
        & (h > 0.8)
        & (h < 45)
        & (s[:, 2] > 1.0)
        & (ratio > 0.46)
        & (s[:, 0] < s[:, 2] * 0.8)
        & ((bits >= 2) | (scatter < 12))
    )
    alpha[far] = np.maximum(alpha[far], (0.49 + 0.31 * strength)[far])
    extra = np.arange(len(a)) >= n
    reject |= extra & ~far
    alpha = np.clip(alpha, 0.012, 0.94)
    a[:, 6:9] = (new - 0.5) / 0.28209479
    a[:, 9] = np.log(alpha / (1 - alpha))
    base = a[:n][~reject[:n]]
    other = [
        a[n:][~reject[n:]],
        np.load(W / "patches.npy"),
        np.load(W / "uprights.npy"),
        np.load(W / "walls.npy"),
    ]
    adds = support_bound(np.concatenate(other), vol)
    combined = np.concatenate([base, adds])
    scene(dst, combined)
    terrain = np.load(W / "grass.npy")
    add_ground(dst, terrain)
    res = dict(
        revision=rev,
        original=n,
        base_pruned=int(reject[:n].sum()),
        first_metre_pruned=int((reject[:n] & (h[:n] < 1)).sum()),
        crowns_strengthened=int((restore[:n] & (alpha[:n] > prior_alpha[:n])).sum()),
        crown_core_before_after=[
            float(np.median(x[core])) for x in [prior_alpha, alpha]
        ],
        new_far_candidates=int(len(other[0])),
        far_strengthened=int((far[:n] & (alpha[:n] > prior_alpha[:n])).sum()),
        patches=len(other[1]),
        upright_elements=len(other[2]),
        facade_elements=len(other[3]),
        support_guard_retained=len(adds),
        objects=len(combined),
        new_verge=len(terrain),
    )
    savej(W / (rev + ".json"), res)
    print(res, flush=True)


if __name__ == "__main__":
    import sys

    build(Path(sys.argv[1]), sys.argv[2])
