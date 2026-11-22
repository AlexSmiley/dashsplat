"""Conservative 3-sigma Gaussian support guard against the swept hull and local road field."""

from .v2seed import *
from scipy.ndimage import maximum_filter


def guard(src, dst):
    tic = time.monotonic()
    v = Volume()
    floor = np.maximum(v.h, v.g + np.where(v.owned, 0.085, 0.015))
    a = np.concatenate(
        [
            reader(src / t["file"])
            for t in J(src / "map.json")["tiles"]
            if t["layer"] == "objects"
        ]
    )
    rot = Rotation.from_quat(a[:, [14, 15, 16, 13]]).as_matrix()
    sig = np.exp(a[:, 10:13])
    rad = 3 * np.sqrt(np.square(rot * sig[:, None, :]).sum(2))
    ij = np.floor((a[:, :2] - v.origin) / v.step).astype(int)
    i, j = ij.T
    dims = np.array(floor.shape[::-1])
    i = np.clip(i, 0, dims[0] - 1)
    j = np.clip(j, 0, dims[1] - 1)
    b = np.ceil(rad[:, :2].max(1) / v.step).astype(int)
    maxb = int(max(b.max(), 1))
    factor = np.ones(len(a), "f")
    saved = np.zeros(len(a), bool)
    reject = np.zeros_like(saved)
    bounds = np.unique(np.r_[0, 1, 2, 3, 4, 6, 8, 12, 17, 24, 32, 48, 64, maxb])
    grids = {
        int(k): maximum_filter(floor, size=2 * int(k) + 1, mode="nearest")
        for k in bounds
        if k <= maxb or k == bounds[np.searchsorted(bounds, maxb)]
    }
    for _ in range(10):
        m = ~saved & ~reject
        ii = np.flatnonzero(m)
        b = np.ceil(rad[ii, :2].max(1) * factor[ii] / v.step).astype(int)
        bins = bounds[np.searchsorted(bounds, b)]
        height = np.empty(len(ii))
        for k in np.unique(bins):
            s = bins == k
            height[s] = grids[k][j[ii[s]], i[ii[s]]]
        ok = a[ii, 2] - rad[ii, 2] * factor[ii] > height + 0.001
        saved[ii[ok]] = True
        fail = ii[~ok]
        factor[fail] *= 0.70
        reject[fail] |= factor[fail] < 0.08
    reject |= ~saved
    a[:, 10:13] += np.log(factor)[:, None]
    scene(dst, a[~reject])
    res = dict(
        objects_before=len(a),
        rejected=int(reject.sum()),
        shrunk=int(((factor < 1) & ~reject).sum()),
        objects_after=int((~reject).sum()),
        seconds=time.monotonic() - tic,
        support_sigmas=3.0,
        floor_grid_m=v.step,
        center_inside=int((a[~reject, 2] <= v.floor(a[~reject, :3])).sum()),
    )
    assert res["center_inside"] == 0
    savej(V / (dst.name + "-guard.json"), res)
    print(res, flush=True)


if __name__ == "__main__":
    import sys

    guard(Path(sys.argv[1]), Path(sys.argv[2]))
