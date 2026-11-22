"""Preserve fitted far facades while applying stronger silhouette cleanup near the orbit."""

from .v2guard import *


def readobjects(p):
    return np.concatenate(
        [
            reader(p / t["file"])
            for t in J(p / "map.json")["tiles"]
            if t["layer"] == "objects"
        ]
    )


def mix(src, post, dst):
    tic = time.monotonic()
    a = readobjects(src)
    b = readobjects(post)
    tree = cKDTree(a[:, :3])
    d, i = tree.query(b[:, :3])
    assert d.max() < 1e-5 and len(np.unique(i)) == len(i)
    have = np.zeros(len(a), "B")
    have[i] = 1
    road = cKDTree(traj.T_world_ego(np.arange(START, END, 0.1))[:, :2, 3])
    dist = road.query(a[:, :2])[0]
    z = np.clip((dist - 25) / 40, 0, 1)
    w = (
        0.89 - 0.66 * z * z * (3 - 2 * z)
    )  # optical thickness blends linearly. A discarded far-field seed keeps part of its trained transmittance.
    orig = np.logaddexp(a[:, 9], 0)
    tau = orig * (1 - w)
    tau[i] += w[i] * np.logaddexp(b[:, 9], 0)
    a[:, 9] = np.log(np.expm1(np.maximum(0.0001, tau)))
    a[i, 6:9] = a[i, 6:9] * (1 - w[i, None]) + b[:, 6:9] * w[i, None]
    a[i, 10:13] = (
        a[i, 10:13] * (1 - w[i, None]) + b[:, 10:13] * w[i, None]
    )  # Actual evidence of sky conflict keeps the narrower axis, including at a far roofline.
    severe = b[:, 10:13] < a[i, 10:13] - 0.62
    a[i, 10:13] = np.where(severe, b[:, 10:13], a[i, 10:13])
    keep = (tau > 0.034) & ((have == 1) | (dist >= 25))
    scene(dst, a[keep])
    res = dict(
        before=len(a),
        after=int(keep.sum()),
        retained_far_only=int(((have == 0) & keep).sum()),
        near_posterior_weight=0.89,
        far_posterior_weight=0.23,
        seconds=time.monotonic() - tic,
    )
    savej(V / (dst.name + "-select.json"), res)
    print(res)


if __name__ == "__main__":
    import sys

    mix(*map(Path, sys.argv[1:]))
