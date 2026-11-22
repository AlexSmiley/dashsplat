"""Initialise optical occupancy by integrating free-space negatives. Unseen seeds survive."""

from .v2seed import *


def clear(src, dst):
    tic = time.monotonic()
    meta = J(src / "map.json")
    a = np.concatenate(
        [reader(src / t["file"]) for t in meta["tiles"] if t["layer"] == "objects"]
    )
    xx = np.ascontiguousarray(np.c_[a[:, :3], np.exp(a[:, 10:12]).max(1)], "f")
    key = np.floor(xx[:, :2] / 15).astype(int)
    code = key[:, 0] * 65536 + key[:, 1]
    order = np.argsort(code)
    b = np.r_[0, np.flatnonzero(np.diff(code[order])) + 1, len(code)]
    cent = (key[order[b[:-1]]] + 0.5) * 15
    bins = [order[i:j].astype("i") for i, j in zip(b[:-1], b[1:])]
    tree = cKDTree(cent)
    stats = np.zeros((len(a), 4), "f")
    lib = make_lib()
    vote = ct.CDLL(str(V / "match.dylib")).votes
    vote.argtypes = (
        [ct.c_void_p, ct.c_void_p, ct.c_int] + [ct.c_void_p] * 6 + [ct.c_int]
    )
    nviews = 0
    for lens, c in enumerate(CAMS):
        last = -10
        C = rig[c]
        K = np.ascontiguousarray(
            C.params * np.r_[[724 / C.width, 469 / C.height] * 2, [1] * 4], "d"
        )
        ray, ok = rays(c)
        for e in schedule()[c]:
            if e["hold"] or e["odo"] - last < 1.75:
                continue
            last = e["odo"]
            T = poses(c, e["t"])
            reach = 145 if c == "front" else 68
            which = tree.query_ball_point(T[4, :2, 3], reach + 12)
            v = (cent[which] - T[4, :2, 3]) @ T[4, :2, 2]
            which = np.asarray(which)[v > -15]
            ids = (
                np.concatenate([bins[i] for i in which])
                if len(which)
                else np.empty(0, "i")
            )
            lab = imask(c, e["n"])
            r, wt = (
                physical_range(c, np.load(e["depth"]).astype("f"))
                if e["depth"]
                else (np.zeros(lab.shape, "f"), np.zeros(lab.shape, "f"))
            )
            ego = traj.T_world_ego(e["t"])
            ng, h = traj.ground_plane(e["t"])
            normal = T[4, :3, :3].T @ ego[:3, :3] @ ng
            den = ray @ normal
            camh = (T[4, :3, 3] - ego[:3, 3]) @ (ego[:3, :3] @ ng) + h
            rg = -camh / np.minimum(-0.01, den)
            ro = (lab == 3) & (den < -0.05) & (rg < 28)
            r[ro] = rg[ro]
            wt[ro] = 1
            vote(
                xx.ctypes.data,
                ids.ctypes.data,
                len(ids),
                T.ctypes.data,
                K.ctypes.data,
                lab.ctypes.data,
                r.ctypes.data,
                wt.ctypes.data,
                stats.ctypes.data,
                lens,
            )
            nviews += 1
    neg, pos, d, sky = stats.T
    delta = np.clip(np.log1p(pos) * 0.36 - np.log1p(neg) * 0.95, -3.7, 0.85)
    a[:, 9] += delta
    reject = (neg >= 4) & (neg > 2.2 * pos + 1.5) & ((sky >= 2) | (d >= 5))
    scene(dst, a[~reject])
    res = dict(
        input=len(a),
        pruned=int(reject.sum()),
        surviving_no_positive=int(((pos == 0) & ~reject).sum()),
        views=nviews,
        seconds=time.monotonic() - tic,
    )
    savej(V / "clear.json", res)
    print(res)


if __name__ == "__main__":
    clear(V / "dense-init", V / "dense-prior")
