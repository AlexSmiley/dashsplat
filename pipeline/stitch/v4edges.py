"""Measured high-frequency road residual, voted in world coordinates. Only phase-consistent, depth-visible paint/texture is sharpened."""

from .v4teacher import *
from .v2eval import Wide


def measure(src):
    model = Wide(src)
    train, _ = split()
    rows = [t for t in J(src / "map.json")["tiles"] if t["layer"] == "ground"]
    tiles = [reader(src / t["file"]) for t in rows]
    stats = [np.zeros((len(a), 18), "f") for a in tiles]
    fn = ct.CDLL(str(OUT / "vote.dylib")).vote
    P = ct.c_void_p
    I = ct.c_int
    fn.argtypes = [P, I, P, P, P, I, I, P, P, P, P, ct.c_float, I, I]
    fn.restype = None
    b = np.array([t["bbox"] for t in rows])
    cent = (b[:, :3] + b[:, 3:]) * 0.5
    rad = np.linalg.norm(b[:, 3:] - cent, axis=1) + 1
    ed = OUT / "edges"
    ed.mkdir(exist_ok=True)
    es = [e for e in train if e["kind"] == "chase" and e["station"] % 3 != 2]

    def teacher(e):
        im, wt, sky, dep = decode(e)
        T = np.array(e["T_world_cam"], "d")
        K = np.array(e["K"], "d")
        H, W = wt.shape
        a = model.draw(
            e["t"], np.repeat(T[None], 33, 0), np.r_[K, np.zeros(4)], (W, H), 0
        )
        err = im.astype("f") - a[:, :, :3]
        conf = (wt > 94) & (a[:, :, 3] > 0.83) & (a[:, :, 5] < 0.045) & (dep > 150)
        den = cv.GaussianBlur(conf.astype("f"), (0, 0), 4)
        low = cv.GaussianBlur(err * conf[:, :, None], (0, 0), 4) / np.maximum(
            0.001, den[:, :, None]
        )
        res = np.ascontiguousarray(np.uint8(np.clip(128 + err - low, 0, 255)))
        w = np.where(conf & (den > 0.8), wt, 0).astype("B")
        v = (cent - T[:3, 3]) @ T[:3, :3]
        ids = np.flatnonzero(
            (v[:, 2] > -rad)
            & (v[:, 2] < 150 + rad)
            & (abs(v[:, 0]) < v[:, 2] * W / (2 * K[0]) + rad * 1.3)
        )
        return e, T, K, res, w, sky, dep, ids

    # Limit in-flight full-res images; maintain ordered, deterministic votes.
    with ThreadPoolExecutor(4) as ex:
        for j in range(0, len(es), 8):
            for e, T, K, res, w, sky, dep, ids in ex.map(teacher, es[j : j + 8]):
                base = 1 / (1 + (e["base_distance_m"] / 7.5) ** 2)
                H, W = w.shape
                for i in ids:
                    fn(
                        tiles[i].ctypes.data,
                        len(tiles[i]),
                        stats[i].ctypes.data,
                        T.ctypes.data,
                        K.ctypes.data,
                        W,
                        H,
                        res.ctypes.data,
                        w.ctypes.data,
                        sky.ctypes.data,
                        dep.ctypes.data,
                        base,
                        1,
                        0,
                    )
        n = 0
        for t, a, s in zip(rows, tiles, stats):
            wt = s[:, 0]
            avg = s[:, 1:4] / np.maximum(0.001, wt[:, None])
            var = np.sqrt(
                np.maximum(0, s[:, 4] / np.maximum(0.001, wt) - avg.mean(1) ** 2)
            )
            res = avg - 128
            best = (
                s[:, 6:9] - 128
            )  # Require agreeing signs on the luminance detail in near-best and weighted views.
            keep = (
                (s[:, 16] >= 2)
                & (wt > 0.32)
                & (abs(res.mean(1)) > 1.0)
                & (res.mean(1) * best.mean(1) > 0)
                & (var < np.maximum(7, abs(res.mean(1)) * 1.15))
            )
            delta = (
                np.clip(1.24 * res + 0.17 * (best - res), -29, 29)
                * np.clip(wt / (wt + 0.75), 0, 1)[:, None]
            )
            np.savez(ed / (t["id"] + ".npz"), p=a[keep, :3], d=delta[keep].astype("f"))
            n += int(keep.sum())
    print(dict(train_stations=len(es), resolved=n), flush=True)


def apply(src, dst):
    rows = []
    n = 0
    for t in J(src / "map.json")["tiles"]:
        a = reader(src / t["file"])
        if t["layer"] == "ground":
            a = a.copy()
            d = np.load(OUT / "edges" / (t["id"] + ".npz"))
            if len(d["p"]):
                dist, i = cKDTree(d["p"]).query(a[:, :3], distance_upper_bound=0.007)
                keep = dist < 0.007
                rgb = (0.5 + a[keep, 6:9] * C0) * 255 + d["d"][i[keep]]
                a[keep, 6:9] = (np.clip(rgb, 0, 255) / 255 - 0.5) / C0
                n += int(keep.sum())
        rows.append(a)
    write_scene(src, dst, rows)
    print(dict(sharpened=n), flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        measure(Path(sys.argv[1]))
    else:
        apply(Path(sys.argv[1]), Path(sys.argv[2]))
