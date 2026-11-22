"""Dense one-ray births, local photographic range refinement, multi-resolution world voxels."""

from .v2data import *
from scipy.spatial.transform import Rotation
from concurrent.futures import ThreadPoolExecutor
import ctypes as ct, subprocess, time, shutil


def unit(a):
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-8)


@lru_cache(40)
def imask(c, n):
    sky, dyn, road = masks(c, n)
    _, ok = rays(c)
    return np.where(
        ok & (dyn == 0), np.where(sky != 0, 2, np.where(road != 0, 3, 1)), 0
    ).astype("B")


def make_lib():
    out = V / "match.dylib"
    src = Path(__file__).with_name("v2match.cpp")
    subprocess.run(
        ["c++", "-O3", "-std=c++17", "-dynamiclib", str(src), "-o", str(out)],
        check=True,
    )
    s = ct.CDLL(str(out)).sweep
    s.argtypes = [ct.c_void_p, ct.c_int] + [ct.c_void_p] * 5
    return s


def extract(c, es, volume, tree, sweep):
    use = [e for e in es if not e["hold"]]
    ds = np.array([e["odo"] for e in use])
    C = rig[c]
    K = np.ascontiguousarray(
        C.params * np.r_[[724 / C.width, 469 / C.height] * 2, [1] * 4], "d"
    )
    ray, ok = rays(c)
    samples = []
    hist = np.zeros(4, "int64")
    rank = []
    keys = []
    ts = []
    nc = 0
    change = []
    for j, e in enumerate(use):
        if not e["depth"]:
            continue
        n, t = e["n"], e["t"]
        rr, weight = physical_range(c, np.load(e["depth"]).astype("f"))
        cls = imask(c, n)
        rr = np.where(cls == 1, rr, 0)
        y, x = np.mgrid[3:466:4, 3:721:4]
        shift = j % 4
        x = (x + shift).clip(3, 720)
        y = (y + (j // 4) % 4).clip(3, 465)
        r = rr[y, x]
        ii = r > 0
        y, x, r = y[ii], x[ii], r[ii]
        if not len(r):
            continue
        rgb0 = rgb(c, n)
        raw = ray[y, x]
        T = poses(c, t)
        row = np.clip((y + 0.5) * 8 / 469, 0, 7).astype(int)
        mat = T[row]
        o = mat[:, :3, 3]
        u = np.einsum("nij,nj->ni", mat[:, :3, :3], raw)
        obs = np.ascontiguousarray(np.column_stack([x, y, o, u, r, weight[y, x]]), "f")
        base = 4.2 if c == "front" else 1.65
        ids = [
            max(0, min(len(use) - 1, np.searchsorted(ds, ds[j] - base))),
            max(0, min(len(use) - 1, np.searchsorted(ds, ds[j] + base))),
        ]
        qq = [use[i] for i in ids]
        rows = np.ascontiguousarray([poses(c, q["t"]) for q in qq], "d")
        ims = np.ascontiguousarray([rgb0] + [rgb(c, q["n"]) for q in qq], "B")
        ms = np.ascontiguousarray([imask(c, q["n"]) for q in qq], "B")
        fit = np.empty((len(x), 2), "f")
        sweep(
            obs.ctypes.data,
            len(obs),
            K.ctypes.data,
            rows.ctypes.data,
            ims.ctypes.data,
            ms.ctypes.data,
            fit.ctypes.data,
        )
        d = fit[:, 0]
        X = o + u * d[:, None]
        g, h, owned = volume.lookup(X)
        good = (X[:, 2] > np.maximum(h, g + np.where(owned, 0.16, 0.015))) & (
            X[:, 2] < g + 65
        )
        x, y, r, d, X, u, row, fit = [v[good] for v in (x, y, r, d, X, u, row, fit)]
        if not len(x):
            continue
        s = fit[:, 1]
        change.extend((abs(d - r) / np.maximum(1, r))[::200])
        hist += np.histogram(d, [0, 15, 60, 120, 211])[0]
        lat = tree.query(X[:, :2])[0]
        level = np.digitize(lat, [12, 29, 59, 105])
        pitch = np.array([0.095, 0.19, 0.41, 0.86, 1.65])[level]
        cell = np.floor(X / pitch[:, None]).astype("i8") + 16000
        key = ((level * 32768 + cell[:, 0]) * 32768 + cell[:, 1]) * 32768 + cell[:, 2]
        pix = rgb0[y, x].astype("f") / 255
        dx = ray[y, x + 1] * rr[y, x + 1, None] - ray[y, x - 1] * rr[y, x - 1, None]
        dy = ray[y + 1, x] * rr[y + 1, x, None] - ray[y - 1, x] * rr[y - 1, x, None]
        normal = unit(np.cross(dx, dy))
        nw = np.einsum("nij,nj->ni", T[row, :3, :3], normal)
        nw = np.where((nw * u).sum(1)[:, None] > 0, -nw, nw)
        sharp = (
            abs(rr[y, x + 1] + rr[y, x - 1] - 2 * r)
            + abs(rr[y + 1, x] + rr[y - 1, x] - 2 * r)
        ) < (0.006 * r + 0.035)
        planar = (
            sharp & (abs((nw * u).sum(1)) > 0.25) & (np.linalg.norm(nw, axis=1) > 0.98)
        )
        nw = np.where(planar[:, None], nw, -u)
        tangent = unit(
            np.cross(
                np.where((abs(nw[:, 2]) > 0.8)[:, None], [0, 1.0, 0], [0, 0, 1.0]), nw
            )
        )
        yv = unit(np.cross(nw, tangent))
        quat = Rotation.from_matrix(np.stack([tangent, yv, nw], 2)).as_quat()[
            :, [3, 0, 1, 2]
        ]
        sig = np.maximum(0.045, np.minimum(pitch * 0.90, 4 * d / K[0] * 0.96))
        sig = np.where(
            d > 60, np.maximum(sig, np.minimum(1.45, 4 * d / K[0] * 0.68)), sig
        )
        confidence = weight[y, x]
        alpha = np.clip(0.24 + 0.30 * s + 0.20 * confidence, 0.16, 0.78)
        a = np.zeros((len(x), 24), "f")
        a[:, :3] = X
        a[:, 3:6] = nw
        a[:, 6:9] = (pix - 0.5) / 0.28209479
        a[:, 9] = np.log(alpha / (1 - alpha))
        a[:, 10:13] = np.log(
            np.stack(
                [sig, sig, np.maximum(0.015, sig * np.where(planar, 0.21, 0.72))], 1
            )
        )
        a[:, 13:17] = quat
        a[:, 17] = 0
        a[:, 18:] = 1e9
        a[:, 19] = t
        samples.append(a)
        rank.append((s + confidence * 0.6 + 0.4 * np.exp(-d / 32)).astype("f"))
        keys.append(key)
        ts.append(np.full(len(a), t, "f"))
        nc += 1
    a = np.concatenate(samples)
    key = np.concatenate(keys)
    score = np.concatenate(rank)
    tm = np.concatenate(ts)
    order = np.lexsort((-score, key))
    first = order[
        np.r_[True, key[order][1:] != key[order][:-1]]
    ]  # preserve all timed evidence separately
    np.savez(
        V / (c + "-births.npz"),
        a=a[first],
        key=key[first],
        score=score[first],
        obskey=key,
        obst=tm,
    )
    return dict(
        frames=nc,
        samples=len(a),
        local_voxels=len(first),
        range_bands=hist.tolist(),
        stereo_fraction_adjust_p50_p90=np.percentile(change, [50, 90]).tolist(),
    )


def scene(dst, aa, groundtiles=None):
    dst.mkdir(parents=True, exist_ok=False)
    (dst / "tiles").mkdir()
    meta = J(GROUND / "map.json")
    tiles = []
    for t in groundtiles or tiles_ground():
        os.link(GROUND / t["file"], dst / t["file"])
        tiles.append(t)
    for fn in ["drive.json", "rig.json"]:
        shutil.copy2(GROUND / fn, dst / fn)
    ij = np.floor(aa[:, :2] / 32).astype("i")
    k = ij[:, 0].astype("int64") * 65536 + ij[:, 1]
    ord = np.argsort(k)
    st = np.r_[0, np.flatnonzero(np.diff(k[ord])) + 1, len(ord)]
    for i0, i1 in zip(st[:-1], st[1:]):
        ids = ord[i0:i1]
        a = aa[ids]
        tx, ty = ij[ids[0]]
        name = f"dense_{tx}_{ty}"
        fn = "tiles/" + name + ".ply"
        with (dst / fn).open("wb") as f:
            f.write(binary_header(len(a)))
            a.astype("<f4").tofile(f)
        tiles.append(
            dict(
                id=name,
                file=fn,
                count=len(a),
                kind="static",
                layer="objects",
                t_min=0,
                t_max=meta["t_end"],
                bbox=np.r_[a[:, :3].min(0), a[:, :3].max(0)].tolist(),
            )
        )
    sky = RS / "scene/live-v4/sky.jpg"
    if sky.is_file():
        shutil.copy2(sky, dst / "sky.jpg")
        meta["sky"] = "sky.jpg"
    savej(dst / "map.json", dict(meta, tiles=tiles, slice_s=[START, END]))


def main():
    tic = time.monotonic()
    vol = Volume()
    tree = cKDTree(traj.T_world_ego(np.arange(START, END, 0.10))[:, :2, 3])
    schedule0 = schedule()
    sweep = make_lib()
    stats = {}
    with ThreadPoolExecutor(3) as ex:
        for c, res in zip(
            CAMS, ex.map(lambda c: extract(c, schedule0[c], vol, tree, sweep), CAMS)
        ):
            stats[c] = res
            print(c, res, flush=True)
    fs = [np.load(V / (c + "-births.npz")) for c in CAMS]
    key = np.concatenate([a["key"] for a in fs])
    score = np.concatenate([a["score"] for a in fs])
    ord = np.lexsort((-score, key))
    take = ord[np.r_[True, key[ord][1:] != key[ord][:-1]]]
    a = np.concatenate(
        [z["a"] for z in fs]
    )[
        take
    ]  # union observed times in the same cell, retain first, first revisit and final interval
    obs = np.concatenate([z["obskey"] for z in fs])
    tm = np.concatenate([z["obst"] for z in fs])
    oo = np.lexsort((tm, obs))
    obs, tm = obs[oo], tm[oo]
    start = np.r_[True, obs[1:] != obs[:-1]]
    rid = np.cumsum(start) - 1
    runbreak = start | np.r_[False, np.diff(tm) > 0.25]
    b = np.flatnonzero(runbreak)
    rr = rid[b]
    beg = tm[b]
    end = tm[np.r_[b[1:] - 1, len(tm) - 1]]
    first = np.flatnonzero(np.r_[True, rr[1:] != rr[:-1]])
    last = np.r_[first[1:] - 1, len(rr) - 1]
    a[:, 19] = end[first]
    more = last > first
    a[more, 20] = beg[first[more] + 1]
    a[more, 21] = end[first[more] + 1]
    more = last > first + 1
    a[more, 22] = beg[last[more]]
    a[more, 23] = end[last[more]]
    scene(V / "dense-init", a)
    savej(
        V / "seed-stats.json",
        dict(cameras=stats, objects=len(a), seconds=time.monotonic() - tic),
    )
    print("dense", len(a), time.monotonic() - tic, flush=True)


if __name__ == "__main__":
    main()
