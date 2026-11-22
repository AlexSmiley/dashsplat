"""Fit a single world-space DC colour to full-resolution, depth/sky-weighted chase teachers. Tangent subdivision only."""

from .v2data import *
from .v2seed import Rotation, shutil, ct, subprocess
import sys, time
from concurrent.futures import ThreadPoolExecutor

OUT = Path(os.getenv("STITCH_V4_WORK", RS / "work/stitch-v4"))
OUT.mkdir(exist_ok=True, parents=True)
SRC = RS / "scene/stitched-opt-v3"
MAN = Path(os.getenv("STITCH_TEACHER", RS / "work/ibr/teacher-85-140/manifest.json"))
C0 = 0.28209479


def split():
    es = J(MAN)
    train = [e for e in es if e["station"] % 10]
    held = [e for e in es if e["station"] % 10 == 0]
    assert not {v["station"] for v in train} & {v["station"] for v in held}
    savej(
        OUT / "split.json",
        dict(
            train=[e["id"] for e in train],
            held=[e["id"] for e in held],
            rule="station modulo 10 == 0 held, including all offsets",
        ),
    )
    return train, held


def decode(e):
    im = cv.imread(e["rgb"])[:, :, ::-1].copy()
    w, sk, z = [
        cv.imread(e[k], cv.IMREAD_UNCHANGED) for k in ["weight", "sky", "depth"]
    ]
    assert z.dtype == np.uint16 and w.ndim == 2 and w.shape == im.shape[:2]
    return im, w, sk, z


def write_scene(src, dst, tiles):
    dst.mkdir(parents=True, exist_ok=False)
    (dst / "tiles").mkdir()
    meta = J(src / "map.json")
    recs = []
    for rec, data in zip(meta["tiles"], tiles):
        if not len(data):
            continue
        a = np.ascontiguousarray(data, "<f4")
        f = dst / rec["file"]
        with f.open("wb") as fp:
            fp.write(binary_header(len(a)))
            a.tofile(fp)
        recs.append(
            dict(
                rec, count=len(a), bbox=np.r_[a[:, :3].min(0), a[:, :3].max(0)].tolist()
            )
        )
    for name in ["sky.jpg", "drive.json", "rig.json"]:
        shutil.copy2(src / name, dst / name)
    savej(dst / "map.json", dict(meta, tiles=recs))


def tally(src, train):
    meta = J(src / "map.json")
    tiles = [reader(src / t["file"]) for t in meta["tiles"]]
    stats = [np.zeros((len(a), 18), "f") for a in tiles]
    b = np.array([t["bbox"] for t in meta["tiles"]])
    cent = (b[:, :3] + b[:, 3:]) * 0.5
    rad = np.linalg.norm(b[:, 3:] - cent, axis=1) + 1
    ground = [t["layer"] == "ground" for t in meta["tiles"]]
    lib = OUT / "vote.dylib"
    subprocess.run(
        [
            "c++",
            "-O3",
            "-std=c++17",
            "-dynamiclib",
            str(Path(__file__).with_name("v4vote.cpp")),
            "-o",
            str(lib),
        ],
        check=True,
    )
    fn = ct.CDLL(str(lib)).vote
    P = ct.c_void_p
    I = ct.c_int
    fn.argtypes = [P, I, P, P, P, I, I, P, P, P, P, ct.c_float, I, I]
    fn.restype = None
    tic = time.monotonic()
    with ThreadPoolExecutor(6) as ex:
        for j, e in enumerate(train):
            T = np.array(e["T_world_cam"], "d")
            K = np.array(e["K"], "d")
            im, wt, sky, d = decode(e)
            H, W = wt.shape
            v = (cent - T[:3, 3]) @ T[:3, :3]
            front = v[:, 2]
            ids = np.flatnonzero(
                (front > -rad)
                & (front < 214 + rad)
                & (abs(v[:, 0]) < front * W / (2 * K[0]) + rad * 1.3)
                & (abs(v[:, 1]) < front * H / (2 * K[1]) + rad * 1.2)
            )
            base = 1 / (1 + (e["base_distance_m"] / 7.5) ** 2)

            def one(i):
                fn(
                    tiles[i].ctypes.data,
                    len(tiles[i]),
                    stats[i].ctypes.data,
                    T.ctypes.data,
                    K.ctypes.data,
                    W,
                    H,
                    im.ctypes.data,
                    wt.ctypes.data,
                    sky.ctypes.data,
                    d.ctypes.data,
                    base,
                    ground[i],
                    e["kind"] != "chase",
                )

            list(ex.map(one, ids))
            if j % 300 == 0:
                print(
                    "teacher",
                    j,
                    len(train),
                    round(time.monotonic() - tic, 1),
                    flush=True,
                )
    return tiles, stats, ground


def fit(src, dst, capacity):
    train, held = split()
    tiles, ss, gg = tally(src, train)
    out = []
    counts = np.zeros(6, int)
    vv = Volume()
    for A, s, g in zip(tiles, ss, gg):
        a = A.copy()
        wt = s[:, 0]
        mean = s[:, 1:4] / np.maximum(0.001, wt[:, None])
        lum = mean.mean(1)
        scatter = np.sqrt(np.maximum(0, s[:, 4] / np.maximum(0.001, wt) - lum**2))
        best = s[:, 6:9]
        c = (a[:, 6:9] * C0 + 0.5) * 255
        grad = s[:, 11] / np.maximum(0.001, wt)
        reliable = (s[:, 16] >= 2) & (wt > 0.25)
        stable = np.clip(1 - (scatter - 5) / 26, 0, 1)
        frac = np.clip(wt / (wt + (0.62 if g else 2.1)), 0, 0.96 if g else 0.8) * stable
        target = 0.35 * mean + 0.65 * best
        target = np.clip(
            target,
            mean - np.maximum(9, scatter[:, None] * 1.15),
            mean + np.maximum(9, scatter[:, None] * 1.15),
        )
        a[:, 6:9] = ((c + frac[:, None] * (target - c)) / 255 - 0.5) / C0
        counts[0 if g else 1] += int((frac > 0.20).sum())
        kill = np.zeros(len(a), bool)
        if not g:
            clear = s[:, 9] + s[:, 10] * 0.65
            kill = (clear > 2.8) & (clear > 4 * wt + 1.5) & (a[:, 9] < 1.5)
            a[:, 9] -= np.clip((clear - 1.2 * wt) / 3, 0, 1) * 0.9
            mv = (
                np.clip(s[:, 13:16] / np.maximum(0.001, wt[:, None]), -0.12, 0.12)
                * (0.18 * stable * reliable)[:, None]
            )  # Real cameras remain geometry anchors. Keep guarded base/trunk contact and do not move thin upright fits.
            margin = a[:, 2] - vv.floor(a[:, :3])
            sel = (
                (margin > 1.6)
                & (np.exp(a[:, 10:13]).min(1) > 0.055)
                & (wt > 1.8)
                & (scatter < 14)
            )
            a[sel, :3] += mv[sel]
            counts[2] += int(kill.sum())
            counts[3] += int((np.linalg.norm(mv[sel], axis=1) > 0.002).sum())
        if g and capacity:
            sig = np.exp(a[:, 10:12])
            sel = (
                reliable
                & (sig.min(1) > 0.032)
                & (grad > (3.2 if capacity == 1 else 5.8))
                & (stable > 0.4)
                & (wt > 0.65)
            )
            ids = np.flatnonzero(sel)
            B = np.repeat(a[ids], 4, axis=0)
            q = (
                Rotation.from_quat(a[ids][:, [14, 15, 16, 13]]).as_matrix()
                if len(ids)
                else np.empty((0, 3, 3))
            )
            offset = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]]) * 0.58
            B[:, :3] += (
                np.einsum("nij,nkj->nki", q[:, :, :2], sig[ids, None, :] * offset[None])
            ).reshape(-1, 3)
            B[:, 10:12] += np.log(
                0.69
            )  # Same tangent plane and normal, equal aggregate optical thickness per footprint; parent is replaced, not overlaid.
            alpha = np.clip(1 / (1 + np.exp(-a[ids, 9])), 0.0001, 0.99998)
            new = 1 - (1 - alpha) ** (1 / (4 * 0.69**2))
            B[:, 9] = np.repeat(np.log(new / (1 - new)), 4)
            a = np.concatenate([a[~sel], B])
            counts[4] += len(ids)
        else:
            a = a[~kill]
        out.append(a)
        counts[5] += len(a)
    write_scene(src, dst, out)
    res = dict(
        train_views=len(train),
        held_views=len(held),
        ground_updated=int(counts[0]),
        objects_updated=int(counts[1]),
        conflicted_pruned=int(counts[2]),
        depth_adjusted=int(counts[3]),
        parents_split=int(counts[4]),
        final=int(counts[5]),
    )
    savej(OUT / (dst.name + "-teacher.json"), res)
    print(res, flush=True)


if __name__ == "__main__":
    fit(Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]))
