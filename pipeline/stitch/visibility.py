"""Remove contradicted object cores only; certify independent road/sky line-of-sight."""

from .views import *
from pipeline.road.scene import reader, binary_header
from scipy.spatial import cKDTree
import shutil, argparse


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-8)


def vote(a):
    tic = time.monotonic()
    xyz = a[:, :3].copy()
    tree = cKDTree(xyz)
    sigma = np.exp(a[:, 10:12]).max(1)
    v = np.zeros((len(a), 4), "i")
    last = np.zeros((len(a), 3), "f") + 1e6
    first = np.zeros((len(a), 3), "f")
    span = np.zeros(len(a), "f")
    labels = np.zeros(len(a), "B")
    es = sorted(
        [e for e in J(CACHE / "manifest.json") if e["yaw"] == 0 and not e["hold"]],
        key=lambda e: e["t"],
    )
    prev = {}
    cnt = 0
    for e in es:
        c, t, n = e["c"], e["t"], e["n"]
        C = rig[c]
        T = T_world_cam(C, t, None, traj)
        o = T[:3, 3]
        if c in prev and np.linalg.norm(o - prev[c]) < 1.4:
            continue
        prev[c] = o
        ii = np.asarray(tree.query_ball_point(o, 40), "i")
        wc = (xyz[ii] - o) @ T[:3, :3]
        ii = ii[
            (wc[:, 2] > 0.8)
            & (np.hypot(wc[:, 0], wc[:, 1]) < wc[:, 2] * (11 if c == "back" else 1.15))
        ]
        if not len(ii):
            continue
        uv, ok = C.project((xyz[ii] - o) @ T[:3, :3])
        ii = ii[ok]
        uv = uv[ok]
        b = np.clip(np.floor(uv[:, 1] * BANDS / C.height), 0, BANDS - 1).astype(int)
        Ts = T_world_cam(C, t, (np.arange(BANDS) + 0.5) * C.height / BANDS, traj)
        Ti = Ts[b]
        wc = np.einsum("nji,nj->ni", Ti[:, :3, :3], xyz[ii] - Ti[:, :3, 3])
        uv, ok = C.project(wc)
        rr = np.linalg.norm(wc, axis=1)
        ii, wc, uv, rr, b = ii[ok], wc[ok], uv[ok], rr[ok], b[ok]
        if not len(ii):
            continue
        fn = "%06d.png" % n
        sky, dyn, road = [
            cv.imread(str(RS / "masks" / k / c / fn), cv.IMREAD_UNCHANGED)
            for k in ("sky", "dyn", "road")
        ]
        h, w = sky.shape
        x = np.clip((uv[:, 0] * w / C.width).astype(int), 0, w - 1)
        y = np.clip((uv[:, 1] * h / C.height).astype(int), 0, h - 1)
        rad = np.clip(1.5 * sigma[ii] * C.params[0] * (w / C.width) / rr + 2, 3, 32)
        dynfree = (
            cv.distanceTransform((dyn == 0).astype("B"), cv.DIST_L2, 3)[y, x] > rad
        )
        sd = cv.distanceTransform((sky > 0).astype("B"), cv.DIST_L2, 3)[y, x]
        rd = cv.distanceTransform((road > 0).astype("B"), cv.DIST_L2, 3)[y, x]
        ego = traj.T_world_ego(t)
        ng, hg = traj.ground_plane(t)
        eh = ((xyz[ii] - ego[:3, 3]) @ ego[:3, :3]) @ ng + hg
        origin = ((Ts[b, :3, 3] - ego[:3, 3]) @ ego[:3, :3]) @ ng + hg
        den = (eh - origin) / rr
        road_range = np.where(den < -0.04, -origin / np.minimum(den, -0.04), 1e5)
        onroad = (
            (rd > rad)
            & (eh > 0.30 + sigma[ii] * 0.4)
            & (road_range > rr + np.maximum(0.45, rr * 0.05))
            & (road_range < 32)
        )
        onsky = (sd > rad) & (rr < 32)
        positive = np.zeros(len(ii), bool)
        deep = np.zeros(len(ii), bool)
        if e["depth"]:
            dep = np.load(e["depth"]).astype("f")
            z = dep[y, x]
            good = (z > 1) & (wc[:, 2] > 0.7)
            positive = (
                good
                & (abs(z - wc[:, 2]) < 0.06 * wc[:, 2] + 0.18)
                & (sd == 0)
                & ~onroad
            )  # strong, conservative independent positive veto
            zz = cv.erode(dep, np.ones((5, 5), "B"))[y, x]
            deep = (
                good
                & (zz > wc[:, 2] * 1.28 + 1.0)
                & (rr < 27)
                & (wc[:, 2] / rr > 0.72)
                & (sd == 0)
                & (rd == 0)
            )
        neg = (onsky | onroad | deep) & dynfree
        independent = np.linalg.norm(last[ii] - o, axis=1) > 1.35
        neg &= independent
        positive &= dynfree
        ni = ii[neg]
        fresh = v[ni, :3].sum(1) == 0
        first[ni[fresh]] = o
        span[ni] = np.maximum(span[ni], np.linalg.norm(first[ni] - o, axis=1))
        last[ni] = o
        labels[ni] |= 1 << CAMS.index(c)
        v[ii[neg & onsky], 0] += 1
        v[ii[neg & onroad], 1] += 1
        v[ii[neg & deep], 2] += 1
        v[ii[positive], 3] += 1
        cnt += 1
    neg = v[:, :3].sum(1)
    hard = (
        (v[:, 0] + v[:, 1] >= 3) & (neg >= 4) & (neg >= 2.5 * v[:, 3] + 2) & (span > 3)
    ) | (
        (v[:, 2] >= 5)
        & (neg >= 3 * v[:, 3] + 3)
        & (span > 5)
        & ((labels & (labels - 1)) != 0)
    )
    # Car actually occupied this conservative metric prism: pose-swept free space even where stereo is ambiguous.
    hull = np.zeros(len(a), bool)
    for T in traj.T_world_ego(np.arange(START, END, 0.06)):
        ids = np.asarray(tree.query_ball_point(T[:3, 3], 3.8), "i")
        q = (xyz[ids] - T[:3, 3]) @ T[:3, :3]
        hull[ids] |= (
            (q[:, 0] > -0.7)
            & (q[:, 0] < 3.3)
            & (abs(q[:, 1]) < 0.85)
            & (q[:, 2] > 0.22)
            & (q[:, 2] < 1.48)
        )
    return hard | hull, dict(
        eligible=len(a),
        views=cnt,
        sky_road_depth_pruned=int(hard.sum()),
        hull_pruned=int(hull.sum()),
        union=int((hard | hull).sum()),
        seconds=time.monotonic() - tic,
    )


def run(src, seed, dst):
    if dst.exists():
        raise FileExistsError(f"Use a new culled-scene directory: {dst}")
    meta = J(src / "map.json")
    refs = [(src, t) for t in meta["tiles"]]
    if seed:
        refs += [
            (seed, t)
            for t in J(seed / "map.json")["tiles"]
            if t["id"].startswith("stereo_")
        ]
    obj = [reader(p / t["file"]) for p, t in refs if t["layer"] == "objects"]
    remove, res = vote(np.concatenate(obj))
    stats = []
    i = 0
    (dst / "tiles").mkdir(parents=True, exist_ok=True)
    for p, t in refs:
        target = dst / t["file"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if t["layer"] == "ground":
            if not target.exists():
                os.link(p / t["file"], target)
            stats.append(t)
            continue
        a = reader(p / t["file"])
        z = a[~remove[i : i + len(a)]]
        i += len(a)
        if not len(z):
            continue
        with target.open("wb") as f:
            f.write(binary_header(len(z)))
            z.tofile(f)
        stats.append(
            dict(t, count=len(z), bbox=np.r_[z[:, :3].min(0), z[:, :3].max(0)].tolist())
        )
    assert i == len(remove)
    for fn in ("rig.json", "drive.json"):
        shutil.copy2(src / fn, dst / fn)
    savej(dst / "map.json", dict(meta, tiles=stats))
    savej(WORK / (dst.name + "-visibility.json"), res)
    print(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--seeds", type=Path)
    a = ap.parse_args()
    run(a.source, a.seeds, a.out)
