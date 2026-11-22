"""Additional held-out road/paint correspondences under extreme viewpoint change.
No offline trajectory in association. Match independently rectified photographic
patches with a broad search, reverse check, uniqueness test, and temporal chain.
"""

from .motion import *
from pipeline.common.rig import _mv
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

cv.setNumThreads(1)
CELL = 0.055
HALF = 10.89
N = int(2 * HALF / CELL) + 1
HP = 14


def run():
    s, d, t, B, stop = inputs()
    cams = [load_rig(RS / "calib/rig.json")[n] for n in NAMES]
    live = Trajectory(OUT / "ego_causal.npz")
    kt = s["t"]
    ns = s["ns"]
    ix = s["ix"]
    ds = s["distance"]
    tc = s["tc"]
    poses = np.stack(
        [
            live.T_world_ego(c.row_time(tc[:, i], c.height / 2)) @ c.T_ego_cam
            for i, c in enumerate(cams)
        ],
        1,
    )
    ego = live.T_world_ego(kt)
    gx, gy = np.meshgrid(np.arange(N) * CELL - HALF, np.arange(N) * CELL - HALF)
    grid = np.stack((gx, gy, np.zeros_like(gx)), axis=-1)
    size = np.stack([[c.width, c.height] for c in cams])
    mask = [c.mask.astype("uint8") for c in cams]

    @lru_cache(maxsize=48)
    def image(k, i):
        return cv.imread(str(D / "frames" / NAMES[i] / f"{ns[k, i]:06d}.jpg"), 0)

    def raster(k, i, G):
        cam = cams[i]
        T = poses[k, i]  # Screen coarse samples before the full warp.
        delta = G[::25, ::25] - T[:3, 3]
        q = delta @ T[:3, :3]
        _, ok = cam.project(q)
        if ok.sum() < 4:
            return None
        q = (G - T[:3, 3]) @ T[:3, :3]
        uv, visible = cam.project(q)
        mp = (uv - 0.5).astype("f")
        im = cv.remap(image(k, i), mp, None, cv.INTER_LINEAR)
        m = (visible & (q[:, :, 2] > 0.1)).astype("uint8")
        im = np.where(m, im, 0).astype("uint8")
        im = cv.GaussianBlur(im, (3, 3), 0.7)
        return im, m, mp

    def peak(template, src, px, py, rad, ms):
        h, w = template.shape
        hh = h // 2
        hw = w // 2
        ax = max(0, int(px) - rad - hw)
        ay = max(0, int(py) - rad - hh)
        bx = min(N, int(px) + rad + hw + 1)
        by = min(N, int(py) + rad + hh + 1)
        roi = src[ay:by, ax:bx]
        if roi.shape[0] < h or roi.shape[1] < w:
            return None
        resp = cv.matchTemplate(roi, template, cv.TM_CCOEFF_NORMED)
        valid = ms[ay + hh : by - hh, ax + hw : bx - hw]
        resp[valid == 0] = -1
        _, val, _, loc = cv.minMaxLoc(resp)
        x, y = loc
        if (
            val < 0.67
            or x < 1
            or y < 1
            or x + 1 >= resp.shape[1]
            or y + 1 >= resp.shape[0]
        ):
            return None
        dx1, dx2, dy1, dy2 = (
            resp[y, x - 1],
            resp[y, x + 1],
            resp[y - 1, x],
            resp[y + 1, x],
        )
        xf = x + np.clip(
            0.5 * (dx1 - dx2) / min(dx1 + dx2 - 2 * val, -1e-6), -0.65, 0.65
        )
        yf = y + np.clip(
            0.5 * (dy1 - dy2) / min(dy1 + dy2 - 2 * val, -1e-6), -0.65, 0.65
        )
        resp[max(y - 5, 0) : y + 6, max(x - 5, 0) : x + 6] = -1
        second = float(resp.max())
        return (ax + hw + xf, ay + hh + yf, float(val), val - second)

    def pixel(mp, x, y):
        return cv.getRectSubPix(mp[:, :, 0], (1, 1), (float(x), float(y)))[
            0, 0
        ] + 0.5, cv.getRectSubPix(mp[:, :, 1], (1, 1), (float(x), float(y)))[0, 0] + 0.5

    def one(z):
        k, reg = z
        origin = ego[k, :3, 3]
        yaw = np.arctan2(ego[k, 1, 0], ego[k, 0, 0])
        G = (
            grid @ Rs(yaw).T + origin
        )  # locally horizontal road in world, purely a correspondence proposal
        near = [raster(k, i, G) for i in range(6)]
        seeds = []
        for ci, z0 in enumerate(near):
            if z0 is None:
                continue
            im, m, mp = z0
            T = poses[k, ci]
            rng = np.linalg.norm(G - T[:3, 3], axis=2)
            safe = cv.erode(m, np.ones((2 * HP + 3, 2 * HP + 3), "uint8"))
            safe &= (rng > 3.1) & (rng < 6.8)
            safe[: HP + 2] = 0
            safe[-HP - 2 :] = 0
            safe[:, : HP + 2] = 0
            safe[:, -HP - 2 :] = 0
            corn = cv.goodFeaturesToTrack(
                im, 16, 0.025, 19, mask=safe.astype("uint8"), blockSize=9
            )
            if corn is None:
                continue
            for fx, fy in corn.reshape(-1, 2):
                px, py = int(fx), int(fy)
                tpl = im[py - HP : py + HP + 1, px - HP : px + HP + 1].copy()
                if np.std(tpl) < 6:
                    continue
                uv = pixel(mp, px, py)
                seeds.append(
                    dict(
                        ci=ci,
                        x=px,
                        y=py,
                        tpl=tpl,
                        obs=[[k, ci, *uv, 1.0, 0.0, 0.0]],
                        last=[px, py],
                        near=im,
                        nearok=cv.erode(m, np.ones((2 * HP + 1, 2 * HP + 1), "uint8")),
                        globalxy=G[py, px, :2].tolist(),
                    )
                )
        sequence = np.unique(
            np.clip(
                np.searchsorted(
                    ds,
                    ds[k]
                    - np.array(
                        [
                            0.5,
                            1.15,
                            2.1,
                            3.4,
                            5,
                            7,
                            9,
                            12,
                            15,
                            18,
                            21,
                            24,
                            27,
                            30,
                            33,
                            36,
                            39,
                            42,
                            45,
                            48,
                            52,
                            56,
                        ]
                    ),
                ),
                0,
                k - 1,
            )
        )[::-1]
        for j in sequence:
            if kt[k] - kt[j] > 23 or ds[k] - ds[j] < 0.2:
                continue
            if not seeds:
                break
            ras = [raster(j, i, G) for i in range(6)]
            valid = []
            for imz in ras:
                valid.append(
                    cv.erode(imz[1], np.ones((2 * HP + 1, 2 * HP + 1), "uint8"))
                    if imz
                    else None
                )
            for seed in seeds:
                lx, ly = seed["last"]
                sx, sy = seed["x"], seed["y"]
                dd = ds[k] - ds[j]
                best = None
                for ci, z0 in enumerate(ras):
                    if z0 is None:
                        continue
                    im, m, mp = z0
                    for scale in [1.0] if dd < 9 else [0.80, 1.0, 1.22]:
                        hh = round(HP * scale)
                        tpl = cv.resize(
                            seed["tpl"],
                            (2 * hh + 1, 2 * hh + 1),
                            interpolation=cv.INTER_AREA,
                        )
                        pk = peak(tpl, im, lx, ly, 23 if dd < 12 else 43, valid[ci])
                        if pk is None:
                            continue
                        x, y, val, margin = pk
                        if val < (0.76 if dd < 14 else 0.70) or margin < 0.045:
                            continue
                        # Reverse photographic association. Do not compare to an offline prediction.
                        sub = cv.getRectSubPix(
                            im, (2 * hh + 1, 2 * hh + 1), (float(x), float(y))
                        )
                        sub = cv.resize(sub, (2 * HP + 1, 2 * HP + 1))
                        rev = peak(sub, seed["near"], sx, sy, 19, seed["nearok"])
                        if rev is None or np.hypot(rev[0] - sx, rev[1] - sy) > 2.4:
                            continue
                        quality = val + 0.18 * min(margin, 0.3)
                        if best is None or quality > best[0]:
                            best = (quality, ci, x, y, val, mp)
                if best:
                    _, ci, x, y, val, mp = best
                    uv = pixel(mp, x, y)
                    seed["obs"].append(
                        [int(j), ci, *uv, val, float(x - sx), float(y - sy)]
                    )
                    seed["last"] = [x, y]
        ret = []
        for q in seeds:
            obs = sorted(q["obs"])
            ks = [v[0] for v in obs]
            dist = ds[ks]
            if len(obs) < 8 or dist[-1] - dist[0] < 26 or np.max(np.diff(dist)) > 11:
                continue
            ret.append((reg, q["ci"], q["globalxy"], obs))
        return ret

    tasks = []
    for reg, a, b in [
        ("car park", 88, 105),
        ("90° turn", 317.5, 325.5),
        ("exit turn", 105, 111.5),
        ("fast straight", 293, 310),
    ]:
        kk = np.flatnonzero((kt >= a) & (kt <= b))
        last = -1e5
        for k in kk:
            if ds[k] - last > 3.6:
                tasks.append((int(k), reg))
                last = ds[k]
    print("road validation target anchors", len(tasks), flush=True)
    rows = []
    seen = set()
    for result in ThreadPoolExecutor(10).map(one, tasks):
        for reg, ci, xy, obs in result:
            tag = (reg, ci, *np.rint(np.array(xy) / 0.65).astype(int))
            if tag in seen:
                continue
            seen.add(tag)
            rows.append(dict(region=reg, near_cam=int(ci), xy=xy, obs=obs))
    (W / "road-tracks.partial").write_text(json.dumps(rows, default=lambda v: v.item()))
    os.replace(W / "road-tracks.partial", W / "road-tracks.json")
    print("road appearance tracks", len(rows), flush=True)


if __name__ == "__main__":
    run()
