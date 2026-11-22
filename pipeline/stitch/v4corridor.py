"""Budget native tangent-plane children by resolved intra-footprint contrast. Anchor disputed paint to one map."""

from .v4teacher import *
from scipy.ndimage import gaussian_filter, maximum_filter


def compact(src, dst):
    route = cKDTree(traj.T_world_ego(np.arange(START, END, 0.10))[:, :2, 3])
    rows = []
    cnt = np.zeros(3, int)
    base = {t["id"]: t for t in J(SRC / "map.json")["tiles"]}
    for t in J(src / "map.json")["tiles"]:
        a = reader(src / t["file"])
        if t["layer"] != "ground":
            rows.append(a)
            continue
        ref = reader(SRC / base[t["id"]]["file"])
        n = len(ref)
        k = (len(a) - n) // 3
        assert len(a) == n + 3 * k
        head = n - k
        C = a[head:].reshape(k, 4, 24)
        tree = cKDTree(ref[:, :3])
        i = tree.query(a[:head, :3], distance_upper_bound=0.0001)[1]
        assert len(set(i.tolist())) == len(i) and (i < n).all()
        ii = np.flatnonzero(~np.isin(np.arange(n), i))
        assert np.allclose(ref[ii, :3], C[:, :, :3].mean(1), atol=0.0001)
        colour = lambda b: (b[..., 6:9] * C0 + 0.5) * 255
        old = colour(ref)
        ave = np.empty_like(old)
        ave[i] = colour(a[:head])
        ave[ii] = colour(C).mean(1)
        lat = route.query(ref[:, :2])[0]
        grid = 0.11
        idx = np.floor(ref[:, :2] / grid).astype(int)
        idx -= idx.min(0)
        xx, yy = idx.T
        W, H = idx.max(0) + 1
        key = yy * W + xx
        den = np.bincount(key, minlength=W * H).reshape(H, W)
        density = gaussian_filter(den.astype("f"), 4.3)
        grey = lambda z: z @ np.array([0.32, 0.55, 0.13], "f")

        def features(rgb):
            l = grey(rgb)
            bg = gaussian_filter(
                np.bincount(key, weights=l, minlength=W * H).reshape(H, W), 4.3
            )[yy, xx] / np.maximum(0.001, density[yy, xx])
            return l - bg

        oc, tc = features(old), features(ave)
        paint = (oc > 14) & (oc > 1.6 * tc + 6) & (old.max(1) - old.min(1) < 96)
        teacher_edges = np.zeros((H, W), "f")
        np.maximum.at(
            teacher_edges, (yy, xx), tc.astype("f")
        )  # Do not restore a second line if the teachers already see one nearby at a different location.
        restore = (
            paint
            & (maximum_filter(teacher_edges, 9)[yy, xx] < np.maximum(16, oc * 0.47))
            & (lat < 15)
        )
        orig = np.where((lat < 6)[:, None], 0.16, 0.26)
        target = ave + orig * (old - ave)
        target[restore] = old[restore]
        delta = target - ave
        edges = np.ptp(colour(C), axis=1).max(1)
        keep = (edges > (6.2 + np.clip(lat[ii] - 5, 0, 22) * 0.37)) & (~restore[ii])
        B = ref.copy()
        B[:, 6:9] = (np.clip(target, 0, 255) / 255 - 0.5) / C0
        D = C[keep].copy()
        cc = colour(D)
        m = cc.mean(
            1
        )  # Mild contrast recovery within four supported children. No new frequencies on unsupported surfaces.
        D[:, :, 6:9] = (
            np.clip(
                m[:, None] + 1.12 * (cc - m[:, None]) + delta[ii[keep], None], 0, 255
            )
            / 255
            - 0.5
        ) / C0
        sel = np.ones(n, bool)
        sel[ii[keep]] = False
        rows.append(np.concatenate([B[sel], D.reshape(-1, 24)]))
        cnt += [k, int(keep.sum()), int(restore.sum())]
    write_scene(src, dst, rows)
    res = dict(
        original_subdivided=int(cnt[0]),
        retained=int(cnt[1]),
        distinct_original_paint=int(cnt[2]),
    )
    savej(OUT / (dst.name + "-budget.json"), res)
    print(res, flush=True)


if __name__ == "__main__":
    compact(Path(sys.argv[1]), Path(sys.argv[2]))
