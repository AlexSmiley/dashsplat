"""Optional last paint experiment: resolve foreground/background colour at trusted ground edges instead of averaging. Never add a marking."""

from .v4teacher import *
from scipy.ndimage import gaussian_filter, maximum_filter


def run(src, dst):
    rows = []
    rt = cKDTree(traj.T_world_ego(np.arange(START, END, 0.1))[:, :2, 3])
    n = 0
    L = np.array([0.30, 0.59, 0.11], "f")
    pre = OUT / "teacher-fit"
    for t in J(src / "map.json")["tiles"]:
        a = reader(src / t["file"])
        if t["layer"] != "ground" or t["id"].startswith("verge_"):
            rows.append(a)
            continue
        b = reader(pre / t["file"])
        old = reader(SRC / t["file"])
        pos = a[:, :3]
        d0, i0 = cKDTree(b[:, :3]).query(pos)
        d1, i1 = cKDTree(old[:, :3]).query(pos)
        idx = np.floor(pos[:, :2] / 0.07).astype(int)
        idx -= idx.min(0)
        x, y = idx.T
        w, h = idx.max(0) + 1
        code = y * w + x
        den = np.bincount(code, minlength=w * h).reshape(h, w).astype("f")
        D = gaussian_filter(den, 5.5)[y, x].clip(0.001)
        col = (a[:, 6:9] * C0 + 0.5) * 255
        tc = (b[i0, 6:9] * C0 + 0.5) * 255
        nc = (old[i1, 6:9] * C0 + 0.5) * 255
        lum = col @ L

        def blur(z):
            return (
                gaussian_filter(
                    np.bincount(code, weights=z, minlength=w * h).reshape(h, w), 5.5
                )[y, x]
                / D
            )

        q = lum - blur(lum)
        qt = tc @ L - blur(tc @ L)
        qo = nc @ L - blur(nc @ L)
        v = np.zeros((h, w), "f")
        np.maximum.at(v, (y, x), qt.astype("f"))
        p = maximum_filter(v, 7)[y, x]
        lat = rt.query(pos[:, :2])[0]
        ok = (
            (lat < 10)
            & (d0 < 0.10)
            & (p > 17)
            & (qt * q > 0)
            & ((qo * q > 0) | (abs(qt) > 14))
            & (abs(q) > 2.5)
        )  # Only amplify an existing edge which the nearest teacher-footprint colour agrees on; damp if native and chase disagree.
        boost = np.clip((p - 16) / 12, 0, 1) * np.where(qo * q > 0, 1, 0.5)
        hi = np.clip(0.85 * q, -15, 25) * boost
        A = a.copy()
        A[ok, 6:9] = (np.clip(col[ok] + hi[ok, None], 0, 255) / 255 - 0.5) / C0
        rows.append(A)
        n += int(ok.sum())
    write_scene(src, dst, rows)
    print("paint-candidate", n, flush=True)


if __name__ == "__main__":
    run(Path(sys.argv[1]), Path(sys.argv[2]))
