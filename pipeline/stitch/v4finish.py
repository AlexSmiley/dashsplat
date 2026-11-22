"""Remove crop-scale colour drift on road without reducing supported paint contrast. Base chroma comes from the dense pre-crop teacher fit and native mosaic."""

from .v4teacher import *
from scipy.ndimage import gaussian_filter


def run(src, pre, dst):
    route = cKDTree(traj.T_world_ego(np.arange(START, END, 0.1))[:, :2, 3])
    rows = []
    affected = 0
    dc = []
    W = np.array([0.30, 0.59, 0.11], "f")
    for t in J(src / "map.json")["tiles"]:
        b = reader(src / t["file"])
        if t["layer"] != "ground":
            rows.append(b)
            continue
        a = reader(pre / t["file"])
        old = reader(
            SRC / t["file"]
        )  # Recover each final centre's pre-crop appearance. Unsplit parents use the average of their own four pre-fit child colours.
        n = len(old)
        k = (len(a) - n) // 3
        u = n - k
        ctree = cKDTree(old[:, :3])
        h = ctree.query(a[:u, :3])[1]
        inds = np.flatnonzero(~np.isin(np.arange(n), h))
        assert len(inds) == k
        precol = np.empty((n, 3), "f")
        rgb = lambda f: (0.5 + C0 * f[..., 6:9]) * 255
        precol[h] = rgb(a[:u])
        precol[inds] = rgb(a[u:].reshape(-1, 4, 24)).mean(1)
        par = ctree.query(b[:, :3])[1]
        d, j = cKDTree(a[:, :3]).query(b[:, :3], distance_upper_bound=0.0001)
        base = precol[par].copy()
        hit = d < 0.0001
        base[hit] = rgb(a[j[hit]])
        lat = route.query(b[:, :2])[0]
        native = rgb(old)[par]
        base += np.where((lat < 6)[:, None], 0.16, 0.26) * (native - base)
        col = rgb(b)
        delta = col - base
        dy = delta @ W
        ix = np.floor(b[:, :2] / 0.12).astype(int)
        ix -= ix.min(0)
        x, y = ix.T
        w, hh = ix.max(0) + 1
        key = y * w + x
        den = np.bincount(key, minlength=hh * w).reshape(hh, w)
        smooth = (
            gaussian_filter(
                np.bincount(key, weights=dy, minlength=hh * w).reshape(hh, w), 4.1
            )[y, x]
            / np.maximum(0.001, gaussian_filter(den.astype("f"), 4.1)[y, x])
        )  # Suppress large colour steps between training rectangles; keep measured high spatial frequencies in luminance. Strong, isolated native paint: preserve its original posterior rather than reinsert a twin.
        chroma = delta - dy[:, None]
        anchored = (
            base + (0.74 * (dy - smooth) + 0.12 * smooth)[:, None] + 0.10 * chroma
        )
        oldpaint = (dy > 18) & ((native @ W - base @ W) > 13)
        anchored[oldpaint] = col[oldpaint] - 0.70 * chroma[oldpaint]
        fac = np.clip((22 - lat) / 7, 0, 1)
        fix = fac[:, None] * (anchored - col)
        b = b.copy()
        b[:, 6:9] = (np.clip(col + fix, 0, 255) / 255 - 0.5) / C0
        rows.append(b)
        affected += int((fac > 0.5).sum())
        if (fac > 0.5).any():
            dc.append(np.mean(abs(fix[fac > 0.5])))
    write_scene(src, dst, rows)
    res = dict(corridor_centres=affected, mean_tile_delta=float(np.mean(dc)))
    savej(OUT / (dst.name + "-finish.json"), res)
    print(res, flush=True)


if __name__ == "__main__":
    run(*map(Path, sys.argv[1:4]))
