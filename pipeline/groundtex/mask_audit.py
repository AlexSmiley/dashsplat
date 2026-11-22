"""Independent decoded-alpha / source-pixel overlap test on all nontransparent interior samples in the three display footprints."""

from collections import Counter
from .v5 import *
from .proof import Atlas, RPT


def audit():
    at = Atlas()
    obs = read(W / "observations.json")
    tr = inputs()[0]
    trees = [
        cKDTree(tr.T_world_ego(np.arange(lo, hi + 0.01, 0.12))[:, :2, 3])
        for lo, hi in SLICES
    ]
    groups = {
        k: np.flatnonzero(
            [tree.query((np.array(k) + 0.5) * M)[0] < 43 for tree in trees]
        )
        for k in at.idx
    }
    canvas = Canvas()
    for p in (W / "state").glob("*.npy"):
        canvas.tiles[tuple(map(int, p.stem.split("_")))] = np.load(p)

    def tile(k):
        t = at.idx[k]
        z = round(C / t["texelSize"])
        pix = t["pixels"]
        cell = t["texelSize"]
        g = t["gutter"]
        size = pix + 2 * g
        i, j = k
        lc = int(np.ceil(g / z)) + 24
        org = np.array(k) * M - g * cell
        a = canvas.get((i * N - lc, j * N - lc, N + 2 * lc, N + 2 * lc))
        v = (a["ok"] > 0) & (a["id"] > 0)
        pre = softedge(a, v, None, None, None)
        crop = lc * z - g
        up = lambda arr, interp: cv.resize(arr, None, fx=z, fy=z, interpolation=interp)[
            crop : crop + size, crop : crop + size
        ]
        label = up(a["id"].astype("f"), cv.INTER_NEAREST).astype("i")
        pre = np.uint8(up(pre, cv.INTER_LINEAR) * 255)
        alpha = np.asarray(Image.open(OUT / t["texture"]).convert("RGBA"))[:, :, 3]
        valid = alpha > 0
        valid[:g] = valid[-g:] = 0
        valid[:, :g] = valid[:, -g:] = 0
        ct = Counter(
            nontransparent=int(valid.sum()),
            unowned=int((valid & (label == 0)).sum()),
            missing_removal=0,
            remove=0,
            dyn=0,
            invalid_rig=0,
        )
        for id in np.unique(label[valid]):
            if not id:
                continue
            r = obs[id - 1]
            cam = inputs()[1][r["cam"]]
            yy, xx = np.nonzero((label == id) & (pre > 0))
            x, y = int(xx.min()), int(yy.min())
            w, h = int(xx.max()) - x + 1, int(yy.max()) - y + 1
            uv = uvgrid(
                r, org + np.array([x, y]) * cell, (w, h), cell, 4 if z == 4 else 2
            )
            ys, xs = np.nonzero((label == id) & valid)
            coords = uv[ys - y, xs - x]
            mm = masks(r["cam"], r["n"])
            ct["invalid_rig"] += int((~cam.mask_at(coords)).sum())
            if "remove" not in mm:
                ct["missing_removal"] += len(coords)
            for kind in ["remove", "dyn"]:
                if kind not in mm:
                    continue
                m = mm[kind]
                ij = np.floor(
                    coords * np.array([m.shape[1] / cam.width, m.shape[0] / cam.height])
                    - 0.5
                ).astype(int)
                sx, sy = ij.T
                hit = np.zeros(len(sx), bool)
                if m.any():
                    for dx, dy in [(0, 0), (1, 0), (0, 1), (1, 1)]:
                        hit |= (
                            m[
                                (sy + dy).clip(0, m.shape[0] - 1),
                                (sx + dx).clip(0, m.shape[1] - 1),
                            ]
                            != 0
                        )
                ct[kind] += int(hit.sum())
        return k, ct

    totals = [Counter() for _ in SLICES]
    ks = [k for k, g in groups.items() if len(g)]
    with ThreadPoolExecutor(4) as ex:
        for k, ct in ex.map(tile, ks):
            for i in groups[k]:
                totals[i].update(ct)
                totals[i]["tiles"] += 1
    result = {f"{lo}-{hi}": dict(ct) for (lo, hi), ct in zip(SLICES, totals)}
    dump(RPT / "slice-masks.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    audit()
