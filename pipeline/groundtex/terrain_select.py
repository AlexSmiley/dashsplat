"""Require low ground classes, depth, multiple views and agreement. Repeat ownership selection without foliage standing in for grass."""

from .outer import *
from .ring_export import ctrl, rem


@lru_cache(20)
def goodmask(id):
    a = cv.imread(str(W / "classes" / f"{id}.png"), 0)
    return cv.erode(np.uint8(np.isin(a, [0, 1, 9])), np.ones((3, 3), "B")), a


def region(id, r):
    box, im, q, sem = cached(id)
    h, w = q.shape
    if min(h, w) < 6:
        return None
    xy = boxxy(box)
    z = surface().take(box)[:, :, 0]
    uv, _, _, _ = project(r, xy, z)
    cam = inputs()[1][r["cam"]]
    m, classes = goodmask(id)
    lab = sam(classes, uv, cam, True)
    valid = (q > 0) & (sam(m, uv, cam, True) > 0)
    q[~valid] = 0
    orig, delta, gain = ctrl(id)
    mp = (xy - orig) / (4 * L)
    shift = rem(delta, mp) / L
    y, x = np.mgrid[:h, :w].astype("f")
    uv0 = np.stack([x, y], -1) + shift
    im = rem(im, uv0) * rem(gain, mp)
    im = np.uint8(im.clip(0, 255))
    q = rem(q, uv0, False)
    lab = rem(lab, uv0, False)
    return box, im, q, lab


def run(workers=6):
    rows = read(W / "outer-rows.json")
    surface()
    cache = W / "terrain-coarse"
    cache.mkdir(exist_ok=True)
    agg = Grid(9)

    def pre(item):
        id, r = item
        res = region(id, r)
        if res is None:
            return None
        box, im, q, lab = res
        good = (q > 0).astype("f")
        eye = rowpose(r["cam"], r["t"])[8, :2, 3]
        wt = q**1.45
        z = np.dstack(
            [
                im * wt[:, :, None],
                wt,
                good,
                good * (lab == 9),
                good * (lab == 1),
                good * eye[0],
                good * eye[1],
            ]
        ).astype("f")
        np.savez_compressed(
            cache / f"{id}.npz", box=box, rgb=im, q=q.astype("f2"), lab=lab
        )
        return box, z

    with ThreadPoolExecutor(workers) as ex:
        for start in range(0, len(rows), workers * 3):
            for v in ex.map(
                pre, enumerate(rows[start : start + workers * 3], start + 1)
            ):
                if v:
                    agg.apply(*v, add=True)
            if start % (workers * 90) == 0:
                print("surface class votes", start, flush=True)
    ids = Grid(1, "i4")
    qual = Grid(1)
    col = Grid(3, "B")
    labels = Grid(1, "B")
    clear = Grid(1)
    semstats = Counter()
    for k in pathkeys():
        with np.load(W / "map" / f"{k[0]}_{k[1]}.npz") as a:
            clear.a[k] = (1 - a["object"].astype("f"))[:, :, None]
    for id, r in enumerate(rows, 1):
        file = cache / f"{id}.npz"
        if not file.exists():
            continue
        with np.load(file) as d:
            box = tuple(map(int, d["box"]))
            im = d["rgb"]
            q = d["q"].astype("f")
            lab = d["lab"]
        v = agg.take(box)
        avg = v[:, :, :3] / v[:, :, 3:4].clip(0.001)
        vv = v[
            :, :, 4
        ]  # A second depth/semantic witness alone is insufficient if all viewing rays have the same origin.
        eye = rowpose(r["cam"], r["t"])[8, :2, 3]
        meaneye = v[:, :, 7:9] / vv[:, :, None].clip(1)
        baseline = np.linalg.norm(meaneye - eye, axis=2)
        agreement = abs(im.astype("f") - avg).mean(2)
        grass = lab == 9
        vote = np.where(
            grass,
            v[:, :, 5],
            np.where(lab == 1, v[:, :, 6], vv - v[:, :, 5] - v[:, :, 6]),
        )
        ok = (
            (vv >= 2)
            & (baseline > 0.30)
            & (vote >= np.minimum(2, vv * 0.65))
            & (cv.GaussianBlur(agreement, (0, 0), 0.75) < np.where(grass, 22, 26))
            & (clear.take(box)[:, :, 0] > 0.5)
        )
        score = np.where(ok, q * np.clip(1 - agreement / 80, 0.5, 1), 0)
        oq = qual.take(box)[:, :, 0]
        old = col.take(box)
        oi = ids.take(box)[:, :, 0]
        ol = labels.take(box)[:, :, 0]
        sel = seam(im, old, score, oq)
        oq[sel] = score[sel]
        oi[sel] = id
        old[sel] = im[sel]
        ol[sel] = lab[sel]
        ids.apply(box, oi[:, :, None])
        qual.apply(box, oq[:, :, None])
        col.apply(box, old)
        labels.apply(box, ol[:, :, None])
        if id % 500 == 0:
            print("source selection", id, flush=True)
    out = W / "terrain-map"
    out.mkdir(exist_ok=True)
    for k in pathkeys():
        p = W / "map" / f"{k[0]}_{k[1]}.npz"
        v = agg.pad(k, 0)
        id = ids.pad(k, 0)[:, :, 0]
        lab = labels.pad(k, 0)[:, :, 0]
        with np.load(p) as d:
            np.savez_compressed(
                out / p.name,
                id=id,
                rgb=col.pad(k, 0),
                prior=d["prior"],
                object=d["object"],
                dist=d["dist"],
                votes=v[:, :, 4].astype("f2"),
                road=(v[:, :, 4] - v[:, :, 5] - v[:, :, 6]).astype("f2"),
                label=lab,
            )
        for val, name in [(0, "road"), (1, "pavement"), (9, "low_terrain")]:
            semstats[name] += int(((id > 0) & (lab == val)).sum())
    dump(W / "terrain-selection.json", dict(coarse_cells=semstats, cell_m=L))
    print("terrain selection ready", flush=True)


if __name__ == "__main__":
    run()
