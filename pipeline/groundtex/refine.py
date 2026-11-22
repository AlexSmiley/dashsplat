"""Second pass: robust local paint registration, smooth strip illumination and connected-mark ownership.
The old splat field is a low-frequency *geometry witness*, never an RGB source in the atlas.
"""

from .atlas import *
from collections import defaultdict

BASE = Path("work/groundtex")
GUIDE = Path("scene/stitched-opt-v4")
FS = W / "fields"
CS = W / "coarse"


@lru_cache(1)
def idx():
    # v4's ground ends with its optimized slice. The independently verified v2
    # ground remains the absolute geometry witness elsewhere on this drive.
    original = [dict(t, root=ROAD) for t in roadtiles()]
    opt = [
        dict(t, root=GUIDE)
        for t in read(GUIDE / "map.json")["tiles"]
        if t["layer"] == "ground"
    ]
    covered = {t["id"] for t in opt}
    return opt + [t for t in original if t["id"] not in covered]


@lru_cache(256)
def guide(k):
    i, j = k
    pad = 9
    x, y = np.array(k) * M
    shape = (N + 2 * pad, N + 2 * pad)
    g = np.zeros(shape + (4,), "f")
    for e in idx():
        a, b, _, A, B, _ = e["bbox"]
        if a > x + M + 0.4 or b > y + M + 0.4 or A < x - 0.4 or B < y - 0.4:
            continue
        v = reader(e["root"] / e["file"])
        sel = (
            (abs(v[:, 0] - x - M / 2) < M / 2 + 0.40)
            & (abs(v[:, 1] - y - M / 2) < M / 2 + 0.40)
            & (v[:, 5] > 0.89)
            & (v[:, 10] < np.log(0.19))
            & (v[:, 11] < np.log(0.22))
        )
        a = v[sel]
        if not len(a):
            continue
        uv = ((a[:, :2] - [x, y]) / C + pad).astype(int).clip(0, shape[0] - 1)
        key = uv[:, 1] * shape[1] + uv[:, 0]
        wg = np.clip(np.exp(a[:, 10] + a[:, 11]) / 0.006, 0.15, 2)
        rgb = np.clip((0.5 + 0.28209479 * a[:, 6:9][:, ::-1]) * 255, 0, 255)
        for ch in range(4):
            g[:, :, ch] += np.bincount(
                key,
                weights=wg * (rgb[:, ch] if ch < 3 else 1),
                minlength=g.shape[0] ** 2,
            ).reshape(shape)
    g = cv.GaussianBlur(g, (0, 0), 2.0)
    g[:, :, :3] /= np.maximum(g[:, :, 3:4], 0.0001)
    return g[pad:-pad, pad:-pad]


def guide_box(box):
    a = np.zeros((box[3], box[2], 4), "f")
    for k, s, b in slices(box):
        a[b] = guide(k)[s]
    return a


def proj_key(r):
    return f"{r['cam']}-{r['n']:06d}"


def bar(im, sigma):
    return cv.GaussianBlur(im, (0, 0), sigma)


def cached(r):
    path = CS / (proj_key(r) + ".npz")
    fld = FS / (proj_key(r) + ".npz")
    r.pop("gain", None)
    r["field"] = str(fld)
    v = np.load(path)
    box = tuple(v["box"].tolist())
    rgb = v["rgb"]
    if v["q"].dtype == np.uint8:
        raise ValueError(
            "Completed tone cache cannot seed a new global solve; use a fresh GT_WORK."
        )
    q = v["q"].astype("f")
    lum = cv.cvtColor(rgb, cv.COLOR_BGR2GRAY).astype("f")
    flat = (
        (abs(lum - bar(lum, 6)) < 4)
        & (lum > 35)
        & (lum < 185)
        & (rgb.max(2) - rgb.min(2) < 55)
    )
    wt = flat * np.minimum(15, q * q) * (1 if r["cam"] == "front" else 0.21)
    return box, rgb.astype("f") * wt[:, :, None], wt


def match(r):
    if (CS / (proj_key(r) + ".npz")).is_file() and (
        FS / (proj_key(r) + ".npz")
    ).is_file():
        return cached(r)
    # Registered first-pass frame -> absolute consensus geometry. Use NCC on high-pass
    # luminance (not RGB), only tangent evidence. Edge-only patches constrain normal
    # displacement; a saturated aperture cannot estimate displacement along a stripe.
    r.pop("gain", None)
    r.pop("field", None)
    box, rgb, q, sem = coarse(r)
    x, y, w, h = box
    g = guide_box(box)
    valid = (q > 0) & (g[:, :, 3] > 0.012)
    p = cv.cvtColor(g[:, :, :3], cv.COLOR_BGR2GRAY)
    s = cv.cvtColor(rgb, cv.COLOR_BGR2GRAY).astype("f")
    p = bar(p, 1) - bar(p, 8)
    s = bar(s, 1.4) - bar(s, 8)
    gx = cv.Sobel(p, cv.CV_32F, 1, 0)
    gy = cv.Sobel(p, cv.CV_32F, 0, 1)
    yy, xx = np.mgrid[0:h:8, 0:w:8]
    ctrl = np.zeros(yy.shape + (3,), "f")
    pad = 11
    rad = 15
    margin = pad + rad + 1
    for j in range(4, yy.shape[0] - 3, 2):
        for i in range(4, yy.shape[1] - 3, 2):
            X = xx[j, i]
            Y = yy[j, i]
            if X < margin or Y < margin or X >= w - margin or Y >= h - margin:
                continue
            sl = np.s_[Y - pad : Y + pad + 1, X - pad : X + pad + 1]
            tpl = p[sl]
            if valid[sl].mean() < 0.89 or tpl.std() < 1.2:
                continue
            search = s[
                Y - pad - rad : Y + pad + rad + 1, X - pad - rad : X + pad + rad + 1
            ]
            cc = cv.matchTemplate(search, tpl, cv.TM_CCOEFF_NORMED)
            ys, xs = np.mgrid[-rad : rad + 1, -rad : rad + 1]
            c = cc - 0.11 * (xs * xs + ys * ys) / rad**2
            V, U = np.unravel_index(np.argmax(c), c.shape)
            score = cc[V, U]
            if score < 0.65 or abs(U - rad) >= rad or abs(V - rad) >= rad:
                continue
            dx = U - rad
            dy = V - rad
            ax = gx[sl]
            ay = gy[sl]
            vals, vec = np.linalg.eigh(
                [
                    [np.mean(ax * ax), np.mean(ax * ay)],
                    [np.mean(ax * ay), np.mean(ay * ay)],
                ]
            )
            vel = np.array([dx, dy], "f")
            if vals[0] < vals[1] * 0.14:
                vel = vec[:, -1] * np.dot(vec[:, -1], vel)
            weight = (score - 0.5) ** 2 * min(2.5, tpl.std() / 4)
            ctrl[j, i] = [*(vel * weight), weight]
    # Low-order smooth correction; spatially unconstrained regions retain the frame's prior.
    strength = bar(ctrl[:, :, 2], 2.3)
    delta = bar(ctrl[:, :, :2], 2.3) / (strength[:, :, None] + 0.0007)
    delta = np.clip(delta, -12, 12)
    origin = (np.array([x, y]) + 0.5) * C
    Y, X = np.mgrid[:h, :w].astype("f")
    xy = np.stack([X, Y], -1)
    shift = cv.remap(
        delta, xy / 8, None, cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE
    )
    mp = xy + shift
    warp = lambda im, flag: cv.remap(im, mp, None, flag)
    rgb = warp(rgb, cv.INTER_LINEAR)
    q = warp(q, cv.INTER_NEAREST)
    sem = warp(sem, cv.INTER_NEAREST)
    # Reference for illumination is the low-frequency neutral photograph population,
    # not v4's DC: store each neutral observation and accumulate a separate tone field.
    lum = cv.cvtColor(rgb, cv.COLOR_BGR2GRAY).astype("f")
    lp = bar(lum, 6)
    flat = (
        (abs(lum - lp) < 4) & (lum > 35) & (lum < 185) & (rgb.max(2) - rgb.min(2) < 55)
    )
    v = (q > 0) & flat
    wt = v.astype("f") * np.minimum(15, q * q) * (1 if r["cam"] == "front" else 0.21)
    m = rgb.astype("f") * wt[:, :, None]
    # Keep previous evidence gates, but require a geometry witness away from the lane.
    prior = g[:, :, 3] > 0.012
    q[~prior] = 0
    k = proj_key(r)
    FS.mkdir(parents=True, exist_ok=True)
    CS.mkdir(parents=True, exist_ok=True)
    np.savez(
        FS / (k + ".npz"),
        xy=origin,
        d=(delta * C).astype("f2"),
        gain=np.ones(yy.shape + (3,), "f2"),
    )
    np.savez_compressed(CS / (k + ".npz"), box=box, rgb=rgb, q=q.astype("f2"), sem=sem)
    r["field"] = str(FS / (k + ".npz"))
    r["local_matches"] = int(np.count_nonzero(ctrl[:, :, 2]))
    return box, m, wt


def qget(box, d, channels):
    a = np.zeros((box[3], box[2], channels), "f")
    for k, s, b in slices(box):
        if k in d:
            a[b] = d[k][s]
    return a


def tone(r, acc):
    k = proj_key(r)
    cache = CS / (k + ".npz")
    v = np.load(cache)
    box = tuple(v["box"])
    rgb = v["rgb"]
    q = v["q"].astype("f")
    sem = v["sem"]
    x, y, w, h = box
    f = np.load(r["field"])
    origin = f["xy"]
    d = f["d"]
    pad = 32
    tar = qget((x - pad, y - pad, w + 2 * pad, h + 2 * pad), acc, 4)
    tar = bar(tar, 14)[pad:-pad, pad:-pad]
    target = tar[:, :, :3] / np.maximum(tar[:, :, 3:], 0.001)
    lum = cv.cvtColor(rgb, cv.COLOR_BGR2GRAY).astype("f")
    m = (
        (q > 0)
        & (tar[:, :, 3] > 0.015)
        & (lum > 35)
        & (lum < 185)
        & (abs(lum - bar(lum, 6)) < 4)
        & (rgb.max(2) - rgb.min(2) < 55)
    ).astype("f")
    dm = np.log(np.maximum(target, 8) / np.maximum(rgb, 8)).clip(-0.37, 0.37)
    ww = bar(m, 12)
    gain = np.exp(bar(dm * m[:, :, None], 12) / (ww[:, :, None] + 0.02))
    g = gain[::8, ::8].astype("f2")
    np.savez(r["field"], xy=origin, d=d, gain=g)
    rgb = np.uint8(np.clip(rgb * gain + 0.5, 0, 255))
    np.savez_compressed(
        cache, box=box, rgb=rgb, q=np.round(np.clip(q, 0, 12) * 20).astype("B"), sem=sem
    )


@lru_cache(240)
def patch(rkey):
    with np.load(CS / (rkey + ".npz")) as v:
        return v["box"], v["q"], v["rgb"], v["sem"]


def tilefix(k, canvas, obs, index):
    # Halo is wider than a complete arrow / bicycle. A component owned by one
    # exposure gets that exposure throughout the marking and its pavement margin.
    i, j = k
    pad = 100
    box = (i * N - pad, j * N - pad, N + 2 * pad, N + 2 * pad)
    a = canvas.get(box)
    guid = guide_box(box)
    g = cv.cvtColor(guid[:, :, :3], cv.COLOR_BGR2GRAY)
    valid = guid[:, :, 3] > 0.012
    mark = (g - bar(g, 11) > 4.6) & valid
    mark = cv.morphologyEx(
        mark.astype("B"),
        cv.MORPH_CLOSE,
        cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7)),
    )
    mark = cv.dilate(mark, cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7)))
    cnt, lab, st, cent = cv.connectedComponentsWithStats(mark, 8)
    ids = sorted(set(id for K, _, _ in slices(box) for id in index[K]))
    h, w = a.shape
    bestq = np.zeros(cnt, "f")
    best = np.zeros(cnt, "i")
    use = np.zeros(cnt, bool)
    use[1:] = (
        (st[1:, 4] >= 24)
        & (st[1:, 2] < 240)
        & (st[1:, 3] < 240)
        & (st[1:, 0] > 3)
        & (st[1:, 1] > 3)
        & (st[1:, 0] + st[1:, 2] < w - 3)
        & (st[1:, 1] + st[1:, 3] < h - 3)
    )
    candidates = []
    for id in ids:
        r = obs[id - 1]
        bb, q, rgb, sm = patch(proj_key(r))
        X = max(box[0], bb[0])
        Y = max(box[1], bb[1])
        E = min(box[0] + w, bb[0] + bb[2])
        F = min(box[1] + h, bb[1] + bb[3])
        if E <= X or F <= Y:
            continue
        sl = np.s_[Y - box[1] : F - box[1], X - box[0] : E - box[0]]
        ss = np.s_[Y - bb[1] : F - bb[1], X - bb[0] : E - bb[0]]
        Q = q[ss]
        m = lab[sl]
        valid = Q > 0
        zz = m[valid]
        area = np.bincount(zz, minlength=cnt)
        qua = np.bincount(zz, weights=Q[valid], minlength=cnt) / np.maximum(1, st[:, 4])
        good = use & (area >= st[:, 4] * 0.998) & (qua > bestq)
        bestq[good] = qua[good]
        best[good] = id
        candidates.append((id, sl, ss))
    picks = best[lab]
    current = a["id"].copy()
    aq = np.zeros_like(a["q"], "f")
    witness = np.zeros(a.shape + (3,), "B")
    usevote = np.zeros(a.shape, bool)
    for id, sl, ss in candidates:
        bb, q, rgb, sm = patch(proj_key(obs[id - 1]))
        q = q[ss]
        im = rgb[ss]
        owner = (picks[sl] == id) & (q > 0)
        same = (current[sl] == id) & (q > 0) & ((picks[sl] == 0) | owner)
        take = owner | same
        aq[sl][take] = q[take]
        witness[sl][take] = im[take]
        a["id"][sl][take] = id
        usevote[sl][take] = True
    # A warped validity edge can leave its old cell. Recover from an actual surviving
    # observation, otherwise reveal the splat fallback. Retain previous evidence mask.
    missing = (~usevote) & (a["ok"] > 0)
    for id, sl, ss in candidates:
        bb, q, rgb, sm = patch(proj_key(obs[id - 1]))
        q = q[ss]
        take = missing[sl] & (q > aq[sl])
        aq[sl][take] = q[take]
        witness[sl][take] = rgb[ss][take]
        a["id"][sl][take] = id
        usevote[sl][take] = True
    a["ok"] &= usevote
    a["rgb"] = witness
    return k, a[pad : pad + N, pad : pad + N].copy(), int(np.count_nonzero(best))


def run(lo, hi, workers):
    W.mkdir(parents=True, exist_ok=True)
    (W / "state").mkdir(exist_ok=True)
    (OUT / "tiles").mkdir(parents=True, exist_ok=True)
    base = read(BASE / "observations.json")
    obs = [dict(r) for r in base if lo <= r["t"] <= hi]
    lb = {(r["cam"], r["n"]): i + 1 for i, r in enumerate(obs)}
    remap = np.array([0] + [lb.get((r["cam"], r["n"]), 0) for r in base], "i")
    canvas = Canvas()
    idx_ = defaultdict(list)
    acc = {}
    ts = time.monotonic()
    for j, r in enumerate(obs):
        box = footprint(r)[0]
        for k, _, _ in slices(box):
            idx_[k].append(j + 1)
    for k in idx_:
        p = BASE / "state" / f"{k[0]}_{k[1]}.npy"
        if not p.is_file():
            continue
        a = np.load(p)
        a["id"] = remap[a["id"]]
        a["ok"][a["id"] == 0] = 0
        canvas.tiles[k] = a
    print("refine", len(obs), len(canvas.tiles), flush=True)
    with ThreadPoolExecutor(workers) as ex:
        for j in range(0, len(obs), workers * 3):
            rr = obs[j : j + workers * 3]
            for r, (box, rgb, wt) in zip(rr, ex.map(match, rr)):
                for k, s, b in slices(box):
                    if k not in acc:
                        acc[k] = np.zeros((N, N, 4), "f")
                    acc[k][s][:, :, :3] += rgb[b]
                    acc[k][s][:, :, 3] += wt[b]
            if j % (workers * 24) == 0:
                print("matches", j, "elapsed", round(time.monotonic() - ts), flush=True)
        list(ex.map(lambda r: tone(r, acc), obs))
        acc.clear()
        print("selecting marks", flush=True)
        ks = list(canvas.tiles)
        out = Canvas()
        comps = 0
        for k, a, n in ex.map(lambda k: tilefix(k, canvas, obs, idx_), ks):
            out.tiles[k] = a
            comps += n
        canvas = out
        print("export", len(ks), "mark assignments", comps, flush=True)
        tiles = [t for t in ex.map(lambda k: export_tile(k, canvas, obs), ks) if t]
    for (i, j), a in canvas.tiles.items():
        np.save(W / "state" / f"{i}_{j}.npy", a)
    dump(W / "observations.json", obs)
    meta = read(Path("scene/stitched-opt-v4/ground/ground.json"))
    meta.update(intervals=[[lo, hi]], tiles=tiles)
    dump(OUT / "ground.json", meta)
    print("published", len(tiles), "elapsed", round(time.monotonic() - ts), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("lo", type=float)
    p.add_argument("hi", type=float)
    p.add_argument("--workers", type=int, default=6)
    a = p.parse_args()
    assert W != BASE
    run(a.lo, a.hi, a.workers)
