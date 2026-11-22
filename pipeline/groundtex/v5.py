"""V5 photographic ground. Build global registration and illumination guides; each displayed texel has one owner."""

from .atlas import *
from collections import Counter, defaultdict
from threading import Lock

BASE = Path("work/groundtex-final/state")
BASEOUT = Path("work/groundtex-final/ground")
SLICES = [(85, 140), (190, 215), (286, 330)]
P = W / "patches"
F = W / "fields"
D8 = 0.08
T = 128
stats = Counter()
mutex = Lock()


def key(r):
    return f"{r['cam']}-{r['n']:06d}"


def extent(box):
    return box[0] + box[2], box[1] + box[3]


def pieces(box):
    x, y, w, h = box
    for j in range(y // T, (y + h - 1) // T + 1):
        for i in range(x // T, (x + w - 1) // T + 1):
            X, Y = max(x, i * T), max(y, j * T)
            E, A = min(x + w, (i + 1) * T), min(y + h, (j + 1) * T)
            yield (
                (i, j),
                np.s_[Y - j * T : A - j * T, X - i * T : E - i * T],
                np.s_[Y - y : A - y, X - x : E - x],
            )


class Grid:
    def __init__(self, n, dtype="f"):
        self.a = {}
        self.n = n
        self.dt = dtype

    def read(self, b):
        z = np.zeros((b[3], b[2], self.n), self.dt)
        for k, s, v in pieces(b):
            if k in self.a:
                z[v] = self.a[k][s]
        return z

    def store(self, b, arr, add=False):
        for k, s, v in pieces(b):
            if k not in self.a:
                self.a[k] = np.zeros((T, T, self.n), self.dt)
            if add:
                self.a[k][s] += arr[v]
            else:
                self.a[k][s] = arr[v]


@lru_cache(1)
def display():
    tr = inputs()[0]
    return cKDTree(
        np.concatenate(
            [
                tr.T_world_ego(np.arange(lo, hi + 0.01, 0.12))[:, :2, 3]
                for lo, hi in SLICES
            ]
        )
    )


@lru_cache(2400)
def mandatory(k):
    return display().query((np.array(k) + 0.5) * M)[0] < 43


@lru_cache(80)
def masks(cam, n):
    a = {}
    for kind in ["road", "remove", "dyn", "sky"]:
        if n not in files(cam, kind):
            continue
        p = (
            (
                Path("maps/movers-v1/remove")
                if kind == "remove"
                else Path("masks") / kind
            )
            / cam
            / f"{n:06d}.png"
        )
        m = cv.imread(str(p), -1)
        a[kind] = np.uint8(m > 0)
    return a


def actual_guard(r, uv, yy, xx, k):
    cam = inputs()[1][r["cam"]]
    mm = masks(r["cam"], r["n"])
    p = uv[yy, xx]
    ok = cam.mask_at(p)
    hits = 0
    if mandatory(k) and "remove" not in mm:
        ok[:] = False
    for kind in ["remove", "dyn", "sky"]:
        if kind not in mm:
            continue
        m = mm[kind]
        g = p * np.array([m.shape[1] / cam.width, m.shape[0] / cam.height]) - 0.5
        ij = np.floor(g).astype(int)
        x, y = ij.T
        bad = np.zeros(len(x), bool)
        for dx, dy in [(0, 0), (1, 0), (0, 1), (1, 1)]:
            bad |= (
                m[(y + dy).clip(0, m.shape[0] - 1), (x + dx).clip(0, m.shape[1] - 1)]
                != 0
            )
        if kind in ("remove", "dyn"):
            hits += int(bad.sum())
        ok &= ~bad
    if mandatory(k):
        with mutex:
            stats["required_native_samples_including_gutters"] += len(ok)
            stats["rejected_native_samples"] += int((~ok).sum())
            stats["removed_mover_or_instance_mask_hits"] += hits
            stats["admitted_without_removal_mask"] += (
                int(ok.sum()) if "remove" not in mm else 0
            )
    return ok


def box8(r):
    if r["cam"] != "front":
        bb, eye, ran = footprint(r)
        return tuple(v // 2 for v in bb), eye, ran
    tr, rig, _ = inputs()
    c = rig["front"]
    T0 = tr.T_world_ego(r["t"]) @ c.T_ego_cam
    eye = T0[:2, 3]
    f = T0[:2, 2]
    f = f / np.linalg.norm(f)
    right = np.array([-f[1], f[0]])
    ra = 36.0
    p = np.array(
        [eye + f * a + right * b for a, b in [(2, -2), (2, 2), (ra, -16), (ra, 16)]]
    )
    b = np.floor((p.min(0) - 0.32) / 0.32).astype(int) * 4
    B = np.ceil((p.max(0) + 0.32) / 0.32).astype(int) * 4
    return tuple(map(int, [*b, *(B - b)])), eye, ra


def raw(r, old):
    bb, eye, ra = box8(r)
    x, y, w, h = bb
    c = r["cam"]
    cam = inputs()[1][c]
    start = np.array([x, y]) * D8
    Y, X = np.mgrid[:h, :w].astype("f")
    grid = start + (np.stack([X, Y], -1) + 0.5) * D8
    # Front-to-front anchor uses the physical near-view gauge; sides start from the existing calibration refinement.
    uv = uvgrid(r, start, (w, h), D8, 3)
    rgb = photo(r, uv, True)
    mm = masks(c, r["n"])
    ok = cam.mask_at(uv)
    sem = np.zeros((h, w), "B")
    for kind, m in mm.items():
        a = sam(m, uv, cam, True)
        if kind == "road":
            sem = a
        else:
            ok &= cv.dilate(a, np.ones((5, 5), "B")) == 0
    if "remove" not in mm:
        for k, s, b in pieces(bb):
            if mandatory(k):
                ok[b] = 0
    dist = np.linalg.norm(grid - eye, axis=2)
    ok &= (dist <= ra) & (dist > 1.65)
    prior = old.read(bb)[:, :, 0]
    ok &= (prior > 0) | (sem > 0)
    edge = cv.distanceTransform(ok.astype("B"), cv.DIST_L2, 5)
    ok &= edge > 1.8
    dx = np.gradient(uv, axis=1) / D8
    dy = np.gradient(uv, axis=0) / D8
    a = (dx**2).sum(2)
    b = (dy**2).sum(2)
    off = (dx * dy).sum(2)
    smin = np.sqrt(np.maximum(0, (a + b - np.hypot(a - b, 2 * off)) * 0.5))
    q = np.where(
        ok, np.clip(smin / 100, 0.07, 2.5) ** 0.54 * np.clip(edge / 10, 0.38, 1), 0
    ).astype("f")
    # A far front pixel must land on previously measured pavement or actual semantic road. Its appearance is independently checked in choose().
    np.savez_compressed(
        P / (key(r) + ".npz"), box=bb, rgb=rgb, q=q.astype("f2"), sem=sem
    )
    obs = (q > 0).astype("f")
    votew = obs if "road" in mm else obs * 0
    front = c == "front"
    geo = (
        q**2 * (14 if front else 0.22) * np.clip(1.2 - abs(dist - 9.5) * 0.055, 0.25, 1)
        if front
        else q**2 * 0.22
    )
    lum = cv.cvtColor(rgb, cv.COLOR_BGR2GRAY).astype("f")
    flat = (
        (abs(lum - cv.GaussianBlur(lum, (0, 0), 4)) < 4)
        & (lum > 40)
        & (lum < 185)
        & (rgb.max(2) - rgb.min(2) < 58)
    )
    tone = geo * flat
    return bb, np.dstack(
        [
            rgb * geo[:, :, None],
            geo,
            rgb * tone[:, :, None],
            tone,
            sem * votew,
            votew,
            obs,
            obs if front else obs * 0,
        ]
    ).astype("f")


def blur(im, s):
    return cv.GaussianBlur(im, (0, 0), s)


@lru_cache(72)
def patch(k):
    with np.load(P / (k + ".npz")) as v:
        return tuple(v["box"]), v["rgb"], v["q"].astype("f"), v["sem"]


def fit(r, guide):
    bb, im, q, sem = patch(key(r))
    x, y, w, h = bb
    pad = 22
    b0 = (x - pad, y - pad, w + pad * 2, h + pad * 2)
    g = guide.read(b0)
    v = g[pad:-pad, pad:-pad]
    target = v[:, :, :3] / np.maximum(v[:, :, 3:4], 0.001)
    base = cv.cvtColor(target, cv.COLOR_BGR2GRAY)
    src = cv.cvtColor(im, cv.COLOR_BGR2GRAY).astype("f")
    base = blur(base, 0.7) - blur(base, 5)
    src = blur(src, 0.7) - blur(src, 5)
    rows, cols = np.mgrid[0:h:4, 0:w:4]
    controls = np.zeros(rows.shape + (3,), "f")
    before = []
    after = []
    valid = (q > 0) & (v[:, :, 3] > 0.025)
    rad = 10 if r["cam"] == "front" else 12
    half = 8
    win = half + rad
    dy0, dx0 = np.mgrid[-rad : rad + 1, -rad : rad + 1]
    gx = cv.Sobel(base, cv.CV_32F, 1, 0)
    gy = cv.Sobel(base, cv.CV_32F, 0, 1)
    for j in range((win + 3) // 4, rows.shape[0] - (win + 3) // 4, 2):
        for i in range((win + 3) // 4, cols.shape[1] - (win + 3) // 4, 2):
            X = cols[j, i]
            Y = rows[j, i]
            sl = np.s_[Y - half : Y + half + 1, X - half : X + half + 1]
            tpl = base[sl]
            if valid[sl].mean() < 0.91 or tpl.std() < 1.35:
                continue
            cc = cv.matchTemplate(
                src[Y - win : Y + win + 1, X - win : X + win + 1],
                tpl,
                cv.TM_CCOEFF_NORMED,
            )
            score = cc - 0.055 * (dx0**2 + dy0**2) / rad**2
            V, U = np.unravel_index(np.argmax(score), score.shape)
            qual = cc[V, U]
            if qual < 0.67 or V in (0, 2 * rad) or U in (0, 2 * rad):
                continue

            def sub(a, b, c):
                return float(
                    np.clip(0.5 * (a - c) / min(-1e-5, a - 2 * b + c), -0.5, 0.5)
                )

            vec = np.array(
                [
                    U - rad + sub(cc[V, U - 1], qual, cc[V, U + 1]),
                    V - rad + sub(cc[V - 1, U], qual, cc[V + 1, U]),
                ]
            )
            sx, sy = gx[sl], gy[sl]
            val, dir = np.linalg.eigh(
                [
                    [np.mean(sx * sx), np.mean(sx * sy)],
                    [np.mean(sx * sy), np.mean(sy * sy)],
                ]
            )
            if val[0] < val[1] * 0.16:
                vec = dir[:, -1] * np.dot(dir[:, -1], vec)
            weight = (qual - 0.48) ** 2 * np.clip(tpl.std() / 4, 0.3, 3)
            controls[j, i] = [*vec * weight, weight]
            before.append(float(cc[rad, rad]))
            after.append(float(qual))
    weight = blur(controls[:, :, 2], 1.65)
    delta = blur(controls[:, :, :2], 1.65) / (weight[:, :, None] + 0.0009)
    Y, X = np.mgrid[:h, :w].astype("f")
    ij = np.stack([X, Y], -1)
    shift = cv.remap(
        delta, ij / 4, None, cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE
    )
    mp = ij + shift
    q = cv.remap(q, mp, None, cv.INTER_NEAREST)
    sem = cv.remap(sem, mp, None, cv.INTER_NEAREST)
    rgb = cv.remap(im, mp, None, cv.INTER_LINEAR)
    # Smooth multiplicative illumination only. Compute the per-patch residual from asphalt, not from painted features or isolated dark shadows.
    t = blur(g[:, :, 4:8], 12)[pad:-pad, pad:-pad]
    tone = t[:, :, :3] / np.maximum(0.001, t[:, :, 3:4])
    lum = cv.cvtColor(rgb, cv.COLOR_BGR2GRAY).astype("f")
    flat = (
        (q > 0)
        & (t[:, :, 3] > 0.015)
        & (lum > 40)
        & (lum < 185)
        & (abs(lum - blur(lum, 5)) < 4.8)
        & (rgb.max(2) - rgb.min(2) < 58)
    )
    m = flat.astype("f")
    loggain = np.log(np.maximum(tone, 10) / np.maximum(rgb, 10)).clip(-0.32, 0.32)
    G = np.exp(blur(loggain * m[:, :, None], 7.5) / (blur(m, 7.5)[:, :, None] + 0.008))
    G = G[::4, ::4]
    origin = (np.array([x, y]) + 0.5) * D8
    pts = origin + np.stack([cols, rows], -1) * D8 + delta * D8
    D = delta * D8
    if "field" in r:
        D = D + localfield(r, pts, "d")
    dst = F / (key(r) + ".npz")
    np.savez(dst, xy=origin, d=D.astype("f2"), gain=G.astype("f2"))
    r["field"] = str(dst)
    r["pairfit"] = [
        len(before),
        round(float(np.median(before)), 3) if before else None,
        round(float(np.median(after)), 3) if after else None,
    ]
    rgb = np.uint8(
        np.clip(
            rgb
            * cv.remap(
                G.astype("f"),
                ij / 4,
                None,
                cv.INTER_LINEAR,
                borderMode=cv.BORDER_REPLICATE,
            )
            + 0.5,
            0,
            255,
        )
    )
    np.savez_compressed(
        P / (key(r) + ".npz"), box=bb, rgb=rgb, q=q.astype("f2"), sem=sem
    )


def choose(r, id, guide, old, owners, quality, colour):
    bb, rgb, q, sem = patch(key(r))
    v = guide.read(bb)
    oldmask = old.read(bb)[:, :, 0] > 0
    votes = v[:, :, 8]
    count = v[:, :, 9]
    c = r["cam"]
    valid = (q > 0) & (oldmask | ((votes >= 1.99) & (votes >= 0.63 * count)))
    boundary = (count >= 2) & (votes / count.clip(1) < 0.35)
    valid &= ~boundary
    bright = cv.cvtColor(rgb, cv.COLOR_BGR2GRAY).astype("f")
    ref = cv.cvtColor(v[:, :, :3] / np.maximum(v[:, :, 3:4], 0.001), cv.COLOR_BGR2GRAY)
    diff = blur(abs(blur(bright, 1.2) - blur(ref, 1.2)), 2)
    q = np.where(valid, q, 0)
    if c == "front":
        q = np.where((q > 0.32) | (diff < 16), q, 0)
        q = np.where(q > 0, 9.0 + q * 6, 0)
    else:
        q = q * (1.12 if c == "back" else 1.0) * np.clip(1.25 - diff * 0.015, 0.6, 1.2)
    oq = quality.read(bb)[:, :, 0]
    ids = owners.read(bb)[:, :, 0]
    im = colour.read(bb)
    sel = seam(
        rgb, im, q, oq
    )  # In overlapping front imagery enforce resolution competition even though all front views outrank side views.
    if c == "front" and (oq > 9).any():
        f = (q > 9) & (oq > 9)
        comp = seam(rgb, im, np.maximum(0, q - 9), np.maximum(0, oq - 9))
        sel[f] = comp[f]
    ids[sel] = id
    oq[sel] = q[sel]
    im[sel] = rgb[sel]
    owners.store(bb, ids[:, :, None])
    quality.store(bb, oq[:, :, None])
    colour.store(bb, im)


# Every cell of a compact marking and a pavement margin gets a single complete photographic observation, with the same camera priority as the road.
def protect(k, ids, col, obs, index):
    pad = 50
    i, j = k
    bb = (i * T - pad, j * T - pad, T + 2 * pad, T + 2 * pad)
    old = ids.read(bb)[:, :, 0]
    im = col.read(bb)
    g = cv.cvtColor(im, cv.COLOR_BGR2GRAY).astype("f")
    m = ((g - blur(g, 7)) > 9) & (old > 0)
    m = cv.morphologyEx(
        m.astype("B"),
        cv.MORPH_CLOSE,
        cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)),
    )
    m = cv.dilate(m, cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7)))
    cnt, lab, st, _ = cv.connectedComponentsWithStats(m, 8)
    fits = (
        (st[:, 4] > 10)
        & (st[:, 2] < 120)
        & (st[:, 3] < 120)
        & (st[:, 0] > 2)
        & (st[:, 1] > 2)
        & (st[:, 0] + st[:, 2] < m.shape[1] - 2)
        & (st[:, 1] + st[:, 3] < m.shape[0] - 2)
    )
    fits[0] = False
    best = np.zeros(cnt, "f")
    owner = np.zeros(cnt, "i")
    for id in sorted(set(z for K, _, _ in pieces(bb) for z in index[K])):
        row = obs[id - 1]
        box, rgb, q, sem = patch(key(row))
        x, y = max(bb[0], box[0]), max(bb[1], box[1])
        a, b = np.minimum(extent(bb), extent(box))
        if a <= x or b <= y:
            continue
        sl = np.s_[y - bb[1] : b - bb[1], x - bb[0] : a - bb[0]]
        s = np.s_[y - box[1] : b - box[1], x - box[0] : a - box[0]]
        q = q[s]
        z = lab[sl]
        zz = z[q > 0]
        num = np.bincount(zz, minlength=cnt)
        score = np.bincount(zz, weights=q[q > 0], minlength=cnt) / st[:, 4].clip(1)
        score += 10 if row["cam"] == "front" else 0
        win = fits & (num == st[:, 4]) & (score > best)
        best[win] = score[win]
        owner[win] = id
    pick = owner[lab]
    apply = (old > 0) & (pick != 0)
    old[apply] = pick[apply]
    return k, old[pad:-pad, pad:-pad].copy()


def softedge(a, v, origin, di, do):
    # Restrict coverage to actual measured cells, erode narrow dangling fingers and use a smooth ~32 cm rim.
    support = blur(v.astype("f"), 2.7)
    hard = v & (support > 0.57)
    sd = cv.distanceTransform(hard.astype("B"), cv.DIST_L2, 5)
    alpha = np.clip((sd - 0.3) / 8, 0, 1)
    return alpha * alpha * (3 - 2 * alpha)


def build(workers):
    assert W != BASE
    W.mkdir(exist_ok=True, parents=True)
    P.mkdir(exist_ok=True)
    F.mkdir(exist_ok=True)
    (W / "state").mkdir(exist_ok=True)
    (OUT / "tiles").mkdir(exist_ok=True, parents=True)
    obs = read(BASE / "observations.json")
    old = Grid(1)
    start = time.monotonic()
    for p in (BASE / "state").glob("*.npy"):
        k = tuple(map(int, p.stem.split("_")))
        a = np.load(p)
        m = cv.resize(a["ok"].astype("f"), (T, T), interpolation=cv.INTER_AREA)
        old.a[k] = (m > 0.48).astype("f")[:, :, None]
    # Initialise nominal front geometry; retain the previous measured side-camera gauge as the starting point.
    for row in obs:
        if row["cam"] == "front":
            row.pop("field", None)
            row["A"] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        row.pop("gain", None)
    guide = Grid(12)
    owners = Grid(1, "i4")
    quality = Grid(1)
    cols = Grid(3, "B")
    index = defaultdict(list)
    with ThreadPoolExecutor(workers) as ex:
        for j in range(0, len(obs), workers * 3):
            batch = obs[j : j + workers * 3]
            for ii, (bb, g) in enumerate(ex.map(lambda r: raw(r, old), batch)):
                guide.store(bb, g, True)
                for k, _, _ in pieces(bb):
                    index[k].append(j + ii + 1)
            if j % (workers * 90) == 0:
                print("measure", j, round(time.monotonic() - start), flush=True)
        for j in range(0, len(obs), workers * 3):
            list(ex.map(lambda r: fit(r, guide), obs[j : j + workers * 3]))
            if j % (workers * 90) == 0:
                print("register", j, round(time.monotonic() - start), flush=True)
        patch.cache_clear()
        dump(W / "observations.json", obs)
        # Keep adjacent exposures adjacent within each lens; hard cuts at low gradient.
        for j in sorted(
            range(len(obs)),
            key=lambda j: (obs[j]["cam"] == "front", obs[j]["cam"], obs[j]["t"]),
        ):
            choose(obs[j], j + 1, guide, old, owners, quality, cols)
        guide.a.clear()
        ks = list(owners.a)
        print("protect", len(ks), round(time.monotonic() - start), flush=True)
        canvas = Canvas()
        for k, ids in ex.map(lambda k: protect(k, owners, cols, obs, index), ks):
            a = np.zeros((N, N), D)
            keep = cv.resize(
                ids.astype("f"), (N, N), interpolation=cv.INTER_NEAREST
            ).astype("i")
            a["id"] = keep
            a["ok"] = keep > 0
            a["rgb"] = cv.resize(cols.a[k], (N, N), interpolation=cv.INTER_LINEAR)
            canvas.tiles[k] = a
            np.save(W / "state" / f"{k[0]}_{k[1]}.npy", a)
        owners.a.clear()
        cols.a.clear()
        quality.a.clear()
        print("export", len(ks), round(time.monotonic() - start), flush=True)
        tiles = [
            v
            for v in ex.map(
                lambda k: export_tile(k, canvas, obs, softedge, actual_guard), ks
            )
            if v
        ]
    meta = read(BASEOUT / "ground.json")
    meta["tiles"] = tiles
    dump(OUT / "ground.json", meta)
    dump(W / "native-mask-check.json", dict(stats))
    print(
        "ready",
        len(tiles),
        sum(v["bytes"] for v in tiles),
        dict(stats),
        round(time.monotonic() - start),
        flush=True,
    )


if __name__ == "__main__":
    import sys

    build(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
