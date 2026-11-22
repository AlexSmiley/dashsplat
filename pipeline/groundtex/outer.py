"""Photographic outer support and explicit low-detail prediction; v5's declared mesh and fine tiles remain the datum."""

from .atlas import *
from .v5 import masks
from collections import Counter

B = Path("work/groundtex-v5/ground")
S = Path("work/groundtex-v5/state")
V4 = Path("work/groundtex-final/ground")
OBJ = Path("scene/stitched-opt-v5-movers-v1")
U = 32
L = M / U
R = 44
P = W / "ring"
SCHEM = {"unsupported": 0, "predicted": 127, "observed": 255}


def cells(box):
    x, y, w, h = map(int, box)
    for j in range(y // U, (y + h - 1) // U + 1):
        for i in range(x // U, (x + w - 1) // U + 1):
            X, Y = max(i * U, x), max(j * U, y)
            E, A = min(i * U + U, x + w), min(j * U + U, y + h)
            yield (
                (i, j),
                np.s_[Y - j * U : A - j * U, X - i * U : E - i * U],
                np.s_[Y - y : A - y, X - x : E - x],
            )


class Grid:
    def __init__(self, c, dt="f"):
        self.a = {}
        self.c = c
        self.dt = dt

    def blank(self, h, w):
        return np.zeros((h, w, self.c), self.dt)

    def take(self, box):
        z = self.blank(box[3], box[2])
        for k, s, v in cells(box):
            if k in self.a:
                z[v] = self.a[k][s]
        return z

    def apply(self, box, a, add=False):
        for k, s, v in cells(box):
            if k not in self.a:
                self.a[k] = self.blank(U, U)
            if add:
                self.a[k][s] += a[v]
            else:
                self.a[k][s] = a[v]

    def pad(self, k, p):
        return self.take((k[0] * U - p, k[1] * U - p, U + 2 * p, U + 2 * p))


def smooth(x):
    v = np.clip(x, 0, 1)
    return v * v * (3 - 2 * v)


@lru_cache(1)
def base():
    return {
        tuple(map(int, t["id"].split("_"))): t for t in read(B / "ground.json")["tiles"]
    }


@lru_cache(1)
def prev():
    return {
        tuple(map(int, t["id"].split("_"))): t
        for t in read(V4 / "ground.json")["tiles"]
    }


@lru_cache(1)
def pathkeys():
    tree, p, _ = height_model()
    keys = {
        tuple(k + d)
        for k in np.unique(np.floor(p[:, :2] / M).astype(int), axis=0)
        for dx in range(-5, 6)
        for dy in range(-5, 6)
        for d in [(dx, dy)]
    }
    return sorted(k for k in keys if tree.query((np.array(k) + 0.5) * M)[0] < R + 7.4)


@lru_cache(5200)
def mesh(k):
    if k in base():
        return np.array(base()[k]["grid"]["z"], "f").reshape(9, 9)
    y, x = np.mgrid[:9, :9]
    z = height(np.array(k) * M + np.stack([x, y], -1) * M / 8)
    return np.round(z, 5).astype("f")


def htile(k, x, y):
    z = mesh(k)
    x = x * 8
    y = y * 8
    i = np.floor(x).astype(int).clip(0, 7)
    j = np.floor(y).astype(int).clip(0, 7)
    u = x - i
    v = y - j
    return np.where(
        u >= v,
        (1 - u) * z[j, i] + (u - v) * z[j, i + 1] + v * z[j + 1, i + 1],
        (1 - v) * z[j, i] + (v - u) * z[j + 1, i] + u * z[j + 1, i + 1],
    )


def boxxy(box):
    x, y, w, h = box
    Y, X = np.mgrid[:h, :w]
    return (np.stack([X + x, Y + y], -1).astype("f") + 0.5) * L


@lru_cache(1)
def surface():
    H = Grid(2)
    y, x = np.mgrid[:U, :U]
    xx = (x + 0.5) / U
    yy = (y + 0.5) / U
    tree = height_model()[0]
    for k in pathkeys():
        xy = (np.stack([xx, yy], -1) + k) * M
        H.a[k] = np.dstack([htile(k, xx, yy), tree.query(xy)[0]]).astype("f")
    return H


def listrows():
    old = read(S / "observations.json")
    index = {(r["cam"], r["n"]): r for r in old}
    rows = []
    hold = read("work/stitch-v2/exposures.json")
    tr = inputs()[0]
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(tr.p[:, :2], axis=0), axis=1))]
    for c in CAMS:
        excluded = {r["n"] for r in hold[c] if r["hold"]}
        pool = {}
        ds = (
            2.5
            if c in ["left_pillar", "right_pillar", "left_repeater", "right_repeater"]
            else 3.0
        )
        for n, e in read(Path("depth") / c / "index.json").items():
            n = int(n)
            t = e["t"]
            if (
                n % 29 == 7
                or n in excluded
                or any(n not in files(c, v) for v in ["remove", "road", "sky", "dyn"])
            ):
                continue
            dist = np.interp(t, tr.t, arc)
            i = int(dist / ds)
            priority = 0.22 * ((c, n) in index) - abs(dist / ds - i - 0.5)
            if i not in pool or priority > pool[i][0]:
                pool[i] = (priority, dict(cam=c, n=n, t=t))
        rows += sorted((v[1] for v in pool.values()), key=lambda r: r["t"])
    return rows


def footprint6(r):
    c = inputs()[1][r["cam"]]
    T = inputs()[0].T_world_ego(r["t"]) @ c.T_ego_cam
    ey = T[:2, 3]
    f = T[:2, 2]
    f /= np.linalg.norm(f)
    s = np.array([-f[1], f[0]])
    ang = (
        np.linspace(-1.65, 1.65, 19)
        if c.name != "front"
        else np.linspace(-0.72, 0.72, 13)
    )
    v = np.vstack([ey, ey + 46 * (np.cos(ang)[:, None] * f + np.sin(ang)[:, None] * s)])
    b = np.floor(v.min(0) / L).astype(int) - 2
    B0 = np.ceil(v.max(0) / L).astype(int) + 2
    return tuple(map(int, [*b, *(B0 - b)]))


@lru_cache(256)
def rowpose(c, t):
    cam = inputs()[1][c]
    rr = np.linspace(0, cam.height, 17)
    return (inputs()[0].T_world_ego(cam.row_time(t, rr)) @ cam.T_ego_cam).astype("f")


def project(r, xy, z):
    cam = inputs()[1][r["cam"]]
    T = rowpose(r["cam"], r["t"])
    xyz = np.dstack([xy, z]).astype("f")

    def p(q):
        return cam.project(q)

    q = (xyz - T[8, :3, 3]) @ T[8, :3, :3]
    uv, ok = p(q)
    j = np.nan_to_num(uv[:, :, 1] * 16 / cam.height).clip(0, 15.999)
    i = j.astype("i")
    w = (j - i).astype("f")
    t = T[i] + (T[i + 1] - T[i]) * w[:, :, None, None]
    q = np.einsum("...i,...ij->...j", xyz - t[:, :, :3, 3], t[:, :, :3, :3])
    uv, ok = p(q)
    return uv.astype("f"), ok, q[:, :, 2], np.linalg.norm(xyz - t[:, :, :3, 3], axis=2)


@lru_cache(20)
def depth(c, n):
    return np.load(Path("depth") / c / f"{n:06d}.npy").astype("f")


def groundvalid(r, uv, ok, zcam, dist, z, xy):
    c = r["cam"]
    cam = inputs()[1][c]
    mm = masks(c, r["n"])
    D = sam(depth(c, r["n"]), uv, cam, True)
    ok &= (D > 0) & (zcam > 0.1) & (dist < 46)
    ratio = D / np.maximum(zcam, 0.1)
    eye = rowpose(c, r["t"])[8, :3, 3]
    above = (eye[2] - z) * (1 - ratio)
    sem = sam(mm["road"], uv, cam, True) > 0
    valid = (
        ok & (abs(above) < np.where(sem, 0.30, 0.18)) & (ratio > 0.78) & (ratio < 1.23)
    )
    for name in ["remove", "dyn", "sky"]:
        m = mm[name]
        if m.any():
            valid &= sam(cv.dilate(m, np.ones((3, 3), "B")), uv, cam, True) == 0
    return valid, sem, above


def measure(item):
    id, r = item
    box = footprint6(r)
    xy = boxxy(box)
    h = surface().take(box)
    z = h[:, :, 0]
    uv, ok, zc, d = project(r, xy, z)
    v, sem, e = groundvalid(r, uv, ok, zc, d, z, xy)
    v &= (h[:, :, 1] < R + 1) & (h[:, :, 1] > 0)
    rgb = photo(
        r, uv, True
    )  # Depth residual slope detects vertical surfaces and uncertain silhouettes.
    slope = np.hypot(np.gradient(e, axis=0), np.gradient(e, axis=1)) / L
    v &= slope < np.where(sem, 0.46, 0.29)
    v = cv.erode(v.astype("B"), np.ones((3, 3), "B")).astype(bool)
    dx = np.gradient(uv, axis=1) / L
    dy = np.gradient(uv, axis=0) / L
    a = (dx * dx).sum(2)
    b = (dy * dy).sum(2)
    c = (dx * dy).sum(2)
    res = np.sqrt(np.maximum(0, (a + b - np.hypot(a - b, 2 * c)) / 2))
    q = np.where(
        v, np.clip(res / 28, 0.06, 2.2) ** 0.52 * np.clip(1 - abs(e) * 1.8, 0.42, 1), 0
    )
    y, x = np.nonzero(q > 0)
    if not len(x):
        np.savez_compressed(
            P / f"{id}.npz",
            box=(0, 0, 1, 1),
            im=np.zeros((1, 1, 3), "B"),
            q=np.zeros((1, 1), "f2"),
            sem=np.zeros((1, 1), "B"),
        )
        return None
    sl = np.s_[y.min() : y.max() + 1, x.min() : x.max() + 1]
    box = (
        box[0] + int(x.min()),
        box[1] + int(y.min()),
        int(np.ptp(x)) + 1,
        int(np.ptp(y)) + 1,
    )
    im = rgb[sl]
    q = q[sl].astype("f")
    sem = sem[sl].astype("B")
    np.savez_compressed(P / f"{id}.npz", box=box, im=im, q=q.astype("f2"), sem=sem)
    return box, np.dstack(
        [im * q[:, :, None], q, (q > 0).astype("f"), sem * (q > 0)]
    ).astype("f")


def objects():
    g = Grid(1)
    types = Counter()
    for e in read(OBJ / "map.json")["tiles"]:
        if e["layer"] != "objects" or e["id"].startswith("traffic_"):
            continue
        a = reader(OBJ / e["file"])
        p = a[:, :3]
        key = np.floor(p[:, :2] / L).astype("i")
        ks = key // U
        enc = (ks[:, 0] + 3000) * 7000 + ks[:, 1] + 3000
        ix = np.argsort(enc)
        edges = np.r_[0, np.flatnonzero(np.diff(enc[ix])) + 1, len(ix)]
        for l, h in zip(edges[:-1], edges[1:]):
            ii = ix[l:h]
            k = tuple(ks[ii[0]])
            if k not in surface().a:
                continue
            xy = p[ii, :2] / M - np.array(k)
            z = htile(k, xy[:, 0], xy[:, 1])
            dz = p[ii, 2] - z
            sel = (dz > 0.25) & (dz < 1.75) & (np.abs(a[ii, 5]) < 0.88)
            ij = key[ii[sel]] - np.array(k) * U
            if len(ij):
                if k not in g.a:
                    g.a[k] = g.blank(U, U)
                np.add.at(g.a[k][:, :, 0], (ij[:, 1], ij[:, 0]), 1)
                types[e["id"].split("_")[0]] += len(ij)
    print("object support", types, flush=True)
    return g


def cached(id):
    with np.load(P / f"{id}.npz") as z:
        return tuple(map(int, z["box"])), z["im"], z["q"].astype("f"), z["sem"]


def runs(workers):
    W.mkdir(parents=True, exist_ok=True)
    P.mkdir(exist_ok=True)
    (W / "ctrl").mkdir(exist_ok=True)
    (W / "map").mkdir(exist_ok=True)
    rows = listrows()
    dump(W / "outer-rows.json", rows)
    surface()
    guide = Grid(6)
    old = base()
    print("world", len(pathkeys()), "photos", len(rows), flush=True)
    with ThreadPoolExecutor(workers) as ex:
        for j in range(0, len(rows), workers * 4):
            for z in ex.map(measure, enumerate(rows[j : j + workers * 4], j + 1)):
                if z:
                    guide.apply(*z, add=True)
            if j % (workers * 100) == 0:
                print("evidence", j, flush=True)
    G = Grid(3)
    oldcol = Grid(4)
    O = objects()
    for k, t in old.items():
        g = t["gutter"]
        s = t["pixels"]
        im = np.asarray(Image.open(B / t["texture"]).convert("RGBA"))[
            g : g + s, g : g + s
        ].astype("f")
        a = im[:, :, 3] / 255
        v = cv.resize(
            np.dstack([im[:, :, :3][:, :, ::-1] * a[:, :, None], a]),
            (U, U),
            interpolation=cv.INTER_AREA,
        )
        oldcol.a[k] = v
    for k, v in guide.a.items():
        o = oldcol.a.get(k)
        avg = v[:, :, :3] / v[:, :, 3:4].clip(0.001)
        G.a[k] = (
            avg
            if o is None
            else np.where(
                o[:, :, 3:4] > 0.89, o[:, :, :3] / o[:, :, 3:4].clip(0.001), avg
            )
        )
    owner = Grid(1, "i")
    qual = Grid(1)
    colour = Grid(3, "B")
    quality = []
    for id, r in enumerate(rows, 1):
        box, im, q, sem = cached(id)
        h, w = q.shape
        if h < 6 or w < 6:
            continue
        g = G.take(box)
        evidence = guide.take(box)
        obj = O.take(box)[
            :, :, 0
        ]  # Coarse projected features guide offsets, never exported radiance.
        gray = lambda b: cv.cvtColor(b.astype("f"), cv.COLOR_BGR2GRAY)
        a = gray(g)
        b = gray(im)
        a -= cv.GaussianBlur(a, (0, 0), 3)
        b -= cv.GaussianBlur(b, (0, 0), 3)
        Y, X = np.mgrid[:h, :w].astype("f")
        uv0 = np.stack([X, Y], -1)
        ctrl = np.zeros(((h + 3) // 4, (w + 3) // 4, 3), "f")
        m = (q > 0) & (evidence[:, :, 4] >= 2)
        patchcount = 0
        for y in range(8, h - 8, 8):
            for x in range(8, w - 8, 8):
                sl = np.s_[y - 4 : y + 5, x - 4 : x + 5]
                tpl = a[sl]
                if m[sl].mean() < 0.92 or tpl.std() < 2.8:
                    continue
                cc = cv.matchTemplate(
                    b[y - 7 : y + 8, x - 7 : x + 8], tpl, cv.TM_CCOEFF_NORMED
                )
                v, u = np.unravel_index(cc.argmax(), cc.shape)
                val = cc[v, u]
                if val < 0.75 or min(u, v) == 0 or max(u, v) == 6:
                    continue

                def sub(a, b, c):
                    return float(
                        np.clip(0.5 * (a - c) / min(-1e-5, a - 2 * b + c), -0.5, 0.5)
                    )

                d = np.array(
                    [
                        u - 3 + sub(cc[v, u - 1], val, cc[v, u + 1]),
                        v - 3 + sub(cc[v - 1, u], val, cc[v + 1, u]),
                    ]
                )
                wt = (val - 0.5) ** 2
                ctrl[y // 4, x // 4] = [*d * wt, wt]
                patchcount += 1
        ctrl = cv.GaussianBlur(ctrl, (0, 0), 1.2)
        delta = ctrl[:, :, :2] / (ctrl[:, :, 2:] + 0.018)
        mp = uv0 + cv.remap(
            delta, uv0 / 4, None, cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE
        )
        im = cv.remap(im, mp, None, cv.INTER_LINEAR)
        q = cv.remap(q, mp, None, cv.INTER_NEAREST)
        sem = cv.remap(sem, mp, None, cv.INTER_NEAREST)
        v = (q > 0) & (evidence[:, :, 4] >= 2)
        diff = np.abs(im.astype("f") - g).mean(2)
        good = (
            v
            & (diff < 32)
            & (np.abs(gray(im) - cv.GaussianBlur(gray(im), (0, 0), 2)) < 12)
        )
        ww = good.astype("f")
        gain = cv.GaussianBlur(
            np.log(np.maximum(g, 12) / np.maximum(im, 12)).clip(-0.24, 0.24)
            * ww[:, :, None],
            (0, 0),
            4,
        ) / (cv.GaussianBlur(ww, (0, 0), 4)[:, :, None] + 0.04)
        gain = np.exp(gain)
        im = np.uint8((im * gain).clip(0, 255))
        dif = cv.GaussianBlur(np.abs(im.astype("f") - g).mean(2), (0, 0), 1.1)
        v &= (dif < np.where(sem, 29, 21)) & (obj < 2)
        q = np.where(v, q * np.clip(1 - dif / 60, 0.45, 1), 0)
        oq = qual.take(box)[:, :, 0]
        ids = owner.take(box)[:, :, 0]
        rgb = colour.take(box)
        sel = (q > 0) & (q > oq * 1.10)
        ids[sel] = id
        oq[sel] = q[sel]
        rgb[sel] = im[sel]
        owner.apply(box, ids[:, :, None])
        qual.apply(box, oq[:, :, None])
        colour.apply(box, rgb)
        np.savez_compressed(
            W / "ctrl" / f"{id}.npz",
            box=box,
            d=(delta * L).astype("f2"),
            gain=gain[::4, ::4].astype("f2"),
        )
        quality.append(patchcount)
        if id % 600 == 0:
            print("registered", id, sum(quality), flush=True)
    for k in pathkeys():
        o = owner.pad(k, 0)[:, :, 0]
        col = colour.pad(k, 0)
        b = oldcol.pad(k, 0)
        v = guide.pad(k, 0)
        obj = O.pad(k, 2)[:, :, 0]
        hard = cv.GaussianBlur(np.minimum(obj, 8), (0, 0), 0.8) > 1.85
        hard = hard[2:-2, 2:-2]
        hard &= (v[:, :, 4] < 2) | (obj[2:-2, 2:-2] >= 4)
        hard &= b[:, :, 3] < 0.7
        o[hard] = 0
        np.savez_compressed(
            W / "map" / f"{k[0]}_{k[1]}.npz",
            id=o,
            rgb=col,
            prior=b.astype("f2"),
            object=hard.astype("B"),
            dist=surface().a[k][:, :, 1].astype("f2"),
            votes=v[:, :, 4].astype("f2"),
            road=v[:, :, 5].astype("f2"),
        )
    dump(
        W / "registration.json",
        dict(
            exposures=len(rows),
            matches=sum(quality),
            constrained_frames=sum(v > 0 for v in quality),
            pitch_m=L,
            tiles=len(pathkeys()),
        ),
    )
    print("evidence ready", flush=True)


if __name__ == "__main__":
    import sys

    runs(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
