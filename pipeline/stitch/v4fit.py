"""Joint anti-aliased raster refinement. Real low-view anchors + high-resolution chase observations; agreement-only nonroad Difix DC."""

from .v2fit import *
from .fit_raster import raster_plan
from .v4teacher import OUT, split, decode, write_scene
import sys


def generators():
    root = RS / "work/genfill"
    direc = root / "fix/chase-opt3/ref-t199"
    m = direc / "fixed.json"
    views = root / "sets/chase-opt3/views.json"
    if not m.exists() or not views.exists():
        return []
    fix = J(m)
    if not fix.get("settings", {}).get("consistency"):
        return []
    rows = fix["views"]
    full = J(views)
    vs = {e["id"]: e for e in full["views"]}
    train, h = split()
    station = sorted(train + h, key=lambda e: e["t"])
    times = np.array([e["t"] for e in station])
    out = []
    holdframes = {c: {e["n"] for e in schedule()[c] if e["hold"]} for c in CAMS}
    for name in sorted(rows):
        f = {
            k: str(direc / k / (name + ".png"))
            for k in ["fixed", "change", "agree", "class"]
        }
        r = vs[name]
        if any(not Path(v).is_file() for v in f.values()):
            return []  # only consume a complete consistency pass
        ref = rows[name].get("ref")
        if (
            station[np.abs(times - r["t"]).argmin()]["station"] % 10 == 0
            or not ref
            or ref["n"] in holdframes[ref["c"]]
        ):
            continue
        M = np.array(r["K"])
        out.append(
            dict(
                **f,
                base=r["id"],
                rgb=f["fixed"],
                T_world_cam=r["T_world_cam"],
                K=[M[0, 0], M[1, 1], M[0, 2], M[1, 2]],
                t=r["t"],
                support=str(root / "sets/chase-opt3/render" / (name + ".npz")),
            )
        )
    return out


@lru_cache(12)
def loadgen(rgb, ag, cl, ch, support):
    im = cv.imread(rgb)[:, :, ::-1]
    agree, cla, change = [cv.imread(p, cv.IMREAD_UNCHANGED) for p in [ag, cl, ch]]
    m = np.load(support)
    z = m["z"].astype("f")
    alpha = m["alpha"]
    alpha = alpha / 255 if alpha.dtype == np.uint8 else alpha
    w = (agree.astype("f") / 255) * (
        (agree >= 220) & (cla <= 1) & (change < 165) & (alpha > 0.65) & (z > 0)
    )
    return im, w, z


@lru_cache(20)
def loadt(path, w, sk, d):
    return decode(dict(rgb=path, weight=w, sky=sk, depth=d))


def window(e, mode, rng, vol):
    if mode == "real":
        args = crop(e, rng)
        if args is None:
            return None
        T, K, w, h, im, lab, o = args
        wt = (lab == 1).float() + 0.55 * (lab == 3)
        return (
            T,
            K,
            w,
            h,
            im,
            wt,
            lab == 2,
            o.get("z", torch.zeros_like(wt)),
            o.get("wt", torch.zeros_like(wt)),
            1.0,
        )
    T = np.array(e["T_world_cam"])
    K = np.array(e["K"])
    if mode == "gen":
        im, wt, z = loadgen(e["rgb"], e["agree"], e["class"], e["change"], e["support"])
        sky = np.zeros_like(wt, "B")
        base = 0.035
    else:
        im, raw, sky, d = loadt(*[e[k] for k in ["rgb", "weight", "sky", "depth"]])
        wt = raw.astype("f") / 255
        z = d.astype("f") * 0.01
        base = 1 / (1 + (e["base_distance_m"] / 7.5) ** 2)
    H, W = wt.shape
    pw = 288
    ph = 112
    small = np.s_[::8, ::8]
    good = wt[small] > 0.22
    y, x = np.nonzero(good)
    if mode == "gen" and len(x):
        zz = z[y * 8, x * 8]
        xyz = (
            T[:3, 3]
            + np.c_[
                (x * 8 + 0.5 - K[2]) / K[0] * zz, (y * 8 + 0.5 - K[3]) / K[1] * zz, zz
            ]
            @ T[:3, :3].T
        )
        elevated = xyz[:, 2] > vol.lookup(xyz)[0] + 1.15
        x, y = x[elevated], y[elevated]
    if not len(x):
        return None
    if mode == "teacher" and rng.random() < 0.45:
        # Explore elevated non-road surfaces and silhouettes, in addition to the naturally dominant pavement.
        ok = y * 8 < H * 0.50
        if rng.random() < 0.20:
            yy, xx = np.nonzero(sky[small] > 0)
            y, x = (yy, xx) if len(xx) else (y, x)
        elif ok.any():
            x, y = x[ok], y[ok]
    i = rng.integers(len(x))
    xx = int(np.clip(x[i] * 8 - rng.integers(50, pw - 50), 0, W - pw))
    yy = int(np.clip(y[i] * 8 - rng.integers(23, ph - 23), 0, H - ph))
    sl = np.s_[yy : yy + ph, xx : xx + pw]
    K[2:] -= [xx, yy]
    wgt = wt[sl].copy()
    Z = z[sl]
    if mode == "gen":
        y, x = np.mgrid[:ph, :pw]
        p = (
            T[:3, 3]
            + np.stack([(x + 0.5 - K[2]) / K[0] * Z, (y + 0.5 - K[3]) / K[1] * Z, Z], 2)
            @ T[:3, :3].T
        )
        g = vol.lookup(p.reshape(-1, 3))[0].reshape(ph, pw)
        wgt *= p[:, :, 2] > g + 1.15
    return (
        T,
        K.tolist(),
        pw,
        ph,
        gpu(im[sl].copy().astype("f") / 255),
        gpu(wgt),
        gpu(sky[sl] > 0),
        gpu(Z),
        gpu(wgt if mode == "teacher" else np.zeros_like(wgt)),
        base,
    )


class Fit:
    def __init__(self, src):
        self.meta = J(src / "map.json")
        self.rows = [reader(src / t["file"]) for t in self.meta["tiles"]]
        self.offset = np.r_[0, np.cumsum([len(a) for a in self.rows])]
        p = np.concatenate([as_param(a) for a in self.rows])
        self.idx = Index(p, False)
        self.field = gpu(p)
        self.orig = self.field[:, 3:7].clone()
        self.g = gpu(
            np.concatenate(
                [
                    np.full(len(a), t["layer"] == "ground")
                    for a, t in zip(self.rows, self.meta["tiles"])
                ]
            ),
            torch.bool,
        )
        self.m = torch.zeros_like(self.orig)
        self.v = torch.zeros_like(self.orig)
        self.cnt = torch.zeros_like(self.m[:, 0])
        self.bg = gpu([22, 26, 30]) / 255
        self.zero = gpu([0, 0, 0])
        self.vol = Volume()

    def update(self, e, mode, rng, rate):
        pack = window(e, mode, rng, self.vol)
        if pack is None:
            return False
        T, K, w, h, im, wt, sky, prior, dw, b = pack
        ii = self.idx.ids(T, K, w, h)
        if not len(ii):
            return False
        ids = gpu(ii, torch.long)
        A = self.field[ids]
        gr = self.g[ids]
        p = A[:, 3:7].detach().requires_grad_()
        alpha = torch.where(gr | (mode == "gen"), A[:, 6], p[:, 3]).sigmoid()
        z, draw = raster_plan(A, T, K, w, h, alpha)
        col = (
            torch.where(gr[:, None], A[:, 3:6], p[:, :3]) if mode == "gen" else p[:, :3]
        )
        pred = draw(col, self.bg)
        logz = z.clamp(0.25, 220).log()
        mom = draw(torch.stack([torch.ones_like(z), logz, (~gr).float()], 1), self.zero)
        cover = mom[:, :, 0].clamp_min(0.02)
        ld = mom[:, :, 1] / cover
        ob = mom[:, :, 2]
        if mode == "gen":
            wt = wt * (
                (ob.detach() > 0.8) & (mom[:, :, 0].detach() - ob.detach() < 0.035)
            )
        valid = (wt > 0.18) & (cover.detach() > 0.24)
        rgb = (pred - im).square()
        norm = wt.sum().clamp_min(0.22 * h * w)
        photo = ((rgb + 0.006**2).sqrt().mean(2) * wt).sum() / norm
        es = edge_loss(pred, im, valid)
        orig = self.orig[ids]
        anch = (p - orig).square()
        loss = b * (photo + 0.27 * es) + (
            0.0005 * anch[:, :3].mean() + 0.0002 * anch[:, 3].mean()
        )
        if mode != "gen":
            loss += 0.21 * b * (ob * sky).sum() / sky.sum().clamp_min(
                0.21 * w * h
            ) + 0.028 * b * (
                (ld - prior.clamp_min(1).log()).abs().clamp_max(0.65)
                * dw
                * (prior > 0)
                * (cover.detach() > 0.3)
            ).sum() / dw.sum().clamp_min(0.22 * w * h)
        if mode == "gen" and int(valid.sum().item()) < 140:
            return False
        loss.backward()
        G = torch.nan_to_num(p.grad)
        act = G.abs().sum(1) > 1e-11
        ix = ids[act]
        G = G[act]
        with torch.no_grad():
            count = self.cnt[ix] + 1
            m = 0.85 * self.m[ix] + 0.15 * G
            v = 0.985 * self.v[ix] + 0.015 * G * G
            delta = (
                m
                / (1 - 0.85 ** count[:, None])
                / (torch.sqrt(v / (1 - 0.985 ** count[:, None])) + 2e-8)
            )
            g = gr[act]
            new = p[act] - rate * np.clip(b, 0.035, 1) * delta * gpu(
                [0.007, 0.007, 0.007, 0.023]
            )
            anchor = self.orig[ix]
            lim = 0.165 if mode == "teacher" else 0.11
            new[:, :3] = torch.clamp(
                torch.maximum(
                    anchor[:, :3] - lim, torch.minimum(anchor[:, :3] + lim, new[:, :3])
                ),
                0,
                1,
            )
            new[:, 3] = torch.where(
                g,
                anchor[:, 3],
                torch.maximum(
                    anchor[:, 3] - 0.55, torch.minimum(anchor[:, 3] + 0.44, new[:, 3])
                ),
            )
            if mode == "gen":
                new[:, 3] = A[act, 6]
                new[:, :3] = torch.where(
                    g[:, None],
                    A[act, 3:6],
                    A[act, 3:6] + (new[:, :3] - A[act, 3:6]).clamp(-0.001, 0.001),
                )
            self.field[ix, 3:7] = new
            self.cnt[ix] = count
            self.m[ix] = m
            self.v[ix] = v
        return True

    def export(self, src, dst):
        p = self.field[:, 3:7].cpu().numpy()
        rows = []
        for i, a in enumerate(self.rows):
            A = a.copy()
            q = p[self.offset[i] : self.offset[i + 1]]
            A[:, 6:9] = (q[:, :3] - 0.5) / C0
            A[:, 9] = q[:, 3]
            rows.append(A)
        write_scene(src, dst, rows)


def fit(src, dst, steps, genuse):
    tic = time.monotonic()
    field = Fit(src)
    rng = np.random.default_rng(487)
    ts, _ = split()
    real = J(V / "train.json")
    groups = [[e for e in real if e["c"] == c] for c in CAMS]
    gens = generators() if genuse else []
    counts = dict(real=0, teacher=0, gen=0)
    n = 0
    batch = None
    while n < steps:
        if n % 8 == 0 or batch is None:
            k = rng.random()
            mode = "teacher" if k < 0.54 else "gen" if gens and k > 0.9 else "real"
            rows = (
                ts
                if mode == "teacher"
                else gens
                if mode == "gen"
                else groups[rng.integers(6)]
            )
            e = rows[rng.integers(len(rows))]
            batch = (e, mode)
        ok = field.update(*batch, rng, 0.42 + 0.58 * (1 - n / steps))
        if ok:
            counts[batch[1]] += 1
            n += 1
        else:
            batch = None
        if ok and n % 800 == 0:
            print(n, counts, round(n / (time.monotonic() - tic), 2), "it/s", flush=True)
    field.export(src, dst)
    savej(
        OUT / (dst.name + "-fit.json"),
        dict(
            steps=steps,
            updates=counts,
            gen_available=len(gens),
            seconds=time.monotonic() - tic,
        ),
    )
    print("exported", dst, counts, flush=True)


if __name__ == "__main__":
    fit(Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), "--gen" in sys.argv[4:])
