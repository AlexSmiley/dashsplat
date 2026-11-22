"""Raster fit only object splats, on native-camera crops and elevated confidence-weighted chase pixels."""

from .v5world import *
from . import v4fit as joint
from .model import gpu, torch


@lru_cache(10)
def native_image(c, n):
    a = cv.imread(str(RS / "data/drive/frames" / c / f"{n:06}.jpg"))
    return photo.apply(c, n, a[:, :, ::-1]), labels(c, n)


def physical_patch(e, rng, klass=None):
    c, n = e["c"], e["n"]
    C = rig[c]
    im, lab = native_image(c, n)
    mask = (
        np.isin(lab, [2, 5, 6, 7, 8, 13, 14, 15, 17]) if klass is None else lab == klass
    )
    yy, xx = np.nonzero(mask[::3, ::3])
    if not len(xx):
        return None
    j = rng.integers(len(xx))
    X, Y = int(xx[j] * 3), int(yy[j] * 3)
    ray = C.unproject(np.array([(X + 0.5) * C.width / 724, (Y + 0.5) * C.height / 469]))
    if not np.isfinite(ray).all():
        return None
    right = unit(np.cross([0, 1, 0], ray))
    down = np.cross(ray, right)
    R = np.stack([right, down, ray], 1)
    T = poses(c, e["t"])[int(Y * 8 / 469)].copy()
    T[:3, :3] = T[:3, :3] @ R
    w, h = 288, 112
    f = C.params[0]
    y, x = np.mgrid[:h, :w]
    uv, ok = C.project(
        np.stack([(x + 0.5 - w / 2) / f, (y + 0.5 - h / 2) / f, np.ones((h, w))], 2)
        @ R.T
    )
    mx = (uv[:, :, 0] - 0.5).astype("f")
    my = (uv[:, :, 1] - 0.5).astype("f")
    pic = cv.remap(im, mx, my, cv.INTER_LINEAR)
    L = cv.remap(
        lab,
        ((mx + 0.5) * 724 / C.width - 0.5).astype("f"),
        ((my + 0.5) * 469 / C.height - 0.5).astype("f"),
        0,
        borderMode=cv.BORDER_CONSTANT,
        borderValue=250,
    )
    L[~ok] = 250
    return T, [f, f, w / 2, h / 2], pic, L


def window(e, mode, rng, vol):
    if mode == "real":
        patch = physical_patch(e, rng)
        if patch is None:
            return None
        T, K, pic, L = patch
        h, w = L.shape
        wt = np.isin(L, [2, 5, 6, 7, 8, 13, 14, 15, 17])
        zero = gpu(np.zeros((h, w), "f"))
        return (
            T,
            K,
            w,
            h,
            gpu(pic.astype("f") / 255),
            gpu(wt.astype("f")),
            gpu(L == 10),
            zero,
            zero,
            1.0,
        )
    if mode == "gen":
        return joint.window_v4(e, mode, rng, vol)
    im, raw, sky, dep = joint.loadt(*[e[k] for k in ["rgb", "weight", "sky", "depth"]])
    w, h = 288, 112
    Z = dep.astype("f") * 0.01
    T = np.array(e["T_world_cam"])
    K = np.array(e["K"], "d")
    H, W = Z.shape

    def elevation(x, y):
        z = Z[y, x]
        P = (
            T[:3, 3]
            + np.stack(
                [(x + 0.5 - K[2]) * z / K[0], (y + 0.5 - K[3]) * z / K[1], z], -1
            )
            @ T[:3, :3].T
        )
        return (z > 0) & (
            P[..., 2] > vol.lookup(P.reshape(-1, 3))[0].reshape(z.shape) + 0.45
        )

    y, x = np.nonzero(raw[::8, ::8] > 120)
    x *= 8
    y *= 8
    m = elevation(x, y)
    x, y = x[m], y[m]
    if not len(x):
        return None
    j = rng.integers(len(x))
    xx = int(np.clip(x[j] - w // 2, 0, W - w))
    yy = int(np.clip(y[j] - h // 2, 0, H - h))
    y, x = np.mgrid[yy : yy + h, xx : xx + w]
    wt = raw[y, x].astype("f") / 255 * elevation(x, y)
    wt[wt < 0.45] = 0
    K[2:] -= [xx, yy]
    return (
        T,
        K,
        w,
        h,
        gpu(im[y, x].astype("f") / 255),
        gpu(wt),
        gpu(sky[y, x] > 0),
        gpu(Z[y, x]),
        gpu(wt),
        1 / (1 + (e["base_distance_m"] / 7.5) ** 2),
    )


def fit(src, steps):
    view = W5 / "objects-input"
    view.mkdir()
    (view / "tiles").mkdir()
    m = J(src / "map.json")
    m["tiles"] = [t for t in m["tiles"] if t["layer"] == "objects"]
    savej(view / "map.json", m)
    for t in m["tiles"]:
        os.link(src / t["file"], view / t["file"])
    F = joint.Fit(view)
    joint.window_v4 = joint.window
    joint.window = window
    ts, held = split()
    gs = joint.generators() if BENCHMARK else []
    reals = [
        e
        for c in CAMS
        for e in physical_schedule()[c]
        if not e["hold"] and semantic(c, e["n"])
    ]
    rng = np.random.default_rng(503)
    counts = {k: 0 for k in ["real", "teacher", "gen"]}
    batch = None
    i = 0
    attempt = 0
    while i < steps and attempt < steps * 8:
        attempt += 1
        if batch is None or i % 6 == 0:
            u = rng.random()
            mode = "real" if u < 0.54 else ("gen" if gs and u > 0.89 else "teacher")
            rows = {"real": reals, "teacher": ts, "gen": gs}[mode]
            batch = rows[rng.integers(len(rows))], mode
        if F.update(*batch, rng, 0.52 + 0.36 * (1 - i / steps)):
            counts[batch[1]] += 1
            i += 1
            if i % 600 == 0:
                print(i, counts, flush=True)
        else:
            batch = None
    assert i == steps
    pars = F.field[:, 3:7].cpu().numpy()
    parts = []
    for j, (t, A) in enumerate(zip(F.meta["tiles"], F.rows)):
        a = A.copy()
        p = pars[F.offset[j] : F.offset[j + 1]]
        a[:, 6:9] = (p[:, :3] - 0.5) / C0
        a[:, 9] = p[:, 3]
        parts.append((t, a))
    assemble(W5 / "fit", src, parts)
    savej(
        W5 / "fit.json",
        dict(
            updates=counts,
            native_crops=[288, 112],
            gen_available=len(gs),
            held_stations=len(set(e["station"] for e in held)),
        ),
    )
    print(counts, flush=True)


if __name__ == "__main__":
    fit(Path(sys.argv[1]), int(sys.argv[2]))
