"""Photograph and depth supervision with optional weak, weighted, pinhole pseudo-views."""

from .v2field import *
import time, resource, argparse
from scipy.spatial import cKDTree


@lru_cache(96)
def load(p):
    with np.load(p) as a:
        return a["rgb"], a["lab"], a["z"], a["wt"]


@lru_cache(32)
def pseudofile(rgb_path, weight_path):
    im = cv.imread(rgb_path, cv.IMREAD_COLOR)
    assert im is not None
    wt = (
        np.load(weight_path)
        if weight_path.endswith(".npy")
        else cv.imread(weight_path, cv.IMREAD_UNCHANGED)
    )
    assert wt is not None and wt.ndim == 2 and wt.shape == im.shape[:2]
    wt = wt.astype("f") / (
        255 if wt.dtype == np.uint8 else 65535 if wt.dtype == np.uint16 else 1
    )
    assert np.isfinite(wt).all() and wt.min() >= 0 and wt.max() <= 1
    return im[:, :, ::-1], wt


def crop(e, rng, pseudo=False):
    if pseudo:
        im, wgt = pseudofile(e["rgb"], e["weight"])
        hh, ww = im.shape[:2]
        y, x = np.where(wgt[::6, ::6] > 0.001)
        lab = np.ones((hh, ww), "B")
        fx, fy, cx, cy = e["K"]
        T = np.array(e["T_world_cam"], dtype="d")
        z = wt = None
        b = 0
    else:
        c, n, yaw = e["c"], e["n"], e["yaw"]
        ww, hh, f, R, _, _, band, _ = pin(c, yaw)
        im, lab, z, wt = load(e["file"])
        sam = rng.random()
        lbl = 1 if sam < 0.80 else 2 if sam < 0.91 else 3
        y, x = np.where(lab[::6, ::6] == lbl)
        if not len(y):
            y, x = np.where(lab[::6, ::6] > 0)
        if not len(y):
            return None
        i = rng.integers(len(y))
        b = band[y[i] * 6, x[i] * 6]
        T = poses(c, e["t"])[b].copy()
        T[:3, :3] = T[:3, :3] @ R
        fx = fy = f
        cx = ww / 2
        cy = hh / 2
        wgt = None
    if not len(y):
        return None
    if pseudo:
        i = rng.integers(len(y))
    w = min(320, ww)
    h = min(96 if ww > 800 else 64, hh)
    x0 = int(np.clip(x[i] * 6 - rng.integers(w // 4, 3 * w // 4), 0, ww - w))
    y0 = int(np.clip(y[i] * 6 - rng.integers(h // 4, 3 * h // 4), 0, hh - h))
    sl = np.s_[y0 : y0 + h, x0 : x0 + w]
    target = im[sl]
    mask = lab[sl] if pseudo else np.where(band[sl] == b, lab[sl], 0)
    opt = {}
    if pseudo:
        opt["weak"] = gpu(wgt[sl])
    elif z.size:
        opt = {"z": gpu(z[sl].astype("f")), "wt": gpu(wt[sl].astype("f") / 255)}
    return (
        T,
        [fx, fy, cx - x0, cy - y0],
        w,
        h,
        gpu(target.astype("f") / 255),
        gpu(mask),
        opt,
    )


def av(x, w):
    return (x * w).sum() / w.sum().clamp_min(1)


def edge_loss(im, target, mask):
    """Finite spatial differences avoid the unstable pooled SSIM adjoint on MPS."""
    total = 0.0
    for dy, dx in [(0, 1), (1, 0), (0, 3), (3, 0)]:
        a = (slice(dy, None), slice(dx, None))
        b = (slice(None, -dy or None), slice(None, -dx or None))
        res = (im[a] - im[b]) - (target[a] - target[b])
        m = mask[a] & mask[b]
        total += av(res.square().add(0.008**2).sqrt().mean(2), m)
    return total * 0.25


def step(s, e, rng, gains, factor, pseudo=False):
    args = crop(e, rng, pseudo)
    if args is None:
        return None
    T, K, w, h, target, mask, opt = args
    ids, d, im, mom, depth, trans, u, valid, zd, rad, q = s.render(T, K, w, h)
    n = len(ids)
    if not n:
        return None
    static = mask == 1
    sky = mask == 2
    road = mask == 3
    validpix = mask > 0
    gain = gpu(0 if pseudo else gains.get((e["c"], e["n"]), 0.0)).requires_grad_(
        not pseudo
    )
    pred = im * gain.exp()
    rgb_err = (pred - target).square().add(0.012**2).sqrt().mean(2)
    op = mom[:, :, 0].clamp_min(0.025)
    logz = mom[:, :, 1] / op
    variance = (mom[:, :, 2] / op - logz.square()).clamp_min(0)
    zexp = depth / (1 - trans).clamp_min(0.025)
    iu = u[:n].detach()
    x = iu[:, 0].long().clamp(0, w - 1)
    y = iu[:, 1].long().clamp(0, h - 1)
    inside = (
        valid[:n].bool()
        & (iu[:, 0] >= 0)
        & (iu[:, 0] < w)
        & (iu[:, 1] >= 0)
        & (iu[:, 1] < h)
    )
    lab = mask[y, x]
    inlab = inside & (lab > 0)
    near = (zd[:n] < zexp[y, x] + 0.5 + zd[:n] * 0.09).detach()
    peak = d[:, 6].sigmoid()
    vis = inlab & near & (lab == 1)
    free = inlab & (lab == 2)
    sig = vis.float()
    loss = (
        av(rgb_err, static.float() + 0.16 * road)
        + 0.22 * edge_loss(pred, target, static)
        + 1.8 * gain.square().sum()
    )
    if pseudo:
        # Keep absolute confidence: a uniform 1% weight must not become a full-strength view.
        loss = 0.10 * (rgb_err * opt["weak"]).mean()
        free = torch.zeros_like(free)
        sig = sig * 0
    else:
        loss = (
            loss
            + 0.27
            * (mom[:, :, 0] * sky).sum()
            / torch.maximum(sky.sum(), 0.15 * validpix.sum()).clamp_min(1)
            + 0.032 * av(variance.clamp_max(1.0), static * (op.detach() > 0.15))
            + 0.025 * av(peak, inlab & (lab == 2))
        )
        if "z" in opt:
            z = opt["z"]
            weight = opt["wt"] * static * (op.detach() > 0.10)
            dl = logz - z.clamp_min(1).log()
            close = z < 42
            far = weight * (~close)
            far_mean = av(dl, far)
            loss = (
                loss
                + 0.12 * av(dl.abs().clamp_max(1.5), weight * close)
                + 0.035 * av((dl - far_mean).abs().clamp_max(0.8), far)
            )
            zz = z[y, x]
            ww = opt["wt"][y, x]
            thresh = 0.35 + 0.14 * zz
            through = inlab & (lab == 1) & (ww > 0.10) & (zz > zd[:n].detach() + thresh)
            free |= through
            loss = (
                loss
                + 0.055 * av(peak, through * ww)
                + 0.035
                * av(
                    (zd[:n].clamp_min(0.25).log() - zz.clamp_min(1).log())
                    .abs()
                    .clamp_max(1),
                    vis * ww * (zz > 1),
                )
            )
            dx = (z[:, 2:] - z[:, :-2]) * 0.5
            dy = (z[2:, :] - z[:-2, :]) * 0.5
            dx = F.pad(dx, (1, 1))
            dy = F.pad(dy, (0, 0, 1, 1))
            fx, fy, cx, cy = K
            nx = -fx * dx[y, x]
            ny = -fy * dy[y, x]
            nz = zz + (x + 0.5 - cx) * dx[y, x] + (y + 0.5 - cy) * dy[y, x]
            normcam = F.normalize(torch.stack([nx, ny, nz], 1), dim=1)
            wn = normcam @ gpu(T[:3, :3].T)
            axis = normal(q[:n])
            flat = (
                vis
                & (abs(zz - zd[:n].detach()) < thresh)
                & (dx[y, x].abs() + dy[y, x].abs() < 0.012 * zz)
                & (zz > 1)
                & (ww > 0.15)
            )
            loss = (
                loss
                + 0.016 * av(1 - (wn * axis).sum(1).abs(), flat * ww)
                + 0.006
                * av((d[:, 9] - d[:, 7:9].mean(1) + 0.6).clamp_min(0), flat * ww)
            )
        # Photograph sees road at the ground-plane intersection; only clear objects in front of it, not trees behind the surface.
        gp = s.floor(d[:, :3].detach())
        above = d[:, 2].detach() - gp
        road_bad = inlab & (lab == 3) & near & (above > 0.18)
        free |= road_bad
        loss = loss + 0.09 * av(peak, road_bad) + 0.004 * av(peak * (1 - peak), inlab)
    loss.backward()
    if not pseudo:
        gains[(e["c"], e["n"])] = float(
            np.clip(gain.detach().item() - 0.035 * gain.grad.item(), -0.045, 0.045)
        )
    screen_grad = u.grad[:n].detach().norm(dim=1)
    ev = torch.stack(
        [sig, sig * rgb_err.detach()[y, x], free.float(), sig * screen_grad], 1
    )
    updated = s.update(ids, d, ev, factor)
    return len(ids), updated, float(loss.item()), int(static.sum().item())


def compact(src):
    dst = V / "packed"
    if dst.exists():
        return dst
    a = np.concatenate(
        [
            reader(src / t["file"])
            for t in J(src / "map.json")["tiles"]
            if t["layer"] == "objects"
        ]
    )
    before = len(a)
    route = traj.T_world_ego(np.arange(START, END, 0.15))[:, :2, 3]
    lat = cKDTree(route).query(a[:, :2])[0]
    level = np.digitize(lat, [12, 29, 59, 105])
    scale = 1.0
    while len(a) > 3650000:
        scale *= 1.16
        pitch = np.array([0.13, 0.26, 0.53, 1.06, 2.1])[level] * scale
        cell = np.floor(a[:, :3] / pitch[:, None]).astype("i8") + 16000
        key = ((level * 32768 + cell[:, 0]) * 32768 + cell[:, 1]) * 32768 + cell[:, 2]
        o = np.lexsort((-a[:, 9], key))
        ii = o[np.r_[True, np.diff(key[o]) != 0]]
        a = a[ii]
        level = level[ii]
    scene(dst, a)
    savej(V / "compact.json", dict(before=before, after=len(a), pitch_factor=scale))
    return dst


def run(args):
    tic = time.monotonic()
    src = Path(args.source) if args.source else compact(V / "dense-prior")
    s = Field(src)
    views = J(V / "train.json")
    rng = np.random.default_rng(548)
    pseudo = J(Path(args.pseudo)) if args.pseudo else []
    gains = {}
    records = []
    elapsed = time.monotonic()
    peak = 0
    steps = args.steps
    # Equal distance bins, with deliberate oversampling of the twisting exit. Draw the six physical lenses evenly (not rear x3).
    strata = {}
    bounds = [8, 38, 84, 180, 290, 400, 520, 620]
    for e in views:
        strata.setdefault((e["c"], int(np.searchsorted(bounds, e["odo"]))), []).append(
            e
        )
    cells = list(strata)
    pw = np.array([1.55 if j in (1, 2) else 1.0 for c, j in cells])
    pw /= pw.sum()
    while len(records) < steps:
        k = len(records)
        is_p = bool(pseudo) and k % 16 == 15
        e = pseudo[rng.integers(len(pseudo))] if is_p else None
        if e is None:
            group = strata[cells[rng.choice(len(cells), p=pw)]]
            e = group[rng.integers(len(group))]
        ret = step(s, e, rng, gains, 0.40 + 0.60 * (1 - k / steps), is_p)
        if ret is None:
            continue
        records.append(ret)
        n = len(records)
        peak = max(peak, torch.mps.driver_allocated_memory())
        if n in [steps // 3, steps * 2 // 3] and steps > 300:
            print("topology", s.grow(), flush=True)
            s.export(V / f"{Path(args.out).name}-checkpoint-{n}")
        elif n % 640 == 0:
            s.reindex()
        if n % 300 == 0:
            print(
                n,
                round(n / (time.monotonic() - elapsed), 2),
                "it/s",
                np.round(np.mean(records[-300:], 0), 3),
                flush=True,
            )
    s.grow(final=True)
    fitsec = time.monotonic() - elapsed
    result = s.export(Path(args.out))
    result.update(
        source=str(src),
        steps=steps,
        seconds=fitsec,
        it_s=steps / fitsec,
        peak_mps=peak,
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        physical_exposures=len(gains),
        train_views=len(views),
        mean=np.mean(records, 0).tolist(),
        pseudo_views=len(pseudo),
        total_s=time.monotonic() - tic,
    )
    savej(V / (Path(args.out).name + "-fit.json"), result)
    print(result, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source")
    ap.add_argument("--out", default=str(RS / "scene/stitched-opt-v2"))
    ap.add_argument("--steps", type=int, default=14400)
    ap.add_argument("--pseudo")
    run(ap.parse_args())
