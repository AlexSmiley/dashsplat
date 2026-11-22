"""Sample a physical-row shutter band, differentiate the composited RGB, sparse bounded Adam."""

from .model import *
import argparse, resource


@lru_cache(160)
def load(p):
    with np.load(p) as a:
        return a["rgb"], a["mask"], a["z"]


def step(scene, e, b, gains):
    c, y, t = e["c"], e["yaw"], e["t"]
    K, R, _, _, band, bounds = design(c, y)
    lo, hi = bounds[b]
    if hi <= lo:
        return None
    rgb, labels, z = load(path(e))
    lab = np.where(band[lo:hi] == b, labels[lo:hi], 0)
    area = int((lab == 1).sum())
    skyarea = int((lab == 2).sum())
    if area < 60 and skyarea < 80:
        return None
    T = world(c, y, t, b)
    params = [K[0, 0], K[1, 1], W / 2, H / 2 - lo]
    ix, d, base, o, model, cam, aux = scene.packet(T, params, W, hi - lo, True)
    key = c + ":" + str(e["n"])
    ex = gpu(gains.get(key, 0.0)).requires_grad_(True)
    target = gpu(rgb[lo:hi] / 255.0)
    mask = gpu(lab)
    pred = aux.image * ex.exp()
    cover = (1 - aux.final_T).detach()
    wm = (mask == 1) * (cover > 0.025)
    res = pred - target
    loss = (
        torch.sqrt(res.square() + 0.015**2).mean(2) * wm
    ).sum() / wm.sum().clamp_min(1) + 1.6 * ex.square().sum()
    # Centroid samples only supervise opacity/geometry if supported in this row band.
    u = aux.means2d.detach()
    xx = u[:, 0].long().clamp(0, W - 1)
    yy = u[:, 1].long().clamp(0, hi - lo - 1)
    inside = (
        aux.valid.bool()
        & (u[:, 0] >= 0)
        & (u[:, 0] < W)
        & (u[:, 1] >= 0)
        & (u[:, 1] < hi - lo)
    )
    sky = inside & (mask[yy, xx] == 2) & (o[:, 0] > 0)
    sig = inside & (mask[yy, xx] == 1) & (o[:, 0] > 0)
    visible = sig & (
        (
            (model.means - cam.position) @ cam.R_wc[2]
            - aux.depth[yy, xx] / cover[yy, xx].clamp_min(0.02)
        )
        .detach()
        .abs()
        < 1.5
    )
    sk = sky.float()
    signal = visible.float()
    loss = loss + 0.11 * (model.opacities * sk).sum() / sk.sum().clamp_min(16)
    if z.size:
        # Target is physical-axis z, not virtual-camera z or Euclidean range.
        depth = gpu(z[lo:hi].astype("f"))[yy, xx]
        axis = gpu(T[:3, :3] @ R.T[:, 2])
        zp = (model.means - cam.position) @ axis
        good = (
            visible
            & (depth > 1)
            & (depth < 48)
            & ((zp.detach() - depth).abs() < 0.10 * depth + 0.24)
        )
        wt = good.float()
        loss = loss + 0.15 * (
            ((zp - depth).abs() / (0.07 * depth + 0.18)) * wt
        ).sum() / wt.sum().clamp_min(32)
    vis = (inside & (mask[yy, xx] != 0)).float()
    prior = (
        d[:, 3].square() * 0.001
        + d[:, 4:6].square().sum(1) * 0.003
        + d[:, 6].square() * 0.009
        + d[:, :3].square().sum(1) * 0.0015
    )
    loss = loss + (vis * prior).sum() / vis.sum().clamp_min(1)
    loss.backward()
    gains[key] = float(
        np.clip(float(ex.detach().item()) - 0.035 * float(ex.grad.item()), -0.04, 0.04)
    )
    n = scene.update(ix, d, signal, sk)
    return len(ix), n, float(loss.item())


def run(args):
    if Path(args.out).exists():
        raise FileExistsError(f"Use a new output scene directory: {args.out}")
    tic = time.monotonic()
    src = Path(args.source).resolve() if args.source else init_source()
    s = Map(src, True, aa=args.aa)
    es = [e for e in J(CACHE / "manifest.json") if not e["hold"]]
    rng = np.random.default_rng(293)
    gains = {}
    rows = []
    peak = 0
    start = time.monotonic()
    epochs = 0
    while len(rows) < args.steps:
        batch = rng.permutation(len(es))
        epochs += 1
        for j in batch:
            e = es[j]
            bb = design(e["c"], e["yaw"])[5]
            bs = [b for b, (l, h) in enumerate(bb) if h > l]
            b = int(rng.choice(bs))
            ret = step(s, e, b, gains)
            if ret is None:
                continue
            rows.append(ret)
            i = len(rows)
            peak = max(peak, torch.mps.driver_allocated_memory())
            if i % 200 == 0:
                print(
                    i,
                    "steps",
                    round(i / (time.monotonic() - start), 1),
                    "it/s",
                    np.round(np.mean(rows[-200:], 0), 3),
                    flush=True,
                )
            if i == args.steps:
                break
    torch.mps.synchronize()
    elapsed = time.monotonic() - start
    outdir = Path(args.out)
    res = s.export(outdir)
    res.update(
        source=str(src),
        seconds=elapsed,
        it_s=len(rows) / elapsed,
        steps=len(rows),
        train_views=len(es),
        physical_exposures_touched=len(gains),
        mean_hot=float(np.mean(rows, 0)[0]),
        mean_updated=float(np.mean(rows, 0)[1]),
        antialias=args.aa,
        mps_driver_peak=peak,
        peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        total_s=time.monotonic() - tic,
        epochs=epochs,
    )
    savej(WORK / (outdir.name + "-fit.json"), res)
    savej(WORK / (outdir.name + "-gains.json"), gains)
    print(res, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=9000)
    ap.add_argument("--source")
    ap.add_argument("--aa", action="store_true")
    ap.add_argument("--out", default=str(RS / "scene/stitched-opt-v1"))
    run(ap.parse_args())
