"""Post-cleanup photometric refit: keep geometry and visibility evidence fixed, cap opacity by confidence."""

from .v2fit import *
from .v2field import _RasterizeGaussiansImpl


class Fixed(Field):
    def __init__(self, src):
        super().__init__(src)
        self.init = self.p[:, 6].clone()
        self.m = torch.zeros_like(self.p[:, 3:7])
        self.v = torch.zeros_like(self.m)
        self.cnt = torch.zeros_like(self.p[:, 6])
        self.gains = {}

    def photo(self, e, rng, factor):
        args = crop(e, rng)
        if args is None:
            return None
        T, K, w, h, target, mask, opt = args
        ii = self.idx.ids(T, K, w, h)
        gg = self.gidx.ids(T, K, w, h)
        if not len(ii):
            return None
        ids = gpu(ii, torch.long)
        D = self.p[ids]
        G = self.g[gpu(gg, torch.long)]
        raw = torch.cat([D, G])
        rr = gpu(T[:3, :3].T)
        tr = gpu(-T[:3, :3].T @ T[:3, 3])
        u, z, co, rad, valid, comp = project_gaussians(
            raw[:, :3],
            raw[:, 7:10].exp(),
            raw[:, 10:14],
            rr,
            tr,
            *K,
            w,
            h,
            near=0.25,
            eps2d=0.22,
        )
        d = D[:, 3:7].detach().requires_grad_()
        n = len(ids)
        col = torch.cat([d[:, :3], G[:, 3:6]])
        alpha = torch.cat([d[:, 3], G[:, 6]]).sigmoid() * comp
        tile = DEFAULT_TILE_SIZE
        bins = bin_and_sort_gaussians(
            u, z, co, rad, valid, w, h, tile, opacities=alpha.detach()
        )
        args = (z, bins.sorted_gaussian_ids, bins.tile_bins, bins.tiles_x, w, h, tile)
        im, _, _ = _RasterizeGaussiansImpl.apply(
            u, co, alpha, col, *args, self.bg, None
        )
        gain = gpu(self.gains.get((e["c"], e["n"]), 0.0)).requires_grad_()
        pred = im * gain.exp()
        obj = mask == 1
        road = mask == 3
        sky = mask == 2
        err = (
            (pred - target).square().mean(2)
        )  # calibrated sky is NOT an RGB target: only penalise object coverage there.
        x = u[:n, 0].long().clamp(0, w - 1)
        y = u[:n, 1].long().clamp(0, h - 1)
        visible = (
            valid[:n].bool()
            & (u[:n, 0] >= 0)
            & (u[:n, 1] >= 0)
            & (u[:n, 0] < w)
            & (u[:n, 1] < h)
        )
        sk = visible & sky[y, x]
        a0 = self.init[ids]
        loss = (
            av(1.7 * err + 0.48 * (err + 0.006**2).sqrt(), obj.float() + 0.15 * road)
            + 0.12 * edge_loss(pred, target, obj)
            + 0.20 * av(d[:, 3].sigmoid(), sk)
            + 0.002 * av((d[:, 3] - a0).square(), visible)
            + 1.8 * gain.square().sum()
        )
        loss.backward()
        self.gains[(e["c"], e["n"])] = np.clip(
            gain.item() - 0.035 * gain.grad.item(), -0.045, 0.045
        )
        with torch.no_grad():
            grad = torch.nan_to_num(d.grad)
            use = grad.abs().sum(1) > 1e-11
            ix = ids[use]
            g = grad[use]
            cnt = self.cnt[ix] + 1
            mom = self.m[ix] * 0.86 + 0.14 * g
            v = self.v[ix] * 0.98 + 0.02 * g * g
            upd = (
                mom
                / (1 - 0.86 ** cnt[:, None])
                / (torch.sqrt(v / (1 - 0.98 ** cnt[:, None])) + 2e-9)
            )
            new = d[use] - upd * gpu([0.009, 0.009, 0.009, 0.065]) * factor
            new[:, :3].clamp_(0, 1)
            original = self.init[
                ix
            ]  # Never turn a sky/free-space-vetoed transparent seed opaque. On stronger seeds depth and location stay fixed; recover lost optical thickness up to ~3x.
            cap = 0.35 + 0.85 * ((original.sigmoid() - 0.035) / 0.23).clamp(0, 1)
            new[:, 3] = torch.maximum(
                original - 0.65, torch.minimum(original + cap, new[:, 3])
            )
            self.p[ix, 3:7] = new
            self.m[ix] = mom
            self.v[ix] = v
            self.cnt[ix] = cnt
        return [n, len(ix), int(obj.sum().item()), float(loss.item())]


def main(src, dst, steps):
    t = time.monotonic()
    s = Fixed(src)
    rng = np.random.default_rng(19)
    vs = J(V / "train.json")
    groups = [[e for e in vs if e["c"] == c] for c in CAMS]
    rows = []
    peak = 0
    tic = time.monotonic()
    while len(rows) < steps:
        g = groups[rng.integers(6)]
        a = s.photo(g[rng.integers(len(g))], rng, 0.52 + 0.48 * (1 - len(rows) / steps))
        if a is not None:
            rows.append(a)
            peak = max(peak, torch.mps.driver_allocated_memory())
        if a is not None and len(rows) % 1200 == 0:
            print(
                len(rows),
                len(rows) / (time.monotonic() - tic),
                np.mean(rows[-1200:], 0),
                flush=True,
            )
    res = s.export(dst)
    res.update(
        steps=steps,
        fit_s=time.monotonic() - tic,
        total_s=time.monotonic() - t,
        mean=np.mean(rows, 0).tolist(),
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        peak_mps=peak,
        physical_exposures=len(s.gains),
    )
    assert res["movement_m"][-1] == 0
    savej(V / (dst.name + "-colour.json"), res)
    print(res, flush=True)


if __name__ == "__main__":
    import sys

    main(Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]))
