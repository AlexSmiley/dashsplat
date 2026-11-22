"""Fully movable oriented Gaussians; frozen mosaic participates only in occlusion/compositing."""

from .v2data import *
from .v2seed import scene
from .model import gpu, torch, C0
from metalsplat.ops.project import project_gaussians
from metalsplat.ops.rasterize import _RasterizeGaussiansImpl
from metalsplat.ops.tiling import bin_and_sort_gaussians, DEFAULT_TILE_SIZE
import ctypes as ct, subprocess, sys
import torch.nn.functional as F


def normal(q):
    w, x, y, z = q.unbind(1)
    return torch.stack(
        [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], 1
    )


def as_param(a):
    return np.column_stack(
        [a[:, :3], 0.5 + C0 * a[:, 6:9], a[:, 9:13], a[:, 13:17]]
    ).astype("f")


class Index:
    def __init__(self, data, move):
        xyz = data[:, :3]
        sig = np.exp(data[:, 7:10]).max(1)
        p = 6
        cell = np.floor(xyz / p).astype("i")
        code = (
            ((cell[:, 0].astype("i8") + 30000) * 65536 + cell[:, 1] + 30000) * 65536
            + cell[:, 2]
            + 30000
        )
        oo = np.argsort(code)
        st = np.r_[0, np.flatnonzero(np.diff(code[oo])) + 1]
        self.order = oo.astype("i")
        self.starts = st
        self.lengths = np.diff(np.r_[st, len(oo)])
        self.pos = (cell[oo[st]] + 0.5) * p
        rad = sig[oo] * 4.5 + (np.minimum(2.4, 0.26 + sig[oo] * 1.6) if move else 0.01)
        self.radius = 5.2 + np.maximum.reduceat(rad, st)
        self.xyzr = np.ascontiguousarray(np.c_[xyz[oo], rad], "f")
        self.cull = lib()

    def ids(self, T, intr, w, h):
        fx, fy, cx, cy = intr
        f = np.array([-cx / fx, (w - cx) / fx, -cy / fy, (h - cy) / fy])
        g = np.hypot(f, 1)
        v = (self.pos - T[:3, 3]) @ T[:3, :3]
        x, y, z = v.T
        r = self.radius
        keep = (
            (z + r > 0.25)
            & (z - r < 225)
            & (x - z * f[0] >= -r * g[0])
            & (x - z * f[1] <= r * g[1])
            & (y - z * f[2] >= -r * g[2])
            & (y - z * f[3] <= r * g[3])
        )
        s, n = self.starts[keep], self.lengths[keep]
        i = np.empty(n.sum(), "i")
        fr = np.r_[f, g]
        T = np.ascontiguousarray(T, "d")
        nn = self.cull(
            self.xyzr.ctypes.data,
            T.ctypes.data,
            fr.ctypes.data,
            s.ctypes.data,
            n.ctypes.data,
            len(s),
            i.ctypes.data,
        )
        return self.order[i[:nn]]


@lru_cache(None)
def lib():
    apple = sys.platform == "darwin"
    path = V / ("index.dylib" if apple else "index.so")
    src = Path(__file__).with_name("v2index.cpp")
    subprocess.run(
        [
            "c++",
            "-O3",
            "-std=c++17",
            *(["-dynamiclib"] if apple else ["-shared", "-fPIC"]),
            str(src),
            "-o",
            str(path),
        ],
        check=True,
    )
    l = ct.CDLL(str(path)).cull
    P = ct.c_void_p
    l.argtypes = [P, P, P, P, P, ct.c_int, P]
    l.restype = ct.c_int
    return l


class Field:
    def __init__(self, src):
        meta = J(src / "map.json")
        gs = np.concatenate(
            [
                as_param(reader(src / t["file"]))
                for t in meta["tiles"]
                if t["layer"] == "ground"
            ]
        )
        self.g = gpu(gs)
        self.gidx = Index(gs, False)
        del gs
        a = np.concatenate(
            [reader(src / t["file"]) for t in meta["tiles"] if t["layer"] == "objects"]
        )
        self.time = a[:, 17:24].copy()
        self.orig = a[:, :3].copy()
        p = as_param(a)
        self.p = gpu(p)
        self.idx = Index(p, True)
        self.m = torch.zeros_like(self.p)
        self.v = torch.zeros_like(self.p)
        self.count = torch.zeros((len(a), 1), device="mps")
        self.stats = torch.zeros((len(a), 4), device="mps")
        self.volume = Volume()
        self.floorgrid = gpu(
            np.maximum(
                self.volume.h, self.volume.g + np.where(self.volume.owned, 0.085, 0.015)
            )
        )
        self.origin = gpu(self.volume.origin)
        self.zeros = gpu([0, 0, 0])
        self.bg = gpu([22 / 255, 26 / 255, 30 / 255])
        self.topology = []

    def floor(self, xyz):
        ij = ((xyz[:, :2] - self.origin) / self.volume.step).floor().long()
        return self.floorgrid[
            ij[:, 1].clamp(0, self.floorgrid.shape[0] - 1),
            ij[:, 0].clamp(0, self.floorgrid.shape[1] - 1),
        ]

    def render(self, T, K, w, h):
        ii = self.idx.ids(T, K, w, h)
        gg = self.gidx.ids(T, K, w, h)
        ids = gpu(ii, torch.long)
        d = self.p[ids].detach().requires_grad_()
        g = self.g[gpu(gg, torch.long)]
        raw = torch.cat([d, g])
        means = raw[:, :3]
        sc = raw[:, 7:10].exp()
        q = F.normalize(raw[:, 10:14], dim=1)
        rr = gpu(T[:3, :3].T)
        tr = gpu(-T[:3, :3].T @ T[:3, 3])
        u, z, conic, rad, valid, comp = project_gaussians(
            means, sc, q, rr, tr, *K, w, h, near=0.25, eps2d=0.22
        )
        alpha = raw[:, 6].sigmoid() * comp
        tile = DEFAULT_TILE_SIZE
        bins = bin_and_sort_gaussians(
            u, z, conic, rad, valid, w, h, tile, opacities=alpha.detach()
        )
        args = (z, bins.sorted_gaussian_ids, bins.tile_bins, bins.tiles_x, w, h, tile)
        im, depth, trans = _RasterizeGaussiansImpl.apply(
            u, conic, alpha, raw[:, 3:6], *args, self.bg, None
        )
        zd = (means @ rr[2] + tr[2]).clamp(0.25, 240)
        logz = zd.log()
        moment, _, _ = _RasterizeGaussiansImpl.apply(
            u,
            conic,
            alpha,
            torch.stack([torch.ones_like(logz), logz, logz.square()], 1),
            *args,
            self.zeros,
            None,
        )
        u.retain_grad()
        return ids, d, im, moment, depth, trans, u, valid, zd, rad, q

    @torch.no_grad()
    def update(self, ids, d, ev, factor):
        grad = torch.nan_to_num(d.grad, nan=0, posinf=0, neginf=0)
        active = grad.abs().sum(1) > 1e-12
        ix = ids[active]
        p = d[active]
        g = grad[active]
        cnt = self.count[ix] + 1
        m = self.m[ix] * 0.88 + 0.12 * g
        v = self.v[ix] * 0.985 + 0.015 * g.square()
        step = m / (1 - 0.88**cnt) / (torch.sqrt(v / (1 - 0.985**cnt)) + 1e-9)
        sigma = p[:, 7:9].exp().mean(1)
        lrs = (
            gpu(
                [
                    1,
                    1,
                    1,
                    0.009,
                    0.009,
                    0.009,
                    0.045,
                    0.016,
                    0.016,
                    0.015,
                    0.009,
                    0.009,
                    0.009,
                    0.009,
                ]
            )
            .expand_as(p)
            .clone()
        )
        lrs[:, :3] = (0.007 + 0.12 * sigma).clamp(0.007, 0.23)[:, None]
        new = p - factor * step * lrs
        new[:, 3:6].clamp_(0, 1)
        new[:, 6].clamp_(-7, 6)
        new[:, 7:10].clamp_(
            -4.6, 1.0
        )  # limit aspect without forcing surfaces to retain their initial normal
        new[:, 7:9] = torch.maximum(
            new[:, 7:9], new[:, 7:9].amax(1, keepdim=True) - 1.45
        )
        new[:, 9] = torch.minimum(new[:, 9], new[:, 7:9].amax(1))
        new[:, 10:14] = F.normalize(new[:, 10:14], dim=1)
        floor = self.floor(new[:, :3])
        bad = new[:, 2] <= floor + 0.012
        new[bad, 6] = -7
        new[:, 2] = torch.maximum(new[:, 2], floor + 0.013)
        self.p[ix] = new
        self.m[ix] = m
        self.v[ix] = v
        self.count[ix] = cnt
        self.stats[ix] += ev[active]
        return len(ix)

    def reindex(self):
        self.idx = Index(self.p.detach().cpu().numpy(), True)

    @torch.no_grad()
    def grow(self, final=False):
        p = self.p.cpu().numpy()
        ev = self.stats.cpu().numpy()
        n = len(p)
        alpha = 1 / (1 + np.exp(-p[:, 6]))
        prune = (alpha < 0.025) | (
            (ev[:, 2] > 5) & (ev[:, 2] > 2.5 * ev[:, 0] + 2) & (alpha < 0.20)
        )
        vol = self.volume
        sc = np.exp(p[:, 7:10])
        from scipy.spatial.transform import Rotation

        rot = Rotation.from_quat(p[:, [11, 12, 13, 10]]).as_matrix()
        vertical = np.sqrt(np.sum((rot[:, 2, :] * sc) ** 2, 1))
        gap = p[:, 2] - vol.floor(p[:, :3])
        prune |= gap <= 0.015
        # Do not leave an opaque ellipse across a car/ground no-go boundary. Check its one-sigma principal-axis probes too.
        for k in range(3):
            for sign in [-1, 1]:
                test = p[:, :3] + sign * rot[:, :, k] * sc[:, k, None]
                floor = vol.floor(test)
                prune |= (test[:, 2] < floor - 0.025) & (vol.lookup(test)[1] > -99990)
        squeeze = np.clip(gap / np.maximum(vertical, 0.01), 0.22, 1)
        p[:, 7:10] += np.log(squeeze)[:, None]
        quality = ev[:, 1] / np.maximum(ev[:, 0], 1)
        choose = (
            (ev[:, 0] >= 4)
            & (quality > 0.023)
            & (ev[:, 3] / np.maximum(ev[:, 0], 1) > 2e-8)
            & (alpha > 0.16)
            & ~prune
        )
        sp = np.flatnonzero(choose)
        cap = int(n * 0.055) if not final else 0
        sp = sp[np.argsort(quality[sp])[-cap:]] if cap else sp[:0]
        large = sc[sp, :2].max(1) > 0.115
        axis = sc[sp, :2].argmax(1)
        delta = (
            rot[sp, :, axis] * sc[sp, axis, None] * np.where(large, 0.48, 0.20)[:, None]
        )
        child = p[sp].copy()
        p[sp, :3] -= delta
        child[:, :3] += delta
        p[sp, 7:9] -= np.where(large, 0.26, 0.06)[:, None]
        child[:, 7:9] = p[sp, 7:9]
        halfa = 1 - np.sqrt(1 - alpha[sp])
        p[sp, 6] = np.log(halfa / (1 - halfa))
        child[:, 6] = p[sp, 6]
        keep = np.flatnonzero(~prune)
        children = child[:, 2] > vol.floor(child[:, :3]) + 0.015
        sel = sp[children]
        newp = np.concatenate([p[keep], child[children]])
        self.time = np.concatenate([self.time[keep], self.time[sel]])
        self.orig = np.concatenate([self.orig[keep], self.orig[sel]])
        self.p = gpu(newp)
        ki = gpu(keep, torch.long)
        self.m = torch.cat([self.m[ki], torch.zeros((len(sel), 14), device="mps")])
        self.v = torch.cat([self.v[ki], torch.zeros((len(sel), 14), device="mps")])
        self.count = torch.cat(
            [self.count[ki], torch.zeros((len(sel), 1), device="mps")]
        )
        self.stats = torch.zeros((len(newp), 4), device="mps")
        self.idx = Index(newp, True)
        res = dict(
            before=n,
            pruned=int(prune.sum()),
            split=int(large[children].sum()),
            clone=int((~large[children]).sum()),
            after=len(newp),
        )
        self.topology.append(res)
        return res

    def export(self, dst):
        p = self.p.cpu().numpy()
        a = np.zeros((len(p), 24), "f")
        a[:, :3] = p[:, :3]
        a[:, 6:9] = (p[:, 3:6] - 0.5) / C0
        a[:, 9:13] = p[:, 6:10]
        a[:, 13:17] = p[:, 10:14]
        a[:, 3:6] = normal(self.p[:, 10:14]).cpu().numpy()
        a[:, 17:24] = self.time
        scene(dst, a)
        delta = np.linalg.norm(self.orig - p[:, :3], axis=1)
        return dict(
            objects=len(p),
            movement_m=np.percentile(delta, [50, 90, 99, 100]).tolist(),
            moved_25cm=int((delta > 0.25).sum()),
            topology=self.topology,
        )
