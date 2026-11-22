"""Constrained DC 3DGS. Spatial bins select local records, preserving source identities."""

from .views import *
import sys, shutil, ctypes as ct, subprocess
from types import SimpleNamespace
import torch

sys.path.insert(0, str(RS / "third_party/metalsplat/src"))
from metalsplat import Camera, render
from pipeline.road.scene import reader, binary_header

C0 = 0.28209479177387814
BG = np.array([22, 26, 30], "f") / 255


def gpu(x, dtype=torch.float32):
    return torch.as_tensor(
        np.ascontiguousarray(x),
        dtype=dtype,
        device="cuda" if torch.cuda.is_available() else "mps",
    )


def init_source():
    for p in [
        RS / "scene/stitched-v3",
        RS / "maps/object-v3/stages/stitched-back",
        RS / "maps/object-v3/stages/stitched-sides",
    ]:
        if (p / "map.json").is_file() and any(
            t.get("layer") == "objects" for t in J(p / "map.json")["tiles"]
        ):
            return p
    raise ValueError("No validated object stage yet")


class Map:
    def __init__(self, src, train=False, aa=False):
        self.aa = aa
        self.src = src
        meta = J(src / "map.json")
        poses = traj.T_world_ego(np.arange(START, END + 0.02, 0.25))[:, :2, 3]
        packs = []
        tiles = []
        source = []
        for t in meta["tiles"]:
            lo, hi = np.array(t["bbox"]).reshape(2, 3)
            d = np.maximum(np.maximum(lo[:2] - poses, poses - hi[:2]), 0)
            if np.min(np.sum(d * d, 1)) > 84**2:
                continue
            a = reader(src / t["file"])
            packs.append(a)
            source.append(np.full(len(a), len(tiles), "i"))
            tiles.append(t)
        a = np.concatenate(packs)
        layer = np.array([t["layer"] == "objects" for t in tiles])
        sid = np.concatenate(source)
        obj = layer[sid]
        cell = np.floor(a[:, :3] / 5).astype("i")
        code = (
            (cell[:, 0].astype("int64") + 30000) * 2**32
            + (cell[:, 1].astype("int64") + 30000) * 2**16
            + (cell[:, 2] + 30000)
        )
        order = np.argsort(code, kind="stable")
        starts = np.r_[0, np.flatnonzero(np.diff(code[order])) + 1]
        self.starts = starts
        self.counts = np.diff(np.r_[starts, len(order)])
        self.tiles = tiles
        self.sid = sid[order]
        self.obj = obj[order]
        self.a = a[order].copy()
        self.n = len(order)
        self.cells = (cell[order[starts]] + 0.5) * 5
        extent = np.exp(self.a[:, 10:13].max(1))
        self.pad = 5 * np.sqrt(3) / 2 + 0.12 + np.maximum.reduceat(extent, starts) * 4.7
        self.background = gpu(BG)
        self.base = gpu(self.a[:, :17])
        self.objg = gpu(self.obj[:, None])
        self.offset = torch.zeros((self.n, 8), device="mps")
        self.train = train
        if train:
            self.m = torch.zeros_like(self.offset)
            self.v = torch.zeros_like(self.offset)
            self.count = torch.zeros((self.n, 1), device="mps")
            self.evidence = torch.zeros((self.n, 3), device="mps")
        self.xyzr = np.ascontiguousarray(
            np.column_stack(
                [self.a[:, :3], 0.12 + extent[order] * np.where(self.obj, 6.7, 4.7)]
            ),
            "f",
        )
        self.pad = 5 * np.sqrt(3) / 2 + np.maximum.reduceat(self.xyzr[:, 3], starts)
        lib = WORK / "cull.dylib"
        srcfile = Path(__file__).with_name("cull.cpp")
        if not lib.exists() or lib.stat().st_mtime < srcfile.stat().st_mtime:
            subprocess.run(
                [
                    "c++",
                    "-std=c++17",
                    "-O3",
                    "-dynamiclib",
                    str(srcfile),
                    "-o",
                    str(lib),
                ],
                check=True,
            )
        self.culler = ct.CDLL(str(lib)).cull
        P = ct.c_void_p
        self.culler.argtypes = [P, P, P, P, P, ct.c_int, P]
        self.culler.restype = ct.c_int

    def ids(self, T, K, w, h):
        R = T[:3, :3]
        v = (self.cells - T[:3, 3]) @ R
        z = v[:, 2]
        f, fv, cx, cy = K
        rad = self.pad
        left, right = -cx / f, (w - cx) / f
        top, bot = -cy / fv, (h - cy) / fv
        keep = (
            (z + rad > 0.25)
            & (z - rad < 85)
            & (v[:, 0] - z * left >= -rad * np.hypot(1, left))
            & (v[:, 0] - z * right <= rad * np.hypot(1, right))
            & (v[:, 1] - z * top >= -rad * np.hypot(1, top))
            & (v[:, 1] - z * bot <= rad * np.hypot(1, bot))
        )
        st, n = self.starts[keep], self.counts[keep]
        ids = np.empty(n.sum(), np.int32)
        f4 = np.array([left, right, top, bot])
        F = np.r_[f4, np.hypot(f4, 1.0)]
        T = np.ascontiguousarray(T, "d")
        nn = self.culler(
            self.xyzr.ctypes.data,
            T.ctypes.data,
            F.ctypes.data,
            st.ctypes.data,
            n.ctypes.data,
            len(n),
            ids.ctypes.data,
        )
        return gpu(ids[:nn], torch.long)

    def packet(self, T, K, w, h, learn=False):
        ix = self.ids(T, K, w, h)
        d = self.offset[ix].detach().requires_grad_(learn)
        base = self.base[ix]
        o = self.objg[ix]
        normal = base[:, 3:6]
        xyz = base[:, :3] + (normal * (0.006 + 0.084 * o)) * d[:, 6:7]
        s = base[:, 10:13] + torch.cat(
            [d[:, 4:6] * (0.13 + 0.42 * o), torch.zeros_like(o)], 1
        )
        alpha = (base[:, 9] + d[:, 3] * (0.6 + 1.8 * o[:, 0])).sigmoid()
        col = (0.5 + C0 * base[:, 6:9] + d[:, :3] * (0.10 + 0.23 * o)).clamp(0, 1)
        model = SimpleNamespace(
            means=xyz,
            scales=s.exp(),
            opacities=alpha,
            colors=col,
            quats=base[:, 13:17].contiguous(),
            sh_degree=0,
        )
        R = T[:3, :3].T
        cam = Camera(gpu(R), gpu(-R @ T[:3, 3]), *K, w, h)
        out = render(
            model,
            cam,
            near=0.25,
            antialias=self.aa,
            eps2d=0.22 if self.aa else 0.18,
            background=self.background,
            return_aux=True,
        )
        return ix, d, base, o, model, cam, out

    @torch.no_grad()
    def update(self, ix, d, signal, sky):
        grad = torch.nan_to_num(d.grad)
        active = grad.abs().sum(1) > 1e-12
        ids = ix[active]
        g = grad[active]
        n = self.count[ids] + 1
        m = self.m[ids] * 0.9 + g * 0.1
        v = self.v[ids] * 0.99 + g * g * 0.01
        lr = gpu([0.034, 0.034, 0.034, 0.032, 0.022, 0.022, 0.016, 0])
        step = m / (1 - 0.9**n) / (torch.sqrt(v / (1 - 0.99**n)) + 2e-9)
        self.offset[ids] = (d[active] - lr * step).clamp(-1, 1)
        self.m[ids] = m
        self.v[ids] = v
        self.count[ids] = n
        self.evidence[ids] += torch.stack(
            [signal[active], sky[active], (g[:, :3].abs().sum(1) > 1e-11).float()], 1
        )
        return len(ids)

    def export(self, dst):
        from scipy.spatial.transform import Rotation

        dst.mkdir(parents=True, exist_ok=True)
        (dst / "tiles").mkdir(exist_ok=True)
        delta = self.offset.cpu().numpy()
        ev = self.evidence.cpu().numpy()
        ob = self.obj.astype("f")
        a = self.a.copy()
        a[:, 6:9] = (
            np.clip(
                0.5 + C0 * a[:, 6:9] + delta[:, :3] * (0.10 + 0.23 * ob[:, None]), 0, 1
            )
            - 0.5
        ) / C0
        a[:, 9] += delta[:, 3] * (0.6 + 1.8 * ob)
        a[:, 10:12] += delta[:, 4:6] * (0.13 + 0.42 * ob[:, None])
        a[:, :3] += a[:, 3:6] * ((0.006 + 0.084 * ob) * delta[:, 6])[:, None]
        prune = (
            self.obj
            & (ev[:, 1] >= 4)
            & (ev[:, 1] > ev[:, 0] * 2 + 1)
            & (delta[:, 3] < -0.22)
        )
        split = (
            self.obj
            & ~prune
            & (ev[:, 0] >= 5)
            & (ev[:, 2] >= 8)
            & (delta[:, 4] > 0.35)
            & (np.exp(a[:, 10]) > 0.18)
            & (np.abs(delta[:, :3]).max(1) > 0.25)
        )
        splitids = np.flatnonzero(split)
        maxsplit = int(self.obj.sum() * 0.025)
        if len(splitids) > maxsplit:
            keep = splitids[np.argsort(ev[splitids, 0])[-maxsplit:]]
            split[:] = False
            split[keep] = True
        meta = J(self.src / "map.json")
        tiles = []
        cnt = 0
        for fn in ("rig.json", "drive.json"):
            shutil.copy2(self.src / fn, dst / fn)
        if meta.get("sky"):
            shutil.copy2(self.src / meta["sky"], dst / meta["sky"])
        for j, t in enumerate(self.tiles):
            use = self.sid == j
            aa = a[use & ~prune].copy()
            sp = split[use & ~prune]
            child = aa[sp].copy()
            if len(child):
                axis = Rotation.from_quat(child[:, [14, 15, 16, 13]]).as_matrix()[
                    :, :, 0
                ]
                off = 0.42 * np.exp(child[:, 10:11]) * axis
                child[:, :3] += off
                aa[sp, :3] -= off
                aa[sp, 10] -= 0.20
                child[:, 10] -= 0.20
                pa = 1 / (1 + np.exp(-child[:, 9]))
                p = 1 - np.sqrt(1 - pa)
                child[:, 9] = np.log(p / (1 - p))
                aa[sp, 9] = child[:, 9]
                aa = np.concatenate([aa, child])
            if not len(aa):
                continue
            assert (
                np.isfinite(aa).all()
                and (aa[:, 17] == 0).all()
                and (aa[:, 18] == 1e9).all()
            )
            q = dst / t["file"]
            q.parent.mkdir(parents=True, exist_ok=True)
            with q.open("wb") as f:
                f.write(binary_header(len(aa)))
                aa.astype("<f4").tofile(f)
            assert reader(q).shape == (len(aa), 24)
            tiles.append(
                dict(
                    t,
                    count=len(aa),
                    bbox=np.r_[aa[:, :3].min(0), aa[:, :3].max(0)].tolist(),
                )
            )
            cnt += len(aa)
        savej(dst / "map.json", dict(meta, tiles=tiles, slice_s=[START, END]))
        return dict(
            input=self.n,
            objects=int(self.obj.sum()),
            output=cnt,
            pruned=int(prune.sum()),
            split=int(split.sum()),
            touched=int((np.abs(delta).max(1) > 1e-6).sum()),
        )
