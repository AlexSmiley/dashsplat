"""Shared helpers for the film's real-data 3D visualisations: a z-buffered point splatter on the GPU (torch MPS),
a look-at camera, easing, and an ffmpeg writer. Everything drawn is project data; nothing is synthesised."""
import math, subprocess
import numpy as np
import torch

DEV = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
BG = (11, 12, 14)


def ease(x):
    """Cubic ease-in-out on [0, 1]."""
    x = min(max(x, 0.0), 1.0)
    return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def ease_t(x):
    x = x.clamp(0, 1)
    return torch.where(x < 0.5, 4 * x ** 3, 1 - (-2 * x + 2) ** 3 / 2)


def smooth_keys(keys, t):
    """keys: list of (time, dict of np arrays/floats). Piecewise eased interpolation."""
    if t <= keys[0][0]:
        return dict(keys[0][1])
    for (t0, a), (t1, b) in zip(keys, keys[1:]):
        if t <= t1:
            s = ease((t - t0) / max(t1 - t0, 1e-9))
            return {k: np.asarray(a[k]) * (1 - s) + np.asarray(b[k]) * s for k in a}
    return dict(keys[-1][1])


def look_at(eye, target, up=(0, 0, 1)):
    """World->camera rotation (OpenCV axes: x right, y down, z forward) and the eye."""
    eye, target, up = (np.asarray(v, float) for v in (eye, target, up))
    f = target - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    d = np.cross(f, r)
    return np.stack([r, d, f]), eye


def pose_to_keys(T, dist=20.0):
    """A camera-to-world pose as look-at parameters (eye, target, up)."""
    R, p = T[:3, :3], T[:3, 3]
    return dict(eye=p, target=p + R[:, 2] * dist, up=-R[:, 1])


class Splatter:
    def __init__(self, W, H, rmax=10):
        self.W, self.H, self.rmax = W, H, rmax
        self.offs = []
        for k in range(rmax + 1):
            rr = k + 0.5
            g = torch.arange(-k, k + 1, device=DEV)
            dy, dx = torch.meshgrid(g, g, indexing='ij')
            keep = dx * dx + dy * dy <= rr * rr
            self.offs.append((dx[keep].reshape(-1), dy[keep].reshape(-1)))

    def render(self, P, C, foot, Rcw, eye, vfov_deg, bg=BG, scale=1.0, alpha=None, near=0.3):
        """P (N,3) world, C (N,3) uint8/float 0..255, foot (N,) metres per source pixel at the point.
        scale multiplies the footprint (1 = surfaces close up; <1 shows points; >1 thick discs).
        alpha (N,) in 0..1 drawn by stochastic inclusion (stable per point)."""
        W, H = self.W, self.H
        f = 0.5 * H / math.tan(math.radians(vfov_deg) / 2)
        Rt = torch.as_tensor(Rcw, dtype=torch.float32, device=DEV)
        e = torch.as_tensor(eye, dtype=torch.float32, device=DEV)
        X = (P - e) @ Rt.T
        z = X[:, 2]
        keep = z > near
        if alpha is not None:
            keep &= alpha > self.rand[: len(alpha)]
        idx = torch.nonzero(keep).squeeze(1)
        X, z = X[idx], z[idx]
        u = f * X[:, 0] / z + W / 2
        v = f * X[:, 1] / z + H / 2
        r = foot[idx] * scale * f / z
        inside = (u > -r - 1) & (u < W + r + 1) & (v > -r - 1) & (v < H + r + 1)
        idx, u, v, z, r = idx[inside], u[inside], v[inside], z[inside], r[inside]
        rc = torch.clamp(torch.round(r - 0.5), 0, self.rmax).long()
        pix, dep, src = [], [], []
        ui, vi = torch.floor(u).long(), torch.floor(v).long()
        for k in range(self.rmax + 1):
            m = torch.nonzero(rc == k).squeeze(1)
            if len(m) == 0:
                continue
            dx, dy = self.offs[k]
            pu = (ui[m][:, None] + dx[None]).reshape(-1)
            pv = (vi[m][:, None] + dy[None]).reshape(-1)
            ok = (pu >= 0) & (pu < W) & (pv >= 0) & (pv < H)
            pix.append((pv * W + pu)[ok])
            dep.append(z[m][:, None].expand(-1, len(dx)).reshape(-1)[ok])
            src.append(idx[m][:, None].expand(-1, len(dx)).reshape(-1)[ok])
        img = torch.tensor(bg, dtype=torch.float32, device=DEV).repeat(H * W, 1)
        if pix:
            pix, dep, src = torch.cat(pix), torch.cat(dep), torch.cat(src)
            zb = torch.full((H * W,), float('inf'), device=DEV)
            zb.scatter_reduce_(0, pix, dep, reduce='amin')
            win = dep <= zb[pix]
            img[pix[win]] = C[src[win]].float()
        return img.reshape(H, W, 3)

    def set_random(self, n, seed=7):
        g = torch.Generator(device='cpu').manual_seed(seed)
        self.rand = torch.rand(n, generator=g).to(DEV)


class Writer:
    def __init__(self, path, W, H, fps=30, crf=12):
        self.p = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}',
                                   '-r', str(fps), '-i', '-', '-c:v', 'libx264', '-preset', 'medium', '-crf', str(crf),
                                   '-g', '15', '-bf', '0', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', path], stdin=subprocess.PIPE)

    def write(self, img):
        if isinstance(img, torch.Tensor):
            img = img.clamp(0, 255).to(torch.uint8).cpu().numpy()
        self.p.stdin.write(np.ascontiguousarray(img).tobytes())

    def close(self):
        self.p.stdin.close()
        self.p.wait()
