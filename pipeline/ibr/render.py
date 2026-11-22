"""Perspective-correct textured sensor meshes, two GPU passes. No optical flow or generated pixels."""

from .pack import *
import ctypes as ct
from collections import OrderedDict
from scipy.spatial.transform import Rotation
from pipeline.livemap import v5_proof as v5

WID, HEI = 1440, 900
FOC = HEI / (2 * np.tan(np.deg2rad(25)))
HERE = Path(__file__).parent


def grid(c, cs):
    h, w = H, B
    ys = np.unique(np.r_[np.arange(0, h, 4), h - 1])
    xs = np.unique(np.r_[np.arange(0, w, 4), w - 1])
    y, x = np.meshgrid(ys, xs, indexing="ij")
    uv = np.stack(((x + 0.5) / w, (y + 0.5) / h), -1)
    ray = np.fromfile(STAGE / cs["rays"], "<f4").reshape(h, w, 3)[y, x]
    i = np.arange(x.size, dtype="I").reshape(x.shape)
    tri = np.stack(
        [
            np.stack((i[:-1, :-1], i[:-1, 1:], i[1:, 1:]), -1),
            np.stack((i[:-1, :-1], i[1:, 1:], i[1:, :-1]), -1),
        ],
        -2,
    ).reshape(-1, 3)
    return y, x, uv, ray, tri


def rowmat(e, f):
    a, b = np.array(e["rowPoses"])
    q0, q1 = a[3:], b[3:]
    q1 = q1 if q0 @ q1 >= 0 else -q1
    q = q0 + f[:, None] * (q1 - q0)
    return Rotation.from_quat(q).as_matrix(), a[:3] + f[:, None] * (b[:3] - a[:3])


def mesh(c, e, data):
    y, x, uv, ray, tri = data
    cat = cv.imread(str(STAGE / e["mask"]), -1)
    z = cv.imread(str(STAGE / e["z"]), -1)[y, x] * 0.01
    lab = cat[y, x]
    R, O = rowmat(e, uv[:, 0, 1])
    dr = np.einsum("hij,hwj->hwi", R, ray)
    r = z / np.maximum(0.01, ray[:, :, 2])
    plane = np.asarray(e["groundPlane"])
    rr = -(O @ plane[:3] + plane[3])[:, None] / np.minimum(-0.0001, dr @ plane[:3])
    road = (lab == 2) & (dr @ plane[:3] < -0.012) & (rr > 0.4) & (rr < 110)
    mix = np.clip((rr - 35) / 35, 0, 1) * (r > 0)
    r = np.where(road, rr * (1 - mix) + r * mix, r)
    sky = lab == 3
    pos = O[:, None, :] + dr * r[:, :, None]
    pos[sky] = dr[sky]
    vt = np.concatenate((pos, uv, lab[:, :, None]), 2).reshape(-1, 6).astype("f")
    k = lab.ravel()[tri]
    dist = r.ravel()[tri]
    same = np.all(k == k[:, :1], axis=1)
    den = dist.min(1)
    spread = dist.max(1) - den
    ok = same & (
        (k[:, 0] == 3)
        | (
            ((k[:, 0] == 1) | (k[:, 0] == 2) | (k[:, 0] >= 8))
            & (den > 0.3)
            & (dist.max(1) < 250)
            & (spread < 0.30 + 0.14 * den)
        )
    )
    ix = np.ascontiguousarray(tri[ok].ravel())
    im = cv.cvtColor(cv.imread(str(STAGE / e["rgb"])), cv.COLOR_BGR2RGBA)
    return vt, ix, im, cat


def selection_frames(es):
    sel = [0]
    for i in range(1, len(es)):
        if es[i]["s"] - es[sel[-1]]["s"] >= 2.4 or es[i]["t"] - es[sel[-1]]["t"] >= 0.9:
            sel.append(i)
    if sel[-1] != len(es) - 1:
        sel.append(len(es) - 1)
    return np.array(sel)


class Scene:
    def __init__(self, bank=None):
        doc = json.load(open(bank or STAGE / "ibr.json"))
        self.info = doc["cameras"]
        self.cam = v5.Camera(BASE)
        self.fb = v5.Render(BASE, self.cam)
        self.cache = OrderedDict()
        self.spec = {}
        self.frames = {}
        self.knots = {}
        self.grid = {}
        self.odom = np.r_[
            0, np.cumsum(np.linalg.norm(np.diff(self.cam.raw[:, :2], axis=0), axis=1))
        ]
        self.g = self.fb.g
        sky = (
            STAGE.parent / "movie-sky.jpg"
            if bank and Path(bank).name == "movie-bank.json"
            else STAGE / "sky.jpg"
        )
        self.pano = cv.imread(str(sky))[:, :, ::-1]
        self.drawn = []
        for c, cs in self.info.items():
            es = cs["keyframes"]
            self.frames[c] = es
            idx = selection_frames(es)
            self.knots[c] = idx
            self.grid[c] = grid(c, cs)
            O = []
            R = []
            for e in es:
                rot, xyz = rowmat(e, np.array([0.5]))
                O.append(xyz[0])
                R.append(rot[0])
            self.spec[c] = (
                np.array([e["t"] for e in es]),
                np.array(O),
                np.array(R),
                np.array([e["s"] for e in es]),
            )
        P = ct.c_void_p
        I = ct.c_int
        self.lib = ct.CDLL(str(STAGE.parent / "metal.dylib"))
        self.lib.ibr_init.argtypes = [ct.c_char_p, I, I]
        self.lib.ibr_init.restype = P
        self.lib.ibr_mesh.argtypes = [P, P, I, P, I, P, I, I, P, I, I]
        self.lib.ibr_mesh.restype = P
        self.lib.ibr_add.argtypes = [P, P, P]
        self.lib.ibr_free.argtypes = [P]
        self.lib.ibr_render.argtypes = [P, P, P]
        self.ctx = self.lib.ibr_init((HERE / "mesh.metal").read_bytes(), WID, HEI)
        assert self.ctx
        self.best = np.zeros((HEI, WID, 4), "f")
        self.sum = np.zeros((HEI, WID, 4), "e")
        self.closest = {}

    def select(self, t, T):
        o = T[:3, 3]
        R = T[:3, :3]
        st = self.cam.raw[round(t * 30)]
        sx = self.odom[round(t * 30)]
        gx, gy = np.meshgrid(
            np.array([-0.87, 0, 0.87]) * WID / (2 * FOC),
            np.array([-0.65, 0.12, 0.72]) * HEI / (2 * FOC),
        )
        ray = np.c_[gx.ravel(), gy.ravel(), np.ones(9)] @ R.T
        rad = np.where(
            ray[:, 2] < -0.07,
            np.clip((st[2] - o[2]) / np.minimum(-0.01, ray[:, 2]), 8, 80),
            65,
        )
        pts = o + ray * rad[:, None]
        cand = {}
        self.closest = {}
        for ci, c in enumerate(CAMS):
            ts, O, A, S = self.spec[c]
            idx = self.knots[c]
            idx = idx[abs(S[idx] - sx) < 62]
            if len(idx) < 2:
                continue
            pos = O[idx]
            v = np.einsum(
                "bij,bnj->bni", A[idx].transpose(0, 2, 1), pts - pos[:, None, :]
            )
            cam = rig[c]
            uv, ok = cam.project(v)
            uu = uv / np.array([cam.width, cam.height])
            inside = ok & (v[:, :, 2] > 0.2) & np.all((uu >= 0) & (uu <= 1), axis=2)
            fraction = inside.mean(1)
            ax = R[:, 2][:2].copy()
            ax /= np.linalg.norm(ax)
            near = int(np.argmin(abs(ts - t)))
            temporal = abs(ts[near] - t)
            # At the three sample locations, project onto each 2D source trajectory segment.
            for offset, gain in [(0, 1.0), (11, 0.49), (-9, 0.37), (24, 0.24)]:
                target = o[:2] + offset * ax
                dxy = np.diff(pos[:, :2], axis=0)
                f = np.clip(
                    ((target - pos[:-1, :2]) * dxy).sum(1)
                    / np.maximum(0.0001, (dxy * dxy).sum(1)),
                    0,
                    1,
                )
                D = np.linalg.norm(pos[:-1, :2] + f[:, None] * dxy - target, axis=1)
                cov = (1 - f) * fraction[:-1] + f * fraction[1:]
                cost = D + 7 * (1 - cov) + 0.05 * abs(ts[idx[:-1]] - t)
                j = np.argmin(cost)
                frac = f[j]
                for ii, w in [(j, 1 - frac), (j + 1, frac)]:
                    if w < 0.012 or (
                        fraction[ii] == 0 and np.dot(A[idx[ii], :, 2], R[:, 2]) < -0.15
                    ):
                        continue
                    ni = idx[ii]
                    dist = np.linalg.norm(pos[ii, :2] - o[:2])
                    q = (
                        gain
                        * (1.3 if c == "front" else 1.0)
                        * np.exp(-dist / 18)
                        * (0.48 + 0.52 * fraction[ii])
                    )
                    k = (c, int(ni), 0)
                    q = max(0.014, q)
                    old = cand.get(k)
                    value = (q * np.sqrt(w), w, ci * 100000 + esid(self.frames[c][ni]))
                    if old is None or value[0] > old[0]:
                        cand[k] = value
                if offset == 0:
                    self.closest[c] = self.frames[c][int(idx[j + (frac >= 0.5)])]["t"]
            if temporal <= 0.35 and self.frames[c][near]["hasInstances"]:
                cand[(c, near, 1)] = (
                    max(0.55, 1 - temporal),
                    1.0,
                    ci * 100000 + esid(self.frames[c][near]),
                )
        return cand

    def load(self, c, i):
        k = c, i
        if k in self.cache:
            self.cache.move_to_end(k)
            return self.cache[k]
        v, ix, im, cat = mesh(c, self.frames[c][i], self.grid[c])
        m = self.lib.ibr_mesh(
            self.ctx,
            v.ctypes.data,
            len(v),
            ix.ctypes.data,
            len(ix),
            im.ctypes.data,
            im.shape[1],
            im.shape[0],
            cat.ctypes.data,
            B,
            H,
        )
        assert m
        self.cache[k] = m
        return m

    def draw(self, t, az=0, raw=None):
        T, st, _ = self.cam.pose(t, az)
        ls = self.select(t, T)
        R = T[:3, :3]
        o = T[:3, 3]
        M = np.c_[R.T, -R.T @ o]
        M[0] *= 2 * FOC / WID
        M[1] *= -2 * FOC / HEI
        for (c, i, mov), (q, w, tag) in ls.items():
            m = self.load(c, i)
            u = np.ascontiguousarray(
                np.r_[
                    M.ravel(),
                    q,
                    w,
                    tag,
                    mov,
                    self.frames[c][i]["displayGain"],
                    1.0,
                    self.spec[c][1][i],
                    1.0,
                    o,
                    1.0,
                ],
                "f",
            )
            self.lib.ibr_add(self.ctx, m, u.ctypes.data)
        assert self.lib.ibr_render(
            self.ctx, self.best.ctypes.data, self.sum.ctypes.data
        )
        self.drawn = list(ls)
        while len(self.cache) > 82:
            self.lib.ibr_free(self.cache.popitem(last=False)[1])
        fallback, obj = self.fb.draw(t, az, raw=raw)
        w = self.sum[:, :, 3].astype("f")
        k = self.best[:, :, 1]
        valid = w > 1e-5
        col = self.sum[:, :, :3].astype("f") * (255 / np.maximum(w, 1e-5))[:, :, None]
        dirs = self.fb.rays @ T[:3, :3].astype("f").T
        pan = cv.remap(
            self.pano,
            (
                (0.5 - np.arctan2(dirs[:, :, 1], dirs[:, :, 0]) / (2 * np.pi)) * 1024
                - 0.5
            ).astype("f"),
            np.clip(
                (0.5 - np.arcsin(np.clip(dirs[:, :, 2], -1, 1)) / np.pi) * 512 - 0.5,
                0,
                511,
            ).astype("f"),
            cv.INTER_LINEAR,
            borderMode=cv.BORDER_WRAP,
        )
        background = (~valid) & (dirs[:, :, 2] > -0.012)
        a0 = np.clip(obj, 0, 1)[:, :, None]
        patched = np.where(
            background[:, :, None], pan + a0 * (fallback.astype("f") - pan), fallback
        )
        out = np.where(
            valid[:, :, None], col, patched
        )  # sky is behind the fused map, not an occluder
        sky = valid & (k == 3)
        skyedge = cv.distanceTransform(sky.astype("B"), cv.DIST_L2, 3)
        sf = (0.68 * np.clip(skyedge / 26, 0, 1) * sky)[:, :, None]
        out = np.where(sky[:, :, None], pan * (1 - sf) + out * sf, out)
        a = np.where(sky, np.clip(obj, 0, 1), 0)[:, :, None]
        out += a * (fallback - out)
        out = np.uint8(np.clip(out + 0.5, 0, 255))
        return out, fallback, valid, k, T


def esid(e):
    return e["n"]


def box(img, st, T, zmap):
    # Rear-axle reference, 4.5 x 1.85 x 1.48 m; depth-tested against the source mesh.
    xx = np.array([-0.9, 3.6, 3.6, -0.9] * 2)
    yy = np.array([-0.925, -0.925, 0.925, 0.925] * 2)
    zz = np.repeat([0.15, 1.48], 4)
    c, s = np.cos(st[3]), np.sin(st[3])
    pts = np.c_[st[0] + c * xx - s * yy, st[1] + s * xx + c * yy, st[2] + zz]
    v = (pts - T[:3, 3]) @ T[:3, :3]
    pix = np.int32(np.round([WID / 2, HEI / 2] + FOC * v[:, :2] / v[:, 2, None]))
    faces = [
        ([0, 1, 5, 4], [112, 118, 124]),
        ([1, 2, 6, 5], [92, 99, 105]),
        ([2, 3, 7, 6], [108, 115, 120]),
        ([3, 0, 4, 7], [78, 86, 92]),
        ([4, 5, 6, 7], [168, 175, 181]),
    ]
    dest = img.copy()
    for idx, color in sorted(faces, key=lambda a: -v[a[0], 2].mean()):
        mask = np.zeros(img.shape[:2], "B")
        cv.fillConvexPoly(mask, pix[idx], 1)
        ok = (mask > 0) & ((zmap == 0) | (zmap > v[idx, 2].mean() - 0.6))
        dest[ok] = color
    return dest
