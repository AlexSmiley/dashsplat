"""Standalone reference player. Its only scene input is the directory containing ibr.json.
Native library is compiled from renderer source, never loaded from a reconstruction workspace."""

import json, sys, subprocess, ctypes as ct, tempfile
from pathlib import Path
from functools import lru_cache
import numpy as np, cv2 as cv
from scipy.spatial.transform import Rotation

cv.setNumThreads(2)
W, H = 1440, 900
F = H / (2 * np.tan(np.deg2rad(25)))
K = np.array([[F, 0, W / 2 - 0.5], [0, F, H / 2 - 0.5], [0, 0, 1.0]])
Y, X = np.mgrid[:H, :W].astype("f")
RAYS = np.stack(((X + 0.5 - W / 2) / F, (Y + 0.5 - H / 2) / F, np.ones_like(X)), 2)


def sm(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def norm(x):
    return x / max(np.linalg.norm(x), 1e-9)


def readj(path):
    return json.loads(path.read_text())


def quad(a, T, im):
    n = np.array(a["plane"][:2])
    d = a["plane"][2]
    tan = np.array([-n[1], n[0]])
    u, z, U, Z = a["extent"]
    x, y, w, h = a["rect"]
    o = np.r_[n * d + tan * u, Z]
    A = np.c_[
        np.r_[tan * (U - u) / max(1, w - 1), 0],
        np.array([0, 0, (z - Z) / max(1, h - 1)]),
        o - T[:3, 3],
    ]
    C = T[:3, :3].T @ A
    pts = np.array([[0, 0, 1], [w - 1, 0, 1], [w - 1, h - 1, 1], [0, h - 1, 1]]) @ C.T
    poly = []
    for j, b in enumerate(pts):
        aa = pts[j - 1]
        if (aa[2] > 0.30) != (b[2] > 0.30):
            poly.append(aa + (b - aa) * (0.30 - aa[2]) / (b[2] - aa[2]))
        if b[2] > 0.30:
            poly.append(b)
    if len(poly) < 3:
        return None
    poly = np.array(poly)
    uv = poly[:, :2] / poly[:, 2, None] * F + [W / 2 - 0.5, H / 2 - 0.5]
    l, t = np.floor(uv.min(0) - 1).clip([0, 0], [W, H]).astype(int)
    r, b = np.ceil(uv.max(0) + 2).clip([0, 0], [W, H]).astype(int)
    if l >= r or t >= b:
        return None
    M = K @ C
    M[0] -= l * M[2]
    M[1] -= t * M[2]
    return np.s_[t:b, l:r], cv.warpPerspective(
        im[y : y + h, x : x + w], M, (r - l, b - t), flags=cv.INTER_LINEAR
    )


def proxy(img, st, T, depth):
    x = np.array([-0.9, 3.6, 3.6, -0.9] * 2)
    y = np.array([-0.925, -0.925, 0.925, 0.925] * 2)
    z = np.repeat([0.15, 1.48], 4)
    c, s = np.cos(st[3]), np.sin(st[3])
    p = (np.c_[st[0] + c * x - s * y, st[1] + s * x + c * y, st[2] + z] - T[:3, 3]) @ T[
        :3, :3
    ]
    uv = np.int32(np.round([W / 2, H / 2] + F * p[:, :2] / p[:, 2, None]))
    out = img.copy()
    for idx, col in sorted(
        [
            ([0, 1, 5, 4], [112, 118, 124]),
            ([1, 2, 6, 5], [92, 99, 105]),
            ([2, 3, 7, 6], [108, 115, 120]),
            ([3, 0, 4, 7], [78, 86, 92]),
            ([4, 5, 6, 7], [168, 175, 181]),
        ],
        key=lambda q: -p[q[0], 2].mean(),
    ):
        mask = np.zeros((H, W), "B")
        cv.fillConvexPoly(mask, uv[idx], 1)
        out[(mask > 0) & (depth > p[idx, 2].mean() - 0.6)] = col
    return out


class Player:
    def __init__(self, root, library=None):
        self.root = Path(root).resolve()
        self.doc = readj(self.asset("ibr.json"))
        j = self.doc
        self.pan = j["panoramas"]
        self.cards = j["cutouts"]
        self.cams = j["bases"]
        self.geo = {}
        for c, es in self.cams.items():
            o = np.array([e["pose"][:3] for e in es])
            ax = Rotation.from_quat([e["pose"][3:] for e in es]).as_matrix()[:, :, 2]
            self.geo[c] = (
                o,
                ax,
                np.array([e["s"] for e in es]),
                np.array([e["t"] for e in es]),
            )
        self.dynamic = {}
        self.mt = {}
        for c, ch in j["movers"].items():
            ls = [f for h in ch for f in readj(self.asset(h["file"]))]
            self.dynamic[c] = ls
            self.mt[c] = np.array([f["t"] for f in ls])
        drive = readj(self.asset(j["drive"]))
        self.ts = np.array(drive["t"])
        self.raw = np.stack(
            [
                np.unwrap(drive[k]) if k == "yaw" else drive[k]
                for k in ["x", "y", "z", "yaw", "speed"]
            ],
            1,
        )
        self.arc = np.r_[
            0, np.cumsum(np.linalg.norm(np.diff(self.raw[:, :2], axis=0), axis=1))
        ]
        self.poses = {}
        pj = readj(self.asset(j["chase"]))
        assert (pj["width"], pj["height"], pj["fps"], pj["vfov"]) == (W, H, 30, 50)
        for f in pj["frames"]:
            T = np.eye(4)
            T[:3, :3] = Rotation.from_quat(f["quat"]).as_matrix() @ np.diag([1, -1, -1])
            T[:3, 3] = f["pos"]
            self.poses[round(f["t"] * 30)] = T
        self.tiles = readj(self.asset(j["ground"]["map"]))["tiles"]
        self.bb = np.array([x["bbox"] for x in self.tiles])
        self.lib = None
        if library:
            self.lib = ct.CDLL(str(library))
            P = ct.c_void_p
            I = ct.c_int
            self.lib.raster.argtypes = [P, I, ct.c_float, I, P, P, I, I, I, P]
            self.lib.raster.restype = None

    def asset(self, rel):
        p = (self.root / rel).resolve()
        assert p.is_relative_to(self.root), rel
        return p

    @lru_cache(128)
    def rgb(self, p):
        im = cv.imread(str(self.asset(p)))
        assert im is not None, p
        return im[:, :, ::-1].copy()

    @lru_cache(100)
    def rgba(self, p, m):
        im = self.rgb(p)
        a = cv.imread(str(self.asset(m)), 0)
        assert a.shape == im.shape[:2]
        return np.dstack([np.uint8(im * (a[:, :, None] / 255) + 0.5), a])

    @lru_cache(32)
    def tile(self, file):
        with self.asset(file).open("rb") as f:
            while True:
                line = f.readline()
                assert line
                if line == b"end_header\n":
                    break
            a = np.fromfile(f, "<f4").reshape(-1, 24)
            return np.column_stack([a, np.zeros(len(a), "f")])

    def selection(self, t, T):
        st = np.interp(t, self.ts, self.arc)
        eye = T[:3, 3]
        target = T[:3, 2]
        choices = []
        for c, (O, A, S, times) in self.geo.items():
            ds = np.diff(O[:, :2], axis=0)
            q = np.clip(
                ((eye[:2] - O[:-1, :2]) * ds).sum(1)
                / np.maximum(1e-6, (ds * ds).sum(1)),
                0,
                1,
            )
            dist = np.linalg.norm(O[:-1, :2] + q[:, None] * ds - eye[:2], axis=1)
            idx = int(
                np.argmin(
                    dist
                    + 0.015 * abs(times[:-1] - t)
                    + np.where(abs(S[:-1] - st) < 55, 0, 1e5)
                )
            )
            f = float(q[idx])
            ax = norm(A[idx] * (1 - f) + A[idx + 1] * f)
            angle = float(np.rad2deg(np.arccos(np.clip(ax @ target, -1, 1))))
            score = float(
                dist[idx]
                + 0.32 * angle
                + 0.025 * abs((1 - f) * times[idx] + f * times[idx + 1] - t)
            )
            choices.append((score, c, idx, f, angle))
        choices.sort()
        best = choices[0][0]
        ls = []
        weights = {}
        den = 0
        for score, c, i, f, ang in choices:
            w = float(sm(1 - (score - best) / 4))
            den += w
            if w <= 0:
                continue
            for idx, share in [(i, 1 - f), (i + 1, f)]:
                if share > 1e-6:
                    ls.append((self.cams[c][idx]["pano"], w * share))
            weights[c] = w
        uniq = {}
        for k, w in ls:
            uniq[k] = uniq.get(k, 0) + w / den
        return uniq, [
            (q[1], q[2], q[3], q[4], weights.get(q[1], 0) / den) for q in choices
        ]

    def in_layer(self, layer, t):
        return any(
            lo <= t <= hi
            for lo, hi in self.doc.get("layerIntervals", {}).get(
                layer, [self.doc["interval"]]
            )
        )

    def active(self, t, T):
        eye = T[:3, 3]
        fw = T[:3, 2]
        ls = []
        for a in self.cards if self.in_layer("cutouts", t) else []:
            f = float(
                sm((t - a["birth"]) / a["fade"]) * sm((a["death"] - t) / a["fade"])
            )
            if f <= 0:
                continue
            cen = np.array(a["center"])
            delta = cen - eye
            span = a["extent"][2] - a["extent"][0]
            if delta @ fw < -span or np.linalg.norm(delta) > 115:
                continue
            ls.append((a, f, False))
        movers = []
        for c in self.dynamic if self.in_layer("movers", t) else []:
            i = int(np.argmin(abs(self.mt[c] - t)))
            e = self.dynamic[c][i]
            if abs(e["t"] - t) <= 0.24:
                movers.extend(e["layers"])
        used = []
        for a in sorted(movers, key=lambda a: -a["rect"][2] * a["rect"][3]):
            cen = np.array(a["center"])
            c = a["source"][0]
            if any(
                c != cc
                and np.linalg.norm(cen[:2] - pos[:2]) < 2.8
                and abs(cen[2] - pos[2]) < 1.6
                for cc, pos in used
            ):
                continue
            used.append((c, cen))
            ls.append((a, 1.0, True))
        return ls

    def draw(self, t):
        assert self.lib
        T = self.poses[round(t * 30)]
        st = np.array([np.interp(t, self.ts, col) for col in self.raw.T])
        O = T[:3, 3]
        wr = RAYS @ T[:3, :3].astype("f").T
        co = wr[2::4, 2::4]
        ang = np.arctan2(co[:, :, 1], co[:, :, 0])
        el = np.rad2deg(np.arctan2(co[:, :, 2], np.linalg.norm(co[:, :, :2], axis=2)))
        weights, who = self.selection(t, T)
        sky = np.zeros((H, W, 3), "f")
        far = np.zeros((H, W, 4), "f")
        zsum = np.zeros((H, W), "f")
        eye = O[:2]

        def sample(im, a, b, lo, hi):
            h, w = im.shape[
                :2
            ]  # Interpolate unwrapped azimuth, never interpolate across the texture's longitude seam.
            aa = np.unwrap(a, axis=1)
            aa = np.unwrap(aa, axis=0)
            uv = np.stack(
                (
                    (aa / (2 * np.pi) + 0.5) * w - 0.5,
                    np.clip((hi - b) / (hi - lo) * h - 0.5, 0, h - 1),
                ),
                2,
            ).astype("f")
            return cv.remap(
                im,
                cv.resize(uv, (W, H)),
                None,
                cv.INTER_LINEAR,
                borderMode=cv.BORDER_WRAP,
            )

        for k, w in weights.items():
            p = self.pan[k]
            sky += w * sample(self.rgb(p["sky"]), ang, el, *self.doc["skyElevation"])
            origin = np.array(p["origin"])
            delta = O[:2] - origin[:2]
            den = (co[:, :, :2] ** 2).sum(2)
            b = co[:, :, :2] @ delta
            disc = b * b - den * (delta @ delta - p["radius"] ** 2)
            z = (-b + np.sqrt(np.maximum(0, disc))) / den
            world = (O - origin) + co * z[:, :, None]
            aa = np.arctan2(world[:, :, 1], world[:, :, 0])
            ee = np.rad2deg(np.arctan2(world[:, :, 2], p["radius"]))
            v = sample(
                self.rgba(p["rgb"], p["matte"]), aa, ee, *self.doc["backdropElevation"]
            )
            lo, hi = self.doc["backdropElevation"]
            ok = cv.resize(
                ((disc > 0) & (z > 0.3) & (ee >= lo) & (ee <= hi)).astype("f"), (W, H)
            )
            coef = w * ok
            far += v * coef[:, :, None]
            zsum += cv.resize(z.astype("f"), (W, H)) * coef * (v[:, :, 3] / 255)
        dz = wr[:, :, 2]
        rg = (st[2] - 0.08 - O[2]) / np.minimum(-0.001, dz)
        rad = np.linalg.norm(O[:2] + wr[:, :, :2] * rg[:, :, None] - st[:2], axis=2)
        alpha = (dz < -0.001) * (rg > 0) * (1 - sm((rad - 160) / 20))
        haze = sm((rad - 30) / 100)[:, :, None]
        ap = np.array([100, 111, 113], "f") + haze * (2 - haze) * np.array(
            [50, 47, 44], "f"
        )
        buf = np.zeros((H, W, 6), "f")
        buf[:, :, :3] = sky + (ap - sky) * alpha[:, :, None]
        dist = np.linalg.norm(
            np.maximum(0, np.maximum(self.bb[:, :2] - eye, eye - self.bb[:, 3:5])),
            axis=1,
        )
        parts = [
            self.tile(e["file"])
            for e, on in zip(self.tiles, dist < 115)
            if on and self.in_layer("ground", t)
        ]
        pts = (
            np.ascontiguousarray(np.concatenate(parts), "f")
            if parts
            else np.empty((0, 25), "f")
        )
        rw = np.ascontiguousarray(np.repeat(T[None], 33, 0), "d")
        intr = np.array([F, F, W / 2, H / 2, 0, 0, 0, 0], "d")
        self.lib.raster(
            pts.ctypes.data,
            len(pts),
            t,
            3,
            rw.ctypes.data,
            intr.ctypes.data,
            0,
            W,
            H,
            buf.ctypes.data,
        )
        res = buf[:, :, :3].copy()
        depth = np.where((dz < -0.001) & (rg > 0), rg, 9999).astype("f")
        g = buf[:, :, 3] > 0.86
        depth[g] = buf[:, :, 4][g] / (buf[:, :, 3][g] * np.linalg.norm(wr[g], axis=1))
        a = far[:, :, 3] / 255
        fz = zsum / np.maximum(1e-5, a)
        visible = (a > 0.001) & (fz < depth + 0.12)
        res = (
            res * (1 - a[:, :, None] * visible[:, :, None])
            + far[:, :, :3] * visible[:, :, None]
        )
        solid = visible & (a > 0.5)
        depth[solid] = fz[solid]
        ids = np.zeros((H, W), "i")
        ids[solid] = -1
        layers = self.active(t, T)
        keys = []
        for mov in [False, True]:
            for card, f, m in sorted(
                layers, key=lambda l: -np.dot(np.array(l[0]["center"]) - O, T[:3, 2])
            ):
                if m != mov:
                    continue
                q = quad(card, T, self.rgba(card["rgb"], card["matte"]))
                if q is None:
                    continue
                region, im = q
                n = np.array(card["plane"][:2], "f")
                den = wr[region][:, :, :2] @ n
                z = np.divide(
                    card["plane"][2] - O[:2] @ n,
                    den,
                    out=np.full(den.shape, 9999, "f"),
                    where=abs(den) > 0.025,
                )
                on = (z > 0.30) & (z < depth[region] + 0.04)
                aa = im[:, :, 3] * (f / 255) * on
                res[region] = (
                    res[region] * (1 - aa[:, :, None])
                    + im[:, :, :3] * (f * on)[:, :, None]
                )
                solid = aa > 0.5
                depth[region][solid] = z[solid]
                ids[region][solid] = card["id"]
                keys.append(card["id"])
        return proxy(np.uint8(np.clip(res + 0.5, 0, 255)), st, T, depth), dict(
            cams=who,
            panos=weights,
            ids=ids,
            drawn=keys,
            ground=int(g.sum()),
            backdrop=int((ids == -1).sum()),
        )
