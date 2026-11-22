"""Pairwise compositing of rigid tracked cards. Pixel colours are in original corrected camera RGB."""

from .popup_layers import *
from .render import box, v5
from .proof import jpg
from .render import Scene as Old
from scipy.ndimage import gaussian_filter1d
from collections import defaultdict

K = np.array([[FOC, 0, WID / 2 - 0.5], [0, FOC, HEI / 2 - 0.5], [0, 0, 1.0]])
Y, X = np.mgrid[:HEI, :WID].astype("f")
RAYS = np.stack(
    ((X + 0.5 - WID / 2) / FOC, (Y + 0.5 - HEI / 2) / FOC, np.ones_like(X)), 2
)


def smooth(x):
    v = np.clip(x, 0, 1)
    return v * v * (3 - 2 * v)


class ExportedChase:
    """Consume Three.js camera-to-world poses directly; only change camera axis conventions."""

    def __init__(self):
        from scipy.spatial.transform import Rotation

        drive = json.load(open(BASE / "drive.json"))
        self.t = np.arange(0, drive["duration"] + 0.04, 1 / 30)
        self.raw = np.stack(
            [
                np.interp(
                    self.t, drive["t"], np.unwrap(drive[n]) if n == "yaw" else drive[n]
                )
                for n in ["x", "y", "z", "yaw", "speed"]
            ],
            1,
        )
        self.poses = {}
        for angle, name in [
            (0, "chase"),
            (-30, "orbit--30"),
            (30, "orbit-30"),
            (0, "stop-200"),
        ]:
            j = json.load(open(ROOT / "poses" / (name + ".json")))
            assert (j["fps"], j["width"], j["height"], j["vfov"], j["near"]) == (
                30,
                WID,
                HEI,
                50,
                0.3,
            )
            ls = j["frames"]
            R = Rotation.from_quat([a["quat"] for a in ls]).as_matrix() @ np.diag(
                [1.0, -1.0, -1.0]
            )
            for a, r in zip(ls, R):
                i = round(a["t"] * 30)
                assert abs(a["t"] - i / 30) < 0.00006
                T = np.eye(4)
                T[:3, :3] = r
                T[:3, 3] = a["pos"]
                self.poses[(angle, i)] = T

    def pose(self, t, az=0, *unused):
        i = round(t * 30)
        return self.poses[az, i], self.raw[i], None


class Road(v5.Render):
    def grade(self, x):
        return x

    def __init__(self, cam):
        super().__init__(BASE, cam)
        self.root = RS / "scene/road-stitched-v2"
        self.doc = json.load(open(self.root / "map.json"))
        self.tiles = self.doc["tiles"]
        assert all(e["layer"] == "ground" for e in self.tiles)
        self.bbox = np.array([e["bbox"] for e in self.tiles])
        self.tmin = np.array([e["t_min"] for e in self.tiles])
        self.tmax = np.array([e["t_max"] for e in self.tiles])
        self.fog = np.array([150.0, 158.0, 157.0])
        ref = json.load(open(self.root / "drive.json"))
        self.ref = np.stack(
            [
                np.interp(cam.t, ref["t"], np.unwrap(ref[n]) if n == "yaw" else ref[n])
                for n in ["x", "y", "z", "yaw"]
            ],
            1,
        )
        self.tree = v5.cKDTree(self.ref[::6, :2])

    @lru_cache(48)
    def tile(self, file):
        a = v5.reader(self.root / file).copy()
        _, i = self.tree.query(a[:, :2])
        i = i * 6
        src = self.ref[i]
        dst = self.cam.raw[i]
        ang = dst[:, 3] - src[:, 3]
        cc = np.cos(ang)
        ss = np.sin(ang)
        dx = a[:, 0] - src[:, 0]
        dy = a[:, 1] - src[:, 1]
        a[:, 0] = dst[:, 0] + cc * dx - ss * dy
        a[:, 1] = dst[:, 1] + ss * dx + cc * dy
        a[:, 2] += dst[:, 2] - src[:, 2]
        x, y = a[:, 3].copy(), a[:, 4].copy()
        a[:, 3] = cc * x - ss * y
        a[:, 4] = ss * x + cc * y
        cw, sw = np.cos(ang / 2), np.sin(ang / 2)
        qw, qx, qy, qz = a[:, 13:17].copy().T
        a[:, 13:17] = np.stack(
            [
                cw * qw - sw * qz,
                cw * qx - sw * qy,
                cw * qy + sw * qx,
                cw * qz + sw * qw,
            ],
            1,
        )
        return np.column_stack([a, np.zeros(len(a), "f")])

    def points(self, t, margin=0):
        ix = np.flatnonzero(self.indices(t, margin))
        return (
            np.vstack([self.tile(self.tiles[i]["file"]) for i in ix])
            if len(ix)
            else np.empty((0, 25), "f")
        )

    def layer(self, t, T, raw, sky, wr, st):
        o = T[:3, 3]
        den = wr[:, :, 2]
        rg = (st[2] - 0.08 - o[2]) / np.minimum(-0.001, den)
        xy = o[:2] + wr[:, :, :2] * rg[:, :, None]
        dist = np.linalg.norm(xy - st[:2], axis=2)
        hz = smooth((dist - 30) / 100)
        near = 0.83 * self.road + 0.17 * self.fog
        apron = near + (self.fog - near) * ((2 - hz) * hz)[:, :, None]
        alpha = (den < -0.001) * (rg > 0) * (1 - smooth((dist - 162) / 18))
        buf = np.zeros((HEI, WID, 6), "f")
        buf[:, :, :3] = sky + (apron - sky) * alpha[:, :, None]
        pts = self.colour(t, raw, st)
        rw = np.ascontiguousarray(np.repeat(T[None], 33, axis=0), "d")
        kn = np.r_[FOC, FOC, WID / 2, HEI / 2, np.zeros(4)].astype("d")
        self.lib.raster(
            pts.ctypes.data,
            len(pts),
            t,
            3,
            rw.ctypes.data,
            kn.ctypes.data,
            0,
            WID,
            HEI,
            buf.ctypes.data,
        )
        za = np.where((den < -0.001) & (rg > 0), rg, 9999).astype("f")
        dense = buf[:, :, 3] > 0.86
        za[dense] = buf[:, :, 4][dense] / (
            buf[:, :, 3][dense] * np.linalg.norm(wr[dense], axis=1)
        )
        return buf[:, :, :3], za, buf[:, :, 3]


@lru_cache(96)
def tex(rgb, alpha):
    im = cv.imread(str(DST / rgb))[:, :, ::-1]
    a = cv.imread(str(DST / alpha), 0)
    return np.dstack((np.uint8(im * (a[:, :, None].astype("f") / 255) + 0.5), a))


@lru_cache(64)
def skytex(path):
    return cv.imread(str(DST / path))[:, :, ::-1]


def rect(a, T, im):
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
    quad = np.array([[0, 0, 1], [w - 1, 0, 1], [w - 1, h - 1, 1], [0, h - 1, 1]]) @ C.T
    poly = []
    for i, b in enumerate(quad):
        aa = quad[i - 1]
        if (aa[2] > 0.32) != (b[2] > 0.32):
            poly.append(aa + (b - aa) * (0.32 - aa[2]) / (b[2] - aa[2]))
        if b[2] > 0.32:
            poly.append(b)
    if len(poly) < 3:
        return None
    poly = np.array(poly)
    uv = poly[:, :2] / poly[:, 2, None] * FOC + [WID / 2 - 0.5, HEI / 2 - 0.5]
    L, A0 = np.floor(uv.min(0) - 1).clip([0, 0], [WID, HEI]).astype(int)
    R, B0 = np.ceil(uv.max(0) + 2).clip([0, 0], [WID, HEI]).astype(int)
    if R <= L or B0 <= A0:
        return None
    M = K @ C
    M[0] -= L * M[2]
    M[1] -= A0 * M[2]
    col = cv.warpPerspective(
        im[y : y + h, x : x + w], M, (R - L, B0 - A0), flags=cv.INTER_LINEAR
    )
    return (L, A0, R, B0, col)


def combine(cards, T, wr, sceneD):
    out = np.zeros((HEI, WID, 4), "f")
    zbuf = np.full((HEI, WID), 9999.0, "f")
    ids = np.zeros((HEI, WID), "i")
    paint = []
    for idd, ls in cards.items():
        imgs = []
        for a, f, w in ls:
            if w < 0.0001:
                continue
            q = rect(a, T, tex(f["rgb"], f["alpha"]))
            if q:
                q = (*q[:4], q[4].astype("f") * np.array([*f["gain"], 1.0], "f"))
            if q:
                imgs.append((*q, w))
        if not imgs:
            continue
        l = min(k[0] for k in imgs)
        t = min(k[1] for k in imgs)
        r = max(k[2] for k in imgs)
        b = max(k[3] for k in imgs)
        n = np.array(ls[0][0]["plane"][:2], "f")
        d = ls[0][0]["plane"][2] - T[:2, 3] @ n
        region = np.s_[t:b, l:r]
        den = wr[region][:, :, :2] @ n
        z = np.divide(d, den, out=np.full(den.shape, 9999, "f"), where=abs(den) > 0.025)
        image = np.zeros((b - t, r - l, 4), "f")
        for L, U, R, Bb, col, w in imgs:
            image[U - t : Bb - t, L - l : R - l] += w * col
        vis = (image[:, :, 3] > 1) & (z > 0.3) & (z < sceneD[region] + 0.30)
        zwhere = z[vis]
        if len(zwhere):
            paint.append((float(np.median(zwhere[::8])), idd, region, z, image, vis))
    for _, idd, region, z, im, valid in sorted(paint, key=lambda x: -x[0]):
        valid &= z <= zbuf[region] + 0.06
        p = out[region]
        a = valid * im[:, :, 3] / 255
        p[:, :, :3] = (
            p[:, :, :3] * (1 - a[:, :, None]) + valid[:, :, None] * im[:, :, :3]
        )
        p[:, :, 3] = p[:, :, 3] * (1 - a) + a
        solid = a > 0.5
        zbuf[region][solid] = z[solid]
        ids[region][solid] = idd
    return out, zbuf, ids


class Popup:
    def __init__(self):
        self.cam = ExportedChase()
        self.road = Road(self.cam)
        self.bank = {c: json.load(open(DST / (c + "-final.json"))) for c in LENSES}
        self.cache = {}
        self.mid = {}
        self.odom = np.r_[
            0, np.cumsum(np.linalg.norm(np.diff(self.cam.raw[:, :2], axis=0), axis=1))
        ]
        self.mtimes = {}
        for c, cs in self.bank.items():
            es = cs["base"]
            self.mid[c] = np.array([np.array(e["rowPoses"])[:, :3].mean(0) for e in es])
            self.mtimes[c] = np.array([e["t"] for e in cs["movers"]])

    def track(self, c, az):
        key = (c, az)
        if key in self.cache:
            return self.cache[key]
        es = self.bank[c]["base"]
        O = self.mid[c][:, :2]
        ds = np.diff(O, axis=0)
        sd = np.maximum(0.00001, (ds * ds).sum(1))
        s = np.array([e["s"] for e in es])
        out = []
        ts = np.arange(80 * 30, 138 * 30 + 1) / 30
        for t in ts:
            T = self.cam.pose(t, az)[0]
            E = T[:2, 3]
            f = np.clip(((E - O[:-1]) * ds).sum(1) / sd, 0, 1)
            dd = np.linalg.norm(O[:-1] + f[:, None] * ds - E, axis=1)
            dd += np.where(abs(s[:-1] - self.odom[round(t * 30)]) < 62, 0, 1e4)
            j = np.argmin(dd)
            out.append(j + f[j])
        out = gaussian_filter1d(out, 3, mode="nearest")
        self.cache[key] = out
        return out

    def pair(self, c, t, az):
        q = self.track(c, az)[int(round((t - 80) * 30))]
        es = self.bank[c]["base"]
        i = min(int(q), len(es) - 2)
        f = float(np.clip(q - i, 0, 1))
        return [(es[i], 1 - f), (es[i + 1], f)]

    def draw(self, t, az=0, raw=None):
        T, st, _ = self.cam.pose(t, az)
        wr = RAYS @ T[:3, :3].astype("f").T
        pairs = {c: self.pair(c, t, az) for c in LENSES}
        fwd = sum(rowmat(e, np.array([0.5]))[0][0, :, 2] * w for e, w in pairs["front"])
        ax = norm(fwd[:2])
        ang = np.arctan2(
            wr[:, :, 1] * ax[0] - wr[:, :, 0] * ax[1],
            wr[:, :, 0] * ax[0] + wr[:, :, 1] * ax[1],
        )
        ang = np.rad2deg(ang)
        front = 1 - smooth(
            (abs(ang) - 19) / 4
        )  # reserve the central third even when the front faces away on a turn
        front = np.maximum(
            front, 1 - smooth((abs((X + 0.5 - WID / 2) / (WID / 2)) - 0.24) / 0.11)
        )
        weights = {
            "front": front,
            "left_pillar": (1 - front) * (ang >= 0),
            "right_pillar": (1 - front) * (ang < 0),
        }
        sky = np.zeros((HEI, WID, 3), "f")
        # Where a lens does not see the requested direction, use the photographic panorama rather than stretching its border.
        coarse = wr[4::8, 4::8].astype("f")
        unit = coarse / np.linalg.norm(coarse, axis=2)[:, :, None]
        ph, pw = self.road.sky.shape[:2]
        pu = (0.5 - np.arctan2(unit[:, :, 1], unit[:, :, 0]) / (2 * np.pi)) * pw - 0.5
        pv = np.clip((0.5 - np.arcsin(unit[:, :, 2]) / np.pi) * ph - 0.5, 0, ph * 0.49)
        pan = cv.resize(
            cv.remap(self.road.sky, pu, pv, cv.INTER_LINEAR, borderMode=cv.BORDER_WRAP),
            (WID, HEI),
        )
        pano = smooth((unit[:, :, 2] + 0.03) / 0.2)
        pano = cv.resize(pano, (WID, HEI))[:, :, None]
        pan = self.road.fog + (pan - self.road.fog) * pano
        sky[:] = 0
        support = np.zeros((HEI, WID), "f")
        far = np.zeros((HEI, WID, 4), "f")
        cyl = FAR / np.maximum(0.1, np.linalg.norm(coarse[:, :, :2], axis=2))
        world = coarse * cyl[:, :, None] + T[:3, 3]
        cz = cv.resize(cyl, (WID, HEI))
        for c, pair in pairs.items():
            wgt = weights[c]
            if wgt.max() < 0.001:
                continue
            for e, w in pair:
                if w < 0.0001:
                    continue
                rot, org = rowmat(e, np.array([0.5]))
                rot = rot[0]
                C = rig[c]
                uv, ok = C.project(coarse @ rot)
                im = skytex(e["sky"])
                uv = (
                    np.nan_to_num(uv).astype("f")
                    * np.float32([im.shape[1] / C.width, im.shape[0] / C.height])
                    - 0.5
                )
                edge = np.minimum.reduce(
                    [
                        uv[:, :, 0],
                        im.shape[1] - 1 - uv[:, :, 0],
                        uv[:, :, 1],
                        im.shape[0] - 1 - uv[:, :, 1],
                    ]
                )
                supported = cv.resize(smooth(edge / 12) * ok, (WID, HEI))
                uv = cv.resize(uv, (WID, HEI))
                col = cv.remap(
                    im, uv, None, cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE
                ).astype("f") * np.array(e["gain"], "f")
                sup = w * wgt * supported
                sky += col * sup[:, :, None]
                support += sup
                h = e["horizon"]
                if h:
                    x, y, X1, Y1 = h["box"]
                    uv, ok = C.project((world - org[0]) @ rot)
                    uv = np.nan_to_num(uv).astype("f") * np.float32(
                        [B / C.width, H / C.height]
                    ) - np.float32([x + 0.5, y + 0.5])
                    uv = cv.resize(uv, (WID, HEI))
                    v = cv.remap(tex(h["rgb"], h["alpha"]), uv, None, cv.INTER_LINEAR)
                    far += (
                        v
                        * (w * wgt * cv.resize(ok.astype("f"), (WID, HEI)))[:, :, None]
                        * np.array([*e["gain"], 1.0], "f")
                    )
        # Normalized convolution over projected sky at infinity fills unsampled directions.
        # The panorama supplies only a very weak prior, never a separate rectangular patch.
        sz = (WID // 8, (HEI + 7) // 8)
        sc = cv.resize(sky, sz, interpolation=cv.INTER_AREA)
        ma = cv.resize(support, sz, interpolation=cv.INTER_AREA)
        base = cv.resize(pan.astype("f"), sz)
        mean = sc.sum((0, 1)) / max(1.0, ma.sum())
        base = 0.85 * mean + 0.15 * base
        for sigma, conf in [(20.0, 0.08), (7.0, 0.15), (2.5, 0.24)]:
            den = cv.GaussianBlur(ma, (0, 0), sigma)
            val = cv.GaussianBlur(sc, (0, 0), sigma) / np.maximum(
                0.001, den[:, :, None]
            )
            blend = smooth(den / conf)[:, :, None]
            base += blend * (val - base)
        base = cv.resize(base, (WID, HEI))
        detail = smooth(support / 0.92)[:, :, None]
        sky = base + (sky / np.maximum(0.001, support[:, :, None]) - base) * detail
        bg, groundD, gA = self.road.layer(
            t, T, self.road.points(t) if raw is None else raw, sky, wr, st
        )
        show = cz < groundD + 0.3
        fa = far[:, :, 3] * show / 255
        bg = bg * (1 - fa[:, :, None]) + far[:, :, :3] * show[:, :, None]
        groundD = np.where(fa > 0.5, np.minimum(groundD, cz), groundD)
        res = bg.copy()
        depth = groundD.copy()
        idmap = np.where(fa > 0.5, 900000, 0).astype("i")
        staticA = fa.copy()
        solidScore = np.zeros_like(depth)
        current = {}
        move_time = {}
        for ci, c in enumerate(LENSES):
            m = self.bank[c]["movers"]
            i = np.argmin(abs(self.mtimes[c] - t))
            f = m[i]
            move_time[c] = f["t"]
            if abs(f["t"] - t) < 0.30:
                for a in f["layers"]:
                    n = np.array(a["plane"][:2])
                    d = a["plane"][2]
                    u, z, U, Z = a["extent"]
                    cen = np.r_[
                        n * d + np.array([-n[1], n[0]]) * (u + U) * 0.5, (z + Z) * 0.5
                    ]
                    area = a["rect"][2] * a["rect"][3]
                    current[(ci, a["id"])] = (a, f, cen, area)
            wgt = weights[c]
            if wgt.max() < 0.001:
                continue
            cards = defaultdict(list)
            for e, w in pairs[c]:
                for a in e["layers"]:
                    cards[(ci + 1) * 100000 + a["id"]].append((a, e, w))
            im, z, ids = combine(cards, T, wr, groundD)
            a = im[:, :, 3] * wgt
            res += im[:, :, :3] * wgt[:, :, None] - bg * a[:, :, None]
            staticA += a * (1 - fa)
            take = (a > solidScore) & (ids != 0)
            solidScore[take] = a[take]
            depth[take] = z[take]
            idmap[take] = ids[take]
        # Coalesce same current-time vehicle across lenses by actual ground contact; draw once.
        chosen = []
        movers = {}
        for (ci, idd), (a, e, cen, area) in sorted(
            current.items(), key=lambda kv: -kv[1][3]
        ):
            if any(
                j != ci
                and np.linalg.norm(other[:2] - cen[:2]) < 2.8
                and abs(cen[2] - other[2]) < 1.6
                for j, other in chosen
            ):
                continue
            chosen.append((ci, cen))
            movers[2000000 + ci * 100000 + idd] = [(a, e, 1.0)]
        if movers:
            im, z, ids = combine(movers, T, wr, depth)
            a = im[:, :, 3]
            res = res * (1 - a[:, :, None]) + im[:, :, :3]
            take = ids != 0
            depth[take] = z[take]
            idmap[take] = ids[take]
        col = np.uint8(np.clip(res + 0.5, 0, 255))
        out = box(col, st, T, depth)
        return out, dict(
            T=T,
            depth=depth,
            ids=idmap,
            alpha=staticA,
            ground=gA,
            frontPair=[e["n"] for e, w in pairs["front"]],
            frac=pairs["front"][1][1],
            frontTime=sum(e["t"] * w for e, w in pairs["front"]),
            moverTimes=move_time,
        )
