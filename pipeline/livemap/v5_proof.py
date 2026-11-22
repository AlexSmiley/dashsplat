"""Compare geometry at the controller's chase pose. No HUD, route ribbon or car proxy."""

from .v5_build import *
from .v4_proof import encoder, finish
from .render import label
from scipy.spatial import cKDTree

PROOF = RS / "pipeline/livemap/report/v5"
WORK = RS / "work/livemap"
WID, HEI = 1440, 900
MOM = [93, 100, 112, 128, 135]


def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


class Camera:
    def __init__(self, root):
        j = json.load(open(root / "drive.json"))
        self.t = np.arange(0, j["duration"] + 0.04, 1 / 30)
        f = lambda n, dt=0: np.interp(
            self.t + dt, j["t"], np.unwrap(j[n]) if n == "yaw" else j[n]
        )
        self.raw = np.column_stack([f(n) for n in ["x", "y", "z", "yaw", "speed"]])
        self.f = self.raw[:, :3].copy()
        self.y = f("yaw", 0.3)
        self.s = f("speed")
        self.s[0] = 0
        for i in range(1, len(self.t)):
            for a, tau in [(self.f, 0.05), (self.y, 0.28), (self.s, 1.2)]:
                a[i] = a[i - 1] + (a[i] - a[i - 1]) * (1 - np.exp(-1 / (30 * tau)))

    def pose(self, t, az=0, elev=15):
        i = round(t * 30)
        pos = self.f[i]
        yaw = self.y[i]
        speed = self.s[i]
        pivot = pos + np.r_[np.cos(yaw) * 1.45, np.sin(yaw) * 1.45, 0.8]
        el = np.deg2rad(elev)
        a = yaw + np.pi + np.deg2rad(az)
        dist = 19.5 * (1 + 0.006 * min(25, max(0, speed)))
        o = pivot + dist * np.array(
            [np.cos(a) * np.cos(el), np.sin(a) * np.cos(el), np.sin(el)]
        )
        fw = unit(pivot - o)
        right = unit(np.cross(fw, [0.0, 0.0, 1.0]))
        up = np.cross(right, fw)
        tilt, pan = np.deg2rad([7.2, 1.1])
        fy = fw * np.cos(tilt) + up * np.sin(tilt)
        uy = up * np.cos(tilt) - fw * np.sin(tilt)
        fw = fy * np.cos(pan) - right * np.sin(pan)
        right = right * np.cos(pan) + fy * np.sin(pan)
        T = np.eye(4)
        T[:3, :3] = np.stack((right, -uy, fw), 1)
        T[:3, 3] = o
        return T, self.raw[i], pivot


def luts(root, tiles):
    index = root / "rsb/index.json"
    grade = json.loads(index.read_text()).get("grade") if index.exists() else None
    if grade:
        return np.array(grade["gains"]), np.array(grade["road"])
    dr = json.load(open(root / "drive.json"))
    p = np.stack([dr[k] for k in ["x", "y", "z"]], 1)
    keep = [0]
    for i in range(len(p)):
        if np.linalg.norm(p[i, :2] - p[keep[-1], :2]) >= 1.5:
            keep.append(i)
    tree = cKDTree(p[keep, :2])
    ss = []
    num = 0
    for tile in tiles:
        a = reader(root / tile["file"])
        a = a[:: max(1, len(a) // 20000)]
        rgb = 0.5 + 0.28209479 * a[:, 6:9]
        l = rgb.mean(1)
        q, i = tree.query(a[:, :2])
        ok = (
            (q <= 2.6)
            & (abs(a[:, 2] - p[np.array(keep)[i], 2]) < 0.6)
            & (a[:, 9] > -0.619)
            & (l >= 0.06)
            & (l <= 0.60)
            & (np.ptp(rgb, axis=1) <= 0.22)
        )
        s = rgb[ok][: 120000 - num]
        ss.append(s)
        num += len(s)
        if num >= 120000:
            break
    med = np.median(np.concatenate(ss), axis=0)
    return np.clip(med.mean() / med, 0.72, 1.4), med


class Render:
    def __init__(self, root, camera):
        self.root = root
        self.cam = camera
        self.doc = json.load(open(root / "map.json"))
        self.tiles = self.doc["tiles"]
        self.g, self.road = luts(root, self.tiles)
        self.road = self.grade(self.road) * 255
        sky = cv.imread(str(root / self.doc["sky"]))[:, :, ::-1]
        small = cv.resize(sky, (256, 128), interpolation=cv.INTER_AREA)
        self.fog = self.grade(small[58:63].mean((0, 1)) / 255) * 255
        self.sky = self.grade(cv.GaussianBlur(sky, (0, 0), 2) / 255) * 255
        self.sky = self.sky.astype("f")
        self.bbox = np.array([x["bbox"] for x in self.tiles])
        self.tmin = np.array([x["t_min"] for x in self.tiles])
        self.tmax = np.array([x["t_max"] for x in self.tiles])
        self.tr = np.array([x["kind"] == "transient" for x in self.tiles])
        self.count = np.array([x["count"] for x in self.tiles])
        self.lib = ct.CDLL(str(V5 / "raster.dylib"))
        P = ct.c_void_p
        I = ct.c_int
        self.lib.raster.argtypes = [P, I, ct.c_float, I, P, P, I, I, I, P]
        self.last = {}
        self.focal = (
            HEI
            * 0.5
            / np.tan(
                np.deg2rad(
                    max(
                        50,
                        np.rad2deg(2 * np.arctan(np.tan(np.deg2rad(32)) / (WID / HEI))),
                    )
                )
                * 0.5
            )
        )
        y, x = np.mgrid[:HEI, :WID]
        self.rays = unit(
            np.stack(
                (
                    (x + 0.5 - WID / 2) / self.focal,
                    (y + 0.5 - HEI / 2) / self.focal,
                    np.ones(x.shape),
                ),
                -1,
            )
        ).astype("f")

    def grade(self, x):
        z = np.clip((x * self.g * 0.97 - 0.07) / 0.93, 0, 1)
        z = (z + smooth(z)) * 0.5
        l = z @ np.array([0.2126, 0.7152, 0.0722])
        return np.maximum(0, l[..., None] + 1.24 * (z - l[..., None]))

    def indices(self, t, margin=0):
        st = self.cam.raw[round(t * 30)]
        f = st[:2] + 25 * np.array([np.cos(st[3]), np.sin(st[3])])
        d = np.linalg.norm(
            np.maximum(0, np.maximum(self.bbox[:, :2] - f, f - self.bbox[:, 3:5])),
            axis=1,
        )
        return (
            (d < 172 + margin * 26)
            & (self.tmin <= t + margin)
            & (self.tmax >= t - margin)
        )

    def points(self, t, margin=0):
        out = []
        for i in np.flatnonzero(self.indices(t, margin)):
            tile = self.tiles[i]
            a = reader(self.root / tile["file"])
            v = a[(a[:, 17] <= t + margin) & (a[:, 18] > t - margin)]
            if len(v):
                out.append(
                    np.column_stack(
                        (v, np.full(len(v), tile["layer"] == "objects", "f"))
                    )
                )
        return (
            np.ascontiguousarray(np.concatenate(out), "f")
            if out
            else np.empty((0, 25), "f")
        )

    def colour(self, t, raw, st):
        a = raw[(raw[:, 17] <= t) & (raw[:, 18] > t)].copy()
        p = st[:2]
        dist = np.linalg.norm(a[:, :2] - p, axis=1)
        focus = p + 25 * np.array([np.cos(st[3]), np.sin(st[3])])
        fd = np.minimum(0.12, 0.3 * (a[:, 18] - a[:, 17]))
        fin = np.clip((t - a[:, 17]) / fd, 0, 1)
        alpha = (
            1
            / (1 + np.exp(-a[:, 9]))
            * smooth(fin)
            * smooth((a[:, 18] - t) / fd)
            * (1 - smooth((dist - 136) / 14))
            * (1 - smooth((np.linalg.norm(a[:, :2] - focus, axis=1) - 160) / 12))
        )
        a[:, 9] = np.log(np.maximum(alpha, 1e-6) / np.maximum(1 - alpha, 1e-6))
        hz = smooth((dist - 50) / 90)[:, None]
        col = self.grade(0.5 + 0.28209479 * a[:, 6:9])
        col += (self.fog / 255 - col) * (hz * (2 - hz))
        a[:, 6:9] = (col - 0.5) / 0.28209479
        s = np.exp(a[:, 10:13])
        far = smooth((dist - 45) / 55)
        a[:, 10:13] = np.log(
            np.minimum(s, (np.median(s, axis=1) * (8 - 5.8 * far))[:, None])
            * (0.85 + 0.15 * fin[:, None])
        )
        return np.ascontiguousarray(a[alpha > 0.002], "f")

    def draw(self, t, angle=0, el=15, raw=None):
        T, st, piv = self.cam.pose(t, angle, el)
        wr = self.rays @ T[:3, :3].astype("f").T
        h, w = self.sky.shape[:2]
        u = (
            (0.5 - np.arctan2(wr[:, :, 1], wr[:, :, 0]) / (2 * np.pi)) * w - 0.5
        ).astype("f")
        v = np.clip(
            (0.5 - np.arcsin(np.clip(wr[:, :, 2], -1, 1)) / np.pi) * h - 0.5,
            0,
            h * 0.495,
        ).astype("f")
        sky = cv.remap(self.sky, u, v, cv.INTER_LINEAR, borderMode=cv.BORDER_WRAP)
        bg = self.fog + (sky - self.fog) * smooth(wr[:, :, 2] / 0.16)[:, :, None]
        o = T[:3, 3]
        z = wr[:, :, 2]
        rg = (st[2] - 0.08 - o[2]) / np.minimum(-1e-4, z)
        ground = (z < -0.001) & (rg > 0)
        dist = np.hypot(
            o[0] + rg * wr[:, :, 0] - st[0], o[1] + rg * wr[:, :, 1] - st[1]
        )
        hz = smooth((dist - 30) / 100)[:, :, None]
        road = self.road * 0.7 + self.fog * 0.3
        ap = road + (self.fog - road) * hz * (2 - hz)
        mix = (ground * (1 - smooth((dist - 162) / 18)))[:, :, None]
        out = np.zeros((HEI, WID, 6), "f")
        out[:, :, :3] = bg + (ap - bg) * mix
        a = self.colour(t, self.points(t) if raw is None else raw, st)
        rw = np.ascontiguousarray(np.repeat(T[None], 33, 0), "d")
        K = np.array([self.focal, self.focal, WID / 2, HEI / 2, 0, 0, 0, 0], "d")
        self.lib.raster(
            a.ctypes.data,
            len(a),
            t,
            3,
            rw.ctypes.data,
            K.ctypes.data,
            0,
            WID,
            HEI,
            out.ctypes.data,
        )
        return np.uint8(np.clip(out[:, :, :3], 0, 255)), out[:, :, 5]

    def load_stats(self, t):
        loaded = np.zeros(len(self.tiles), bool)
        tt = np.arange(84, t + 0.01, 0.2)
        for now in tt:
            st = self.cam.raw[round(now * 30)]
            f = st[:2] + 25 * np.array([np.cos(st[3]), np.sin(st[3])])
            d = np.linalg.norm(
                np.maximum(0, np.maximum(self.bbox[:, :2] - f, f - self.bbox[:, 3:5])),
                axis=1,
            )
            wanted = np.where(
                self.tr,
                (self.tmin <= now + 4.5) & (self.tmax >= now - 0.5),
                self.tmin <= now + 4,
            ) & (d < 172)
            keep = np.where(
                self.tr,
                (self.tmin <= now + 6) & (self.tmax >= now - 3),
                self.tmin <= now + 8,
            ) & (d < 217)
            loaded = (loaded & keep) | wanted
        live = dead = 0
        for i in np.flatnonzero(loaded):
            a = reader(self.root / self.tiles[i]["file"])
            live += int(((a[:, 17] <= t) & (a[:, 18] > t)).sum())
            dead += int((a[:, 18] < t - 3).sum())
        return dict(
            manifest_wanted_or_retained=int(self.count[loaded].sum()),
            alive=live,
            dead_over_3s=dead,
            tiles=int(loaded.sum()),
        )


def jpeg(f, img, q=86):
    cv.imwrite(str(f), img[:, :, ::-1], [cv.IMWRITE_JPEG_QUALITY, q])


def proof(root, stage):
    PROOF.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    cam = Camera(root)
    new = Render(root, cam)
    old = Render(RS / "scene/live-v4", cam)
    res = {}
    if stage == "stills":
        for t in MOM:
            st0, st1 = old.load_stats(t), new.load_stats(t)
            ro, rn = old.points(t), new.points(t)
            res[str(t)] = dict(v4=st0, v5=st1)
            ls = []
            for angle in [-60, 0, 60]:
                a, al = old.draw(t, angle, raw=ro)
                b, bl = new.draw(t, angle, raw=rn)
                img = np.hstack(
                    (
                        label(a, f"v4 | {t} s | orbit {angle:+} deg"),
                        label(b, f"v5 | {t} s | orbit {angle:+} deg"),
                    )
                )
                jpeg(PROOF / f"{t}-{angle:+}.jpg", img)
                ls.append(cv.resize(img, (960, 300), interpolation=cv.INTER_AREA))
                if angle == 0:
                    res[str(t)]["pose"] = cam.pose(t)[0].tolist()
                    res[str(t)]["slab_window_opaque_fraction"] = [
                        float((v[85:175, 485:1085] > 0.35).mean()) for v in (al, bl)
                    ]
            jpeg(PROOF / f"{t}-sheet.jpg", np.vstack(ls))
            print(t, st0, st1, flush=True)
        # A high-angle stress test is additional to, not a substitute for, the specified poses.
        a, _ = old.draw(112, 60, 68)
        b, _ = new.draw(112, 60, 68)
        jpeg(PROOF / "112-high.jpg", np.hstack((a, b)))
        (V5 / "slice-proof.json").write_text(json.dumps(res, indent=2))
        return
    path = WORK / "chase-v5-108-118.mp4"
    enc = encoder(path, WID, HEI, 30, "9500k")
    sheet = []
    for i in range(300):
        t = 108 + i / 30
        if i % 15 == 0:
            raw = new.points(t + 7 / 30, 0.27)
        a, _ = new.draw(t, raw=raw)
        enc.stdin.write(a.tobytes())
        if i % 30 == 0:
            sheet.append(label(cv.resize(a, (384, 240)), f"{t:.1f} s"))
    info = finish(enc, path, 300, 30)
    import shutil

    shutil.copy2(path, PROOF / path.name)
    jpeg(
        PROOF / "sequence.jpg",
        np.vstack([np.hstack(sheet[i : i + 2]) for i in range(0, 10, 2)]),
    )
    (V5 / "slice-movie.json").write_text(json.dumps(info))
    print(info, flush=True)


if __name__ == "__main__":
    proof(
        Path(os.environ.get("V5_DEST", OUT)),
        sys.argv[1] if len(sys.argv) > 1 else "stills",
    )
