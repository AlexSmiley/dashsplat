"""View published scene tiles. Sky never contributes to depth, alpha or geometry coverage."""

from .v4_build import *
from .render import rgb, save, label, reader


class View:
    def __init__(self, root, pan=None):
        self.root = root
        self.doc = json.loads((root / "map.json").read_text())
        self.sky = cv.imread(str(pan or root / self.doc["sky"]))[:, :, ::-1]
        self.v4 = self.doc.get("sky") is not None
        self.lib = ct.CDLL(str(V4 / "raster.dylib"))
        P = ct.c_void_p
        I = ct.c_int
        self.lib.raster.argtypes = [P, I, ct.c_float, I, P, P, I, I, I, P]

    @lru_cache(128)
    def read(self, file):
        return reader(self.root / file)

    def points(self, t, p, margin=0, tag_transient=False):
        out = []
        for tile in self.doc["tiles"]:
            x, y, _, X, Y, _ = tile["bbox"]
            obj = tile["layer"] == "objects"
            radius = (95 if not obj else 75 if not self.v4 else 1500) + margin * 28
            if (
                tile["t_min"] > t + margin
                or tile["t_max"] < t - margin
                or max(x - p[0], 0, p[0] - X) ** 2 + max(y - p[1], 0, p[1] - Y) ** 2
                > radius**2
            ):
                continue
            a = self.read(tile["file"])
            ii = (
                (a[:, 17] <= t + margin)
                & (a[:, 18] > t - margin)
                & ((a[:, 0] - p[0]) ** 2 + (a[:, 1] - p[1]) ** 2 <= radius**2)
            )
            v = a[ii]
            if len(v):
                out.append(
                    np.column_stack(
                        (
                            v,
                            np.full(
                                len(v),
                                tile["kind"] == "transient" if tag_transient else obj,
                                "f",
                            ),
                        )
                    )
                )
        return (
            np.ascontiguousarray(np.vstack(out), "f") if out else np.empty((0, 25), "f")
        )

    @staticmethod
    @lru_cache(None)
    def rays(c, size):
        w, h = size
        y, x = np.mgrid[:h, :w]
        return (
            rig[c].unproject(
                np.stack(
                    ((x + 0.5) * rig[c].width / w, (y + 0.5) * rig[c].height / h), -1
                )
            )
            if c
            else unit(
                np.stack(
                    (
                        (x + 0.5 - w / 2) / (w * 0.8660254),
                        (y + 0.5 - h / 2) / (w * 0.8660254),
                        np.ones((h, w)),
                    ),
                    -1,
                )
            )
        ).astype("f")

    def image(self, t, c=None, angle=0, size=(640, 360), pts=None):
        w, h = size
        T = traj("live").T_world_ego(t)
        p = T[:3, 3]
        ray = self.rays(c, size)
        if c:
            C = rig[c]
            full = rows("live", c, t)[0]
            rw = full[np.linspace(0, H - 1, 33).round().astype("i")]
            K = np.r_[
                C.params[:4] * [w / C.width, h / C.height, w / C.width, h / C.height],
                C.params[4:],
            ]
            wr = np.einsum(
                "hij,hwj->hwi",
                full[np.linspace(0, H - 1, h).round().astype("i"), :3, :3],
                ray,
            )
        else:
            yaw = np.arctan2(T[1, 0], T[0, 0]) + angle
            spin = np.array([np.cos(yaw), np.sin(yaw), 0])
            fw = spin * np.cos(0.215) - [0, 0, np.sin(0.215)]
            right = unit(np.cross(fw, [0, 0, 1.0]))
            cam = np.eye(4)
            cam[:3, :3] = np.stack((right, np.cross(fw, right), fw), 1)
            cam[:3, 3] = p - 6 * spin + [0, 0, 3]
            rw = np.repeat(cam[None], 33, 0)
            K = np.r_[[w * 0.8660254] * 2, w / 2, h / 2, [0] * 4]
            wr = ray @ cam[:3, :3].T
        ph, pw = self.sky.shape[:2]
        sx = (
            (0.5 - np.arctan2(wr[:, :, 1], wr[:, :, 0]) / (2 * np.pi)) * pw - 0.5
        ).astype("f")
        sy = np.clip(
            (0.5 - np.arcsin(np.clip(wr[:, :, 2], -1, 1)) / np.pi) * ph - 0.5, 0, ph - 1
        ).astype("f")
        bg = cv.remap(self.sky, sx, sy, cv.INTER_LINEAR, borderMode=cv.BORDER_WRAP)
        pts = self.points(t, p) if pts is None else pts
        out = np.zeros((h, w, 6), "f")
        rw = np.ascontiguousarray(rw, "d")
        K = np.ascontiguousarray(K, "d")
        self.lib.raster(
            pts.ctypes.data,
            len(pts),
            t,
            2 if self.v4 else 1,
            rw.ctypes.data,
            K.ctypes.data,
            bool(c),
            w,
            h,
            out.ctypes.data,
        )
        out[:, :, :3] += bg * (1 - out[:, :, 3:4])
        return out
