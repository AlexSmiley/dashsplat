"""Read only published PLYs. No depth image or photographs are used by the renderer."""

from .build import *
from pipeline.road.render import label


class Scene:
    def __init__(self, mode, stage="final"):
        self.root = scene(mode, stage)
        self.mode = mode
        self.index = json.loads((self.root / "map.json").read_text())["tiles"]
        self.lib = ct.CDLL(str(W / "raster.dylib"))
        P = ct.c_void_p
        I = ct.c_int
        self.lib.raster.argtypes = [P, I, ct.c_float, I, P, P, I, I, I, P]

    def local(self, t, p):
        out = []
        for item in self.index:
            x, y, _, X, Y, _ = item["bbox"]
            r = 95 if item["layer"] == "ground" else 75
            if (
                max(x - p[0], 0, p[0] - X) ** 2 + max(y - p[1], 0, p[1] - Y) ** 2
                > r * r
            ):
                continue
            a = reader(self.root / item["file"])
            ii = (
                (a[:, 17] <= t)
                & (a[:, 18] > t)
                & ((a[:, 0] - p[0]) ** 2 + (a[:, 1] - p[1]) ** 2 < r * r)
            )
            aa = a[ii]
            out.append(
                np.column_stack(
                    (aa, np.full(len(aa), int(item["layer"] == "objects"), "f"))
                )
            )
        return np.ascontiguousarray(np.concatenate(out), "f")

    def view(self, t, c=None, angle=0, size=(640, 374), points=None, physical=None):
        w, h = size
        pose = traj(self.mode).T_world_ego(t)
        p = pose[:3, 3]
        rot = pose[:2, 0]
        yaw = np.arctan2(rot[1], rot[0])
        head = np.r_[np.cos(yaw), np.sin(yaw), 0.0]
        if c:
            C = rig[c]
            rw = rows(self.mode, c, t if physical is None else physical)[0][
                np.linspace(0, H - 1, 33).round().astype(int)
            ]
            K = np.r_[
                C.params[:4] * [w / C.width, h / C.height, w / C.width, h / C.height],
                C.params[4:],
            ]
        else:
            spin = np.array([np.cos(yaw + angle), np.sin(yaw + angle), 0.0])
            o = (
                p - 6 * spin + [0, 0, 3.0]
            )  # maintain the specified chase pitch; an orbit looks through the car.
            fw = spin * np.cos(0.215) - [0, 0, np.sin(0.215)]
            right = unit(np.cross(fw, [0, 0, 1.0]))
            down = np.cross(fw, right)
            cam = np.eye(4)
            cam[:3, :3] = np.stack((right, down, fw), 1)
            cam[:3, 3] = o
            rw = np.repeat(cam[None], 33, 0)
            K = np.r_[[w * 0.8660254] * 2, w / 2, h / 2, [0] * 4]
        if points is None:
            points = self.local(t, p)
        out = np.zeros((h, w, 6), "f")
        out[:, :, :3] = [22, 26, 30]
        rw = np.ascontiguousarray(rw, "d")
        K = np.ascontiguousarray(K, "d")
        self.lib.raster(
            points.ctypes.data,
            len(points),
            t,
            self.mode == "live",
            rw.ctypes.data,
            K.ctypes.data,
            bool(c),
            w,
            h,
            out.ctypes.data,
        )
        return out


def rgb(a):
    return np.clip(a[:, :, :3], 0, 255).astype("B")


def save(path, a):
    path.parent.mkdir(parents=True, exist_ok=True)
    for q in [87, 80, 74, 68, 60]:
        buf = cv.imencode(".jpg", a[:, :, ::-1], [cv.IMWRITE_JPEG_QUALITY, q])[1]
        if len(buf) < 500000:
            path.write_bytes(buf.tobytes())
            return
    raise ValueError("oversize image")


def inspect(mode, stage):
    sc = Scene(mode, stage)
    times = [90.3, 94.5, 101.3] if stage == "front" else [94.5, 112.6, 135.0]
    ims = []
    for t in times:
        e = min(
            [x for x in frames()["front"] if x["hold"]], key=lambda x: abs(x["t"] - t)
        )
        t = e["t"]
        im = cv.imread(str(RS / "data/drive/frames/front" / f"{e['n']:06d}.jpg"))
        ref = cv.resize(im, (410, 264), interpolation=cv.INTER_AREA)[:, :, ::-1]
        ims.append(
            np.hstack(
                [
                    label(i, n)
                    for i, n in [
                        (ref, f"held-out {t:.2f}"),
                        (rgb(sc.view(t, "front", size=(410, 264))), "native pose"),
                        (rgb(sc.view(t, size=(465, 264))), stage + " chase"),
                    ]
                ]
            )
        )
    if stage in ["sides", "back"]:
        names = CAMS[1:5] if stage == "sides" else ["back"] * 3
        ims2 = []
        for j, c in enumerate(names):
            target = ([94, 115, 135, 101] if stage == "sides" else [94, 115, 135])[j]
            e = min(
                [e for e in frames()[c] if e["hold"]],
                key=lambda e: abs(e["t"] - target),
            )
            t = e["t"]
            ref = cv.resize(
                cv.imread(str(RS / "data/drive/frames" / c / f"{e['n']:06d}.jpg")),
                (400, 256),
            )[:, :, ::-1]
            ims2.append(
                np.hstack(
                    (
                        label(ref, f"{c} photo {t:.2f}"),
                        label(rgb(sc.view(t, c, size=(400, 256))), "native pose"),
                        label(rgb(sc.view(t, size=(432, 256))), stage + " chase"),
                    )
                )
            )
        save(W / "stages" / f"{mode}-{stage}-added.jpg", np.vstack(ims2))
    save(W / "stages" / f"{mode}-{stage}.jpg", np.vstack(ims))


if __name__ == "__main__":
    import sys

    inspect(*sys.argv[1:])
