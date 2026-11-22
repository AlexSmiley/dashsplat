"""Independent native-fisheye, row-timed renderer; never fits test exposure or excludes holes."""

from .views import *
from pipeline.road.scene import reader
from concurrent.futures import ThreadPoolExecutor
import ctypes as ct, argparse, subprocess
from functools import lru_cache

REPORT = WORK / "report"
PORT = REPORT / "selected"
BG = np.array([22, 26, 30], "f")
DIMS = (362, 235)
NOVEL = (544, 320)
TIMES = [93, 101, 112, 128]


def jpg(p, im, q=88):
    p.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv.imencode(".jpg", im[:, :, ::-1], [cv.IMWRITE_JPEG_QUALITY, q])
    assert ok
    p.write_bytes(buf.tobytes())


def readjpg(p):
    return cv.imread(str(p))[:, :, ::-1]


def label(im, s):
    out = cv.copyMakeBorder(im, 24, 0, 0, 0, cv.BORDER_CONSTANT, value=(23, 26, 29))
    cv.putText(
        out, s, (7, 17), cv.FONT_HERSHEY_SIMPLEX, 0.44, (225, 231, 234), 1, cv.LINE_AA
    )
    return out


class Render:
    def __init__(self, src):
        self.src = src
        self.rast = ct.CDLL(str(RS / "maps/object-v3/raster.dylib")).raster
        P = ct.c_void_p
        I = ct.c_int
        self.rast.argtypes = [P, I, ct.c_float, I, P, P, I, I, I, P]
        self.tiles = []
        p = traj.T_world_ego(np.arange(START, END + 0.01, 0.25))[:, :2, 3]
        for tile in J(src / "map.json")["tiles"]:
            bb = np.array(tile["bbox"])
            d = np.maximum(np.maximum(bb[:2] - p, p - bb[3:5]), 0)
            if (d * d).sum(1).min() > 84**2:
                continue
            a = reader(src / tile["file"])
            geom = np.column_stack(
                [a, np.full(len(a), tile["layer"] == "objects", "f")]
            )
            self.tiles.append((bb, geom))

    @lru_cache(8)
    def local(self, tx, ty):
        p = np.array([tx, ty]) * 12 + 6
        out = []
        for bb, a in self.tiles:
            d = np.maximum(np.maximum(bb[:2] - p, p - bb[3:5]), 0)
            if d @ d < 100**2:
                out.append(a[((a[:, :2] - p) ** 2).sum(1) < 100**2])
        return np.ascontiguousarray(np.concatenate(out), "f")

    def draw(self, t, T, k, size, fish):
        w, h = size
        q = self.local(*np.floor(T[16, :2, 3] / 12).astype(int))
        out = np.zeros((h, w, 6), "f")
        out[:, :, :3] = BG
        T = np.ascontiguousarray(T, "d")
        k = np.ascontiguousarray(k, "d")
        self.rast(
            q.ctypes.data,
            len(q),
            t,
            0,
            T.ctypes.data,
            k.ctypes.data,
            fish,
            w,
            h,
            out.ctypes.data,
        )
        return out

    def real(self, e):
        c, t = e["c"], e["t"]
        C = rig[c]
        w, h = DIMS
        T = T_world_cam(C, t, np.linspace(0, C.height, 33), traj)
        k = np.r_[
            C.params[:4] * [w / C.width, h / C.height, w / C.width, h / C.height],
            C.params[4:],
        ]
        return self.draw(t, T, k, DIMS, 1)

    def orbit(self, t, a, size=NOVEL):
        w, h = size
        T = traj.T_world_ego(t)
        yaw = np.arctan2(T[1, 0], T[0, 0]) + a
        head = np.r_[np.cos(yaw), np.sin(yaw), 0.0]
        up = np.array([0, 0, 1.0])
        fw = head * np.cos(0.215) - up * np.sin(0.215)
        right = np.cross(head, up)
        C = np.eye(4)
        C[:3, :3] = np.stack([right, np.cross(fw, right), fw], 1)
        C[:3, 3] = T[:3, 3] - 6 * head + 3 * up
        return self.draw(
            t,
            np.repeat(C[None], 33, 0),
            np.r_[w * 0.8660254, w * 0.8660254, w / 2, h / 2, np.zeros(4)],
            size,
            0,
        )


@lru_cache(None)
def body(c):
    w, h = DIMS
    x, y = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    C = rig[c]
    return C.mask_at(np.stack([x * C.width / w, y * C.height / h], -1))


def reference(e):
    c, n = e["c"], e["n"]
    fn = "%06d" % n
    im = cv.imread(
        str(RS / "data/drive/frames" / c / (fn + ".jpg")),
        cv.IMREAD_REDUCED_COLOR_4 if c == "front" else cv.IMREAD_REDUCED_COLOR_2,
    )
    rgb = photo.apply(
        c, n, cv.resize(im, DIMS, interpolation=cv.INTER_AREA)[:, :, ::-1]
    )
    valid = body(c).copy()
    for k in ("sky", "dyn"):
        m = cv.imread(str(RS / "masks" / k / c / (fn + ".png")), cv.IMREAD_UNCHANGED)
        assert m is not None
        valid &= (
            cv.resize((m != 0).astype("B"), DIMS, interpolation=cv.INTER_NEAREST) == 0
        )
    return rgb, valid


def scores(ir, render, valid):
    x = ir.astype("f") / 255.0
    y = np.clip(render[:, :, :3] / 255, 0, 1)
    mse = ((x - y)[valid] ** 2).mean()
    mu = lambda z: cv.GaussianBlur(z, (11, 11), 1.5)
    mx, my = mu(x), mu(y)
    vx = mu(x * x) - mx * mx
    vy = mu(y * y) - my * my
    cov = mu(x * y) - mx * my
    ss = ((2 * mx * my + 0.01**2) * (2 * cov + 0.03**2)) / (
        (mx * mx + my * my + 0.01**2) * (vx + vy + 0.03**2)
    )
    ok = cv.erode(valid.astype("B"), np.ones((11, 11), "B")).astype(bool)
    return dict(
        psnr=float(-10 * np.log10(mse)),
        ssim=float(ss[ok].mean()),
        coverage=float((render[:, :, 3][valid] > 0.5).mean()),
        pixels=int(valid.sum()),
    )


def held():
    ground = np.load(RS / "maps/ground-v2/schedule.npz")["ns"]
    plan = J(RS / "maps/object-v3/plan.json")
    out = []
    for i, c in enumerate(CAMS):
        used = set(ground[:, i])
        used |= {n for e in plan[c] if e["train"] for n in e["window"]}
        es = [
            e
            for e in J(RS / "depth/keyframes.json")["frames"][c]
            if START <= e["t_key"] <= END
        ]
        out.extend(
            dict(c=c, n=e["n"], t=e["t"], clean=e["n"] not in used) for e in es[7::8]
        )
    return sorted(out, key=lambda e: e["t"])


def run(src, stage, orbitframes=96):
    s = Render(src)
    root = REPORT / stage
    tic = time.monotonic()
    es = held()

    def one(e):
        im, valid = reference(e)
        a = s.real(e)
        res = scores(im, a, valid)
        stem = "%06d.jpg" % e["n"]
        jpg(root / "camera" / e["c"] / stem, np.clip(a[:, :, :3], 0, 255).astype("B"))
        return dict(**e, **res)

    with ThreadPoolExecutor(3) as ex:
        rows = list(ex.map(one, es))
    savej(root / "metrics.json", rows)
    for t in TIMES:

        def angle(j):
            a = s.orbit(t, j * 2 * np.pi / orbitframes)
            im = np.clip(a[:, :, :3], 0, 255).astype("B")
            jpg(root / "orbit" / str(t) / ("%03d.jpg" % j), im)
            return dict(
                coverage=float((a[:, :, 3] > 0.5).mean()),
                objects=float((a[:, :, 5] > 0.3).mean()),
            )

        with ThreadPoolExecutor(3) as ex:
            values = list(ex.map(angle, range(orbitframes)))
        savej(root / "orbit" / str(t) / "coverage.json", values)
    print(stage, "eval seconds", time.monotonic() - tic, flush=True)


def compare():
    r = {k: J(REPORT / k / "metrics.json") for k in ["before", "after"]}
    out = {}
    for c in CAMS:
        out[c] = {}
        for kind in ("all", "clean"):
            out[c][kind] = {}
            for k in r:
                q = [e for e in r[k] if e["c"] == c and (kind == "all" or e["clean"])]
                out[c][kind][k] = dict(
                    n=len(q),
                    **{
                        v: float(np.mean([e[v] for e in q]))
                        for v in ("psnr", "ssim", "coverage")
                    },
                )
        target = dict(
            front=93,
            back=101,
            left_pillar=101,
            right_pillar=112,
            left_repeater=128,
            right_repeater=93,
        )[c]
        e = min(
            (e for e in r["before"] if e["c"] == c and e["clean"]),
            key=lambda e: abs(e["t"] - target),
        )
        im, _ = reference(e)
        q = [label(im, f"{c}  {e['t']:.2f}s / photo")]
        for k in r:
            a = next(x for x in r[k] if x["c"] == c and x["n"] == e["n"])
            q.append(
                label(
                    readjpg(REPORT / k / "camera" / c / ("%06d.jpg" % e["n"])),
                    f"{k}  {a['psnr']:.2f} dB / {a['ssim']:.3f} SSIM",
                )
            )
        jpg(PORT / ("camera-" + c + ".jpg"), np.hstack(q))
    savej(REPORT / "metrics.json", out)
    for t in TIMES:
        q = [
            label(
                readjpg(REPORT / k / "orbit" / str(t) / "000.jpg"),
                f"{k}  /  chase  {t}s",
            )
            for k in r
        ]
        jpg(PORT / f"chase-{t}.jpg", np.hstack(q))
        a = REPORT / "before/orbit" / str(t) / "%03d.jpg"
        b = REPORT / "after/orbit" / str(t) / "%03d.jpg"
        fn = REPORT / f"orbit-{t}.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-framerate",
                "12",
                "-i",
                str(a),
                "-framerate",
                "12",
                "-i",
                str(b),
                "-filter_complex",
                "hstack=inputs=2",
                "-c:v",
                "libx264",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                str(fn),
            ],
            check=True,
        )
    for t in (101, 128):
        rows = []
        for k in r:
            folder = REPORT / k / "orbit" / str(t)
            n = len(list(folder.glob("*.jpg")))
            xs = [
                label(
                    cv.resize(
                        readjpg(folder / ("%03d.jpg" % (n * j // 4))), (300, 177)
                    ),
                    f"{k}  {t}s / {90 * j} deg",
                )
                for j in range(4)
            ]
            rows.append(np.hstack(xs))
        jpg(PORT / f"orbit-{t}.jpg", np.vstack(rows))
    print(json.dumps(out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["before", "after", "compare"])
    ap.add_argument("--source", type=Path)
    ap.add_argument("--orbit-frames", type=int, default=96)
    args = ap.parse_args()
    compare() if args.stage == "compare" else run(
        args.source, args.stage, args.orbit_frames
    )
