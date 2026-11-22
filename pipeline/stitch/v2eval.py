"""Independent contract compositor: all layers to 215 m, sky only for unscored display."""

from .evaluate import *
from .v2data import V, GROUND
from . import evaluate as v1, perceptual, verify
from scipy.spatial import cKDTree
import os, shutil, time

RESULT = RS / "work/stitch/report/v2"
RESULT.mkdir(parents=True, exist_ok=True)


@lru_cache(1)
def viewer_cameras():
    return J(RESULT / "viewer-cameras.json")


def viewer_pose(t, key):
    v = viewer_cameras()["sites"][str(t)]["frames"][str(key)]
    return np.asarray(v["T_world_cam"], "d"), v["K"]


def world_ray(T, K, x, y):
    # Broadcast x and y on the physical pixel grid; convert the captured controller matrix from OpenCV axes.
    fx, fy, cx, cy = K
    uc = ((x + 0.5 - cx) / fx).astype("f")
    vc = ((y + 0.5 - cy) / fy).astype("f")
    r = (
        T[:3, 2].astype("f")
        + uc[:, :, None] * T[:3, 0].astype("f")
        + vc[:, :, None] * T[:3, 1].astype("f")
    )
    return r / np.linalg.norm(r, axis=2)[:, :, None]


class Wide(Render):
    def __init__(self, src, ground_only=False):
        self.src = src
        self.rast = ct.CDLL(str(RESULT / "wide.dylib")).raster
        P = ct.c_void_p
        I = ct.c_int
        self.rast.argtypes = [P, I, ct.c_float, I, P, P, I, I, I, P]
        chunks = []
        for t in J(src / "map.json")["tiles"]:
            if ground_only and t["layer"] != "ground":
                continue
            a = reader(src / t["file"])
            chunks.append(
                np.column_stack([a, np.full(len(a), t["layer"] == "objects", "f")])
            )
        a = np.concatenate(chunks)
        key = np.floor(a[:, :3] / 9).astype(int)
        code = (
            ((key[:, 0].astype("i8") + 30000) * 65536 + key[:, 1] + 30000) * 65536
            + key[:, 2]
            + 30000
        )
        o = np.argsort(code)
        b = np.r_[0, np.flatnonzero(np.diff(code[o])) + 1, len(o)]
        self.pos = (key[o[b[:-1]]] + 0.5) * 9
        self.data = a[o]
        self.start = b[:-1]
        self.n = np.diff(b)
        self.radius = 7.8 + 3 * np.maximum.reduceat(
            np.exp(self.data[:, 10:13]).max(1), self.start
        )
        self.sky_path = RS / "scene/live-v4/sky.jpg"
        self.pan = readjpg(self.sky_path) if self.sky_path.exists() else None

    def draw(self, t, T, k, size, fish):
        w, h = size
        v = (self.pos - T[16, :3, 3]) @ T[16, :3, :3]
        x, y, z = v.T
        r = self.radius + 2
        keep = (np.linalg.norm(v, axis=1) < 215 + r) & (z > -r)
        fx, fy, cx, cy = k[:4]
        # Pinhole planes and a conservative bounding cone for KB4. Exact pixel exposure uses 33 poses in the C++ compositor.
        if fish:
            corner = np.hypot(max(cx, w - cx) / fx, max(cy, h - cy) / fy)
            theta = np.linspace(0, np.pi / 2, 250)
            u = theta**2
            rs = theta * (1 + u * (k[4] + u * (k[5] + u * (k[6] + u * k[7]))))
            angle = min(np.pi / 2, theta[min(np.searchsorted(rs, corner), 249)] + 0.11)
            keep &= np.hypot(x, y) * np.cos(angle) - z * np.sin(angle) < r
        else:
            for val, d in [
                (-x, -cx / fx),
                (x, (w - cx) / fx),
                (-y, -cy / fy),
                (y, (h - cy) / fy),
            ]:
                limit = abs(d)
                keep &= val - z * limit < r * np.hypot(1, limit)
        ids = np.where(keep)[0]
        q = (
            np.concatenate(
                [self.data[self.start[j] : self.start[j] + self.n[j]] for j in ids]
            )
            if len(ids)
            else self.data[:0]
        )
        out = np.zeros((h, w, 6), "f")
        out[:, :, :3] = BG
        T = np.ascontiguousarray(T, "d")
        k = np.ascontiguousarray(k, "d")
        self.rast(
            q.ctypes.data,
            len(q),
            t,
            2,
            T.ctypes.data,
            k.ctypes.data,
            fish,
            w,
            h,
            out.ctypes.data,
        )
        return out

    def viewpoint(self, t, key):
        T, K = viewer_pose(t, key)
        return self.draw(
            t, np.repeat(T[None], 33, 0), np.r_[K, np.zeros(4)], (1440, 900), 0
        )

    def display(self, t, key, out):
        h, w = out.shape[:2]
        y, x = np.mgrid[:h, :w]
        T, K = viewer_pose(t, key)
        ray = world_ray(T, K, x, y)
        if self.pan is not None:
            H, W = self.pan.shape[:2]
            mx = (
                (0.5 - np.arctan2(ray[:, :, 1], ray[:, :, 0]) / (2 * np.pi)) * W - 0.5
            ) % W
            my = np.clip((0.5 - np.arcsin(ray[:, :, 2]) / np.pi) * H - 0.5, 0, H - 1)
            bg = cv.remap(
                self.pan,
                mx.astype("f"),
                my.astype("f"),
                cv.INTER_LINEAR,
                borderMode=cv.BORDER_WRAP,
            )
        else:
            bg = np.array([205, 212, 217]) - np.clip(ray[:, :, 2], 0, 1)[:, :, None] * [
                55,
                52,
                42,
            ]
        return np.uint8(
            np.clip(out[:, :, :3] + (1 - out[:, :, 3:4]) * (bg - BG), 0, 255)
        )


def build():
    src = RS / "pipeline/livemap/raster.cpp"
    text = src.read_text()
    assert (
        "if (live < 2 && r > 135)" in text or "r > 135" in text
    )  # mode 2 uses offline lifetimes but range is explicitly limited, same for all stages
    text = (
        text.replace("if (live < 2 && r > 135)", "if (r > 215)")
        .replace("if (r > 135)", "if (r > 215)")
        .replace("if (live)", "if (live == 1)")
    )
    # Preserve v1's native metric compositor. For high-resolution novel poses, include the entire 3-sigma ellipse instead of chopping it at 120 px.
    text = (
        text.replace(
            "if (p.x < -70 || p.x > w + 70 || p.y < -70 || p.y > h + 70)",
            "if (fish && (p.x < -70 || p.x > w + 70 || p.y < -70 || p.y > h + 70))",
        )
        .replace(
            "min(120.f, 3 * sqrtf(A))", "(fish ? min(120.f,3*sqrtf(A)) : 3*sqrtf(A))"
        )
        .replace(
            "min(120.f, 3 * sqrtf(C))", "(fish ? min(120.f,3*sqrtf(C)) : 3*sqrtf(C))"
        )
    )
    text = text.replace(
        "float r = sqrt(v.x * v.x + v.y * v.y + v.z * v.z);",
        """if (!fish) {
      // A pinhole derivative at an off-screen, grazing centre diverges; MetalSplat uses this same FOV-clamped EWA Jacobian. Never smear an off-frustum sphere across the frame.
      float rad=3*expf(max({a[10],a[11],a[12]}));float lx=-K[2]/K[0],rx=(w-K[2])/K[0],ly=-K[3]/K[1],ry=(h-K[3])/K[1];
      if(v.z<.3 || v.x-v.z*lx < -rad*hypotf(1,lx) || v.x-v.z*rx > rad*hypotf(1,rx) || v.y-v.z*ly < -rad*hypotf(1,ly) || v.y-v.z*ry > rad*hypotf(1,ry))continue;
      float ux=.65f*w/K[0],uy=.65f*h/K[1];p.fx[2]=-K[0]/v.z*clamp(v.x/v.z,-ux,ux);p.fy[2]=-K[1]/v.z*clamp(v.y/v.z,-uy,uy);
    }
    float r = sqrt(v.x * v.x + v.y * v.y + v.z * v.z);""",
    )
    f = RESULT / "wide.cpp"
    f.write_text(text)
    subprocess.run(
        [
            "c++",
            "-O3",
            "-std=c++17",
            "-dynamiclib",
            str(f),
            "-o",
            str(RESULT / "wide-new.dylib"),
        ],
        check=True,
    )
    (RESULT / "wide-new.dylib").replace(RESULT / "wide.dylib")


def runstage(src, st, quick=False, ground=False, orbit_only=False):
    tic = time.monotonic()
    r = Wide(src, ground)
    dst = RESULT / st
    names = {"default": 0, "left": 80, "right": 16, "raised": "raised"}
    ks = [0, 80, 16, "raised"] if quick else list(range(96)) + ["raised"]
    areas = []
    for t in TIMES:

        def one(key):
            out = r.viewpoint(t, key)
            im = r.display(t, key, out)
            actual = viewer_cameras()["sites"][str(t)]["frames"][str(key)]
            if isinstance(key, int):
                jpg(dst / "orbit" / str(t) / f"{key:03d}.jpg", im)
            for tag, k in names.items():
                if key == k:
                    jpg(dst / "views" / str(t) / (tag + ".jpg"), im, 92)
            h, w = im.shape[:2]
            sl = np.s_[h // 8 : 3 * h // 4, w // 4 : 3 * w // 4]
            dep = out[:, :, 4] / np.maximum(0.001, out[:, :, 3])
            rec = dict(
                t=t,
                key=key,
                az=actual["az"],
                elev=actual["elev"],
                coverage=float((out[:, :, 3] > 0.5).mean()),
                center_near_object=float(
                    ((out[:, :, 5][sl] > 0.5) & (dep[sl] < 6)).mean()
                ),
            )
            if t == 112 and key == 0:
                np.savez(
                    dst / "road112.npz",
                    rgb=out[:, :, :3],
                    alpha=out[:, :, 3],
                    objects=out[:, :, 5],
                )
                np.savez_compressed(
                    dst / "sky112.npz",
                    opacity=out[:400, :, 5].astype("float16"),
                    distance=dep[:400].astype("float16"),
                )
            return rec

        with ThreadPoolExecutor(4) as ex:
            areas.extend(ex.map(one, ks))
    savej(dst / "orbit.json", areas)
    print(st, "orbits done", time.monotonic() - tic, flush=True)
    if not quick and not orbit_only:

        def physical(e):
            ref, valid = reference(e)
            out = r.real(e)
            jpg(
                dst / "camera" / e["c"] / ("%06d.jpg" % e["n"]),
                np.clip(out[:, :, :3], 0, 255).astype("B"),
            )
            m = scores(ref, out, valid)
            return dict(**e, **m)

        with ThreadPoolExecutor(5) as ex:
            savej(dst / "metrics.json", list(ex.map(physical, held())))
    savej(
        dst / "timing.json",
        dict(
            seconds=time.monotonic() - tic,
            sky="live-v4/sky.jpg" if r.pan is not None else "neutral overcast gradient",
        ),
    )
    print(st, "done", time.monotonic() - tic, flush=True)


def boards(stages, quick=False):
    dst = RESULT / ("review-" + stages[-1] if quick else "selected")
    for t in TIMES:
        rows = []
        for st in stages:
            tiles = [
                label(
                    cv.resize(
                        readjpg(RESULT / st / f"views/{t}/{name}.jpg"),
                        (480, 300),
                        interpolation=cv.INTER_AREA,
                    ),
                    f"{st} / {t}s / {tag}",
                )
                for name, tag in [
                    ("default", "default"),
                    ("left", "ego-left 60deg"),
                    ("right", "ego-right 60deg"),
                    ("raised", "40deg elevation"),
                ]
            ]
            rows.append(np.hstack(tiles))
        jpg(dst / f"views-{t}.jpg", np.vstack(rows), 86)
        if not quick:
            args = []
            for s in stages:
                args += [
                    "-framerate",
                    "12",
                    "-i",
                    str(RESULT / s / f"orbit/{t}/%03d.jpg"),
                ]
            subprocess.run(
                ["ffmpeg", "-loglevel", "error", "-y"]
                + args
                + [
                    "-filter_complex",
                    f"hstack=inputs={len(stages)}",
                    "-c:v",
                    "libx264",
                    "-crf",
                    "20",
                    "-pix_fmt",
                    "yuv420p",
                    str(RESULT / f"orbit-{t}.mp4"),
                ],
                check=True,
            )
    for t, view, roi in [
        (112, "default", (0, 30, 1440, 480)),
        (101, "raised", (160, 210, 1280, 770)),
    ]:
        x, y, x1, y1 = roi
        rows = [
            label(
                readjpg(RESULT / st / f"views/{t}/{view}.jpg")[y:y1, x:x1],
                f"{st} / {t}s / {view} / native-resolution crop",
            )
            for st in stages
        ]
        jpg(dst / f"detail-{t}.jpg", np.vstack(rows), 84)
    if quick:
        return
    rows = {s: J(RESULT / s / "metrics.json") for s in stages}
    summary = {}
    for c, t in zip(CAMS, [93, 101, 101, 112, 128, 93]):
        summary[c] = {}
        ee = min(
            (v for v in held() if v["c"] == c and v["clean"]),
            key=lambda v: abs(v["t"] - t),
        )
        if abs(ee["t"] - t) > 0.9:
            ee = min((v for v in held() if v["c"] == c), key=lambda v: abs(v["t"] - t))
        ref = reference(ee)[0]
        ims = [label(ref, f"{c} / photo / {ee['t']:.2f}s")]
        for st in stages:
            aa = [a for a in rows[st] if a["c"] == c]
            v = next(a for a in aa if a["n"] == ee["n"])
            ims.append(
                label(
                    readjpg(RESULT / st / "camera" / c / ("%06d.jpg" % v["n"])),
                    f"{st} / {v['psnr']:.2f} dB",
                )
            )
            summary[c][st] = {
                q: {
                    "n": len(seq),
                    **{
                        k: float(np.mean([a[k] for a in seq]))
                        for k in ["psnr", "ssim", "coverage"]
                    },
                }
                for q, seq in [("all", aa), ("clean", [a for a in aa if a["clean"]])]
            }
        jpg(dst / f"camera-{c}.jpg", np.hstack(ims))
    savej(RESULT / "metrics.json", summary)
    aliases = RESULT / "perceptual"
    aliases.mkdir(exist_ok=True)
    for s, k in zip(stages, ["before", "after"]):
        p = aliases / k
        if not p.exists():
            p.symlink_to(RESULT / s, target_is_directory=True)
    perceptual.REPORT = aliases
    perceptual.run()
    verify.WORK = RESULT
    verify.run(RS / "scene/stitched-opt-v2")
    truth = np.load(RESULT / "ground/road112.npz")
    y, x = np.mgrid[:900, :1440]
    mask = (y > 560) & (x > 270) & (x < 1120) & (truth["alpha"] > 0.95)
    d = {}
    for st in stages + ["ground"]:
        a = np.load(RESULT / st / "road112.npz")
        gray = cv.cvtColor(a["rgb"], cv.COLOR_RGB2GRAY)
        eg = np.hypot(cv.Sobel(gray, -1, 1, 0), cv.Sobel(gray, -1, 0, 1))
        d[st] = dict(
            n=int(mask.sum()),
            rgb_mae_to_mosaic=float(abs(truth["rgb"] - a["rgb"])[mask].mean()),
            gradient_mean=float(eg[mask].mean()),
            gradient_p90=float(np.percentile(eg[mask], 90)),
            object_opacity=float(a["objects"][mask].mean()),
        )
    savej(RESULT / "road112.json", d)
    road_boards = [
        label(
            readjpg(RESULT / st / "views/112/default.jpg")[560:900, 270:1120],
            "112 s / " + ("original fused mosaic" if st == "ground" else st),
        )
        for st in ["ground", *stages]
    ]
    jpg(RESULT / "mosaic-comparison-112.jpg", np.vstack(road_boards), 94)
    # Fixed before final rendering, from the reference viewer's interval between the two lamp heads.
    box = [490, 85, 840, 170]
    counts = {}
    x, y, xx, yy = box
    for st in stages:
        a = np.load(RESULT / st / "sky112.npz")["opacity"].astype("f")[y:yy, x:xx]
        m = (a > 0.15).astype("B")
        n, _, stats, _ = cv.connectedComponentsWithStats(m)
        counts[st] = dict(
            mean_opacity=float(a.mean()),
            p95=float(np.percentile(a, 95)),
            fraction_over_15pct=float(m.mean()),
            largest_connected_width_px=int(stats[1:, cv.CC_STAT_WIDTH].max())
            if n > 1
            else 0,
        )
    savej(RESULT / "streetlight-gap112.json", dict(native_box_xyxy=box, models=counts))


if __name__ == "__main__":
    import sys

    a = sys.argv[1:]
    if a[0] == "build":
        build()
    elif a[0] == "boards":
        boards(a[1:3], len(a) > 3)
    else:
        runstage(
            Path(a[1]), a[0], "quick" in a[2:], a[0] == "ground", "orbits" in a[2:]
        )
