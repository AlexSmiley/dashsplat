"""Image and temporal evaluation of actual PLYs; no photographic backdrop or proxy sky coverage."""

from .v4_render import *
import shutil

OUT = RS / "pipeline/livemap/report/v4"
WORK = RS / "work/livemap"
MOMENTS = [
    (93.7, "exit", None),
    (112.6, "first-road", None),
    (135, "arterial", None),
    (102.82, "crossing-car", "front"),
    (131.78, "passing-bus", "left_pillar"),
]


def variants():
    root = RS / "scene/live-v4"
    return [View(RS / "scene/live-v3", root / "sky.jpg"), View(root)]


def check(c):
    scored = json.loads((RS / "pipeline/livemap/measurements.json").read_text())[
        "validation"
    ]["cameras"][c]["live"]["physical_ids"]
    extra = [3329] if c == "left_repeater" else []
    old, new = variants()
    w, h = 480, 310
    items = []
    pan = []
    selected = [
        min(scored, key=lambda n: abs(frame_times(c)[n] - t)) for t in [94, 114, 135]
    ] + extra
    scheduled = json.loads((V4 / "slice-plan.json").read_text())[c]
    assert not set(scored + extra).intersection(
        k for e in scheduled for k in e["window"]
    )
    for n in sorted(scored + extra):
        t = frame_times(c)[n]
        ref = cv.resize(
            cv.imread(str(RS / "data/drive/frames" / c / f"{n:06d}.jpg"))[:, :, ::-1],
            (w, h),
            interpolation=cv.INTER_AREA,
        )
        native = [
            cv.imread(str(RS / "masks" / k / c / f"{n:06d}.png"), cv.IMREAD_UNCHANGED)
            for k in ["sky", "road", "dyn"]
        ]
        sk, ro, dyn = native
        b = lattice(c)[1] & (sk == 0) & ((ro == 0) | (dyn != 0))
        mask = cv.resize(b.astype("B"), (w, h), interpolation=cv.INTER_NEAREST) > 0
        moving = (
            cv.resize(
                (b & (dyn > 0)).astype("B"), (w, h), interpolation=cv.INTER_NEAREST
            )
            > 0
        )
        sky0 = (
            cv.resize(
                (lattice(c)[1] & (sk != 0)).astype("B"),
                (w, h),
                interpolation=cv.INTER_NEAREST,
            )
            > 0
        )
        newrow = [
            label(ref, f"{c} photo {t:.2f} s" + (" | unscored" if n in extra else ""))
        ]
        met = {}
        for s, tag in [(old, "v3"), (new, "v4")]:
            surf = s.image(t, c, size=(w, h))
            im = rgb(surf)
            geom = surf[:, :, 3] > 0.5
            obj = surf[:, :, 5] > 0.45
            se = np.mean((im.astype(float) - ref) ** 2, 2)
            met[tag] = dict(
                pixels=int(mask.sum()),
                geometry=int(np.sum(geom & mask)),
                objects=int(np.sum(obj & mask)),
                movers=int(moving.sum()),
                mover_object=int(np.sum(obj & moving)),
                sky=int(sky0.sum()),
                sky_occluded=int(np.sum(obj & sky0)),
                sse=float(se[mask].sum()),
            )
            newrow.append(
                label(
                    im,
                    f"{tag} | geometry {100 * geom[mask].mean():.0f}% | objects {100 * obj[mask].mean():.0f}%",
                )
            )
        if n in selected:
            pan.append(np.hstack(newrow))
        if n in scored:
            items.append(dict(n=n, t=t, **met))
    save(OUT / f"camera-{c}.jpg", np.vstack(pan))
    res = {"frames": items}
    for tag in ["v3", "v4"]:
        su = lambda k: sum(i[tag][k] for i in items)
        res[tag] = dict(
            non_ground_geometry=su("geometry") / su("pixels"),
            non_ground_objects=su("objects") / su("pixels"),
            photo_psnr=10 * np.log10(255**2 * su("pixels") / su("sse")),
            moving_object_coverage=su("mover_object") / max(1, su("movers")),
            sky_silhouette=su("sky_occluded") / su("sky"),
            min_frame_geometry=min(
                i[tag]["geometry"] / i[tag]["pixels"] for i in items
            ),
        )
    print(c, res["v4"], flush=True)
    return c, res


def encoder(path, w, h, fps, bitrate):
    return subprocess.Popen(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{w}x{h}",
            "-framerate",
            str(fps),
            "-i",
            "-",
            "-c:v",
            "h264_videotoolbox",
            "-b:v",
            bitrate,
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "faststart",
            str(path),
        ],
        stdin=subprocess.PIPE,
    )


def finish(enc, path, frames, fps):
    enc.stdin.close()
    assert enc.wait() == 0
    stream = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_streams",
                "-of",
                "json",
                str(path),
            ]
        )
    )["streams"][0]
    assert (
        int(stream["nb_frames"]) == frames
        and abs(float(stream["duration"]) - frames / fps) < 0.005
    )
    return dict(frames=frames, fps=fps, seconds=frames / fps, bytes=path.stat().st_size)


def moment(spec):
    t, tag, c = spec
    old, s = variants()
    pos = traj("live").T_world_ego(t)[:3, 3]
    pts = s.points(t, pos)
    pan = [
        label(
            rgb(v.image(t, size=(640, 360), pts=pts if v == s else None)),
            f"{name} | chase {t:.2f} s",
        )
        for v, name in [(old, "v3"), (s, "v4")]
    ]
    save(OUT / f"chase-{tag}.jpg", np.hstack(pan))
    if c:
        n = min(frame_times(c), key=lambda n: abs(frame_times(c)[n] - t))
        ref = cv.resize(
            cv.imread(str(RS / "data/drive/frames" / c / f"{n:06d}.jpg"))[:, :, ::-1],
            (480, 310),
        )
        save(
            OUT / f"traffic-{tag}.jpg",
            np.hstack(
                [
                    label(ref, "source " + c),
                    label(rgb(old.image(t, c, size=(480, 310))), "v3"),
                    label(rgb(s.image(t, c, size=(480, 310))), "v4"),
                ]
            ),
        )
    path = WORK / f"orbit-v4-{tag}.mp4"
    enc = encoder(path, 512, 288, 15, "1800k")
    ls = []
    for i in range(180):
        im = rgb(s.image(t, angle=2 * np.pi * i / 180, size=(512, 288), pts=pts))
        enc.stdin.write(im.tobytes())
        if i % 30 == 0:
            ls.append(label(im, f"{t:.2f} s | {i * 2} deg"))
    info = finish(enc, path, 180, 15)
    if info["bytes"] < 20000000:
        shutil.copy2(path, OUT / path.name)
    save(OUT / f"orbit-{tag}.jpg", np.vstack((np.hstack(ls[:3]), np.hstack(ls[3:]))))
    return tag, dict(t=t, **info)


def chase_pose(t):
    T = traj("live").T_world_ego(t)
    s = unit(np.r_[T[:2, 0], 0.0])
    f = s * np.cos(0.215) - [0, 0, np.sin(0.215)]
    right = unit(np.cross(f, [0, 0, 1.0]))
    return T[:3, 3] - 6 * s + [0, 0, 3], np.stack((right, np.cross(f, right), f), 1)


def movie():
    _, s = variants()
    path = WORK / "chase-v4-94-104.mp4"
    w, h = 768, 432
    enc = encoder(path, w, h, 30, "4500k")
    prev = None
    errs = {"near": [], "far": []}
    sheet = []
    ray = s.rays(None, (w, h))[::4, ::4]
    for i in range(300):
        t = 94 + i / 30
        if i % 15 == 0:
            pts = s.points(t + 0.24, traj("live").T_world_ego(t + 0.24)[:3, 3], 0.27)
        out = s.image(t, size=(w, h), pts=pts)
        im = rgb(out)
        enc.stdin.write(im.tobytes())
        if i in [0, 50, 100, 150, 200, 250]:
            sheet.append(label(cv.resize(im, (384, 216)), f"{t:.2f} s"))
        p, R = chase_pose(t)
        alpha = out[:, :, 3]
        depth = out[:, :, 4] / np.maximum(alpha, 0.001)
        if prev:
            pp, RR, pim, dd, al = prev
            world = pp + (ray @ RR.T) * dd[:, :, None]
            v = (world - p) @ R
            ux = (
                w * 0.8660254 * v[:, :, 0] / np.maximum(v[:, :, 2], 0.01) + w / 2 - 0.5
            ).astype("f")
            uy = (
                w * 0.8660254 * v[:, :, 1] / np.maximum(v[:, :, 2], 0.01) + h / 2 - 0.5
            ).astype("f")
            cur = cv.remap(im, ux, uy, cv.INTER_LINEAR)
            refz = cv.remap(depth, ux, uy, cv.INTER_LINEAR)
            ok = (
                (al > 0.8)
                & (ux > 1)
                & (ux < w - 2)
                & (uy > 1)
                & (uy < h - 2)
                & (abs(refz - np.linalg.norm(v, axis=2)) < np.maximum(0.18, 0.045 * dd))
            )
            err = np.abs(cur.astype(float) - pim).mean(2)
            for k, b in [("near", ok & (dd < 25)), ("far", ok & (dd > 40))]:
                if b.sum() > 16:
                    errs[k].append(
                        [
                            float(err[b].mean()),
                            float((err[b] > 25).mean()),
                            float(np.percentile(err[b], 90)),
                        ]
                    )
        prev = p, R, im[::4, ::4].copy(), depth[::4, ::4].copy(), alpha[::4, ::4].copy()
    info = finish(enc, path, 300, 30)
    save(
        OUT / "chase-sequence.jpg",
        np.vstack((np.hstack(sheet[:3]), np.hstack(sheet[3:]))),
    )
    if info["bytes"] < 20000000:
        shutil.copy2(path, OUT.parent / path.name)
    return dict(
        start=94, **info, warp={k: np.mean(v, 0).tolist() for k, v in errs.items()}
    )


def proof():
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    cam = dict(ThreadPoolExecutor(3).map(check, CAMS))
    orb = dict(ThreadPoolExecutor(3).map(moment, MOMENTS))
    v = movie()
    doc = dict(cameras=cam, orbits=orb, movie=v)
    (V4 / "slice-proof.json").write_text(json.dumps(doc, indent=2))
    print(v, flush=True)


if __name__ == "__main__":
    proof()
