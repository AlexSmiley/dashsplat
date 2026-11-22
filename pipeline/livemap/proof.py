"""Photographic holdout, geometry coverage, chase pairs and 360 degree orbit evidence."""

from .render import *
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import subprocess

REPORT = RS / "pipeline/livemap/report"
MOMENTS = [(93.7, "exit"), (112.6, "first-road"), (135.0, "arterial")]


def compare(c):
    plan = frames()[c]
    j = NAMES.index(c)
    gp = np.load(RS / "maps/ground-v2/schedule.npz")
    gone = set(gp["ns"][:, j])
    sc = {q: Scene(q) for q in ("live", "stitched")}
    out = {}
    pan = []
    hold = [e for e in plan if e["hold"] and e["n"] not in gone]
    assert hold
    chosen = [
        min(hold, key=lambda e: abs(e["t"] - target))["n"] for target in [94, 114, 135]
    ]
    examples = [
        e
        for e in plan
        if e["hold"] and c == "left_repeater" and e["n"] == 3329 and e not in hold
    ]
    for e in sorted(hold + examples, key=lambda e: e["t"]):
        n = e["n"]
        C = rig[c]
        pw, ph = 480, 310
        full = cv.imread(str(RS / "data/drive/frames" / c / f"{n:06d}.jpg"))[:, :, ::-1]
        ref = cv.resize(full, (pw, ph), interpolation=cv.INTER_AREA)
        sky, dyn, road = [
            cv.imread(str(RS / "masks" / k / c / f"{n:06d}.png"), cv.IMREAD_UNCHANGED)
            for k in ["sky", "dyn", "road"]
        ]
        b = lattice(c)[1] & (sky == 0) & (dyn == 0)
        static = cv.resize(
            b.astype("B"), (pw, ph), interpolation=cv.INTER_NEAREST
        ).astype(bool)
        nonground = cv.resize(
            (b & (road == 0)).astype("B"), (pw, ph), interpolation=cv.INTER_NEAREST
        ).astype(bool)
        row = [
            label(
                ref,
                f"{c} | photo {e['t']:.2f} s"
                + (" | not scored" if e in examples else ""),
            )
        ]
        out[n] = {}
        for mode, s in sc.items():
            t = e["t"]
            points = s.local(t, traj(mode).T_world_ego(t)[:3, 3])
            a = s.view(t, c, size=(pw, ph), points=points)
            base = rgb(s.view(t, c, size=(pw, ph), points=points[points[:, -1] == 0]))
            im = rgb(a)
            row.append(label(im, mode + " v3 | held-out camera"))
            col = photo(mode).apply(c, n, ref)
            diff = abs(im.astype(float) - ref)
            cd = abs(im.astype(float) - col)
            drawn = a[:, :, 3] > 0.50
            obj = a[:, :, 5] > 0.45
            mm = nonground & obj
            dist, safe, _, _ = maps(mode, c, n)
            rd = cv.resize(dist, (pw, ph), interpolation=cv.INTER_NEAREST)
            rok = cv.resize(
                safe.astype("B"), (pw, ph), interpolation=cv.INTER_NEAREST
            ).astype(bool)
            dd = a[:, :, 4] / np.maximum(a[:, :, 3], 0.001)
            dg = mm & rok & (rd > 2) & (rd < 40)
            sk = cv.resize(
                (sky > 0).astype("B"), (pw, ph), interpolation=cv.INTER_NEAREST
            ).astype(bool)
            out[n][mode] = dict(
                baseline_nonground_sse=float(
                    np.sum((base[nonground].astype(float) - ref[nonground]) ** 2) / 3
                ),
                depth_checked=int(dg.sum()),
                in_front=int((dg & (dd < rd - np.maximum(0.5, 0.20 * rd))).sum()),
                behind=int((dg & (dd > rd + np.maximum(0.5, 0.2 * rd))).sum()),
                sky=int(sk.sum()),
                sky_overlap=int((sk & obj).sum()),
                static=int(static.sum()),
                static_sse=float((diff[static] ** 2).sum() / 3),
                static_covered=int((drawn & static).sum()),
                nonground=int(nonground.sum()),
                object_pixels=int(mm.sum()),
                nonground_sse=float((diff[nonground] ** 2).sum() / 3),
                object_abs=float(diff[mm].sum() / 3),
                corrected_object_abs=float(cd[mm].sum() / 3),
            )
        if n in chosen or e in examples:
            pan.append(np.hstack(row))
        if e in examples:
            del out[n]
    save(REPORT / f"camera-{c}.jpg", np.vstack(pan))
    met = {}
    for mode in sc:
        a = [v[mode] for v in out.values()]
        sb = lambda k: sum(x[k] for x in a)
        met[mode] = dict(
            ground_only_nonground_psnr=10
            * np.log10(255**2 * sb("nonground") / sb("baseline_nonground_sse")),
            near_depth_compared=sb("depth_checked"),
            render_too_near=sb("in_front") / max(1, sb("depth_checked")),
            render_too_far=sb("behind") / max(1, sb("depth_checked")),
            sky_silhouette_overlap=sb("sky_overlap") / max(1, sb("sky")),
            frames=len(a),
            physical_ids=list(out),
            static_psnr=10 * np.log10(255**2 * sb("static") / sb("static_sse")),
            static_coverage=sb("static_covered") / sb("static"),
            nonground_psnr=10
            * np.log10(255**2 * sb("nonground") / sb("nonground_sse")),
            nonground_object_coverage=sb("object_pixels") / sb("nonground"),
            matched_object_mae=sb("object_abs") / max(1, sb("object_pixels")),
            corrected_mae=sb("corrected_object_abs") / max(1, sb("object_pixels")),
        )
    return c, met


def moment(item, video=True):
    t, tag = item
    sc = [Scene(m) for m in ("live", "stitched")]
    p = [s.local(t, traj(s.mode).T_world_ego(t)[:3, 3]) for s in sc]
    pics = [s.view(t, size=(720, 408), points=q) for s, q in zip(sc, p)]
    save(
        REPORT / f"chase-{tag}.jpg",
        np.hstack(
            [label(rgb(a), f"{s.mode} v3 | chase {t:.1f} s") for s, a in zip(sc, pics)]
        ),
    )
    s = sc[1]
    l = []
    for i in [0, 60, 120, 180, 240, 300]:
        a = s.view(t, angle=np.deg2rad(i), size=(430, 250), points=p[1])
        l.append(label(rgb(a), f"{t:.1f} s | orbit {i} deg"))
    save(REPORT / f"orbit-{tag}.jpg", np.vstack([np.hstack(l[:3]), np.hstack(l[3:])]))
    if video:
        folder = W / "orbits"
        folder.mkdir(exist_ok=True)
        out = folder / (tag + ".mp4")
        enc = subprocess.Popen(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                "960x270",
                "-framerate",
                "18",
                "-i",
                "-",
                "-c:v",
                "h264_videotoolbox",
                "-b:v",
                "3000k",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "faststart",
                str(out),
            ],
            stdin=subprocess.PIPE,
        )
        try:
            for k in range(216):
                pair = []
                for view, pts in zip(sc, p):
                    im = rgb(
                        view.view(
                            t, angle=2 * np.pi * k / 216, size=(480, 270), points=pts
                        )
                    )
                    msg = f"{view.mode}  {t:.1f} s"
                    cv.putText(
                        im,
                        msg,
                        (10, 21),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (14, 17, 20),
                        3,
                        cv.LINE_AA,
                    )
                    cv.putText(
                        im,
                        msg,
                        (10, 21),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (230, 234, 238),
                        1,
                        cv.LINE_AA,
                    )
                    pair.append(im)
                enc.stdin.write(np.hstack(pair).tobytes())
        finally:
            enc.stdin.close()
            status = enc.wait()
        assert status == 0
        info = json.loads(
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_streams",
                    "-select_streams",
                    "v:0",
                    "-of",
                    "json",
                    str(out),
                ]
            )
        )["streams"][0]
        assert (
            int(info["nb_frames"]) == 216
            and float(info["duration"]) == 12
            and info["width"] == 960
            and info["height"] == 270
        )
    return tag, dict(
        time=t,
        live_object_fraction=float(np.mean(pics[0][:, :, 5] > 0.45)),
        stitched_object_fraction=float(np.mean(pics[1][:, :, 5] > 0.45)),
    )


def verify(mode):
    s = Scene(mode)
    gm = json.loads((RS / "scene" / f"road-{mode}-v2/map.json").read_text())
    assert [i for i in s.index if i["layer"] == "ground"] == gm["tiles"]
    cnt = 0
    retired = 0
    hist = np.zeros(3, int)
    terrain = np.load(W / mode / "final-terrain.npz")
    extra_path = W / mode / "final-extra-runs.npz"
    extra = np.load(extra_path) if extra_path.exists() and mode == "stitched" else None
    land_checked = overflow_checked = 0
    for tile in s.index:
        if tile["layer"] == "ground":
            assert os.path.samefile(
                s.root / tile["file"], RS / "scene" / f"road-{mode}-v2" / tile["file"]
            )
            continue
        a = reader(s.root / tile["file"])
        assert (
            tile["kind"] == "static"
            and len(a) == tile["count"]
            and np.isfinite(a).all()
        )
        assert (
            np.max(abs(np.r_[a[:, :3].min(0), a[:, :3].max(0)] - tile["bbox"])) < 0.0001
        )
        assert abs((a[:, 13:17] ** 2).sum(1) - 1).max() < 1e-5
        assert abs((a[:, 3:6] ** 2).sum(1) - 1).max() < 1e-5
        b, d, e, i, j, k, l = a[:, 17:].T
        assert np.all(
            (b >= 0)
            & (d > b)
            & (b <= e)
            & (e <= d)
            & (e <= i)
            & (i <= j)
            & (j <= k)
            & (k <= l)
            & ((j <= d) | (j == INF))
            & ((l <= d) | (l == INF))
        )
        assert np.all((i == INF) == (j == INF)) and np.all((k == INF) == (l == INF))
        cnt += len(a)
        retired += int((d < INF).sum())
        hist += np.bincount((i < INF).astype(int) + (k < INF), minlength=3)
        tile_xy = np.array([int(i) for i in tile["id"].split("_")[1:]])
        ti = terrain["index"][(terrain["tile_xy"] == tile_xy).all(1)]
        land = a[ti]
        land_checked += len(land)
        if len(land):
            rot = Rotation.from_quat(land[:, [14, 15, 16, 13]]).as_matrix()
            sig = np.exp(land[:, 10:12])
            ex = np.linalg.norm(rot[:, :2, :2] * sig[:, None, :], axis=2)
            lo = np.floor((land[:, :2] - ex) * 4.4).astype(int)
            hi = np.floor((land[:, :2] + ex) * 4.4).astype(int)
            span = hi - lo
            for dx in range(span[:, 0].max() + 1):
                for dy in range(span[:, 1].max() + 1):
                    keep = (span[:, 0] >= dx) & (span[:, 1] >= dy)
                    xy = (lo[keep] + [dx + 0.5, dy + 0.5]) / 4.4
                    first, _ = ground(mode, np.column_stack((xy, land[keep, 2])))
                    assert np.all(first > land[keep, 17])
                    assert np.all(land[keep, 18] <= first + 0.251)
        if extra is not None:
            ii = (extra["tile_xy"] == tile_xy).all(1)
            rr = extra["index"][ii]
            x0, x1 = extra["start"][ii], extra["end"][ii]
            assert ((x0 >= a[rr, 21]) & (x1 >= x0) & (x1 <= a[rr, 22])).all()
            overflow_checked += len(rr)
        if mode == "stitched":
            assert np.all(b == 0) & np.all(d == INF)
    assert {
        p.relative_to(s.root).as_posix() for p in (s.root / "tiles").glob("*.ply")
    } == {i["file"] for i in s.index}
    assert land_checked == len(terrain["index"])
    assert extra is None or overflow_checked == len(extra["index"])
    return dict(
        records=cnt,
        retired=retired,
        visibility_runs=hist.tolist(),
        terrain_checked=land_checked,
        overflow_checked=overflow_checked,
    )


def proof(video=True):
    REPORT.mkdir(parents=True, exist_ok=True)
    res = dict(validation={m: verify(m) for m in ("live", "stitched")})
    res["cameras"] = dict(ThreadPoolExecutor(3).map(compare, CAMS))
    res["chase"] = dict(ThreadPoolExecutor(3).map(lambda e: moment(e, video), MOMENTS))
    (W / "proof.json").write_text(json.dumps(res, indent=2))
    print("verified", res["validation"], flush=True)


if __name__ == "__main__":
    proof()
