"""Render independent, contiguous frame ranges, encode short GOPs and publish a verified scene-relative clip."""

from .morph_chase import *
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp, fcntl, uuid


@lru_cache(None)
def clip_clock(a, b):
    frames = track(a, b)
    arc = choose(frames)
    t = np.array([x[0] for x in frames])
    plan = None
    if b <= 110:
        plan = turn_plan(a, b)
    if (a, b) == (93, 110):
        # Keep the source trajectories continuous when independently filtered files meet.
        head = 1 - smooth((t - a) / 1.1)
        tail = 1 - smooth((b - t) / 1.1)
        left = turn_plan(85, 93)
        rightArc = choose(track(110, 125))[0]
        arc = (
            arc
            + head * (choose(track(85, 93))[-1] - arc[0])
            + tail * (rightArc - arc[-1])
        )
        plan = {
            c: (
                p + head * (left[c][0][-1] - p[0]),
                (w + head * (left[c][1][-1] - w[0])) * (1 - tail),
            )
            for c, (p, w) in plan.items()
        }
    elif a == 125:
        arc = arc + (1 - smooth((t - a) / 1.1)) * (choose(track(110, 125))[-1] - arc[0])
    return arc, ((a, plan) if plan is not None else None)


def require_ghost_gate(file, n):
    digest = hashlib.sha256(file.read_bytes()).hexdigest()
    audit = W / "release-gates" / f"{digest}.json"
    if not audit.exists():
        raise RuntimeError(
            f"Not published: {file} has no after-image audit for its encoded SHA-256 {digest}."
        )
    r = J(audit)
    if not (
        r["encodedSHA256"] == digest
        and r["frames"] == n
        and r["framesAboveSolo"] == 0
        and r["structureFramesAboveSolo"] == 0
    ):
        raise RuntimeError(
            f"Not published: {file} exceeds its single-photograph control ({audit})."
        )


def segment(args):
    a, b, lo, hi = args
    ts = track(a, b)
    arc, plan = clip_clock(a, b)
    out = W / f"{a}-{b}"
    v = out / f"part-{lo:04}.mp4"
    proc = encode(v, OW, OH)
    metrics = []
    try:
        for i in range(lo, hi):
            t, T = ts[i]
            im, d = render(
                t,
                T,
                arc[i],
                i % 15 == 0 or i == hi - 1,
                plan,
            )
            proc.stdin.write(im.tobytes())
            metrics.append(d)
            if i % 30 == 0 or i == len(ts) - 1:
                cv.imwrite(str(out / f"{t:.2f}.jpg"), im, [cv.IMWRITE_JPEG_QUALITY, 95])
    finally:
        proc.stdin.close()
        rc = proc.wait()
    assert rc == 0
    print("encoded", a, lo, hi, flush=True)
    return v, metrics


def run(a, b, boundary=False):
    ts = track(a, b)
    N = len(ts)
    out = W / f"{a}-{b}"
    out.mkdir(exist_ok=True)
    cuts = list(range(0, N, 50))
    args = [(a, b, i, min(i + 50, N)) for i in cuts]
    if boundary:
        assert (a, b) in [(93, 110), (125, 140)]
        args = [q for q in args if q[2] == 0 or (a == 93 and q[3] > N - 34)]
    with ProcessPoolExecutor(5, mp_context=mp.get_context("spawn")) as pool:
        res = list(pool.map(segment, args))
    stats = J(out / "film.json") if boundary else [None] * N
    for (_, _, lo, hi), (_, items) in zip(args, res):
        stats[lo:hi] = items
    savej(out / "film.json", stats)
    (out / "concat.txt").write_text(
        "".join(f"file 'part-{lo:04}.mp4'\n" for lo in cuts)
    )
    file = out / "chase-candidate.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(out / "concat.txt"),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(file),
        ],
        check=True,
    )
    probe = Jprobe(file)
    assert (
        probe["codec_name"],
        probe["width"],
        probe["height"],
        int(probe["nb_frames"]),
        probe["pix_fmt"],
    ) == ("h264", OW, OH, N, "yuv420p")
    assert abs(float(probe["duration"]) - N / 30) < 0.015
    require_ghost_gate(file, N)
    g = HOST / "generated"
    g.mkdir(exist_ok=True)
    asset = f"generated/morph-photo-{a}-{b}.mp4"
    dst = HOST / asset
    unique = g / f".morph-{uuid.uuid4().hex}.mp4"

    path = HOST / "generated.json"
    with open(W / "publish.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        doc = J(path) if path.exists() else dict(clips=[])
        ls = [p for p in doc["clips"] if p.get("file") != asset]
        packed = "generated/morph-photo-85-140.mp4"
        inside_packed = any(
            p["file"] == packed and p["t0"] <= a <= b <= p["t1"] for p in ls
        )
        assert not any(
            max(a, p["t0"]) < min(b, p["t1"])
            for p in ls
            if p["file"] != packed or not inside_packed
        ), "overlaps another producer clip"
        os.link(file, unique)
        os.replace(unique, dst)
        os.replace(file, out / "chase.mp4")
        file = out / "chase.mp4"
        if not inside_packed:
            ls.append(
                dict(file=asset, t0=a, t1=b, fps=30, pose="chase", vfov=50, ego="free")
            )
            doc["clips"] = sorted(ls, key=lambda x: x["t0"])
            savej(path, doc)
    summary = dict(
        scene="stitched-opt-v3",
        clip=asset,
        trackSHA256=hashlib.sha256(
            (W / f"chase-{a}-{b}.json").read_bytes()
        ).hexdigest(),
        sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
        bytes=file.stat().st_size,
        photoMean=float(np.mean([f["photo"] for f in stats])),
        probe=probe,
    )
    savej(out / "receipt.json", summary)
    print(summary, flush=True)


def Jprobe(file):
    return json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,width,height,nb_frames,duration,r_frame_rate,pix_fmt",
                "-of",
                "json",
                str(file),
            ]
        )
    )["streams"][0]


def continuous():
    # The browser fades to splats between video elements. One clip, without duplicate endpoint frames.
    spans = [(85, 93), (93, 110), (110, 125), (125, 140)]
    out = W / "85-140"
    out.mkdir(exist_ok=True)
    file = out / "chase-candidate.mp4"
    docs = [J(W / f"{a}-{b}" / "receipt.json") for a, b in spans]
    inputs = [HOST / d["clip"] for d in docs]
    assert all(
        hashlib.sha256(p.read_bytes()).hexdigest() == r["sha256"]
        for p, r in zip(inputs, docs)
    )
    joints = []
    for k in range(1, len(inputs)):
        a, b = spans[k - 1]
        _, c = spans[k]
        j = (b - a) * 30
        left = J(W / f"{a}-{b}" / "film.json")[j]
        right = J(W / f"{b}-{c}" / "film.json")[0]
        assert left["uses"] == right["uses"] and np.array_equal(
            track(a, b)[-1][1], track(b, c)[0][1]
        )
        images = []
        for p, idx in [(inputs[k - 1], j), (inputs[k], 0)]:
            v = cv.VideoCapture(str(p))
            v.set(cv.CAP_PROP_POS_FRAMES, idx)
            ok, im = v.read()
            assert ok
            images.append(im)
            v.release()
        joints.append(
            dict(
                t=b,
                meanAbsoluteRGB=float(np.mean(abs(images[0].astype("f") - images[1]))),
            )
        )
    args = ["ffmpeg", "-loglevel", "error", "-y"]
    fil = []
    for k, (a, b) in enumerate(spans):
        args += ["-i", str(inputs[k])]
        n = (b - a) * 30 + int(k == len(spans) - 1)
        fil.append(f"[{k}:v]trim=end_frame={n},setpts=PTS-STARTPTS[v{k}]")
    fil += ["[v0][v1][v2][v3]concat=n=4:v=1:a=0[v]"]
    subprocess.run(
        args
        + [
            "-filter_complex",
            ";".join(fil),
            "-map",
            "[v]",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-b:v",
            "6800k",
            "-maxrate",
            "9000k",
            "-bufsize",
            "9000k",
            "-threads",
            "6",
            "-g",
            "15",
            "-keyint_min",
            "15",
            "-bf",
            "0",
            "-sc_threshold",
            "0",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(file),
        ],
        check=True,
    )
    pr = Jprobe(file)
    assert (
        pr["r_frame_rate"] == "30/1"
        and pr["nb_frames"] == "1651"
        and pr["codec_name"] == "h264"
        and pr["pix_fmt"] == "yuv420p"
    )
    pkt = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_packets",
                "-select_streams",
                "v:0",
                "-show_entries",
                "packet=pts_time,flags",
                "-of",
                "json",
                str(file),
            ]
        )
    )["packets"]
    pts = np.array([float(x["pts_time"]) for x in pkt])
    assert np.max(abs(pts - np.arange(len(pts)) / 30)) < 0.000002
    assert max(np.diff(np.flatnonzero(["K" in x["flags"] for x in pkt]))) <= 15
    cap = cv.VideoCapture(str(file))
    err = []
    for k, (a, b) in enumerate(spans):
        src = cv.VideoCapture(str(inputs[k]))
        for t in [a, (a + b) / 2, b]:
            ims = []
            for v, n in [(cap, round((t - 85) * 30)), (src, round((t - a) * 30))]:
                v.set(cv.CAP_PROP_POS_FRAMES, n)
                ok, im = v.read()
                assert ok
                ims.append(im)
            err.append(
                dict(
                    t=t,
                    fromClip=[a, b],
                    rgbMAE=float(np.mean(abs(ims[0].astype("f") - ims[1]))),
                )
            )
        src.release()
    cap.release()
    assert max(e["rgbMAE"] for e in err) < 4
    assert file.stat().st_size < 65e6
    require_ghost_gate(file, 1651)
    rel = "generated/morph-photo-85-140.mp4"
    own = {d["clip"] for d in docs} | {rel}
    path = HOST / "generated.json"
    with open(W / "publish.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        doc = J(path)
        other = [c for c in doc["clips"] if c["file"] not in own]
        assert not any(max(85, c["t0"]) < min(140, c["t1"]) for c in other)
        dest = HOST / rel
        temp = dest.parent / f".morph-{uuid.uuid4().hex}.mp4"
        os.link(file, temp)
        os.replace(temp, dest)
        os.replace(file, out / "chase.mp4")
        doc["clips"] = sorted(
            other
            + [
                dict(file=rel, t0=85, t1=140, fps=30, pose="chase", vfov=50, ego="free")
            ],
            key=lambda r: r["t0"],
        )
        savej(path, doc)
    receipt = dict(
        clip=rel,
        bytes=dest.stat().st_size,
        sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
        probe=pr,
        segments=docs,
        boundaries=joints,
        encodeError=err,
    )
    savej(out / "receipt.json", receipt)
    print(rel, pr, joints, "MB", receipt["bytes"] / 1e6, flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["continuous"]:
        continuous()
    else:
        run(int(sys.argv[1]), int(sys.argv[2]), "boundary" in sys.argv[3:])
