"""Synchronized footage / chase / actual optimised PLY comparisons and photo-aligned edge/stretch measurements."""

from .morph_clip import *
from copy import copy


def sob(im):
    g = cv.cvtColor(im, cv.COLOR_BGR2GRAY).astype("f")
    return np.hypot(cv.Sobel(g, -1, 1, 0), cv.Sobel(g, -1, 0, 1)) * 0.125


def examine(a, b, reuse=False):
    ts = track(a, b)
    arc = choose(ts)
    out = W / f"{a}-{b}"
    cap = cv.VideoCapture(str(out / "chase.mp4"))
    assert cap.isOpened()
    prior = cv.VideoCapture(str(out / "compare.mp4")) if reuse else None
    file = out / "compare-next.mp4"
    enc = encode(file, 1440, 324, 9)
    chaseHash = hashlib.sha256((out / "chase.mp4").read_bytes()).hexdigest()
    fh = copy(ground_player())
    fh.tiles = J(HOST / "map.json")["tiles"]
    fh.bb = np.array([e["bbox"] for e in fh.tiles])
    pics = []
    data = []
    prev = None
    pop = []
    starts = []
    ST = seq("front")
    stats = J(out / "film.json")
    for idx, ((t, T), s) in enumerate(zip(ts, arc)):
        ok, im = cap.read()
        assert ok
        if prior is None or idx % 30 == 0:
            wr = RAY @ T[:3, :3].T
            gr = plane(t, T, wr)
            fallback, _ = floor(t, T, wr, gr)
            buf = fh.splats(t, T, [52, 100])
            spl = np.uint8(
                np.clip(
                    buf[:, :, :3][:, :, ::-1] + (1 - buf[:, :, 3:4]) * fallback, 0, 255
                )
            )
        if prior is None:
            now = int(np.argmin(abs(ST[0] - t)))
            real = cv.imread(
                str(RS / "data/drive/frames/front" / f"{ST[1][now]:06}.jpg"),
                cv.IMREAD_REDUCED_COLOR_2,
            )
            ims = [
                cv.resize(k, (480, 300), interpolation=cv.INTER_AREA)
                for k in [real, im, spl]
            ]
            board = np.vstack([np.hstack(ims), np.zeros((24, 1440, 3), "B")])
            for u, title in enumerate(
                [
                    "PHYSICAL FRONT / same drive time",
                    "TASK M / viewer chase pose",
                    "STITCHED-OPT-V3 / ground + object PLYs",
                ]
            ):
                cv.putText(
                    board,
                    f"{title}   {t:.2f} s",
                    (u * 480 + 7, 316),
                    0,
                    0.42,
                    (230, 230, 230),
                    1,
                    cv.LINE_AA,
                )
        else:
            ok, board = prior.read()
            assert ok
            board[:300, 480:960] = cv.resize(
                im, (480, 300), interpolation=cv.INTER_AREA
            )
        enc.stdin.write(board.tobytes())
        small = cv.resize(im, (480, 300), interpolation=cv.INTER_AREA)
        g = cv.cvtColor(small, cv.COLOR_BGR2GRAY)
        if prev is not None:
            v = cv.calcOpticalFlowFarneback(prev, g, None, 0.5, 3, 15, 3, 5, 1.1, 0)
            p = coord(480, 300) + v
            cur = remap(g, p)
            err = abs(prev.astype("f") - cur)
            roi = err[20:265, 15:-15]
            pop.append(float(np.percentile(roi, 95)))
            starts.append(float((roi > 24).mean()))
        prev = g
        if idx % 30 != 0:
            continue
        lens, n0, f, _ = stats[idx]["uses"][0]
        i0 = int(np.searchsorted(seq(lens)[1], n0))
        q = pull(lens, i0 + int(f > 0.5), T, wr, gr)
        uv = resize(q["p"], OW, OH)
        valid = (
            (q["mask"] > 0.94)
            & (q["fg"] < 0.15)
            & (np.abs(XY[:, :, 0] - L / 2) < L * 0.20)
        )
        ax = resize(valid.astype("f"), OW, OH) > 0.99
        ax[:25] = False
        ax[-25:] = False
        ideal = remap(q["tex"], uv)
        strength = sob(im)
        source = sob(q["tex"])
        actual = sob(ideal)
        ref = remap(source, uv)
        gs = sob(spl)
        pct = lambda img: float(np.percentile(img[ax], 90)) if ax.any() else None
        sample = dict(
            t=t,
            pixels=int(ax.sum()),
            sourceNativeP90=pct(ref),
            sourceWarpedP90=pct(actual),
            chaseP90=pct(strength),
            splatP90=pct(gs),
            anisotropyP95=stats[idx]["stretchP95"],
        )
        data.append(sample)
        if idx % 90 == 0 or idx == len(ts) - 1:
            pics.append(cv.resize(board, (1080, 243)))
    enc.stdin.close()
    assert enc.wait() == 0
    cap.release()
    if prior is not None:
        prior.release()
    os.replace(file, out / "compare.mp4")
    cv.imwrite(str(out / "comparison-sheet.jpg"), np.vstack(pics))
    use = [x for x in data if x["pixels"] > 5000]
    avg = lambda k: float(np.median([q[k] for q in use])) if use else None
    report = dict(
        clip=[a, b],
        chaseSHA256=chaseHash,
        samples=data,
        medianSharpnessP90={
            k: avg(k)
            for k in ["sourceNativeP90", "sourceWarpedP90", "chaseP90", "splatP90"]
        },
        p95WarpedLumaChangeP50=float(np.median(pop)),
        p95WarpedLumaChangeP95=float(np.percentile(pop, 95)),
        highChangeFractionP95=float(np.percentile(starts, 95)),
        anisotropyP95median=float(
            np.median([d["stretchP95"] for d in stats if d["stretchP95"] is not None])
        ),
        video=Jprobe(out / "compare.mp4"),
    )
    savej(out / "comparison.json", report)
    print({k: v for k, v in report.items() if k != "samples"}, flush=True)


if __name__ == "__main__":
    examine(int(sys.argv[1]), int(sys.argv[2]), "reuse" in sys.argv[3:])
