"""Encode 110–125 s privately; replay the exact encoded frames against each endpoint's original photo chart."""

import hashlib
import json
import subprocess
from .morph_m2 import OUT, ALL, View, camera, frame
from . import horizon_proof as hp, morph_sources as ms
from .morph_clip import encode, Jprobe
from .morph_sources import OW, OH, F, J, savej, cv, np
from .morph_ghost import references, pack, unpack, audit, refmask
from concurrent.futures import ProcessPoolExecutor
import csv, multiprocessing as mp

SPAN = OUT / "110-125"
SPAN.mkdir(exist_ok=True)


def segment(bounds):
    lo, hi = bounds
    ts, arc = camera(50)
    hist = None
    rows = []
    store = []
    hp.F = F
    paths = [SPAN / f"{kind}-{lo:03d}.mp4" for kind in ["part", "solo"]]
    writers = [encode(path, OW, OH) for path in paths]
    try:
        for i in range(max(0, lo - 6), hi):
            t, T = ts[i]
            im, lab, patches, solo, meta, hist = frame(View(t, T, 50), arc[i], hist)
            if i < lo:
                continue
            for w, rgb in zip(writers, [im, solo]):
                w.stdin.write(rgb.tobytes())
            refs = references(lab, patches)
            store.append(
                (
                    meta,
                    pack(lab),
                    [(j, r, pack(m)) for j, (r, m) in refs.items()],
                    pack(refmask(solo)),
                )
            )
    finally:
        for w in writers:
            w.stdin.close()
            assert w.wait() == 0
    cap, sc = (cv.VideoCapture(str(p)) for p in paths)
    assert all(int(c.get(cv.CAP_PROP_FRAME_COUNT)) == hi - lo for c in (cap, sc))
    for i, (meta, l, rs, ref) in enumerate(store, lo):
        ok, im = cap.read()
        assert ok
        ok, solo = sc.read()
        assert ok
        l = unpack(l)
        refs = {j: (r, unpack(m)) for j, r, m in rs}
        res = audit(im, l, refs, "all")
        p = res.pop("allPoints")
        res.pop("points")
        res.pop("mark")
        x, y, u, v = p
        res["crossOwner"] = int((l[y, x] != l[v, u]).sum())
        control = audit(solo, np.zeros_like(l), {0: (ALL, unpack(ref))})
        rows.append(dict(i=i, meta=meta, candidate=res, solo=control))
        if i % 30 == 0:
            cv.imwrite(
                str(SPAN / f"{meta['t']:.2f}.jpg"), im, [cv.IMWRITE_JPEG_QUALITY, 94]
            )
    cap.release()
    sc.release()
    savej(SPAN / f"gate-{lo:03d}.json", rows)
    print("audited", lo, hi, flush=True)
    return rows


def run():
    spans = [(lo, min(lo + 50, 451)) for lo in range(0, 451, 50)]
    with ProcessPoolExecutor(5, mp_context=mp.get_context("spawn")) as pool:
        rows = [r for rs in pool.map(segment, spans) for r in rs]
    assert [r["i"] for r in rows] == list(range(451))
    savej(SPAN / "frames.json", rows)

    # Lossless packet concat; audit codec bytes from the assembled delivery, not merely
    # the frame counts of its intermediate parts. ffmpeg framemd5 is independent of PTS.
    def digests(path):
        raw = (
            subprocess.check_output(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-i",
                    str(path),
                    "-map",
                    "0:v:0",
                    "-f",
                    "framemd5",
                    "-",
                ]
            )
            .decode()
            .splitlines()
        )
        return [v.rsplit(",", 1)[-1].strip() for v in raw if not v.startswith("#")]

    sub = [SPAN / f"part-{lo:03d}.mp4" for lo, hi in spans]
    index = SPAN / "concat.txt"
    index.write_text("".join(f"file '{p.name}'\n" for p in sub))
    dest = SPAN / "pillar-photo-110-125.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(index),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(dest),
        ],
        check=True,
    )
    assert digests(dest) == [h for p in sub for h in digests(p)]
    pr = Jprobe(dest)
    assert (pr["width"], pr["height"], int(pr["nb_frames"]), pr["r_frame_rate"]) == (
        OW,
        OH,
        451,
        "30/1",
    )

    def stats(part, key):
        ls = [r[key] for r in part]
        return dict(
            all=100 * sum(r["twins"] for r in ls) / sum(r["edges"] for r in ls),
            structure=100
            * sum(r["structureTwins"] for r in ls)
            / sum(r["structureEdges"] for r in ls),
            worst=100 * max(r["fraction"] for r in ls),
            worstT=max(part, key=lambda r: r[key]["fraction"])["meta"]["t"],
        )

    stats0 = {k: stats(rows, k) for k in ["candidate", "solo"]}
    before = J(ms.W / "single-owner-v1/110-125/result.json")["stats"]
    worst = max(rows, key=lambda r: r["candidate"]["fraction"])
    report = dict(
        frames=len(rows),
        span=[110, 125],
        codec=pr["codec_name"],
        width=OW,
        height=OH,
        fps=30,
        sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
        decodedFramesIdentical=True,
        stats=stats0,
        previous=before,
        framesOverTwiceSolo=sum(
            r["candidate"]["fraction"] > 2 * r["solo"]["fraction"] for r in rows
        ),
        structureFramesOverTwiceSolo=sum(
            r["candidate"]["structureFraction"] > 2 * r["solo"]["structureFraction"]
            for r in rows
        ),
        worstBox=worst["candidate"]["worstBox"],
        crossOwnerPairs=sum(r["candidate"]["crossOwner"] for r in rows),
        allPairs=sum(r["candidate"]["twins"] for r in rows),
        coverage={
            k: float(np.mean([r["meta"][k] for r in rows]))
            for k in [
                "photo",
                "pillar",
                "bridge",
                "unknown",
                "stretch6",
                "ground",
                "groundStitched",
            ]
        },
        registered=sum(d is not None for r in rows for d in r["meta"]["registration"]),
        bridgesTested=sum(len(r["meta"]["registration"]) for r in rows),
        moverMaskMissing=sum(not q["remove"] for r in rows for q in r["meta"]["picks"]),
        framesAboveSolo=sum(
            r["candidate"]["fraction"] > r["solo"]["fraction"] for r in rows
        ),
        structureFramesAboveSolo=sum(
            r["candidate"]["structureFraction"] > r["solo"]["structureFraction"]
            for r in rows
        ),
        release=(
            "withhold: twice-solo gate failed"
            if stats0["candidate"]["all"] > 2 * stats0["solo"]["all"]
            or stats0["candidate"]["structure"] > 2 * stats0["solo"]["structure"]
            else "await per-frame gate and visual review"
        ),
    )
    with (SPAN / "per-second.csv").open("w") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "second",
                "frames",
                "candidateTwinPct",
                "candidateStructurePct",
                "soloTwinPct",
                "soloStructurePct",
            ]
        )
        for sec in range(110, 126):
            rr = [r for r in rows if int(r["meta"]["t"]) == sec]
            a, b = (stats(rr, k) for k in ["candidate", "solo"])
            w.writerow(
                [sec, len(rr), a["all"], a["structure"], b["all"], b["structure"]]
            )
    savej(SPAN / "result.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    run()
