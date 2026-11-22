"""Offline source bank for a small live orbit: short GOPs for actual sensor RGB, quantized continuous inverse-depth/matte grids, pose and indexed selection schedule. No chase imagery inside."""

from .morph_clip import *
import struct, zlib, zipfile


def packet(a, b):
    out = W / f"{a}-{b}"
    st = J(out / "film.json")
    dest = out / "live-photo"
    dest.mkdir(exist_ok=True)
    N = GW * GH
    runs = []
    geoBytes = 0
    texBytes = 0
    metaBytes = 0
    for c in CAMS:
        ids = set()
        ts, ns, T, _ = seq(c)
        for row in st:
            for lens, n, f, amount in row["uses"]:
                if lens == c:
                    idx = int(np.searchsorted(ns, n))
                    ids.update([idx, idx + 1])
        if not ids:
            continue
        ids = sorted(ids)
        cut = (
            [0]
            + [i for i in range(1, len(ids)) if ids[i] - ids[i - 1] > 5]
            + [len(ids)]
        )
        for k, (u, v) in enumerate(zip(cut, cut[1:])):
            seqids = range(ids[u], ids[v - 1] + 1)
            stem = f"{c}-{ns[seqids.start]}"
            video = dest / (stem + ".mp4")
            geo = dest / (stem + ".grid")
            index = []
            proc = None
            prev = None
            with geo.open("wb") as f:
                f.write(struct.pack("<4s5I", b"LDI1", GW, GH, len(seqids), 8, 16))
                for count, i in enumerate(seqids):
                    pose, A, rgb = source(c, i)
                    ph, pw = rgb.shape[:2]
                    if proc is None:
                        proc = subprocess.Popen(
                            [
                                "ffmpeg",
                                "-v",
                                "error",
                                "-y",
                                "-f",
                                "rawvideo",
                                "-s",
                                f"{pw}x{ph}",
                                "-r",
                                "36",
                                "-pix_fmt",
                                "bgr24",
                                "-i",
                                "-",
                                "-an",
                                "-c:v",
                                "h264_videotoolbox",
                                "-b:v",
                                "11M",
                                "-g",
                                "18",
                                "-bf",
                                "0",
                                "-pix_fmt",
                                "yuv420p",
                                "-movflags",
                                "+faststart",
                                str(video),
                            ],
                            stdin=subprocess.PIPE,
                        )
                    proc.stdin.write(rgb.tobytes())
                    d = np.uint16(np.rint(np.clip(A[:, :, :2], 0, 1) * 65535)).astype(
                        "<u2"
                    )
                    m = np.uint8(np.rint(255 * np.clip(A[:, :, [2, 3, 4, 6]], 0, 1)))
                    raw = np.concatenate([d.view("B").reshape(GH, GW, 4), m], 2)
                    keyframe = count % 16 == 0
                    payload = zlib.compress(
                        (raw if keyframe else raw - prev).tobytes(), 4
                    )
                    prev = raw
                    offset = f.tell()
                    f.write(payload)
                    index.append(
                        dict(
                            n=int(ns[i]),
                            t=float(ts[i]),
                            camToWorld=pose[:3].ravel().tolist(),
                            offset=offset,
                            length=len(payload),
                            absolute=keyframe,
                        )
                    )
            proc.stdin.close()
            assert proc.wait() == 0
            geoBytes += geo.stat().st_size
            texBytes += video.stat().st_size
            runs.append(
                dict(
                    lens=c,
                    texture=video.name,
                    grid=geo.name,
                    records=index,
                    width=pw,
                    height=ph,
                )
            )
            assert int(Jprobe(video)["nb_frames"]) == len(index)
    lensdata = {
        c: dict(
            model=rig[c].model,
            params=rig[c].params.tolist(),
            size=[rig[c].width, rig[c].height],
        )
        for c in CAMS
    }
    doc = dict(
        format="chase-ldi-experiment",
        version=1,
        clip=[a, b],
        coordinate="sensor +z forward, +y down; metric world z-up",
        sourceFPS=36,
        geometry=dict(
            width=GW,
            height=GH,
            stride=8,
            channels=[
                "background inverse axial m^-1 u16/65535",
                "foreground inverse axial m^-1 u16/65535",
                "near alpha u8/255",
                "exclude vehicle alpha u8/255",
                "ground alpha u8/255",
                "valid alpha u8/255",
            ],
            coding="little endian interleaved top-down. zlib on whole frames; bytewise modulo-256 difference to previous unless absolute; seek to previous absolute.",
        ),
        projection="Camera.project from rig, scaled to texture. Reconcile depth with tangent ground and use infinite sky; perform bounded chase-space inverse-map regularization and bilinear colour lookup. Re-estimate residual flow after changing the eye; no screen-fixed colour crossfade.",
        fallback="host ground only plus photographic sky; small unobserved inter-lens gaps are extrapolated",
        rig=lensdata,
        sources=runs,
        selection=[
            dict(
                t=r["t"],
                baseArc=r["baseArc"],
                pairs=[[c, n, f] for c, n, f, w in r["uses"]],
            )
            for r in st
        ],
    )
    return finalize(a, b, doc, dest, out, runs, geoBytes, texBytes)


def finalize(a, b, doc, dest, out, runs, geoBytes, texBytes):
    import shutil

    shutil.copyfile(RS / "work/ibr/horizon-l/horizon-r4/sky.jpg", dest / "sky.jpg")
    doc["sky"] = {"file": "sky.jpg", "elevation": [-12, 82]}
    for sel, (_, T) in zip(doc["selection"], track(a, b)):
        t = sel["t"]
        E = TR.T_world_ego(t)
        n, h = TR.ground_plane(t)
        n = E[:3, :3] @ n
        sel["groundPlane"] = [*n.tolist(), float(h - n @ E[:3, 3])]
        sel["chaseCV"] = T[:3].ravel().tolist()
    savej(dest / "bank.json", doc)
    metaBytes = (dest / "bank.json").stat().st_size
    # Decode a delta chain and compare each reconstructed final grid to the source-quantized data.
    for run in runs:
        data = (dest / run["grid"]).read_bytes()
        prev = None
        for r in run["records"]:
            x = np.frombuffer(
                zlib.decompress(data[r["offset"] : r["offset"] + r["length"]]), "B"
            )
            assert x.size == GW * GH * 8
            prev = x if r["absolute"] else (prev + x)
        A = source(
            run["lens"],
            int(np.searchsorted(seq(run["lens"])[1], run["records"][-1]["n"])),
        )[1]
        assert np.array_equal(
            prev.reshape(GH, GW, 8)[:, :, 4],
            np.uint8(np.rint(255 * np.clip(A[:, :, 2], 0, 1))),
        )
    z = out / "live-photo-bank.zip"
    with zipfile.ZipFile(
        z, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1
    ) as archive:
        for p in dest.iterdir():
            archive.write(p, p.name)
    result = dict(
        sourceFrames=sum(len(r["records"]) for r in runs),
        streams=len(runs),
        textureBytes=texBytes,
        gridBytes=geoBytes,
        metadataBytes=metaBytes,
        bankBytes=z.stat().st_size,
        MBperChaseSecond=z.stat().st_size / (b - a) / 1e6,
        gridMBperChaseSecond=geoBytes / (b - a) / 1e6,
    )
    savej(out / "live-payload.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "finalize":
        out = W / "110-125"
        dest = out / "live-photo"
        doc = J(dest / "bank.json")
        runs = doc["sources"]
        finalize(
            110,
            125,
            doc,
            dest,
            out,
            runs,
            sum((dest / r["grid"]).stat().st_size for r in runs),
            sum((dest / r["texture"]).stat().st_size for r in runs),
        )
    else:
        packet(110, 125)
