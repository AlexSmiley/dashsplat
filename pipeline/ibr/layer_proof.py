"""Decode only a published scene. Usage: python layer_proof.py SCENE NATIVE_RASTER OUTPUT road|exit|stills|audit"""

from layer_player import *
from collections import Counter
import hashlib


def save(p, a):
    assert cv.imwrite(str(p), a[:, :, ::-1], [cv.IMWRITE_JPEG_QUALITY, 89])


def movie(r, out, name):
    start, end = (93, 104) if name == "exit" else (105, 120)
    film = out / f"{name}-{start}-{end}.mp4"
    enc = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{W}x{H}",
            "-r",
            "30",
            "-i",
            "-",
            "-an",
            "-c:v",
            "h264_videotoolbox",
            "-b:v",
            "14M",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(film),
        ],
        stdin=subprocess.PIPE,
    )
    st = []
    frames = []
    for i in range((end - start) * 30):
        t = start + i / 30
        im, d = r.draw(t)
        enc.stdin.write(im.tobytes())
        frames.append(
            dict(
                t=t,
                camera=d["cams"][0][0],
                pano=d["panos"],
                cards=len(set(d["drawn"])),
                moverPixels=int((d["ids"] >= 1000000).sum()),
                backdropPixels=d["backdrop"],
            )
        )
        if i % 30 == 0:
            a = cv.resize(im, (384, 240), interpolation=cv.INTER_AREA)
            cv.putText(a, f"{t:.0f}s", (7, 23), 0, 0.62, (245, 245, 245), 1, cv.LINE_AA)
            st.append(a)
        if i % 120 == 0:
            print(name, i, flush=True)
    enc.stdin.close()
    assert enc.wait() == 0
    chk = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,r_frame_rate,nb_frames,duration",
                "-of",
                "json",
                str(film),
            ]
        )
    )["streams"][0]
    assert (
        chk["width"],
        chk["height"],
        chk["r_frame_rate"],
        int(chk["nb_frames"]),
        float(chk["duration"]),
    ) == (W, H, "30/1", len(frames), end - start)
    st += [np.zeros_like(st[0])] * ((-len(st)) % 3)
    save(
        out / (name + "-sequence.jpg"),
        np.vstack([np.hstack(st[i : i + 3]) for i in range(0, len(st), 3)]),
    )
    (out / (name + ".json")).write_text(
        json.dumps(
            dict(
                sceneSha256=hashlib.sha256(
                    r.asset("ibr.json").read_bytes()
                ).hexdigest(),
                video=chk,
                frames=frames,
            )
        )
    )
    print(chk, flush=True)


def stills(r, out):
    records = []
    imgs = []
    for t in [93, 99 + 8 / 30, 100, 102 + 25 / 30, 112, 118 + 22 / 30, 135, 140]:
        im, d = r.draw(t)
        name = f"{t:.3f}.jpg"
        save(out / name, im)
        T = r.poses[round(t * 30)]
        record = dict(
            t=t,
            source=d["cams"],
            backdropPixels=d["backdrop"],
            groundPixels=d["ground"],
            mover46Pixels=int((d["ids"] == 1000046).sum()),
            visibleIds=[int(i) for i in np.unique(d["ids"]) if i > 0],
        )
        records.append(record)
        a = cv.resize(im, (560, 350))
        cv.putText(
            a,
            f"{t:.2f} s   {d['cams'][0][0]}",
            (10, 22),
            0,
            0.53,
            (255, 255, 255),
            1,
            cv.LINE_AA,
        )
        imgs.append(a)
    save(
        out / "witnesses.jpg",
        np.vstack([np.hstack(imgs[i : i + 2]) for i in range(0, len(imgs), 2)]),
    )
    (out / "witnesses.json").write_text(json.dumps(records, indent=2))
    print([(x["t"], x["source"][0][0], x["mover46Pixels"]) for x in records])


def audit(r, out):
    paths = {
        str(p.relative_to(r.root)): p.stat().st_size
        for p in r.root.rglob("*")
        if p.is_file()
    }
    cats = Counter()
    for p, n in paths.items():
        cats[(p.split("/")[0] if "/" in p else "metadata")] += n
    report = {
        "format": r.doc["format"],
        "revision": r.doc["contentRevision"],
        "sceneDigest": hashlib.sha256(r.asset("ibr.json").read_bytes()).hexdigest(),
        "bytes": sum(paths.values()),
        "files": len(paths),
        "groups": dict(cats),
        "cutouts": len(r.cards),
        "baseKnots": len(r.pan),
        "bases": {c: len(es) for c, es in r.cams.items()},
        "lifetimes": {
            "multiple": sum(a["observations"] > 1 for a in r.cards),
            "single": sum(a["observations"] == 1 for a in r.cards),
        },
        "stream": {},
    }
    idx = readj(r.asset("rsb/index.json"))["tiles"]
    assert all(
        idx[e["id"]]["count"] == e["count"]
        and paths["rsb/" + idx[e["id"]]["file"]] == idx[e["id"]]["bytes"]
        for e in r.tiles
    )
    report["groundSplats"] = sum(e["count"] for e in r.tiles)
    for start, end in [(93, 104), (105, 120), (93, 140)]:
        seen = set()
        loads = []
        ground = set()
        meta = {
            "ibr.json",
            "chase.json",
            "drive.json",
            "rig.json",
            "rsb/index.json",
            "map.json",
        }
        cameras = Counter()
        for i in range((end - start) * 30):
            t = start + i / 30
            T = r.poses[round(t * 30)]
            pg, ls = r.selection(t, T)
            cameras[ls[0][0]] += 1
            now = set()
            for k in pg:
                p = r.pan[k]
                now.update(p[s] for s in ["sky", "rgb", "matte"])
            for a, f, m in r.active(t, T):
                now.update([a["rgb"], a["matte"]])
            for c, chs in r.doc["movers"].items():
                meta.update(
                    h["file"] for h in chs if h["first"] - 0.24 <= t <= h["last"] + 0.24
                )
            d = np.linalg.norm(
                np.maximum(
                    0, np.maximum(r.bb[:, :2] - T[:2, 3], T[:2, 3] - r.bb[:, 3:5])
                ),
                axis=1,
            )
            ground.update(
                "rsb/" + idx[e["id"]]["file"] for e, k in zip(r.tiles, d < 115) if k
            )
            loads.append(sum(paths[p] for p in now - seen))
            seen |= now
        layer = sum(loads)
        total = layer + sum(paths[p] for p in meta | ground)
        report["stream"][f"{start}-{end}"] = dict(
            layersBytes=layer,
            layersMBs=layer / 1e6 / (end - start),
            layerPeakRollingMB=float(
                np.convolve(np.array(loads), np.ones(30), "valid").max() / 1e6
            ),
            cacheAndMetadataBytes=total - layer,
            allMBs=total / 1e6 / (end - start),
            dominantCameras=dict(cameras),
        )
    (out / "audit.json").write_text(json.dumps(report, indent=2))
    print(report, flush=True)


if __name__ == "__main__":
    root, lib, folder, task = sys.argv[1:]
    out = Path(folder)
    out.mkdir(parents=True, exist_ok=True)
    r = Player(root, lib)
    {
        "road": lambda: movie(r, out, "road"),
        "exit": lambda: movie(r, out, "exit"),
        "stills": lambda: stills(r, out),
        "audit": lambda: audit(r, out),
    }[task]()
