"""Score both exported scenes on the Studio; holdouts never enter the GPU payload.

recover <receipts.jsonl> <review> <native> <scene> -- verify streamed tiles and restore local metadata.
score <scene> <dir> -- independent native-photo and teacher composites.
compare <metal-dir> <cuda-dir> <output> -- paired metrics and three chase pairs.
"""

import hashlib, json, os, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2 as cv
from pipeline.stitch import v2eval as E
from pipeline.stitch.v4teacher import decode, split, schedule, CAMS


def recover(receipts, review, native, scene):
    rows = [json.loads(line) for line in receipts.read_text().splitlines()]
    info = {r["receipt"]: r for r in rows if "receipt" in r}
    m = info["scene"]["map"]
    expected = {r["media"]: r["sha256"] for r in rows if "media" in r}
    ground = info["ground_hashes"]["tiles"]
    for t in m["tiles"]:
        f = Path(t["file"])
        assert f.parent == Path("tiles")
        if t["layer"] == "ground":
            src, sha = native / f, ground[str(f)]
        else:
            src, sha = review / f.name, expected[f.name]
        with src.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != sha:
                raise ValueError(f"corrupt tile {f}")
        dest = scene / f
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.link(src, dest)
    for name in ("sky.jpg", "rig.json", "drive.json"):
        os.link(native / name, scene / name)
    (scene / "map.json").write_text(json.dumps(m))
    print("verified and restored", len(m["tiles"]), "tiles", flush=True)


def score(scene, dest):
    dest.mkdir(parents=True, exist_ok=True)
    if not (E.RESULT / "wide.dylib").is_file():
        E.build()
    model = E.Wide(scene)
    held = split()[1]
    chases = [e for e in held if e["kind"] == "chase"]
    shots = {min(chases, key=lambda e: abs(e["t"] - t))["id"] for t in (226, 236, 246)}
    assert len(shots) == 3

    def teacher(e):
        target, w, _, _ = decode(e)
        h, ww = w.shape
        T = np.array(e["T_world_cam"])
        K = e["K"]
        im = model.draw(
            e["t"], np.repeat(T[None], 33, axis=0), np.r_[K, np.zeros(4)], (ww, h), 0
        )
        weight = w.astype("f") / 255
        err = np.abs(target - im[:, :, :3]).mean(2)
        if e["id"] in shots:
            y, x = np.mgrid[:h, :ww]
            ray = E.world_ray(T, K, x, y)
            pan = model.pan
            H, W = pan.shape[:2]
            mx = (
                (0.5 - np.arctan2(ray[:, :, 1], ray[:, :, 0]) / (2 * np.pi)) * W - 0.5
            ) % W
            my = np.clip(
                (0.5 - np.arcsin(np.clip(ray[:, :, 2], -1, 1)) / np.pi) * H - 0.5,
                0,
                H - 1,
            )
            sky = cv.remap(
                pan,
                mx.astype("f"),
                my.astype("f"),
                cv.INTER_LINEAR,
                borderMode=cv.BORDER_WRAP,
            )
            display = np.uint8(
                np.clip(im[:, :, :3] + (1 - im[:, :, 3:4]) * (sky - E.BG), 0, 255)
            )
            assert cv.imwrite(str(dest / (e["id"] + ".png")), display[:, :, ::-1])
        assert weight.sum() > 0
        return dict(
            id=e["id"],
            t=e["t"],
            group=e["station"],
            mae=float(np.sum(err * weight) / weight.sum()),
        )

    def photograph(e):
        ref, valid = E.reference(e)
        res = model.real(e)
        assert valid.any()
        return dict(
            id=f"{e['c']}/{e['n']}",
            t=e["t"],
            group=int((e["t"] - 224) // 2),
            mae=float(np.abs(ref - res[:, :, :3])[valid].mean()),
        )

    with ThreadPoolExecutor(4) as pool:
        ts = list(pool.map(teacher, held))
        ps = list(
            pool.map(photograph, [e for c in CAMS for e in schedule()[c] if e["hold"]])
        )
    (dest / "scores.json").write_text(json.dumps(dict(teacher=ts, photo=ps), indent=2))


def compare(metal, cuda, out):
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads((d / "scores.json").read_text()) for d in (metal, cuda)]
    summary = {}
    for kind in ("teacher", "photo"):
        a, b = (r[kind] for r in rows)
        assert [x["id"] for x in a] == [x["id"] for x in b]
        means = [float(np.mean([x["mae"] for x in arr])) for arr in (a, b)]
        ms = [
            np.mean([x["mae"] for x in a if x["group"] == g])
            for g in sorted({x["group"] for x in a})
        ]
        se = float(np.std(ms, ddof=1) / np.sqrt(len(ms)))
        delta = means[1] - means[0]
        summary[kind] = dict(
            views=len(a),
            metal=means[0],
            cuda=means[1],
            delta=delta,
            studio_group_se=se,
            groups=len(ms),
            passes=abs(delta) <= min(0.1, se),
        )
    summary["chase"] = []
    for fn in sorted(metal.glob("*.png")):
        tag = fn.stem
        t = next(x["t"] for x in rows[0]["teacher"] if x["id"] == tag)
        imgs = [cv.imread(str(d / fn.name)) for d in (metal, cuda)]
        err = np.abs(imgs[0].astype("i2") - imgs[1])
        p99 = float(np.percentile(err, 99))
        imgs = [
            E.label(im, f"{device}    {t:.3f} s")
            for im, device in zip(imgs, ("Studio / Metal", "NVIDIA / CUDA"))
        ]
        assert cv.imwrite(
            str(out / (tag + ".jpg")), np.hstack(imgs), [cv.IMWRITE_JPEG_QUALITY, 94]
        )
        summary["chase"].append(
            dict(id=tag, t=t, mae=float(err.mean()), p99=p99, max=int(err.max()))
        )
    assert len(summary["chase"]) == 3
    (out / "comparison.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
    if not all(summary[k]["passes"] for k in ("teacher", "photo")) or any(
        v["p99"] > 1 for v in summary["chase"]
    ):
        raise SystemExit("fit comparison outside tolerance")


if __name__ == "__main__":
    {"recover": recover, "score": score, "compare": compare}[sys.argv[1]](
        *(Path(p) for p in sys.argv[2:])
    )
