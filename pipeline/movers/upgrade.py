"""Bake the hindsight winner textures after the trajectory is fixed; compare generations."""

import pickle, json
from pathlib import Path
import cv2, numpy as np
from .inventory import W
from .demo import R, run as attach
from .model import bake
from .appearance import fit_profile
from .proof import orb, text

FOCUS = {2, 5, 6, 14}
OUT = R / "report"
OUT.mkdir(exist_ok=True)


def plate(path, s, size):
    a = np.concatenate(
        [
            np.frombuffer(
                (path / f"{s['id']}{suf}.ply")
                .read_bytes()
                .split(b"end_header\n", 1)[1],
                "<f4",
            ).reshape(-1, 24)
            for suf in ["", ".pred"]
        ]
    )
    start = a[:, 17].min()
    end = a[:, 17].max()
    return np.vstack(
        [orb(a, t, size) for t in [start + 0.04, (start + end) / 2, end + 0.26]]
    )


def smooth_corrections(track):
    from scipy.ndimage import gaussian_filter1d, maximum_filter1d

    t = dict(track)
    xy = np.c_[t["x"], t["y"]]
    dist = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    bad = np.r_[False, dist > 1.46]
    if not bad.any():
        return t
    w = gaussian_filter1d(
        maximum_filter1d(bad.astype(float), size=17), 2, mode="nearest"
    )
    new = xy + (gaussian_filter1d(xy, 3.7, axis=0, mode="nearest") - xy) * w[:, None]
    t["x"] = new[:, 0]
    t["y"] = new[:, 1]
    return t


def run():
    dest = R / "hindsight/objects"
    dest.mkdir(parents=True, exist_ok=True)
    src = Path("scene/stitched-opt-v2-movers-v1")
    models = {}
    objs = json.load(open(src / "objects.json"))
    for o in objs["objects"]:
        uid = o["id"]
        s, rows, track, size, stats, scan = pickle.load(
            open(W / f"stitched-{uid}.pickle", "rb")
        )
        track = smooth_corrections(track)
        o["track"]["x"] = np.round(track["x"], 6).tolist()
        o["track"]["y"] = np.round(track["y"], 6).tolist()
        sz, cabin, fit = fit_profile((s, rows, track, size, stats, scan))
        data = (s, rows, track, sz, stats, scan)
        models[s["name"]] = {**fit, **bake(data, "stitched", dest, True, cabin)}
        o["size"] = np.round(sz, 4).tolist()
        print(uid, models[s["name"]], flush=True)
        if uid in FOCUS:
            sheets = []
            for label, path, sz0 in [
                ("BEFORE", src / "objects", size),
                ("AFTER", dest, sz),
            ]:
                im = plate(path, s, sz0)
                cap = np.full((23, im.shape[1], 3), 24, "uint8")
                text(
                    cap,
                    f"{label}   {s['name']}   early / middle / late   |   0, 45, 90, 135, 180, 225, 270, 315 degrees",
                )
                sheets.extend([cap, im])
            cv2.imwrite(
                str(OUT / f"{s['name']}-before-after.jpg"),
                np.vstack(sheets),
                [cv2.IMWRITE_JPEG_QUALITY, 84],
            )
    (R / "hindsight/objects.json").write_text(json.dumps(objs))
    (OUT / "shape-texture.json").write_text(json.dumps(models, indent=2))
    attach(True)


if __name__ == "__main__":
    run()
