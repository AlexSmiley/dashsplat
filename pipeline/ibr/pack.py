"""Snapshot available shared RGB-D keyframes onto the original distorted sensor lattice."""

from pathlib import Path
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import json, os, sys
import cv2 as cv
import numpy as np
from pipeline.livemap.geometry import RS, CAMS, rig, lattice, traj, photo, H, B
from scipy.spatial.transform import Rotation

STAGE = RS / "work/ibr/v1"
BASE = RS / "scene/live-v5"
T0, T1 = 80, 137.25


def dump(p, obj):
    tmp = p.with_suffix(".partial")
    tmp.write_text(json.dumps(obj, separators=(",", ":"), allow_nan=False))
    os.replace(tmp, p)


def pose(T):
    return np.r_[T[:3, 3], Rotation.from_matrix(T[:3, :3]).as_quat()].tolist()


@lru_cache(None)
def timed(c, t):
    C = rig[c]
    ts = C.row_time(t, np.array([0.0, C.height]))
    T = traj("live").T_world_ego(ts) @ C.T_ego_cam
    return [pose(v) for v in T]


def build():
    STAGE.mkdir(parents=True, exist_ok=True)
    j = json.load(open(RS / "depth/keyframes.json"))
    dr = json.load(open(BASE / "drive.json"))
    s = np.r_[0, np.cumsum(np.hypot(np.diff(dr["x"]), np.diff(dr["y"])))]
    doc = {
        "version": 1,
        "sourceInterval": [T0, T1],
        "fallback": "map.json",
        "depthUnitM": 0.01,
        "meshStride": 4,
        "cameras": {},
    }
    missing = {}
    for c in CAMS:
        C = rig[c]
        ray, valid = lattice(c)
        dn = STAGE / "rgbd" / c
        dn.mkdir(parents=True, exist_ok=True)
        idx = json.load(open(RS / "depth" / c / "index.json"))
        entries = [e for e in j["frames"][c] if T0 <= e["t"] <= T1]
        movep = RS / "masks/dyn" / c / "tracks.json"
        tracks = json.loads(movep.read_text())["tracks"] if movep.exists() else {}
        camspec = dict(
            native=[C.width, C.height],
            image=[1448, 938],
            depth=[B, H],
            model=C.model,
            params=C.params.tolist(),
            rowReadoutS=C.rs_readout_s,
            timeOffsetS=C.time_offset_s,
        )
        camfile = "rgbd/" + c + "/rays.bin"
        ray.astype("<f4").tofile(STAGE / camfile)
        camspec["rays"] = camfile
        doc["cameras"][c] = camspec
        backups = [
            (root / c, json.load(open(root / c / "index.json")))
            for root in [
                RS / "maps/object-v3/stitched/depth",
                RS / "maps/live-v4/inputs/depth",
            ]
            if (root / c / "index.json").exists()
        ]

        def one(e):
            n = e["n"]
            key = f"{n:06d}"
            zfile = RS / "depth" / c / (key + ".npy")
            source = idx.get(str(n))
            origin = "shared"
            if source is None or not zfile.exists():
                for folder, other in backups:
                    if str(n) in other and (folder / (key + ".npy")).exists():
                        source = other[str(n)]
                        zfile = folder / (key + ".npy")
                        origin = str(folder.relative_to(RS))
                        break
                else:
                    return None
            fn = "rgbd/" + c + "/" + key
            frame = dict(
                n=n,
                t=e["t"],
                s=float(np.interp(e["t"], dr["t"], s)),
                rowPoses=timed(c, e["t"]),
                rgb=fn + ".jpg",
                z=fn + "-z.png",
                mask=fn + "-m.png",
                depthModel=source["model"],
                depthSource=origin,
            )
            dst = [STAGE / frame[k] for k in ("rgb", "z", "mask")]
            ego = traj("live").T_world_ego(e["t"])
            norm, hh = traj("live").ground_plane(e["t"])
            nw = ego[:3, :3] @ norm
            frame["groundPlane"] = np.r_[nw, hh - nw @ ego[:3, 3]].tolist()
            frame["hasInstances"] = bool(
                json.load(open(RS / "masks/dyn" / c / (key + ".json")))
            )
            # On a repeat pack, promote a completed shared side depth over archived DA3.
            old = previous.get(n)
            reuse = (
                old
                and old["depthModel"] == source["model"]
                and old.get("depthSource", "shared") == origin
            )
            if reuse and all(p.is_file() for p in dst):
                return frame
            im = cv.imread(
                str(RS / "data/drive/frames" / c / (key + ".jpg")),
                cv.IMREAD_REDUCED_COLOR_2 if C.width == 2896 else cv.IMREAD_COLOR,
            )
            assert im.shape[:2] == (938, 1448)
            rgb = photo("live").apply(c, n, im[:, :, ::-1])
            sky, road, ins = [
                cv.imread(
                    str(RS / "masks" / m / c / (key + ".png")), cv.IMREAD_UNCHANGED
                )
                for m in ["sky", "road", "dyn"]
            ]
            if any(x is None for x in [sky, road, ins]):
                return None
            z = np.load(zfile).astype("f")
            assert z.shape == (H, B)
            z = np.where(valid & np.isfinite(z) & (z > 0.15) & (z < 350), z, 0)
            cat = np.where(road, 2, 1).astype("H")
            cat[sky != 0] = 3
            ring = cv.dilate((ins != 0).astype("B"), np.ones((7, 7), "B")) > 0
            cat[ring] = 4
            # Unknown or parked is never permission to bake an instance into an old view.
            for iid in np.unique(ins[ins > 0]):
                m = ins == iid
                motion = tracks.get(str(iid), {}).get("moving")
                cat[m] = (
                    8
                    + 3 * int(iid)
                    + (2 if motion is True else 1 if motion is False else 0)
                )
                # Complete small depth holes strictly inside this instance with its own reliable median.
                zm = z[m & (z > 0)]
                if len(zm) >= 12 and np.percentile(zm, 85) - np.percentile(
                    zm, 15
                ) < max(2, 0.18 * np.median(zm)):
                    z[m & (z == 0)] = np.median(zm)
            cat[~valid] = 0
            q = np.rint(np.clip(z * 100, 0, 65535)).astype("H")
            for p, img, opts in zip(
                dst,
                [rgb[:, :, ::-1], q, cat],
                [
                    [cv.IMWRITE_JPEG_QUALITY, 83],
                    [cv.IMWRITE_PNG_COMPRESSION, 2],
                    [cv.IMWRITE_PNG_COMPRESSION, 3],
                ],
            ):
                tmp = p.with_name(p.stem + ".part" + p.suffix)
                assert cv.imwrite(str(tmp), img, opts)
                os.replace(tmp, p)
            return frame

        prev = STAGE / "ibr.json"
        previous = (
            {e["n"]: e for e in json.load(open(prev))["cameras"][c]["keyframes"]}
            if prev.exists()
            else {}
        )
        with ThreadPoolExecutor(6) as pool:
            ls = list(pool.map(one, entries))
            camspec["keyframes"] = [e for e in ls if e]
            missing[c] = [e["n"] for e, v in zip(entries, ls) if v is None]
            print(
                c,
                len(camspec["keyframes"]),
                len(missing[c]),
                Counter(e["depthModel"] for e in ls if e),
                flush=True,
            )
    dump(STAGE / "ibr.json", doc)
    dump(STAGE.parent / "missing.json", missing)


if __name__ == "__main__":
    build()
