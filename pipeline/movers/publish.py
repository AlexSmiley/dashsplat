"""Make an isolated scene revision. Only changed tagged rows are copied/repacked."""

import json, os, pickle
from pathlib import Path
from collections import defaultdict
import numpy as np
from .inventory import W
from .model import bake
from .tracking import playback

NAMES = {"live": "live-v5", "stitched": "stitched-opt-v2"}


def source_tiles(mode, src, dst, spec):
    mp = json.loads((src / "map.json").read_text())
    idx = json.loads((src / "rsb/index.json").read_text())
    runs = defaultdict(list)
    trimmed = 0
    links = 0
    # Discovered tracks have exact camera/frame pixel labels. Legacy seeded
    # scenes still use local fragment IDs within their explicit time windows.
    exact = defaultdict(set)
    for s in spec:
        if "frame" in s:
            exact[s["cam"], s["frame"]].add(s["id"])
    spec = [s for s in spec if "frame" not in s]
    if mode == "live":
        tag = json.load(open("maps/live-v5/slice-instances.json"))
        for f in tag["frames"]:
            match = exact.get((f["cam"], f["frame"]), ())
            for v in f["instances"]:
                if v["id"] in match:
                    for r in v["tile_runs"]:
                        runs[r["tile"]].append((r["first"], r["count"]))
            for s in spec:
                if (
                    s.get("input") == "extra"
                    or not s["window"][0] <= f["time"] <= s["window"][1]
                ):
                    continue
                for v in f["instances"]:
                    if v["id"] in s["fragments"].get(f["cam"], []):
                        for r in v["tile_runs"]:
                            runs[r["tile"]].append((r["first"], r["count"]))

    def link(a, b):
        b.parent.mkdir(parents=True, exist_ok=True)
        if b.exists() and os.path.samefile(a, b):
            return
        tmp = b.with_suffix(".link-temp")
        tmp.unlink(missing_ok=True)
        os.link(a, tmp)
        os.replace(tmp, b)

    for name in ["drive.json", "rig.json", "sky.jpg"]:
        if (src / name).exists():
            link(src / name, dst / name)
    generated = src / "generated.json"
    if generated.is_file():
        for clip in json.loads(generated.read_text()).get("clips", []):
            file = Path(clip["file"])
            if file.is_absolute() or ".." in file.parts:
                raise ValueError(f"non-scene clip path: {file}")
            link(src / file, dst / file)
        link(generated, dst / "generated.json")
        idx["generated"] = True
    (dst / "rsb").mkdir(exist_ok=True)
    tiles = []
    for v in mp["tiles"]:
        file = Path(v["file"])
        a = src / file
        b = dst / file
        cache = v["id"] + ".rsb"
        (dst / file.parent).mkdir(parents=True, exist_ok=True)
        if v["id"] not in runs:
            link(a, b)
            links += 1
            if (src / "rsb" / cache).exists():
                link(src / "rsb" / cache, dst / "rsb" / cache)
        else:
            blob = a.read_bytes()
            head, body = blob.split(b"end_header\n", 1)
            width = body.__len__() // v["count"]
            arr = np.frombuffer(body, np.uint8).reshape(v["count"], width)
            keep = np.ones(len(arr), bool)
            for i, n in runs[v["id"]]:
                assert i + n <= len(arr)
                keep[i : i + n] = False
            count = int(keep.sum())
            trimmed += len(arr) - count
            idx["tiles"].pop(v["id"], None)
            (dst / "rsb" / cache).unlink(missing_ok=True)
            if not count:
                continue
            header = head.replace(
                f"element vertex {len(arr)}\n".encode(),
                f"element vertex {count}\n".encode(),
            )
            assert header != head
            tmp = b.with_suffix(".tmp")
            tmp.write_bytes(header + b"end_header\n" + arr[keep].tobytes())
            os.replace(tmp, b)
            v["count"] = count
        tiles.append(v)
    if mode == "stitched":
        # The object copy extends beyond the parent optimisation slice. Only add
        # missing ground tile IDs from the same road/pose revision.
        road = Path("scene/road-stitched-v2")
        existing = {v["id"] for v in tiles}
        roadidx = json.loads((road / "rsb/index.json").read_text())
        for v in json.loads((road / "map.json").read_text())["tiles"]:
            if v["layer"] != "ground" or v["id"] in existing:
                continue
            link(road / v["file"], dst / v["file"])
            cache = v["id"] + ".rsb"
            link(road / "rsb" / cache, dst / "rsb" / cache)
            idx["tiles"][v["id"]] = roadidx["tiles"][v["id"]]
            tiles.append(v)
            links += 1
    mp["tiles"] = tiles
    (dst / "map.json").write_text(json.dumps(mp))
    idx["objects"] = True
    next_index = dst / "rsb/index.next"
    next_index.write_text(json.dumps(idx))
    os.replace(next_index, dst / "rsb/index.json")
    return links, trimmed


def run(only=None):
    for mode, nm in NAMES.items():
        if only and mode != only:
            continue
        src = Path("scene", nm)
        dst = Path("scene", nm + "-movers-v1")
        dest = dst / "objects"
        dest.mkdir(parents=True, exist_ok=True)
        objs = []
        specs = []
        summary = {}
        for s0 in json.loads(
            (
                (W / "subjects.json")
                if (W / "subjects.json").exists()
                else Path(__file__).with_name("subjects.json")
            ).read_text()
        ):
            if s0.get("review_only"):
                continue
            path = W / f"{mode}-{s0['id']}.pickle"
            if not path.exists():
                continue
            data = pickle.load(open(path, "rb"))
            s, rows, track, size, st, scan = data
            track = playback(track, mode)
            specs.append(s)
            summary[s["name"]] = bake(data, mode, dest)
            id = s["id"]
            objs.append(
                dict(
                    id=id,
                    **{"class": s["class"]},
                    size=np.round(size, 4).tolist(),
                    file=f"objects/{id}.ply",
                    predicted_file=f"objects/{id}.pred.ply",
                    t_first=float(track["t"][0]),
                    t_last=float(track["t"][-1]),
                    track={k: np.round(v, 6).tolist() for k, v in track.items()},
                )
            )
            print(mode, id, summary[s["name"]], flush=True)
        links, removed = source_tiles(mode, src, dst, specs)
        tmp = dst / "objects.tmp"
        tmp.write_text(json.dumps(dict(objects=objs), separators=(",", ":")))
        os.replace(tmp, dst / "objects.json")
        json.dump(
            dict(models=summary, source=nm, links=links, retired_source_rows=removed),
            open(W / (mode + "-models.json"), "w"),
            indent=2,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=NAMES)
    run(parser.parse_args().mode)
