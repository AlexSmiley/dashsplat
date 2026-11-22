"""Read physical Tesla exposures, preserving the front SEI clock across clip boundaries."""

import collections
import json
from pathlib import Path

import av
import cv2
import numpy as np
from pipeline.ingest.ingest import CAMS, parse_packet
from pipeline.ingest.ingest import main as index


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".next")
    tmp.write_text(json.dumps(obj, allow_nan=False, separators=(",", ":")))
    tmp.replace(path)


def inventory(source, bounds, rig_path):
    rig = json.loads(Path(rig_path).read_text())["cameras"]
    groups = {}
    for p in sorted(Path(source).rglob("*.mp4")):
        name, sep, c = p.stem.rpartition("-")
        if c not in CAMS or not sep:
            continue
        group = groups.setdefault(name, {})
        if c in group:
            raise ValueError(f"duplicate raw exposure group {name}: {p} and {group[c]}")
        group[c] = str(p.resolve())
    if not groups:
        raise ValueError("no Tesla <timestamp>-<camera>.mp4 groups")
    clock_path = Path(source) / "window-clock.json"
    clock = json.loads(clock_path.read_text()) if clock_path.is_file() else None
    if clock and (clock['version'] != 1 or bounds != clock['window'] or set(groups) != set(clock['native_ids'])):
        raise ValueError("window clock must match the requested slice and raw groups")
    counts = collections.Counter()
    info, seqs, speeds, span, seconds = [], [], [], 0, 0.0
    for name, paths in sorted(groups.items()):
        if set(paths) != set(CAMS):
            raise ValueError(
                f"incompatible/incomplete six-camera rig: {name} missing {set(CAMS) - set(paths)}"
            )
        starts = dict(counts)
        for c in CAMS:
            with av.open(paths[c]) as f:
                st = f.streams.video[0]
                if (st.width, st.height) != (rig[c]["width"], rig[c]["height"]):
                    raise ValueError(
                        f"incompatible rig/resolution: {name} {c}: {st.width}x{st.height}"
                    )
                # MP4 sample count is cheap but can include non-picture samples.
                # Count sliced access units, as the original native-index ingest does.
                first = last = None
                for pkt in f.demux(st):
                    if not pkt.size:
                        continue
                    sei, picture = parse_packet(bytes(pkt))
                    if not picture:
                        continue
                    counts[c] += 1
                    if (
                        c == "front"
                        and sei
                        and "seq" in sei
                        and "speed" in sei
                        and "lat" in sei
                    ):
                        rec = (
                            int(sei["seq"]),
                            float(pkt.pts * st.time_base),
                            float(sei["speed"]),
                        )
                        if first is None:
                            first = rec
                        last = rec
                        seqs.append(rec[0])
                        speeds.append(rec[2])
                if c == "front":
                    if first is None or last[1] <= first[1]:
                        raise ValueError(f"no usable Tesla SEI clock: {name}")
                    span += last[0] - first[0]
                    seconds += last[1] - first[1]
                    seq_range = [first[0], last[0]]
        info.append(
            dict(
                name=name,
                paths=paths,
                starts={c: starts.get(c, 0) for c in CAMS},
                seq=seq_range,
            )
        )
    fps = span / seconds
    order = np.argsort(seqs, kind="stable")
    seq = np.asarray(seqs)[order]
    speed = np.asarray(speeds)[order]
    if (
        not 10 <= fps <= 70
        or np.any(np.diff(seq) <= 0)
        or any(a["seq"][1] >= b["seq"][0] for a, b in zip(info, info[1:]))
    ):
        raise ValueError(
            "discontinuous or duplicate Tesla SEI clock: separate individual drives"
        )
    zero = int(seq[0])
    if clock:
        zero, fps = int(clock["seq0"]), float(clock["fps"])
        for group in info:
            group["native_ids"] = clock["native_ids"][group["name"]]
    t = (seq - zero) / fps
    a, b = bounds or [t[0], t[-1]]
    if not 0 <= a < b <= t[-1] + 1 / fps:
        raise ValueError(f"window {a, b} outside recorded clock 0..{t[-1]:.3f}")
    # Break at missing telemetry and parked intervals; retain short slowdowns inside fit context.
    moving = np.abs(speed) >= 0.1
    cut = np.r_[
        0, np.flatnonzero((moving[1:] != moving[:-1]) | (np.diff(t) > 0.5)) + 1, len(t)
    ]
    spans = []
    stops = []
    windows = []
    for i, j in zip(cut[:-1], cut[1:]):
        lo, hi = float(t[i]), float(t[j - 1])
        if not moving[i]:
            stops.append([lo, hi])
            continue
        if clock and i == 0 and 0 <= lo - a < 1 / fps:
            lo = a
        if clock and j == len(t) and 0 <= b - hi < 1 / fps:
            hi = b
        lo, hi = max(lo, a), min(hi, b)
        if hi - lo < 0.3:
            continue
        spans.append([float(lo), float(hi)])
        start = lo
        while start < hi:
            end = min(start + 24, hi)
            windows.append([float(start), float(end)])
            if end >= hi:
                break
            start = end - 3
    if not windows:
        raise ValueError("no moving footage in requested time window")
    write(
        "inventory.json",
        dict(
            clips=info,
            seq0=zero,
            fps=fps,
            counts=dict(counts),
            duration_s=float(t[-1]),
            stationary=stops,
            moving=spans,
            windows=windows,
            window=[float(a), float(b)],
        ),
    )
    print(f"{len(info)} groups; {fps:.4f} Hz; motion windows {windows}", flush=True)


def ingest():
    # Single decoder at a time, one codec thread; no MPS/torch. Reuse the canonical
    # SEI parser and telemetry synchronizer. Decode from the start of each relevant GOP file.
    cv2.setNumThreads(1)
    inv = json.loads(Path("inventory.json").read_text())
    fps = inv["fps"]
    zero = inv["seq0"]
    a, b = inv["window"]
    data = Path("data")
    dest = data / "drive"
    sizes = {}
    raw = {c: [] for c in CAMS}
    for group in inv["clips"]:
        if (group["seq"][1] - zero) / fps < a - 0.5 or (
            group["seq"][0] - zero
        ) / fps > b + 0.5:
            continue
        local = {c: [] for c in CAMS}
        for c, p in group["paths"].items():
            out = dest / "frames" / c
            out.mkdir(parents=True, exist_ok=True)
            with av.open(p) as f:
                st = f.streams.video[0]
                st.codec_context.thread_count = 1
                sizes[c] = [st.width, st.height]
                pending = collections.defaultdict(collections.deque)
                idx = 0
                for pkt in f.demux(st):
                    if pkt.size:
                        sei, picture = parse_packet(bytes(pkt))
                        if not picture:
                            continue
                        pending[pkt.pts].append((sei, float(pkt.pts * st.time_base)))
                    for fr in pkt.decode():
                        sei, pts = pending[fr.pts].popleft()
                        n = group["native_ids"][c][idx] if "native_ids" in group else group["starts"][c] + idx
                        idx += 1
                        if not sei or "seq" not in sei:
                            continue
                        t = (sei["seq"] - zero) / fps
                        if not a - 0.25 <= t <= b + 0.25:
                            continue
                        path = out / f"{n:06d}.jpg"
                        if not cv2.imwrite(
                            str(path),
                            fr.to_ndarray(format="bgr24"),
                            [cv2.IMWRITE_JPEG_QUALITY, 95],
                        ):
                            raise IOError(path)
                        local[c].append(
                            dict(sei, n=n, idx=idx - 1, pts=pts, clip=group["name"])
                        )
            print(group["name"], c, len(local[c]), flush=True)
        if len(local["front"]) < 2:
            continue
        (data / "clips" / group["name"]).mkdir(parents=True, exist_ok=True)
        for c in CAMS:
            raw[c].extend(local[c])
    if any(len(raw[c]) < 2 for c in CAMS):
        raise ValueError("insufficient synchronized exposures in window")
    write(dest / "raw_tables.json", {**raw, "_sizes": sizes})
    index(str(data))
    # Ingest defines a local front clock. Restore the master raw-clip epoch/rate,
    # retaining its measured inter-camera offsets; no old frames or poses are read.
    meta = json.loads((dest / "meta.json").read_text())
    ratio = meta["fps"] / fps
    shift = (meta["seq0"] - zero) / fps
    p = dest / "timeline.npz"
    with np.load(p) as f:
        tl = dict(f)
    tl["t"] = tl["t"] * ratio + shift
    for k in ("x", "y"):
        tl[k] *= ratio
    np.savez(p, **tl)
    for c in CAMS:
        p = dest / "cams" / f"{c}.json"
        rows = json.loads(p.read_text())
        for r in rows:
            r["t"] = r["t"] * ratio + shift
            r["ego_x"] *= ratio
            r["ego_y"] *= ratio
        write(p, rows)
    meta.update(
        fps=fps,
        seq0=zero,
        duration_s=float(tl["t"][-1]),
        distance_m=meta["distance_m"] * ratio,
        absolute_window=inv["window"],
    )
    write(dest / "meta.json", meta)
