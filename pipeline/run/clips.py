"""Chunk scheduling and the installed genfill CLI's cross-boundary seed contract."""

import json
import math
import shutil
import subprocess
from pathlib import Path


def selection(words):
    if words in (["none"], ["all"]):
        return words[0]
    if len(words) != 2:
        raise ValueError("--clip requires all, none, or two drive-clock seconds")
    lo, hi = map(float, words)
    if not (math.isfinite(lo) and math.isfinite(hi) and 0 <= lo < hi):
        raise ValueError("invalid --clip bounds")
    return [lo, hi]


def schedule(windows, selection):
    rows = []
    for i, (a, b) in enumerate(windows):
        if i:
            a = max(a, windows[i - 1][1])
        if selection == "none":
            continue
        if isinstance(selection, list):
            a, b = max(a, selection[0]), min(b, selection[1])
        # An endpoint belongs to both adjacent videos; its RGB/provenance and
        # geometry seed are carried exactly, not independently hallucinated.
        while b - a > 0.0001:
            end = min(a + 4, b)
            rows.append(dict(window=i, start=a, end=end))
            a = end
    return rows


def render(out, scene, a, b, previous=None):
    from stages import HOME, binlink, command, mod, read, write
    import os

    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    scene = Path(scene).resolve()
    binlink(
        HOME / "maps/object-v3/raster.dylib",
        Path(os.environ["RS"]) / "maps/object-v3/raster.dylib",
    )
    track = out / "track.json"
    # chase accepts 1 sample for a sub-frame tail; explicitly append the exact
    # endpoint so fractional telemetry boundaries don't leave a viewer gap.
    tool = HOME / "tools/run-chase/chase-track.js"
    with track.open("w") as f:
        command("bun", tool, scene, a, b, 30, 1440, 900, stdout=f)
    doc = read(track)
    frames = doc["frames"]
    expected = round(b, 4)
    if frames[-1]["t"] != expected:
        last = json.loads(
            subprocess.check_output(
                ["bun", str(tool), str(scene), str(b), str(b), "30", "1440", "900"]
            )
        )["frames"][0]
        frames.append(last)
        write(track, doc)
    if len(frames) < 2:
        raise ValueError("no decodable chase interval")
    tag = "repaired"
    carry = []
    n0 = 0
    if previous:
        p = Path(previous)
        old = read(p / "frames.json")["frames"][-1]
        if abs(old["t"] - frames[0]["t"]) > 1e-4:
            raise ValueError("non-contiguous chase seed")
        for sub, ext in [
            ("render", "npz"),
            ("render", "png"),
            (tag, "png"),
            (tag + "-prov", "png"),
        ]:
            (out / sub).mkdir(exist_ok=True)
            shutil.copy2(p / sub / f"{old['n']:04d}.{ext}", out / sub / f"0000.{ext}")
        carry = ["--seed", tag]
        n0 = 1
    mod(
        "genfill.clip", "render", scene, track, out, "--workers", 3
    )  # deliberately no --movers
    py = HOME / "envs/genfill/bin/python"
    command(
        py,
        "-m",
        "pipeline.genfill.clip",
        "fix",
        out,
        tag,
        "--timestep",
        199,
        "--ff-in",
        0.5,
        "--ff-out",
        0.5,
        "--n0",
        n0,
        *carry,
    )
    name = f"chase-{round(a * 10000)}-{round(b * 10000)}"
    bundle = out / "bundle"
    bundle.mkdir(exist_ok=True)
    command(py, "-m", "pipeline.genfill.clip", "encode", out, tag, name, bundle)
    entry = read(bundle / "generated.json")["clips"][0]
    video = bundle / entry["file"]
    probe = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "stream=nb_frames,width,height,codec_name",
                "-of",
                "json",
                str(video),
            ]
        )
    )["streams"][0]
    if (
        entry["vehicles"] != "overlay"
        or entry["provenance"] != "below"
        or abs(entry["t0"] - a) > 1e-4
        or abs(entry["t1"] - b) > 1e-4
        or int(probe["nb_frames"]) != len(frames)
        or (probe["width"], probe["height"]) != (1440, 1800)
    ):
        raise ValueError("invalid encoded clip")
    write(
        out / "clip.json",
        dict(entry=entry, probe=probe, seed=str(previous) if previous else None),
    )
