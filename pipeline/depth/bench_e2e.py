#!/usr/bin/env python3
"""End-to-end time to produce depth + masks for one six-camera keyframe, with the production code path.

Builds a private root (symlinks to data/calib/poses, own depth/ and masks/) holding K consecutive
distance keyframes, then times, per keyframe: Mask2Former on all six cameras, and depth for all six
cameras (JPEG decode, rectification, model with its causal window, resampling, write) for each
configuration. Each stage runs twice and the second (warm) pass is timed. One model is loaded at a
time, so the total is the sum of the stages (a real-time build would keep all models resident).

Usage: bench_e2e.py [--t0 300 --k 12] [--configs production,da3large,da3base,da3small]
Writes $RS/work/depth/report/e2e.json.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

MAIN = Path(os.environ.get("RS", "~/dev/reality-splat")).expanduser()
CAMS = ["front", "back", "left_pillar", "right_pillar", "left_repeater", "right_repeater"]
CONFIGS = {
    "production": {"front": "da3-large-1.1", "back": "da3-large-1.1", "left_pillar": "mapanything-nomask",
                   "right_pillar": "mapanything-nomask", "left_repeater": "mapanything-nomask",
                   "right_repeater": "mapanything-nomask"},
    "da3large": {c: "da3-large-1.1" for c in CAMS},
    "da3base": {c: "da3-base" for c in CAMS},
    "da3small": {c: "da3-small" for c in CAMS},
}


def setup(alt, t0, k):
    alt.mkdir(parents=True, exist_ok=True)
    for d in ["data", "calib", "poses", "pipeline"]:
        if not (alt / d).exists():
            (alt / d).symlink_to(MAIN / d)
    (alt / "work" / "depth").mkdir(parents=True, exist_ok=True)
    if not (alt / "work" / "depth" / "cache").exists():
        (alt / "work" / "depth" / "cache").symlink_to(MAIN / "work" / "depth" / "cache")
    (alt / "depth").mkdir(exist_ok=True)
    kf = json.loads((MAIN / "depth" / "keyframes.json").read_text())
    front = [e for e in kf["frames"]["front"] if e.get("kind", "distance") == "distance" and e["t"] >= t0][:k]
    keys = {round(e["t"], 6) for e in front}
    sel = {c: [e for e in kf["frames"][c] if round(e.get("t_key", e["t"]), 6) in keys] for c in CAMS}
    (alt / "depth" / "keyframes.json").write_text(json.dumps({"version": 1, "frames": sel, "selections": []}))
    return sel


def contention():
    out = subprocess.run(["ps", "-Ao", "pcpu,rss,command", "--sort=-pcpu" if sys.platform.startswith("linux") else "-r"], capture_output=True, text=True).stdout.splitlines()[1:12]
    return [" ".join(l.split()[:2] + [" ".join(l.split()[2:])[:90]]) for l in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t0", type=float, default=300.0)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--configs", default="production,da3large,da3base,da3small")
    args = ap.parse_args()
    alt = MAIN / "work" / "depth" / "e2e_root"
    sel = setup(alt, args.t0, args.k)
    os.environ["RS"] = str(alt)
    sys.path.insert(0, str(MAIN / "pipeline" / "depth"))
    sys.path.insert(0, str(MAIN / "pipeline" / "masks"))
    import rscommon
    rscommon.setup_env()
    import torch
    import models
    import run_depth
    import run_masks
    from seg import Segmenter
    K = len(sel["front"])
    res = dict(t0=args.t0, keyframes=K, t_keys=[e["t"] for e in sel["front"]],
               memory_free_pct_start=rscommon.memory_free_pct(), other_processes=contention())
    rscommon.wait_for_memory(20, what="Mask2Former")
    seg = Segmenter()
    times = {}
    for rep in range(2):
        t = time.time()
        for c in CAMS:
            run_masks.segment_camera(c, sel[c], (724, 469), seg, overwrite=True)
        times[rep] = time.time() - t
    res["masks_s_per_keyframe"] = times[1] / K
    del seg
    models.empty_cache()
    traj = rscommon.trajectory()
    ns = argparse.Namespace(size="724x469", width=518, window=5, window_multiview=3, win_spacing=1.0, posed=True,
                            edge=0.06, max_depth=150.0, overwrite=True, limit=0, every=1, min_free=0, scale="none",
                            scale_factor=1.0, cam_factors={})
    fb = {}

    def fallback():
        if "m" not in fb:
            fb["m"] = models.load_model("da3metric-large")
        return fb["m"]

    res["configs"] = {}
    for cfg in args.configs.split(","):
        per_cam = {}
        groups = {}
        for c in CAMS:
            groups.setdefault(CONFIGS[cfg][c], []).append(c)
        for mname, cams in groups.items():
            rscommon.wait_for_memory(20, what=f"loading {mname}")
            m = models.load_model(mname)
            for c in cams:
                for rep in range(2):
                    t = time.time()
                    run_depth.process_camera(c, sel[c], m, ns, traj, fallback)
                    dt = time.time() - t
                per_cam[c] = dt / len(sel[c])
            del m
            models.empty_cache()
        depth = sum(per_cam.values())
        res["configs"][cfg] = dict(models=CONFIGS[cfg], depth_s_per_camera=per_cam, depth_s_per_keyframe=depth,
                                   total_s_per_keyframe=depth + res["masks_s_per_keyframe"])
        print(cfg, json.dumps(res["configs"][cfg]), flush=True)
    res["memory_free_pct_end"] = rscommon.memory_free_pct()
    out = MAIN / "work" / "depth" / "report" / "e2e.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print("masks per keyframe", res["masks_s_per_keyframe"])


if __name__ == "__main__":
    main()
