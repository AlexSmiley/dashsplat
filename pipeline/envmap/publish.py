"""Publish an environment map as a new variant scene, leaving the base scene untouched: the variant is
hard links to every file of the base scene, except the files that later steps rewrite in place (map.json,
generated.json, rsb/index.json), which are copies; map.json gains "environment", and environment/ holds
the map, its provenance and environment.json (pipeline/envmap/fill.py).
"""

import argparse
import json
import os
import shutil
from pathlib import Path

RS = Path(os.environ.get("RS", Path.home() / "dev/reality-splat")).resolve()
COPY = {"map.json", "generated.json", "rsb/index.json"}


def publish(base, name, src, reach=None):
    B, N, src = RS / "scene" / base, RS / "scene" / name, RS / src
    if not N.exists():
        for root, _, files in os.walk(B):
            rel = Path(root).relative_to(B)
            (N / rel).mkdir(parents=True, exist_ok=True)
            for f in files:
                r = (rel / f).as_posix()
                (shutil.copy2 if r in COPY else os.link)(B / r, N / r)
    m = json.loads((N / "map.json").read_text())
    m["environment"] = "environment/environment.json"
    tmp = N / "map.json.tmp"
    tmp.write_text(json.dumps(m))
    tmp.replace(N / "map.json")
    (N / "environment").mkdir(exist_ok=True)
    for f in ("sky.jpg", "provenance.png"):
        shutil.copy2(src / f, N / "environment" / f)
    e = json.loads((src / "environment.json").read_text())
    if reach:
        r = json.loads((RS / reach).read_text())
        e["far_band"] = (f"none: from the chase camera ({r['camera_travel_m']} m of travel), content frozen at 1 km "
                         f"would slide up to {r['distances']['1000']['max_px']} px, at 10 km up to "
                         f"{r['distances']['10000']['max_px']} px (threshold {r['threshold_px']} px)")
    (N / "environment" / "environment.json").write_text(json.dumps(e, indent=1))
    print("published", N, json.dumps({k: e[k] for k in ("horizon", "share_upper_hemisphere", "far_band")}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="stitched-opt-v5-movers-v1")
    ap.add_argument("--name", default="stitched-opt-v5-movers-v1-env1")
    ap.add_argument("--src", default="work/envmap/300-316")
    ap.add_argument("--reach", default="work/envmap/300-316/reach.json")
    a = ap.parse_args()
    publish(a.base, a.name, a.src, a.reach)
