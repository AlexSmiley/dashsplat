#!/usr/bin/env python3
"""Apply per-camera depth scale factors to already written depth maps (idempotent).

Each depth/<cam>/index.json entry records the factor already applied (scale_applied); this rescales
the file by new/old and updates the entry, so running it twice with the same factors changes nothing.
The factors are also recorded in keyframes.json -> depth.scale_factors.

Usage: apply_scale.py --factors front=0.957,left_pillar=1.05,...
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rscommon import RS  # noqa: E402

import numpy as np  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--factors", required=True)
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    fac = {k: float(v) for k, v in (s.split("=") for s in args.factors.split(","))}
    for cam, f in fac.items():
        ip = RS / "depth" / cam / "index.json"
        if not ip.exists():
            continue
        idx = json.loads(ip.read_text())
        changed = 0
        for n, e in idx.items():
            old = e.get("scale_applied", 1.0)
            if abs(old - f) < 1e-9:
                continue
            p = RS / "depth" / cam / f"{int(n):06d}.npy"
            if not p.exists():
                continue
            z = np.load(p).astype(np.float32) * (f / old)
            np.save(p, z.astype(np.float16))
            if e.get("road_h_ratio") is not None:
                e["road_h_ratio"] = round(e["road_h_ratio"] * f / old, 4)
            e["scale_applied"] = f
            changed += 1
        ip.write_text(json.dumps(idx))
        print(f"{cam}: factor {f} applied to {changed} files")
    kp = RS / "depth" / "keyframes.json"
    kf = json.loads(kp.read_text())
    d = kf.setdefault("depth", {})
    d.setdefault("scale_factors", {}).update(fac)
    if args.note:
        d["scale_factors_note"] = args.note
    tmp = kp.with_suffix(".tmp")
    tmp.write_text(json.dumps(kf, indent=1))
    tmp.replace(kp)


if __name__ == "__main__":
    main()
