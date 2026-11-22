#!/usr/bin/env python3
"""Summarise bake-off metrics.json files into a markdown table (aggregated over moments).

Usage: summarize.py --cams front [--inp $RS/work/depth/bakeoff] > summary.md
Columns: median road-height ratio and its spread over moments (scale error, 1.00 ideal), median
triangulation scale ratio, triangulation AbsRel after a median rescale (shape error), temporal
consistency AbsRel 1 m / 4 m apart (raw and after per-frame road-plane scale), seconds per image.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rscommon import RS  # noqa: E402


def agg(vals, fn=np.median):
    v = np.array([x for x in vals if x is not None and np.isfinite(x)], float)
    return fn(v) if len(v) else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cams", default="front")
    ap.add_argument("--inp", default=str(RS / "work/depth/bakeoff"))
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    rows = {}
    for cam in args.cams.split(","):
        p = Path(args.inp) / cam / "metrics.json"
        if not p.exists():
            continue
        M = json.loads(p.read_text())
        print(f"\n### {cam} ({len(next(iter(M.values()))['moments'])} moments, trajectory: "
              f"{next(iter(M.values())).get('trajectory')})\n")
        print("| config | licence | road h ratio (med) | road ratio IQR | log-scale err (med abs) | tri scale (med) "
              "| tri AbsRel aligned | cons 1 m | cons 4 m | cons 4 m road-corr | chained: tri scale | chained: cons 4 m "
              "| s/image (bake-off) |")
        print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for c, m in M.items():
            mo = m["moments"].values()
            rr = [x["road_h_ratio"] for x in mo]
            tri = [x["tri_scale"] for x in mo]
            lr = [abs(np.log(x)) for x in rr if x]
            q = np.array([x for x in rr if x])
            r = dict(cam=cam, config=c, licence=m["licence"], road=agg(rr),
                     road_iqr=float(np.percentile(q, 75) - np.percentile(q, 25)) if len(q) else np.nan,
                     logerr=agg(lr), tri=agg(tri), tri_rel=agg([x["tri_absrel_aligned"] for x in mo]),
                     cons1=agg([x["cons1"] for x in mo]), cons4=agg([x["cons4"] for x in mo]),
                     cons4c=agg([x["cons4_corr"] for x in mo]), sec=m["s_per_image"],
                     ctri=agg([x.get("chain_tri_scale") for x in mo]), ccons=agg([x.get("chain_cons4") for x in mo]))
            rows[f"{cam}/{c}"] = r
            print(f"| {c} | {m['licence']} | {r['road']:.2f} | {r['road_iqr']:.2f} | {r['logerr']:.2f} | {r['tri']:.2f} "
                  f"| {100 * r['tri_rel']:.1f}% | {100 * r['cons1']:.1f}% | {100 * r['cons4']:.1f}% "
                  f"| {100 * r['cons4c']:.1f}% | {r['ctri']:.2f} | {100 * r['ccons']:.1f}% | {r['sec']:.3f} |")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
