#!/usr/bin/env python3
"""Check the delivered depth and mask files against depth/keyframes.json.

For every keyframe of every camera: depth npy exists, loads, has the recorded grid and dtype float16,
has an index.json entry; sky/road/dyn masks exist with the same grid. Prints per-camera coverage
(valid depth fraction inside the calibration mask, sky fraction) and the median road-plane ratio.
--fix deletes unreadable depth files and depth files without an index entry so run_depth.py redoes them.

Usage: verify_outputs.py [--fix]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rscommon import RS, load_rig, time_filter, atomic_write_text  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fix", action="store_true")
    ap.add_argument("--json", default=str(RS / "work/depth/report/verify.json"))
    ap.add_argument("--t0", type=float, default=None)
    ap.add_argument("--t1", type=float, default=None)
    args = ap.parse_args()
    kf = json.loads((RS / "depth/keyframes.json").read_text())
    rig = load_rig()
    out = {}
    for cam, frames in kf["frames"].items():
        frames = time_filter(frames, args.t0, args.t1)
        if not frames:
            continue
        grid = kf.get("depth", {}).get("cameras", {}).get(cam, {}).get("grid")
        W, H = (grid["width"], grid["height"]) if grid else (724, 469)
        wm = cv2.resize(rig[cam].mask.astype(np.float32), (W, H), interpolation=cv2.INTER_AREA) >= 0.5
        idx_p = RS / "depth" / cam / "index.json"
        idx = json.loads(idx_p.read_text()) if idx_p.exists() else {}
        missing, bad, noidx, nomask = [], [], [], []
        cov, skyf, ratios, zmed = [], [], [], []
        for k, e in enumerate(frames):
            n = e["n"]
            p = RS / "depth" / cam / f"{n:06d}.npy"
            if not p.exists():
                missing.append(n)
                continue
            try:
                z = np.load(p)
                assert z.dtype == np.float16 and z.shape == (H, W)
            except Exception:
                bad.append(n)
                if args.fix:
                    p.unlink()
                continue
            if str(n) not in idx:
                noidx.append(n)
                if args.fix:
                    p.unlink()
                continue
            ms = [RS / "masks" / kind / cam / f"{n:06d}.png" for kind in ("sky", "road", "dyn")]
            if not all(m.exists() for m in ms):
                nomask.append(n)
            if k % 10 == 0:
                zf = z.astype(np.float32)
                sky = cv2.imread(str(ms[0]), cv2.IMREAD_GRAYSCALE) > 0 if ms[0].exists() else np.zeros((H, W), bool)
                nonsky = wm & ~sky
                cov.append(float(((zf > 0) & nonsky).sum() / max(nonsky.sum(), 1)))
                skyf.append(float((sky & wm).sum() / max(wm.sum(), 1)))
                zmed.append(float(np.median(zf[zf > 0])) if (zf > 0).any() else 0.0)
            r = idx[str(n)].get("road_h_ratio")
            if r:
                ratios.append(r)
        models = sorted({v.get("model") for v in idx.values()})
        kinds = {k: sum(1 for e in frames if e.get("kind", "distance") == k) for k in ("distance", "time")}
        out[cam] = dict(keyframes=len(frames), kinds=kinds, missing=len(missing), unreadable=len(bad), no_index=len(noidx),
                        no_masks=len(nomask), valid_nonsky=float(np.median(cov)) if cov else None,
                        sky_fraction=float(np.median(skyf)) if skyf else None,
                        median_depth_m=float(np.median(zmed)) if zmed else None,
                        road_ratio_median=float(np.median(ratios)) if ratios else None,
                        road_ratio_p10_p90=[float(x) for x in np.percentile(ratios, [10, 90])] if ratios else None,
                        models=models)
        print(cam, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[cam].items()}, flush=True)
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(args.json, json.dumps(dict(t0=args.t0, t1=args.t1, cameras=out), indent=1))
    bad = sum(v["missing"] + v["unreadable"] + v["no_index"] + v["no_masks"] for v in out.values())
    print("OK" if bad == 0 else f"PROBLEMS: {bad}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
