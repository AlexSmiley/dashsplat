#!/usr/bin/env python3
"""Select keyframes and record them in $RS/depth/keyframes.json (shared by the depth and mask tools).

Rules, on the front camera clock within [t0, t1]:
  distance  a frame becomes a keyframe when the car has travelled at least --step metres (planar
            odometer from the telemetry trajectory) since the previous distance keyframe;
  time      while the car is slower than --slow-mps, a frame also becomes a keyframe when --stopped-hz
            worth of time has passed since the previous keyframe of either kind. Static-map builders
            can skip these (entry "kind": "time"); the moving-object work uses them.
--time-only selects only time keyframes (for the parked start of the drive).
Every other camera uses its frame nearest in time to each front keyframe and inherits the kind.

Usage: keyframes.py --t0 286 --t1 330 [--step 0.5 --stopped-hz 2 --slow-mps 1] [--time-only] [--cams all]
Existing entries are kept (and backfilled with kind "distance" if they have none); new ones are merged
in, sorted by frame number. The file is replaced atomically.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rscommon import RS, CAMS, frame_table, frame_at_time, frame_time, atomic_write_text  # noqa: E402

import numpy as np  # noqa: E402

KF_PATH = RS / "depth" / "keyframes.json"


def timeline():
    tl = np.load(RS / "data/drive/timeline.npz")
    d = np.hypot(np.diff(tl["x"]), np.diff(tl["y"]))
    return tl["t"], np.concatenate([[0], np.cumsum(d)]), tl["speed"]


def select(t0, t1, step, stopped_hz=2.0, slow_mps=1.0, time_only=False, last_dist_odo=None):
    ns, ts = frame_table("front")
    tt, oo, sp = timeline()
    odo = np.interp(ts, tt, oo)
    speed = np.interp(ts, tt, sp)
    idx = np.nonzero((ts >= t0) & (ts <= t1))[0]
    out = []
    last_o, last_t = last_dist_odo, None
    for i in idx:
        slow = speed[i] < slow_mps
        kind = None
        if not time_only and (last_o is None or odo[i] - last_o >= step):
            kind = "distance"
            last_o = odo[i]
        elif (slow or time_only) and stopped_hz > 0 and (last_t is None or ts[i] - last_t >= 1.0 / stopped_hz - 1e-3):
            kind = "time"
        if kind:
            out.append(dict(n=int(ns[i]), t=float(ts[i]), odo=round(float(odo[i]), 3),
                            speed=round(float(speed[i]), 2), kind=kind))
            last_t = ts[i]
    return out


def load():
    if KF_PATH.exists():
        return json.loads(KF_PATH.read_text())
    return {"version": 1, "frames": {}, "selections": []}


def save(kf):
    KF_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(KF_PATH, json.dumps(kf, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t0", type=float, required=True)
    ap.add_argument("--t1", type=float, required=True)
    ap.add_argument("--step", type=float, default=0.5)
    ap.add_argument("--stopped-hz", type=float, default=2.0)
    ap.add_argument("--slow-mps", type=float, default=1.0)
    ap.add_argument("--time-only", action="store_true")
    ap.add_argument("--cams", default="all")
    args = ap.parse_args()
    cams = CAMS if args.cams == "all" else args.cams.split(",")
    kf = load()
    for es in kf["frames"].values():
        for e in es:
            e.setdefault("kind", "distance")
    # continue the distance rule from the last existing front distance keyframe before t0
    prev = [e for e in kf["frames"].get("front", []) if e["kind"] == "distance" and e.get("t_key", e["t"]) < args.t0]
    last = max(prev, key=lambda e: e["t"])["odo"] if prev else None
    sel = select(args.t0, args.t1, args.step, args.stopped_hz, args.slow_mps, args.time_only, last)
    added = {}
    for cam in cams:
        have = {e["n"]: e for e in kf["frames"].get(cam, [])}
        a = 0
        for e in sel:
            n = e["n"] if cam == "front" else frame_at_time(cam, e["t"])
            if n not in have:
                have[n] = dict(n=n, t=frame_time(cam, n), t_key=e["t"], odo=e["odo"], speed=e["speed"], kind=e["kind"])
                a += 1
        kf["frames"][cam] = [have[k] for k in sorted(have)]
        added[cam] = a
    kf["selections"].append(dict(t0=args.t0, t1=args.t1, step_m=args.step, stopped_hz=args.stopped_hz,
                                 slow_mps=args.slow_mps, time_only=args.time_only, cams=cams,
                                 rule="distance: front-clock travel >= step since previous distance keyframe; "
                                      "time: while speed < slow_mps, >= 1/stopped_hz since previous keyframe; "
                                      "other cameras take the frame nearest in time"))
    kf["kinds"] = {"distance": "every --step m of travel; use for static maps",
                   "time": "time-based while slow or stopped; static-map builders can skip, moving-object work uses"}
    save(kf)
    nd = sum(e["kind"] == "distance" for e in sel)
    print(f"[{args.t0}, {args.t1}] s: {nd} distance + {len(sel) - nd} time keyframes selected; added per camera {added}")


if __name__ == "__main__":
    main()
