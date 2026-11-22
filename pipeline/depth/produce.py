#!/usr/bin/env python3
"""Polite background producer for depth and masks over the whole drive, in priority order.

Phases (each step is a separate, resumable process, run with nice; the tools wait while system free
memory is below --min-free %, and each process holds one model at a time):
  1. priority windows 286-330 s and 190-215 s, all six cameras; then the end-to-end benchmark (bench_e2e.py)
  2. the rest of the drive 80-361 s, front + pillars first, then repeaters + rear
  3. the parked start 0-80 s, 2 Hz time-based keyframes only
Per window: keyframes.py (0.5 m distance + 2 Hz time-based below 1 m/s) -> run_masks.py -> run_depth.py
-> verify_outputs.py -> run_masks.py --motion. Progress: work/depth/produce_status.json and the log.

Usage: nice -n 10 python pipeline/depth/produce.py [--phases 1,2,3]
Check: cat $RS/work/depth/produce_status.json ; tail $RS/work/depth/produce.log
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rscommon import RS, atomic_write_text  # noqa: E402

PY = sys.executable
D = RS / "pipeline" / "depth"
M = RS / "pipeline" / "masks"
STATUS = RS / "work" / "depth" / "produce_status.json"
FRONT_SIDES = "front,left_pillar,right_pillar"
REAR = "left_repeater,right_repeater,back"
ALL = "front,left_pillar,right_pillar,left_repeater,right_repeater,back"
END = 361.5


def plan():
    p = []
    for t0, t1 in [(286, 330), (190, 215)]:
        p.append(dict(phase=1, name=f"priority {t0}-{t1}", t0=t0, t1=t1, keyframes=[], cams=[ALL]))
    p.append(dict(phase=1, name="e2e benchmark", bench=True))
    rest = [(80, 140), (140, 190), (215, 286), (330, END)]
    p.append(dict(phase=2, name="rest front+pillars", ranges=rest, cams=[FRONT_SIDES]))
    p.append(dict(phase=2, name="rest repeaters+rear", ranges=rest, cams=[REAR]))
    p.append(dict(phase=3, name="parked 0-80 (time only)", t0=0, t1=80, cams=[ALL], time_only=True))
    return p


def run(cmd, log):
    t = time.time()
    log.write(f"\n$ {' '.join(map(str, cmd))}\n")
    log.flush()
    r = subprocess.run(list(map(str, cmd)), stdout=log, stderr=subprocess.STDOUT, cwd=RS)
    log.write(f"# exit {r.returncode} after {time.time() - t:.0f} s\n")
    log.flush()
    return r.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phases", default="1,2,3")
    ap.add_argument("--min-free", default="20")
    args = ap.parse_args()
    phases = {int(x) for x in args.phases.split(",")}
    status = json.loads(STATUS.read_text()) if STATUS.exists() else {"steps": {}}
    log = open(RS / "work" / "depth" / "produce.log", "a")

    def mark(key, **kw):
        status["steps"].setdefault(key, {}).update(kw, updated=time.strftime("%H:%M:%S"))
        status["current"] = key
        atomic_write_text(STATUS, json.dumps(status, indent=1))

    mf = ["--min-free", args.min_free]
    for step in plan():
        if step["phase"] not in phases:
            continue
        ranges = step.get("ranges") or ([(step["t0"], step["t1"])] if "t0" in step else [])
        key = step["name"]
        if status["steps"].get(key, {}).get("state") == "done":
            continue
        mark(key, state="running", started=time.strftime("%H:%M:%S"))
        if step.get("bench"):
            rc = run([PY, D / "bench_e2e.py", "--t0", "300", "--k", "12"], log)
            mark(key, state="done" if rc == 0 else "failed", finished=time.strftime("%H:%M:%S"))
            continue
        ok = True
        for t0, t1 in ranges:
            kcmd = [PY, D / "keyframes.py", "--t0", t0, "--t1", t1, "--cams", "all"]
            if step.get("time_only"):
                kcmd.append("--time-only")
            ok &= run(kcmd, log) == 0
        for cams in step["cams"]:
            for t0, t1 in ranges:
                mark(key, state="running", now=f"masks {cams} {t0}-{t1}")
                ok &= run([PY, M / "run_masks.py", "--cams", cams, "--t0", t0, "--t1", t1] + mf, log) == 0
            for t0, t1 in ranges:
                mark(key, state="running", now=f"depth {cams} {t0}-{t1}")
                ok &= run([PY, D / "run_depth.py", "--cams", cams, "--t0", t0, "--t1", t1] + mf, log) == 0
        for t0, t1 in ranges:
            vj = RS / "work" / "depth" / "report" / f"verify_{int(t0)}_{int(t1)}.json"
            v = run([PY, D / "verify_outputs.py", "--t0", t0, "--t1", t1, "--json", vj], log)
            mark(key, **{f"verify_{int(t0)}_{int(t1)}": "ok" if v == 0 else "problems (see json)"})
        run([PY, M / "run_masks.py", "--cams", "all", "--motion"], log)
        mark(key, state="done" if ok else "done-with-errors", finished=time.strftime("%H:%M:%S"))
    status["current"] = "finished"
    atomic_write_text(STATUS, json.dumps(status, indent=1))


if __name__ == "__main__":
    main()
