"""Distil repaired chase views into the stitched splat map: one pass of the stitch optimiser
(pipeline/stitch/v2fit.py, used unmodified through its --pseudo interface).

  .venv/bin/python -m pipeline.genfill.distil pseudo SET FIX OUT.json   # weighted pseudo-view manifest
  .venv/bin/python -m pipeline.genfill.distil fit --source scene/stitched-opt-v3 --out OUT [--pseudo M] [--steps N]
  .venv/bin/python -m pipeline.genfill.distil check MANIFEST SCENE...                  # does a scene reproduce the repairs?

pseudo: one entry per repaired view, in v2fit's format: {"rgb": fixed PNG, "weight": uint8 PNG,
"K": [fx, fy, cx, cy], "T_world_cam": 4x4 camera-to-world, OpenCV axes} plus id/t/class fractions.
weight = 0 for fixer-only and sky pixels (class 2, 3); otherwise the cross-view agreement
(agree/255 from consistency.py), or 0.25 where no neighbour sees the pixel. v2fit multiplies this by
its own pseudo gain (0.10 of a photograph's loss) and draws a pseudo view on 1 update in 16; pseudo
pixels never vote on depth, exposure, sky or densification.
fit: runs v2fit with STITCH_V2_WORK pointed at a private mirror of work/stitch-v2 (symlinks), so its
checkpoints and fit report land in work/genfill/distil/, then gives the exported scene the source
scene's ground tiles, sky panorama and map keys (v2fit's export links the v2 mosaic ground).
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2 as cv
import numpy as np

RS = Path(os.environ.get("RS", Path.home() / "dev/reality-splat")).resolve()
D = RS / "work/genfill/distil"


def pseudo(set_dir, fix, out):
    set_dir, fix, out = (RS / set_dir).resolve(), (RS / fix).resolve(), (RS / out).resolve()
    views = {v["id"]: v for v in json.loads((set_dir / "views.json").read_text())["views"]}
    man = json.loads((fix / "fixed.json").read_text())["views"]
    wdir = out.parent / (out.stem + "-weights")
    wdir.mkdir(parents=True, exist_ok=True)
    entries, tot = [], []
    for vid in sorted(man):
        cl = cv.imread(str(fix / "class" / f"{vid}.png"), cv.IMREAD_UNCHANGED)
        ag = cv.imread(str(fix / "agree" / f"{vid}.png"), cv.IMREAD_UNCHANGED)
        if cl is None or ag is None:
            continue
        w = np.where(ag == 128, 0.25, ag / 255.0) * (cl <= 1)
        cv.imwrite(str(wdir / f"{vid}.png"), (w * 255 + 0.5).astype("B"))
        v = views[vid]
        K = np.array(v["K"])
        entries.append(dict(id=vid, t=v["t"], rgb=str(fix / "fixed" / f"{vid}.png"),
                            weight=str(wdir / f"{vid}.png"), K=[K[0, 0], K[1, 1], K[0, 2], K[1, 2]],
                            T_world_cam=v["T_world_cam"], mean_weight=float(w.mean())))
        tot.append(w.mean())
    out.write_text(json.dumps(entries, indent=1))
    print(len(entries), "pseudo views, mean weight", round(float(np.mean(tot)), 3), flush=True)


def mirror():
    m = D / "v2work"
    m.mkdir(parents=True, exist_ok=True)
    for p in (RS / "work/stitch-v2").iterdir():
        q = m / p.name
        if not q.exists() and not q.is_symlink():
            q.symlink_to(p)
    return m


def patch_scene(src, dst):
    sm = json.loads((src / "map.json").read_text())
    dm = json.loads((dst / "map.json").read_text())
    for t in dm["tiles"]:
        if t["layer"] == "ground":
            (dst / t["file"]).unlink(missing_ok=True)
    ground = [t for t in sm["tiles"] if t["layer"] == "ground"]
    for t in ground:
        (dst / t["file"]).parent.mkdir(parents=True, exist_ok=True)
        os.link(src / t["file"], dst / t["file"])
    out = {k: v for k, v in sm.items() if k != "tiles"}
    out["tiles"] = ground + [t for t in dm["tiles"] if t["layer"] != "ground"]
    if sm.get("sky"):
        shutil.copy(src / sm["sky"], dst / sm["sky"])
    (dst / "map.json").write_text(json.dumps(out, indent=1))


def fit(source, out, pseudo_manifest=None, steps=14400):
    env = dict(os.environ, STITCH_V2_WORK=str(mirror()),
               PYTHONPATH=str(RS / "envs/stitch/lib/python3.12/site-packages"))
    cmd = [str(RS / ".venv/bin/python"), "-m", "pipeline.stitch.v2fit", "--source", source, "--out", out,
           "--steps", str(steps)] + (["--pseudo", pseudo_manifest] if pseudo_manifest else [])
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=RS, env=env, check=True)
    patch_scene(RS / source, RS / out)
    print("patched", out, flush=True)


def check(manifest, scenes, every=8):
    """Render each scene at every n-th pseudo pose and measure how close it is to the repaired target,
    weighted by the pseudo weight (RGB levels), and to the first scene's render."""
    from pipeline.genfill.views import Scene

    entries = json.loads((RS / manifest).read_text())[::every]
    out = {}
    first = {}
    for name in scenes:
        sc = Scene(name)
        to_fix, to_first = [], []
        for e in entries:
            im = cv.imread(e["rgb"])[:, :, ::-1].astype("f")
            w = cv.imread(e["weight"], cv.IMREAD_UNCHANGED).astype("f") / 255
            fx, fy, cx, cy = e["K"]
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            r, a, _ = sc.render(np.array(e["T_world_cam"]), K, im.shape[1::-1], e["t"])
            d = np.abs(r - im).mean(-1)
            to_fix.append(float((d * w).sum() / max(w.sum(), 1)))
            if e["id"] in first:
                to_first.append(float(np.abs(r - first[e["id"]]).mean()))
            else:
                first[e["id"]] = r
        out[name] = dict(weighted_mae_to_repaired=float(np.mean(to_fix)),
                         mae_to_first_scene=float(np.mean(to_first)) if to_first else 0.0, views=len(entries))
        print(name, out[name], flush=True)
    (D / "check.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pseudo")
    p.add_argument("set")
    p.add_argument("fix")
    p.add_argument("out")
    f = sub.add_parser("fit")
    f.add_argument("--source", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--pseudo")
    f.add_argument("--steps", type=int, default=14400)
    c = sub.add_parser("check")
    c.add_argument("manifest")
    c.add_argument("scenes", nargs="+")
    a = ap.parse_args()
    if a.cmd == "pseudo":
        pseudo(a.set, a.fix, a.out)
    elif a.cmd == "check":
        check(a.manifest, a.scenes)
    else:
        fit(a.source, a.out, a.pseudo, a.steps)
