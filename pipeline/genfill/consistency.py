"""Cross-view agreement of repaired views, for distillation.

For every repaired view, warp each neighbour that exists (same chase kind at the previous and next
station, 2 m of travel apart; and mid <-> orbit 30 at the same station) into it through the
rendered depth, occlusion-tested. Per pixel, excess = |neighbour - this| minus the same quantity for
the plain renders (the splats' own view-dependence and occlusion error), after a 1.5 px blur, in
RGB levels; the median over visible neighbours is kept. Writes, per view:
  agree/<id>.png   255 x clip(1 - excess / (2 tol)); 128 where no neighbour sees the pixel
and demotes repaired or kept pixels with excess >= tol to class 2 (fixer-only) in class/<id>.png;
the single-view classes are kept in class-single/<id>.png. Run after fix_views.py.

  envs/genfill/bin/python -m pipeline.genfill.consistency SET --fix FIXDIR [--tol 10]
"""

import argparse
import json
import shutil
from pathlib import Path

import cv2 as cv
import numpy as np

from pipeline.genfill.evaluate import RS, render_aux, rgb, warp


def neighbours(v, views):
    k, kind = v["station"], v["chase"]
    out = [f"chase-{k - 1:04d}-{kind}", f"chase-{k + 1:04d}-{kind}"]
    out += [f"chase-{k:04d}-{x}" for x in ("orb-30", "orb+30")] if kind == "mid" else (
        [f"chase-{k:04d}-mid"] if kind in ("orb-30", "orb+30") else [])
    return [n for n in out if n in views]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("set", type=Path)
    ap.add_argument("--fix", type=Path, required=True)
    ap.add_argument("--tol", type=float, default=10.0)
    a = ap.parse_args()
    set_dir, fix = (RS / a.set).resolve(), (RS / a.fix).resolve()
    views = {v["id"]: v for v in json.loads((set_dir / "views.json").read_text())["views"]}
    man = json.loads((fix / "fixed.json").read_text())
    have = [vid for vid in man["views"] if (fix / "fixed" / f"{vid}.png").exists()]
    (fix / "agree").mkdir(exist_ok=True)
    (fix / "class-single").mkdir(exist_ok=True)
    blur = lambda im: cv.GaussianBlur(im.astype("f"), (0, 0), 1.5)
    tot = dict(covered=0, seen=0, agree=0, demoted=0)
    for vid in have:
        v = views[vid]
        nb = [n for n in neighbours(v, views) if n in man["views"]]
        alpha, z = render_aux(set_dir, vid)
        single = fix / "class-single" / f"{vid}.png"
        if not single.exists():
            shutil.copy(fix / "class" / f"{vid}.png", single)
        cl = cv.imread(str(single), cv.IMREAD_UNCHANGED)
        f0, r0 = blur(rgb(fix / "fixed" / f"{vid}.png")), blur(rgb(set_dir / "render" / f"{vid}.png"))
        ex = []
        for n in nb:
            _, zn = render_aux(set_dir, n)
            wf, vis = warp(rgb(fix / "fixed" / f"{n}.png"), views[n], zn, v, z)
            wr, _ = warp(rgb(set_dir / "render" / f"{n}.png"), views[n], zn, v, z)
            e = np.abs(blur(wf) - f0).mean(-1) - np.abs(blur(wr) - r0).mean(-1)
            ex.append(np.where(vis & (alpha >= 0.5), np.maximum(e, 0), np.nan))
        if ex:
            with np.errstate(all="ignore"):
                med = np.nanmedian(np.stack(ex), 0)
        else:
            med = np.full(alpha.shape, np.nan)
        seen = np.isfinite(med)
        ag = np.where(seen, np.clip(1 - np.nan_to_num(med) / (2 * a.tol), 0, 1) * 255, 128).astype("B")
        bad = seen & (np.nan_to_num(med) >= a.tol) & (cl <= 1)
        cl2 = cl.copy()
        cl2[bad] = 2
        cv.imwrite(str(fix / "agree" / f"{vid}.png"), ag)
        cv.imwrite(str(fix / "class" / f"{vid}.png"), cl2)
        cov = alpha >= 0.5
        man["views"][vid].update(agree_seen=float((seen & cov).sum() / max(cov.sum(), 1)),
                                 agree_ok=float((seen & cov & (np.nan_to_num(med) < a.tol)).sum() / max((seen & cov).sum(), 1)),
                                 kept=float((cl2 == 0).mean()), repaired=float((cl2 == 1).mean()),
                                 fixer_only=float((cl2 == 2).mean()), sky=float((cl2 == 3).mean()))
        tot["covered"] += int(cov.sum())
        tot["seen"] += int((seen & cov).sum())
        tot["agree"] += int((seen & cov & (np.nan_to_num(med) < a.tol)).sum())
        tot["demoted"] += int(bad.sum())
    man["settings"]["consistency"] = dict(tol_levels=a.tol, blur_px=1.5, totals=tot)
    (fix / "fixed.json").write_text(json.dumps(man, indent=1))
    print(len(have), "views", tot, "agree fraction of seen covered", round(tot["agree"] / max(tot["seen"], 1), 3),
          flush=True)


if __name__ == "__main__":
    main()
