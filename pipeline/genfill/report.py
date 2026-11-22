"""Report images for pipeline/genfill/REPORT.md (each under 500 KB)."""

import sys
from pathlib import Path

import cv2 as cv
import numpy as np

from pipeline.genfill.evaluate import RS, label, pairsheet, rgb, save_small, sheet

G = RS / "work/genfill"
OUT = RS / "pipeline/genfill/report"


def ibr_sheet(t, out, width=720):
    S, F = G / "sets/ibr-v1", G / "fix/ibr-v1/ref-t199"
    rs = lambda im: cv.resize(im, (width, width * 896 // 1408), interpolation=cv.INTER_AREA)
    a = [rs(rgb(S / "render" / f"ibr-{t}.png")), rs(rgb(F / "fixed" / f"ibr-{t}.png")),
         rs(rgb(S / "render" / f"splat-{t}.png")), rs(rgb(F / "fixed" / f"splat-{t}.png"))]
    names = [f"image-based prototype {t} s (grey box = car)", "... repaired", f"live-v4 splats, same camera",
             "... repaired"]
    a = [label(x, n) for x, n in zip(a, names)]
    save_small(out, np.vstack([np.hstack(a[:2]), np.hstack(a[2:])]))


def main(chase_ids):
    OUT.mkdir(parents=True, exist_ok=True)
    real = G / "sets/real-v4"
    fr = G / "fix/real-v4/ref-t199"
    for vid, name, title in [
        ("real-right_pillar-004875", "02-real-buildings-from-reference", "held-out right pillar 4875"),
        ("real-back-004209", "03-real-fail-wrong-paint-sharpened", "held-out back 4209"),
        ("real-right_repeater-003330", "04-real-fail-arrow-invented", "held-out right repeater 3330"),
    ]:
        sheet(real, fr, vid, OUT / f"{name}.jpg", title)
    sheet(G / "sets/real-v4-far", G / "fix/real-v4-far/ref-t199", "real-right_pillar-004875",
          OUT / "05-real-far-reference.jpg", "same view, reference >= 10 m away")
    ch, fc = G / "sets/chase-v4", G / "fix/chase-v4/ref-t199"
    for i, (vid, title) in enumerate(chase_ids):
        sheet(ch, fc, vid, OUT / f"{6 + i:02d}-{vid}.jpg", title)
    n = 6 + len(chase_ids)
    pairsheet(ch, fc, "chase-0240-mid", "chase-0241-mid", OUT / f"{n:02d}-consistency-travel.jpg")
    pairsheet(ch, fc, "chase-0026-mid", "chase-0026-orb+30", OUT / f"{n + 1:02d}-consistency-orbit.jpg")
    ibr_sheet("112.0", OUT / f"{n + 2:02d}-ibr-vs-splat-112.jpg")
    ibr_sheet("97.0", OUT / f"{n + 3:02d}-ibr-vs-splat-97.jpg")


if __name__ == "__main__":
    ids = [x.split("=", 1) for x in sys.argv[1:]]
    main([(a, b) for a, b in ids])
