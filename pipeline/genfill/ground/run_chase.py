"""Real-ESRGAN x2plus on 8 perspective chase renders; sheet render | Difix ref t199 | ESRGAN. Run with envs/genfill-edit."""

import sys
import time
from pathlib import Path

import cv2 as cv
import numpy as np

from pipeline.genfill.ground.common import RS, WORK, tiled
from pipeline.genfill.ground.evaluate import jpeg_under, label
from pipeline.genfill.ground.run_esrgan import ESRGAN, sync

IDS = ["chase-0024-mid", "chase-0030-orb+60", "chase-0236-orb-30", "chase-0240-mid",
       "chase-0026-mid", "chase-0232-mid", "chase-0034-orb-30", "chase-0246-orb+30"]
SRC = RS / "work/genfill/sets/chase-v4/render"
DIFIX = RS / "work/genfill/fix/chase-v4/ref-t199/fixed"
OUT = WORK / "chase"


def main(report):
    m = ESRGAN()
    OUT.mkdir(parents=True, exist_ok=True)
    rows, ts, stats = [], [], {}
    for i in IDS:
        img = cv.imread(str(SRC / f"{i}.png"))[:, :, ::-1]
        sync()
        t0 = time.perf_counter()
        hi = tiled(np.ascontiguousarray(img), lambda t, yx: m(t), 512, 64, scale=2)
        sync()
        ts.append(time.perf_counter() - t0)
        out = cv.resize(hi, img.shape[1::-1], interpolation=cv.INTER_AREA)
        cv.imwrite(str(OUT / f"{i}.png"), out[:, :, ::-1])
        dfx = cv.imread(str(DIFIX / f"{i}.png"))
        d = lambda a: float(np.abs(a.astype(np.int16) - img[:, :, ::-1].astype(np.int16)).mean())
        stats[i] = dict(esrgan_mean_abs=d(out[:, :, ::-1]), difix_mean_abs=d(dfx), seconds=ts[-1])
        half = lambda z: cv.resize(z, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)
        rows.append(np.hstack([label(half(img[:, :, ::-1]).copy(), i), label(half(dfx).copy(), "Difix ref t199"),
                               label(half(out[:, :, ::-1]).copy(), "Real-ESRGAN x2 (to 1x)")]))
        print(i, stats[i], flush=True)
    q, n = jpeg_under(np.vstack(rows), Path(report) / "chase-render-difix-esrgan.jpg")
    print("sheet", q, n, "s per frame", np.median(ts))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else WORK / "report")
