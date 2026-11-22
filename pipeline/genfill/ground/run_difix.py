"""Refine the three bird's-eye patches with Difix (noref / ref, t99 / t199). Run with envs/genfill/bin/python."""

import time

import cv2 as cv
import numpy as np
import torch

from pipeline.genfill.difix import Difix, sync
from pipeline.genfill.ground.common import PATCHES, WORK, load_patch, save_json, tiled

TILE, OVERLAP, REF_OFFSET = 768, 128, 384


def main():
    imgs = {p: load_patch(p)[0] for p in PATCHES}
    times = {}
    for ref in (False, True):
        m = Difix(ref=ref)
        for t in (99, 199):
            key = f"difix-{'ref' if ref else 'noref'}-t{t}"
            for p, img in imgs.items():
                h, w = img.shape[:2]
                ts = []

                def fn(tile, yx):
                    r = None
                    if ref:  # nearby tile of the same mosaic patch, half a tile away (50 % overlap)
                        y = yx[0] + REF_OFFSET if yx[0] + REF_OFFSET + TILE <= h else yx[0] - REF_OFFSET
                        x = yx[1] + REF_OFFSET if yx[1] + REF_OFFSET + TILE <= w else yx[1] - REF_OFFSET
                        r = np.ascontiguousarray(img[y : y + TILE, x : x + TILE])
                    sync(m.device)
                    t0 = time.perf_counter()
                    o = m(tile, r, timestep=t)
                    sync(m.device)
                    ts.append(time.perf_counter() - t0)
                    return o

                out = tiled(img, fn, TILE, OVERLAP)
                (WORK / "out" / p).mkdir(parents=True, exist_ok=True)
                cv.imwrite(str(WORK / "out" / p / f"{key}.png"), out[:, :, ::-1])
                times.setdefault(key, {})[p] = dict(tiles=len(ts), s_per_tile_median=float(np.median(ts[1:] or ts)), first=ts[0])
                print(key, p, times[key][p], flush=True)
        del m
        torch.mps.empty_cache()
    save_json(WORK / "timing-difix.json", dict(tile=TILE, overlap=OVERLAP, times=times))


if __name__ == "__main__":
    main()
