"""Refine the bird's-eye patches with Real-ESRGAN x2plus (BSD-3) via spandrel on MPS. Run with envs/genfill-edit.

Modes:
  esrgan-x2-native: 2 cm mosaic in, 1 cm out (saved), area-downsampled back to 2 cm for the metrics.
  esrgan-x2-half:   mosaic area-downsampled to 4 cm, x2 back to 2 cm (does it restore lost detail?).
"""

import time

import cv2 as cv
import numpy as np
import spandrel
import torch

from pipeline.genfill.ground.common import PATCHES, RS, WORK, load_patch, save_json, tiled

WEIGHTS = RS / "third_party/Real-ESRGAN/weights/RealESRGAN_x2plus.pth"


class ESRGAN:
    def __init__(self, path=WEIGHTS, device="mps"):
        self.model = spandrel.ModelLoader().load_from_file(str(path)).model.eval().to(device)
        self.device = device
        self.scale = 2

    @torch.no_grad()
    def __call__(self, rgb):
        x = torch.from_numpy(rgb).permute(2, 0, 1)[None].float().div(255).to(self.device)
        y = self.model(x).clamp(0, 1)[0].permute(1, 2, 0).mul(255).add(0.5).byte().cpu().numpy()
        return y


def sync():
    torch.mps.synchronize()


def run(m, img, tile, overlap):
    ts = []

    def fn(t, yx):
        sync()
        t0 = time.perf_counter()
        o = m(t)
        sync()
        ts.append(time.perf_counter() - t0)
        return o

    return tiled(img, fn, tile, overlap, scale=2), ts


def main():
    m = ESRGAN()
    times = {}
    for p in PATCHES:
        img, _ = load_patch(p)
        (WORK / "out" / p).mkdir(parents=True, exist_ok=True)
        hi, ts = run(m, img, 512, 64)
        cv.imwrite(str(WORK / "out" / p / "esrgan-x2-native-1cm.png"), hi[:, :, ::-1])
        lo = cv.resize(hi, img.shape[1::-1], interpolation=cv.INTER_AREA)
        cv.imwrite(str(WORK / "out" / p / "esrgan-x2-native.png"), lo[:, :, ::-1])
        times.setdefault("esrgan-x2-native", {})[p] = dict(tile_in=512, tiles=len(ts), s_per_tile_median=float(np.median(ts[1:])), first=ts[0])
        half = cv.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv.INTER_AREA)
        up, ts = run(m, half, 384, 32)
        cv.imwrite(str(WORK / "out" / p / "esrgan-x2-half.png"), up[:, :, ::-1])
        times.setdefault("esrgan-x2-half", {})[p] = dict(tile_in=384, tiles=len(ts), s_per_tile_median=float(np.median(ts[1:])), first=ts[0])
        print(p, {k: v[p] for k, v in times.items()}, flush=True)
    save_json(WORK / "timing-esrgan.json", times)


if __name__ == "__main__":
    main()
