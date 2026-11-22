#!/usr/bin/env python3
"""Speed and memory benchmark on the M5 Ultra (MPS) at realistic input sizes.

For every model: one image (front camera) and a batch of six (all six cameras at one moment,
rectified to pinhole views of the same size; posed models get the rig poses). Peak memory is the
largest MPS driver allocation seen while the call runs (weights + activations + allocator cache).

Usage: bench_speed.py [--models all] [--out $RS/work/depth/report/speed.json]
"""
import argparse
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "masks"))
from rscommon import RS, setup_env, load_rig, trajectory, T_world_cam, frame_at_time, load_image, default_views, CAMS  # noqa

setup_env()
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import models  # noqa: E402

# (model, width, mode) mode: mono = independent images; posed = pose-conditioned multi-view;
# unposed = multi-view without poses
RUNS = [
    ("da3metric-large", 518, "mono"), ("da3metric-large", 1036, "mono"),
    ("da3-small", 518, "posed"), ("da3-base", 518, "posed"),
    ("da3-large-1.1", 518, "posed"), ("da3-large-1.1", 1036, "posed"),
    ("da3nested-1.1", 518, "mono"), ("da3nested-1.1", 518, "posed"), ("da3nested-1.1", 518, "unposed"),
    ("mapanything-apache", 518, "mono"), ("mapanything-apache", 518, "posed"),
    ("mapanything-apache", 1036, "posed"),
    ("mapanything", 518, "posed"),
    ("moge2-vitl", 518, "mono"), ("moge2-vitb", 518, "mono"), ("moge2-vitl", 1036, "mono"),
    ("unidepth-v2-l", 518, "mono"),
]


class PeakMem:
    def __enter__(self):
        self.peak = torch.mps.driver_allocated_memory()
        self.run = True
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()
        return self

    def _loop(self):
        while self.run:
            self.peak = max(self.peak, torch.mps.driver_allocated_memory())
            time.sleep(0.002)

    def __exit__(self, *a):
        self.run = False
        self.th.join()
        self.peak = max(self.peak, torch.mps.driver_allocated_memory())


def timed(fn, reps=5):
    fn()
    models.sync()
    ts = []
    with PeakMem() as pm:
        for _ in range(reps):
            t0 = time.time()
            fn()
            models.sync()
            ts.append(time.time() - t0)
    return float(np.median(ts)), pm.peak / 2**30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="all")
    ap.add_argument("--out", default=str(RS / "work/depth/report/speed.json"))
    ap.add_argument("--t", type=float, default=125.0)
    args = ap.parse_args()
    rig = load_rig()
    traj = trajectory()
    views = {}
    imgs = {}
    for w in (518, 1036):
        for c in CAMS:
            v = default_views(rig[c], w)
            v = v[1] if len(v) == 3 else v[0]
            # all six at the front aspect so they batch: wide views get the same size (w x 14k)
            views[(c, w)] = v
    out = json.loads(Path(args.out).read_text()) if Path(args.out).exists() else {}
    runs = RUNS if args.models == "all" else [r for r in RUNS if r[0] in args.models.split(",")]
    cur = None
    for mname, w, mode in runs:
        if cur is None or cur.name != mname:
            cur = None
            torch.mps.empty_cache()
            cur = models.load_model(mname)
        ims, Ks, Ts = [], [], []
        for c in CAMS:
            v = views[(c, w)]
            n = frame_at_time(c, args.t)
            im = v.rectify(load_image(c, n))
            if (im.shape[1], im.shape[0]) != views[("front", w)].size:
                # batch needs one size: centre-crop/pad to the front view size
                W, H = views[("front", w)].size
                canvas = np.zeros((H, W, 3), np.uint8)
                hh, ww = min(H, im.shape[0]), min(W, im.shape[1])
                canvas[:hh, :ww] = im[:hh, :ww]
                im = canvas
            Rv = np.eye(4)
            Rv[:3, :3] = v.R
            ims.append(im)
            Ks.append(v.K)
            Ts.append(T_world_cam(rig[c], n, traj) @ Rv)
        key = f"{mname}@{w}/{mode}"
        rec = dict(model=mname, width=w, height=ims[0].shape[0], mode=mode, licence=cur.licence)
        try:
            if mode == "mono":
                rec["single_s"], rec["single_gb"] = timed(lambda: cur.predict(ims[:1], Ks[:1], None))
            else:
                # front temporal pair as the smallest multi-view call
                rec["single_s"], rec["single_gb"] = (None, None)
            if mode == "posed":
                f6 = lambda: cur.predict(ims, Ks, Ts)
            elif mode == "unposed":
                f6 = lambda: cur.predict(ims, Ks, None, use_pose=False)
            else:
                f6 = lambda: cur.predict(ims, Ks, None)
            b, gb = timed(f6, reps=3)
            rec["batch6_s"], rec["batch6_per_image_s"], rec["batch6_gb"] = b, b / 6, gb
        except Exception as e:
            rec["error"] = repr(e)[:300]
        out[key] = rec
        print(key, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec.items()}, flush=True)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=1))
    # segmentation
    from seg import Segmenter
    s = Segmenter()
    for size in [(724, 469), (1036, 672)]:
        im = load_image("front", frame_at_time("front", args.t), size)
        t1, g1 = timed(lambda: s([im]))
        t6, g6 = timed(lambda: [s([im]) for _ in range(6)], reps=2)
        out[f"mask2former-swinL@{size[0]}"] = dict(model="mask2former-swin-large-cityscapes-panoptic", width=size[0],
                                                  height=size[1], single_s=t1, single_gb=g1, batch6_s=t6,
                                                  batch6_per_image_s=t6 / 6, batch6_gb=g6, licence="MIT code, Cityscapes weights")
        print("mask2former", size, t1, g1, t6 / 6, flush=True)
    Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
