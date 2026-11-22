"""Fixer input comparison: the image-based prototype's frames (pipeline/ibr/report, real photographs
re-projected through depth meshes: sharp but torn) against splat renders of live-v4 from exactly
the same camera (the controller replay pipeline/livemap/v5_proof.Camera that made those videos).
Both are repaired with the same reference photo. Writes a render set usable by fix_views.py.

  .venv/bin/python -m pipeline.genfill.ibr_compare FRAMES_DIR OUT_SET
FRAMES_DIR holds ibr-<t>.png frames extracted from the prototype's videos (1440x900).
"""

import sys
from pathlib import Path

import cv2 as cv
import numpy as np

from pipeline.genfill import views as V
from pipeline.livemap.v5_proof import Camera

SRC = (1440, 900)
SIZE = (1408, 896)


def main(frames, out):
    frames, out = Path(frames), Path(out)
    scene = V.Scene("scene/live-v4")
    cam = Camera(V.RS / "scene/live-v5")
    K0 = V.K_vfov(50.0, SRC)
    K = V.K_vfov(50.0, SIZE)
    K[0, 0] = K0[0, 0] * SIZE[0] / SRC[0]  # an anisotropic resize of the 1440x900 frame
    K[1, 1] = K0[1, 1] * SIZE[1] / SRC[1]
    views = []
    for p in sorted(frames.glob("ibr-*.png")):
        t = float(p.stem[4:])
        T, _, _ = cam.pose(t, 0)
        rgb, alpha, z = scene.render(T, K, SIZE, t)
        refs = V.choose_refs(scene, T, z, K, SIZE)
        for r in refs:
            r["path"] = V.ensure_ref(out, r, SIZE)
        ibr = cv.resize(cv.imread(str(p))[:, :, ::-1], SIZE, interpolation=cv.INTER_AREA)
        for name, im, a in (("splat", rgb, alpha), ("ibr", ibr.astype("f"), np.ones_like(alpha))):
            vid = f"{name}-{t:.1f}"
            V.write_view(out, vid, im, a, z)
            views.append(dict(id=vid, kind=name, t=t, t_map=t, T_world_cam=T.tolist(), K=K.tolist(),
                              size=list(SIZE), refs=refs, coverage=float((a > 0.5).mean())))
        print(t, [(r["c"], round(r["t"], 1), round(r["score"], 2)) for r in refs[:1]], flush=True)
    V.savej(out / "views.json", dict(scene=str(scene.root), size=list(SIZE),
                                     note="ibr alpha is unknown and set to 1", views=views))


if __name__ == "__main__":
    main(*sys.argv[1:3])
