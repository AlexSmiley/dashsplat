"""Make a portable, license-free-of-weights test bundle from the exact 81-frame window."""

import json
import shutil
import zipfile
from pathlib import Path

import cv2 as cv
import numpy as np

from pipeline.genvideo.evaluate import save_video

P = Path("work/genvideo/round2")
D = P / "cosmos"
D.mkdir(exist_ok=True)
frames = [cv.imread(str(P / "raw" / f"{i:03d}-view.png")) for i in range(81)]
zs = [np.load(P / "raw" / f"{i:03d}-z.npy").astype("f") for i in range(81)]
near, far = np.percentile(
    np.concatenate([z[::8, ::8][z[::8, ::8] > 0] for z in zs]), [1, 99]
)


def gray(a):
    return cv.cvtColor(a, cv.COLOR_GRAY2BGR)


save_video(D / "input.mp4", frames)
save_video(D / "opaque.mp4", [gray((z > 0).astype("B") * 255) for z in zs])
save_video(
    D / "disparity.mp4",
    [
        gray(
            np.where(
                z > 0,
                ((1 / z.clip(0.01) - 1 / far) / (1 / near - 1 / far)).clip(0, 1) * 255,
                0,
            ).astype("B")
        )
        for z in zs
    ],
)
save_video(
    D / "edges.mp4",
    [gray(cv.Canny(cv.cvtColor(a, cv.COLOR_BGR2GRAY), 45, 110)) for a in frames],
)
shutil.copyfile(P / "inputs/000-81/ref0.png", D / "front.png")
base = {
    "video_path": "input.mp4",
    "image_context_path": "front.png",
    "max_frames": 81,
    "prompt": "Match the photographed light and surfaces of this road, retaining the source layout.",
    "guidance": 3,
    "num_steps": 35,
    "seed": 42,
    "resolution": "720",
    "keep_input_resolution": True,
}
for key, conf in [
    ("edge", {"edge": {"control_path": "edges.mp4", "control_weight": 1}}),
    (
        "multi",
        {
            "edge": {"control_path": "edges.mp4", "control_weight": 0.45},
            "depth": {
                "control_path": "disparity.mp4",
                "mask_path": "opaque.mp4",
                "control_weight": 0.35,
            },
            "vis": {"control_weight": 0.2, "preset_blur_strength": "low"},
        },
    ),
]:
    (D / f"{key}.json").write_text(json.dumps(dict(base, name=key, **conf), indent=2))
for f in ("run-cosmos.sh", "COSMOS.md"):
    shutil.copyfile(Path(__file__).parent / f, D / f)
with zipfile.ZipFile(P / "cosmos-110-118.zip", "w") as arc:
    for f in sorted(D.iterdir()):
        arc.write(f, "streetnvs-cosmos/" + f.name)
