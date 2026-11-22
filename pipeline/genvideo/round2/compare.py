"""Time-matched three-column movie. Round one has a narrower FOV; pad it rather than stretching it."""

import sys
from pathlib import Path

import cv2 as cv
import numpy as np

from pipeline.genvideo.evaluate import frames, save_video, sheet


def compare(p, winner):
    old = frames(p.parent / "generated.mp4")
    new = frames(p / f"{winner}.mp4")
    out = []
    rows = []
    for i, im in enumerate(new):
        t = 110 + i / 10
        index = round((t - 105) * 10)
        dense = cv.imread(str(p / "raw" / f"{i:03d}-view.png"))
        r1 = cv.copyMakeBorder(
            cv.resize(old[index], (768, 480)),
            0,
            0,
            32,
            32,
            cv.BORDER_CONSTANT,
            value=(15, 15, 15),
        )
        r2 = im.copy()
        row = np.hstack((dense, r1, r2))
        rows.append(row)
        w = 832
        banner = np.full((32, 3 * w, 3), 24, "B")
        for j, tag in enumerate(
            [
                f"{t:.1f} s  |  Dense conditioning",
                "Round 1  |  live-v5, upscaled",
                "Round 2  |  stitched-opt-v3, " + winner,
            ]
        ):
            cv.putText(
                banner, tag, (10 + j * w, 23), 0, 0.6, (245, 245, 245), 1, cv.LINE_AA
            )
        out.append(np.vstack((banner, row)))
    save_video(p / "three-column.mp4", out)
    inds = np.linspace(0, len(out) - 1, 8).round().astype(int)
    cv.imwrite(
        str(p / "three-column.jpg"),
        np.vstack(
            [cv.resize(out[i], (1248, 256), interpolation=cv.INTER_AREA) for i in inds]
        ),
    )
    cv.imwrite(str(p / "winner.jpg"), sheet(new, inds))


if __name__ == "__main__":
    compare(Path("work/genvideo/round2"), sys.argv[1])
