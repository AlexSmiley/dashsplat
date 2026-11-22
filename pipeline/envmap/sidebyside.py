"""One small side-by-side clip from viewer recordings: for each segment, the published scene (left) and the
environment variant (right), each trimmed to start where its picture first moves (play pressed or the
drag begun) and cut to the shorter length, scaled to 720x450, labelled, then the segments in a row.
H.264, raised CRF until the file is under 4.9 MB.

    python -m pipeline.envmap.sidebyside out.mp4 'label' old.webm new.webm ['label' old new ...]
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import cv2 as cv
import numpy as np


def motion_start(path, thr=1.5):
    cap = cv.VideoCapture(str(path))
    fps = cap.get(cv.CAP_PROP_FPS) or 30
    ok, f0 = cap.read()
    g0 = cv.cvtColor(cv.resize(f0, (360, 225)), cv.COLOR_BGR2GRAY).astype("f")
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            return 0.0, i / fps
        i += 1
        g = cv.cvtColor(cv.resize(f, (360, 225)), cv.COLOR_BGR2GRAY).astype("f")
        if np.abs(g - g0).mean() > thr:
            start = max(0, i - 2) / fps
            n = i
            while cap.read()[0]:
                n += 1
            return start, n / fps


def label_png(text, path):
    im = np.zeros((34, 720, 4), "B")
    im[..., 3] = 150
    cv.putText(im, text, (12, 23), cv.FONT_HERSHEY_SIMPLEX, 0.62, (235, 235, 235, 255), 1, cv.LINE_AA)
    cv.imwrite(str(path), im)


def main(out, segs):
    tmp = Path(tempfile.mkdtemp())
    parts = []
    for k, (title, old, new) in enumerate(segs):
        (s0, e0), (s1, e1) = motion_start(old), motion_start(new)
        dur = min(e0 - s0, e1 - s1)
        label_png(f"published  |  {title}", tmp / f"l{k}a.png")
        label_png(f"environment map (env1)  |  {title}", tmp / f"l{k}b.png")
        p = tmp / f"seg{k}.mp4"
        vf = ("[0:v]scale=720:450,fps=30,setsar=1[a];[1:v]scale=720:450,fps=30,setsar=1[b];"
              "[a][2:v]overlay=0:0[a2];[b][3:v]overlay=0:0[b2];[a2][b2]hstack=inputs=2[v]")
        subprocess.run(["nice", "-n", "15", "ffmpeg", "-v", "error", "-y", "-ss", f"{s0:.3f}", "-t", f"{dur:.3f}", "-i", str(old),
                        "-ss", f"{s1:.3f}", "-t", f"{dur:.3f}", "-i", str(new), "-i", str(tmp / f"l{k}a.png"),
                        "-i", str(tmp / f"l{k}b.png"), "-filter_complex", vf, "-map", "[v]", "-c:v", "libx264",
                        "-crf", "16", "-pix_fmt", "yuv420p", "-an", str(p)], check=True)
        parts.append(p)
        print(title, "old starts", round(s0, 2), "new starts", round(s1, 2), "seconds", round(dur, 2), flush=True)
    lst = tmp / "list.txt"
    lst.write_text("".join(f"file '{p}'\n" for p in parts))
    for crf in (24, 28, 32, 36):
        subprocess.run(["nice", "-n", "15", "ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
                        "-c:v", "libx264", "-crf", str(crf), "-preset", "slow", "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart", "-an", str(out)], check=True)
        if Path(out).stat().st_size < 4.9e6:
            break
    print("wrote", out, Path(out).stat().st_size, "crf", crf, flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], [tuple(a[i:i + 3]) for i in range(1, len(a), 3)])
