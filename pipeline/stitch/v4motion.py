"""Motion evidence from the actual 110–125 s Chrome screencasts. Exclude HUD, car, route ribbon, and flow-inconsistent pixels."""

import cv2 as cv, numpy as np, json, sys
from pathlib import Path

cv.setNumThreads(2)


def run(src, dst):
    cap = cv.VideoCapture(str(src))
    dt = cap.get(cv.CAP_PROP_FPS)
    prev = None
    rows = []
    proof = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % 3 != 0:
            i += 1
            continue
        frame = cv.resize(frame, (720, 450), interpolation=cv.INTER_AREA)
        g = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        t = i / dt
        if i % 30 == 0:
            proof.append(frame)
        if prev is not None:
            opt = dict(
                pyr_scale=0.5,
                levels=4,
                winsize=17,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )
            flow = cv.calcOpticalFlowFarneback(g, prev, None, **opt)
            back = cv.calcOpticalFlowFarneback(prev, g, None, **opt)
            Y, X = np.mgrid[:450, :720].astype("f")
            xx = X + flow[:, :, 0]
            yy = Y + flow[:, :, 1]
            warp = cv.remap(prev, xx, yy, cv.INTER_LINEAR)
            fb = np.linalg.norm(flow + cv.remap(back, xx, yy, cv.INTER_LINEAR), axis=2)
            valid = (fb < 0.65) & (xx > 3) & (xx < 716) & (yy > 2) & (yy < 446)
            rgn = {
                "verge": ((X < 302) | (X > 422)) & (Y > 148) & (Y < 315),
                "distant": (X > 142) & (X < 620) & (Y > 108) & (Y < 154),
                "road": (X > 258)
                & (X < 463)
                & (Y > 185)
                & (Y < 359)
                & ((X < 320) | (X > 386)),
            }
            row = dict(t=float(t))
            for name, m in rgn.items():
                err = np.abs(warp.astype("f") - g)[valid & m]
                row[name] = (
                    dict(
                        mean=float(err.mean()),
                        over20=float((err > 20).mean()),
                        n=len(err),
                    )
                    if len(err)
                    else None
                )
            rows.append(row)
        prev = g
        i += 1
    cap.release()
    out = dict(
        frames=i,
        decoded_fps=dt,
        rows=rows,
        regions={
            k: dict(
                mean=float(np.mean([r[k]["mean"] for r in rows])),
                worst_mean=float(max(r[k]["mean"] for r in rows)),
                p95_over20=float(np.percentile([r[k]["over20"] for r in rows], 95)),
            )
            for k in ["road", "verge", "distant"]
        },
    )
    dst.mkdir(exist_ok=True, parents=True)
    (dst / (src.stem + "-motion.json")).write_text(json.dumps(out))
    f = [cv.resize(im, (360, 225), interpolation=cv.INTER_AREA) for im in proof[:15]]
    cv.imwrite(
        str(dst / (src.stem + "-motion-sheet.jpg")),
        np.vstack([np.hstack(f[j : j + 5]) for j in [0, 5, 10]]),
    )
    print(src, out["regions"], flush=True)


if __name__ == "__main__":
    run(Path(sys.argv[1]), Path(sys.argv[2]))
