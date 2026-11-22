"""Join paired windows and measure differences in the generated camera's pixel grid."""

import argparse
import json
import subprocess
from pathlib import Path

import cv2 as cv
import numpy as np


def frames(p):
    r = cv.VideoCapture(str(p))
    out = []
    while True:
        ok, im = r.read()
        if not ok:
            break
        out.append(im)
    r.release()
    assert out, str(p)
    return out


def save_video(p, ims):
    h, w = ims[0].shape[:2]
    proc = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{w}x{h}",
            "-r",
            "10",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-g",
            "5",
            "-keyint_min",
            "5",
            "-sc_threshold",
            "0",
            "-crf",
            "17",
            "-threads",
            "2",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(p),
        ],
        stdin=subprocess.PIPE,
    )
    for im in ims:
        proc.stdin.write(im.tobytes())
    proc.stdin.close()
    assert proc.wait() == 0


def sheet(ims, inds):
    tiles = [cv.resize(ims[i], (312, 180)) for i in inds]
    return np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:])])


def hp(im):
    g = cv.cvtColor(im, cv.COLOR_BGR2GRAY).astype("f")
    return g - cv.GaussianBlur(g, (0, 0), 3.0)


def stats(x):
    return (
        {"n": len(x), "median": float(np.median(x)), "p90": float(np.percentile(x, 90))}
        if len(x)
        else {"n": 0, "median": None, "p90": None}
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("first", type=Path)
    p.add_argument("second", type=Path, nargs="?")
    p.add_argument("--out", type=Path, default=Path("work/genvideo"))
    p.add_argument("--raw", type=Path, default=Path("work/genvideo/raw"))
    p.add_argument(
        "--anchors",
        type=Path,
        help="JSON list of [label,frame,x,y] at output resolution",
    )
    a = p.parse_args()
    root = a.raw
    out = a.out
    out.mkdir(exist_ok=True, parents=True)
    g = frames(a.first)
    assert len(g) in (49, 81)
    overlap = {}
    if a.second:
        b = frames(a.second)
        assert len(g) == len(b) == 81
        disagreements = [
            float(np.abs(g[69 + i].astype("f") - b[i]).mean()) for i in range(12)
        ]
        overlap = {
            "start_s": 111.9,
            "end_s": 113.0,
            "unblended_rgb_mae_255": disagreements,
        }
        for i in range(12):
            g[69 + i] = cv.addWeighted(g[69 + i], (12 - i) / 13, b[i], (i + 1) / 13, 0)
        g += b[12:]
        assert len(g) == 150
    h, w = g[0].shape[:2]
    tr = np.load(root / "track.npz")
    poses = tr["c2w"]
    K = np.diag([w / (tr["K"][0, 2] * 2), h / (tr["K"][1, 2] * 2), 1]) @ tr["K"]
    yy, xx = np.mgrid[:h, :w]
    rays = np.stack([xx + 0.5, yy + 0.5, np.ones_like(xx)], -1) @ np.linalg.inv(K).T

    def image(i, k, gray=False):
        return cv.resize(
            cv.imread(str(root / f"{i:03d}-{k}.png"), 0 if gray else 1),
            (w, h),
            interpolation=cv.INTER_AREA,
        )

    def depth(i):
        return cv.resize(
            np.load(root / f"{i:03d}-z.npy").astype("f"),
            (w, h),
            interpolation=cv.INTER_NEAREST,
        )

    src = [image(i, "view") for i in range(len(g))]
    save_video(out / "generated.mp4", g)
    save_video(out / "side-by-side.mp4", [np.hstack((s, t)) for s, t in zip(src, g)])
    inds = np.linspace(0, len(g) - 1, 8).round().astype(int)
    cv.imwrite(str(out / "generated.jpg"), sheet(g, inds))
    cv.imwrite(str(out / "conditioning.jpg"), sheet(src, inds))
    actual_splat = (
        [image(i, "splat") for i in range(len(g))]
        if (root / "000-splat.png").exists()
        else src
    )
    if actual_splat is not src:
        cv.imwrite(str(out / "splat.jpg"), sheet(actual_splat, inds))
        save_video(
            out / "splat-vs-generated.mp4",
            [np.hstack((s, t)) for s, t in zip(actual_splat, g)],
        )
    temporal = {"generated": [], "conditioning": [], "splat": []}
    affected = []
    valid_change = []
    per_frame = []
    tracks = []
    # Human-picked white lane dash / arrow strokes (on the 512x320 revised-pose render of the input).
    # Snap onto the bright stroke within 4px, back-project once, then follow in ENU.
    anchors = [
        ("left-turn arrow", 0, 155, 243),
        ("right-turn arrow", 0, 266, 241),
        ("exit divider", 4, 213, 233),
        ("left lane dash", 65, 231, 189),
        ("right solid line", 65, 307, 261),
        ("right lane dash", 77, 305, 240),
        ("through-lane dash", 98, 222, 225),
        ("turn-lane parallel line", 98, 339, 239),
        ("later dash", 131, 232, 188),
        ("turn arrow", 131, 390, 239),
    ]
    if a.anchors:
        anchors = json.loads(a.anchors.read_text())
    for desc, i, x, y in anchors:
        if i >= len(g):
            continue
        if not a.anchors:
            x, y = round(x * w / 512), round(y * h / 320)
        a0 = hp(src[i])
        ry, rx = np.unravel_index(np.argmax(a0[y - 4 : y + 5, x - 4 : x + 5]), (9, 9))
        x += int(rx) - 4
        y += int(ry) - 4
        z = depth(i)[y, x]
        if z > 0 and a0[y, x] > 6:
            tracks.append(
                (desc, i, (rays[y, x] * z) @ poses[i, :3, :3].T + poses[i, :3, 3])
            )
    measured = []
    rn = 6
    search = 18
    for i, (s, v) in enumerate(zip(src, g)):
        z = depth(i)
        hole = image(i, "hole", True)
        valid = (hole < 128) & (z > 0)
        diff = (
            np.linalg.norm(
                cv.cvtColor(s.astype("f") / 255, cv.COLOR_BGR2LAB)
                - cv.cvtColor(v.astype("f") / 255, cv.COLOR_BGR2LAB),
                axis=2,
            )
            > 20
        )
        affected.append(float(diff.mean()))
        valid_change.append(float(diff[valid].mean()))
        A, B = hp(s), hp(v)
        for desc, seed, X in tracks:
            if abs(i - seed) > 10:
                continue
            cp = (X - poses[i, :3, 3]) @ poses[i, :3, :3]
            uv = cp @ K.T
            if cp[2] <= 0:
                continue
            x, y = np.rint(uv[:2] / cp[2] - 0.5).astype(int)
            margin = rn + search + 1
            if not (
                margin < x < w - margin
                and 0.53 * h < y < h - margin
                and abs(z[y, x] - cp[2]) < max(0.6, 0.05 * cp[2])
            ):
                continue
            crop = A[y - rn : y + rn + 1, x - rn : x + rn + 1]
            if crop.max() < 8 or crop.std() < 3:
                continue
            scores = cv.matchTemplate(
                B[
                    y - rn - search : y + rn + search + 1,
                    x - rn - search : x + rn + search + 1,
                ],
                crop,
                cv.TM_CCOEFF_NORMED,
            )
            _, score, _, best = cv.minMaxLoc(scores)
            dx, dy = np.array(best) - search
            measured.append(
                {
                    "feature": desc,
                    "frame": i,
                    "score": score,
                    "dx": int(dx),
                    "dy": int(dy),
                    "drift": float(np.hypot(dx, dy)) if score >= 0.60 else None,
                }
            )
        if i:
            C = poses[i]
            P = poses[i - 1]
            q = ((rays * z[..., None]) @ C[:3, :3].T + C[:3, 3] - P[:3, 3]) @ P[:3, :3]
            uv = q @ K.T
            mx, my = (
                (uv[..., :2] / q[..., 2:].clip(0.01) - 0.5)
                .astype("f")
                .transpose(2, 0, 1)
            )
            zp = cv.remap(depth(i - 1), mx, my, cv.INTER_NEAREST)
            prev = cv.remap(
                image(i - 1, "hole", True),
                mx,
                my,
                cv.INTER_NEAREST,
                borderMode=cv.BORDER_CONSTANT,
                borderValue=255,
            )
            keep = (
                valid
                & (prev < 128)
                & (mx >= 1)
                & (my >= 1)
                & (mx < w - 2)
                & (my < h - 2)
                & (q[..., 2] > 0)
                & (abs(zp - q[..., 2]) < np.maximum(0.25, 0.03 * q[..., 2]))
            )
            for label, ims in [
                ("generated", g),
                ("conditioning", src),
                ("splat", actual_splat),
            ]:
                warp = cv.remap(ims[i - 1], mx, my, cv.INTER_LINEAR)
                err = np.abs(ims[i].astype("f") - warp).mean(-1)
                temporal[label].append(float(err[keep].mean()))
        per_frame.append(
            {
                "frame": i,
                "t": float(tr["t"][i]),
                "changed": float(affected[-1]),
                "valid_changed": float(valid_change[-1]),
            }
        )
    dr = [m["drift"] for m in measured if m["drift"] is not None]
    frame_groups = {}
    for m in measured:
        frame_groups.setdefault(m["frame"], []).append(m["drift"])
    within3 = sum(
        all(d is not None and d <= 3 for d in ds) for ds in frame_groups.values()
    )
    summary = {
        "frames": len(g),
        "overlap": overlap,
        "fps": 10,
        "size": [w, h],
        "sheet_indices": inds.tolist(),
        "change_dE76_gt20": {
            "all": float(np.mean(affected)),
            "valid": float(np.mean(valid_change)),
        },
        "reprojected_mae_255": {k: float(np.mean(v)) for k, v in temporal.items()},
        "paint": {
            "anchors": len(tracks),
            "tested": len(measured),
            "frames_tested": len(frame_groups),
            "frames_all_patches_within_3px": within3,
            "matched": stats(dr),
            "within_3px_over_all": sum(d <= 3 for d in dr) / max(1, len(measured)),
            "per_feature": {
                d: stats(
                    [
                        m["drift"]
                        for m in measured
                        if m["feature"] == d and m["drift"] is not None
                    ]
                )
                for d, _, _ in tracks
            },
        },
    }
    (out / "metrics.json").write_text(
        json.dumps(
            {"summary": summary, "tracks": measured, "frames": per_frame}, indent=2
        )
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
