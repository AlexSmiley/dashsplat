"""Metrics and before | after | |difference| sheets for every refined patch. Run with envs/genfill/bin/python.

Sheet layout: top row = whole 30.7 m patch at 4 cm/px, bottom row = 15.4 m detail at the native 2 cm/px
(window with the most paint). Difference = max-channel |after - before| x4, inferno colour map.
"""

import io
import sys
from pathlib import Path

import cv2 as cv
import numpy as np

from pipeline.genfill.ground.common import PATCHES, WORK, load_patch, metrics, paint, save_json

DETAIL = 768


def jpeg_under(img_bgr, path, limit=500_000):
    for q in (90, 85, 80, 75, 70, 65, 60, 55, 50):
        ok, buf = cv.imencode(".jpg", img_bgr, [cv.IMWRITE_JPEG_QUALITY, q])
        if len(buf) < limit:
            break
    Path(path).write_bytes(buf.tobytes())
    return q, len(buf)


def label(img, text):
    cv.putText(img, text, (12, 34), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(img, text, (12, 34), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv.LINE_AA)
    return img


def diffmap(a, b):
    d = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(-1)
    return cv.applyColorMap(np.clip(d * 4, 0, 255).astype(np.uint8), cv.COLORMAP_INFERNO)


def detail_window(p0):
    s = cv.boxFilter(p0.astype(np.float32), -1, (DETAIL, DETAIL), normalize=True, borderType=cv.BORDER_CONSTANT)
    h, w = p0.shape
    c = DETAIL // 2
    s = s[c : h - c + 1, c : w - c + 1]
    y, x = np.unravel_index(np.argmax(s), s.shape)
    return int(y), int(x)


def sheet(before, after, name, setting, yx, out):
    b, a = before[:, :, ::-1], after[:, :, ::-1]
    half = lambda x: cv.resize(x, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)
    y, x = yx
    crop = lambda z: z[y : y + DETAIL, x : x + DETAIL]
    top = np.hstack([label(half(b).copy(), f"{name}: before (4 cm/px)"), label(half(a).copy(), setting), label(half(diffmap(a, b)), "|diff| x4")])
    bot = np.hstack([label(crop(b).copy(), "detail 2 cm/px"), label(crop(a).copy(), setting), label(crop(diffmap(a, b)).copy(), "|diff| x4")])
    return jpeg_under(np.vstack([top, bot]), out)


ZOOM = ["difix-noref-t199", "difix-noref-t99", "difix-ref-t199", "difix-ref-t99", "esrgan-x2-native-1cm"]


ZOOM_CENTRE = {"arrows": (620, 680), "bike": (520, 765), "only": (720, 760)}  # row, col of the main symbol


def zoom(p, before, yx, out, n=320):
    """Before and each setting on an n x n crop around the patch's main symbol, 2 cm px shown 2x nearest."""
    y, x = ZOOM_CENTRE[p][0] - n // 2, ZOOM_CENTRE[p][1] - n // 2
    up = lambda z: cv.resize(z, (2 * n, 2 * n), interpolation=cv.INTER_NEAREST)
    panels = [label(up(before[y : y + n, x : x + n, ::-1]).copy(), "before 2 cm (2x)")]
    for k in ZOOM:
        a = cv.imread(str(WORK / "out" / p / f"{k}.png"))
        a = a[2 * y : 2 * y + 2 * n, 2 * x : 2 * x + 2 * n] if k.endswith("-1cm") else up(a[y : y + n, x : x + n])
        panels.append(label(np.ascontiguousarray(a), k.replace("-1cm", " (1 cm out)")))
    return jpeg_under(np.vstack([np.hstack(panels[:3]), np.hstack(panels[3:])]), out)


def main(report):
    report = Path(report)
    report.mkdir(parents=True, exist_ok=True)
    allm = {}
    for p in PATCHES:
        before, mask = load_patch(p)
        p0, _ = paint(before, mask)
        yx = detail_window(p0)
        for f in sorted((WORK / "out" / p).glob("*.png")):
            if f.stem.endswith("-1cm"):
                continue
            after = cv.imread(str(f))[:, :, ::-1]
            m = metrics(before, after, mask)
            q, n = sheet(before, after, p, f.stem, yx, report / f"{p}-{f.stem}.jpg")
            m["sheet"] = f"report/{p}-{f.stem}.jpg"
            allm.setdefault(p, {})[f.stem] = m
            print(p, f.stem, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()}, f"jpeg q{q} {n}B", flush=True)
        # controls: what trivial, layout-preserving operations score on the same metrics
        ok, buf = cv.imencode(".jpg", before[:, :, ::-1], [cv.IMWRITE_JPEG_QUALITY, 85])
        controls = {"control-jpeg85": cv.imdecode(buf, cv.IMREAD_COLOR)[:, :, ::-1], "control-blur-s0.8": cv.GaussianBlur(before, (0, 0), 0.8)}
        for k, after in controls.items():
            allm[p][k] = metrics(before, after, mask)
            print(p, k, {a: (round(v, 3) if isinstance(v, float) else v) for a, v in allm[p][k].items()}, flush=True)
        zoom(p, before, yx, report / f"{p}-zoom.jpg")
        allm[p]["_detail_window_rowcol"] = yx
        allm[p]["_paint_px"] = int(p0.sum())
        allm[p]["_road_px"] = int(mask.sum())
    save_json(WORK / "metrics.json", allm)
    save_json(report / "metrics.json", allm)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else WORK / "report")
