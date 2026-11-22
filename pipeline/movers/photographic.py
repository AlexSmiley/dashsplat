"""Keep source native RGB detail, put it in the background's exposure space.

The morph-chase photographic renderer already normalises its source exposures.
PLY colours therefore use that same reference. For raw front plates a renderer
can apply the inverse front gain at the actor projection (no double correction).
"""

from functools import lru_cache
import numpy as np
from pipeline.photo.model import Photometry, basis


@lru_cache(None)
def calibration(live):
    return Photometry("live-ground-v2" if live else "stitched-ground-v2")


@lru_cache(192)
def shade_grid(live, epoch, c):
    p = calibration(live)
    x, y = np.meshgrid((np.arange(49) + 0.5) / 49, (np.arange(37) + 0.5) / 37)
    return (
        (basis(np.stack((x, y), -1).reshape(-1, 2)) @ p.shade[epoch, c])
        .reshape(37, 49, 3)
        .astype("f")
    )


def gain(cam, n, t, xy, live):
    p = calibration(live)
    i = p.names.index(cam)
    ns = p.ns[:, i]
    j = max(0, np.searchsorted(ns, n, "right") - 1)
    if live:
        jj = np.flatnonzero((ns <= n) & (p.ready <= t)) if p.ready is not None else []
        j = jj[-1] if len(jj) else -1
        if j < 0:
            return np.ones((len(xy), 3))
        g = p.gain[j, i]
        e = p.epoch[j]
    else:
        _, g, _ = p.params(cam, n)
        e = p.epoch[j]
    uv = xy / [724, 469] * [49, 37] - 0.5
    v = shade_grid(live, int(e), i)
    x = np.clip(uv[:, 0], 0, 47.999)
    y = np.clip(uv[:, 1], 0, 35.999)
    a = x.astype(int)
    b = y.astype(int)
    x = (x - a)[:, None]
    y = (y - b)[:, None]
    s = (1 - y) * ((1 - x) * v[b, a] + x * v[b, a + 1]) + y * (
        (1 - x) * v[b + 1, a] + x * v[b + 1, a + 1]
    )
    return np.exp(np.clip(g + s, -0.65, 0.65))


def native_colour(im, xy, cam, n, t, live):
    h, w = im.shape[:2]
    uv = xy * [w / 724, h / 469] - 0.5
    x = np.clip(uv[:, 0], 0, w - 1.001)
    y = np.clip(uv[:, 1], 0, h - 1.001)
    i = x.astype(int)
    j = y.astype(int)
    x = (x - i)[:, None]
    y = (y - j)[:, None]
    rgb = (1 - y) * ((1 - x) * im[j, i] + x * im[j, i + 1]) + y * (
        (1 - x) * im[j + 1, i] + x * im[j + 1, i + 1]
    )
    return np.clip(rgb * gain(cam, n, t, xy, live), 0, 255)


def metadata(data, decisions, t0, end):
    sz = data[3]
    ds = []
    last = None
    for v in decisions:
        if v["release"] > end:
            break
        key = (v["state"], v["context"], round(v["confidence"], 1))
        if key != last:
            ds.append([round(v["release"], 5), *key])
            last = key
    return dict(
        photometry={
            "space": "pipeline/photo reference RGB",
            "source": "live-ground-v2",
            "canonical_background_gain": [1, 1, 1],
            "raw_background": "inverse front-camera gain/shading at displayed source exposure",
        },
        contact_shadow=dict(
            shape="elliptical-gaussian",
            offset_xyz=[0, 0, 0.015],
            sigma_xy_m=np.round(sz[:2] * [0.33, 0.39], 3).tolist(),
            peak_opacity=0.32,
            blend="multiply",
            ground_test=True,
        ),
        motion_confidence={
            "fields": ["available_at", "state", "context", "evidence_strength"],
            "changes": ds,
        },
    )


def bake_epochs(data, dst):
    from .model import bake, ply, FIELDS
    from .inventory import SIZE
    from tempfile import TemporaryDirectory
    from pathlib import Path

    sp, rr, tr, sz, stat, scan = data
    epochs = stat["class_epochs"]
    uid = sp["id"]
    if len(epochs) == 1 or epochs[-1][0] <= scan["t"][0]:
        return bake(data, "live", dst, True, photographic=True)
    records = [[], []]
    stats = []
    with TemporaryDirectory(dir=dst) as tmp:
        tmp = Path(tmp)
        for k, (start, cls) in enumerate(epochs):
            end = epochs[k + 1][0] if k + 1 < len(epochs) else np.inf
            keep = (scan["t"] >= start) & (scan["t"] < end)
            if not keep.any():
                continue
            try:
                m = bake(
                    (
                        dict(sp, **{"class": cls}),
                        [
                            r
                            for r in rr
                            if r["release"] < end and r["release"] >= start - 0.5
                        ],
                        tr,
                        np.array(SIZE[cls]),
                        stat,
                        {key: v[keep] for key, v in scan.items()},
                    ),
                    "live",
                    tmp,
                    True,
                    photographic=True,
                )
            except ValueError as e:
                if str(e).startswith("no photographs"):
                    continue
                raise
            stats.append(m)
            for j, suf in enumerate(["", ".pred"]):
                a = (
                    np.frombuffer(
                        (tmp / f"{uid}{suf}.ply")
                        .read_bytes()
                        .split(b"end_header" + bytes([10]), 1)[1],
                        "<f4",
                    )
                    .reshape(-1, len(FIELDS))
                    .copy()
                )
                a[:, 18] = np.minimum(a[:, 18], end)
                records[j].append(a[a[:, 18] > a[:, 17]])
    if not stats:
        raise ValueError("no photographs for class epochs")
    for arr, suf in zip(records, ["", ".pred"]):
        ply(dst / f"{uid}{suf}.ply", np.concatenate(arr))
    return dict(stats[-1], first_photo=stats[0]["first_photo"], class_epochs=epochs)
