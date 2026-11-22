"""Test simultaneous masks in the other calibrated camera, marginalising range.

The old range guess (especially car vs bus) is a prior, not a requirement.
This uses pixels released at the handover, never a backwards merge of IDs.
"""

import numpy as np
from .online import camera
from .packets import iou


def transfer(a, b):
    c = camera(a["c"])
    dst = camera(b["c"])
    l, t, r, bt = a["bb"]
    if min(r - l, bt - t) < 4:
        return 0.0
    # A pillar can still see the full flank when the repeater sees only a
    # sliver. Its centre ray need not project inside the new image. Sample
    # the silhouette perimeter and compare only what the new shutter sees.
    gx, gy = np.meshgrid(np.linspace(l, r, 5), np.linspace(t, bt, 5))
    rays = c.unproject(np.c_[gx.ravel(), gy.ravel()] * [c.width / 724, c.height / 469])
    rays /= np.maximum(np.linalg.norm(rays, axis=1, keepdims=True), 1e-6)
    w = rays @ a["C"][:3, :3].T
    rg = max(4.0, float(b["range"]))
    dist = rg * np.array([0.42, 0.58, 0.78, 1.0, 1.32, 1.75])
    P = w[None, :, :] * dist[:, None, None] + a["C"][:3, 3]
    local = (P - b["C"][:3, 3]) @ b["C"][:3, :3]
    uv, ok = dst.project(local)
    uv *= [724 / dst.width, 469 / dst.height]
    ext = np.clip(np.c_[uv.min(1), uv.max(1)], [0, 0, 0, 0], [724, 469, 724, 469])
    return max(
        [
            iou(e, b["bb"])
            for e, m, z in zip(ext, ok, local)
            if m.any() and z[:, 2].min() > 0
        ]
        or [0.0]
    )


def cross_cost(rows, rr):
    if rows[-1]["c"] == rr[-1]["c"]:
        return 99.0
    scores = []
    colors = []
    for b in rr[-3:]:
        a = min(rows[-8:], key=lambda a: abs(a["rt"] - b["rt"]))
        gap = abs(a["rt"] - b["rt"])
        if gap > 0.10:
            continue
        colors.append(np.mean(abs(a["color"] - b["color"]) * [0.30, 1.0, 1.0]))
        scores.append(transfer(a, b))
    a = rows[-1]
    b = rr[-1]
    gap = b["rt"] - a["rt"]
    A = a["bb"]
    B = b["bb"]
    ca = camera(a["c"])
    cb = camera(b["c"])
    edgeA = min(A[0], 724 - A[2])
    edgeB = min(B[0], 724 - B[2])
    angle = np.degrees(np.arccos(np.clip(a["dir"] @ b["dir"], -1, 1)))
    col = np.mean(abs(a["color"] - b["color"]) * [0.24, 1, 1])
    # Neighbouring FOVs see opposite ends of an occluded body; at a seam
    # even the centre ray can land outside BOTH calibrated valid masks.
    nearA = [r for r in rows[-18:] if b["rt"] - r["rt"] < 0.68]
    exit_ok = (
        len(nearA) > 1 and edgeA <= min(nearA[0]["bb"][0], 724 - nearA[0]["bb"][2]) + 2
    )
    heights = [
        (X[3] - X[1]) * cm.width / (724 * cm.params[1]) for X, cm in [(A, ca), (B, cb)]
    ]
    seam = 99.0
    if (
        -0.08 <= gap < 0.58
        and edgeA < 9
        and edgeB < 42
        and angle < 16 + min(5, max(0, gap) * 12)
        and exit_ok
        and col < 14
        and 0.30 < heights[0] / max(0.0001, heights[1]) < 3.7
    ):
        seam = (
            0.22
            + 0.027 * angle
            + 0.020 * col
            + 0.32 * max(gap, 0)
            + 0.15 * abs(np.log(heights[0] / heights[1]))
        )
    if not scores or np.median(colors) > 15:
        return seam
    ov = np.median(scores)
    return min(
        seam,
        0.82 - ov + 0.031 * np.median(colors)
        if ov > 0.26 and max(scores) > 0.35
        else 99.0,
    )
