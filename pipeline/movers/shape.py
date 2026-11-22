"""Closed loft, end caps, underside, and tyre sidewalls. Local axes x nose/y left/z up."""

import numpy as np
from scipy.spatial.transform import Rotation


# Quad patches have consistent winding corrected against an outward reference.
def patches(cls, size, cabin=0.0):
    L, W, H = size
    out = []

    def add(a, b, c, d, n, material):
        out.append((np.array([a, b, c, d]), np.array(n), material))

    def ring(x, w, top, shoulder, base=0.15):
        y = w * W * 0.5
        z = top * H
        s = shoulder * H
        b = base * H
        u = min(0.12 * H, (z - s) * 0.38)
        q = min(0.06 * H, (s - b) * 0.24)
        return np.array(
            [
                [x * L / 2, -0.83 * y, b],
                [x * L / 2, -0.98 * y, b + q],
                [x * L / 2, -y, s],
                [x * L / 2, -0.85 * y, z - u],
                [x * L / 2, -0.69 * y, z],
                [x * L / 2, 0.69 * y, z],
                [x * L / 2, 0.85 * y, z - u],
                [x * L / 2, y, s],
                [x * L / 2, 0.98 * y, b + q],
                [x * L / 2, 0.83 * y, b],
            ]
        )

    if cls in ("bus", "truck"):
        profile = [
            (-1, 0.89, 0.89, 0.74),
            (-0.98, 0.98, 0.97, 0.83),
            (-0.90, 1, 1, 0.87),
            (0.82, 1, 1, 0.87),
            (0.96, 0.99, 0.97, 0.81),
            (1, 0.90, 0.86, 0.72),
        ]
    elif cls in ("person", "rider", "bicycle", "motorcycle"):
        # Separate head, torso, arms, thighs and shins; cyclists sit forward
        # above two wheels. This is a rigid articulated-looking stand-in, not gait estimation.
        def limb(start, end, rx, ry, mat="body"):
            a, b = np.array(start) * size, np.array(end) * size
            axis = b - a
            axis /= max(1e-8, np.linalg.norm(axis))
            ref = (
                np.array([0.0, 1.0, 0.0])
                if abs(axis[1]) < 0.8
                else np.array([1.0, 0.0, 0.0])
            )
            u = np.cross(axis, ref)
            u /= np.linalg.norm(u)
            v = np.cross(axis, u)
            rings = []
            for f, frac in [
                (0, 0.18),
                (0.15, 0.82),
                (0.45, 1.0),
                (0.83, 0.8),
                (1.0, 0.16),
            ]:
                aa = np.arange(13) * np.pi / 6
                rings.append(
                    a
                    + (b - a) * f
                    + frac
                    * (np.cos(aa)[:, None] * rx * u + np.sin(aa)[:, None] * ry * v)
                )
            for r0, r1 in zip(rings, rings[1:]):
                for j in range(12):
                    add(
                        r0[j],
                        r0[j + 1],
                        r1[j + 1],
                        r1[j],
                        r0[j] + r0[j + 1] - 2 * (r0.mean(0)),
                        mat,
                    )

        seated = cls != "person"
        lx = 0.12 if seated else 0.0
        hz = 0.46 if seated else 0.52
        limb((0, 0, hz), (lx, 0, 0.84), 0.12, 0.18)
        limb((lx, 0, 0.83), (lx + 0.015, 0, 1.0), 0.10, 0.09, "head")
        for side in (-1, 1):
            limb((lx, side * 0.28, 0.80), (lx + 0.07, side * 0.36, 0.65), 0.055, 0.05)
            limb(
                (lx + 0.07, side * 0.36, 0.65),
                (lx + (0.20 if seated else 0.01), side * 0.38, 0.53),
                0.044,
                0.043,
                "head",
            )
            knee = (0.13 if seated else 0.045 * side, side * 0.17, 0.27)
            limb((0, side * 0.15, hz), knee, 0.082, 0.072)
            limb(
                knee,
                (-0.06 if seated else -0.035 * side, side * 0.18, 0.045),
                0.063,
                0.052,
            )
            limb(
                (-0.05, side * 0.18, 0.043),
                (0.065, side * 0.18, 0.035),
                0.05,
                0.052,
                "tyre",
            )
        if seated:
            for x in (-0.31, 0.31):
                a = np.arange(25) * np.pi / 12
                rad = 0.195 * H
                for i in range(24):

                    def wheel(j, r, y):
                        return [x * L + r * np.cos(a[j]), y, rad + r * np.sin(a[j])]

                    add(
                        wheel(i, rad, -0.034),
                        wheel(i + 1, rad, -0.034),
                        wheel(i + 1, rad, 0.034),
                        wheel(i, rad, 0.034),
                        [np.cos(a[i]), 0, np.sin(a[i])],
                        "tyre",
                    )
        return out
    else:
        profile = [
            (-1, 0.78, 0.54, 0.43),
            (-0.94, 0.96, 0.62, 0.48),
            (-0.82, 1, 0.80, 0.53),
            (-0.61, 0.99, 0.98, 0.53),
            (-0.40, 0.98, 1, 0.54),
            (0.17, 0.98, 0.97, 0.54),
            (0.48, 1, 0.64, 0.51),
            (0.86, 0.98, 0.56, 0.44),
            (1, 0.81, 0.48, 0.39),
        ]
    if cls == "car":
        profile = [
            (x + cabin * max(0.0, 1 - abs(x + 0.14) / 0.86), w, t, s)
            for x, w, t, s in profile
        ]
    rings = [ring(*p, base=0.11 if cls == "bus" else 0.15) for p in profile]
    for a, b in zip(rings, rings[1:]):
        for j in range(10):
            k = (j + 1) % 10
            q = (a[j] + a[k] + b[j] + b[k]) * 0.25
            n = [0, q[1], q[2] - H * 0.48]
            mat = (
                "window"
                if cls == "car" and j in (2, 3, 5, 6) and q[2] > 0.62 * H
                else "body"
            )
            add(a[j], b[j], b[k], a[k], n, mat)
    for e, n in [(rings[0], -1), (rings[-1], 1)]:
        for j in range(4):
            add(e[j], e[9 - j], e[8 - j], e[j + 1], [n, 0, 0], "body")
    rad = min(0.205 * H, 0.48) if cls != "bus" else min(0.19 * H, 0.54)
    for xc in (-0.315 * L, 0.31 * L):
        for sg in [-1, 1]:
            for j in range(24):
                a, b = j * np.pi / 12, (j + 1) * np.pi / 12

                def pt(a, k, off):
                    return [
                        xc + rad * k * np.cos(a),
                        sg * (W * 0.498 + off),
                        rad * (1 + k * np.sin(a)),
                    ]

                for ri, ro, off, mat in [
                    (0.001, 0.54, 0.012, "hub"),
                    (0.54, 0.72, 0.011, "rim"),
                    (0.72, 1.0, 0.004, "tyre"),
                ]:
                    add(
                        pt(a, ri, off),
                        pt(a, ro, off),
                        pt(b, ro, off),
                        pt(b, ri, off),
                        [0, sg, 0],
                        mat,
                    )
                add(
                    pt(a, 1, 0.004),
                    pt(a, 1, -0.19),
                    pt(b, 1, -0.19),
                    pt(b, 1, 0.004),
                    [np.cos(a), 0, np.sin(a)],
                    "tyre",
                )
    return out


def samples(cls, size, pitch, cabin=0.0):
    P = []
    N = []
    S = []
    Q = []
    M = []
    patch = []
    for h, (corners, n, mat) in enumerate(patches(cls, size, cabin)):
        a, b, c, d = corners
        nu = max(
            1, int(np.ceil(max(np.linalg.norm(b - a), np.linalg.norm(c - d)) / pitch))
        )
        nv = max(
            1, int(np.ceil(max(np.linalg.norm(d - a), np.linalg.norm(c - b)) / pitch))
        )
        v, u = np.mgrid[:nv, :nu]
        u = (u.ravel()[:, None] + 0.5) / nu
        v = (v.ravel()[:, None] + 0.5) / nv
        pos = (1 - v) * ((1 - u) * a + u * b) + v * ((1 - u) * d + u * c)
        U = ((1 - v) * (b - a) + v * (c - d)) / nu
        V = ((1 - u) * (d - a) + u * (c - b)) / nv
        norm = np.cross(U, V)
        norm /= np.maximum(1e-12, np.linalg.norm(norm, axis=1)[:, None])
        norm *= np.where(norm @ n >= 0, 1, -1)[:, None]
        tangent = U / np.maximum(1e-12, np.linalg.norm(U, axis=1)[:, None])
        bit = np.cross(norm, tangent)
        rot = Rotation.from_matrix(np.stack([tangent, bit, norm], axis=2)).as_quat()[
            :, [3, 0, 1, 2]
        ]
        scale = np.c_[
            0.67 * np.linalg.norm(U, axis=1),
            0.67 * abs((V * bit).sum(1)),
            np.full(len(U), pitch * 0.16),
        ]
        P.append(pos)
        N.append(norm)
        Q.append(rot)
        S.append(np.maximum(0.003, scale))
        M.extend([mat] * len(U))
        patch.extend([h] * len(U))
    return tuple(np.concatenate(v) for v in [P, N, S, Q]) + (
        np.array(M),
        np.array(patch),
    )


def silhouette(cls, size, cabin=0.0):
    p = patches(cls, size, cabin)
    b = np.vstack([a for a, _, m in p if m == "body" or m == "window"])
    wheels = np.array(
        [[x * size[0], s * 0.49 * size[1], 0] for x in [-0.315, 0.31] for s in [-1, 1]]
    )
    return np.unique(np.round(np.vstack([b, wheels]), 5), axis=0)
