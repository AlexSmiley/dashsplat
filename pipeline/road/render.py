"""CPU orthographic / chase proof renders directly from exported PLY properties."""

import ctypes as ct
import json
from pathlib import Path
import cv2 as cv
import numpy as np
from .scene import RS, W, reader

DIMS = (750, 440)
TIMES = [
    (5.0, "first-movement"),
    (87.9, "entrance"),
    (93.7, "turn-arrow"),
    (101.2, "exit-arrow"),
    (112.5, "first-road"),
    (200.2, "light-stop"),
    (318.6, "only"),
    (322.2, "arterial-turn"),
]
REPORT = RS / "pipeline/road/report"
DEBUG = RS / "maps/ground-debug"
BG = np.array([20.0, 24.0, 29.0], "f")


class Scene:
    def __init__(self, name, version=1):
        self.version = version
        self.work = RS / "maps/ground-v2" if version == 2 else W
        self.name = name
        self.root = RS / "scene" / f"road-{name}-v{version}"
        self.index = json.loads((self.root / "map.json").read_text())
        self.drive = json.loads((self.root / "drive.json").read_text())
        self.lib = ct.CDLL(str(W / "discs.dylib"))
        self.lib.draw.argtypes = [
            ct.c_void_p,
            ct.c_int,
            ct.c_int,
            ct.c_void_p,
            ct.c_int,
        ]

    def pose(self, t):
        d = self.drive
        return np.array([np.interp(t, d["t"], d[k]) for k in ("x", "y", "z", "yaw")])

    def local(self, t, position, radius, raw=False):
        z = []
        colors = []
        px, py = position[:2]
        for tile in self.index["tiles"]:
            x, y, _, X, Y, _ = tile["bbox"]
            if max(x - px, 0, px - X) ** 2 + max(y - py, 0, py - Y) ** 2 > radius**2:
                continue
            a = reader(self.root / tile["file"])
            ii = np.flatnonzero(
                (a[:, 17] <= t)
                & (a[:, 18] > t)
                & (abs(a[:, 0] - px) < radius)
                & (abs(a[:, 1] - py) < radius)
            )
            if len(ii):
                z.append(a[ii])
                if raw:
                    colors.append(
                        np.memmap(
                            self.work / self.name / (tile["id"] + ".rgb"),
                            "uint8",
                            "r",
                            shape=(len(a), 3),
                        )[ii]
                    )
        return (np.vstack(z) if z else np.zeros((0, 24), "f")), np.vstack(
            colors
        ) if colors else None

    def draw(
        self,
        t,
        view="chase",
        centre=None,
        span=25,
        raw=False,
        mode="rgb",
        size=DIMS,
        fixed_yaw=None,
    ):
        x, y, z, yaw = self.pose(t)
        yaw = yaw if fixed_yaw is None else fixed_yaw
        p = np.array([x, y, z]) if centre is None else np.asarray(centre)
        head = np.array([np.cos(yaw), np.sin(yaw), 0])
        up = np.array([0, 0, 1.0])
        right = np.cross(head, up)
        w, h = size
        f = 0.5 * w / np.tan(np.pi / 6)
        persp = view == "chase"
        cam = p - 6 * head + 3 * up if persp else p + 12 * up
        fw = head * np.cos(0.215) - up * np.sin(0.215) if persp else -up
        down = np.cross(fw, right)
        a, rawc = self.local(t, p, 119 if persp else span + 3, raw)
        canvas = np.broadcast_to(BG, (h, w, 3)).copy()
        if not len(a):
            return canvas.astype("uint8"), dict(covered=0, visible=0, coarse=0)
        xyz = a[:, :3] - cam
        dep = xyz @ fw
        r = xyz @ right
        v = xyz @ down
        foc = f / np.maximum(0.05, dep) if persp else np.full(len(a), w / span)
        uu = w / 2 + r * foc
        vv = h / 2 + v * foc
        sel = (
            (dep > 0.15)
            & (dep < 126)
            & (uu >= -60)
            & (uu < w + 60)
            & (vv >= -60)
            & (vv < h + 60)
        )
        a = a[sel]
        r = r[sel]
        v = v[sel]
        dep = dep[sel]
        foc = foc[sel]
        uu = uu[sel]
        vv = vv[sel]
        nx = a[:, 3:6]
        sa = np.exp(2 * a[:, 10])
        sz = np.exp(2 * a[:, 12])
        jx = (
            foc[:, None] * (right - (r / dep)[:, None] * fw)
            if persp
            else right[None] * foc[:, None]
        )
        jy = (
            foc[:, None] * (down - (v / dep)[:, None] * fw)
            if persp
            else down[None] * foc[:, None]
        )
        ax = (jx * nx).sum(1)
        ay = (jy * nx).sum(1)
        A = sa * (jx * jx).sum(1) + (sz - sa) * ax * ax
        B = sa * (jx * jy).sum(1) + (sz - sa) * ax * ay
        C = sa * (jy * jy).sum(1) + (sz - sa) * ay * ay
        # Each PLY stores an arbitrary tangent-frame rotation. The symmetric
        # v1 shortcut accounts for scale_0; add the second axis explicitly.
        qw, qx, qy, qz = a[:, 13:17].T
        by = np.column_stack(
            (
                2 * (qx * qy - qw * qz),
                1 - 2 * (qx * qx + qz * qz),
                2 * (qy * qz + qw * qx),
            )
        )
        ds = np.exp(2 * a[:, 11]) - sa
        bx = (jx * by).sum(1)
        byy = (jy * by).sum(1)
        A += ds * bx * bx
        B += ds * bx * byy
        C += ds * byy * byy
        det = A * C - B * B
        A += 0.20
        C += 0.20
        det2 = A * C - B * B
        aa = (
            1
            / (1 + np.exp(-a[:, 9]))
            * np.sqrt(np.maximum(0, det) / np.maximum(1e-8, det2))
        )
        if self.name == "live":
            aa *= np.clip((t - a[:, 17]) / 0.25, 0, 1) * np.clip(
                (a[:, 18] - t) / 0.25, 0, 1
            )
        rgb = (
            np.clip((0.5 + a[:, 6:9] * 0.28209479) * 255, 0, 255)
            if rawc is None
            else rawc[sel]
        )
        age = t - np.minimum(t, a[:, 19])
        age = np.where(t >= a[:, 20], np.minimum(age, t - np.minimum(t, a[:, 21])), age)
        age = np.where(t >= a[:, 22], np.minimum(age, t - np.minimum(t, a[:, 23])), age)
        lev = np.clip(np.rint(np.log2(0.181818 / np.exp(a[:, 10]))).astype(int), 0, 2)
        if mode == "clock":
            stops = np.array([0, 0.04, 0.25, 1, 5, 25])
            palette = np.array(
                [
                    [97, 232, 167],
                    [227, 183, 85],
                    [177, 111, 125],
                    [102, 95, 158],
                    [80, 77, 132],
                    [58, 65, 103],
                ]
            )
            rgb = palette[
                np.clip(np.searchsorted(stops, age, "right") - 1, 0, len(palette) - 1)
            ]
        if mode == "level":
            rgb = np.array([[223, 137, 53], [92, 183, 176], [147, 168, 236]])[lev]
        if mode == "age":
            rgb = np.where(age[:, None] < 0.025, rgb, rgb * 0.36)
        order = np.argsort(-dep, kind="stable")
        disc = np.column_stack(
            (
                uu,
                vv,
                C / det2,
                -B / det2,
                A / det2,
                3 * np.sqrt(A),
                3 * np.sqrt(C),
                aa,
                rgb,
                dep,
            )
        )[order].astype("f")
        self.lib.draw(canvas.ctypes.data, w, h, disc.ctypes.data, len(disc))
        rgb = np.clip(canvas, 0, 255).astype("uint8")
        return rgb, dict(
            covered=float((abs(canvas - BG).max(2) > 4).mean()),
            visible=int((age < 0.03).sum()),
            coarse=int((lev == 0).sum()),
            levels=np.bincount(lev, minlength=3).tolist(),
        )


def label(rgb, name):
    row = np.full((28, rgb.shape[1], 3), 25, "uint8")
    cv.putText(row, name, (9, 19), 0, 0.49, (224, 224, 224), 1, cv.LINE_AA)
    return np.vstack((row, rgb))


def save(p, im):
    cv.imwrite(str(p), im[:, :, ::-1], [cv.IMWRITE_JPEG_QUALITY, 87])
    assert p.stat().st_size < 500000


def proof():
    REPORT.mkdir(exist_ok=True, parents=True)
    DEBUG.mkdir(exist_ok=True, parents=True)
    sc = [Scene(n) for n in ("live", "stitched")]
    metrics = {}
    for t, n in TIMES:
        views = []
        metrics[n] = {}
        for view in ["top", "chase"]:
            row = []
            for v in sc:
                pic, met = v.draw(t, view)
                row.append(label(pic, f"{v.name} | {view} | {t:.2f} s"))
                metrics[n][v.name + "-" + view] = met
            views.append(np.hstack(row))
        save(REPORT / ("ground-" + n + ".jpg"), np.vstack(views))
    # Follow the same physical parent as it changes resolution. Pick a real
    # supersession along the exit, then hold the nadir camera fixed on that patch.
    v = sc[0]
    parent = None
    score = -1
    from scipy.spatial import cKDTree

    for t0 in np.arange(88, 103, 0.8):
        p = v.pose(t0)
        a, _ = v.local(t0, p, 10)
        sel = (
            (a[:, 18] < t0 + 1.7)
            & (a[:, 18] > t0 + 0.27)
            & (a[:, 10] > -1.86)
            & (a[:, 17] < t0 - 0.26)
            & (a[:, 17] > t0 - 3)
        )
        ix = np.flatnonzero(sel)
        if not len(ix):
            continue
        final, _ = sc[0].local(min(t0 + 3, 105), p, 13)
        tree = cKDTree(final[:, :2])
        for q in ix[:: max(1, len(ix) // 35)]:
            ids = tree.query_ball_point(a[q, :2], 0.75)
            if len(ids) < 140:
                continue
            z = final[ids]
            contrast = np.percentile(z[:, 7], [90, 10])
            fine = np.mean(z[:, 10] < -2.9)
            value = (contrast[0] - contrast[1]) * fine
            if value > score:
                parent = a[q]
                score = value
    if parent is not None:
        xy = parent[:3]
        ts = [parent[17] + 0.26, parent[18] - 0.23, parent[18] + 0.55]
        pan = []
        for mode in ["rgb", "level"]:
            row = []
            for t in ts:
                pic, met = v.draw(
                    float(t),
                    "top",
                    centre=xy,
                    span=1.9,
                    mode=mode,
                    size=(380, 300),
                    fixed_yaw=v.pose(parent[18])[3],
                )
                row.append(label(pic, f"{mode}   {t:.2f} s"))
            pan.append(np.hstack(row))
        save(REPORT / "ground-refinement.jpg", np.vstack(pan))
        metrics["refinement"] = dict(centre=xy.tolist(), times=[float(x) for x in ts])
    # Same geometry, weights and exact sample identities, only change radiometry.
    pics = []
    for t in [93.7, 297.67, 318.62]:
        cells = []
        for raw in [True, False]:
            im, _ = sc[1].draw(t, "top", span=16, raw=raw, size=(590, 305))
            cells.append(label(im, ("raw" if raw else "normalised") + f" | {t:.2f} s"))
        pics.append(np.hstack(cells))
    save(REPORT / "ground-photo-before-after.jpg", np.vstack(pics))
    (DEBUG / "renders.json").write_text(json.dumps(metrics, indent=2))
    print("rendered", len(TIMES), "poses", flush=True)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", choices=["live", "stitched"])
    p.add_argument("--version", type=int, choices=[1, 2], default=1)
    p.add_argument("--time", type=float)
    p.add_argument("--view", choices=["top", "chase"], default="chase")
    p.add_argument("--mode", choices=["rgb", "level", "age", "clock"], default="rgb")
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    if a.scene:
        if a.time is None or a.output is None:
            p.error("--scene requires --time and --output")
        pic, info = Scene(a.scene, a.version).draw(a.time, a.view, mode=a.mode)
        a.output.parent.mkdir(parents=True, exist_ok=True)
        cv.imwrite(str(a.output), pic[:, :, ::-1])
        print(info)
    else:
        proof()
