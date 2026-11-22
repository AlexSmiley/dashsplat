"""Orthographic road radiance raster. Public API: MosaicGrid, accumulate_window.
Image columns/rows are world metric basis vectors; crop-only montages never move
one exposure relative to another. Best-view colours and observation timestamps
are retained independently of the optional road consensus display mask.
"""

from .pairs import *
from .geometry import ground_to_pixels
from concurrent.futures import ThreadPoolExecutor


class MosaicGrid:
    def __init__(self, origin, basis, size, cell=0.02):
        self.origin = np.asarray(origin)
        self.basis = np.asarray(basis)
        self.cell = cell
        w, h = size
        self.size = size
        self.color = np.zeros((h, w, 3), "uint8")
        self.weight = np.zeros((h, w), "f")
        self.first_seen = np.full((h, w), np.nan, "f")
        self.last_seen = self.first_seen.copy()
        self.reference = np.zeros((h, w), "uint8")
        self.ref_time = np.full((h, w), -1e4, "f")
        self.ref_camera = np.full((h, w), 255, "uint8")
        self.support = np.zeros((h, w), "uint8")

    def world_to_grid(self, xy):
        return (np.asarray(xy) - self.origin) @ self.basis / self.cell

    def grid_to_world(self, xy):
        return self.origin + (np.asarray(xy) * self.cell) @ self.basis.T

    def accumulate(self, box, color, weight, time, camera):
        """Offer a raw registered view. Replace a cell only when a better view sees it."""
        x, y, w, h = box
        sl = np.s_[y : y + h, x : x + w]
        ok = weight > 0
        gray = cv.cvtColor(color, cv.COLOR_BGR2GRAY)
        rt = self.ref_time[sl]
        rc = self.ref_camera[sl]
        ref = self.reference[sl]
        sep = (abs(time - rt) > 0.45) | (rc != camera)
        agree = ok & sep & (rt > -9999) & (np.abs(ref.astype("i2") - gray) < 22)
        s = self.support[sl]
        s[agree] = np.minimum(s[agree].astype(int) + 1, 255)
        fresh = ok & ((rt < -9999) | ((time - rt) > 1.2))
        rt[fresh] = time[fresh]
        rc[fresh] = camera
        ref[fresh] = gray[fresh]
        best = ok & (weight > self.weight[sl])
        self.color[sl][best] = color[best]
        self.weight[sl][best] = weight[best]
        f = self.first_seen[sl]
        l = self.last_seen[sl]
        f[ok] = np.fmin(f[ok], time[ok])
        l[ok] = np.fmax(l[ok], time[ok])
        self.support[sl] = s

    def image(self, consensus=True):
        out = self.color.copy()
        ok = self.weight > 0
        if consensus:
            ok &= (
                cv.morphologyEx(
                    (self.support > 0).astype("uint8"),
                    cv.MORPH_CLOSE,
                    np.ones((5, 5), "uint8"),
                )
                > 0
            )
        out[~ok] = [24, 24, 24]
        return out


def raster(grid, cam, tr, tf, img):
    # Find supported footprint before constructing the fine remap. The projection
    # LUT is sampled every 8 cm and bilinearly interpolated to 2 cm; recompute
    # native body mask and range at the output. This is a rendering approximation,
    # independently of the exact row-time, full-resolution measurement API.
    mid = tr.T_world_ego(cam.row_time(tf, cam.height / 2))
    loc = grid.world_to_grid(mid[:2, 3])
    rad = int(np.ceil(12 / grid.cell)) + 1
    lo = np.maximum(np.floor(loc - rad).astype(int), 0)
    hi = np.minimum(np.ceil(loc + rad).astype(int) + 1, grid.size)
    cx = np.arange(lo[0], hi[0], 30)
    cy = np.arange(lo[1], hi[1], 30)
    if not len(cx) or not len(cy):
        return None
    xx, yy = np.meshgrid(cx, cy)
    uv, ok, z = ground_to_pixels(
        cam, tr, tf, grid.grid_to_world(np.stack((xx, yy), -1))
    )
    iy, ix = np.nonzero(ok)
    if not len(ix):
        return None
    lo = np.maximum(lo, [cx[ix.min()] - 32, cy[iy.min()] - 32])
    hi = np.minimum(hi, [cx[ix.max()] + 33, cy[iy.max()] + 33])
    x, y = lo
    w, h = hi - lo
    inc = 4
    u = np.arange(0, w + inc, inc)
    v = np.arange(0, h + inc, inc)
    uu, vv = np.meshgrid(u, v)
    G = grid.grid_to_world(np.stack((uu + x, vv + y), -1))
    uv, ok, z = ground_to_pixels(cam, tr, tf, G, 12.3)
    xx, yy = np.meshgrid(np.arange(w, dtype="f") / inc, np.arange(h, dtype="f") / inc)
    mp = cv.remap((uv - 0.5).astype("f"), xx, yy, cv.INTER_LINEAR)
    valid = cv.remap(ok.astype("uint8"), xx, yy, cv.INTER_NEAREST)
    valid &= cv.remap(cam.mask.astype("uint8"), mp, None, cv.INTER_NEAREST)
    raw = cv.remap(img, mp, None, cv.INTER_LINEAR)
    valid &= road_color(raw)
    dist = (xx * inc + x - loc[0]) ** 2 + (yy * inc + y - loc[1]) ** 2
    valid &= dist < (12 / grid.cell) ** 2
    incidence = abs(cam.T_ego_cam[2, 3]) / np.sqrt(
        dist * grid.cell**2 + cam.T_ego_cam[2, 3] ** 2
    )
    gr = cv.cvtColor(raw, cv.COLOR_BGR2GRAY)
    sharp = np.clip(np.mean(abs(cv.Laplacian(gr, cv.CV_32F))) / 2, 0.45, 1.7)
    weight = valid * sharp * incidence / np.maximum(1, dist * grid.cell**2)
    tm = cam.row_time(tf, mp[:, :, 1] + 0.5).astype("f")
    return (int(x), int(y), int(w), int(h)), raw, weight.astype("f"), tm


def trajectory(name):
    if name == "planar":
        return Trajectory(WORK / "missing-planar.npz", timeline=D / "timeline.npz")
    stem = {
        "offline-v1": "ego_v1",
        "causal-v1": "ego_causal_v1",
        "offline": "ego",
        "causal": "ego_causal",
    }[name]
    p = (
        WORK / (stem + ".npz")
        if name in ("offline", "causal")
        else OUT / (stem + ".npz")
    )
    return Trajectory(p if p.exists() else OUT / (stem + ".npz"))


def accumulate_window(start, end, tr, cell=0.02, save=None):
    s, d, t, B, stop = inputs()
    ks = np.flatnonzero((s["t"] >= start) & (s["t"] <= end))
    ts = s["t"][ks]
    p = tr.T_world_ego(ts)[:, :2, 3]
    xy = p[-1] - p[0]
    heading = np.arctan2(xy[1], xy[0])
    basis = rot(heading)
    ab = p @ basis
    lo = ab.min(0) - 12.25
    hi = ab.max(0) + 12.25
    grid = MosaicGrid(lo @ basis.T, basis, np.ceil((hi - lo) / cell).astype(int), cell)
    cams = [load_rig(RS / "calib/rig.json")[n] for n in NAMES]

    def render(item):
        k, i = item
        pic = cv.imread(str(D / "frames" / NAMES[i] / f"{s['ns'][k, i]:06d}.jpg"))
        return i, raster(grid, cams[i], tr, s["tc"][k, i], pic)

    with ThreadPoolExecutor(int(os.environ.get("ROAD_THREADS", 8))) as pool:
        for a in range(0, len(ks), 4):
            for i, z in pool.map(
                render, [(k, i) for k in ks[a : a + 4] for i in range(6)]
            ):
                if z:
                    grid.accumulate(*z, i)
    if save:
        save = Path(save)
        save.mkdir(exist_ok=True, parents=True)
        img = grid.image()
        cv.imwrite(str(save / "best.jpg"), img, [cv.IMWRITE_JPEG_QUALITY, 91])
        from .display import pavement

        pavement_image, road_mask = pavement(
            img, tr, dict(origin=grid.origin, basis=grid.basis, cell=cell), ts
        )
        cv.imwrite(
            str(save / "pavement.jpg"), pavement_image, [cv.IMWRITE_JPEG_QUALITY, 92]
        )
        cv.imwrite(str(save / "road-mask.png"), road_mask * 255)
        del pavement_image, road_mask
        cv.imwrite(
            str(save / "preview.jpg"),
            cv.resize(
                img,
                None,
                fx=min(1, 1350 / img.shape[1]),
                fy=min(1, 1350 / img.shape[1]),
                interpolation=cv.INTER_AREA,
            ),
        )
        np.savez_compressed(
            save / "grid.npz",
            origin=grid.origin,
            basis=basis,
            cell=cell,
            first_seen=grid.first_seen,
            last_seen=grid.last_seen,
            support=grid.support,
        )
        print(
            save, grid.size, "observed", np.isfinite(grid.first_seen).sum(), flush=True
        )
    return grid


if __name__ == "__main__":
    import sys

    for name in sys.argv[1:] or [
        "planar",
        "offline-v1",
        "causal-v1",
        "offline",
        "causal",
    ]:
        for seg, a, b in SEG:
            accumulate_window(
                a, b, trajectory(name), save=PRODUCT / "mosaics" / name / seg
            )
