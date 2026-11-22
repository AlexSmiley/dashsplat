"""Protect the driven path + observed road; crop registered stitched splats by cross-view agreement."""

from .v3_input import *
from .popup_render import Road, ExportedChase
from scipy.spatial import cKDTree
from scipy.ndimage import binary_dilation
from .v3_pano import gain
import shutil

CELL = 0.40


class RoadMask:
    def __init__(self, p):
        j = np.load(p)
        self.origin = j["origin"]
        self.ro = j["road"]
        self.clean = j["clean"]
        self.center = j["center"]
        self.shape = self.ro.shape

    def query(self, xy):
        ij = np.floor((xy - self.origin) / CELL).astype(int)
        x, y = ij.T
        inside = (x >= 0) & (x < self.shape[1]) & (y >= 0) & (y < self.shape[0])
        x = x.clip(0, self.shape[1] - 1)
        y = y.clip(0, self.shape[0] - 1)
        return self.ro[y, x] & inside, self.clean[y, x] & inside, self.center[y, x]


def masks():
    r = ExportedChase()
    path = r.raw[80 * 30 : 146 * 30 : 3, :2]
    origin = np.floor((path.min(0) - 70) / CELL) * CELL
    w, h = np.ceil((path.max(0) + 70 - origin) / CELL).astype(int)
    N = w * h
    count = np.zeros(N, "f")
    roads = count.copy()
    rgb = np.zeros((N, 3), "f")
    sq = rgb.copy()
    sets = getsets()
    for s in sets:
        for c, e in s.items():
            cat = cv.imread(e["mask"], -1)
            semantic = sem(c, e)
            allowed = cat == 2
            if c in C3 and semantic.exists():
                ss = cv.imread(str(semantic), 0)
                allowed |= np.isin(ss, [1, 9]) & (cat == 1)
            yy, xx = np.where(allowed[3::6, 3::6])
            x = 3 + 6 * xx
            y = 3 + 6 * yy
            O, D = wrays(c, e, x, y)
            pts, ok = floorpts(e, O, D)
            ij = np.floor((pts[:, :2] - origin) / CELL).astype(int)
            ok &= (
                (ij[:, 0] >= 0)
                & (ij[:, 1] >= 0)
                & (ij[:, 0] < w)
                & (ij[:, 1] < h)
                & (np.linalg.norm(pts[:, :2] - O[:, :2], axis=1) < 32)
            )
            ij = ij[ok]
            key = ij[:, 0] + ij[:, 1] * w
            u, ix = np.unique(key, return_index=True)
            x = x[ok][ix]
            y = y[ok][ix]
            col = cv.imread(e["rgb"])[2 * y, 2 * x, ::-1] * gain(c, e["t"])
            count[u] += 1
            roads[u] += cat[y, x] == 2
            rgb[u] += col
            sq[u] += col * col
    yy, xx = np.mgrid[:h, :w]
    xy = np.c_[xx.ravel() + 0.5, yy.ravel() + 0.5] * CELL + origin
    center = cKDTree(path).query(xy)[0]
    road = (center < 2.35) | ((roads >= 2) & (roads >= count * 0.50))
    road = binary_dilation(road.reshape(h, w), iterations=1)
    std = np.sqrt(
        np.maximum(
            0,
            sq / np.maximum(1, count[:, None])
            - (rgb / np.maximum(1, count[:, None])) ** 2,
        )
    ).mean(1)
    clean = (count >= 3) & (std < 21) & (center < 13.5)
    np.savez_compressed(
        W / "roadmask.npz",
        origin=origin,
        road=road,
        clean=clean.reshape(h, w),
        center=center.reshape(h, w).astype("f"),
    )
    return RoadMask(W / "roadmask.npz")


def build():
    m = masks()
    cam = ExportedChase()
    r = Road(cam)
    path = cam.raw[93 * 30 : 141 * 30 : 5, :2]
    tree = cKDTree(path)
    items = []
    (BUILD / "tiles").mkdir(parents=True, exist_ok=True)
    n0 = n1 = 0
    for e in r.tiles:
        lo = np.array(e["bbox"][:2])
        hi = np.array(e["bbox"][3:5])
        if (
            np.min(
                np.linalg.norm(np.maximum(0, np.maximum(lo - path, path - hi)), axis=1)
            )
            > 85
        ):
            continue
        a = r.tile(e["file"])[:, :24]
        inside = tree.query(a[:, :2])[0] < 82
        n0 += int(inside.sum())
        ro, clean, perp = m.query(a[:, :2])
        sc = np.exp(a[:, 10:12])
        aspect = sc.max(1) / np.maximum(0.01, sc.min(1))
        ok = inside & (
            (perp < 2.85)
            | (
                clean
                & (aspect < 3.4)
                & (sc.max(1) < 0.42)
                & (a[:, 5] > 0.95)
                & (a[:, 9] > -0.5)
            )
        )
        a = np.ascontiguousarray(a[ok], dtype="<f4")
        n1 += len(a)
        if not len(a):
            continue
        hdr = b""
        line = b""
        # ASCII PLY header, followed by the unchanged 24-field vertex layout.
        with (r.root / e["file"]).open("rb") as f:
            while line != b"end_header\n":
                line = f.readline()
                assert line
                hdr += (
                    f"element vertex {len(a)}\n".encode()
                    if line.startswith(b"element vertex ")
                    else line
                )
        (BUILD / e["file"]).write_bytes(hdr + a.tobytes())
        items.append(
            dict(e, count=len(a), bbox=np.r_[a[:, :3].min(0), a[:, :3].max(0)].tolist())
        )
    dump(BUILD / "map.json", dict(t_end=140.0, tiles=items, sky="sky.jpg"))
    shutil.copy2(BASE / "drive.json", BUILD / "drive.json")
    shutil.copy2(BASE / "rig.json", BUILD / "rig.json")
    shutil.copy2(W / "chase.json", BUILD / "chase.json")
    dump(W / "ground-stats.json", dict(before=n0, after=n1, tiles=len(items)))
    print("ground", len(items), n1, "of", n0, flush=True)


if __name__ == "__main__":
    build()
