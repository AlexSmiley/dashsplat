"""Conservative road-colour/connected-pavement support and fixed metric crop panels."""

from .mosaic import *
from scipy.ndimage import distance_transform_edt


def pavement(image, tr, grid, ts, recover_paint=False):
    # Fine centreline samples seed connected pavement; distant islands that happen
    # to be gray (facades, passing vehicles) do not grow across grass or a curb.
    H, W = image.shape[:2]
    step = 5
    thumb = cv.resize(
        image,
        ((W + step - 1) // step, (H + step - 1) // step),
        interpolation=cv.INTER_AREA,
    )
    sz = (thumb.shape[1], thumb.shape[0])
    center = np.zeros(thumb.shape[:2], "uint8")
    T = tr.T_world_ego(ts)
    xy = (T[:, :2, 3] - grid["origin"]) @ grid["basis"] / float(grid["cell"])
    xy *= np.array(sz) / [W, H]
    cv.polylines(center, [np.rint(xy).astype("i4")], False, 1, 3)
    dist = distance_transform_edt(1 - center) * float(grid["cell"]) * step
    b, g, r = cv.split(thumb.astype("f"))
    vegetation = (g - r > 8) & (g > b * 1.03)
    saturated = np.maximum.reduce([r, g, b]) - np.minimum.reduce([r, g, b])
    valid = ~(
        (b == 24) & (r == 24) & (g == 24)
    )  # asphalt has near-neutral or cool chroma; also preserve yellow and white paint.
    pavement = (
        (~vegetation)
        & ((saturated < 43) | ((b > r) & (g > r)) | ((r > g * 0.94) & (g > b * 1.21)))
        & valid
        & (dist < 8.0)
    )
    opening = cv.morphologyEx(
        pavement.astype("uint8"), cv.MORPH_OPEN, np.ones((3, 3), "uint8")
    )
    n, ids = cv.connectedComponents(opening)
    labels = np.unique(ids[(dist < 2.15) & (opening != 0)])
    good = np.isin(ids, labels) & (ids != 0)
    good = cv.morphologyEx(
        good.astype("uint8"), cv.MORPH_CLOSE, np.ones((7, 7), "uint8")
    )
    good = cv.dilate(good, np.ones((3, 3), "uint8"))
    if recover_paint:
        # Classifier holes can cut squares out of well-observed white letters.
        # Restore only actual radiance inside small enclosed components, never
        # paint/inpaint missing pixels or change the fixed measurement mask.
        _, holes, stats, _ = cv.connectedComponentsWithStats(1 - good)
        area = stats[:, cv.CC_STAT_AREA] * (float(grid["cell"]) * step) ** 2
        small = area < 1.6
        small[0] = False
        small[np.unique(np.r_[holes[0], holes[-1], holes[:, 0], holes[:, -1]])] = False
        good |= (
            small[holes]
            & valid
            & (dist < 8)
            & (np.maximum.reduce((b, g, r)) > 48)
            & (saturated < 60)
        ).astype("uint8")
    mask = cv.resize(good, (W, H), interpolation=cv.INTER_NEAREST)
    img = image.copy()
    img[mask == 0] = 24
    return img, mask


def segment(name, seg, a, b):
    path = PRODUCT / "mosaics" / name / seg
    img = cv.imread(str(path / "best.jpg"))
    s = inputs()[0]
    ts = s["t"][(s["t"] >= a) & (s["t"] <= b)]
    gr = np.load(path / "grid.npz")
    first, last = gr["first_seen"], gr["last_seen"]
    seen = np.isfinite(first)
    assert (
        np.array_equal(seen, np.isfinite(last))
        and np.all(last[seen] >= first[seen])
        and first[seen].min() > a - 0.3
        and last[seen].max() < b + 0.3
    )
    del first, last, seen
    tr = trajectory(name)
    base, mask = pavement(img, tr, gr, ts)
    del base
    im, display_mask = pavement(img, tr, gr, ts, True)
    cv.imwrite(str(path / "display-mask.png"), display_mask * 255)
    cv.imwrite(str(path / "pavement.jpg"), im, [cv.IMWRITE_JPEG_QUALITY, 92])
    cv.imwrite(str(path / "road-mask.png"), mask * 255)
    return im, {k: gr[k] for k in ["origin", "basis", "cell"]}, tr


# Locations selected on the planar baseline at 1350-pixel preview width, before
# observing final mosaics; use nearest ego packet as the common physical anchor.
CROPS = {
    "park": [
        (201, 458, 5.0, "entrance-arrow"),
        (527, 203, 6.4, "turn-arrow"),
        (915, 294, 6.4, "exit-arrows"),
    ],
    "turn": [
        (200, 248, 6.0, "only-approach"),
        (426, 429, 6.8, "only-turn"),
        (535, 452, 5.8, "turn-pointer"),
    ],
    "straight": [(352, 52, 7.5, "stop-bar"), (1040, 58, 7.0, "lane-crossing")],
}


def make():
    names = ["planar", "offline-v1", "causal-v1", "offline", "causal"]
    out = PRODUCT / "debug"
    out.mkdir(exist_ok=True, parents=True)
    s, d, t, *_ = inputs()
    for seg, a, b in SEG:
        info = [segment(n, seg, a, b) for n in names]
        im0, g0, tr0 = info[0]
        ts = s["t"][(s["t"] >= a) & (s["t"] <= b)]
        T0 = tr0.T_world_ego(ts)
        B0 = g0["basis"]
        panelpx = 300
        for x, y, span, key in CROPS[seg]:
            pos = g0["origin"] + B0 @ (
                [x, y] * np.array(float(g0["cell"]) * im0.shape[1] / 1350)
            )
            k = np.argmin(np.linalg.norm(T0[:, :2, 3] - pos, axis=1))
            tm = ts[k]
            psi = np.arctan2(T0[k, 1, 0], T0[k, 0, 0])
            lever = rot(-psi) @ (pos - T0[k, :2, 3])
            offset = (np.arange(panelpx) - (panelpx - 1) / 2) * span / panelpx
            mesh = np.stack(np.meshgrid(offset, offset), -1)
            panels = []
            for nm, (img, g, tr) in zip(names, info):
                T = tr.T_world_ego(tm)
                az = np.arctan2(T[1, 0], T[0, 0])
                cent = T[:2, 3] + rot(az) @ lever
                B = rot(az - psi) @ B0
                G = cent + mesh @ B.T
                pt = ((G - g["origin"]) @ g["basis"] / g["cell"]).astype("f")
                crop = cv.remap(
                    img,
                    pt,
                    None,
                    cv.INTER_LINEAR,
                    borderMode=cv.BORDER_CONSTANT,
                    borderValue=(24, 24, 24),
                )
                row = np.full((32, panelpx, 3), 32, "uint8")
                cv.putText(
                    row,
                    nm,
                    (9, 22),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.56,
                    (236, 236, 236),
                    1,
                    cv.LINE_AA,
                )
                panels.append(np.vstack([row, crop]))
            frame = np.hstack(panels)
            base = np.full((34, frame.shape[1], 3), 24, "uint8")
            cv.putText(
                base,
                f"{seg}: {key}, {span:.1f} m square | t = {tm:.2f} s | best view, 2 cm native grid",
                (10, 23),
                0,
                0.58,
                (236, 236, 236),
                1,
                cv.LINE_AA,
            )
            dest = out / f"compare-{seg}-{key}.jpg"
            cv.imwrite(
                str(dest), np.vstack([base, frame]), [cv.IMWRITE_JPEG_QUALITY, 88]
            )
            assert dest.stat().st_size < 500000


if __name__ == "__main__":
    make()
