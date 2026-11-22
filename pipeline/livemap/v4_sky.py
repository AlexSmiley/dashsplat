"""Rotation-only sky. JPEG u=.5-azimuth/(2π), upper edge +z (matches viewer)."""

from .v4_packet import *
from scipy.ndimage import gaussian_filter, distance_transform_edt


def sky(p, root, out):
    w, h = 1024, 512
    weight = np.zeros(w * h)
    acc = np.zeros((3, w * h))
    used = 0
    for c, es in p.items():
        ray, ok = lattice(c)
        for e in es[::5]:
            n = e["n"]
            ms = cv.imread(str(root / "masks/sky" / c / f"{n:06d}.png"), 0)
            valid = cv.erode(ms, np.ones((9, 9), "B")) > 0
            y, x = np.where((valid & ok)[::4, ::4])
            y *= 4
            x *= 4
            if not len(y):
                continue
            a = photo("live").apply(
                c,
                n,
                cv.resize(
                    cv.imread(str(RS / "data/drive/frames" / c / f"{n:06d}.jpg"))[
                        :, :, ::-1
                    ],
                    (B, H),
                    interpolation=cv.INTER_AREA,
                ),
            )[y, x]
            rw = rows("live", c, e["t"])[0]
            v = np.einsum("nij,nj->ni", rw[y, :3, :3], ray[y, x])
            i = (
                np.floor((0.5 - np.arctan2(v[:, 1], v[:, 0]) / (2 * np.pi)) * w).astype(
                    int
                )
                % w
            ) + w * np.clip(
                np.floor((0.5 - np.arcsin(np.clip(v[:, 2], -1, 1)) / np.pi) * h),
                0,
                h - 1,
            ).astype(int)
            g = np.maximum(0.05, ray[y, x, 2]) ** 2
            weight += np.bincount(i, weights=g, minlength=w * h)
            used += 1
            for j in range(3):
                acc[j] += np.bincount(i, weights=g * a[:, j], minlength=w * h)
    weight = weight.reshape(h, w)
    coverage = float(np.mean(weight[: h // 2] > 0))
    den = gaussian_filter(weight, 3, mode=("nearest", "wrap"))
    valid = den > 0.04
    rgb = np.stack(
        [
            gaussian_filter(x.reshape(h, w), 3, mode=("nearest", "wrap"))
            / np.maximum(den, 0.04)
            for x in acc
        ],
        2,
    )
    mask = np.tile(valid, (1, 3))
    nn = distance_transform_edt(~mask, return_distances=False, return_indices=True)[
        :, :, w : 2 * w
    ]
    f = rgb[nn[0], nn[1] % w]
    smooth = gaussian_filter(f, (14, 14, 0), mode=("nearest", "wrap", "nearest"))
    blend = np.clip(den / 2, 0, 1)[:, :, None]
    im = np.uint8(np.clip(f * blend + smooth * (1 - blend), 0, 255))
    cv.imwrite(str(out), im[:, :, ::-1], [cv.IMWRITE_JPEG_QUALITY, 90])
    return dict(frames=used, upper_hemisphere_measured=coverage)
