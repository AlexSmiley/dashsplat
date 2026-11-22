"""Exposure and sensor shading in original JPEG RGB code values; no tone mapping."""

from pathlib import Path
import os
import numpy as np


def basis(uv):
    # All fields vanish at optical image centre; offsets are exposure gains.
    x, y = ((np.asarray(uv) - 0.5) * 2).T
    poly = np.stack(
        (x, y, x * x, y * y, x * y, x * x * y, x * y * y, (x * x + y * y) ** 2), -1
    )
    nodes = np.linspace(-1, 1, 5)
    hx = np.maximum(0, 1 - abs(x[:, None] - nodes) * 2)
    hy = np.maximum(0, 1 - abs(y[:, None] - nodes) * 2)
    grid = (hy[:, :, None] * hx[:, None, :]).reshape(-1, 25)
    grid[:, 12] -= 1
    return np.column_stack((poly, grid))


class Photometry:
    def __init__(self, variant="stitched", root=None):
        root = Path(root or os.environ.get("RS", "."))
        a = np.load(root / "photo" / (variant + ".npz"))
        self.names = a["names"].tolist()
        self.sizes = a["sizes"]
        self.ns = a["ns"]
        self.ready = a["ready"] if "ready" in a else None
        self.gain = a["gain"]
        self.shade = a["shade"]
        self.epoch = a["epoch"]
        self.live = variant in ("live", "live-v2", "live-ground-v2")

    def params(self, cam, n):
        i = self.names.index(cam) if isinstance(cam, str) else cam
        ns = self.ns[:, i]
        k = np.clip(np.searchsorted(ns, n, "right") - 1, 0, len(ns) - 1)
        g = self.gain[k, i]
        if not self.live and k + 1 < len(ns) and ns[k + 1] != ns[k]:
            g = g + (self.gain[k + 1, i] - g) * np.clip(
                (n - ns[k]) / (ns[k + 1] - ns[k]), 0, 1
            )
        return i, g, self.shade[self.epoch[k], i]

    def available_at(self, cam, n):
        """Earliest timestamp with all causal row/pose/exposure inputs available."""
        if not self.live or self.ready is None:
            raise ValueError("readiness is defined for the live calibration only")
        i = self.names.index(cam) if isinstance(cam, str) else cam
        return float(
            self.ready[
                np.clip(
                    np.searchsorted(self.ns[:, i], n, "right") - 1,
                    0,
                    len(self.ready) - 1,
                )
            ]
        )

    def sample(self, cam, n, uv, rgb):
        """Correct RGB samples (0..255); UV are original COLMAP pixel centres."""
        i, g, v = self.params(cam, n)
        shape = np.shape(rgb)
        q = np.asarray(uv).reshape(-1, 2) / self.sizes[i]
        return np.clip(
            np.asarray(rgb).reshape(-1, 3)
            * np.exp(np.clip(g + basis(q) @ v, -0.65, 0.65)),
            0,
            255,
        ).reshape(shape)

    def apply(self, cam, n, image):
        """Correct an RGB full frame or a proportionally resized RGB frame."""
        import cv2 as cv

        i, g, v = self.params(cam, n)
        h, w = image.shape[:2]
        x, y = np.meshgrid((np.arange(65) + 0.5) / 65, (np.arange(49) + 0.5) / 49)
        lm = (basis(np.stack((x, y), -1).reshape(-1, 2)) @ v).reshape(49, 65, 3)
        scale = cv.resize(
            np.exp(np.clip(lm + g, -0.65, 0.65)).astype("f"),
            (w, h),
            interpolation=cv.INTER_LINEAR,
        )
        return np.clip(image * scale + 0.5, 0, 255).astype(image.dtype)
