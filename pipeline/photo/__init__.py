"""Shared JPEG radiometry. RGB in and RGB out, in the capture's native look."""

from functools import lru_cache
from pathlib import Path
import os
from .model import Photometry


@lru_cache(maxsize=6)
def _open(root, variant, mtime):
    return Photometry(variant, root)


def load(variant="stitched", root=None):
    root = Path(root or os.environ.get("RS", ".")).resolve()
    return _open(
        root, variant, (root / "photo" / (variant + ".npz")).stat().st_mtime_ns
    )


def apply(cam, n, image, *, variant="stitched", root=None):
    return load(variant, root).apply(cam, n, image)
