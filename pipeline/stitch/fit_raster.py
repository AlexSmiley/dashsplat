"""V4/V5 appearance fit: fixed geometry, gradients to DC and opacity, reusable binning."""

from functools import lru_cache
from pathlib import Path
import ctypes as ct
import torch
from .model import gpu


def raster_plan(field, T, K, w, h, alpha):
    cuda = field.is_cuda
    if cuda:
        from metalsplat.reference.project_ref import project_gaussians
        from metalsplat.reference.tiling_ref import (
            bin_and_sort_gaussians,
            DEFAULT_TILE_SIZE,
        )
    else:
        from metalsplat.ops.project import project_gaussians
        from metalsplat.ops.tiling import bin_and_sort_gaussians, DEFAULT_TILE_SIZE
        from metalsplat.ops.rasterize import _RasterizeGaussiansImpl
    result = project_gaussians(
        field[:, :3],
        field[:, 7:10].exp(),
        field[:, 10:14],
        gpu(T[:3, :3].T),
        gpu(-T[:3, :3].T @ T[:3, 3]),
        *K,
        w,
        h,
        near=0.28,
        eps2d=0.25,
    )
    if cuda:
        u, z, co, rad, valid, comp = (
            result.means2d,
            result.depths,
            result.conics,
            result.radii,
            result.valid,
            result.compensation,
        )
    else:
        u, z, co, rad, valid, comp = result
    opacity = alpha * comp
    tile = DEFAULT_TILE_SIZE
    bins = bin_and_sort_gaussians(
        u, z, co, rad, valid, w, h, tile, opacities=opacity.detach()
    )
    if cuda:

        def draw(color, background):
            return _Draw.apply(
                u,
                co,
                opacity,
                color,
                bins.sorted_gaussian_ids,
                bins.tile_bins,
                background,
                w,
                h,
                tile,
            )
    else:

        def draw(color, background):
            return _RasterizeGaussiansImpl.apply(
                u,
                co,
                opacity,
                color,
                z,
                bins.sorted_gaussian_ids,
                bins.tile_bins,
                bins.tiles_x,
                w,
                h,
                tile,
                background,
                None,
            )[0]

    return z, draw


@lru_cache(None)
def _native(device_index):
    from torch.utils.cpp_extension import load

    # No torch ABI at the kernel boundary; the extension cache hashes the source and serializes builds.
    so = load(
        name="rs_fit_raster",
        sources=[str(Path(__file__).with_name("fit_raster.cu"))],
        is_python_module=False,
    )
    lib = ct.CDLL(so)
    for fn, n in [(lib.rs_fit_forward, 10), (lib.rs_fit_backward, 12)]:
        fn.argtypes = [ct.c_int] * 3 + [ct.c_void_p] * (n + 1)
        fn.restype = ct.c_int
    return lib


def _launch(fn, w, h, tile, *buffers):
    rc = fn(
        w,
        h,
        tile,
        *(x.data_ptr() for x in buffers),
        torch.cuda.current_stream(buffers[0].device).cuda_stream,
    )
    if rc:
        raise RuntimeError(f"CUDA stitch kernel launch failed ({rc})")


class _Draw(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, co, alpha, color, ids, bins, bg, w, h, tile):
        if u.requires_grad or co.requires_grad:
            raise ValueError("appearance fit expects fixed geometry")
        u, co, alpha, color, ids, bins, bg = [
            v.contiguous() for v in (u, co, alpha, color, ids, bins, bg)
        ]
        im = u.new_empty((h, w, 3))
        trans = u.new_empty((h, w))
        last = ids.new_empty((h, w))
        ctx.dims = w, h, tile
        with torch.cuda.device(u.device):
            _launch(
                _native(u.device.index).rs_fit_forward,
                w,
                h,
                tile,
                u,
                co,
                alpha,
                color,
                ids,
                bins,
                bg,
                im,
                trans,
                last,
            )
        ctx.save_for_backward(u, co, alpha, color, ids, bins, bg, trans, last)
        return im

    @staticmethod
    def backward(ctx, di):
        u, co, a, c, ids, bins, bg, tr, last = ctx.saved_tensors
        da = torch.zeros_like(a)
        dc = torch.zeros_like(c)
        with torch.cuda.device(u.device):
            _launch(
                _native(u.device.index).rs_fit_backward,
                *ctx.dims,
                u,
                co,
                a,
                c,
                ids,
                bins,
                bg,
                tr,
                last,
                di.contiguous(),
                da,
                dc,
            )
        return None, None, da, dc, None, None, None, None, None, None
