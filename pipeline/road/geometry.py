"""Road pixels ↔ ENU metres. COLMAP pixel centres; all projections use source-row time."""

import numpy as np
from pipeline.common.rig import _mv


def pixels_to_ground(cam, trajectory, t_frame, uv, max_range=12.0):
    """Return world XYZ and validity, preserving arbitrary leading UV dimensions."""
    uv = np.asarray(uv)
    t = cam.row_time(t_frame, uv[..., 1])
    T = trajectory.T_world_ego(t)
    n, h = trajectory.ground_plane(t)
    ray = cam.unproject(uv) @ cam.T_ego_cam[:3, :3].T
    o = cam.T_ego_cam[:3, 3]
    den = (n * ray).sum(-1)
    dist = -(np.einsum("...i,i->...", n, o) + h) / np.where(
        abs(den) > 1e-9, den, np.nan
    )
    ego = o + dist[..., None] * ray
    world = _mv(T[..., :3, :3], ego) + T[..., :3, 3]
    valid = (
        cam.mask_at(uv)
        & (dist > 0)
        & (den < 0)
        & (np.linalg.norm(ego[..., :2], axis=-1) <= max_range)
    )
    return world, valid


def ground_to_pixels(cam, trajectory, t_frame, xy, max_range=12.0):
    """Return (uv, valid, world_z) on the local road tangent. Iterate row exposure."""
    xy = np.asarray(xy, float)
    shape = xy.shape[:-1]
    xy = xy.reshape(-1, 2)
    rows = np.linspace(0, cam.height, 33)
    times = cam.row_time(t_frame, rows)
    E = trajectory.T_world_ego(times)
    CW = E @ cam.T_ego_cam
    n, h = trajectory.ground_plane(times)
    nw = _mv(E[:, :3, :3], n)
    dw = h - (nw * E[:, :3, 3]).sum(1)
    r = np.full(len(xy), 16.0)
    for it in range(4):
        j = np.clip(r.astype(int), 0, 31)
        w = np.clip(r - j, 0, 1)
        mix = lambda a: (
            a[j] + (a[j + 1] - a[j]) * w.reshape((-1,) + (1,) * (a.ndim - 1))
        )
        N = mix(nw)
        z = -(np.einsum("ni,ni->n", xy, N[:, :2]) + mix(dw)) / N[:, 2]
        v = np.column_stack((xy, z)) - mix(CW[:, :3, 3])
        R = mix(CW[:, :3, :3])
        q = np.einsum("nji,nj->ni", R, v)
        uv, valid = cam.project(q)
        r = np.clip(
            np.nan_to_num(uv[:, 1], nan=cam.height / 2) * 32 / cam.height, 0, 32
        )
    valid &= (np.einsum("ni,ni->n", N, v) < 0) & (
        np.linalg.norm(xy - mix(E[:, :2, 3]), axis=1) <= max_range
    )
    return uv.reshape(shape + (2,)), valid.reshape(shape), z.reshape(shape)
