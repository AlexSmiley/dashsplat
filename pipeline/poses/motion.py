"""Telemetry baseline, knots, exact stop equality, master-clock product utilities."""

from .tracks import *
from scipy.spatial.transform import Rotation as Rot
from pipeline.common.rig import _exp, _log, _skew, _mv, _T, Trajectory


def inputs():
    with np.load(W / "schedule.npz") as a:
        s = dict(a)
    with np.load(D / "timeline.npz") as a:
        d = dict(a)
    t = d["t"]
    speed = np.maximum(d["speed"], 0)
    stop = speed < 0.01
    yaw = np.unwrap(d["yaw"]).copy()
    x = d["x"].copy()
    y = d["y"].copy()
    a = np.flatnonzero(stop & ~np.r_[False, stop[:-1]])
    b = np.flatnonzero(stop & ~np.r_[stop[1:], False])
    for i, j in zip(a, b):
        yaw[i : j + 1] = yaw[i]
        x[i : j + 1] = x[i]
        y[i : j + 1] = y[i]
    base = np.column_stack((x, y, yaw))
    return s, d, t, base, stop


def sample(t, base, tt):
    return np.stack([np.interp(tt, t, v) for v in base.T], -1)


def Rs(yaw):
    return _exp(np.stack((yaw * 0, yaw * 0, yaw), -1))


def assemble(base, c):
    return _T(
        Rs(base[..., 2]) @ _exp(c[..., :3]),
        np.pad(base[..., :2], [(0, 0)] * (base.ndim - 1) + [(0, 1)]) + c[..., 3:],
    )


def knots(s, t, stop):
    ix = s["ix"]
    ds = s["distance"]
    selected = [0]
    for k in range(1, len(ix)):
        j = selected[-1]
        i = ix[k]
        border = stop[i] or stop[ix[k - 1]] or (k + 1 < len(ix) and stop[ix[k + 1]])
        if (
            border
            or ds[k] - ds[j] >= 1.7
            or t[i] - t[ix[j]] >= 0.14
            or k == len(ix) - 1
        ):
            selected.append(k)
    sk = np.array(selected)
    ids = np.arange(len(sk))
    duplicate = stop[ix[sk[1:]]] & stop[ix[sk[:-1]]]
    ids[1:] -= np.cumsum(duplicate)
    first = np.r_[True, ~duplicate]
    return sk, ids, first


def brackets(ts, tt):
    j = np.clip(np.searchsorted(ts, tt, "right") - 1, 0, len(ts) - 2)
    return j, np.clip((tt - ts[j]) / (ts[j + 1] - ts[j]), 0, 1)


def corr_interp(ts, c, tt):
    j, a = brackets(ts, tt)
    return c[j] * (1 - a[..., None]) + c[j + 1] * a[..., None]


def export(name, t, poses, sp, sr, nt, stop):
    # Gauge: first ego origin is exactly ENU zero, including elevation.
    poses = poses.copy()
    poses[:, :3, 3] -= poses[0, :3, 3].copy()
    for a, b in zip(
        np.flatnonzero(stop & ~np.r_[False, stop[:-1]]),
        np.flatnonzero(stop & ~np.r_[stop[1:], False]),
    ):
        poses[a : b + 1] = poses[a]
    OUT.mkdir(exist_ok=True, parents=True)
    dest = OUT / (name + ".npz")
    tmp = OUT / (name + ".partial.npz")
    q = quality(t, nt, stop, name)
    np.savez(
        tmp,
        t=t,
        T_world_ego=poses,
        sigma_pos=sp,
        sigma_rot=sr,
        n_tracks=nt,
        quality_flag=q,
    )
    os.replace(tmp, dest)
    return poses


def quality(t, nt, stop, name):
    # Distinguish intentional stillness from a moving frame with weak vision.
    q = np.asarray(stop, dtype="uint8")
    q |= (2 * ((~stop) & (nt < (215 if name == "ego" else 600)))).astype("uint8")
    for j in np.flatnonzero(np.diff(t) > 0.075) + 1:
        q[(t >= t[j]) & (t < t[j] + 0.35)] |= 4
    return q
