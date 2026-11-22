"""Physical distorted grid, causal availability and row-timed transforms."""

from functools import lru_cache
import cv2 as cv, numpy as np, json
from .inputs import W, RS, CAMS, NAMES
from pipeline.common.rig import load_rig, Trajectory
from pipeline.photo.model import Photometry

cv.setNumThreads(1)
rig = load_rig(RS / "calib/rig.json")
H, B = 469, 724


@lru_cache(None)
def traj(mode):
    return Trajectory(
        RS / "poses" / ("ego_causal.npz" if mode == "live" else "ego.npz")
    )


@lru_cache(None)
def photo(mode):
    return Photometry(mode + "-ground-v2", RS)


@lru_cache(None)
def lattice(name):
    c = rig[name]
    y, x = np.mgrid[:H, :B]
    uv = np.stack(((x + 0.5) * c.width / B, (y + 0.5) * c.height / H), -1)
    ray = c.unproject(uv)
    valid = c.mask_at(uv) & (ray[:, :, 2] > np.cos(np.radians(88)))
    return np.nan_to_num(ray).astype("f"), valid


def maps(mode, c, n):
    key = f"{n:06d}"
    z = np.load(W / mode / "depth" / c / (key + ".npy")).astype("f")
    assert z.shape == (H, B)
    ray, ok = lattice(c)
    dist = z / np.maximum(ray[:, :, 2], 0.01)
    masks = [
        cv.imread(str(RS / "masks" / k / c / (key + ".png")), cv.IMREAD_UNCHANGED)
        for k in ("sky", "dyn", "road")
    ]
    assert (
        all(m is not None and m.shape == (H, B) for m in masks)
        and masks[1].dtype == np.uint16
    )
    bad = cv.dilate(
        ((masks[0] != 0) | (masks[1] != 0)).astype("B"), np.ones((5, 5), "B")
    ).astype(bool)
    base = ok & (z > 0) & ~bad
    lo = cv.erode(dist, np.ones((3, 3), "B"))
    hi = cv.dilate(dist, np.ones((3, 3), "B"))
    edge = hi - lo > 0.04 * np.maximum(dist, 0.1)
    safe = base & ~cv.dilate(edge.astype("B"), np.ones((5, 5), "B")).astype(bool)
    return dist, safe, masks[2] != 0, base


def rows(mode, name, t):
    c = rig[name]
    times = c.row_time(t, (np.arange(H) + 0.5) * c.height / H)
    tr = traj(mode)
    T = tr.T_world_ego(times)
    cw = T @ c.T_ego_cam
    n, h = tr.ground_plane(times)
    nc = n @ c.T_ego_cam[:3, :3]
    height = h + n @ c.T_ego_cam[:3, 3]
    return cw, nc, height


def projected(name, xyz, cw):
    c = rig[name]
    j = np.full(len(xyz), H // 2, "i")
    for _ in range(3):
        # indexed vector multiply; slabs avoid one interpolated matrix per splat
        R = cw[j, :3, :3]
        v = np.einsum("nji,nj->ni", R, xyz - cw[j, :3, 3])
        uv, ok = c.project(v)
        j = np.clip(np.nan_to_num(uv[:, 1]) * H / c.height, 0, H - 1).astype(int)
    return uv, ok & (v[:, 2] > 0), v, j


@lru_cache(None)
def frame_times(c):
    r = json.loads((RS / "data/drive/cams" / f"{c}.json").read_text())
    return {x["n"]: x["t"] for x in r}


@lru_cache(None)
def release_table():
    tr = traj("live")
    return tr.t, tr.t + np.load(RS / "work/road/causal-lag.npy")


def ready(mode, c, e):
    if mode != "live":
        return e["t"]
    # Depth window uses earlier physical frames, whose pose-release times can vary.
    ct = rig[c]
    tl, rl = release_table()
    times = [frame_times(c)[q] for q in e["window"]]
    latest = photo(mode).available_at(c, e["n"])
    for t in times:
        a = t + ct.time_offset_s - abs(ct.rs_readout_s or 0) * 0.5
        b = t + ct.time_offset_s + abs(ct.rs_readout_s or 0) * 0.5
        i = max(0, np.searchsorted(tl, a) - 1)
        j = min(len(tl), np.searchsorted(tl, b) + 1)
        latest = max(latest, b, rl[i:j].max())
    return float(latest)


def frames():
    return json.loads((W / "plan.json").read_text())
