"""Re-measure calibrated observations at v2 centres and candidates removed by the previous sky pass."""

from .v2posterior import observations
from .v2select import readobjects
from .v2data import *

W = Path(os.getenv("STITCH_V3_WORK", RS / "work/stitch-v3")).resolve()
W.mkdir(exist_ok=True, parents=True)
BASE = Path(os.getenv("STITCH_V3_BASE", RS / "scene/stitched-opt-v2"))
FIT = Path(os.getenv("STITCH_V3_FIT", V / "photo-fit"))


def candidates():
    a = readobjects(BASE)
    b = readobjects(FIT)
    route = cKDTree(traj.T_world_ego(np.arange(START, END, 0.1))[:, :2, 3])
    lat = route.query(b[:, :2])[0]
    h = b[:, 2] - Volume().lookup(b[:, :3])[0]
    sel = (lat > 24) & (h > 0.85) & (h < 48) & (b[:, 9] > -2.8)
    b = b[sel]
    d = cKDTree(a[:, :3]).query(b[:, :3])[0]
    b = b[d > 0.28]
    return np.ascontiguousarray(np.concatenate([a, b]), "f"), len(a)


def run():
    a, n = candidates()
    st, di, nv = observations(a)
    np.savez(W / "observations.npz", stats=st, dirs=di, n=n, nv=nv)
    floor = Volume().lookup(a[:, :3])[0]
    h = a[:, 2] - floor
    alpha = 1 / (1 + np.exp(-a[:, 9]))
    rgb = a[:, 6:9] * 0.28209479 + 0.5
    ref = st[:, 7:10] / np.maximum(0.001, st[:, 6, None]) / 255
    res = {}
    for name, m in [
        ("first_metre", h < 1),
        (
            "crown",
            (h > 2)
            & (h < 12)
            & (rgb[:, 1] > rgb[:, 0] * 1.04)
            & (rgb[:, 1] > rgb[:, 2] * 1.02),
        ),
        ("additional", np.arange(len(a)) >= n),
    ]:
        if name != "additional":
            m &= np.arange(len(a)) < n
        q = m & (st[:, 6] > 0.65)
        res[name] = dict(
            n=int(m.sum()),
            alpha=np.percentile(alpha[m], [10, 50, 90]).tolist() if m.any() else [],
            under18=float((alpha[m] < 0.18).mean()) if m.any() else 0,
            real_rgb_n=int(q.sum()),
            stored=np.median(rgb[q], 0).tolist() if q.any() else [],
            photo=np.median(ref[q], 0).tolist() if q.any() else [],
            paired_brightness_delta=float(np.median((rgb[q] - ref[q]).mean(1)) * 255)
            if q.any()
            else None,
        )
    savej(W / "diagnosis.json", res)
    print(res, flush=True)


if __name__ == "__main__":
    run()
