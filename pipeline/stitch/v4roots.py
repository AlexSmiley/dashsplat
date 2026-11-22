"""Protect grounded vertical chains already visible in v3 from false free-space in a single-image warp."""

from .v4teacher import *
from scipy.ndimage import maximum_filter


def run(src, dst):
    refm = J(SRC / "map.json")
    m = J(src / "map.json")
    vol = Volume()
    rt = cKDTree(traj.T_world_ego(np.arange(START, END, 0.1))[:, :2, 3])
    objs = [reader(SRC / t["file"]) for t in refm["tiles"] if t["layer"] == "objects"]
    Q = np.concatenate(objs)
    g, h, _ = vol.lookup(Q[:, :3])
    rel = Q[:, 2] - g
    lat = rt.query(Q[:, :2])[0]
    alpha = (
        1 / (1 + np.exp(-Q[:, 9]))
    )  # An occupied column needs a substantial connected presence below 1.2 m and at successive heights into the crown; roads with no such column are not restored.
    pix = 0.30
    ij = np.floor(Q[:, :2] / pix).astype(int)
    ij -= ij.min(0)
    x, y = ij.T
    nx, ny = ij.max(0) + 1
    centres = (lat < 58) & (Q[:, 2] > h + 0.015) & (alpha > 0.12)
    grid = []
    for lo, hi, win in [(0.18, 1.2, 5), (0.8, 1.8, 5), (1.55, 2.9, 7), (2.6, 5.5, 9)]:
        mask = centres & (rel >= lo) & (rel < hi)
        mass = (
            np.bincount((y * nx + x)[mask], weights=alpha[mask], minlength=ny * nx)
            .reshape(ny, nx)
            .astype("f")
        )
        grid.append(maximum_filter(mass, size=win) > 1.25)
    chain = np.logical_and.reduce(grid)
    allowed = centres & (rel > 0.12) & (rel < 5.2) & chain[y, x]
    offset = 0
    rows = []
    missing = reinforced = 0
    protected = cKDTree(np.load(OUT / "canopy.npz")["original"])
    exact = 0
    for t in m["tiles"]:
        a = reader(src / t["file"])
        if t["layer"] == "ground":
            rows.append(a)
            continue
        rr = reader(SRC / t["file"])
        select = allowed[offset : offset + len(rr)]
        offset += len(rr)
        b = rr[select]
        if not len(b):
            rows.append(a)
            continue
        b = b[protected.query(b[:, :3], distance_upper_bound=0.028)[0] >= 0.028]
        d, i = cKDTree(a[:, :3]).query(b[:, :3], distance_upper_bound=0.034)
        existing = d < 0.034
        a = a.copy()
        keep = np.flatnonzero(existing)
        same = keep[np.max(abs(a[i[keep], 10:13] - b[keep, 10:13]), axis=1) < 0.30]
        reinforced += int((a[i[same], 9] < b[same, 9] - 0.10).sum())
        np.maximum.at(a[:, 9], i[same], b[same, 9])
        add = b[~existing]
        rows.append(np.concatenate([a, add]))
        missing += len(add)
        exact += len(same)
    assert offset == len(Q)
    write_scene(src, dst, rows)
    savej(
        OUT / "root-guard.json",
        dict(
            eligible=int(allowed.sum()),
            restored=missing,
            reinforced=reinforced,
            matched=exact,
        ),
    )
    print(J(OUT / "root-guard.json"), flush=True)


if __name__ == "__main__":
    run(Path(sys.argv[1]), Path(sys.argv[2]))
