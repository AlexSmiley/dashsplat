"""Independent multi-view height anchors off the carriageway. No depth model."""

import numpy as np
from pipeline.common.rig import Trajectory
from pipeline.photo.sample import RS, CAMS
from pipeline.road.geometry import pixels_to_ground
from collections import defaultdict


def run():
    m = np.load(RS / "work/road/pairs.npy")
    s = np.load(RS / "work/poses/schedule.npz")
    out = RS / "maps/ground-work"
    out.mkdir(exist_ok=True, parents=True)
    for name in ("live", "stitched"):
        tr = Trajectory(
            RS / "poses" / ("ego_causal.npz" if name == "live" else "ego.npz")
        )
        GG = []
        SS = []
        lat = []
        times = []
        for side in (0, 4):
            k, c = m[:, side : side + 2].astype(int).T
            uv = m[:, side + 2 : side + 4]
            g = np.zeros((len(k), 3))
            ss = np.zeros((len(k), 2))
            l = np.zeros(len(k))
            tm = np.zeros(len(k))
            for i, cam in enumerate(CAMS):
                u = c == i
                tf = s["tc"][k[u], i]
                t = cam.row_time(tf, uv[u, 1])
                E = tr.T_world_ego(t)
                o = (E @ cam.T_ego_cam)[:, :3, 3]
                G, ok = pixels_to_ground(cam, tr, tf, uv[u])
                ray = G - o
                g[u] = G
                ss[u] = ray[:, :2] / ray[:, 2, None]
                l[u] = abs(np.einsum("ni,ni->n", G - E[:, :3, 3], E[:, :3, 1]))
                tm[u] = t
            GG.append(g)
            SS.append(ss)
            lat.append(l)
            times.append(tm)
        e = GG[0][:, :2] - GG[1][:, :2]
        J = SS[0] - SS[1]
        den = (J * J).sum(1)
        z = -(e * J).sum(1) / np.maximum(den, 1e-8)
        valid = (
            (den > 5)
            & (np.linalg.norm(e + J * z[:, None], axis=1) < 0.09)
            & (m[:, 8] > 0.85)
            & (np.minimum(*lat) > 3.4)
            & (z > 0.045)
            & (z < 0.30)
        )
        xy = (GG[0][:, :2] + SS[0] * z[:, None] + GG[1][:, :2] + SS[1] * z[:, None]) / 2
        tm = np.maximum(*times) + (0.5 if name == "live" else 0)
        hist = defaultdict(list)
        records = []
        for ii in np.flatnonzero(valid)[np.argsort(tm[valid])]:
            key = tuple(np.floor(xy[ii] / 1.5).astype(int))
            o = hist[key]
            if not o or abs(tm[ii] - o[-1][0]) > 0.09:
                o.append((tm[ii], z[ii]))
            if len(o) >= 3 and np.ptp([q[0] for q in o]) > 0.45:
                zz = np.array([q[1] for q in o])
                med = np.median(zz)
                if np.median(abs(zz - med)) < 0.032:
                    records.append([*key, tm[ii], med])
        np.save(out / (name + "-height.npy"), np.array(records).reshape(-1, 4))
        print(
            name,
            "supported raised cells",
            len(set(tuple(r[:2]) for r in records)),
            flush=True,
        )


if __name__ == "__main__":
    run()
