"""Signed residual along the exposure-time image velocity, on fast accepted tracks.
A diagnostic only: landmarks/trajectory share the fit and are not a clock reference.
"""

from .validate import *
from pipeline.common.rig import _mv


def main():
    s = Fit().load()
    tr = Trajectory(OUT / "ego.npz")
    ids = np.repeat(np.arange(len(s.n)), s.n)
    shift = s.corr[s.kids[0], 3:] + np.r_[s.base[0, :2], 0]
    out = {}
    for i, cam in enumerate(s.C):
        ix = np.flatnonzero((s.cam == i) & (s.d["speed"][s.ix[s.k]] > 14))
        sub = []
        for a in range(0, len(ix), 14000):
            v = ix[a : a + 14000]
            k = s.k[v]
            X = s.pts[ids[v]] - shift
            uv = s.uv[v]
            t = cam.row_time(s.tc[k, i], uv[:, 1])

            def proj(dt):
                T = tr.T_world_ego(t + dt) @ cam.T_ego_cam
                return cam.project(
                    _mv(T[:, :3, :3].transpose(0, 2, 1), X - T[:, :3, 3])
                )[0]

            e = uv - proj(0)
            vel = (proj(0.001) - proj(-0.001)) / 0.002
            den = (vel * vel).sum(1)
            ok = (den > 150**2) & (den < 4000**2) & (np.linalg.norm(e, axis=1) < 4)
            ms = np.einsum("ij,ij->i", vel, e) / np.maximum(den, 1) * 1000
            sub.append(np.column_stack((t[ok], ms[ok])))
        a = np.vstack(sub)
        out[cam.name] = [
            len(a),
            float(np.median(a[:, 1])) if len(a) else None,
            [
                float(np.median(v))
                for begin, end in [(110, 190), (200, 252), (290, 361.5)]
                if len(v := a[(a[:, 0] >= begin) & (a[:, 0] < end), 1])
            ],
        ]
    json.dump(out, open(W / "timing.json", "w"), indent=2)
    print(out)


if __name__ == "__main__":
    main()
