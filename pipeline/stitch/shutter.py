"""Native residual against measured BA pixels; exact and quantised source-row pose."""

from .views import *


def proof():
    a = np.load(RS / "work/poses/offline-fit.npz")
    s = np.load(RS / "work/poses/schedule.npz")["tc"]
    old = Trajectory(RS / "poses/ego_v1.npz")
    k, cc = a["k"], a["cam"]
    ts = s[k, cc]
    points = a["pts"][np.repeat(np.arange(len(a["n"])), a["n"])]
    out = {}
    for i, c in enumerate(CAMS):
        use = (cc == i) & (ts >= START) & (ts < END)
        t = ts[use]
        C = rig[c]
        uv = a["uv"][use]
        row = uv[:, 1]
        exact = T_world_cam(C, t, row, traj)
        v1 = T_world_cam(C, t, row, old)
        loc = np.einsum("nji,nj->ni", v1[:, :3, :3], points[use] - v1[:, :3, 3])
        xyz = np.einsum("nij,nj->ni", exact[:, :3, :3], loc) + exact[:, :3, 3]
        good = C.mask_at(uv) & (loc[:, 2] > 1)
        res = []
        ref = C.project(loc)[0]
        for band in [1, BANDS, 0]:
            r = (
                (np.floor(row * band / C.height) + 0.5) * C.height / band
                if band
                else row
            )
            T = T_world_cam(C, t, r, traj)
            xx = np.einsum("nji,nj->ni", T[:, :3, :3], xyz - T[:, :3, 3])
            pred, ok = C.project(xx)
            good &= ok
            res.append(
                (
                    np.linalg.norm(pred - uv, axis=-1),
                    np.linalg.norm(pred - ref, axis=-1),
                )
            )
        # Test camera-space to virtual and back at the same landmarks. These coordinates use COLMAP centres.
        errors = []
        for yaw in designs(c):
            K, R, *_ = design(c, yaw)
            v = loc @ R
            pix = (v @ K.T) / v[:, 2:3]
            ok = (
                good
                & (v[:, 2] > 0)
                & (pix[:, 0] >= 0.5)
                & (pix[:, 0] < W - 0.5)
                & (pix[:, 1] >= 0.5)
                & (pix[:, 1] < H - 0.5)
            )
            rays = (pix[ok] @ np.linalg.inv(K).T) @ R.T
            errors.extend(np.linalg.norm(C.project(rays)[0] - ref[ok], axis=1))
        out[c] = dict(
            n=int(good.sum()),
            residual_px=[np.percentile(r[0][good], [50, 90]).tolist() for r in res],
            approximation_px=[
                np.percentile(r[1][good], [50, 90]).tolist() for r in res[:2]
            ],
            roundtrip_max=float(max(errors)),
        )
    savej(WORK / "shutter.json", out)
    print(json.dumps(out))


if __name__ == "__main__":
    proof()
