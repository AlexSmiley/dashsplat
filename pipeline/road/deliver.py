"""Verify complete products and unit/plane conventions before atomic replacement."""

from .evaluate import *
from pipeline.common.rig import ground_plane
import hashlib, shutil


def run():
    ctx = inputs()
    s, d, t, B, stop = ctx
    cams = load_rig(RS / "calib/rig.json")
    result = {}
    products = []
    lag = np.load(WORK / "causal-lag.npy")
    assert lag.max() <= 0.5
    result["lag_max_all_s"] = float(lag.max())
    result["lag_max_moving_s"] = float(lag[~stop].max())
    result["rig_sha256"] = hashlib.sha256(
        (RS / "calib/rig.json").read_bytes()
    ).hexdigest()
    for name in ("ego", "ego_causal"):
        backup = OUT / (name + "_v1.npz")
        assert backup.exists()
        p = WORK / (name + ".npz")
        tr = Trajectory(p)
        v = dict(np.load(p))
        R = tr.poses[:, :3, :3]
        n, h = tr.ground_plane(t)
        assert np.array_equal(t, tr.t) and np.array_equal(
            tr.poses[0, :3, 3], np.zeros(3)
        )
        assert np.max(abs(np.linalg.det(R) - 1)) < 1e-9
        assert all(np.isfinite(x).all() and len(x) == len(t) for x in v.values())
        norm = np.stack(
            (
                np.sin(v["susp_pitch"]),
                np.cos(v["susp_pitch"]) * np.sin(v["susp_roll"]),
                np.cos(v["susp_pitch"]) * np.cos(v["susp_roll"]),
            ),
            1,
        )
        assert np.max(abs(n - norm)) < 1e-9
        for k in (
            "T_world_ego",
            "road_pitch",
            "road_roll",
            "susp_pitch",
            "susp_roll",
            "susp_heave",
        ):
            assert (np.diff(v[k], axis=0)[stop[:-1] & stop[1:]] == 0).all()
        nw = np.einsum("nij,nj->ni", R, n)
        ds = np.r_[0, np.linalg.norm(np.diff(tr.poses[:, :2, 3], axis=0), axis=1)]
        avg = np.average(nw, axis=0, weights=ds)
        H = tr.poses[:, 2, 3] - h / nw[:, 2]
        zz = np.diff(H)
        motion = np.diff(tr.poses[:, :2, 3], axis=0)
        expect = -np.einsum("ni,ni->n", nw[:-1, :2], motion) / nw[:-1, 2]
        diff = zz - expect
        total = np.cumsum(ds)
        j = np.searchsorted(total, total + 100).clip(0, len(t) - 1)
        ix = total[j] - total > 99.5
        result[name] = dict(
            mean_normal_tilt_deg=float(
                np.rad2deg(np.arctan2(np.linalg.norm(avg[:2]), avg[2]))
            ),
            surface_z_range=[float(H.min()), float(H.max())],
            grade_integral_100m_p95_cm=float(
                np.percentile(
                    abs(np.r_[0, np.cumsum(diff)][j] - np.r_[0, np.cumsum(diff)])[ix],
                    95,
                )
                * 100
            ),
            backup_sha256=hashlib.sha256(backup.read_bytes()).hexdigest(),
        )
        products.append((p, OUT / (name + ".npz")))
        # Independent source pixel -> plane -> row-iterated camera check per lens.
        from .geometry import ground_to_pixels

        m = np.load(WORK / "pairs.npy")[::50]
        roundtrip = {}
        for ci, cam in enumerate(cams.values()):
            # Rig dictionary order need not agree with channel order.
            ci = NAMES.index(cam.name)
            u = m[m[:, 1] == ci]
            u = u[
                (s["t"][u[:, 0].astype(int)] > 293)
                & (s["t"][u[:, 0].astype(int)] < 326)
            ][:30]
            err = []
            for row in u:
                tf = s["tc"][int(row[0]), ci]
                uv = row[2:4].astype("d")
                G, ok = pixels_to_ground(cam, tr, tf, uv)
                vv, valid, _ = ground_to_pixels(cam, tr, tf, G[:2])
                if ok and valid:
                    err.append(np.linalg.norm(vv - uv))
            roundtrip[cam.name] = max(err) if err else None
        assert max(v for v in roundtrip.values() if v is not None) < 0.15
        result[name]["roundtrip_max_px"] = roundtrip
    for p, dst in products:
        tmp = dst.with_suffix(".road-partial.npz")
        shutil.copyfile(p, tmp)
        os.replace(tmp, dst)
        assert (
            hashlib.sha256(p.read_bytes()).digest()
            == hashlib.sha256(dst.read_bytes()).digest()
        )
    # Verify the public default resolves to the delivered decomposition, not a cached v1.
    n, h = ground_plane(t[1000])
    n1, h1 = Trajectory(OUT / "ego.npz").ground_plane(t[1000])
    assert np.allclose(n, n1) and h == h1
    json.dump(result, open(WORK / "checks.json", "w"), indent=2)
    print(result, flush=True)


if __name__ == "__main__":
    run()
