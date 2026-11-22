"""Lift dense masked native colour samples onto visible object-local surfaces.

Photo UVs are registered to the projected body, never a billboard. Keep PLY
birth/death/observed runs; leave unsupported geometry in the inferred PLY.
"""

from pathlib import Path
import pickle, json, os, cv2, numpy as np
from scipy.spatial import cKDTree
from .demo import W, R
from .model import INF, projection, interp, ply
from .shape import samples
from .photographic import native_colour
from .online import camera
from pipeline.common.rig import Trajectory


def augment(data, o, dest, live):
    s, rows, track, size, stats, scan = data
    ego = Trajectory("poses/ego" + ("_causal" if live else "") + ".npz")
    p = dest / o["file"]
    base = np.frombuffer(
        p.read_bytes().split(b"end_header" + bytes([10]), 1)[1], "<f4"
    ).reshape(-1, 24)
    base = base[base[:, 9] < 4.1]
    patches = []
    active = {}
    last = {}
    count = 0
    cache = {}
    for r in rows:
        c = r["c"]
        tf = r["t"]
        t = r["release"]
        cam = camera(c)
        bb = np.array(r["bb"])
        bw = bb[2:] - bb[:2]
        if (
            not r["key"]
            or r.get("held")
            or min(bw) < 9
            or t > o["t_last"]
            or t - last.get(c, -1e3) < 0.12
        ):
            continue
        pose_track = track
        cls = s["class"]
        sz = size
        if live:
            k = np.searchsorted(scan["t"], t)
            epochs = stats["class_epochs"]
            if k >= len(scan["t"]) or scan["t"][k] - t > 0.35:
                continue
            t = scan["t"][k]
            cls = next((x for at, x in reversed(epochs) if at <= t), cls)
            from .inventory import SIZE

            sz = np.array(SIZE[cls])
            dt = np.array([-0.6, 0.0])
            vx, vy = scan["vx"][k], scan["vy"][k]
            pose_track = dict(
                t=t + dt,
                x=scan["x"][k] + vx * dt,
                y=scan["y"][k] + vy * dt,
                z=np.full(2, scan["z"][k]),
                yaw=np.full(2, np.arctan2(vy, vx)),
            )
        if cls not in cache:
            cache[cls] = samples(cls, sz, 0.065 if cls == "bus" else 0.041)
        P, N, S, Q, M, patch = cache[cls]
        uv, ok, face, z = projection(
            cam, ego, tf, pose_track, P, N, r if "C" in r else None
        )
        use = ok & (face > 0.18) & (z > 1)
        ii = np.flatnonzero(use)
        if len(ii) < 40:
            continue
        ext = np.r_[uv[ii].min(0), uv[ii].max(0)]
        wh = ext[2:] - ext[:2]
        ratio = bw / np.maximum(1, wh)
        if (
            np.min(ratio) < 0.66
            or np.max(ratio) > 1.5
            or np.any(
                abs((ext[:2] + ext[2:] - bb[:2] - bb[2:]) * 0.5)
                > np.maximum(0.47 * wh, 8)
            )
        ):
            continue
        # Depth ordering before interpolation keeps far wheels out of the near flank.
        xy = (uv[ii] - ext[:2]) * ratio + bb[:2]
        u, v = (
            np.clip(np.rint(xy[:, 0] * 1.5).astype(int), 0, 1085),
            np.clip(np.rint(xy[:, 1] * 1.5).astype(int), 0, 703),
        )
        zz = np.full((704, 1086), 1.0e5, "f")
        np.minimum.at(zz, (v, u), z[ii].astype("f"))
        zz = cv2.erode(zz, np.ones((3, 3), np.uint8))
        pick = z[ii] < zz[v, u] + 0.18
        ii = ii[pick]
        xy = xy[pick]
        if len(ii) < 30:
            continue
        root = Path(r.get("root", W / "extra" if s.get("input") == "extra" else "."))
        mask = (
            cv2.imread(str(root / "masks/dyn" / c / f"{r['n']:06}.png"), -1) == r["id"]
        )
        l, top, bx, bt = np.clip(
            np.rint(bb).astype(int), [0, 0, 0, 0], [723, 468, 724, 469]
        )
        l = max(0, l - 2)
        top = max(0, top - 2)
        bx = min(724, bx + 2)
        bt = min(469, bt + 2)
        k = cam.width // 724
        mat = cv2.resize(
            mask[top:bt, l:bx].astype("uint8"),
            ((bx - l) * k, (bt - top) * k),
            interpolation=cv2.INTER_LINEAR,
        )
        mat = cv2.erode(mat, np.ones((3, 3), np.uint8))
        area = mat.sum()
        step = max(1, int(np.ceil(np.sqrt(area / 7300))))
        Y, X = np.where(mat[::step, ::step] > 0)
        src = np.c_[l + (X * step + 0.5) / k, top + (Y * step + 0.5) / k]
        if len(src) < 48:
            continue
        dist, idx = cKDTree(xy).query(src, k=4)
        close = dist[:, 0] < max(1.9, 0.035 * np.min(wh))
        idx = ii[idx[close]]
        src = src[close]
        dist = dist[close]
        if len(src) < 48:
            continue
        # Interpolate object-local samples only across the same visible face, and
        # stand the measured skin 3 cm outside its lower-frequency support.
        weight = 1 / np.maximum(0.08, dist) ** 2
        weight *= np.einsum("nij,nj->ni", N[idx], N[idx[:, 0]]) > 0.45
        weight *= abs(z[idx] - z[idx[:, 0], None]) < 0.26
        weight /= weight.sum(1, keepdims=True)
        xyz = np.einsum("nk,nkj->nj", weight, P[idx])
        norm = np.einsum("nk,nkj->nj", weight, N[idx])
        norm /= np.maximum(
            1e-6, np.linalg.norm(norm, axis=1, keepdims=True)
        )  # Intersect the photographic ray with the local tangent plane. Weighted
        # centre interpolation alone collapses native pixels into stripes.
        original = (src - bb[:2]) / ratio + ext[:2]
        sensor = cam.unproject(original * [cam.width / 724, cam.height / 469])
        from pipeline.common.rig import T_world_cam

        C = r["C"] if "C" in r else T_world_cam(cam, tf, trajectory=ego)
        at = interp(pose_track, cam.row_time(tf, cam.height * 0.5))
        ca, sa = np.cos(at[3]), np.sin(at[3])
        rot = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
        O = (C[:3, 3] - at[:3]) @ rot
        D = (sensor @ C[:3, :3].T) @ rot
        den = np.sum(D * norm, axis=1)
        travel = np.sum((xyz - O) * norm, axis=1) / np.where(
            abs(den) > 0.12, den, -0.12
        )
        hit = O + travel[:, None] * D
        close = np.linalg.norm(hit - xyz, axis=1) < 0.14
        xyz[close] = hit[close]
        xyz += norm * 0.038
        im = cv2.imread(f"data/drive/frames/{c}/{r['n']:06}.jpg")[:, :, ::-1]
        rgb = native_colour(im, src, c, r["n"], t, live)
        cls_norm = np.median(norm, 0)
        sector = (
            int(np.floor((np.arctan2(cls_norm[1], cls_norm[0]) + np.pi) * 4 / np.pi))
            % 8
        )
        axis = idx[:, 0]
        px = np.clip(z[axis] / cam.params[0] * step, 0.009, 0.14)
        sig = np.c_[px * 0.88 / np.maximum(0.42, face[axis]), px * 0.86, px * 0.25]
        tt = np.full((len(src), 7), INF)
        tt[:, 1] = next((at for at, _ in epochs if at > t), INF) if live else INF
        tt[:, 0] = t
        tt[:, 2] = t + 1 / 36
        patches.append(
            np.c_[
                xyz,
                norm,
                (rgb / 255 - 0.5) / 0.28209479,
                np.full(len(src), 4.15),
                np.log(sig),
                Q[axis],
                tt,
            ].astype("f")
        )
        key = (c, sector, cls)
        old = active.get(key)
        if old is not None:
            patches[old][:, 18] = np.minimum(
                patches[old][:, 18],
                np.maximum(
                    np.nextafter(patches[old][:, 17], np.float32(np.inf)), t + 0.035
                ),
            )
        active[key] = len(patches) - 1
        last[c] = t
        count += len(src)
    if patches:
        tmp = p.with_suffix(".skin-next")
        ply(tmp, np.concatenate([base, *patches]))
        os.replace(tmp, p)
    return dict(native_samples=count, photographs=len(patches), measured_file=o["file"])


def run():
    live = not R.name.startswith("photo-demo")
    from .online_scene import scene_name

    dest = Path("scene") / scene_name() if live else R / "hindsight"
    oo = json.load(open(dest / "objects.json"))["objects"]
    ds = (
        {d[0]["id"]: d for d in pickle.load(open(R / "online.pickle", "rb"))}
        if live
        else {}
    )
    report = {}
    for o in oo:
        uid = o["id"]
        d = ds[uid] if live else pickle.load(open(R / f"fit-{uid}.pickle", "rb"))
        d = (
            d[0],
            d[1],
            {k: np.asarray(v) for k, v in o["track"].items()},
            np.array(o["size"]),
            *d[4:],
        )
        report[uid] = augment(d, o, dest, live)
    (R / "report/native-skin.json").write_text(json.dumps(report, indent=2))
    print(
        "dense native samples",
        sum(x["native_samples"] for x in report.values()),
        flush=True,
    )
    if not live:
        from .demo import attach, targets

        for parent in targets():
            attach(parent, dest, parent + "-movers-photo-v1", assets=dest)
        attach(
            "live-v5",
            dest,
            "live-v5-traffic-photo-hindsight",
            [(190, 215), (286, 330)],
            dest,
        )


if __name__ == "__main__":
    run()
