"""Map novel-view depth support back to unreserved physical depth exposures; no training or pseudo data."""

from .v2eval import *
from .v2data import exposure_range, schedule, masks, poses, rays


def sample(r, t, deg):
    out = r.viewpoint(t, deg)
    h, w = out.shape[:2]
    y, x = np.mgrid[5:h:15, 5:w:15]
    T, K = viewer_pose(t, deg)
    o = T[:3, 3]
    u = world_ray(T, K, x, y)
    alpha = out[y, x, 3]
    obj = out[y, x, 5]
    dist = out[y, x, 4] / np.maximum(0.05, alpha)
    world = o + u * dist[:, :, None]
    region = (x / w, y / h)
    return world.reshape(-1, 3), [v.ravel() for v in [*region, obj, alpha, dist]]


def inputs(src):
    scenes = {}
    r = Wide(src)
    sched = schedule()
    res = {}
    for t in TIMES:
        for angle in [0, 80, 16, "raised"]:
            scenes[t, angle] = sample(r, t, angle)
        packs = [scenes[t, angle] for angle in [0, 80, 16, "raised"]]
        pts = np.concatenate([p[0] for p in packs])
        sz = len(packs[0][0])
        match = np.zeros(len(pts), "i")
        seen = match.copy()
        cz = np.zeros(len(pts), "f")
        lz = np.zeros(len(pts), "f")
        step = np.zeros(len(pts), "f")
        direction = np.zeros_like(pts)
        sep = np.zeros(len(pts), "f")
        camset = np.zeros(len(pts), "B")
        for ci, c in enumerate(CAMS):
            use = [e for e in sched[c] if e["depth"] and abs(e["t"] - t) < 10]
            last = -100
            C = rig[c]
            for e in use:
                if e["odo"] - last < 1.4:
                    continue
                last = e["odo"]
                P = poses(c, e["t"])
                v = (pts - P[4, :3, 3]) @ P[4, :3, :3]
                uv, _ = C.project(v)
                row = np.clip(
                    np.nan_to_num(uv[:, 1], nan=0) * 8 / C.height, 0, 7
                ).astype(int)
                v = np.einsum("ni,nij->nj", pts - P[row, :3, 3], P[row, :3, :3])
                uv, _ = C.project(v)
                x = np.floor(np.nan_to_num(uv[:, 0], nan=-100) * 724 / C.width).astype(
                    "i"
                )
                y = np.floor(np.nan_to_num(uv[:, 1], nan=-100) * 469 / C.height).astype(
                    "i"
                )
                ii = np.flatnonzero(
                    (v[:, 2] > 0.25) & (x >= 1) & (y >= 1) & (x < 723) & (y < 468)
                )
                x, y = x[ii], y[ii]
                sky, dyn, road = masks(c, e["n"])
                z, wt = exposure_range(e)
                ok = (
                    (sky[y, x] == 0)
                    & (road[y, x] == 0)
                    & (dyn[y, x] == 0)
                    & (wt[y, x] > 0.045)
                )
                ii, x, y = ii[ok], x[ok], y[ok]
                rr = np.linalg.norm(v[ii], axis=1)
                dd = z[y, x]
                seen[ii] += 1
                delta = abs(dd - rr) / np.maximum(rr, 1)
                agree = delta < 0.14 + 0.35 / np.maximum(rr, 1)
                jj = ii[agree]
                match[jj] += 1
                camset[jj] |= 1 << ci
                d = (pts[jj] - P[row[jj], :3, 3]) / rr[agree, None]
                old = match[jj] > 1
                sep[jj[old]] = np.maximum(
                    sep[jj[old]],
                    np.rad2deg(
                        np.arccos(np.clip((d[old] * direction[jj[old]]).sum(1), -1, 1))
                    ),
                )
                direction[jj[~old]] = d[~old]
                cz[ii] += np.minimum(delta, 2)
                lz[ii] = np.maximum(lz[ii], dd)
                step[ii] += dd > 60
        for ai, angle in enumerate([0, 80, 16, "raised"]):
            sl = slice(sz * ai, sz * (ai + 1))
            x, y, ob, alpha, dist = packs[ai][1]
            areas = {
                "upper silhouette": (x > 0.08) & (x < 0.92) & (y > 0.08) & (y < 0.34),
                "central foliage/facades": (x > 0.17)
                & (x < 0.83)
                & (y >= 0.26)
                & (y < 0.59),
                "verge and kerb": (y > 0.50) & (y < 0.77) & ((x < 0.28) | (x > 0.72)),
            }
            qq = {}
            for name, roi in areas.items():
                m = roi & (ob > 0.45) & (alpha > 0.45)
                z = dist[m]
                N = int(m.sum())
                s = seen[sl][m]
                mm = match[sl][m]
                se = sep[sl][m]
                err = cz[sl][m] / np.maximum(s, 1)
                qq[name] = dict(
                    sampled_cells=int(roi.sum()),
                    objects=N,
                    covered_fraction=float((alpha[roi] > 0.5).mean()),
                    range_median=float(np.median(z)) if N else None,
                    two_depth_fraction=float((mm >= 2).mean()) if N else None,
                    two_separated_degrees_fraction=float(((mm >= 2) & (se >= 5)).mean())
                    if N
                    else None,
                    median_capped_relative_disagreement=float(np.median(err[s > 0]))
                    if np.any(s > 0)
                    else None,
                )
            res[
                f"{t}s {dict([(0, 'default'), (80, 'left'), (16, 'right'), ('raised', 'raised')])[angle]}"
            ] = qq
    savej(RESULT / "region-evidence.json", res)


def semantics():
    reps = [
        min(
            (e for e in held() if e["c"] == c and e["clean"]),
            key=lambda e: abs(e["t"] - t),
        )
        for t in TIMES
        for c in CAMS
    ]
    summary = {}
    for st in ("v1", "v2"):
        r = Wide(RS / "scene" / ("stitched-opt-" + st))
        parts = {
            k: [] for k in ["sky", "roads", "nonroad_mse", "road_mse", "nonroad_cover"]
        }
        for e in reps:
            a = r.real(e)
            im, val = reference(e)
            sky, dyn, road = masks(e["c"], e["n"])
            sk, dy, ro = [
                cv.resize((m != 0).astype("B"), DIMS, interpolation=cv.INTER_NEAREST)
                for m in [sky, dyn, road]
            ]
            valid = body(e["c"]) & (dy == 0)
            sk = valid & cv.erode(sk, np.ones((5, 5), "B")).astype(bool)
            ro = val & cv.erode(ro, np.ones((5, 5), "B")).astype(bool)
            ob = val & (
                cv.dilate(
                    (
                        road
                        if road.shape == ro.shape
                        else cv.resize(
                            (road != 0).astype("B"),
                            DIMS,
                            interpolation=cv.INTER_NEAREST,
                        )
                    ),
                    np.ones((5, 5), "B"),
                )
                == 0
            )
            mse = ((im.astype("f") - a[:, :, :3]) / 255) ** 2
            for k, v in [
                ("sky", a[:, :, 3][sk]),
                ("roads", a[:, :, 5][ro]),
                ("nonroad_mse", mse[ob]),
                ("road_mse", mse[ro]),
                ("nonroad_cover", a[:, :, 3][ob] > 0.5),
            ]:
                parts[k].append(v)
        d = {k: float(np.mean(np.concatenate(v))) for k, v in parts.items()}
        d.update(
            nonroad_psnr=-10 * np.log10(d["nonroad_mse"]),
            road_psnr=-10 * np.log10(d["road_mse"]),
            frames=len(reps),
        )
        summary[st] = d
    savej(RESULT / "semantics.json", summary)


if __name__ == "__main__":
    inputs(RS / "scene/stitched-opt-v2")
    semantics()
