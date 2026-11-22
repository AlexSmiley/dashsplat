"""Panoptic tracks, ray bundles and physical facade planes. No alpha cards in the scene."""

from .v5world import *
from scipy.spatial import ConvexHull
from scipy.ndimage import maximum_filter

CATS = {5: "pole/shaft", 6: "signal head", 7: "sign panel"}


def semframes(c):
    last = -1e8
    for e in schedule()[c]:
        if e["hold"] or not semantic(c, e["n"]) or e["odo"] - last < 1.2:
            continue
        last = e["odo"]
        yield e


def rays_at(c, e, x, y):
    C = rig[c]
    uv = np.c_[(x + 0.5) * C.width / 724, (y + 0.5) * C.height / 469]
    r = C.unproject(uv)
    T = poses(c, e["t"])[np.clip((y * 8 / 469).astype(int), 0, 7)]
    return T[:, :3, 3], np.einsum("nij,nj->ni", T[:, :3, :3], r)


def project(c, e, P):
    C = rig[c]
    T = poses(c, e["t"])
    q = (P - T[4, :3, 3]) @ T[4, :3, :3]
    uv, ok = C.project(q)
    row = np.nan_to_num(uv[:, 1] * 8 / C.height).astype(int).clip(0, 7)
    M = T[row]
    q = np.einsum("ni,nij->nj", P - M[:, :3, 3], M[:, :3, :3])
    uv, ok = C.project(q)
    return uv, ok & (q[:, 2] > 0.5), np.linalg.norm(q, axis=1)


def collect():
    obs = []
    points = []
    ims = 0
    for c in CAMS:
        es = list(semframes(c))
        prev = None
        for e in es:
            n = e["n"]
            s = labels(c, n)
            im = rgb(c, n)
            ims += 1
            for k in CATS:
                _, ll, bb, ctr = cv.connectedComponentsWithStats(
                    (s == k).astype("B"), 8
                )
                for idx, (x, y, w, h, area) in enumerate(bb[1:], 1):
                    if (
                        area < 8
                        or h < (8 if k == 5 else 3)
                        or (k == 5 and (w > 1.15 * h or h > 330))
                    ):
                        continue
                    yy, xx = np.nonzero(ll[y : y + h, x : x + w] == idx)
                    xx += x
                    yy += y
                    cx, cy = np.median(xx), np.median(yy)
                    o, u = rays_at(
                        c, e, np.array([cx, cx, cx]), np.array([cy, y, y + h - 1])
                    )
                    if not np.isfinite(u).all():
                        continue
                    obs.append(
                        dict(
                            c=c,
                            n=n,
                            t=e["t"],
                            k=k,
                            bb=[int(v) for v in [x, y, w, h]],
                            O=o[0].tolist(),
                            ray=u[0].tolist(),
                            top=u[1].tolist(),
                            bottom=u[2].tolist(),
                            rgb=np.median(im[yy, xx], 0).tolist(),
                        )
                    )
            # Match repeat-observed building corners with row-timed rays, rather than relying on monocular far depth.
            gray = cv.cvtColor(im, cv.COLOR_RGB2GRAY)
            mask = (s == 2).astype("B")
            f = cv.goodFeaturesToTrack(gray, 1800, 0.003, 3, mask=mask, blockSize=5)
            if prev is not None and f is not None:
                ep, gp, mp = prev
                baseline = abs(e["odo"] - ep["odo"])
                if 0.7 < baseline < 20:
                    pp = f.reshape(-1, 2)
                    pq, st, _ = cv.calcOpticalFlowPyrLK(
                        gray, gp, pp, None, winSize=(27, 27), maxLevel=4
                    )
                    rev, sr, _ = cv.calcOpticalFlowPyrLK(
                        gp, gray, pq, None, winSize=(27, 27), maxLevel=4
                    )
                    ix, iy = pq.round().astype(int).T
                    valid = (
                        (st.ravel() > 0)
                        & (sr.ravel() > 0)
                        & (np.linalg.norm(pp - rev, axis=1) < 0.6)
                        & (mp[iy.clip(0, 468), ix.clip(0, 723)] > 0)
                    )
                    pp, pq = pp[valid], pq[valid]
                    if len(pp):
                        a, U = rays_at(c, e, *pp.T)
                        b, V = rays_at(c, ep, *pq.T)
                        cross = np.cross(U, V)
                        ss = (cross * cross).sum(1)
                        d = (np.cross(b - a, V) * cross).sum(1) / np.maximum(1e-9, ss)
                        f2 = (np.cross(b - a, U) * cross).sum(1) / np.maximum(1e-9, ss)
                        P = a + d[:, None] * U
                        ok = (
                            (ss > 0.00015)
                            & (d > 5)
                            & (d < 230)
                            & (f2 > 0)
                            & (
                                np.linalg.norm(P - b - f2[:, None] * V, axis=1)
                                < 0.08 + 0.004 * d
                            )
                        )
                        points.append(P[ok])
            prev = e, gray, mask
    savej(W5 / "instances.json", obs)
    np.save(
        W5 / "building-tracks.npy",
        np.concatenate(points) if points else np.empty((0, 3)),
    )
    print(
        "images / components / facade corners",
        ims,
        len(obs),
        sum(map(len, points)),
        flush=True,
    )


def tracks():
    obs = J(W5 / "instances.json")
    vol = Volume()
    parts = []
    records = []
    used = set()
    skipped = []
    for k in CATS:
        ii = np.array([i for i, o in enumerate(obs) if o["k"] == k], int)
        if len(ii) < 3:
            continue
        oo = [obs[i] for i in ii]
        O = np.array([o["O"] for o in oo])
        D = np.array([o["ray"] for o in oo])
        U = unit(D[:, :2])
        ts = np.array([o["t"] for o in oo])
        N = np.c_[-U[:, 1], U[:, 0]]
        off = (N * O[:, :2]).sum(1)
        sol = []
        indices = []
        for i in range(len(ii)):
            ids = np.flatnonzero(
                (np.arange(len(ii)) > i)
                & (abs(ts - ts[i]) < 8)
                & (np.linalg.norm(O[:, :2] - O[i, :2], axis=1) > 1.6)
            )
            den = N[ids] @ U[i]
            rng = (off[ids] - N[ids] @ O[i, :2]) / np.where(abs(den) > 0.055, den, 1e-8)
            P = O[i, :2] + rng[:, None] * U[i]
            r2 = ((P - O[ids, :2]) * U[ids]).sum(1)
            z1 = O[i, 2] + rng * D[i, 2] / np.linalg.norm(D[i, :2])
            z2 = O[ids, 2] + r2 * D[ids, 2] / np.linalg.norm(D[ids, :2], axis=1)
            ok = (
                (rng > 2)
                & (rng < 100)
                & (r2 > 2)
                & (r2 < 100)
                & (abs(z1 - z2) < (2 if k == 5 else 0.85))
            )
            P = P[ok]
            ids = ids[ok]
            near = rt.query(P)[0] < 40
            sol.extend(P[near])
            indices.extend(zip(np.full(near.sum(), i), ids[near]))
        if not sol:
            continue
        sol = np.array(sol)
        tree = cKDTree(sol)
        support = tree.query_ball_point(sol, 0.60, return_length=True)
        seen = set()
        local = set()
        for idx in np.argsort(-support):
            if support[idx] < 3:
                break
            ij = tuple(np.floor(sol[idx] / 0.5).astype(int))
            if ij in seen:
                continue
            seen.add(ij)
            x = sol[idx]
            dist = np.linalg.norm(x - O[:, :2], axis=1)
            along = ((x - O[:, :2]) * U).sum(1)
            inside = (
                (along > 2) & (along < 95) & (abs(N @ x - off) < 0.14 + 0.004 * dist)
            )
            z = O[:, 2] + along * D[:, 2] / np.linalg.norm(D[:, :2], axis=1)
            seedz = z[indices[idx][0]]
            inside &= abs(z - seedz) < (2.7 if k == 5 else 0.75)
            ids = [i for i in np.flatnonzero(inside) if i not in local]
            by = {}
            for i in ids:
                key = (oo[i]["c"], oo[i]["n"])
                err = abs(N[i] @ x - off[i])
                if key not in by or err < by[key][0]:
                    by[key] = err, i
            ids = np.array([v[1] for v in by.values()], int)
            if len(ids) < 4 or np.ptp(O[ids, :2], axis=0).max() < 2.8:
                continue
            xy = np.linalg.lstsq(N[ids], off[ids], rcond=None)[0]
            ids = ids[abs(N[ids] @ xy - off[ids]) < 0.13 + 0.004 * dist[ids]]
            if (
                len(ids) < 4
                or rt.query(xy)[0] > 40
                or any(
                    np.linalg.norm(xy - r["xy"]) < 0.55
                    and r["class"] == CATS[k]
                    and abs(r["top"] - seedz) < 3
                    for r in records
                )
            ):
                continue
            d = np.linalg.norm(xy - O[ids, :2], axis=1)
            t = np.array([oo[i]["top"] for i in ids])
            b = np.array([oo[i]["bottom"] for i in ids])
            hi = O[ids, 2] + d * t[:, 2] / np.linalg.norm(t[:, :2], axis=1)
            lo = O[ids, 2] + d * b[:, 2] / np.linalg.norm(b[:, :2], axis=1)
            g = float(vol.lookup(np.array([[*xy, 0]]))[0][0])
            z0 = float(np.median(lo))
            z1 = float(np.median(hi))
            width = np.array(
                [
                    oo[i]["bb"][2]
                    * rig[oo[i]["c"]].width
                    / 724
                    / rig[oo[i]["c"]].params[0]
                    * di
                    for i, di in zip(ids, d)
                ]
            )
            rw = float(np.median(width))
            color = np.median([oo[i]["rgb"] for i in ids], 0)
            tm = float(np.median(ts[ids]))
            good = (
                z1 > g + 0.6
                and z1 < g + 18
                and (k != 5 or z1 - z0 > 1.0)
                and np.percentile(abs(hi - z1), 60) < max(0.5, 0.21 * (z1 - z0))
            )
            local.update(ids)
            used.update(ii[ids])
            bottom_ok = z0 < g + 0.7 and z0 > g - 0.8
            rec = dict(
                id=f"{k}-{len(records) + 1:03}",
                class_=CATS[k],
                xy=xy.tolist(),
                path_m=float(rt.query(xy)[0]),
                floor=g,
                bottom=z0,
                top=z1,
                views=len(ids),
                component_ids=ii[ids].tolist(),
                time=tm,
                geometry=bool(good),
                status="unresolved",
            )
            rec["class"] = rec.pop("class_")
            records.append(rec)
            if not good:
                continue
            if k == 5:
                radius = np.clip(rw * 0.34, 0.025, 0.135)
                base = g if bottom_ok else max(g, z0)
                zz = np.linspace(
                    base, z1, max(2, int((z1 - base) / max(0.045, radius * 0.85)) + 1)
                )
                parts.append(
                    surfels(
                        np.c_[np.tile(xy, (len(zz), 1)), zz],
                        color,
                        [radius, radius, radius * 1.35],
                        0.965,
                        tm,
                    )
                )
                rec["status"] = (
                    "base-to-observed-top" if bottom_ok else "lower-shaft-unconfirmed"
                )
                rec["radius"] = float(radius)
            else:
                # Use actual masked native pixels from the closest confirmed exposure. Fit width and height metrically, not a cutout.
                j = ids[d.argmin()]
                ob = oo[j]
                x0, y0, w, h = ob["bb"]
                c, n = ob["c"], ob["n"]
                image = cv.imread(str(RS / "data/drive/frames" / c / f"{n:06}.jpg"))[
                    :, :, ::-1
                ]
                mask = labels(c, n)[y0 : y0 + h, x0 : x0 + w] == k
                direction = unit(O[j, :2] - xy)
                tangent = np.r_[-direction[1], direction[0], 0]
                width = float(np.clip(rw, 0.12, 3.0))
                height = float(np.clip(z1 - z0, 0.15, 3))
                nu, nv = max(3, int(width / 0.045)), max(3, int(height / 0.045))
                yy, xx = np.mgrid[:nv, :nu]
                u = (xx + 0.5) / nu
                v = (yy + 0.5) / nv
                sel = mask[
                    np.minimum(h - 1, (v * h).astype(int)),
                    np.minimum(w - 1, (u * w).astype(int)),
                ]
                P = np.r_[xy, z1] + (u[sel, None] - 0.5) * tangent * width
                P[:, 2] -= v[sel] * height
                uv, vis, _ = project(c, ob, P)
                px, py = uv.astype(int).T
                C = image[
                    py.clip(0, image.shape[0] - 1), px.clip(0, image.shape[1] - 1)
                ]
                B = surfels(
                    P[vis],
                    C[vis],
                    [width / nu * 0.68, height / nv * 0.68, 0.019],
                    0.95,
                    tm,
                )
                B[:, 13:17] = Rotation.from_matrix(
                    np.stack([tangent, [0, 0, 1], np.r_[direction, 0]], 1)
                ).as_quat()[[3, 0, 1, 2]]
                parts.append(B)
                rec["status"] = "panel-only"
    for r in records:
        if r["status"] == "panel-only":
            mounts = [
                s["id"]
                for s in records
                if s["status"] == "base-to-observed-top"
                and np.linalg.norm(np.subtract(s["xy"], r["xy"])) < 0.4
                and s["top"] > r["bottom"] - 0.35
            ]
            r["mounts"] = mounts
            if mounts:
                r["status"] = "panel-and-grounded-mount"
    # Mask components not localised must not disappear from the completeness denominator.
    savej(
        W5 / "inventory.json",
        dict(
            records=records,
            unassigned_component_ids=sorted(set(range(len(obs))) - used),
        ),
    )
    np.save(W5 / "thin.npy", np.concatenate(parts) if parts else np.empty((0, 24), "f"))
    print("tracks", len(records), len(used), "of", len(obs), flush=True)


def facades():
    pts = np.load(W5 / "building-tracks.npy")
    vol = Volume()
    g = vol.lookup(pts)[0]
    pts = pts[(pts[:, 2] > g + 0.2) & (pts[:, 2] < g + 38)]
    rng = np.random.default_rng(51)
    planes = []
    bags = []
    # Fit piecewise vertical planes over 18 m cells plus context. Track corners on building masks are the geometric anchors.
    cells = np.unique(np.floor(pts[:, :2] / 18).astype(int), axis=0)
    parts = []
    for cell in cells:
        ctr = (cell + 0.5) * 18
        ix = np.flatnonzero(abs(pts[:, :2] - ctr).max(1) < 12)
        if len(ix) < 28:
            continue
        P = pts[ix]
        for _ in range(2):
            if len(P) < 28:
                break
            ix = rng.integers(len(P), size=(130, 2))
            U = unit(P[ix[:, 1], :2] - P[ix[:, 0], :2])
            N = np.c_[-U[:, 1], U[:, 0]]
            bias = (N * P[ix[:, 0], :2]).sum(1)
            dis = abs(P[:, :2] @ N.T - bias)
            score = (dis < 0.35).sum(0)
            j = score.argmax()
            keep = dis[:, j] < 0.45
            Y = P[keep]
            P = P[~keep]
            if (
                len(Y) < 26
                or np.ptp(Y[:, 2]) < 1.15
                or np.ptp(Y[:, :2], axis=0).max() < 2.2
            ):
                continue
            mid = np.median(Y[:, :2], 0)
            _, _, vh = np.linalg.svd(Y[:, :2] - mid, full_matrices=False)
            tan = vh[0]
            normal = np.r_[tan[1], -tan[0], 0]
            uv = np.c_[(Y[:, :2] - mid) @ tan, Y[:, 2]]
            step = np.clip(rt.query(mid)[0] * 0.0015, 0.065, 0.18)
            u, z = np.meshgrid(
                np.arange(uv[:, 0].min() - 0.35, uv[:, 0].max() + 0.35, step),
                np.arange(uv[:, 1].min() - 0.35, uv[:, 1].max() + 0.35, step),
            )
            Q = np.c_[u.ravel(), z.ravel()]
            hull = ConvexHull(uv)
            ok = (Q @ hull.equations[:, :2].T + hull.equations[:, 2] < 0.35).all(1) & (
                cKDTree(uv).query(Q)[0] < 2.3
            )
            Q = Q[ok]
            X = np.c_[mid + Q[:, :1] * tan, Q[:, 1]]
            X = X[(np.floor(X[:, :2] / 18) == cell).all(1)]
            if len(X) < 50:
                continue
            A = surfels(
                X, np.ones(3) * 110, [step * 0.76, step * 0.76, 0.026], 0.96, START
            )
            A[:, 3:6] = normal
            A[:, 13:17] = Rotation.from_matrix(
                np.stack([np.r_[tan, 0], [0, 0, 1], normal], 1)
            ).as_quat()[[3, 0, 1, 2]]
            bags.append(A)
            planes.append(
                dict(
                    id=len(planes),
                    xy=mid.tolist(),
                    normal=normal.tolist(),
                    corners=len(Y),
                    n=len(X),
                    residual_m=float(np.median(abs((Y[:, :2] - mid) @ normal[:2]))),
                )
            )
    if not bags:
        np.save(W5 / "walls.npy", np.empty((0, 24), "f"))
        savej(W5 / "planes.json", [])
        return
    A = np.concatenate(bags)
    cnt = np.zeros(len(A), int)
    sky = cnt.copy()
    best = np.zeros(len(A))
    rgbest = np.zeros((len(A), 3))
    s0 = np.full(len(A), 1e8)
    s1 = -s0.copy()
    tree = cKDTree(A[:, :2])
    for c in CAMS:
        for e in semframes(c):
            O = poses(c, e["t"])[4, :3, 3]
            ids = np.array(tree.query_ball_point(O[:2], 200), int)
            uv, ok, dist = project(c, e, A[ids, :3])
            ids, uv, dist = ids[ok], uv[ok], dist[ok]
            C = rig[c]
            x = (uv[:, 0] * 724 / C.width).astype(int).clip(0, 723)
            y = (uv[:, 1] * 469 / C.height).astype(int).clip(0, 468)
            s = labels(c, e["n"])[y, x]
            sky[ids[s == 10]] += 1
            ok = s == 2
            ids, uv, dist = ids[ok], uv[ok], dist[ok]
            cnt[ids] += 1
            s0[ids] = np.minimum(s0[ids], e["odo"])
            s1[ids] = np.maximum(s1[ids], e["odo"])
            score = C.params[0] / np.maximum(1, dist)
            pick = score > best[ids]
            ids, uv = ids[pick], uv[pick]
            best[ids] = score[pick]
            if len(ids):
                patch = cv.imread(
                    str(RS / "data/drive/frames" / c / f"{e['n']:06}.jpg")
                )
                x, y = uv.astype(int).T
                rgbest[ids] = photo.sample(
                    c,
                    e["n"],
                    uv,
                    patch[y.clip(0, C.height - 1), x.clip(0, C.width - 1), ::-1],
                )
    ok = (cnt >= 4) & (s1 - s0 > 3) & (sky < cnt * 0.22)
    A[:, 6:9] = (rgbest / 255 - 0.5) / C0
    np.save(W5 / "walls.npy", A[ok])
    offset = 0
    for q in planes:
        n = q["n"]
        q["verified_texels"] = int(ok[offset : offset + n].sum())
        offset += n
    savej(W5 / "planes.json", planes)
    print("facade planes / texels", len(planes), sum(ok), flush=True)


if __name__ == "__main__":
    collect()
    tracks()
    facades()
