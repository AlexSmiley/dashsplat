"""Intersect bearings of persistent upright image ridges. Fit XY from all inlier images; retain observed shaft intervals only."""

from .v3terrain import surfels, W
from .v2seed import *
from scipy.ndimage import maximum_filter


def lines():
    ls = cv.createLineSegmentDetector(cv.LSD_REFINE_STD)
    obs = []
    seq = J(V / "train.json")
    last = {}
    vol = Volume()
    route = cKDTree(traj.T_world_ego(np.arange(START, END, 0.12))[:, :2, 3])
    for e in seq:
        c = e["c"]
        yaw = e["yaw"]
        ck = (c, yaw)
        k = e["odo"]
        if k - last.get(ck, -100) < 1.3:
            continue
        last[ck] = k
        w, h, f, R, mx, my, rows, rz = pin(c, yaw)
        M = poses(c, e["t"])
        img = np.load(e["file"])
        rgb0 = img["rgb"]
        lab = img["lab"]
        gray = cv.cvtColor(rgb0, cv.COLOR_RGB2GRAY)
        seg = ls.detect(gray)[0]
        if seg is None:
            continue
        for x, y, xx, yy in seg.reshape(-1, 4):
            if y > yy:
                x, xx, y, yy = xx, x, yy, y
            if (
                yy - y < max(14, h * 0.034)
                or abs(x - xx) > 0.30 * (yy - y)
                or x < 9
                or xx < 9
                or x > w - 10
                or xx > w - 10
            ):
                continue
            fy = np.linspace(y + 1, yy - 1, 9)
            fx = np.linspace(x, xx, 9)
            iy = fy.round().astype(int).clip(0, h - 1)
            ix = fx.round().astype(int)
            ox = np.arange(-8, 9)
            L = gray[iy[:, None], ix[:, None] + ox].astype(
                float
            )  # narrow ridge contrasted on both sides, not a facade/sky step
            choices = []
            for wd in [2, 4, 6]:
                for j in [-2, -1, 0, 1, 2]:
                    dif = L[:, 8 + j, None] - L[:, [8 + j - wd, 8 + j + wd]]
                    amp = np.min(abs(dif), 1) * ((dif[:, 0] * dif[:, 1]) > 0)
                    choices.append((np.median(amp) - wd * 0.4, j, wd))
            score, off, wd = max(choices)
            if score < 8:
                continue
            fx += off
            ix += off
            row = rows[iy, ix]
            if (row < 0).any() or (lab[iy, ix] == 0).mean() > 0.23:
                continue
            mat = M[row]
            u = unit(
                np.einsum(
                    "nij,nj->ni",
                    mat[:, :3, :3] @ R,
                    np.stack([(fx - w / 2) / f, (fy - h / 2) / f, np.ones(len(fx))], 1),
                )
            )
            bearing = unit(u[:, :2])
            az = np.arctan2(bearing[:, 1], bearing[:, 0])
            if np.ptp(np.unwrap(az)) > 2.8 / f:
                continue
            O = mat[4, :3, 3]
            D = unit(bearing.mean(0))
            slope = u[:, 2] / np.linalg.norm(u[:, :2], axis=1)
            col = np.median(rgb0[iy, ix], 0)
            sky = (lab[iy, ix - wd] == 2) | (lab[iy, ix + wd] == 2)
            obs.append(
                [
                    *O,
                    *D,
                    slope.min(),
                    slope.max(),
                    k,
                    e["t"],
                    f,
                    wd,
                    *col,
                    float(sky.mean()),
                    score,
                ]
            )
    if len(obs) < 2:
        np.save(W / "lines.npy", np.empty((0, 24), "f"))
        savej(W / "lines.json", dict(ridges=len(obs), proposals=0, shafts=[]))
        return
    a = np.array(obs)
    O = a[:, :2]
    D = a[:, 3:5]
    N = np.c_[-D[:, 1], D[:, 0]]
    order = np.argsort(a[:, 7])
    a, O, D, N = [x[order] for x in (a, O, D, N)]
    proposals = []
    k = a[:, 7]
    for i in range(len(a)):
        ii = np.arange(np.searchsorted(k, k[i] + 2.2), np.searchsorted(k, k[i] + 25))
        A = N[ii] @ D[i]
        B = ((O[ii] - O[i]) * N[ii]).sum(1)
        r = B / np.where(abs(A) > 0.025, A, 1)
        P = O[i] + D[i] * r[:, None]
        r2 = ((P - O[ii]) * D[ii]).sum(1)
        z0 = a[i, 2] + a[i, 5:7] * r[:, None]
        z1 = a[ii, 2, None] + a[ii, 5:7] * r2[:, None]
        ov = np.minimum(z0[:, 1], z1[:, 1]) - np.maximum(z0[:, 0], z1[:, 0])
        ok = (
            (abs(A) > 0.033)
            & (r > 3)
            & (r < 190)
            & (r2 > 3)
            & (r2 < 190)
            & (ov > 0.65)
            & (abs(a[i, 11:14] - a[ii, 11:14]).mean(1) < 42)
        )
        proposals.append(P[ok])
    P = np.concatenate(proposals)
    pitch = 0.42
    ij = np.floor(P / pitch).astype(int)
    un, ct = np.unique(ij, return_counts=True, axis=0)
    q = (un[ct >= 5] + 0.5) * pitch
    votes = ct[ct >= 5]
    q = q[np.argsort(-votes)]
    oTree = cKDTree(O)
    out = []
    records = []
    used = {}
    for xy in q:
        cell = tuple(np.floor(xy / 0.65).astype(int))
        if any(
            np.linalg.norm(xy - j) < 0.85
            for di in [-1, 0, 1]
            for dj in [-1, 0, 1]
            for j in used.get((cell[0] + di, cell[1] + dj), [])
        ):
            continue
        idx = np.array(oTree.query_ball_point(xy, 190), int)
        delta = xy - O[idx]
        r = (delta * D[idx]).sum(1)
        err = abs((delta * N[idx]).sum(1))
        sel = (r > 3) & (err < 0.34 + r * 0.0027)
        ii = idx[sel]
        for it in range(3):
            if len(ii) < 5:
                break
            rr = np.linalg.norm(xy - O[ii], axis=1)
            lo = a[ii, 2] + a[ii, 5] * rr
            hi = a[ii, 2] + a[ii, 6] * rr
            mid = (lo + hi) / 2
            anchor = np.median(mid)
            ok = (lo < anchor + 0.65) & (hi > anchor - 0.65)
            ii = ii[ok]
            if len(ii) < 5:
                break
            dist = np.linalg.norm(xy - O[ii], axis=1)
            ww = 1 / (0.065 + 0.0017 * dist) ** 2
            M = N[ii]
            xy = np.linalg.lstsq(
                M * np.sqrt(ww[:, None]), (M * O[ii]).sum(1) * np.sqrt(ww), rcond=None
            )[0]
            dd = xy - O[ii]
            err = abs((dd * M).sum(1))
            ii = ii[err < 0.055 + dist * 0.0019]
        if len(np.unique(a[ii, 8].round(1))) < 5 or np.ptp(k[ii]) < 5.6:
            continue
        dist = np.linalg.norm(xy - O[ii], axis=1)
        sv = np.linalg.svd(N[ii], compute_uv=False)
        if sv[-1] / sv[0] < 0.034:
            continue
        lo = a[ii, 2] + a[ii, 5] * dist
        hi = a[ii, 2] + a[ii, 6] * dist
        bottom = np.quantile(lo, 0.27)
        top = np.quantile(hi, 0.72)
        g = vol.lookup(np.r_[xy, 0.0][None])[0][0]
        bottom = max(bottom, g + 0.16)
        height = top - bottom
        if height < 1.05 or top > g + 18 or bottom > g + 8 or route.query(xy)[0] < 2.8:
            continue
        radius = np.clip(0.48 * np.median(a[ii, 10] * dist / a[ii, 9]), 0.044, 0.17)
        z = np.linspace(bottom, top, max(3, int(np.ceil(height / (radius * 1.65))) + 1))
        xyz = np.c_[np.repeat(xy[None], len(z), 0), z]
        col = np.median(a[ii, 11:14], 0)
        B = surfels(
            xyz, col, [radius, radius, radius * 1.75], 0.89, float(a[ii, 8].max())
        )
        out.append(B)
        records.append(
            dict(
                xy=xy.tolist(),
                lo=float(bottom),
                hi=float(top),
                distinct=int(len(np.unique(a[ii, 8].round(1)))),
                baseline=float(np.ptp(k[ii])),
                sky_side=float(a[ii, 14].mean()),
                radius=float(radius),
                time=float(np.median(a[ii, 8])),
            )
        )
        used.setdefault(tuple(np.floor(xy / 0.65).astype(int)), []).append(xy)
    np.save(W / "lines.npy", np.concatenate(out) if out else np.empty((0, 24), "f"))
    savej(W / "lines.json", dict(ridges=len(a), proposals=len(P), shafts=records))
    print(len(a), len(P), "confirmed uprights", len(records), flush=True)


if __name__ == "__main__":
    lines()
