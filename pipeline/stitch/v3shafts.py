"""Validate triangulated uprights in real held-in images, including views not used to propose the ridge."""

from .v3terrain import *


def validate():
    raw = J(W / "lines.json")["shafts"]
    cand = []
    for l in sorted(raw, key=lambda L: -L["distinct"]):
        if (
            l["distinct"] < 7
            or (l["sky_side"] < 0.06 and l["distinct"] < 13)
            or l["hi"] - l["lo"] < 1.15
        ):
            continue
        if any(
            np.linalg.norm(np.subtract(l["xy"], m["xy"])) < 1.05
            and min(l["hi"], m["hi"]) > max(l["lo"], m["lo"]) - 0.5
            for m in cand
        ):
            continue
        cand.append(l)
    if not cand:
        np.save(W / "uprights.npy", np.empty((0, 24), "f"))
        savej(
            W / "uprights.json",
            dict(
                proposals=len(raw), dedup_candidates=0, images=0, accepted=0, records=[]
            ),
        )
        return
    heights = 40
    vol = Volume()
    floors = vol.lookup(np.array([l["xy"] for l in cand]))[0]
    xyz = np.array(
        [
            [(*l["xy"], z) for z in np.linspace(g + 0.09, l["hi"], heights)]
            for l, g in zip(cand, floors)
        ]
    )
    n = len(xyz)
    P = xyz.reshape(-1, 3)
    evidence = np.zeros((n, heights))
    seen = evidence.copy()
    colour = np.zeros((n, heights, 3))
    width = evidence.copy()
    resolution = evidence.copy()
    last = {}
    num = 0
    for e in J(V / "train.json"):
        c = e["c"]
        yaw = e["yaw"]
        key = (c, yaw)
        if e["odo"] - last.get(key, -100) < 2.4:
            continue
        last[key] = e["odo"]
        w, h, f, R, _, _, band, _ = pin(c, yaw)
        M = poses(c, e["t"])
        X = (P - M[4, :3, 3]) @ M[4, :3, :3] @ R
        z = X[:, 2].clip(0.01)
        x = (
            np.nan_to_num(w / 2 + f * X[:, 0] / z - 0.5)
            .round()
            .astype(int)
            .clip(0, w - 1)
        )
        y = (
            np.nan_to_num(h / 2 + f * X[:, 1] / z - 0.5)
            .round()
            .astype(int)
            .clip(0, h - 1)
        )
        T = M[band[y, x].clip(0, 7)]
        X = np.einsum("ni,nij->nj", P - T[:, :3, 3], T[:, :3, :3] @ R)
        z = X[:, 2].clip(0.01)
        x = (w / 2 + f * X[:, 0] / z - 0.5).round().astype(int)
        y = (h / 2 + f * X[:, 1] / z - 0.5).round().astype(int)
        ids = np.flatnonzero(
            (z > 3) & (z < 155) & (x >= 14) & (x < w - 14) & (y >= 2) & (y < h - 2)
        )
        if not len(ids):
            continue
        with np.load(e["file"]) as a:
            img = a["rgb"]
            lab = a["lab"]
            dep = a["z"]
            wt = a["wt"]
        x = x[ids]
        y = y[ids]
        z = z[ids]
        valid = lab[y, x] != 0
        if dep.size:
            valid &= ~((wt[y, x] > 20) & (dep[y, x] * 1.18 + 1.2 < z))
        aidx = ids // heights
        hh = ids % heights
        lum = cv.cvtColor(img, cv.COLOR_RGB2GRAY).astype(float)
        radius = np.array([l["radius"] for l in cand])[aidx]
        wd = np.clip(np.ceil(1.5 + 1.75 * f * radius / z), 2, 11).astype(int)
        v = np.stack([lum[y, x + q] for q in [-1, 0, 1]], 1)
        side = np.stack([lum[y, x - wd], lum[y, x + wd]], 1)
        diff = v[:, :, None] - side[:, None, :]
        contrast = abs(diff).min(2) * ((diff[:, :, 0] * diff[:, :, 1]) > 0)
        ii = contrast.argmax(1)
        amp = contrast.max(1)
        ix = x + ii - 1
        seen[aidx[valid], hh[valid]] += 1
        ok = valid & (amp > 9)
        evidence[aidx[ok], hh[ok]] += 1
        colour[aidx[ok], hh[ok]] += img[y[ok], ix[ok]]
        # Estimate metric width from actual half-contrast crossings, weighted toward the highest resolution views. The ridge search window is not the physical diameter.
        mid = lum[y, ix]
        background = (side[:, 0] + side[:, 1]) * 0.5
        diff = mid - background
        wp = np.ones(len(ids))
        for sign in [-1, 1]:
            still = np.ones(len(ids), bool)
            for step in range(1, 9):
                still &= (
                    lum[y, (ix + sign * step).clip(0, w - 1)] - background
                ) * diff > 0.48 * diff**2
                wp += still
        measurement = np.clip(wp * z / f / 2.85, 0.019, 0.098)
        reswt = np.clip((f / z / 22) ** 2, 0.15, 12)
        width[aidx[ok], hh[ok]] += measurement[ok] * reswt[ok]
        resolution[aidx[ok], hh[ok]] += reswt[ok]
        num += 1
    out = []
    records = []
    for i, l in enumerate(cand):
        frac = evidence[i] / np.maximum(1, seen[i])
        ok = (evidence[i] >= 5) & (frac > 0.51)
        idx = np.flatnonzero(ok)
        zs = xyz[i, :, 2]
        if l["distinct"] < 10 or len(idx) < 13 or np.median(frac[idx]) < 0.62:
            continue
        # Require a supported base and contiguous column: no dark midair fragments at the edge of a canopy.
        runs = np.split(
            idx, np.flatnonzero((np.diff(idx) > 3) | (np.diff(zs[idx]) > 0.8)) + 1
        )
        runs = [
            j
            for j in runs
            if len(j) >= 10
            and zs[j[0]] < floors[i] + 0.75
            and zs[j[-1]] - zs[j[0]] > 1.4
        ]
        if not runs:
            continue
        idx = max(runs, key=len)
        r = float(
            np.clip(
                width[i, idx].sum() / max(0.001, resolution[i, idx].sum()), 0.023, 0.081
            )
        )
        z = np.linspace(
            zs[idx[0]],
            zs[idx[-1]],
            max(3, int((zs[idx[-1]] - zs[idx[0]]) / (1.38 * r)) + 1),
        )
        C = colour[i, idx] / evidence[i, idx, None]
        color = np.stack([np.interp(z, zs[idx], C[:, j]) for j in range(3)], 1)
        B = surfels(
            np.c_[np.tile(l["xy"], (len(z), 1)), z],
            color,
            [r * 0.83, r * 0.83, 1.7 * r],
            0.91,
            l["time"],
        )
        out.append(B)
        records.append(
            dict(
                **l,
                observed_radius=r,
                bottom_clearance=float(z[0] - floors[i]),
                heights_confirmed=len(idx),
                photographs_median=float(np.median(evidence[i, idx])),
                agreement=float(np.median(evidence[i, idx] / seen[i, idx])),
            )
        )
    np.save(W / "uprights.npy", np.concatenate(out) if out else np.empty((0, 24), "f"))
    savej(
        W / "uprights.json",
        dict(
            proposals=len(raw),
            dedup_candidates=n,
            images=num,
            accepted=len(records),
            records=records,
        ),
    )
    print("photographic ridge validation", len(raw), n, len(records), flush=True)


if __name__ == "__main__":
    validate()
