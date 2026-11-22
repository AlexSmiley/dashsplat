"""Stream-decode every published texture; shared mesh edges and opacity invariants."""

import hashlib
from .proof import *


def verify():
    at = Atlas()
    grid = {}
    area = [0.0, 0.0, 0.0]
    weighted = {}
    counts = {}
    unique = set()
    edge = []
    for k, v in at.idx.items():
        p = OUT / v["texture"]
        b = p.read_bytes()
        assert len(b) == v["bytes"] and hashlib.sha256(b).hexdigest()[:12] in p.stem
        assert p.name not in unique
        unique.add(p.name)
        s = v["pixels"]
        g = v["gutter"]
        d = v["texelSize"]
        assert s * d == M
        expected = np.r_[np.array(k) * M, (np.array(k) + 1) * M]
        assert np.allclose(expected, v["bounds"], atol=1e-5, rtol=0)
        im = np.asarray(Image.open(p).convert("RGBA"))
        assert im.shape == (s + 2 * g, s + 2 * g, 4)
        alpha = im[g:-g, g:-g, 3]
        area[0] += float((alpha > 0).sum() * d * d)
        area[1] += float((alpha >= 250).sum() * d * d)
        area[2] += float(((alpha > 0) & (alpha < 250)).sum() * d * d)
        grid[k] = np.array(v["grid"]["z"]).reshape(9, 9)
        assert np.isfinite(grid[k]).all()
        counts[str(d)] = 1 + counts.get(str(d), 0)
        weighted[str(d)] = weighted.get(str(d), 0.0) + float(
            (alpha >= 250).sum() * d * d
        )
    for (x, y), z in grid.items():
        if (x + 1, y) in grid:
            edge.extend(abs(grid[x + 1, y][:, 0] - z[:, -1]))
        if (x, y + 1) in grid:
            edge.extend(abs(grid[x, y + 1][0, :] - z[-1, :]))
    assert max(edge, default=0) < 0.0001
    tr = inputs()[0]
    dist = np.r_[0, np.cumsum(np.linalg.norm(np.diff(tr.p[:, :2], axis=0), axis=1))]
    intervals = at.meta["intervals"]
    total = sum(np.diff(np.interp([lo, hi], tr.t, dist))[0] for lo, hi in intervals)
    bytes_ = (
        sum(t["bytes"] for t in at.idx.values()) + (OUT / "ground.json").stat().st_size
    )
    obs = read(W / "observations.json")
    assert len({(r["cam"], r["n"]) for r in obs}) == len(obs)
    stats = {}
    for cam in CAMS:
        rr = [r for r in obs if r["cam"] == cam]
        fit = [r["registration"] for r in rr if "after_cm" in r["registration"]]
        stats[cam] = dict(
            exposures=len(rr),
            registered=len(fit),
            before_cm=float(np.median([t["before_cm"] for t in fit])),
            after_cm=float(np.median([t["after_cm"] for t in fit])),
        )
    if any("pairfit" in r for r in obs):
        # The previous strip similarities are not the registration used by V5 front exposures.
        for cam in CAMS:
            rr = [r for r in obs if r["cam"] == cam]
            pairs = [r["pairfit"] for r in rr if r.get("pairfit", [0])[0] > 0]
            stats[cam] = dict(
                exposures=len(rr),
                registered=len(pairs),
                patch_matches=sum(p[0] for p in pairs),
                median_initial_ncc=float(np.median([p[1] for p in pairs])),
                median_aligned_ncc=float(np.median([p[2] for p in pairs])),
            )
    v = dict(
        tiles=len(unique),
        textures_bytes=bytes_ - (OUT / "ground.json").stat().st_size,
        manifest_bytes=(OUT / "ground.json").stat().st_size,
        total_bytes=bytes_,
        route_metres=total,
        MB_per_100m=bytes_ / 1e6 / (total / 100),
        surface_square_metres=dict(present=area[0], opaque=area[1], soft=area[2]),
        tiles_per_resolution=counts,
        opaque_area_by_resolution=weighted,
        shared_vertices=len(edge),
        max_mesh_edge_error_m=max(edge, default=0),
        sources=stats,
    )
    dump(RPT / "verified.json", v)
    print(v)


if __name__ == "__main__":
    verify()
