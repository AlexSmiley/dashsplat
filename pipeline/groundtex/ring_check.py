"""Verify both immutable planes, measure uncovered eligible area and class/edge transitions, and rerun paired paint holdouts."""

from .outer import *
from . import proof
from .proof import RPT, top_xy, groundphoto
from .edge_locations import edges, edge_locations
from collections import defaultdict


def compare():
    orig = proof.OUT

    class Decoded(proof.Atlas):
        def __init__(self, path):
            proof.OUT = path
            super().__init__()
            self.path = path

        @lru_cache(50)
        def data(self, k, l):
            a = np.asarray(
                Image.open(self.path / self.idx[k]["texture"]).convert("RGBA")
            ).copy()
            a[:, :, :3] = np.uint8(a[:, :, :3].astype("f") * a[:, :, 3:4] / 255 + 0.5)
            return a

    a, b = Decoded(B), Decoded(orig)
    proof.OUT = orig
    obs = read(S / "observations.json") + read(W / "outer-rows.json")
    src = {r["n"] for r in obs if r["cam"] == "front"}
    tbl = table("front")
    by = {r["n"]: i for i, r in enumerate(tbl)}
    data = {}
    for lo, hi in [(85, 140), (190, 215), (286, 330)]:
        for fold in [0, 1]:
            out = []
            tiles = defaultdict(lambda: [0, 0, 0, 0])
            held = [e for e in tbl if e["n"] % 29 == 7 and lo + 1 <= e["t"] <= hi - 1]
            for e in held[fold::2]:
                n = e["n"]
                assert n not in src
                alts = [
                    r
                    for r in tbl[max(0, by[n] - 4) : by[n] + 5]
                    if r["n"] not in src
                    and r["n"] != n
                    and abs(r["t"] - e["t"]) <= 0.15
                ]
                if not alts:
                    continue
                alt = min(alts, key=lambda r: abs(r["t"] - e["t"]))
                xy, _, _ = top_xy(e["t"], 10.5, 0, 680, 800, 0.01)
                ref, v = groundphoto(dict(cam="front", n=n, t=e["t"]), xy)
                solo, v2 = groundphoto(dict(cam="front", n=alt["n"], t=alt["t"]), xy)
                p = a.lookup(xy)
                q = b.lookup(xy)
                valid = cv.erode(
                    (v & v2 & (p[:, :, 3] >= 250) & (q[:, :, 3] >= 250)).astype("B"),
                    np.ones((9, 9), "B"),
                ).astype(bool)
                gray = cv.cvtColor(ref, cv.COLOR_RGB2GRAY).astype("f")
                paint = (
                    cv.dilate(
                        np.uint8(gray - cv.GaussianBlur(gray, (0, 0), 16) > 13),
                        cv.getStructuringElement(cv.MORPH_ELLIPSE, (45, 45)),
                    )
                    > 0
                )
                mask = cv.dilate(edges(ref, False)[0], np.ones((3, 3), "B"))
                d = {}
                for j, (label, rgb) in enumerate(
                    [("v5", p[:, :, :3]), ("v6", q[:, :, :3]), ("floor", solo)]
                ):
                    yy, xx, flags = edge_locations(rgb, mask, valid & paint)
                    d[label] = [int(flags.sum()), len(flags)]
                    if j < 2:
                        keys, indices = np.unique(
                            np.floor(xy[yy, xx] / M).astype("i"),
                            axis=0,
                            return_inverse=True,
                        )
                        for k, n1, n2 in zip(
                            keys,
                            np.bincount(indices),
                            np.bincount(indices, weights=flags, minlength=len(keys)),
                        ):
                            z = tiles[f"{k[0]}_{k[1]}"]
                            z[j * 2] += int(n2)
                            z[j * 2 + 1] += int(n1)
                out.append(dict(n=n, adjacent=alt["n"], **d))
            data[f"{lo}-{hi}-fold{fold}"] = dict(
                rows=out,
                tiles=tiles,
                totals={
                    k: np.sum([r[k] for r in out], axis=0).tolist()
                    for k in ["v5", "v6", "floor"]
                },
            )
            print(lo, hi, fold, data[f"{lo}-{hi}-fold{fold}"]["totals"], flush=True)
    dump(RPT / "paint.json", data)


def verify():
    p = OUT / "ground.json"
    m = read(p)
    assert m["provenance"]["values"] == SCHEM
    grids = {}
    total = Counter()
    areas = defaultdict(Counter)
    v5 = base()
    tr = inputs()[0]
    trees = [
        (f"{a}-{b}", cKDTree(tr.T_world_ego(np.arange(a, b, 0.15))[:, :2, 3]))
        for a, b in [(85, 140), (190, 215), (286, 330)]
    ]
    steps = Counter()
    edge_pairs = Counter()
    sides = {}
    edge = Counter()
    tree = height_model()[0]
    for t in m["tiles"]:
        k = tuple(map(int, t["id"].split("_")))
        assert k not in grids
        grids[k] = np.array(t["grid"]["z"]).reshape(9, 9)
        g = t["gutter"]
        s = t["pixels"]
        dx = t["texelSize"]
        assert abs(dx * s - M) < 1e-9 and np.isfinite(grids[k]).all()
        b, meta = [OUT / t[key] for key in ["texture", "provenance"]]
        for f, length in [(b, t["bytes"]), (meta, t["provenanceBytes"])]:
            raw = f.read_bytes()
            assert len(raw) == length and hashlib.sha256(raw).hexdigest()[:12] in f.stem
        im = np.asarray(Image.open(b).convert("RGBA"))
        pr = np.asarray(Image.open(meta))
        assert pr.ndim == 2 and pr.shape == im.shape[:2] == (s + 2 * g, s + 2 * g)
        assert np.all(np.isin(pr, [0, 127, 255])) and np.array_equal(
            pr == 0, im[:, :, 3] == 0
        )
        pr = pr[g : g + s, g : g + s]
        im = im[g : g + s, g : g + s]
        for name in ["full"] + [
            q for q, tree in trees if tree.query((np.array(k) + 0.5) * M)[0] < 47.5
        ]:
            for val, field in [(255, "observed_m2"), (127, "predicted_m2")]:
                areas[name][field] += int((pr == val).sum()) * dx * dx
        total["textures_bytes"] += t["bytes"]
        total["provenance_bytes"] += t["provenanceBytes"]
        total[f"{dx:g}_m_tiles"] += 1
        if k in v5:
            t0 = v5[k]
            a0 = np.asarray(Image.open(B / t0["texture"]).convert("RGBA"))[
                g : g + s, g : g + s, 3
            ]
            core = a0 >= 128
            rgb = im[:, :, :3].astype("i2")
            val = im[:, :, 3] > 240
            for axis in [0, 1]:
                step = np.abs(np.diff(rgb, axis=axis)).mean(2)
                lo = np.s_[:-1, :] if axis == 0 else np.s_[:, :-1]
                hi = np.s_[1:, :] if axis == 0 else np.s_[:, 1:]
                e = (core[lo] != core[hi]) & val[lo] & val[hi]
                near = (
                    cv.dilate(e.astype("B"), np.ones((5, 5), "B")).astype(bool)
                    & ~e
                    & val[lo]
                    & val[hi]
                )
                steps["native_boundary_rgb_sum"] += float(step[e].sum())
                steps["native_boundary_samples"] += int(e.sum())
                steps["adjacent_rgb_sum"] += float(step[near].sum())
                steps["adjacent_samples"] += int(near.sum())
        if tree.query((np.array(k) + 0.5) * M)[0] > 37:
            idx = np.arange(0, s, max(1, s // 64))
            yy, xx = np.meshgrid(idx, idx, indexing="ij")
            physical = tree.query(
                np.array(k) * M + (np.stack([xx, yy], -1) + 0.5) * dx
            )[0]
            outside = physical > 44.2
            edge["outside_samples"] += int(outside.sum())
            edge["outside_covered"] += int((outside & (im[yy, xx, 3] > 0)).sum())
        small = cv.resize(im, (U, U), interpolation=cv.INTER_AREA).astype("f")
        sides[k] = [small[0], small[-1], small[:, 0], small[:, -1]]
    diff = []
    for (i, j), z in grids.items():
        for k, ours, other, u, v in [
            ((i + 1, j), z[:, -1], (slice(None), 0), 3, 2),
            ((i, j + 1), z[-1, :], (0, slice(None)), 1, 0),
        ]:
            if k not in grids:
                continue
            diff.extend(abs(ours - grids[k][other]))
            a = sides[(i, j)][u]
            b = sides[k][v]
            ok = (a[:, 3] > 245) & (b[:, 3] > 245)
            edge_pairs["sum"] += float(np.abs(a[ok, :3] - b[ok, :3]).mean(1).sum())
            edge_pairs["samples"] += int(ok.sum())
    assert edge["outside_samples"] > 0 and edge["outside_covered"] == 0
    assert max(diff, default=0) < 0.0001
    by = read(W / "tile-areas.json")
    stats = {key: sum(t[key] for t in by) for key in by[0] if key != "id"}
    assert stats["unexplained_core40_m2"] < 0.001
    v = dict(
        tiles=len(grids),
        manifest_bytes=p.stat().st_size,
        total_bytes=p.stat().st_size
        + total["textures_bytes"]
        + total["provenance_bytes"],
        shared_height_checks=len(diff),
        max_height_delta_m=max(diff),
        independently_sampled_outer_edge=edge,
        areas=areas,
        coverage=stats,
        tone_steps=steps,
        shared_tile_colour=edge_pairs,
        **total,
    )
    dump(RPT / "verified.json", v)
    print(v, flush=True)


if __name__ == "__main__":
    import sys

    verify() if "verify" in sys.argv else compare()
