"""Fixed physical holdout, decoded atlas, independent single-photo and injected-twin controls."""

from .proof import (
    Atlas,
    W,
    RPT,
    read,
    dump,
    table,
    top_xy,
    groundphoto,
    C,
    N,
    np,
    cv,
)
from pipeline.stitch.v4outline import edges, count
from functools import lru_cache
from collections import defaultdict
from .edge_locations import edge_locations


@lru_cache(20)
def state(k):
    return np.load(W / "state" / f"{k[0]}_{k[1]}.npy")


def owners(at, xy):
    flat = xy.reshape(-1, 2)
    ids = np.zeros(len(flat), "i")
    for k, ix in at.groups(flat):
        if k not in at.idx:
            continue
        loc = np.floor(flat[ix] / C).astype(int) - np.array(k) * N
        ids[ix] = state(k)["id"][loc[:, 1].clip(0, N - 1), loc[:, 0].clip(0, N - 1)]
    return ids.reshape(xy.shape[:2])


def main(lo=85, hi=140, fold=0):
    if fold not in [0, 1]:
        raise ValueError("fold must be 0 or 1")
    suffix = "-fresh" if fold else ""
    at = Atlas()
    obs = read(W / "observations.json")
    src = {r["n"] for r in obs if r["cam"] == "front"}
    tbl = table("front")
    by_n = {v["n"]: i for i, v in enumerate(tbl)}
    held = [r for r in tbl if r["n"] % 29 == 7 and lo + 1 <= r["t"] <= hi - 1]
    rows = []
    tiles = defaultdict(lambda: [0, 0])
    anchors = []
    for e in held[fold::2]:
        n = e["n"]
        assert n not in src
        alts = [
            v
            for v in tbl[max(0, by_n[n] - 4) : by_n[n] + 5]
            if v["n"] != n and v["n"] not in src
        ]
        if not alts:
            continue
        v = min(alts, key=lambda v: abs(v["t"] - e["t"]))
        if abs(v["t"] - e["t"]) > 0.15:
            continue
        xy, _, _ = top_xy(e["t"], 10.5, 0, 680, 800, 0.01)
        ref, valid = groundphoto(dict(cam="front", n=n, t=e["t"]), xy)
        solo, v2 = groundphoto(dict(cam="front", n=v["n"], t=v["t"]), xy)
        rgb, alpha = at.rgba(xy)
        out = rgb.clip(0, 255).astype("B")
        valid &= v2 & (alpha[:, :, 0] > 0.98)
        valid = cv.erode(valid.astype("B"), np.ones((9, 9), "B")).astype(bool)
        g = cv.cvtColor(ref, cv.COLOR_RGB2GRAY).astype("f")
        base = cv.GaussianBlur(g, (0, 0), 16)
        paint = cv.dilate(
            np.uint8((g - base) > 13),
            cv.getStructuringElement(cv.MORPH_ELLIPSE, (45, 45)),
        ).astype(bool)
        region = paint & valid
        rmask = cv.dilate(edges(ref, False)[0], np.ones((3, 3), "B"))
        idmap = owners(at, xy)
        seam = (idmap[:, 1:] != idmap[:, :-1]) & valid[:, 1:] & valid[:, :-1]
        low = (abs(g[:, 1:] - g[:, :-1]) < 4) & (~paint[:, 1:])
        delta = np.mean(abs(out[:, 1:].astype("f") - out[:, :-1]), axis=2)
        non = cv.dilate(seam.astype("B"), np.ones((11, 11), "B")).astype(bool) & ~seam
        seam &= low
        non &= low & valid[:, 1:] & valid[:, :-1]
        ey, ex, flag = edge_locations(out, rmask, region)
        loc = xy[ey, ex]
        keys = np.floor(loc / (C * N)).astype(int)
        unique, iv = np.unique(keys, axis=0, return_inverse=True)
        for key, total, hits in zip(
            unique,
            np.bincount(iv),
            np.bincount(iv, weights=flag, minlength=len(unique)),
        ):
            vtile = tiles[f"{key[0]}_{key[1]}"]
            vtile[0] += int(hits)
            vtile[1] += int(total)
        for xy0, source in zip(loc[flag], idmap[ey[flag], ex[flag]]):
            rr = obs[source - 1]
            anchors.append(
                [
                    round(float(xy0[0]), 3),
                    round(float(xy0[1]), 3),
                    n,
                    rr["cam"],
                    rr["n"],
                ]
            )
        rows.append(
            dict(
                n=n,
                t=e["t"],
                floor_n=v["n"],
                paint_samples=int(region.sum()),
                atlas=[int(flag.sum()), len(flag)],
                floor=count(solo, rmask, region),
                self=count(ref, rmask, region),
                injected=count(
                    cv.addWeighted(solo, 0.5, np.roll(solo, 9, 1), 0.5, 0),
                    rmask,
                    region,
                ),
                seam_samples=int(seam.sum()),
                seam_mean=float(delta[seam].mean()) if seam.any() else None,
                seam_p90=float(np.percentile(delta[seam], 90)) if seam.any() else None,
                nearby_mean=float(delta[non].mean()) if non.any() else None,
            )
        )
    agg = {
        k: np.sum([e[k] for e in rows], axis=0).tolist()
        for k in ["atlas", "floor", "self", "injected"]
    }
    dump(RPT / f"audit-{lo}-{hi}{suffix}.json", dict(rows=rows, totals=agg))
    dump(
        RPT / f"tiles-{lo}-{hi}{suffix}.json",
        dict(
            tile_size_m=C * N,
            fields=["suspected", "strong_edges"],
            tiles=dict(tiles),
            anchor_fields=["x", "y", "held_frame", "owner_camera", "owner_frame"],
            anchors=anchors,
        ),
    )
    print(agg, flush=True)


if __name__ == "__main__":
    import sys

    main(*map(int, sys.argv[1:]))
