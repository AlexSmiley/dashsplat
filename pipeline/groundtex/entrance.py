"""Conservatively replace rays occluded by the entrance building in the stationary front exposures."""

from .finish_v5 import *


def correct():
    at = Atlas()
    obs = read(W / "observations.json")
    canvas = Canvas()
    for p in (W / "state").glob("*.npy"):
        canvas.tiles[tuple(map(int, p.stem.split("_")))] = np.load(p)
    start = pose(85)[0]
    ks = [k for k in at.idx if np.linalg.norm((np.array(k) + 0.5) * M - start) < 48]
    allowed = np.array([False] + [r["cam"] == "front" and r["t"] < 91 for r in obs])
    repl = Counter()
    changed = set()
    for k in ks:
        a = canvas.tiles[k]
        idmap = a["id"]
        front = allowed[idmap] & (a["ok"] > 0)
        Y, X = np.mgrid[:N, :N]
        xy = (np.array(k) * N + np.stack([X, Y], -1) + 0.5) * C
        bad = np.zeros((N, N), "B")
        for id in np.unique(idmap[front]):
            r = obs[id - 1]
            cam = inputs()[1]["front"]
            m = masks("front", r["n"]).get("road")
            uv, ok, _ = ground_to_pixels(
                cam, inputs()[0], r["t"], corrected_xy(r, xy), 37
            )
            sem = (
                sam(m, uv.astype("f"), cam, True) > 0
                if m is not None
                else np.zeros_like(ok)
            )
            far = np.linalg.norm(xy - pose(r["t"])[0], axis=2) > 8
            bad |= (idmap == id) & front & far & ~(sem & ok)
        bad = cv.dilate(bad, np.ones((5, 5), "B")).astype(bool) & (a["ok"] > 0)
        repl["occluded_cells"] += int(bad.sum())
        if not bad.any():
            continue
        result = np.zeros_like(idmap)
        qual = np.zeros((N, N), "f")
        for id, r in enumerate(obs, 1):
            if (
                not 86 <= r["t"] <= 110
                or np.linalg.norm(pose(r["t"])[0] - (np.array(k) + 0.5) * M) > 40
            ):
                continue
            mm = masks(r["cam"], r["n"])
            if "road" not in mm or "remove" not in mm:
                continue
            cam = inputs()[1][r["cam"]]
            uv, ok, _ = ground_to_pixels(
                cam,
                inputs()[0],
                r["t"],
                corrected_xy(r, xy),
                25 if r["cam"] == "front" else 13,
            )
            uv = uv.astype("f")
            ok &= sam(mm["road"], uv, cam, True) > 0
            for name in ["remove", "dyn", "sky"]:
                if name in mm:
                    ok &= sam(mm[name], uv, cam, True) == 0
            score = np.clip(
                1 - np.linalg.norm(xy - pose(r["t"])[0], axis=2) / 30, 0, 1
            ) + 2 * (r["cam"] == "front")
            ok = cv.erode(ok.astype("B"), np.ones((5, 5), "B")).astype(bool)
            hit = bad & ok & (score > qual)
            result[hit] = id
            qual[hit] = score[hit]
        a["id"][bad] = result[bad]
        a["ok"][bad] = np.uint8(result[bad] != 0)
        repl["recovered_from_photographed_road"] += int((bad & (result != 0)).sum())
        repl["left_for_fallback"] += int((bad & (result == 0)).sum())
        np.save(W / "state" / f"{k[0]}_{k[1]}.npy", a)
        changed.update((k[0] + dx, k[1] + dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1])
    ks = sorted(changed & at.idx.keys())
    sub = Atlas()
    near = set(
        (k[0] + dx, k[1] + dy) for k in ks for dx in [-1, 0, 1] for dy in [-1, 0, 1]
    )
    sub.idx = {k: v for k, v in sub.idx.items() if k in near}
    gain = Verge(sub)
    items = dict(at.idx)
    with ThreadPoolExecutor(4) as ex:
        for k, t in zip(
            ks,
            ex.map(
                lambda k: export_tile(k, canvas, obs, softedge, actual_guard, gain), ks
            ),
        ):
            if t:
                items[k] = t
            else:
                items.pop(k, None)
    meta = at.meta
    meta["tiles"] = list(items.values())
    dump(OUT / "ground.json", meta)
    dump(RPT / "entrance.json", dict(tiles_reencoded=len(ks), **repl))
    print(repl, len(ks))


if __name__ == "__main__":
    correct()
