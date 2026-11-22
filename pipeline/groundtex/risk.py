"""Temporal exclusion over the two stationary-car footprints. Re-export only affected tiles and their gutter neighbours."""

from .v5 import *
from .proof import RPT, Atlas, label, save
from pipeline.road.geometry import pixels_to_ground

RISK = [(198, 206, 204.37, "front"), (286, 292, 288.87, "left_pillar")]


def blocked(obs, lo, hi):
    g = Grid(1, "B")
    count = 0
    for r in obs:
        if not lo <= r["t"] <= hi:
            continue
        m = masks(r["cam"], r["n"]).get("remove")
        if m is None or not m.any():
            continue
        cam = inputs()[1][r["cam"]]
        bb, eye, ran = box8(r)
        x, y, w, h = bb
        Y, X = np.mgrid[:h, :w]
        xy = (np.stack([X + x, Y + y], -1) + 0.5) * D8
        uv, ok, _ = ground_to_pixels(
            cam, inputs()[0], r["t"], corrected_xy(r, xy), min(28, ran)
        )
        hit = ok & (sam(m, uv.astype("f"), cam, True) > 0)
        hit = cv.dilate(
            hit.astype("B"), cv.getStructuringElement(cv.MORPH_ELLIPSE, (9, 9))
        )
        v = g.read(bb)
        v[:, :, 0] |= hit
        g.store(bb, v)
        count += 1
        if lo == 198:
            # The stop-line traffic is 34–65 m away. A short front texture footprint is not an audit of those cars.
            ys, xs = np.nonzero(m)
            sel = np.arange(0, len(xs), 4)
            uvp = (np.stack([xs[sel], ys[sel]], 1) + 0.5) * [
                cam.width / m.shape[1],
                cam.height / m.shape[0],
            ]
            xyz, good = pixels_to_ground(cam, inputs()[0], r["t"], uvp, 80)
            cells = np.floor(xyz[good, :2] / D8).astype(int)
            if len(cells):
                b = cells.min(0) - 5
                sz = cells.max(0) - b + 6
                bb = tuple(map(int, [*b, *sz]))
                v = g.read(bb)
                loc = cells - b
                v[loc[:, 1], loc[:, 0], 0] = 1
                v[:, :, 0] = cv.dilate(
                    v[:, :, 0], cv.getStructuringElement(cv.MORPH_ELLIPSE, (9, 9))
                )
                g.store(bb, v)
    return {k: v[:, :, 0].astype(bool) for k, v in g.a.items() if v.any()}, count


def switch():
    obs = read(W / "observations.json")
    canvas = Canvas()
    meta = read(OUT / "ground.json")
    tiles = {tuple(map(int, t["id"].split("_"))): t for t in meta["tiles"]}
    affected = set()
    summary = {}
    for f in (W / "state").glob("*.npy"):
        canvas.tiles[tuple(map(int, f.stem.split("_")))] = np.load(f)
    for lo, hi, _, _ in RISK:
        block, n = blocked(obs, lo, hi)
        ids = Grid(1, "i4")
        scores = Grid(1)
        regions = Grid(1, "B")
        cnt = Counter()
        for k, v in block.items():
            regions.a[k] = v.astype("B")[:, :, None]
        for id, r in enumerate(obs, 1):
            if (
                lo <= r["t"] <= hi
                or "remove" not in masks_fast(r)
                or min(abs(r["t"] - lo), abs(r["t"] - hi)) > 110
            ):
                continue
            bb, eye, ran = box8(r)
            reg = regions.read(bb)[:, :, 0].astype(bool)
            if not reg.any():
                continue
            bb, rgb, q, sem = patch(key(r))
            s = scores.read(bb)[:, :, 0]
            rr = ids.read(bb)[:, :, 0]
            qual = np.where(q > 0, q + 5 * (r["cam"] == "front"), 0)
            win = reg & (qual > s)
            s[win] = qual[win]
            rr[win] = id
            scores.store(bb, s[:, :, None])
            ids.store(bb, rr[:, :, None])
        for k, v in block.items():
            if k not in canvas.tiles:
                continue
            a = canvas.tiles[k]
            reg = np.repeat(np.repeat(v, 2, 0), 2, 1) & (a["ok"] > 0)
            old = a["id"]
            chosen = (
                np.zeros_like(old)
                if k not in ids.a
                else np.repeat(np.repeat(ids.a[k][:, :, 0], 2, 0), 2, 1)
            )
            time = np.array([0] + [r["t"] for r in obs])
            need = reg & (time[old] >= lo) & (time[old] <= hi)
            old[need] = chosen[need]
            a["ok"][need & (chosen == 0)] = 0
            cnt["footprint_4cm_cells"] += int(reg.sum())
            cnt["already_from_other_times"] += int((reg & ~need).sum())
            cnt["replaced_with_other_times"] += int((need & (chosen != 0)).sum())
            cnt["no_outside_photo_left_transparent"] += int(
                (need & (chosen == 0)).sum()
            )
            cnt["remaining_owners_in_risk_window"] += int(
                (reg & (time[old] >= lo) & (time[old] <= hi) & (old > 0)).sum()
            )
            if need.any():
                np.save(W / "state" / f"{k[0]}_{k[1]}.npy", a)
                affected.update(
                    (k[0] + dx, k[1] + dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1]
                )
        summary[f"{lo}-{hi}"] = dict(nonempty_source_masks=n, **cnt)
    with ThreadPoolExecutor(6) as ex:
        ks = sorted(affected & tiles.keys())
        v = list(
            ex.map(lambda k: export_tile(k, canvas, obs, softedge, actual_guard), ks)
        )
    for k, t in zip(ks, v):
        if t:
            tiles[k] = t
        else:
            tiles.pop(k, None)
    meta["tiles"] = list(tiles.values())
    dump(OUT / "ground.json", meta)
    dump(RPT / "risk-ownership.json", summary)
    print(summary, flush=True)


def masks_fast(r):
    return ["remove"] if r["n"] in files(r["cam"], "remove") else []


def plates():
    at = Atlas()
    obs = read(W / "observations.json")
    for lo, hi, t, camname in RISK:
        row = min(
            [r for r in obs if r["cam"] == camname], key=lambda r: abs(t - r["t"])
        )
        cam = inputs()[1][camname]
        m = masks(camname, row["n"])["remove"]
        cc, _, st, _ = cv.connectedComponentsWithStats(m, 8)
        x, y, w, h, _ = st[1 + np.argmax(st[1:, 4])]
        sx, sy = cam.width / m.shape[1], cam.height / m.shape[0]
        x, y, w, h = int(x * sx), int(y * sy), int(w * sx), int(h * sy)
        cx, cy = x + w / 2, y + h / 2
        ratio = max(w / 430, h / 270, 0.7)
        xx, yy = np.meshgrid(np.arange(500) - 250, np.arange(330) - 165)
        uv = np.stack([cx + xx * ratio, cy + yy * ratio], -1).astype("f")
        xyz, valid = pixels_to_ground(
            cam, inputs()[0], row["t"], uv, 80 if lo == 198 else 32
        )
        pic = photo(row, uv, False)[:, :, ::-1]
        im, a = at.rgba(np.nan_to_num(xyz[:, :, :2]))
        view = (
            np.where(valid[:, :, None], im + (1 - a) * 28, 28).clip(0, 255).astype("B")
        )
        mask = sam(m, uv, cam, True)
        edge = cv.morphologyEx(mask, cv.MORPH_GRADIENT, np.ones((3, 3), "B")) > 0
        pic = pic.copy()
        pic[edge] = [240, 190, 64]
        save(
            f"risk-{lo}-{hi}.jpg",
            np.hstack(
                [
                    label(pic, f"{camname} {row['n']}: removal mask outline"),
                    label(view, "Atlas in same camera: actual road or transparent"),
                ]
            ),
        )


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        switch()
    plates()
