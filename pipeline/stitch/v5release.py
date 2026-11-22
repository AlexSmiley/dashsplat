"""Vet fitted columns along their full height; replace conflicting geometry, and publish disjoint spatial slices."""

from .v5structures import *
from collections import Counter


def validated():
    inv = J(W5 / "inventory.json")
    rr = inv["records"]
    thin = np.load(W5 / "thin.npy")
    pts = []
    ranges = []
    classes = []
    for r in rr:
        lo = len(pts)
        k = next(k for k, v in CATS.items() if v == r["class"])
        h0 = max(r["floor"], r["bottom"])
        h0 = r["floor"] + 0.055 if r["status"] == "base-to-observed-top" else h0
        hi = r["top"]
        z = np.linspace(h0, hi, 17)
        pts.extend(np.c_[np.tile(r["xy"], (len(z), 1)), z])
        classes.extend([k] * len(z))
        ranges.append(slice(lo, lo + len(z)))
    if not pts:
        return thin
    P = np.array(pts)
    cls = np.array(classes)
    yes = np.zeros(len(P))
    no = yes.copy()
    cams = Counter()
    for c in CAMS:
        for e in semframes(c):
            uv, ok, d = project(c, e, P)
            C = rig[c]
            x = (uv[:, 0] * 724 / C.width).astype(int).clip(0, 723)
            y = (uv[:, 1] * 469 / C.height).astype(int).clip(0, 468)
            s = labels(c, e["n"])
            ds = depth(e)[y, x]
            visible = ok & (d < 80) & (d > 1.5) & ((ds < 1) | (ds * 1.26 + 1.6 > d))
            for k in CATS:
                match = cv.dilate((s == k).astype("B"), np.ones((3, 3), "B"))[y, x] > 0
                yes += visible & (cls == k) & match
                no += (
                    visible
                    & (cls == k)
                    & ~match
                    & np.isin(s[y, x], [0, 1, 2, 5, 6, 7, 9, 10])
                )
            cams[c] += 1
    keep = []
    tree = cKDTree(thin[:, :2])
    done = []
    for i, r in sorted(enumerate(rr), key=lambda p: -p[1]["views"]):
        sl = ranges[i]
        fraction = yes[sl] / np.maximum(1, yes[sl] + no[sl])
        good = (yes[sl] >= 3) & (fraction > 0.29)
        r["supported_height_bins"] = int(good.sum())
        r["checked_height_bins"] = len(good)
        r["supporting_photos_median"] = float(np.median(yes[sl]))
        r["status_fit"] = r["status"]
        overlap = next(
            (
                s
                for s in done
                if s["class"] == r["class"]
                and np.linalg.norm(np.subtract(r["xy"], s["xy"])) < 0.43
                and min(s["top"], r["top"]) >= max(s["bottom"], r["bottom"]) - 0.3
            ),
            None,
        )
        if overlap:
            r["status"] = "duplicate proposal"
            r["same_as"] = overlap["id"]
            continue
        if not r["geometry"] or sum(good) < 7:
            r["status"] = "unverified"
            continue
        done.append(r)
        idx = np.array(
            tree.query_ball_point(r["xy"], 0.012 if r["class"] == CATS[5] else 1.6), int
        )
        z = thin[idx, 2]
        tm = thin[idx, 19]
        idx = idx[
            (abs(tm - r["time"]) < 0.002)
            & (z > min(r["bottom"], r["floor"]) - 0.12)
            & (z < r["top"] + 0.12)
        ]
        if not len(idx):
            r["status"] = "unverified"
            continue
        r["installed"] = len(idx)
        keep.extend(idx)
        r["status"] = "partial"
        if (
            r["class"] == CATS[5]
            and r["status_fit"] == "base-to-observed-top"
            and good.sum() >= 13
            and good[:3].any()
            and good[-3:].any()
        ):
            r["status"] = "continuous shaft"
    for r in done:
        if (
            r["class"] != CATS[5]
            and r.get("installed")
            and r["supported_height_bins"] >= 12
        ):
            mounts = [
                q["id"]
                for q in done
                if q["status"] == "continuous shaft"
                and np.linalg.norm(np.subtract(q["xy"], r["xy"])) < 0.48
                and q["top"] >= r["bottom"] - 0.3
            ]
            r["mounts"] = mounts
            r["status"] = (
                "panel + continuous shaft" if mounts else "panel; mount unverified"
            )
    inv["counts"] = dict(Counter(r["status"] for r in rr))
    inv["tested_frames"] = dict(cams)
    savej(W5 / "catalogue.json", inv)
    out = thin[np.unique(keep).astype(int)]
    if (
        BENCHMARK
    ):  # Keep triangulated photographic shafts excluded by the panoptic classifier; match before replacement.
        protected = np.load(RS / "work/stitch-v3/uprights.npy")
        if (W5 / "arms.npy").exists() and len(out):
            probe = cKDTree(out[:, :3])
            arms = np.load(W5 / "arms.npy")
            checked = []
            offset = 0
            for a in J(W5 / "arms.json")["arms"]:
                size = len(np.arange(0, a["length_m"] + 0.01, 0.07))
                item = arms[offset : offset + size]
                offset += size
                a["installed"] = bool(probe.query(a["start"])[0] < 0.21)
                checked.append(a)
                if a["installed"]:
                    out = np.concatenate([out, item])
            assert offset == len(arms)
            inv["arms"] = checked
        out = np.concatenate([out, protected])
        inv["other_ridge_tracks"] = J(RS / "work/stitch-v3/uprights.json")["records"]
        savej(W5 / "catalogue.json", inv)
    return out


def install(src):
    thin = validated()
    walls = np.load(W5 / "walls.npy")
    offices = np.empty((0, 24), "f")
    if BENCHMARK:
        offices = np.load(W5 / "long-planes.npy")
        if len(offices):
            # The long-baseline solution supersedes local coplanar hints; do not stack two sets of windows.
            close = (
                cKDTree(offices[:, :3]).query(walls[:, :3], distance_upper_bound=0.75)[
                    0
                ]
                < 0.75
            )
            walls = walls[~close]
    extras = [(thin, "columns"), (walls, "facade"), (offices, "office-plane")]
    trees = [cKDTree(b[:, :3]) if len(b) else None for b, k in extras]
    parts = []
    rm = [0, 0, 0]
    for t in J(src / "map.json")["tiles"]:
        if t["layer"] != "objects":
            continue
        A = reader(src / t["file"])
        keep = np.ones(len(A), bool)
        for j, ((B, k), tree) in enumerate(zip(extras, trees)):
            if tree is None:
                continue
            cap = 0.40 if j else 0.18
            d, ix = tree.query(A[:, :3], distance_upper_bound=cap)
            ii = np.flatnonzero(d < cap)
            ix = ix[ii]
            delta = A[ii, :3] - B[ix, :3]
            match = (
                (
                    np.linalg.norm(delta[:, :2], axis=1)
                    < np.maximum(0.055, 1.15 * np.exp(B[ix, 10]))
                )
                if not j
                else (
                    (abs((delta * B[ix, 3:6]).sum(1)) < 0.30)
                    & (np.linalg.norm(delta, axis=1) < np.exp(B[ix, 10]) * 2.9)
                )
            )
            keep[ii[match]] = False
            rm[j] += int(match.sum())
        parts.append((t, A[keep]))
    for B, k in extras:
        if len(B):
            parts.extend(group(B, k))
    assemble(W5 / "final", src, parts)
    savej(
        W5 / "installed.json",
        dict(thin=len(thin), wall=len(walls), office=len(offices), old_removed=rm),
    )
    print(J(W5 / "installed.json"), flush=True)


def merge(dest, dirs):
    dest.mkdir(parents=True, exist_ok=False)
    (dest / "tiles").mkdir()
    seen = {}
    tiles = []
    ss = []
    for src in dirs:
        m = J(src / "map.json")
        ss.append(m["slice_s"])
        for t in m["tiles"]:
            if t["id"] in seen:
                if t["layer"] == "ground" and os.path.samefile(
                    src / t["file"], seen[t["id"]]
                ):
                    continue
                if t["layer"] == "ground" and np.allclose(
                    t["bbox"], next(r["bbox"] for r in tiles if r["id"] == t["id"])
                ):
                    continue
                raise ValueError(("overlapping tile", t["id"]))
            seen[t["id"]] = src / t["file"]
            os.link(src / t["file"], dest / t["file"])
            tiles.append(t)
    for f in ["sky.jpg", "rig.json", "drive.json"]:
        shutil.copy2(dirs[0] / f, dest / f)
    m = J(dirs[0] / "map.json")
    m.pop("slice_s", None)
    savej(dest / "map.json", dict(m, tiles=tiles, slices=ss))
    print("merged", ss, len(tiles), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "merge":
        merge(Path(sys.argv[2]), [Path(s) for s in sys.argv[3:]])
    else:
        install(Path(sys.argv[1]))
