"""Use clean-source footprint evidence, independent training-photo registration and a low-frequency verge gain. No photo mixtures."""

from .v5 import *
from .proof import Atlas, RPT, top_xy, groundphoto, pose
from . import refine

BASELINE = Path("work/groundtex-v5/before")
refine.GUIDE = Path("scene/stitched-opt-v5")


def high(im):
    g = cv.cvtColor(im.astype("f"), cv.COLOR_RGB2GRAY)
    return blur(g, 0.8) - blur(g, 5)


def local_repair(canvas, obs, zz, refpair):
    # Select a complete photograph, not the one with the highest nominal Jacobian through a blind spot.
    cell = 0.04
    xy, c, yaw = top_xy(
        zz["t"], zz["along"], zz["side"], zz["w"] // 4, zz["h"] // 4, cell
    )
    refrow = next(r for r in obs if (r["cam"], r["n"]) == refpair)
    rr, _ = groundphoto(refrow, xy)
    ref = high(rr)
    ax = np.stack([(xy[0, 1] - xy[0, 0]) / cell, (xy[1, 0] - xy[0, 0]) / cell], 1)
    half = np.array([zz["w"], zz["h"]]) * 0.005
    pad = 8
    rank = []
    front_seen = []
    for id, r in enumerate(obs, 1):
        cam = inputs()[1][r["cam"]]
        dist = np.linalg.norm(c - pose(r["t"])[0])
        if (
            dist > 37
            or abs(r["t"] - zz["t"]) > 14
            or r["n"] not in files(r["cam"], "remove")
        ):
            continue
        uv, valid, _ = ground_to_pixels(
            cam, inputs()[0], r["t"], corrected_xy(r, xy), 37
        )
        mm = masks(r["cam"], r["n"])
        for t in ["remove", "dyn", "sky"]:
            if t in mm:
                valid &= sam(mm[t], uv.astype("f"), cam, True) == 0
        if valid.mean() < 0.85:
            continue
        src, _ = groundphoto(r, xy)
        im = high(src)
        tpl = ref[pad:-pad, pad:-pad]
        cc = cv.matchTemplate(im, tpl, cv.TM_CCOEFF_NORMED)
        Y, X = np.mgrid[-pad : pad + 1, -pad : pad + 1]
        idx = np.unravel_index(np.argmax(cc - 0.025 * (X**2 + Y**2) / pad**2), cc.shape)
        score = float(cc[idx])
        shift = (np.array(idx[::-1]) - pad) * cell
        frac = valid.mean()
        dx = np.gradient(uv, axis=1) / cell
        dy = np.gradient(uv, axis=0) / cell
        a = (dx**2).sum(2)
        b = (dy**2).sum(2)
        o = (dx * dy).sum(2)
        smin = float(
            np.median(
                np.sqrt(np.maximum(0, (a + b - np.hypot(a - b, 2 * o)) * 0.5))[valid]
            )
        )
        if r["cam"] == "front":
            front_seen.append(score)
        if score < 0.60 or frac < 0.945:
            continue
        rank.append(
            (
                int(r["cam"] == "front"),
                score * min(100, smin) ** 0.35,
                id,
                shift,
                score,
                smin,
            )
        )
    if not rank:
        raise RuntimeError(zz["tag"] + " has no clean complete photo")
    _, _, id, shift, ncc, res = max(rank, key=lambda c: c[:2])
    r = obs[id - 1]
    d = ax @ shift
    r.setdefault("patch_shifts", []).append(
        [c.tolist(), ax.tolist(), (half + 0.64).tolist(), d.tolist()]
    )
    used = removed = recovered = 0
    for k, a in canvas.tiles.items():
        o = np.array(k) * M
        if np.linalg.norm(o + M / 2 - c) > np.linalg.norm(half) + 9:
            continue
        Y, X = np.mgrid[:N, :N]
        loc = o + (np.stack([X, Y], -1) + 0.5) * C
        dist = np.max((abs((loc - c) @ ax) - half) / 0.9, axis=2)
        core = dist < 0.65
        cam = inputs()[1][r["cam"]]
        uv, valid, _ = ground_to_pixels(
            cam, inputs()[0], r["t"], corrected_xy(r, loc), 37
        )
        mm = masks(r["cam"], r["n"])
        sem = (
            sam(mm["road"], uv.astype("f"), cam, True) > 0
            if "road" in mm
            else np.zeros((N, N), bool)
        )
        v = valid.copy()
        for t in ["remove", "dyn", "sky"]:
            if t in mm:
                v &= (
                    cv.dilate(
                        sam(mm[t], uv.astype("f"), cam, True), np.ones((3, 3), "B")
                    )
                    == 0
                )
        # Native road/kerb segmentation sets the grass edge; in the median preserve both verified painted borders. At an unmeasured edge do not smear paint into grass.
        if zz["tag"] == "median":
            support = (a["ok"] > 0) | sem
        else:
            support = cv.morphologyEx(
                sem.astype("B"),
                cv.MORPH_CLOSE,
                cv.getStructuringElement(cv.MORPH_ELLIPSE, (9, 9)),
            ).astype(bool)
        sel = core & v & support
        off = core & v & ~support
        used += int(sel.sum())
        recovered += int((sel & (a["ok"] == 0)).sum())
        removed += int((off & (a["ok"] > 0)).sum())
        a["ok"][sel] = 1
        a["id"][sel] = id
        a["ok"][off] = 0
        a["id"][off] = 0
    return dict(
        cam=r["cam"],
        frame=r["n"],
        front_candidates=len(front_seen),
        best_front_ncc=max(front_seen, default=0),
        selected_ncc=ncc,
        source_worst_axis_px_per_m=res,
        shift_m=d.tolist(),
        assigned_4cm_cells=used,
        recovered=recovered,
        cut=removed,
    )


def fieldfit(at, obs):
    # Baseline audit locates tiles; *all* their pavement windows and only training photographs determine corrections.
    hot = set()
    for lo, hi in SLICES:
        for k, (h, n) in read(BASELINE / f"tiles-{lo}-{hi}.json")["tiles"].items():
            if h >= 4:
                hot.add(tuple(map(int, k.split("_"))))
    samples = []
    fobs = [r for r in obs if r["cam"] == "front"]
    world = np.array([pose(r["t"])[0] for r in fobs])
    tree = cKDTree(world)
    T0 = 0.04
    patchRad = 12
    search = 5
    win = patchRad + search
    registered = 0
    fitrows = []
    for k in sorted(hot):
        pos = (np.array(k) + 0.5) * M
        s = int(M / T0)
        iy, ix = np.mgrid[-win : s + win, -win : s + win]
        xy = np.array(k) * M + (np.stack([ix, iy], -1) + 0.5) * T0
        rgba = at.lookup(xy)
        atlas = high(rgba[:, :, :3])
        valid = rgba[:, :, 3] >= 250
        vectors = defaultdict(list)
        for i in tree.query_ball_point(pos, 22):
            r = fobs[i]
            cam = inputs()[1]["front"]
            p, f, side = pose(r["t"])
            along = (xy - p) @ f
            rs = dict(cam="front", n=r["n"], t=r["t"])
            uv, validuv, _ = ground_to_pixels(cam, inputs()[0], r["t"], xy, 19)
            nom = photo(rs, uv.astype("f"), False)[:, :, ::-1]
            ok = validuv & valid & (along > 6.5) & (along < 13.5)
            mm = masks("front", r["n"])
            for z in ["remove", "dyn", "sky"]:
                if z in mm:
                    ok &= sam(mm[z], uv.astype("f"), cam, True) == 0
            ref = high(nom)
            gx = cv.Sobel(ref, cv.CV_32F, 1, 0)
            gy = cv.Sobel(ref, cv.CV_32F, 0, 1)
            for j in range(win + 8, s + win, 16):
                for l in range(win + 8, s + win, 16):
                    sl = np.s_[
                        j - patchRad : j + patchRad + 1, l - patchRad : l + patchRad + 1
                    ]
                    tpl = ref[sl]
                    if ok[sl].mean() < 0.97 or tpl.std() < 2.5:
                        continue
                    cc = cv.matchTemplate(
                        atlas[j - win : j + win + 1, l - win : l + win + 1],
                        tpl,
                        cv.TM_CCOEFF_NORMED,
                    )
                    yy, xx = np.mgrid[-search : search + 1, -search : search + 1]
                    v, u = np.unravel_index(
                        np.argmax(cc - 0.02 * (xx**2 + yy**2) / search**2), cc.shape
                    )
                    best = cc[v, u]
                    if best < 0.77 or min(u, v) <= 0 or max(u, v) >= 2 * search:
                        continue

                    def frac(a, b, c):
                        return np.clip(
                            0.5 * (a - c) / min(-0.00001, a - 2 * b + c), -0.5, 0.5
                        )

                    dx = (
                        np.array(
                            [
                                u - search + frac(cc[v, u - 1], best, cc[v, u + 1]),
                                v - search + frac(cc[v - 1, u], best, cc[v + 1, u]),
                            ]
                        )
                        * T0
                    )
                    sx, sy = gx[sl], gy[sl]
                    vals, V = np.linalg.eigh(
                        [
                            [np.mean(sx * sx), np.mean(sx * sy)],
                            [np.mean(sx * sy), np.mean(sy * sy)],
                        ]
                    )
                    if vals[0] < 0.18 * vals[1]:
                        dx = V[:, -1] * (V[:, -1] @ dx)
                    vectors[((l - win - 8) // 16, (j - win - 8) // 16)].append(dx)
                    registered += 1
        for (x, y), vec in vectors.items():
            if len(vec) < 3:
                continue
            dx = np.median(vec, axis=0)
            disp = np.median(np.linalg.norm(vec - dx, axis=1))
            if disp > 0.07:
                continue
            samples.append((np.array(k) * 16 + [x, y], dx, min(len(vec), 7)))
            fitrows.append(float(np.linalg.norm(dx)))
    if not samples:
        raise RuntimeError("No supported front training correlations")
    pts = np.array([s[0] for s in samples])
    base = pts.min(0) - 5
    sz = pts.max(0) - base + 6
    v = np.zeros((sz[1], sz[0], 3), "f")
    for p, d, w in samples:
        x, y = p - base
        v[y, x] = [*d * w, w]
    v = blur(v, 1.15)
    delta = v[:, :, :2] / (v[:, :, 2:3] + 0.42)
    name = W / "world-registration.npz"
    np.savez_compressed(
        name, origin=base * 0.64 + 0.34, delta=delta.astype("f2"), pitch=0.64
    )
    for r in obs:
        r["worldfit"] = str(name)
    return dict(
        flagged_tiles=len(hot),
        training_patch_matches=registered,
        consensus_sites=len(samples),
        median_correction_cm=float(np.median(fitrows) * 100),
        max_correction_cm=float(np.max(np.linalg.norm(delta, axis=2)) * 100),
    )


class Verge:
    def __init__(self, at):
        self.col = {}
        self.a = {}
        self.gains = {}
        Q = 64
        for k, t in at.idx.items():
            if not mandatory(k):
                continue
            g = t["gutter"]
            s = t["pixels"]
            base = cv.resize(
                at.data(k, 0)[g : g + s, g : g + s], (Q, Q), interpolation=cv.INTER_AREA
            ).astype("f")
            alpha = base[:, :, 3] / 255.0
            rgb = base[:, :, :3] / np.maximum(alpha[:, :, None], 0.01)
            ref = refine.guide(k)
            guide = cv.resize(
                ref[:, :, :3][:, :, ::-1], (Q, Q), interpolation=cv.INTER_AREA
            )
            confidence = cv.resize(ref[:, :, 3], (Q, Q), interpolation=cv.INTER_AREA)
            m = (
                (alpha > 0.12)
                & (alpha < 0.98)
                & (confidence > 0.03)
                & (rgb.min(2) > 26)
                & (guide.min(2) > 26)
            ).astype("f")
            v = np.dstack(
                [
                    np.log(np.maximum(guide, 10) / np.maximum(rgb, 10)).clip(
                        -0.35, 0.35
                    )
                    * m[:, :, None],
                    m,
                ]
            )
            self.col[k] = v
            self.a[k] = alpha
        for i, j in self.col:
            u = np.zeros((Q * 3, Q * 3, 4), "f")
            a = np.zeros((Q * 3, Q * 3), "f")
            for y in range(3):
                for x in range(3):
                    k = (i + x - 1, j + y - 1)
                    sl = np.s_[y * Q : (y + 1) * Q, x * Q : (x + 1) * Q]
                    if k in self.col:
                        u[sl] = self.col[k]
                        a[sl] = self.a[k]
            v = blur(u, 3.3)
            sd = cv.distanceTransform(np.uint8(a >= 0.93), cv.DIST_L2, 5) * 0.16
            scale = np.exp(-sd * sd / 0.82)
            res = v[:, :, :3] / (v[:, :, 3:] + 0.025) * scale[:, :, None]
            self.gains[i, j] = np.exp(res[Q - 5 : 2 * Q + 5, Q - 5 : 2 * Q + 5]).astype(
                "f"
            )

    def __call__(self, out, k, origin, cell):
        if k not in self.gains:
            return
        h, w = out.shape[:2]
        y, x = np.mgrid[:h, :w].astype("f")
        xy = (origin - (np.array(k) * M)) + (np.stack([x, y], -1) + 0.5) * cell
        gain = cv.remap(
            self.gains[k],
            (xy / 0.16 + 4.5).astype("f"),
            None,
            cv.INTER_LINEAR,
            borderMode=cv.BORDER_REPLICATE,
        )
        out[:, :, :3] = np.uint8(np.clip(out[:, :, :3] * gain + 0.5, 0, 255))


def run():
    at = Atlas()
    obs = read(W / "observations.json")
    canvas = Canvas()
    for path in (W / "state").glob("*.npy"):
        canvas.tiles[tuple(map(int, path.stem.split("_")))] = np.load(path)
    fit = fieldfit(at, obs)
    print("registration", fit, flush=True)
    zz = read("pipeline/groundtex/patches-v5.json")
    patches = {}
    for idx, ref in [
        (3, ("left_pillar", 4360)),
        (5, ("back", 3950)),
        (6, ("left_pillar", 4586)),
    ]:
        patches[zz[idx]["tag"]] = local_repair(canvas, obs, zz[idx], ref)
    print("coherent patches", patches, flush=True)
    verge = Verge(at)
    print("verge gains", len(verge.gains), flush=True)
    dump(W / "observations.json", obs)
    for k, a in canvas.tiles.items():
        np.save(W / "state" / f"{k[0]}_{k[1]}.npy", a)
    ks = list(at.idx)
    meta = at.meta
    results = []
    with ThreadPoolExecutor(6) as ex:
        for j, t in enumerate(
            ex.map(
                lambda k: export_tile(k, canvas, obs, softedge, actual_guard, verge), ks
            )
        ):
            if t:
                results.append(t)
            if j % 200 == 0:
                print("export", j, flush=True)
    meta["tiles"] = results
    dump(OUT / "ground.json", meta)
    dump(RPT / "native-mask-check.json", dict(stats))
    dump(RPT / "patch-registration.json", dict(hotspots=fit, features=patches))
    print("ready", len(results), flush=True)


if __name__ == "__main__":
    run()
