"""Build phototextured, track-locked upright planes; no dense depth survives into the renderer."""

from .popup_sem import *
from .render import rowmat, rig, H, B, WID, HEI, FOC, ct
from .pack import BASE, dump
from pipeline.livemap.prepare import ground
from concurrent.futures import ThreadPoolExecutor
from scipy.optimize import linear_sum_assignment
from functools import lru_cache
import sys, os

DST = ROOT / "bank"
GROUP = {2: 2, 3: 2, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 16: 2}
FAR = 92.0


def norm(x):
    return x / max(1e-8, np.linalg.norm(x))


@lru_cache(None)
def rays(c):
    return np.fromfile(STAGE / f"rgbd/{c}/rays.bin", "<f4").reshape(H, B, 3)


def wrays(c, e, x, y):
    R, O = rowmat(e, (y + 0.5) / H)
    return O, np.einsum("nij,nj->ni", R, rays(c)[y, x])


def floorpts(e, o, dr):
    n = np.array(e["groundPlane"])
    dot = dr @ n[:3]
    r = -(o @ n[:3] + n[3]) / np.minimum(dot, -0.009)
    p = o + dr * r[:, None]
    b, z = ground("live", p)
    r = np.where(
        (b <= e["t"]) & np.isfinite(z) & (abs(z - p[:, 2]) < 0.42),
        r + np.clip((z - p[:, 2]) / np.minimum(dr[:, 2], -0.015), -2, 2),
        r,
    )
    return o + dr * r[:, None], (dot < -0.014) & (r > 1) & (r < 70)


def segment(c, e, static):
    cat = cv.imread(str(STAGE / e["mask"]), -1)
    z = cv.imread(str(STAGE / e["z"]), -1) * 0.01
    movers = cat >= 8
    Omid = np.array(e["rowPoses"]).mean(0)[:3]
    lab = np.zeros(cat.shape, "H")
    ls = []
    if static:
        sem = cv.imread(str(semfile(c, e)), 0)
        assert sem is not None
        terrain = np.isin(sem, [0, 1, 9]) | (cat == 2)
        staticmask = (cat == 1) & ~terrain
        groups = [
            (typ, np.isin(sem, [a for a, b in GROUP.items() if b == typ]) & staticmask)
            for typ in sorted(set(GROUP.values()))
        ]
    else:
        terrain = cat == 2
        groups = [(int(v), cat == v) for v in np.unique(cat[movers])]
    for typ, binm in groups:
        # Close narrow segmentation cracks for connectivity only; keep the original matte.
        conn = cv.morphologyEx(binm.astype("B"), cv.MORPH_CLOSE, np.ones((3, 3), "B"))
        num, idx, stats, _ = cv.connectedComponentsWithStats(conn, 8)
        if not static:
            # A single tracked instance may have disconnected wheels, glazing or panels.
            iy, ix = np.where(binm)
            idx = binm.astype("B")
            stats = [
                None,
                (
                    int(ix.min()),
                    int(iy.min()),
                    int(ix.max() - ix.min() + 1),
                    int(iy.max() - iy.min() + 1),
                    len(ix),
                ),
            ]
        for sid, (x0, y0, w, h, area) in enumerate(stats[1:], 1):
            if area < (38 if typ in [5, 6, 7] or not static else 110) or w < 2 or h < 3:
                continue
            win = np.s_[y0 : y0 + h, x0 : x0 + w]
            m = binm[win] & (idx[win] == sid)
            segments = [(0, w)]
            if static and typ in [2, 4, 8] and w >= 96 and area > 2200:
                # Split only if two sizeable neighbours have a persistent range step, never per pixel.
                cuts = []
                zz = z[win]
                med = []
                for a in range(0, w, 10):
                    zz0 = zz[:, a : a + 10]
                    q = zz0[(zz0 > 0) & m[:, a : a + 10]]
                    med.append(float(np.median(q)) if len(q) > 12 else np.nan)
                for j in range(3, len(med) - 3):
                    a, b = np.array(med[j - 3 : j]), np.array(med[j : j + 3])
                    if (
                        np.isfinite(a).all()
                        and np.isfinite(b).all()
                        and abs(np.log(np.median(a) / np.median(b))) > 0.46
                    ):
                        cuts.append((abs(np.log(np.median(a) / np.median(b))), 10 * j))
                div = [0, w]
                for _, x in sorted(cuts, reverse=True):
                    if min(abs(x - u) for u in div) > 40 and len(div) < 4:
                        div.append(x)
                div.sort()
                segments = list(zip(div[:-1], div[1:]))
            for lo, hi in segments:
                yy, xx = np.where(m[:, lo:hi])
                xx += lo + x0
                yy += y0
                if len(xx) < 30:
                    continue
                x, inv = np.unique(xx, return_inverse=True)
                bottom = np.zeros(len(x), int)
                np.maximum.at(bottom, inv, yy)
                g = (bottom >= np.percentile(bottom, 45)) & (bottom < H - 5)
                beneath = np.zeros(len(x), bool)
                for dy in (1, 2, 4, 7):
                    beneath |= terrain[np.minimum(H - 1, bottom + dy), x]
                g &= beneath if static else True
                fo, fd = wrays(c, e, x, bottom)
                points, ok = floorpts(e, fo, fd)
                g &= ok
                depth = z[yy, xx]
                good = depth > 0.15
                samp = np.flatnonzero(good)[:: max(1, int(good.sum()) // 350)]
                dp = np.zeros((0, 3))
                if len(samp):
                    oo, dd = wrays(c, e, xx[samp], yy[samp])
                    dp = (
                        oo
                        + dd
                        * (
                            depth[samp]
                            / np.maximum(0.03, rays(c)[yy[samp], xx[samp], 2])
                        )[:, None]
                    )
                anchored = g.sum() >= max(2, min(9, len(x) * 0.13))
                centerpx = np.array([int(np.median(xx)), int(np.median(yy))])
                oo, dd = wrays(c, e, centerpx[:1], centerpx[1:])
                axis = norm(dd[0, :2])
                if anchored:
                    foot = points[g]
                    pos = np.median(foot, axis=0)
                    mode = "contact"
                elif len(dp) >= 4:
                    pos = np.median(dp, axis=0)
                    foot = dp
                    mode = "median"
                else:
                    pos = np.r_[Omid[:2] + axis * FAR, Omid[2]]
                    foot = pos[None]
                    mode = "backdrop"
                dist = np.linalg.norm(pos[:2] - Omid[:2])
                if dist > 65 or mode == "backdrop":
                    pos[:2] = Omid[:2] + axis * FAR
                    mode = "backdrop"
                if dist < 2.5:
                    continue
                n = norm(pos[:2] - Omid[:2])
                if (
                    static
                    and typ in [2, 4, 8]
                    and w > 38
                    and len(foot) > 12
                    and mode != "backdrop"
                ):
                    cen = np.median(foot[:, :2], 0)
                    rr = np.linalg.norm(foot[:, :2] - cen, axis=1)
                    f = foot[rr <= np.percentile(rr, 80), :2]
                    cov = np.cov(f.T)
                    val, vec = np.linalg.eigh(cov)
                    nn = vec[:, 0]
                    if val[1] > max(0.7, val[0] * 7) and abs(nn @ n) > 0.55:
                        n = nn * np.sign(nn @ n)
                tangent = np.array([-n[1], n[0]])
                corners = np.array(
                    [
                        [xx.min(), yy.min()],
                        [xx.max(), yy.min()],
                        [xx.min(), yy.max()],
                        [xx.max(), yy.max()],
                    ]
                )
                oo, dd = wrays(c, e, corners[:, 0], corners[:, 1])
                rr = ((pos[:2] - oo[:, :2]) @ n) / np.maximum(0.18, dd[:, :2] @ n)
                p = oo + dd * rr[:, None]
                span = np.ptp(p[:, :2] @ tangent)
                hspan = np.ptp(p[:, 2])
                lid = len(ls) + 1
                lab[yy, xx] = lid
                ls.append(
                    dict(
                        local=lid,
                        kind=typ,
                        pos=pos.tolist(),
                        normal=n.tolist(),
                        span=float(np.clip(span, 0.1, 120)),
                        height=float(np.clip(hspan, 0.1, 45)),
                        source=mode,
                        bbox=[
                            int(xx.min()),
                            int(yy.min()),
                            int(xx.max() + 1),
                            int(yy.max() + 1),
                        ],
                        pixels=len(xx),
                    )
                )
    return lab, ls


def associate(es, obs):
    tracks = []
    prev = []
    for e, ls in zip(es, obs):
        cost = np.full((len(prev), len(ls)), 1e4)
        for i, a in enumerate(prev):
            p = np.array(a["pos"][:2])
            n = np.array(a["normal"])
            tan = np.array([-n[1], n[0]])
            for j, b in enumerate(ls):
                if a["kind"] != b["kind"]:
                    continue
                q = np.array(b["pos"][:2]) - p
                diff = abs(q @ n)
                tang = abs(q @ tan)
                rad = 0.5 * (a["span"] + b["span"])
                far = a["source"] == "backdrop" and b["source"] == "backdrop"
                dtol = (
                    14
                    if far
                    else 2.8
                    + 0.09
                    * min(65, np.linalg.norm(p - np.asarray(e["rowPoses"][0][:2])))
                )
                xtol = max(1.1, 0.80 * rad)
                area = abs(np.log(a["pixels"] / b["pixels"]))
                if diff < dtol and tang < xtol and area < 2.55:
                    cost[i, j] = (
                        0.42 * diff / dtol + 0.40 * tang / xtol + 0.18 * area / 2.55
                    )
        if cost.size:
            for i, j in zip(*linear_sum_assignment(cost)):
                if cost[i, j] < 0.89:
                    ls[j]["track"] = prev[i]["track"]
        for a in ls:
            if "track" not in a:
                a["track"] = len(tracks) + 1
                tracks.append([])
            tracks[a["track"] - 1].append(a)
        prev = ls
    # One fixed world plane per track. Prefer ground contacts over inferred centres.
    for tr in tracks:
        anchored = [a for a in tr if a["source"] == "contact"]
        ls = anchored if anchored else tr
        pp = np.array([a["pos"][:2] for a in ls])
        nn = np.array([a["normal"] for a in ls])
        ref = nn[len(nn) // 2]
        nn *= np.where(nn @ ref < 0, -1, 1)[:, None]
        n = norm(np.median(nn, 0))
        d = float(np.median(pp @ n))
        for a in tr:
            a["plane"] = [*n, d]
    return tracks


def skyframe(c, e):
    # Smoothly extend only sky colour into missing directions; no object RGB is copied.
    im = cv.resize(
        cv.imread(str(STAGE / e["rgb"])), (362, 235), interpolation=cv.INTER_AREA
    ).astype("f")
    cat = cv.resize(
        cv.imread(str(STAGE / e["mask"]), -1),
        (362, 235),
        interpolation=cv.INTER_NEAREST,
    )
    mask = cv.erode((cat == 3).astype("B"), np.ones((5, 5), "B")).astype("f")
    default = np.array([151, 151, 137], "f")
    blur = lambda a, s: cv.GaussianBlur(a, (0, 0), s)
    bg = np.ones_like(im) * default
    for sigma in [36, 14, 4, 1.2]:
        den = blur(mask, sigma)
        num = blur(im * mask[:, :, None], sigma)
        mix = np.clip(den / 0.30, 0, 1)[:, :, None]
        bg = bg * (1 - mix) + mix * num / np.maximum(0.0001, den[:, :, None])
    return np.uint8(np.clip(bg, 0, 255))


def make_atlas(c, e, ls, lab, static):
    # Rectify each object into its OWN world plane once. The rendered primitive is a quad.
    im = cv.imread(str(STAGE / e["rgb"]))
    tiles = []
    meta = []
    C = rig[c]
    midR, midO = rowmat(e, np.array([0.5]))
    midO = midO[0]
    planeG = np.array(e["groundPlane"])
    for a in ls:
        nn = np.array(a["plane"][:2])
        d = a["plane"][2]
        tan = np.array([-nn[1], nn[0]])
        x, y, X, Y = a["bbox"]
        yy, xx = np.where(lab[y:Y, x:X] == a["local"])
        xx += x
        yy += y
        stride = max(1, len(yy) // 1200)
        xo = xx[::stride]
        yo = yy[::stride]
        O, D = wrays(c, e, xo, yo)
        den = D[:, :2] @ nn
        rr = (d - O[:, :2] @ nn) / np.where(abs(den) > 0.14, den, 1e-8)
        ok = (rr > 1) & (rr < 230) & (abs(den) > 0.14)
        if np.mean(ok) < 0.85:
            continue
        P = O[ok] + D[ok] * rr[ok, None]
        us = P[:, :2] @ tan
        zs = P[:, 2]
        u0, u1 = us.min(), us.max()
        z0, z1 = zs.min(), zs.max()
        uc = np.median(us)
        xy = nn * d + tan * uc
        floor = -(xy @ planeG[:2] + planeG[3]) / planeG[2]
        if a["source"] == "contact":
            z0 = max(z0, floor - 0.12)
        if z1 <= z0 + 0.04 or u1 <= u0 + 0.04:
            continue
        scale = min(
            98.0,
            max(1.0, 1.60 * (X - x) / (u1 - u0)),
            max(1.0, 1.60 * (Y - y) / (z1 - z0)),
        )
        ww = int(np.clip(np.ceil((u1 - u0) * scale) + 4, 6, 760))
        hh = int(np.clip(np.ceil((z1 - z0) * scale) + 4, 6, 480))
        u0 -= 1 / scale
        u1 += 1 / scale
        z0 -= 1 / scale
        z1 += 1 / scale
        u, z = np.meshgrid(np.linspace(u0, u1, ww), np.linspace(z1, z0, hh))
        w = np.stack((nn[0] * d + tan[0] * u, nn[1] * d + tan[1] * u, z), -1)
        v = (w - midO) @ midR[0]
        uv, valid = C.project(v)
        jrow = np.clip(
            uv[:, :, 1] / C.height, 0, 1
        )  # row poses vary slowly; evaluate by 16 horizontal time bins
        R, O = rowmat(e, (np.arange(16) + 0.5) / 16)
        jj = np.clip(np.nan_to_num(jrow * 16).astype(int), 0, 15)
        v = np.einsum("...ji,...j->...i", R[jj], w - O[jj])
        uv, valid = C.project(v)
        uv = np.nan_to_num(uv) * np.r_[B / C.width, H / C.height] - 0.5
        uv = uv.astype("f")
        m = (lab[y:Y, x:X] == a["local"]).astype("B")
        pad = cv.copyMakeBorder(m, 2, 2, 2, 2, cv.BORDER_CONSTANT)
        dist = cv.distanceTransform(pad, cv.DIST_L2, 3)[2:-2, 2:-2]
        alpha = np.clip((dist - 0.35) / 1.6, 0, 1)
        A = cv.remap(alpha, uv - np.float32([x, y]), None, cv.INTER_LINEAR)
        rgb = cv.remap(im, uv * 2 + 0.5, None, cv.INTER_LINEAR)
        A *= valid
        if (A > 0.4).sum() < 7:
            continue
        tiles.append((rgb, np.uint8(A * 255 + 0.5)))
        meta.append(
            dict(
                id=a["track"],
                kind=a["kind"],
                plane=[float(nn[0]), float(nn[1]), float(d)],
                extent=[float(u0), float(z0), float(u1), float(z1)],
                anchor=a["source"],
            )
        )
    folder = DST / ("base" if static else "movers") / c
    folder.mkdir(parents=True, exist_ok=True)
    stem = folder / f"{e['n']:06}"
    sky = None
    if static:
        sky = str(stem.relative_to(DST)) + "-sky.jpg"
        cv.imwrite(str(DST / sky), skyframe(c, e), [cv.IMWRITE_JPEG_QUALITY, 81])
    # Pack horizontally in shelves with two pixels of gutter; rects exclude gutter.
    rects = []
    x = y = sh = maxx = 0
    for rgb, m in tiles:
        h, w = m.shape
        if x + w > 1400:
            x = 0
            y += sh + 2
            sh = 0
        rects.append([x, y, w, h])
        x += w + 2
        maxx = max(maxx, x)
        sh = max(sh, h)
    paths = {}
    if tiles:
        rgb = np.zeros((y + sh, maxx, 3), "B")
        mask = np.zeros(rgb.shape[:2], "B")
        for a, rect, (im, ma) in zip(meta, rects, tiles):
            x, y, w, h = rect
            rgb[y : y + h, x : x + w] = im
            mask[y : y + h, x : x + w] = ma
            a["rect"] = rect
        for suffix, img, opts, k in [
            (".jpg", rgb, [cv.IMWRITE_JPEG_QUALITY, 86], "rgb"),
            ("-a.png", mask, [cv.IMWRITE_PNG_COMPRESSION, 3], "alpha"),
        ]:
            paths[k] = str(stem.relative_to(DST)) + suffix
            assert cv.imwrite(str(DST / paths[k]), img, opts)
    return dict(
        n=e["n"],
        t=e["t"],
        s=e["s"],
        rowPoses=e["rowPoses"],
        sky=sky,
        layers=meta,
        **paths,
    )


def build(c):
    cs = bank()[c]
    es = bases(cs["keyframes"])
    tmp = ROOT / "ids" / c
    tmp.mkdir(parents=True, exist_ok=True)

    def seg(e):
        lab, ls = segment(c, e, True)
        cv.imwrite(str(tmp / f"{e['n']}.png"), lab)
        return ls

    with ThreadPoolExecutor(4) as pool:
        obs = list(pool.map(seg, es))
        tracks = associate(es, obs)

    def pack(arg):
        e, ls = arg
        return make_atlas(c, e, ls, cv.imread(str(tmp / f"{e['n']}.png"), -1), True)

    with ThreadPoolExecutor(4) as pool:
        base = list(pool.map(pack, zip(es, obs)))
    dynamic = pack_movers(c, cs)
    j = dict(
        native=cs["native"],
        params=cs["params"],
        model=cs["model"],
        base=base,
        movers=dynamic,
        tracks=len(tracks),
    )
    dump(DST / (c + ".json"), j)
    print(
        c,
        "base",
        len(base),
        "tracks",
        len(tracks),
        "mean cuts",
        np.mean([len(b["layers"]) for b in base]),
        "dynamic",
        len(dynamic),
        flush=True,
    )


def pack_movers(c, cs):
    def mov(e):
        if not e["hasInstances"]:
            return dict(n=e["n"], t=e["t"], layers=[])
        lab, ls = segment(c, e, False)
        for a in ls:
            a["track"] = 1000000 + (a["kind"] - 8) // 3
            a["plane"] = [*a["normal"], float(np.dot(a["normal"], a["pos"][:2]))]
        return make_atlas(c, e, ls, lab, False)

    with ThreadPoolExecutor(4) as pool:
        return list(
            pool.map(mov, [e for e in cs["keyframes"] if 92.7 <= e["t"] <= 135.4])
        )


if __name__ == "__main__":
    for c in LENSES:
        build(c)
