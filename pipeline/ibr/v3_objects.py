"""Ground-validated footprints, gap-tolerant cross-lens identities and one canonical card per object."""

from .v3_ground import *
from .v3_pano import gain
from scipy.spatial.transform import Rotation
from scipy.optimize import linear_sum_assignment
from collections import Counter, defaultdict
from .render import rowmat

TMP = W / "cards"
layers.STAGE = STAGE
layers.DST = TMP
layers.semfile = sem


def inspect(c, e, protect, static, cachelab=False):
    lab, ls = layers.segment(c, e, static)
    z = cv.imread(e["z"], -1) * 0.01
    im = cv.resize(cv.imread(e["rgb"], 0), (B, H))
    high = abs(cv.Laplacian(im, cv.CV_32F))
    O = np.array(e["rowPoses"])[:, :3].mean(0)
    out = []
    for a in ls:
        x, y, X, Y = a["bbox"]
        v = np.s_[y:Y, x:X]
        yy, xx = np.where(lab[v] == a["local"])
        step = max(1, len(xx) // 500)
        xx = x + xx[::step]
        yy = y + yy[::step]
        oo, dd = wrays(c, e, xx, yy)
        axial = z[yy, xx]
        valid = axial > 0.4
        if valid.sum() < 8:
            continue
        wp = (
            oo[valid]
            + dd[valid]
            * (axial[valid] / np.maximum(0.06, rays(c)[yy, xx, 2][valid]))[:, None]
        )
        dp = np.median(wp, 0)
        md = np.linalg.norm(dp[:2] - O[:2])
        cp = np.array(a["pos"])
        cd = np.linalg.norm(cp[:2] - O[:2])
        cert = a["source"] == "contact" and abs(cd - md) <= 0.25 * md
        pos = cp if cert else dp
        dist = np.linalg.norm(pos[:2] - O[:2])
        if static and (md > 45 or dist > 47):
            continue
        if dist < 1.7 or dist > 160:
            continue
        if static and protect.query(pos[None, :2])[0][0]:
            continue
        n = norm(
            pos[:2] - O[:2]
        )  # Trees and poles face the best camera; stable tangents for coherent facades only.
        if a["kind"] in [2, 4] and cert and np.array(a["normal"]) @ n > 0.6:
            n = np.array(a["normal"])
        clip = min(x, y, B - X, H - Y) < 3
        sharp = min(30.0, np.mean(high[yy, xx]))
        a.update(
            pos=pos.tolist(),
            normal=n.tolist(),
            plane=[*n, float(n @ pos[:2])],
            source="contact" if cert else "depth",
            depthRange=float(md),
            contactRange=float(cd) if a["source"] == "contact" else None,
            span=float(np.clip(a["span"] * dist / max(cd, 1), 0.18, 18)),
            clip=bool(clip),
            score=float(
                np.sqrt(a["pixels"])
                * (1 + sharp / 18)
                * (0.24 if clip else 1)
                * (0.94 if not cert else 1.12)
            ),
            c=c,
            n=e["n"],
            t=e["t"],
        )
        out.append(a)
    if cachelab:
        p = TMP / "ids" / c
        p.mkdir(parents=True, exist_ok=True)
        assert cv.imwrite(str(p / f"{e['n']}.png"), lab)
    return lab, out


def associate(observations):
    tracks = []
    for (c, n, t), ls in sorted(observations.items(), key=lambda kv: kv[0][2]):
        active = [i for i, tr in enumerate(tracks) if t - tr[-1]["t"] < 4.8]
        cost = np.full((len(active), len(ls)), 99.0)
        for ii, i in enumerate(active):
            tr = tracks[i]
            last = tr[-1]
            pp = np.array([a["pos"][:2] for a in tr[-5:]])
            p = np.median(pp, 0)
            pred = p + np.clip(last["pos"][:2] - p, -0.65, 0.65) * min(1, t - last["t"])
            nrm = np.array(last["normal"])
            tan = np.array([-nrm[1], nrm[0]])
            for j, b in enumerate(ls):
                if last["kind"] != b["kind"]:
                    continue
                d = b["pos"][:2] - pred
                rad = 0.5 * (last["span"] + b["span"])
                utol = min(7, max(0.9, 0.7 * rad))
                ztol = min(4.0, 0.85 + 0.08 * b["depthRange"])
                u = abs(d @ tan) / utol
                v = abs(d @ nrm) / ztol
                ratio = abs(np.log(max(0.2, last["span"]) / max(0.2, b["span"])))
                if u < 1.15 and v < 1.15 and ratio < 1.5:
                    cost[ii, j] = 0.45 * u + 0.45 * v + 0.1 * min(1, ratio)
        matched = set()
        if cost.size:
            for ii, j in zip(*linear_sum_assignment(cost)):
                if cost[ii, j] < 0.91:
                    tracks[active[ii]].append(ls[j])
                    matched.add(j)
        tracks.extend([a] for j, a in enumerate(ls) if j not in matched)
    return tracks


def blocked(card, mask):
    u, z, U, Z = card["extent"]
    n = np.array(card["plane"][:2])
    d = card["plane"][2]
    us = np.linspace(u, U, max(3, int((U - u) / 0.15) + 1))
    pos = n * d + us[:, None] * [-n[1], n[0]]
    return bool(mask.query(pos)[0].any())


def promote(c, e, ls, lab, static, mask, folder):
    f = layers.make_atlas(c, e, ls, lab, False)
    out = []
    if not f["layers"]:
        return out
    rgb = cv.imread(str(TMP / f["rgb"]))
    al = cv.imread(str(TMP / f["alpha"]), 0)
    lookup = {a["track"]: a for a in ls}
    tiles = []
    for a in f["layers"]:
        obs = lookup[a["id"]]
        u, z, U, Z = a["extent"]
        nn = np.array(a["plane"][:2])
        xy = nn * a["plane"][2] + np.array([-nn[1], nn[0]]) * (u + U) / 2
        gp = e["groundPlane"]
        floor = float(-(xy @ gp[:2] + gp[3]) / gp[2])
        if static and (blocked(a, mask) or U - u > 23 or Z - z > 24):
            continue
        a["extent"] = [u, floor, U, floor + Z - z]
        x, y, w, h = a.pop("rect")
        ma = al[y : y + h, x : x + w]
        col = rgb[y : y + h, x : x + w].astype("f") * gain(c, e["t"])[::-1]
        tiles.append((np.uint8(np.clip(col + 0.5, 0, 255)), ma))
        a["source"] = [c, e["n"], e["t"]]
        a["contactRange"] = obs["contactRange"]
        a["depthRange"] = obs["depthRange"]
        out.append(a)
    if not tiles:
        return out
    # Per exposure shelf atlas; RGB/alpha have identical texel rectangles. No discarded-quad bytes in export.
    x = y = h = width = 0
    for a, (im, ma) in zip(out, tiles):
        H0, W0 = ma.shape
        if x + W0 > 1536:
            x = 0
            y += h + 2
            h = 0
        a["rect"] = [x, y, W0, H0]
        x += W0 + 2
        h = max(h, H0)
        width = max(width, x)
    im = np.zeros((y + h, width, 3), "B")
    ma = np.zeros((y + h, width), "B")
    stem = f"{folder}/{c}-{e['n']:06}"
    (BUILD / folder).mkdir(parents=True, exist_ok=True)
    for a, (col, alpha) in zip(out, tiles):
        x, y, ww, hh = a["rect"]
        im[y : y + hh, x : x + ww] = col
        ma[y : y + hh, x : x + ww] = alpha
        a["rgb"] = stem + ".jpg"
        a["matte"] = stem + ".png"
    assert cv.imwrite(str(BUILD / (stem + ".jpg")), im, [cv.IMWRITE_JPEG_QUALITY, 90])
    assert cv.imwrite(str(BUILD / (stem + ".png")), ma, [cv.IMWRITE_PNG_COMPRESSION, 4])
    return out


def build():
    mask = RoadMask(W / "roadmask.npz")
    sets = getsets()
    b = load()
    pool = ThreadPoolExecutor(4)
    inputs = {(c, s[c]["n"]): (c, s[c]) for s in sets for c in C3 if c in s}
    ls = list(inputs.values())

    def one(ce):
        c, e = ce
        return (c, e["n"], e["t"]), inspect(c, e, mask, True, True)[1]

    obs = dict(pool.map(one, ls))
    tracks = associate(obs)
    poses = json.load(open(W / "chase.json"))["frames"]
    tt = np.array([p["t"] for p in poses])
    o = np.array([p["pos"] for p in poses])
    fw = -Rotation.from_quat([p["quat"] for p in poses]).as_matrix()[:, :, 2]
    best = defaultdict(list)
    meta = {}
    numCarry = 0
    for i, tr in enumerate(tracks, 1):
        pick = max(tr, key=lambda a: a["score"])
        if (len(tr) == 1 and (pick["clip"] or pick["pixels"] < 180)) or (
            pick["clip"] and pick["kind"] == 8
        ):
            continue
        birth = min(a["t"] for a in tr)
        latest = max(a["t"] for a in tr)
        p = np.array(pick["pos"][:2])
        behind = ((p - o[:, :2]) * fw[:, :2]).sum(1) < -max(2.0, 0.45 * pick["span"])
        outside = np.linalg.norm(p - o[:, :2], axis=1) > 100
        hits = (
            np.convolve((behind | outside).astype(int), np.ones(16, dtype=int), "valid")
            == 16
        )
        ends = np.flatnonzero((tt[: len(hits)] > latest) & hits)
        death = float(tt[ends[0]] + 0.5) if len(ends) else HI + 0.6
        if death <= LO:
            continue
        pick["track"] = i
        best[pick["c"], pick["n"]].append(pick)
        meta[i] = dict(
            birth=birth - 0.35,
            death=death,
            fade=0.45,
            observations=len(tr),
            lastSeen=latest,
        )
        numCarry += sum(np.diff(sorted(set(a["t"] for a in tr))) > 0.6)

    def atlas(kls):
        key, ls = kls
        c, e = inputs[key]
        lab = cv.imread(str(TMP / "ids" / c / f"{e['n']}.png"), -1)
        return promote(c, e, ls, lab, True, mask, "objects")

    cards = [
        dict(a, **meta[a["id"]]) for sub in pool.map(atlas, best.items()) for a in sub
    ]
    dump(W / "cards.json", cards)
    print(
        "tracks",
        len(tracks),
        "exported",
        len(cards),
        "link gaps",
        int(numCarry),
        flush=True,
    )

    # Use nearest-time acquisition records, INCLUDING explicit empty frames. Never hold a stale mover indefinitely.
    def dynamic(ce):
        c, e = ce
        out = []
        if e["hasInstances"]:
            lab, ls = inspect(c, e, mask, False)
            ci = C3.index(c)
            for a in ls:
                a["track"] = 1000000 + ci * 100000 + (a["kind"] - 8) // 3
            out = promote(c, e, ls, lab, False, mask, "movers")
        return dict(t=e["t"], n=e["n"], layers=out)

    movers = {
        c: list(
            pool.map(
                dynamic, [(c, e) for e in b[c] if LO - 0.23 <= e["t"] <= HI + 0.23]
            )
        )
        for c in C3
    }
    dump(W / "dynamic.json", movers)
    print("mover frames", sum(map(len, movers.values())), flush=True)


if __name__ == "__main__":
    build()
