"""Render both exact intervals, orbit comparisons, and motion-compensated hand-over diagnostics."""

from .render import *
from pipeline.livemap.v4_proof import encoder, finish
from pipeline.livemap.render import label

OUT = RS / "pipeline/ibr/report"
OUT.mkdir(parents=True, exist_ok=True)


def jpg(p, im):
    assert cv.imwrite(str(p), im[:, :, ::-1], [cv.IMWRITE_JPEG_QUALITY, 89])


def tagged(r, t, az, raw=None):
    a, b, v, k, T = r.draw(t, az, raw)
    st = r.cam.raw[round(t * 30)]
    z = r.best[:, :, 2]
    return box(a, st, T, z), b, v, k, T


def exposure(r, t):
    es = r.frames["front"]
    e = min(es, key=lambda e: abs(e["t"] - t))
    im = cv.imread(str(STAGE / e["rgb"]))[:, :, ::-1]
    h, w = im.shape[:2]
    sz = (round(HEI * w / h), HEI)
    dst = np.zeros((HEI, WID, 3), "B")
    left = (WID - sz[0]) // 2
    dst[:, left : left + sz[0]] = cv.resize(im, sz, interpolation=cv.INTER_AREA)
    return dst, e


def texture(r, img, base, t, T, k):
    # Sample gradients only inside same-source static regions, away from cut edges.
    tag = r.best[:, :, 3]
    z = r.best[:, :, 2]
    m = (k == 1) & (z > 3)
    m &= cv.dilate(tag, None, iterations=3) == cv.erode(tag, None, iterations=3)
    m = cv.erode(m.astype("B"), np.ones((7, 7), "B")).astype(bool)
    y, x = np.where(m[4::8, 4::8])
    y = y * 8 + 4
    x = x * 8 + 4
    tags = tag[y, x].astype(int)
    ray = np.c_[(x + 0.5 - WID / 2) / FOC, (y + 0.5 - HEI / 2) / FOC, np.ones(len(y))]
    world = T[:3, 3] + (ray @ T[:3, :3].T) * z[y, x, None]

    def grad(im):
        L = cv.cvtColor(im, cv.COLOR_RGB2GRAY).astype("f")
        return (
            cv.magnitude(cv.Sobel(L, -1, 1, 0, ksize=3), cv.Sobel(L, -1, 0, 1, ksize=3))
            * 0.125
        )

    src = []
    out = []
    old = []
    G = grad(img)
    B0 = grad(base)
    for j in np.unique(tags):
        c = CAMS[j // 100000]
        n = j % 100000
        e = next(e for e in r.frames[c] if e["n"] == n)
        ii = np.where(tags == j)[0]
        w = world[ii]
        row = np.full(len(ii), 0.5)
        for it in range(2):
            rot, o = rowmat(e, row)
            vec = np.einsum("nji,nj->ni", rot, w - o)
            uv, ok = rig[c].project(vec)
            row = np.clip(uv[:, 1] / rig[c].height, 0, 1)
        uv *= np.array([1448 / rig[c].width, 938 / rig[c].height])
        ii = ii[ok]
        uv = uv[ok].astype(int)
        good = (uv[:, 0] >= 1) & (uv[:, 0] < 1447) & (uv[:, 1] >= 1) & (uv[:, 1] < 937)
        ii = ii[good]
        uv = uv[good]
        im = cv.imread(str(STAGE / e["rgb"]))[:, :, ::-1]
        im = np.uint8(np.clip(im * np.array(e["displayGain"]), 0, 255))
        src.extend(grad(im)[uv[:, 1], uv[:, 0]].tolist())
        out.extend(G[y[ii], x[ii]].tolist())
        old.extend(B0[y[ii], x[ii]].tolist())
    return dict(
        samples=len(src),
        p90_source_ibr_v5=[float(np.percentile(ls, 90)) for ls in [src, out, old]],
    )


def stills():
    r = Scene()
    data = []
    for t in [112, 100, 128, 135]:
        ls = []
        photo, e = exposure(r, t)
        raw = r.fb.points(t)
        for az in [0, -30, 30, -60, 60]:
            a, b, v, k, T = tagged(r, t, az, raw)
            item = dict(
                t=t,
                az=az,
                selectedFront=r.closest.get("front"),
                contemporaryFront=e["t"],
                coverage=float(v.mean()),
                holesByThird=[float((~l).mean()) for l in np.array_split(v, 3)],
                moverPixels=int((k >= 8).sum()),
                layers={str(i): float((k == i).mean()) for i in [1, 2, 3]},
            )
            source, witness = exposure(r, item["selectedFront"])
            img = np.hstack(
                [
                    label(b, "v5"),
                    label(a, f"IBR | {t} s | orbit {az:+d} deg"),
                    label(source, f"viewpoint-selected front | {witness['t']:.2f} s"),
                    label(photo, f"contemporaneous front | {e['t']:.2f} s"),
                ]
            )
            jpg(OUT / f"{t}-{az:+d}.jpg", img)
            ls.append(cv.resize(img, (1536, 248), interpolation=cv.INTER_AREA))
            if az == 0:
                item["texture"] = texture(r, a, b, t, T, k)
                item["frontArcLagM"] = float(
                    np.interp(t, r.spec["front"][0], r.spec["front"][3])
                    - np.interp(
                        item["selectedFront"], r.spec["front"][0], r.spec["front"][3]
                    )
                )
            data.append(item)
        jpg(OUT / f"{t}-sheet.jpg", np.vstack(ls))
        print("stills", t, flush=True)
    dump(OUT / "stills.json", data)


def temporal(prev, img, v, k, T, r):
    ys, xs = np.mgrid[4:HEI:8, 4:WID:8]
    sl = np.s_[4::8, 4::8]
    z = r.best[sl][:, :, 2]
    tag = r.best[sl][:, :, 3].astype(int)
    g = (k[sl] == 1) & v[sl]
    Y = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    cur = (Y, r.best[:, :, 2].copy(), r.best[:, :, 3].copy(), (k == 1) & v, T)
    res = None
    if prev is not None:
        im, pz, pt, pv, P = prev
        dirs = np.stack(
            [(xs + 0.5 - WID / 2) / FOC, (ys + 0.5 - HEI / 2) / FOC, np.ones(xs.shape)],
            -1,
        )
        w = (dirs @ T[:3, :3].T) * z[:, :, None] + T[:3, 3]
        q = (w - P[:3, 3]) @ P[:3, :3]
        uv = q[:, :, :2] / np.maximum(q[:, :, 2, None], 0.1) * FOC + [
            WID / 2 - 0.5,
            HEI / 2 - 0.5,
        ]
        ix = np.round(uv[:, :, 0]).astype(int).clip(0, WID - 1)
        iy = np.round(uv[:, :, 1]).astype(int).clip(0, HEI - 1)
        g &= (
            pv[iy, ix]
            & (abs(pz[iy, ix] - q[:, :, 2]) < np.maximum(0.35, 0.055 * z))
            & (ix > 0)
            & (iy > 0)
            & (ix < WID - 1)
            & (iy < HEI - 1)
        )
        py = cv.remap(im, uv.astype("f"), None, cv.INTER_LINEAR)
        err = abs(py.astype("f") - Y[sl])
        change = pt[iy, ix] != tag
        res = [
            [int(m.sum()), float(err[m].sum()), int((err[m] > 20).sum())]
            for m in [g & ~change, g & change]
        ]
    return cur, res


def movie(name):
    bank = STAGE.parent / "movie-bank.json"
    r = Scene(bank if bank.exists() else None)
    start, end = (105, 120) if name == "road" else (93, 104)
    N = (end - start) * 30
    path = OUT / f"{name}-{start}-{end}.mp4"
    enc = encoder(path, WID, HEI, 30, "14500k")
    prev = None
    data = []
    sheet = []
    for n in range(N):
        t = start + n / 30
        if n % 15 == 0:
            raw = r.fb.points(t + 7 / 30, 0.28)
        a, b, v, k, T = tagged(r, t, 0, raw)
        prev, err = temporal(prev, a, v, k, T, r)
        enc.stdin.write(a.tobytes())
        m = k >= 8
        ages = [abs(t - r.frames[c][i]["t"]) for c, i, mode in r.drawn if mode]
        data.append(
            dict(
                t=t,
                hole=float((~v).mean()),
                mover_pixels=int(m.sum()),
                confirmed_moving_pixels=int((m & ((k - 8) % 3 == 2)).sum()),
                max_mover_age=max(ages, default=0),
                primary_front=r.closest.get("front"),
                matched=err,
            )
        )
        if n % 30 == 0:
            sheet.append(
                label(
                    cv.resize(a, (384, 240), interpolation=cv.INTER_AREA), f"{t:.1f} s"
                )
            )
        if n % 90 == 0:
            print(name, n, "/", N, flush=True)
    info = finish(enc, path, N, 30)
    pad = [np.zeros_like(sheet[0])] * ((-len(sheet)) % 3)
    s = sheet + pad
    jpg(
        OUT / f"{name}-sequence.jpg",
        np.vstack([np.hstack(s[i : i + 3]) for i in range(0, len(s), 3)]),
    )
    dump(OUT / f"{name}.json", dict(video=info, frames=data))
    print(info, flush=True)


if __name__ == "__main__":
    stills() if sys.argv[1] == "stills" else movie(sys.argv[1])
