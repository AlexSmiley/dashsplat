"""Reconstruct small, textured surfaces from three independent physical exposures, including distant roof and sign edges."""

from .v3terrain import *
from scipy.spatial import Delaunay
from functools import lru_cache


@lru_cache(20)
def image(path):
    with np.load(path) as a:
        im = a["rgb"]
        lab = a["lab"]
    return im, cv.cvtColor(im, cv.COLOR_RGB2GRAY), lab


def triplet(es, vol):
    c = es[1]["c"]
    yaw = es[1]["yaw"]
    w, h, f, R, _, _, band, _ = pin(c, yaw)
    im, gray, lab = image(es[1]["file"])
    mask = cv.erode((lab == 1).astype("B"), np.ones((3, 3), "B"))
    p = cv.goodFeaturesToTrack(gray, 2400, 0.003, 4, mask=mask, blockSize=5)
    if p is None or len(p) < 8:
        return None
    p = p.reshape(-1, 2)
    args = dict(
        winSize=(19, 19),
        maxLevel=4,
        criteria=(cv.TERM_CRITERIA_COUNT | cv.TERM_CRITERIA_EPS, 24, 0.025),
    )
    uv = []
    ok = np.ones(len(p), bool)
    for j in [0, 2]:
        other, g, l = image(es[j]["file"])
        q, st, err = cv.calcOpticalFlowPyrLK(gray, g, p, None, **args)
        q = q.reshape(-1, 2)
        rev, sr, _ = cv.calcOpticalFlowPyrLK(g, gray, q, None, **args)
        rev = rev.reshape(-1, 2)
        x, y = np.round(q).astype("i").T
        x = x.clip(0, w - 1)
        y = y.clip(0, h - 1)
        ok &= (
            (st.ravel() != 0)
            & (sr.ravel() != 0)
            & (err.ravel() < 16)
            & (np.linalg.norm(rev - p, axis=1) < 0.62)
            & (l[y, x] == 1)
        )
        uv.append(q)
    if sum(ok) < 8:
        return None
    views = []
    pix = p[ok]
    for e, q in zip(es, [uv[0][ok], pix, uv[1][ok]]):
        x, y = np.round(q).astype(int).T
        m = poses(c, e["t"])[band[y.clip(0, h - 1), x.clip(0, w - 1)].clip(0, 7)]
        d = unit(
            np.einsum(
                "nij,nj->ni",
                m[:, :3, :3] @ R,
                np.c_[
                    (q[:, 0] + 0.5 - w / 2) / f,
                    (q[:, 1] + 0.5 - h / 2) / f,
                    np.ones(len(q)),
                ],
            )
        )
        views.append((m[:, :3, 3], d))
    o0, u0 = views[0]
    o, u = views[1]
    o2, u2 = views[2]
    cc = np.cross(u0, u2)
    sn = np.linalg.norm(cc, axis=1)
    b = o2 - o0
    t0 = (np.cross(b, u2) * cc).sum(1) / np.maximum(sn**2, 1e-8)
    t2 = (np.cross(b, u0) * cc).sum(1) / np.maximum(sn**2, 1e-8)
    A = o0 + t0[:, None] * u0
    B = o2 + t2[:, None] * u2
    X = (A + B) * 0.5
    dist = np.linalg.norm(X - o, axis=1)
    g, hull, _ = vol.lookup(X)
    keep = (
        (sn > 0.012)
        & (t0 > 2)
        & (t2 > 2)
        & (dist > 4)
        & (dist < 210)
        & (np.linalg.norm(unit(X - o) - u, axis=1) * f < 0.85)
        & (np.linalg.norm(B - A, axis=1) / dist * f < 1.4)
        & (X[:, 2] > np.maximum(g + 0.35, hull + 0.25))
        & (X[:, 2] < g + 35)
    )
    X = X[keep]
    q = pix[keep]
    dist = dist[keep]
    if len(X) < 8:
        return None
    # Neighbouring independent feature tracks support a local surface only where their depths and pixel spacing agree. Never triangulate through the sky.
    tri = Delaunay(q).simplices
    D = dist[tri]
    Q = q[tri]
    side = np.max(np.linalg.norm(Q - np.roll(Q, 1, axis=1), axis=2), axis=1)
    x0, y0 = q.round().astype(int).T
    keep = (np.ptp(D, axis=1) < 0.55 + 0.068 * D.min(1)) & (side < 22) & (side > 3)
    tri = tri[keep]
    Q = Q[keep]
    D = D[keep]
    if not len(tri):
        return None
    weights = np.array(
        [[1 / 3] * 3, [0.65, 0.175, 0.175], [0.175, 0.65, 0.175], [0.175, 0.175, 0.65]]
    )
    pix = np.einsum("ij,njk->nik", weights, Q)
    xp, yp = np.round(pix).astype(int).transpose(2, 0, 1)
    ab = mask[yp, xp].all(1)
    Xtri = X[tri]
    normal = unit(np.cross(Xtri[:, 1] - Xtri[:, 0], Xtri[:, 2] - Xtri[:, 0]))
    size = np.linalg.norm(Xtri - np.roll(Xtri, 1, axis=1), axis=2).max(1)
    ab &= (size < 2.8) & (np.linalg.norm(normal, axis=1) > 0.98)
    world = np.einsum("ij,njk->nik", weights, Xtri)[ab].reshape(-1, 3)
    col = im[yp, xp][ab].reshape(-1, 3)
    n = np.repeat(normal[ab], 4, 0)
    rho = np.repeat(D.mean(1)[ab], 4)
    s = np.repeat(np.clip(size[ab] * 0.28, 0.06, 1.1), 4)
    return np.c_[
        world,
        n,
        col,
        rho,
        s,
        np.full(len(n), es[1]["t"]),
        np.full(len(n), es[1]["odo"]),
    ]


def run():
    v = Volume()
    data = J(V / "train.json")
    rows = []
    n = 0
    for c in CAMS:
        for yaw in designs(c):
            es = sorted(
                [e for e in data if e["c"] == c and e["yaw"] == yaw],
                key=lambda e: e["odo"],
            )
            d = np.array([e["odo"] for e in es])
            last = -100
            for i, e in enumerate(es):
                if d[i] - last < 2.8:
                    continue
                last = d[i]
                base = 3.3 if c == "front" else 1.25
                idx = [np.abs(d - (d[i] + j)).argmin() for j in [-base, 0, base]]
                if len(set(idx)) < 3:
                    continue
                a = triplet([es[j] for j in idx], v)
                n += 1
                if a is not None and len(a):
                    rows.append(a)
    a = np.concatenate(rows)
    r = a[:, 9]
    pitch = np.select([r < 18, r < 50, r < 95], [0.18, 0.32, 0.55], 0.88)
    k = np.c_[np.digitize(r, [18, 50, 95]), np.floor(a[:, :3] / pitch[:, None])].astype(
        int
    )
    _, ids = np.unique(k, axis=0, return_inverse=True)
    o = np.argsort(ids)
    s = np.r_[0, np.flatnonzero(np.diff(ids[o])) + 1]
    num = np.diff(np.r_[s, len(a)])
    span = np.maximum.reduceat(a[o, 12], s) - np.minimum.reduceat(a[o, 12], s)
    rgbmean = np.add.reduceat(a[o, 6:9], s) / num[:, None]
    lum = a[o, 6:9].mean(1)
    scatter = np.sqrt(
        np.maximum(0, np.add.reduceat(lum**2, s) / num - rgbmean.mean(1) ** 2)
    )
    ok = (span > 0.8) & (scatter < 19)
    B = a[o[s[ok]]]
    normal = B[:, 3:6]
    t = unit(
        np.cross(
            np.where((abs(normal[:, 2]) > 0.87)[:, None], [0, 1.0, 0], [0, 0, 1.0]),
            normal,
        )
    )
    vax = unit(np.cross(normal, t))
    quat = Rotation.from_matrix(np.stack([t, vax, normal], 2)).as_quat()[
        :, [3, 0, 1, 2]
    ]
    sz = np.clip(B[:, 10], 0.08, np.minimum(1.12, pitch[o[s[ok]]] * 1.38))
    out = surfels(
        B[:, :3],
        rgbmean[ok],
        np.c_[sz, sz, np.maximum(0.018, sz * 0.18)],
        0.78,
        B[:, 11],
    )
    out[:, 3:6] = normal
    out[:, 13:17] = quat
    np.save(W / "patches.npy", out)
    res = dict(
        triplets=n,
        proposed_samples=len(a),
        voxels=len(s),
        repeat_verified=len(out),
        range_hist=np.histogram(B[:, 9], [0, 30, 60, 100, 211])[0].tolist(),
    )
    savej(W / "patches.json", res)
    print(res, flush=True)


if __name__ == "__main__":
    run()
