"""Fit vertical office faces by multiview photometric plane sweep when metric tracks are too sparse at 200–400 m."""

from .v5structures import *


@lru_cache(48)
def eye(c, n):
    a = cv.imread(
        str(RS / "data/drive/frames" / c / f"{n:06}.jpg"), cv.IMREAD_REDUCED_COLOR_2
    )
    a = photo.apply(c, n, a[:, :, ::-1])
    g = cv.cvtColor(a, cv.COLOR_RGB2GRAY).astype("f")
    edge = g - cv.GaussianBlur(g, (0, 0), 2.7)
    return a, edge


def fit(name, t, rect):
    c = "front"
    C = rig[c]
    es = [e for e in schedule()[c] if not e["hold"]]
    base = min([e for e in es if semantic(c, e["n"])], key=lambda e: abs(e["t"] - t))
    candidates = [
        min(es, key=lambda e: abs(e["t"] - (base["t"] + dt)))
        for dt in [-2.2, -1.45, -0.85, -0.48, 0.48, 0.85, 1.45, 2.2]
    ]
    poses1 = [poses(c, e["t"])[3] for e in candidates]
    ims = [eye(c, e["n"])[1] for e in candidates]
    im, edge = eye(c, base["n"])
    lab = labels(c, base["n"])
    x0, y0, x1, y1 = rect
    mask = np.zeros_like(lab, "B")
    mask[y0:y1, x0:x1] = 1
    mask &= lab == 2
    mask = cv.erode(mask, np.ones((3, 3), "B"))
    y, x = np.nonzero(cv.resize(mask, (1448, 938), interpolation=0) & (abs(edge) > 2))
    rng = np.random.default_rng(714)
    take = rng.permutation(len(x))[:950]
    x = x[take]
    y = y[take]
    if len(x) < 70:
        return [], dict(name=name, reason="insufficient anchors", anchors=len(x))
    O, U = rays_at(c, base, (x + 0.5) / 2 - 0.5, (y + 0.5) / 2 - 0.5)
    ref = edge[y, x]
    centre = O.mean(0)
    forward = unit(U.mean(0)[:2])
    K = C.params
    angles = np.linspace(-79, 79, 39)
    depths = np.geomspace(65, 520, 49)
    cameras = [(M[:3, 3], M[:3, :3]) for M in poses1]

    def score(angle, r):
        theta = np.deg2rad(angle)
        N = np.array(
            [
                forward[0] * np.cos(theta) - forward[1] * np.sin(theta),
                forward[0] * np.sin(theta) + forward[1] * np.cos(theta),
            ]
        )
        mid = centre[:2] + r * forward
        den = U[:, :2] @ N
        z = ((mid - O[:, :2]) @ N) / np.maximum(0.02, den)
        P = O + z[:, None] * U
        sc = []
        for (o, R), g in zip(cameras, ims):
            q = (P - o) @ R
            rad = np.linalg.norm(q[:, :2], axis=1)
            a = np.arctan2(rad, q[:, 2])
            u = a * a
            mul = (
                a
                * (1 + u * (K[4] + u * (K[5] + u * (K[6] + u * K[7]))))
                / np.maximum(0.0001, rad)
            )
            px = (q[:, :2] * mul[:, None] * K[:2] + K[2:4]) / 2 - 0.5
            ix, iy = px.round().astype(int).T
            ok = (ix > 0) & (iy > 0) & (ix < 1447) & (iy < 937) & (q[:, 2] > 0)
            target = g[iy.clip(0, 937), ix.clip(0, 1447)]
            v = ref[ok].clip(-42, 42)
            w = target[ok].clip(-42, 42)
            sc.append(
                float((v @ w) / (np.linalg.norm(v) * np.linalg.norm(w) + 0.01))
                if len(v) > len(ref) * 0.65
                else -0.5
            )
        return np.mean(np.sort(sc)[2:]), N, mid, sc

    best = (-2, None, None)
    for a in angles:
        for r in depths:
            s = score(a, r)[0]
            if s > best[0]:
                best = (s, a, r)
    _, a, r = best
    for angle in np.linspace(a - 5, a + 5, 12):
        for dist in np.linspace(r * 0.955, r * 1.045, 22):
            s = score(angle, dist)[0]
            if s > best[0]:
                best = (s, angle, dist)
    score0, angle, r = best
    _, N, mid, tests = score(angle, r)
    info = dict(
        name=name,
        t=base["t"],
        source=base["n"],
        anchors=len(ref),
        range_m=float(r),
        angle_deg=float(angle),
        ncc=score0,
        view_ncc=tests,
        normal=N.tolist(),
        xy=mid.tolist(),
    )
    print(info, flush=True)
    if score0 < 0.25:
        return [], dict(info, texels=0)
    # One world-space grid per fitted plane, sampled from one registered physical exposure. Verify every texel in other physical exposures, preserving sky holes.
    xx, yy = np.meshgrid([x0, x1], [y0, y1])
    o, u = rays_at(c, base, xx.ravel(), yy.ravel())
    d = ((mid - o[:, :2]) @ N) / (u[:, :2] @ N)
    corn = o + d[:, None] * u
    tx = np.array([-N[1], N[0]])
    bounds = (corn[:, :2] - mid) @ tx
    step = np.clip(r / C.params[0] * 0.96, 0.045, 0.115)
    uu, zz = np.meshgrid(
        np.arange(bounds.min(), bounds.max(), step),
        np.arange(corn[:, 2].min(), corn[:, 2].max(), step),
    )
    P = np.c_[mid + uu.ravel()[:, None] * tx, zz.ravel()]
    uv, seen, _ = project(c, base, P)
    x, y = (uv / 4).astype(int).T
    ok = (
        seen
        & (x >= x0)
        & (x < x1)
        & (y >= y0)
        & (y < y1)
        & (lab[y.clip(0, 468), x.clip(0, 723)] == 2)
    )
    P, uv = P[ok], uv[ok]
    q = np.zeros(len(P), int)
    neg = q.copy()
    dis = []
    for e in [
        e
        for e in semframes(c)
        if abs(e["t"] - base["t"]) < 10 and abs(e["odo"] - base["odo"]) > 1.3
    ]:
        v, vis, rr = project(c, e, P)
        x, y = (v / 4).astype(int).T
        l = labels(c, e["n"])[y.clip(0, 468), x.clip(0, 723)]
        q += vis & (l == 2)
        neg += vis & (l == 10)
        dis.append(e["odo"])
    ok = (q >= 3) & (neg < 0.35 * q)
    P = P[ok]
    uv = uv[ok]
    native = cv.imread(str(RS / "data/drive/frames" / c / f"{base['n']:06}.jpg"))[
        :, :, ::-1
    ]
    x, y = uv.astype(int).T
    col = photo.sample(c, base["n"], uv, native[y, x])
    A = surfels(P, col, [step * 0.73, step * 0.73, 0.026], 0.979, t)
    n = np.r_[N, 0]
    A[:, 3:6] = n
    A[:, 13:17] = Rotation.from_matrix(
        np.stack([np.r_[tx, 0], [0, 0, 1], n], 1)
    ).as_quat()[[3, 0, 1, 2]]
    return [A], dict(
        info,
        texels=len(A),
        confirming_views=len(dis),
        baseline_m=float(np.ptp(dis)) if dis else 0,
    )


if __name__ == "__main__":
    parts = []
    rec = []
    for x in [
        ("brick-west", 112, (320, 162, 407, 229)),
        ("brick-east", 112, (408, 162, 572, 241)),
        ("white-west", 124, (385, 171, 428, 225)),
        ("white-east", 124, (426, 169, 487, 226)),
    ]:
        a, r = fit(*x)
        parts += a
        rec.append(r)
    np.save(
        W5 / "long-planes.npy",
        np.concatenate(parts) if parts else np.empty((0, 24), "f"),
    )
    savej(W5 / "long-planes.json", rec)
