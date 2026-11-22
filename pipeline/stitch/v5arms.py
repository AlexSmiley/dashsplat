"""Recover skyline arms attached to measured columns; require a contrast ridge on both sides in displaced photos."""

from .v5structures import *


def recover():
    rr = [
        r
        for r in J(W5 / "inventory.json")["records"]
        if r["class"] == CATS[5]
        and r["geometry"]
        and r["views"] >= 7
        and r["top"] - r["floor"] > 4
    ]
    records = []
    parts = []
    for r in rr:
        top = np.array([*r["xy"], r["top"]])
        theta = np.arange(40) * np.pi / 20
        dirs = np.c_[np.cos(theta), np.sin(theta)]
        angles = np.array([-0.04, 0.12, 0.30])
        length = np.arange(0.24, 8.05, 0.24)
        rad = np.tile(length, 40 * 3 * 3)
        di = np.repeat(np.repeat(dirs, 3 * 3, axis=0), len(length), axis=0)
        dz = np.repeat(np.tile(np.repeat(angles, 3), 40), len(length))
        bias = np.repeat(np.tile([-0.6, 0, 0.65], 120), len(length))
        P = top + np.c_[di * rad[:, None], bias + dz * rad]
        P1 = P + np.c_[di * 0.15, dz * 0.15]
        cnt = np.zeros(len(P))
        tot = cnt.copy()
        color = np.zeros((len(P), 3))
        last = {}
        mn = np.ones(len(P)) * 1e6
        mx = -mn
        for c in CAMS:
            for e in semframes(c):
                if e["odo"] - last.get(c, -1e10) < 4.0:
                    continue
                T = poses(c, e["t"])[4]
                d = np.linalg.norm(top - T[:3, 3])
                if d > 85 or d < 5:
                    continue
                uv, ok, _ = project(c, e, top[None])
                if not ok[0]:
                    continue
                last[c] = e["odo"]
                C = rig[c]
                uv, vis, _ = project(c, e, P)
                uv1, _, _ = project(c, e, P1)
                uv *= np.array([1448 / C.width, 938 / C.height])
                uv1 *= np.array([1448 / C.width, 938 / C.height])
                perp = unit((uv1 - uv)[:, ::-1]) * [-1, 1]
                pix = uv.round().astype(int)
                x, y = pix.T
                v = vis & (x > 6) & (x < 1441) & (y > 6) & (y < 931)
                lab = labels(c, e["n"])
                l = lab[(y // 2).clip(0, 468), (x // 2).clip(0, 723)]
                im = cv.imread(str(RS / "data/drive/frames" / c / f"{e['n']:06}.jpg"))
                im = cv.resize(im, (1448, 938), interpolation=cv.INTER_AREA)
                lum = cv.cvtColor(im, cv.COLOR_BGR2GRAY).astype("f")

                def sample(q):
                    x, y = q.round().astype(int).T
                    return lum[y.clip(0, 937), x.clip(0, 1447)]

                a = sample(uv + perp * 3.5)
                b = sample(uv - perp * 3.5)
                mid = np.minimum.reduce([sample(uv + perp * i) for i in [-0.8, 0, 0.8]])
                visible = v & np.isin(l, [5, 6, 7, 10, 255])
                yes = visible & (mid + 10 < np.minimum(a, b)) & (abs(a - b) < 42)
                cnt += yes
                tot += visible
                color[yes] += im[y[yes], x[yes], ::-1]
                mn[yes] = np.minimum(mn[yes], e["odo"])
                mx[yes] = np.maximum(mx[yes], e["odo"])
        shape = (40, 3, 3, len(length))
        good = ((cnt >= 4) & (cnt / np.maximum(1, tot) > 0.46) & (mx - mn > 3)).reshape(
            shape
        )
        score = np.cumprod(good.astype(int), axis=3).sum(3)
        order = np.argsort(-score.ravel())
        accepted = []
        for j in order:
            ai, zi, bi = np.unravel_index(j, score.shape)
            n = int(score[ai, zi, bi])
            if n < 5:
                break
            if any(min((ai - k) % 40, (k - ai) % 40) < 7 for k in accepted):
                continue
            lo = length[0]
            hi = length[n - 1]
            sl = np.arange(n) + j * len(length)
            q = np.arange(0, hi + 0.01, 0.070)
            direction = np.r_[dirs[ai], angles[zi]]
            X = top + np.r_[0, 0, [-0.6, 0, 0.65][bi]] + q[:, None] * direction
            C = color[sl] / np.maximum(1, cnt[sl, None])
            C = np.stack([np.interp(q, length[:n], C[:, k]) for k in range(3)], 1)
            B = surfels(X, C, [0.052, 0.052, 0.065], 0.94, r["time"])
            parts.append(B)
            accepted.append(ai)
            records.append(
                dict(
                    track=r["id"],
                    length_m=round(hi, 2),
                    direction=direction.tolist(),
                    start=X[0].tolist(),
                    end=X[-1].tolist(),
                    min_views=int(cnt[sl].min()),
                    min_baseline=float(np.min(mx[sl] - mn[sl])),
                )
            )
            if len(accepted) == 2:
                break
    np.save(W5 / "arms.npy", np.concatenate(parts) if parts else np.empty((0, 24), "f"))
    savej(W5 / "arms.json", dict(tested=len(rr), arms=records))
    print(
        "arms",
        len(records),
        "on",
        len(set(r["track"] for r in records)),
        "of",
        len(rr),
        flush=True,
    )


if __name__ == "__main__":
    recover()
