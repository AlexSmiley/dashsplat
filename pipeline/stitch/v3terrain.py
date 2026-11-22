"""Recover photographically observed verge/kerb surfaces in holes outside the immutable road mosaic."""

from .v3evidence import W
from .v2seed import *


def surfels(xyz, color, scale, alpha, tm):
    a = np.zeros((len(xyz), 24), "f")
    a[:, :3] = xyz
    a[:, 5] = 1
    a[:, 6:9] = (np.clip(color, 0, 255) / 255 - 0.5) / 0.28209479
    a[:, 9] = np.log(alpha / (1 - alpha))
    a[:, 10:13] = np.log(scale)
    a[:, 13] = 1
    a[:, 18:] = 1e9
    a[:, 19] = tm
    return a


def terrain():
    v = Volume()
    route = cKDTree(traj.T_world_ego(np.arange(START, END, 0.12))[:, :2, 3])
    pitch = 0.18
    samples = []
    nframe = 0
    for c in CAMS:
        ray, ok = rays(c)
        last = -100
        for e in schedule()[c]:
            if not e["depth"] or e["hold"] or e["odo"] - last < 0.9:
                continue
            last = e["odo"]
            i = e["n"]
            y, x = np.mgrid[3 + (i % 3) : 466 : 3, 3 + ((i // 3) % 3) : 721 : 3]
            lab = imask(c, i)[y, x]
            rr, wt = exposure_range(e)
            d = rr[y, x]
            im = rgb(c, i)[y, x].astype("f")
            row = (8 * y / 469).astype(int)
            M = poses(c, e["t"])[row]
            u = np.einsum("...ij,...j->...i", M[..., :3, :3], ray[y, x])
            o = M[..., :3, 3]
            ok = (
                (lab == 1)
                & (wt[y, x] > 0.05)
                & (u[:, :, 2] < -0.057)
                & (d > 1.4)
                & (d < 28)
            )
            u, o, d, im = [z[ok] for z in [u, o, d, im]]
            X = o + u * d[:, None]
            g, h, owned = v.lookup(X)
            dz = X[:, 2] - g
            dg = (g - o[:, 2]) / np.minimum(-0.05, u[:, 2])
            sel = (
                (abs(dz) < 0.46)
                & (abs(d - dg) < 0.19 * d + 0.25)
                & (dg > 1.4)
                & (dg < 27)
            )
            X = X[sel]
            u = u[sel]
            o = o[sel]
            dg = dg[sel]
            dz = dz[sel]
            im = im[
                sel
            ]  # depth offsets are trusted only for shallow curbs; colour votes must agree from displaced photographs.
            X = o + u * (dg + np.clip(dz, -0.06, 0.24) / u[:, 2])[:, None]
            g, h, owned = v.lookup(X)
            take = (
                (~owned.astype(bool))
                & (h < g)
                & (X[:, 2] < g + 0.38)
                & (X[:, 2] > g - 0.2)
            )
            X = X[take]
            im = im[take]
            ix = np.floor(X[:, :2] / pitch).astype("i8")
            code = ix[:, 0] * 100000 + ix[:, 1]
            order = np.argsort(code)
            start = np.r_[0, np.flatnonzero(np.diff(code[order])) + 1]
            count = np.diff(np.r_[start, len(order)])
            if not len(order):
                continue
            col = np.add.reduceat(im[order], start) / count[:, None]
            xyz = np.add.reduceat(X[order], start) / count[:, None]
            samples.append(
                np.c_[xyz, col, np.full(len(xyz), last), np.full(len(xyz), e["t"])]
            )
            nframe += 1
    a = np.concatenate(samples)
    ij = np.floor(a[:, :2] / pitch).astype("i8")
    key = ij[:, 0] * 100000 + ij[:, 1]
    o = np.argsort(key)
    a = a[o]
    s = np.r_[0, np.flatnonzero(np.diff(key[o])) + 1]
    cnt = np.diff(np.r_[s, len(a)])
    mean = np.add.reduceat(a, s) / cnt[:, None]
    lum = a[:, 3:6].mean(1)
    scatter = np.sqrt(
        np.maximum(0, np.add.reduceat(lum**2, s) / cnt - mean[:, 3:6].mean(1) ** 2)
    )
    span = np.maximum.reduceat(a[:, 6], s) - np.minimum.reduceat(a[:, 6], s)
    v0, h0, own = v.lookup(mean[:, :3])
    lat = route.query(mean[:, :2])[0]
    good = (
        (cnt >= 3)
        & (span >= 1.05)
        & (scatter < 25)
        & (h0 < v0)
        & (~own.astype(bool))
        & (lat > 2.65)
        & (lat < 32)
    )
    b = mean[good]
    sig = pitch * 0.85 * np.ones((len(b), 3), "f")
    sig[:, 2] = 0.018
    out = surfels(b[:, :3], b[:, 3:6], sig, 0.94, np.maximum.reduceat(a[:, 7], s)[good])
    np.save(W / "verge.npy", out)
    savej(
        W / "terrain.json",
        dict(
            frames=nframe,
            proposed=len(s),
            retained=len(b),
            pitch_m=pitch,
            median_distinct_exposures=float(np.median(cnt[good])),
            median_baseline=float(np.median(span[good])),
        ),
    )
    print(J(W / "terrain.json"), flush=True)


if __name__ == "__main__":
    terrain()
