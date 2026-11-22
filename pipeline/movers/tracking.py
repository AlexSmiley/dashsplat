"""Robust image-space fit of a rolling rigid vehicle on the local road tangent."""

import numpy as np
from scipy.optimize import least_squares
from .inventory import lift, SIZE
from .shape import silhouette
from pipeline.common.rig import Trajectory, load_rig


class Factors:
    def __init__(self, rows, cls, variant):
        self.rows = rows
        self.cls = cls
        self.tr = Trajectory(
            "poses/ego" + ("_causal" if variant == "live" else "") + ".npz"
        )
        self.cs = load_rig()
        self.t = np.array([r["rt"] for r in rows])
        self.release = np.array([r["release"] for r in rows])
        self.bb = np.array([r["bb"] for r in rows])
        self.c = np.array([r["c"] for r in rows])
        self.n = len(rows)
        self.size = np.array(SIZE[cls])
        self.outline = silhouette(cls, self.size)
        self.g = np.zeros((self.n, 3))
        self.plane = np.zeros((self.n, 3))
        self.R = np.zeros((self.n, 3, 3))
        self.O = self.g.copy()
        self.p = np.zeros((self.n, 8))
        self.hz = self.t.copy()
        self.dt = self.t.copy()
        self.tf = np.array([r["t"] for r in rows])
        self.d = np.array([r["depth"] for r in rows])
        self.V = self.g.copy()
        self.omega = self.g.copy()
        self.factor = np.zeros(self.n)
        self.first = [r["key"] for r in rows]
        heights = np.load(
            "maps/ground-work/"
            + ("live" if variant == "live" else "stitched")
            + "-height.npy"
        )
        for c, cam in self.cs.items():
            i = np.flatnonzero(self.c == c)
            if not len(i):
                continue
            G, d, C, nw, E, h = lift(
                cam, self.tr, self.tf[i], np.array([rows[k]["foot"] for k in i])
            )
            self.g[i] = G
            self.R[i] = C[:, :3, :3]
            self.O[i] = C[:, :3, 3]
            self.p[i] = cam.params * np.array(
                [
                    724 / cam.width,
                    469 / cam.height,
                    724 / cam.width,
                    469 / cam.height,
                    1,
                    1,
                    1,
                    1,
                ]
            )
            self.plane[i, :2] = -nw[:, :2] / nw[:, 2, None]
            self.plane[i, 2] = (
                E[:, 2, 3] - (self.plane[i, :2] * E[:, :2, 3]).sum(1) - h / nw[:, 2]
            )
            self.factor[i] = cam.rs_readout_s / 469
            self.hz[i] = [rows[k]["foot"][1] for k in i]
            vel, omega = self.tr.velocity(self.t[i])
            self.V[i] = vel + np.cross(omega, C[:, :3, 3] - E[:, :3, 3])
            self.omega[i] = omega
            for k in i:
                hit = heights[
                    (heights[:, 0] == np.floor(self.g[k, 0] / 1.5))
                    & (heights[:, 1] == np.floor(self.g[k, 1] / 1.5))
                ]
                hit = hit[hit[:, 2] <= self.t[k]] if variant == "live" else hit
                if len(hit):
                    self.plane[k, 2] += hit[-1, 3]

    def height(self, i, xy):
        return (self.plane[i, :2] * xy).sum(-1) + self.plane[i, 2]

    def project(self, i, world):
        local = np.einsum("nki,nij->nkj", world - self.O[i, None, :], self.R[i])
        rho = np.linalg.norm(local[:, :, :2], axis=2)
        th = np.arctan2(rho, local[:, :, 2])
        th2 = th * th
        p = self.p[i]
        a, b, c, d = p[:, 4:, None].transpose(1, 0, 2)
        f = (
            th
            * (1 + th2 * (a + th2 * (b + th2 * (c + th2 * d))))
            / np.maximum(1e-12, rho)
        )
        return local[:, :, :2] * f[:, :, None] * p[:, None, :2] + p[
            :, None, 2:4
        ], local[:, :, 2]

    def edges(self, i, ctr, vel, yaw, scale):
        v = self.outline * scale
        cs, sn = np.cos(yaw), np.sin(yaw)
        rot = np.stack(
            [cs * v[:, 0] - sn * v[:, 1], sn * v[:, 0] + cs * v[:, 1], v[:, 2]], axis=1
        )
        xyz = np.c_[ctr, self.height(i, ctr)]
        world = xyz[:, None, :] + rot[None, :, :]
        uv, z = self.project(i, world)
        dt = (uv[:, :, 1] - self.hz[i, None]) * self.factor[i, None]
        shift = (
            np.r_[vel, 0][None, None, :]
            - self.V[i, None, :]
            - np.cross(self.omega[i, None, :], world - self.O[i, None, :])
        )
        uv, z = self.project(i, world + dt[:, :, None] * shift)
        return np.c_[uv.min(1), uv.max(1)], z.min(1)

    def fit(self, i, t0, start=None, prior=None, sz=None, stationary=False):
        if len(i) > 58:
            i = i[np.unique(np.linspace(0, len(i) - 1, 58).astype(int))]
        tt = self.t[i] - t0
        bb = self.bb[i]
        wh = np.maximum(bb[:, 2:] - bb[:, :2], 3)
        w = np.c_[np.maximum(2, wh[:, 0] * 0.08), np.maximum(1.3, wh[:, 1] * 0.063)]
        w = np.c_[w, w]
        w *= np.where(np.array(self.first)[i, None], 1, 1.3)
        g = self.g[i, :2]
        usable = np.linalg.norm(g - self.O[i, :2], axis=1) < 130
        if start is None:
            ii = np.where(usable)[0]
            ii = ii if len(ii) >= 3 else np.arange(len(i))
            slope, base = np.linalg.lstsq(
                np.c_[tt[ii], np.ones(len(ii))], g[ii], rcond=None
            )[0]
            slope *= min(1, 28 / max(0.01, np.linalg.norm(slope)))
            start = np.r_[base, slope]
        yaw0 = np.arctan2(start[3], start[2])
        x0 = np.r_[
            start[:4], ([yaw0] if stationary else []), ([1, 1, 1] if sz is None else [])
        ]
        shape = sz is None

        def res(x, raw=False):
            vel = x[2:4] if not stationary else np.zeros(2)
            yaw = x[4] if stationary else np.arctan2(vel[1], vel[0])
            scale = x[-3:] if shape else sz
            xy = x[:2] + tt[:, None] * vel
            b, z = self.edges(
                i, xy, vel, yaw, scale
            )  # clipped or pole-split masks only bound the visible silhouette
            b = np.clip(b, [0.5, 0.5, 0.5, 0.5], [723.5, 468.5, 723.5, 468.5])
            e = (b - bb) / w
            e *= np.where(e * np.array([-1, -1, 1, 1]) > 0, 0.79, 1.0)
            extras = []
            # An observed tyre/contact lies within the oriented footprint. Range uncertainty
            # grows quadratically near the horizon. This also breaks the head/side-view ambiguity.
            D = g - xy
            cs, sn = np.cos(yaw), np.sin(yaw)
            along = D[:, 0] * cs + D[:, 1] * sn
            across = -D[:, 0] * sn + D[:, 1] * cs
            dist = np.linalg.norm(g - self.O[i, :2], axis=1)
            sigma = 0.55 + 0.0017 * np.minimum(dist, 200) ** 2
            extras.extend(
                (np.maximum(0, abs(along) - 0.5 * self.size[0] * scale[0]) / sigma)[
                    usable
                ]
            )
            extras.extend(
                (np.maximum(0, abs(across) - 0.5 * self.size[1] * scale[1]) / sigma)[
                    usable
                ]
            )
            if prior is not None:
                extras.extend((x[:4] - prior) / [2.6, 2.6, 4.0, 4.0])
            if shape:
                extras.extend(np.sqrt(len(i)) * (scale - 1) / [0.09, 0.07, 0.08])
            if raw:
                return e
            return np.r_[e.ravel(), extras, np.maximum(0, np.abs(vel) - 32) * 1.3]

        lo = np.r_[
            [-np.inf] * 4,
            ([-np.inf] if stationary else []),
            ([0.76, 0.80, 0.76] if shape else []),
        ]
        hi = np.r_[
            [np.inf] * 4,
            ([np.inf] if stationary else []),
            ([1.30, 1.20, 1.27] if shape else []),
        ]
        sol = least_squares(
            res, x0, bounds=(lo, hi), loss="soft_l1", f_scale=1.2, max_nfev=35
        )
        return (
            sol.x[:4],
            sol.x[-3:] if shape else sz,
            float(np.median(abs(res(sol.x, True)))),
        )


def make_track(f, variant):
    if variant == "live":
        return live_track(f)
    # A car that is stationary relative to the road must explain its pixel motion better
    # with a constant location/free orientation than a translating rigid template.
    whole = np.arange(f.n)
    midpoint = np.mean(f.t)
    glob, sc, ed = f.fit(whole, midpoint)
    _, _, es = f.fit(whole, midpoint, glob, sz=sc, stationary=True)
    if np.linalg.norm(glob[2:]) < 1.75 or es < max(0.8, ed * 1.3):
        return None
    knots = []
    prev = None
    end = f.release[-1]
    h = 0.083333
    for t in np.arange(f.release.min(), end + 0.75, h):
        cutoff = t + 1.05
        i = np.flatnonzero((f.release <= cutoff) & (f.t >= t - 1.05))
        predict = f.n and (len(i) < 4 or t - f.release[i[-1]] > 0.20)
        if (not len(i) or np.ptp(f.t[i]) < 0.18) and prev is None:
            continue
        prior = prev.copy() if prev is not None else None
        if prior is not None:
            prior[:2] += prior[2:] * h
        if predict or len(i) < 5:
            if prev is None:
                continue
            cur = prior
        else:
            seed = np.r_[glob[:2] + glob[2:] * (t - midpoint), glob[2:]]
            cur, _, cost = f.fit(i, t, seed, prior, sc)
            if prior is not None and cost > 0.8:
                alt, _, err = f.fit(i, t, prior, prior, sc)
                if err < cost:
                    cur, cost = alt, err
            if prior is not None:
                # Bounded innovation, separate tangential/angular smoothing from the image update.
                delta = cur[:2] - prior[:2]
                cur[:2] = prior[:2] + delta * 0.72
                dv = cur[2:] - prior[2:]
                cur[2:] = prior[2:] + dv * 0.55
        prev = cur
        knots.append((t, *cur))
    if len(knots) < 4:
        return None
    k = np.array(knots)
    ts = np.arange(k[0, 0], k[-1, 0] + 1e-6, 1 / 36)
    xy = np.stack([np.interp(ts, k[:, 0], k[:, j]) for j in (1, 2)], axis=1)
    yaw = np.interp(ts, k[:, 0], np.unwrap(np.arctan2(k[:, 4], k[:, 3])))
    recent = np.searchsorted(f.release, ts, side="right") - 1
    recent = np.maximum(0, recent)
    z = f.height(recent, xy)
    state = ((ts - f.release[recent] > 0.043) | (ts < f.release[0])).astype(int)
    return (
        dict(t=ts, x=xy[:, 0], y=xy[:, 1], z=z, yaw=yaw, state=state),
        sc,
        dict(moving_residual=ed, stationary_residual=es, velocity=glob[2:].tolist()),
    )


def released_rows(rows):
    """Complete shutter + both ego samples used by row interpolation and twist."""
    cs = load_rig()
    tl = Trajectory("poses/ego_causal.npz").t
    avail = tl + np.load("work/road/causal-lag.npy")
    out = []
    for r0 in rows:
        r = r0.copy()
        cam = cs[r["c"]]
        mid = r["t"] + cam.time_offset_s
        half = 0.5 * abs(cam.rs_readout_s or 0)
        i = max(0, np.searchsorted(tl, mid - half, side="right") - 1)
        j = min(len(tl), np.searchsorted(tl, mid + half, side="right") + 1)
        r["release"] = float(max(r["release"], mid + half, avail[i:j].max()))
        out.append(r)
    return sorted(out, key=lambda r: (r["release"], r["rt"]))


def live_track(f):
    # Filter at 12 Hz, advance a display pose at 36 Hz. Between updates the
    # viewer moves towards the last released fit, never towards the next one.
    h = 1 / 36
    times = (
        np.arange(np.ceil(f.release[0] * 36), np.floor((f.release[-1] + 0.72) * 36) + 1)
        / 36
    )
    base = None
    anchor = 0.0
    last_i = -1
    rows = []
    fits = []
    sc = np.ones(3)
    motion = None
    for j, t in enumerate(times):
        last = np.searchsorted(f.release, t, side="right") - 1
        if last < 0:
            continue
        if j % 3 == 0 and last != last_i:
            last_i = last
            i = np.flatnonzero((f.release <= t) & (f.t >= t - 1.1))
            if len(i) >= 5 and np.ptp(f.t[i]) >= (0.46 if base is None else 0.22):
                prior = (
                    np.r_[base[:2] + (t - anchor) * base[2:], base[2:]]
                    if base is not None
                    else None
                )
                cur, _, cost = f.fit(i, t, None, prior, sc)
                if prior is not None and cost > 0.8:
                    alt, _, err = f.fit(i, t, prior, prior, sc)
                    if err < cost:
                        cur, cost = alt, err
                if base is None:
                    if np.linalg.norm(cur[2:]) < 2.3:
                        continue
                    _, _, stat = f.fit(i, t, cur, sz=sc, stationary=True)
                    if stat < max(0.95, cost * 1.4):
                        continue
                    motion = dict(
                        moving_residual=cost,
                        stationary_residual=stat,
                        velocity=cur[2:].tolist(),
                    )
                    xy = cur[:2].copy()
                    vel = cur[2:].copy()
                else:
                    cur[:2] = prior[:2] + 0.92 * (cur[:2] - prior[:2])
                    cur[2:] = prior[2:] + 0.72 * (cur[2:] - prior[2:])
                base = cur
                anchor = t
                fits.append((t, *cur, f.height(last, cur[:2])))
        if base is None:
            continue
        target = base[:2] + (t - anchor) * base[2:]
        if rows:
            vel += 0.38 * (base[2:] - vel)
            xy += vel * h
            delta = target - xy
            xy += delta * min(0.64, 1.2 / max(1e-6, np.linalg.norm(delta)))
        state = int(t - f.release[last] > 0.09)
        rows.append((t, *xy, f.height(last, xy), np.arctan2(vel[1], vel[0]), state))
    if len(rows) < 4:
        return None
    tr = {k: v for k, v in zip(("t", "x", "y", "z", "yaw", "state"), np.array(rows).T)}
    tr["state"] = tr["state"].astype(int)
    scan = dict(zip(("t", "x", "y", "vx", "vy", "z"), np.array(fits).T))
    return tr, sc, motion, scan


def playback(track, mode):
    """Delay live knots one tick so both interpolation endpoints are released.

    Advance x/y at the last observable display velocity to compensate the delay.
    Texture fits remain on their own release clock.
    """
    if mode != "live":
        return track
    h = 1 / 36
    out = dict(track, t=track["t"] + h)
    for key in ("x", "y"):
        step = np.diff(track[key], prepend=track[key][0])
        out[key] = track[key] + step
    return out
