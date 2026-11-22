"""M2: a shared chase depth chart, calibrated same-instant lens ownership, exact ground, no radiance blends."""

from . import morph_sources as ms
from .morph_chase import floor, pair, plane
from .morph_ghost import refmask, rect_for, audit
from .morph_ownership import cut, grad
from . import horizon_proof as hp
from .morph_sources import (
    RS,
    TR,
    CAMS,
    GW,
    GH,
    OW,
    OH,
    L,
    H,
    XY,
    seq,
    source,
    rig,
    coord,
    track,
    choose,
    resize,
    remap,
    proj,
    blur,
    smooth,
    savej,
    np,
    cv,
    lru_cache,
)
import ctypes as ct

OUT = RS / "work/ibr/m2"
OUT.mkdir(parents=True, exist_ok=True)
ALL = (0, 0, OW, OH)


@lru_cache(None)
def raster():
    lib = ct.CDLL(str(OUT / "m2_surface.dylib"))
    fn = lib.surface
    fn.argtypes = [ct.c_void_p] + [ct.c_int] * 4 + [ct.c_void_p]
    fn.restype = None
    return fn


@lru_cache(None)
def unit(c):
    C = rig[c]
    uv = (coord(GW, GH) + 0.5) * [C.width / GW, C.height / GH]
    v = C.unproject(uv)
    return (v / v[..., 2:]).astype("f")


@lru_cache(48)
def fields(c, i):
    S, A, img = source(c, i)
    A = A.copy()
    C = rig[c]
    t, n = seq(c)[0][i], int(seq(c)[1][i])
    p = RS / "maps/movers-v1/remove" / c / f"{n:06}.png"
    mover = p.is_file()
    if mover:
        m = cv.imread(str(p), cv.IMREAD_GRAYSCALE)
        assert m is not None
        A[..., 3] = cv.dilate(
            cv.resize((m != 0).astype("f"), (GW, GH), interpolation=cv.INTER_AREA),
            np.ones((3, 3), "B"),
        )
    # The source row is physical, including effective clock offset and readout.
    rows = (
        TR.T_world_ego(C.row_time(t, (np.arange(GH) + 0.5) * C.height / GH))
        @ C.T_ego_cam
    )
    rays = np.einsum("hij,hwj->hwi", rows[:, :3, :3], unit(c)).astype("f")
    valid = cv.resize(C.mask.astype("f"), (GW, GH), interpolation=cv.INTER_AREA) > 0.94
    return S, A, img, rows.astype("f"), rays, valid, mover


class View:
    def __init__(self, t, T, deg):
        self.t, self.T = t, T
        self.f = 450 / np.tan(np.deg2rad(deg / 2))
        self.ray = np.dstack(
            [(XY + 0.5 - [L / 2, H / 2]) / (self.f / 5), np.ones((H, L))]
        ).astype("f")
        self.wr = self.ray @ T[:3, :3].T
        E = TR.T_world_ego(t)
        n, h = TR.ground_plane(t)
        self.normal = E[:3, :3] @ n
        self.offset = h - E[:3, 3] @ self.normal
        self.gr = plane(t, T, self.wr)

    def sheet(self, c, i):
        S, A, img, rows, rays, valid, mover = fields(c, i)
        g = np.maximum(
            0,
            -(rays @ self.normal)
            / (rows[:, :3, 3] @ self.normal + self.offset)[:, None],
        ).astype("f")
        # Far relief + the measured near static matte. Keep ground (including planar verges)
        # and the silhouette in the same continuous UV chart. The narrow skirt interpolates
        # geometry, never RGB from two sides of a discontinuity.
        far = blur(np.clip(A[..., 0], 0, 0.047), 1.15)
        near = blur(A[..., 2], 0.8)
        iv = far + (np.clip(A[..., 1], 0.033, 0.17) - far) * near
        iv = np.maximum(iv, g)
        road = A[..., 4] > 0.45
        iv[road] = g[road]
        iv = np.maximum(iv, 1e-5)
        # A closer branch above a farther foot would fold downward on lifting the eye.
        # Continue its measured depth for a short skirt until the ground catches it.
        fy = rig[c].params[1] * GH / rig[c].height
        step = 0.83 / (fy * max(2.5, self.T[2, 3] - S[2, 3]))
        for y in range(1, GH):
            iv[y] = np.where(road[y], g[y], np.maximum(iv[y], iv[y - 1] - step))
        xyz = (
            rays
            + (rows[:, :3, 3, None].transpose(0, 2, 1) - self.T[:3, 3]) * iv[..., None]
        ) @ self.T[:3, :3]
        zz = xyz[..., 2]
        uv = xyz[..., :2] / np.maximum(0.025, zz[..., None]) * (self.f / 5) + [
            L / 2 - 0.5,
            H / 2 - 0.5,
        ]
        v = np.ascontiguousarray(
            np.dstack(
                [
                    uv,
                    iv / np.maximum(0.025, zz),
                    (zz > 0.025) & valid & (A[..., 3] < 0.12),
                ]
            ),
            "f",
        )
        z = np.full((H, L), -1, "f")
        raster()(v.ctypes.data, GW, GH, L, H, z.ctypes.data)
        return z

    def project(self, c, i, z):
        S, A, im, rows, rays, valid, mover = fields(c, i)
        dr = self.wr @ S[:3, :3]
        off = (self.T[:3, 3] - S[:3, 3]) @ S[:3, :3]
        p = proj(c, dr + off * z[..., None], GW, GH)
        # Interpolate row-pose between bands and re-evaluate at the projected source row.
        for _ in range(2):
            y = np.clip(p[..., 1], 0, GH - 1)
            k = y.astype("i")
            l = np.minimum(k + 1, GH - 1)
            a = y - k
            xyz = []
            for j in [k, l]:
                rel = self.wr + (self.T[:3, 3] - rows[j, :3, 3]) * z[..., None]
                xyz.append(np.einsum("hwi,hwij->hwj", rel, rows[j, :3, :3]))
            ray = xyz[0] + (xyz[1] - xyz[0]) * a[..., None]
            p = proj(c, ray, GW, GH)
        border = np.minimum.reduce(
            [p[..., 0], GW - 1 - p[..., 0], p[..., 1], GH - 1 - p[..., 1]]
        )
        ok = (
            (border > 0.45)
            & (ray[..., 2] > 0.06)
            & (remap(valid.astype("f"), p) > 0.9)
            & (remap(A[..., 3], p) < 0.10)
        )
        uv = ((p + 0.5) * [im.shape[1] / GW, im.shape[0] / GH] - 0.5).astype("f")
        col = remap(im, uv)
        return dict(
            c=c,
            i=i,
            n=int(seq(c)[1][i]),
            uv=uv,
            ok=ok,
            border=border,
            img=im,
            col=col,
            road=remap(A[..., 4], p),
            mover=mover,
            gain=np.ones(3, "f"),
        )


def closest(c, t):
    ts = seq(c)[0] + rig[c].time_offset_s
    k = int(np.clip(np.searchsorted(ts, t), 1, len(ts) - 1))
    return k if abs(ts[k] - t) < abs(ts[k - 1] - t) else k - 1


def picks(V, s):
    t, T = V.t, V.T
    ii, j, f = pair("front", s, t)
    b = j if f > 0.5 else ii
    ts, ns, S, arc = seq("front")
    baseclock = ts[b]
    base = [("front", b)] + [(c, closest(c, baseclock)) for c in CAMS[1:]]
    # Only search bridges for the actual off-axis gaps, in the physical base lens frame.
    rel = V.wr @ S[b, :3, :3]
    bearing = np.rad2deg(np.arctan2(rel[..., 0], rel[..., 2]))
    strip = (abs(bearing) > 20) & (abs(bearing) < 34)
    # Coverage probe at several ranges, including sky: the nearest observing exposure wins
    # at each witness. Limit source churn by using fixed metre offsets on the delayed clock.
    pts = []
    for sg in [-1, 1]:
        y, x = np.where(strip & (np.sign(bearing) == sg))
        if not len(x):
            continue
        m = (y % 16 == 4) & (x % 5 == 0)
        r = V.wr[y[m], x[m]]
        g = V.gr[y[m], x[m]]
        if len(g):
            pts.append((r, np.maximum(g, np.where(g > 0.002, 0.025, 0.00001))))
    donors = []
    if pts:
        r = np.concatenate([p[0] for p in pts])
        z = np.concatenate([p[1] for p in pts])
        unseen = np.ones(len(z), bool)
        for d in (4, 9, 16, 26, 42, 65, 90):
            i, j, f = pair("front", max(0, s - d), t)
            i = j if f > 0.5 else i
            P = S[i]
            v = (r + (T[:3, 3] - P[:3, 3]) * z[:, None]) @ P[:3, :3]
            uv = proj("front", v[:, None, :], GW, GH)[:, 0]
            ok = (
                (v[:, 2] > 0)
                & (uv[:, 0] > 1)
                & (uv[:, 0] < GW - 2)
                & (uv[:, 1] > 1)
                & (uv[:, 1] < GH - 2)
            )
            if (ok & unseen).sum() >= 2 and i != b and ("front", i) not in donors:
                donors.append(("front", i))
                unseen &= ~ok
    return base + donors, strip, bearing


def photogain(q, refs, ground, gmask):
    # Fit one smooth (constant over this photograph) residual correction; only robust,
    # low-gradient corresponding surface samples count. Flat measured ground is an anchor
    # when the *angular* gap leaves no direct front/pillar overlap. Front is the tone gauge.
    samples = []
    weights = []
    for col, ok in [*refs, (ground, gmask & (q["road"] > 0.65))]:
        good = (
            ok
            & q["ok"]
            & (grad(q["col"]) < 14)
            & (grad(col) < 14)
            & (col.min(2) > 20)
            & (q["col"].min(2) > 20)
        )
        if good.sum() > 14:
            samples.append(
                np.median(
                    np.log(
                        (col[good].astype("f") + 2) / (q["col"][good].astype("f") + 2)
                    ),
                    axis=0,
                )
            )
            weights.append(min(300, good.sum()))
    return (
        np.clip(
            np.exp(np.average(samples, axis=0, weights=weights)), 0.61, 1.55
        ).astype("f")
        if samples
        else np.ones(3, "f")
    )


def adjusted(col, gain):
    return np.uint8(np.clip(col * gain + 0.5, 0, 255))


def register_bridge(q, neighbours, plane_mask):
    # Geometry is the registration model; test a single residual translation, in full
    # output pixels. Require round-trip-consistent features from BOTH neighbouring lenses.
    # Never apply a residual on ground, or treat absent/occluded tracks as agreement.
    samples = []
    core = cv.erode((~plane_mask).astype("B"), np.ones((5, 5), "B")).astype(bool)
    for other in neighbours:
        mask = core & q["ok"] & other["ok"]
        ys, xs = np.where(mask)
        if len(xs) < 50:
            continue
        x0 = max(0, xs.min() - 3)
        x1 = min(L, xs.max() + 4)
        y0 = max(0, ys.min() - 3)
        y1 = min(H, ys.max() + 4)
        sl = np.s_[y0:y1, x0:x1]

        def gray(p):
            uv = resize(p["uv"][sl], (x1 - x0) * 5, (y1 - y0) * 5)
            return cv.cvtColor(remap(p["img"], uv), cv.COLOR_BGR2GRAY)

        a, b = gray(other), gray(q)
        m = cv.erode(
            cv.resize(
                mask[sl].astype("B"),
                (a.shape[1], a.shape[0]),
                interpolation=cv.INTER_NEAREST,
            ),
            np.ones((11, 11), "B"),
        )
        pts = cv.goodFeaturesToTrack(a, 65, 0.012, 7, mask=m)
        if pts is None or len(pts) < 6:
            continue
        v, ok, _ = cv.calcOpticalFlowPyrLK(
            a, b, pts, None, winSize=(15, 15), maxLevel=2
        )
        u, ok2, _ = cv.calcOpticalFlowPyrLK(b, a, v, None, winSize=(15, 15), maxLevel=2)
        d = (v - pts).reshape(-1, 2)
        sel = (
            ok.ravel().astype(bool)
            & ok2.ravel().astype(bool)
            & (np.linalg.norm((u - pts).reshape(-1, 2), axis=1) < 0.5)
            & (np.linalg.norm(d, axis=1) < 3.0)
        )
        if sel.sum() >= 5:
            samples.append(d[sel])
    if len(samples) != 2:
        return None
    shift = np.median(np.vstack(samples), 0)
    if (
        np.linalg.norm(shift) > 2.5
        or max(np.percentile(np.linalg.norm(d - shift, axis=1), 80) for d in samples)
        > 1
    ):
        return None
    # Only the non-ground coordinate field moves. Keep a two-cell guard at the contact.
    target = XY + shift[None, None, :] / 5 * core[..., None]
    q["uv"] = remap(q["uv"], target)
    q["ok"] &= remap(q["ok"].astype("f"), target) > 0.97
    q["col"] = remap(q["img"], q["uv"])
    return shift.tolist()


def frame(V, s, prev=None):
    t, T = V.t, V.T
    sel, strip, bearing = picks(V, s)
    zs = [V.sheet(*p) for p in sel]
    # The independent observer sees the SAME 3-D point at a seam. Use nearest photo for
    # relief; make only the geometry continuous at borders, never the sampled radiance.
    z = V.gr.copy()
    weights = np.zeros_like(z)
    for k, v in enumerate(zs):
        ok = v >= 0
        if k >= 3:
            ok &= strip
        w = np.minimum(cv.distanceTransform(ok.astype("B"), cv.DIST_L2, 3) / 3, 1) * (
            1 - weights
        )
        z += w * (np.maximum(v, V.gr) - V.gr)
        weights += w
    # Fit the two depth layers as upright street-side surfaces. At a skyline, raw
    # per-pixel depth can lift a leaf by 200 px and the adjacent sky by zero. A z-buffer
    # cannot recover the missing sky. Keep vertical trunk and roof lines intact.
    fw = T[:2, 2] / np.linalg.norm(T[:2, 2])
    along = V.wr[..., :2] @ fw
    upright = z / np.maximum(0.2, along)
    above = (z > V.gr + 0.002) & (weights > 0.5) & (V.gr < 0.048)
    # Fit near and background range separately, then hand over the chart across azimuth.
    # A column chart shortens lower trunks in ambiguous views but cannot draw a second
    # copy of a silhouette. Ground is analytic right up to contact with the nearer plane.
    fg = above & (upright > 1 / 25)
    bgsel = above & (upright <= 1 / 25) & (upright > 1 / 180)

    def profile(mask, lo, hi, default, sigma):
        count = mask.sum(0).astype("f")
        val = np.where(mask, np.clip(upright, lo, hi), 0).sum(0)
        den = ms.gaussian_filter1d(count, sigma)
        return (ms.gaussian_filter1d(val, sigma) + default * 4) / (den + 4), den

    far, _ = profile(bgsel, 1 / 180, 1 / 40, 1 / 95, 13)
    near, n = profile(fg, 1 / 30, 1 / 17, 1 / 24, 18)
    # For texture with no reliable occlusion boundary keep one conservative plane.
    # A large disparity inferred from leaves must not bend the entire office roof or wires.
    far = np.clip(far, 1 / 135, 1 / 85)
    iv = far + (near - far) * smooth((n - 1) / 14) * 0.16
    z = np.maximum(V.gr, along * iv[None, :]).astype("f")
    qs = [V.project(*pick, z) for pick in sel]
    bg, ga = floor(t, T, V.wr, V.gr)
    ground = cv.resize(bg, (L, H), interpolation=cv.INTER_AREA)
    planeMask = (z == V.gr) & (V.gr > 0.003)
    registration = []
    for q in qs[3:]:
        left = np.sum(q["ok"] & qs[1]["ok"] & strip)
        right = np.sum(q["ok"] & qs[2]["ok"] & strip)
        registration.append(
            register_bridge(q, [qs[0], qs[1 if left > right else 2]], planeMask)
        )
    # Recover front gauge on the ground used as disconnected-overlap reference.
    front = qs[0]
    off = (
        front["ok"]
        & planeMask
        & (front["road"] > 0.7)
        & (ga > 0.85)
        & (grad(ground) < 10)
        & (grad(front["col"]) < 10)
    )
    gg = (
        np.clip(
            np.median(
                (front["col"][off].astype("f") + 2) / (ground[off].astype("f") + 2),
                axis=0,
            ),
            0.72,
            1.32,
        )
        if off.sum() > 40
        else np.ones(3)
    )
    bg = adjusted(bg, gg)
    ground = adjusted(ground, gg)
    refs = [(front["col"], front["ok"])]
    for k in list(range(3, len(qs))) + [1, 2]:
        q = qs[k]
        q["gain"] = photogain(q, refs, ground, (ga > 0.85) & planeMask)
        # Disconnected optical footprints have no skyline overlap even at infinity. Use
        # contemporaneous, masked sky to estimate the missing chromatic gauge, preserving
        # its detailed texture. Taper continuously into the measured ground correction.
        _, qa, qi, *_ = fields(q["c"], q["i"])
        _, fa, fi, *_ = fields("front", front["i"])
        qsky = (qa[..., 5] > 0.93) & (qa[..., 6] > 0.9)
        fsky = (fa[..., 5] > 0.93) & (fa[..., 6] > 0.9)
        if k in [1, 2] and qsky.sum() > 120 and fsky.sum() > 120:
            skyGain = np.clip(
                np.median(resize(fi, GW, GH)[fsky], 0)
                / np.maximum(1, np.median(resize(qi, GW, GH)[qsky], 0)),
                0.55,
                1.55,
            )
            # Smooth low-order gain in sensor coordinates, applied to a *single* texture.
            y = np.linspace(0, 1, qi.shape[0])[:, None, None]
            f = smooth((0.70 - y) / 0.40)
            g = (skyGain * f + q["gain"] * (1 - f)).astype("f")
            q["img"] = adjusted(qi, g)
            q["gain"] = np.ones(3, "f")
            q["col"] = remap(q["img"], q["uv"])
        else:
            q["col"] = adjusted(q["col"], q["gain"])
        refs.append((q["col"], q["ok"] & (strip if k >= 3 else True)))
    # Graph cuts use large costs along structures/markings and across disagreement. A fixed
    # lens role, rather than the changing acquisition number, receives temporal hysteresis.
    lab = np.zeros((H, L), "B")
    best = np.full((H, L), -7.0, "f")
    simul = qs[0]["ok"] | qs[1]["ok"] | qs[2]["ok"]
    stitched = (~simul) & planeMask & (ga > 0.45)
    best[stitched] = 3.8
    col = ground.copy()
    seen = np.zeros_like(z, bool)
    hist = None
    if prev is not None:
        oldT, oldrole = prev
        rr = (V.wr + (T[:3, 3] - oldT[:3, 3]) * z[..., None]) @ oldT[:3, :3]
        xy = rr[..., :2] / np.maximum(0.01, rr[..., 2:]) * (V.f / 5) + [
            L / 2 - 0.5,
            H / 2 - 0.5,
        ]
        hist = cv.remap(
            oldrole,
            xy.astype("f"),
            None,
            cv.INTER_NEAREST,
            borderMode=cv.BORDER_CONSTANT,
            borderValue=255,
        )
    for k in list(reversed(range(3, len(qs)))) + [2, 1, 0]:
        q = qs[k]
        role = min(4, k + 1)
        legal = q["ok"].copy()
        if k >= 3:
            S, A, _, _, _, _, _ = fields(q["c"], q["i"])
            uv = (q["uv"] + 0.5) * [
                GW / q["img"].shape[1],
                GH / q["img"].shape[0],
            ] - 0.5
            sample = remap(A, uv)
            # Visibility check, not just frustum membership. A far-back lens that sees
            # the drive-through wall instead of the street cannot bridge the blind wedge.
            ray = (V.wr + (T[:3, 3] - S[:3, 3]) * z[..., None]) @ S[:3, :3]
            zi = z / np.maximum(0.1, ray[..., 2])
            obs = np.maximum(sample[..., 0], sample[..., 2] * sample[..., 1])
            consistent = (obs < zi * 1.65 + 0.008) & (sample[..., 4] < 0.48)
            legal &= strip & ~planeMask & consistent
            if np.linalg.norm(T[:2, 3] - S[:2, 3]) > 36:
                legal &= (sample[..., 5] > 0.9) & (V.wr[..., 2] > 0.035)
        legal &= ~stitched
        # Select the nearest bridge with usable samples. A pillar replaces the bridge as
        # soon as there is overlap to route a low-texture seam. No cross-time flow.
        sc = (2.4 if k == 0 else 1.9 if k < 3 else 0.45 - 0.1 * (k - 3)) + np.log(
            np.clip(q["border"] / 4, 0.12, 1)
        )
        if k >= 3:
            sc -= 0.7 * (np.linalg.norm(seq("front")[2][q["i"], :3, 3] - T[:3, 3]) / 30)
        if hist is not None:
            sc += 0.5 * (hist == role)
        delta = (sc - best).astype("f")
        halo = cv.dilate(
            np.clip(np.maximum(grad(col), grad(q["col"])) / 14, 0, 2),
            np.ones((5, 5), "B"),
        )
        dif = abs(col.astype("f") - q["col"]).mean(2)
        barrier = 3 + 70 * halo**2 + np.minimum(dif, 55)
        win = cut(delta, barrier, legal, seen | ((ga > 0.75) & planeMask))
        lab[win] = k + 1
        best[win] = sc[win]
        col[win] = q["col"][win]
        seen |= win
    # Signed field interpolation only chooses an owner; the full-resolution RGB is always
    # a single bilinear lookup. Keep the 44-pixel halo for independent solo-warp audits.
    score = np.full((OH, OW), -1, "f")
    hard = np.zeros((OH, OW), "B")
    for j in np.unique(lab):
        w = resize(blur((lab == j).astype("f"), 0.60), OW, OH)
        m = w > score
        hard[m] = j
        score[m] = w[m]
    out = np.empty_like(bg)
    patches = []
    mag = np.zeros((H, L), "f")
    for j in np.unique(hard):
        x, y, w, h = rc = rect_for(hard, j)
        sl = np.s_[y : y + h, x : x + w]
        if j:
            q = qs[j - 1]
            uv = resize(q["uv"], OW, OH)[sl]
            rgb = adjusted(remap(q["img"], uv), q["gain"])
            dx = np.gradient(q["uv"], axis=1) / 5
            dy = np.gradient(q["uv"], axis=0) / 5
            tr = (dx * dx + dy * dy).sum(2)
            det = abs(dx[..., 0] * dy[..., 1] - dx[..., 1] * dy[..., 0])
            hi = np.sqrt(0.5 * (tr + np.sqrt(np.maximum(0, tr * tr - 4 * det * det))))
            mag[lab == j] = (hi / np.maximum(0.0001, det))[lab == j]
        else:
            rgb = bg[sl]
        m = hard[sl] == j
        out[sl][m] = rgb[m]
        patches.append((int(j), rc, rgb))
    solo = remap(front["img"], resize(qs[0]["uv"], OW, OH))
    meta = dict(
        t=t,
        registration=registration,
        photo=float((lab > 0).mean()),
        pillar=float(((lab == 2) | (lab == 3)).mean()),
        bridge=float((lab > 3).mean()),
        unknown=float(((lab == 0) & (ga < 0.7)).mean()),
        stretch6=float((mag > 6).mean()),
        ground=float(planeMask.mean()),
        groundStitched=float(((lab == 0) & (ga > 0.7) & planeMask).mean()),
        picks=[
            dict(cam=q["c"], n=q["n"], remove=q["mover"], gain=q["gain"].tolist())
            for q in qs
        ],
    )
    return out, hard, patches, solo, meta, (T, np.minimum(lab, 4))


def camera(deg):
    frames = track(110, 125)
    out = []
    for t, T in frames:
        T = T.copy()
        if deg != 50:
            # Scale about the follow pivot along the ORIGINAL arm, not along the tilted optical axis.
            D = 16.2 * (
                1
                + 0.004
                * min(
                    25,
                    np.average(
                        [
                            max(0, np.interp(t - k * 0.3, ms.dt, ms.drive["speed"]))
                            for k in range(9)
                        ],
                        weights=np.exp(-np.arange(9) * 0.3 / 1.2),
                    ),
                )
            )
            fw = T[:2, 2] / np.linalg.norm(T[:2, 2])
            arm = np.r_[-fw * np.cos(np.deg2rad(16.9)), np.sin(np.deg2rad(16.9))]
            T[:3, 3] += (
                D * (np.tan(np.deg2rad(25)) / np.tan(np.deg2rad(deg / 2)) - 1) * arm
            )
        out.append((t, T))
    return out, choose(out)


def pilot():
    for deg, ids in [(50, [0, 60, 180, 225, 450]), (44, [60, 225, 450])]:
        hp.F = 450 / np.tan(np.deg2rad(deg / 2))
        ts, arc = camera(deg)
        rows = []
        for i in ids:
            t, T = ts[i]
            im, lab, patches, solo, meta, hist = frame(View(t, T, deg), arc[i])
            refs = {j: (r, refmask(rgb)) for j, r, rgb in patches}
            meta["rawAudit"] = audit(im, lab, refs)
            rows.append(meta)
            cv.imwrite(
                str(OUT / f"{t:.2f}-{deg}.jpg"), im, [cv.IMWRITE_JPEG_QUALITY, 94]
            )
            print(
                t,
                deg,
                meta["rawAudit"]["fraction"],
                {
                    k: round(meta[k], 3)
                    for k in ["pillar", "bridge", "unknown", "stretch6"]
                },
                flush=True,
            )
        savej(OUT / f"probe-{deg}.json", rows)


if __name__ == "__main__":
    pilot()
