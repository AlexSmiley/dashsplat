"""Continuous, backwards-sampled RGB-D photographic warps. Morph temporal and spatial donors in chase pixels before mixing their radiance."""

from .morph_sources import *
from .horizon_proof import Hybrid
from . import layer_player as lp
import ctypes as ct, subprocess, sys, hashlib, time


@lru_cache(None)
def ground_player():
    h = Hybrid.__new__(Hybrid)
    h.host = HOST
    h.tiles = [e for e in J(HOST / "map.json")["tiles"] if e["layer"] == "ground"]
    h.bb = np.array([e["bbox"] for e in h.tiles])
    h.lib = ct.CDLL(str(RS / "work/stitch/report/v2/wide.dylib"))
    P = ct.c_void_p
    I = ct.c_int
    h.lib.raster.argtypes = [P, I, ct.c_float, I, P, P, I, I, I, P]
    h.lib.raster.restype = None
    return h


@lru_cache(None)
def sky():
    return cv.imread(str(RS / "work/ibr/horizon-l/horizon-r4/sky.jpg"))


def floor(t, T, wr, gd):
    im = sky()
    el = np.arctan2(wr[:, :, 2], np.linalg.norm(wr[:, :, :2], axis=2))
    az = np.unwrap(np.arctan2(wr[:, :, 1], wr[:, :, 0]), axis=1)
    p = np.stack(
        [
            (0.5 + az / (2 * np.pi)) * im.shape[1] - 0.5,
            np.clip((82 - np.rad2deg(el)) / 94 * im.shape[0] - 0.5, 0, im.shape[0] - 1),
        ],
        -1,
    ).astype("f")
    col = cv.remap(
        im, resize(p, OW, OH), None, cv.INTER_LINEAR, borderMode=cv.BORDER_WRAP
    )
    ground = ground_player().splats(t, T, [38, 65])
    al = ground[:, :, 3:4]
    rgb = ground[:, :, :3][:, :, ::-1]
    ap = resize(smooth((-el - 0.005) / 0.14), OW, OH)[:, :, None]
    bg = (1 - ap) * col + ap * np.array([102, 105, 99], "f")
    out = rgb + (1 - al) * bg
    return np.uint8(np.clip(out + 0.5, 0, 255)), cv.resize(
        al, (L, H), interpolation=cv.INTER_AREA
    )


def plane(t, T, wr):
    E = TR.T_world_ego(t)
    n, h = TR.ground_plane(t)
    n = E[:3, :3] @ n
    return np.clip(-(wr @ n) / max(1, (T[:3, 3] - E[:3, 3]) @ n + h), 0, 1 / 2).astype(
        "f"
    )


def pair(c, s, t):
    times, _, poses, arc = seq(c)
    hi = min(len(arc) - 1, np.searchsorted(times, t) + 1)
    i = int(np.clip(np.searchsorted(arc[:hi], s, side="right") - 1, 0, hi - 1))
    j = min(i + 1, len(arc) - 1)
    v = float(np.clip((s - arc[i]) / max(0.00001, arc[j] - arc[i]), 0, 1))
    return (i, j, v)


def pull(c, i, T, wr, gr):
    S, A, im = source(c, i)
    dr = wr @ S[:3, :3]
    delta = (T[:3, 3] - S[:3, 3]) @ S[:3, :3]
    ph, pw = im.shape[:2]
    # Reference chart: an uncut photographic sheet with a rounded horizon and exact tangent-plane
    # intersection below. Limit depth-induced displacement FIRST, not a blur of an already folded chart.
    baseInv = np.maximum(gr, 0)
    z0 = 0.007 + np.maximum(0, baseInv - 0.007) * smooth(baseInv / 0.022)

    def uv(iv):
        return proj(c, dr + delta * iv[:, :, None], GW, GH)

    def bounded(p0, d, limit, sigma):
        # A shared bound in source texels per output cell protects the relative Jacobian even on weak depth.
        d = np.clip(d, -limit, limit).astype("f")
        v = blur(d, sigma)
        dx = np.linalg.norm(np.gradient(v, axis=1), axis=2)
        dy = np.linalg.norm(np.gradient(v, axis=0), axis=2)
        amp = 1 / np.maximum(
            1, cv.dilate(np.maximum(dx, dy), np.ones((7, 7), "B")) / 0.105
        )
        return p0 + v * blur(amp, 3)[:, :, None]

    p0 = uv(z0)
    v = remap(A, p0)
    inv = np.clip(
        dr[:, :, 2] * v[:, :, 0] / np.maximum(0.5, 1 - delta[2] * v[:, :, 0]), 0, 0.048
    )
    z = (1 - v[:, :, 4]) * inv + v[:, :, 4] * gr
    # Edge-aware ground contact: retain measured road and extend it only gently under non-road scenery.
    road = blur(v[:, :, 4], 4)
    blend = smooth((road - 0.12) / 0.66)
    reference = z0 * blend + (1 - blend) * np.minimum(z0, 0.023)
    pbase = uv(reference)
    p = bounded(pbase, uv(z) - pbase, 3.5, 4.6)
    v = remap(A, p)
    # Single foreground sheet: choose a smooth absolute relief range, keep its photograph's shape
    # rather than tracing a possibly multivalued inverse depth at a pole boundary.
    fr = np.full_like(gr, 0.05)
    f0 = uv(fr)
    av = remap(A, f0)
    di = np.clip(
        dr[:, :, 2] * av[:, :, 1] / np.maximum(0.5, 1 - delta[2] * av[:, :, 1]),
        0.034,
        0.076,
    )
    fp = bounded(f0, uv(di) - f0, 4, 4.5)
    av = remap(A, fp)
    border = lambda p: np.minimum.reduce(
        [p[:, :, 0], GW - 1 - p[:, :, 0], p[:, :, 1], GH - 1 - p[:, :, 1]]
    )
    alpha = (
        smooth((border(p) - 0.3) / 5)
        * smooth((v[:, :, 6] - 0.25) / 0.6)
        * (dr[:, :, 2] > 0)
    )
    alpha *= 1 - smooth((v[:, :, 3] - 0.06) / 0.32)
    alpha *= 1 - smooth((v[:, :, 2] - 0.32) / 0.54)
    fg = (
        smooth((av[:, :, 2] - 0.16) / 0.55)
        * (1 - smooth((av[:, :, 3] - 0.08) / 0.38))
        * smooth((border(fp) - 0.3) / 5)
        * av[:, :, 6]
        * (dr[:, :, 2] > 0)
    )
    # A camera ahead of the chase cannot photograph surfaces behind its optical centre.
    alpha *= smooth((dr[:, :, 2] + delta[2] * z0 - 0.08) / 0.32)
    fg *= smooth((dr[:, :, 2] + delta[2] * fr - 0.08) / 0.32)
    # Severely minified road from a far-back lens has almost no vertical source samples. In the
    # near field let measured stitched ground carry it, rather than extending paint through a verge.
    skip = smooth((np.linalg.norm(delta[:]) - 10) / 19) * smooth((gr - 0.045) / 0.045)
    alpha *= 1 - 0.88 * skip * v[:, :, 4]
    w = fg + alpha * (1 - fg)
    fg = fg / np.maximum(0.001, w)
    p = (p + 0.5) * [pw / GW, ph / GH] - 0.5
    fp = (fp + 0.5) * [pw / GW, ph / GH] - 0.5
    p = p.astype("f")
    fp = fp.astype("f")
    thumb = remap(im, p) * (1 - fg[:, :, None]) + remap(im, fp) * fg[:, :, None]
    return dict(
        i=i,
        c=c,
        n=int(seq(c)[1][i]),
        p=p,
        fp=fp,
        fg=fg.astype("f"),
        mask=np.clip(w, 0, 1).astype("f"),
        thumb=np.uint8(thumb + 0.5),
        tex=im,
    )


def align(a, b, am, bm, limit=5):
    # Positive flow maps coordinates in reference A into B. Suppress flow outside reliable overlap; use
    # a bounded, locally smooth displacement rather than accepting independently folding correspondences.
    mask = am * bm
    ga = cv.cvtColor(a, cv.COLOR_BGR2GRAY)
    gb = cv.cvtColor(b, cv.COLOR_BGR2GRAY)
    both = mask > 0.16
    if both.sum() < 65:
        return np.zeros((H, L, 2), "f")
    xs = np.where(both)[1]
    ys = np.where(both)[0]
    x0 = max(0, xs.min() - 6)
    x1 = min(L, xs.max() + 7)
    y0 = max(0, ys.min() - 6)
    y1 = min(H, ys.max() + 7)
    roi = np.s_[y0:y1, x0:x1]
    flow = np.zeros((H, L, 2), "f")
    u = ga[roi].copy()
    v = gb[roi].copy()
    v[~both[roi]] = u[~both[roi]]
    d = cv.calcOpticalFlowFarneback(u, v, None, 0.5, 3, 15, 3, 5, 1.1, 0)
    d = blur(d, 1.35)
    flow[roi] = np.clip(d, -limit, limit)
    return flow * smooth(mask / 0.6)[:, :, None]


def deform(q, p):
    for k in ["p", "fp", "fg", "mask", "thumb"]:
        q[k] = remap(q[k], p)


def group(c, br, T, wr, gr):
    i, j, f = br
    a = pull(c, i, T, wr, gr)
    b = pull(c, j, T, wr, gr)
    v = align(a["thumb"], b["thumb"], a["mask"], b["mask"], 2.8)
    deform(a, XY - f * v)
    deform(b, XY + (1 - f) * v)
    wa = (1 - f) * a["mask"]
    wb = f * b["mask"]
    m = wa + wb
    col = np.uint8(
        (wa[:, :, None] * a["thumb"] + wb[:, :, None] * b["thumb"])
        / np.maximum(0.001, m[:, :, None])
        + 0.5
    )
    a["share"] = wa / np.maximum(0.001, m)
    b["share"] = wb / np.maximum(0.001, m)
    return (
        [a, b],
        m,
        col,
        float(np.percentile(np.linalg.norm(v, axis=2)[m > 0.75], 90) * 5)
        if (m > 0.75).any()
        else 0,
    )


def render(t, T, s, details=False, plan=None, fixed=None, resolver=None):
    wr = RAY @ T[:3, :3].T
    gr = plane(t, T, wr)
    bg, ga = floor(t, T, wr, gr)
    base = np.zeros((H, L, 3), "B")
    mask = np.zeros((H, L), "f")
    allq = []
    uses = []
    flows = []
    total = np.zeros_like(mask)
    if fixed is None:
        sp = float(np.interp(t, dt, ds))
        target = max(0, sp - s)
        picks = [
            ("front", s),
            ("front", max(0, s - max(10, target))),
            ("front", max(0, s - max(25, target * 2.2))),
            ("front", min(sp, s + 8)),
            ("front", sp),
            ("left_pillar", min(sp, s + 7)),
            ("right_pillar", min(sp, s + 7)),
            ("left_pillar", max(0, s - 12)),
            ("right_pillar", max(0, s - 12)),
        ]
        # The turn can put the nearest historical front lens 45 degrees off axis. Look for additional
        # front views of the two flank rays before borrowing a pillar, using a continuous delayed clock.
        ts, _, hist, arc = seq("front")
        cand = np.flatnonzero(
            (arc >= max(0, s - 72)) & (arc <= sp) & (ts <= t) & (ts >= max(0, t - 30))
        )
        extra = []
        if len(cand):
            for sg in [-1, 1]:
                ax = T[:3, 2] + sg * 0.56 * T[:3, 0]
                ax /= np.linalg.norm(ax)
                sl = np.clip(hist[cand, :3, 2] @ ax, -1, 1)
                err = 1 - sl + 0.00075 * abs(arc[cand] - s)
                j = cand[np.argmin(err)]
                extra.append(("front", arc[j]))
        # Prefer a front image looking along the street to an XY-nearer image looking across it.
        # The source clock remains within ordered video brackets; orientation only adjusts their reliability.
        z = next(idx for idx, (c, _) in enumerate(picks) if c != "front")
        picks = picks[:z] + extra + picks[z:]
        leading = []
        if plan is not None:
            j = int(np.clip(round((t - plan[0]) * 30), 0, len(plan[1]["front"][0]) - 1))
            p0 = pair("front", s, t)[0]
            fwd = seq("front")[2][p0, :3, 2]
            off = float(np.rad2deg(np.arccos(np.clip(fwd @ T[:3, 2], -1, 1))))
            turn = float(smooth((off - 17) / 15))
            for c in CAMS:
                spick, conf = plan[1][c]
                w = turn * conf[j]
                if w > 0.015:
                    leading.append((c, spick[j], w))
            # Keep front when aligned; when it looks across the street a single wide pillar can own
            # nearly the entire frame. Ease out again before reverting to the delayed front video.
            leading.sort(key=lambda x: -x[2])
            picks = leading + [(c, sp, 1.0) for c, sp in picks]
        else:
            picks = [(c, sp, 1.0) for c, sp in picks]
        brackets = [(c, pair(c, pos, t), prior) for c, pos, prior in picks]
    else:
        brackets = fixed
    seen = set()
    for c, br, prior in brackets:
        idx = (c, br[0])
        if idx in seen:
            continue
        seen.add(idx)
        S = seq(c)[2][br[0]]
        angle = float(np.rad2deg(np.arccos(np.clip(S[:3, 2] @ T[:3, 2], -1, 1))))
        if angle > 115:
            continue
        qs, m, col, fl = group(c, br, T, wr, gr)
        # Condition number from the inverse map, not depth confidence alone: e.g. a lens 35 m
        # back compresses four road lanes into a few source scanlines. Use the measured ground there.
        pmap = qs[0]["p"]
        dx = np.gradient(pmap, axis=1) / 5
        dy = np.gradient(pmap, axis=0) / 5
        tr = (dx * dx + dy * dy).sum(2)
        det = abs(dx[:, :, 0] * dy[:, :, 1] - dx[:, :, 1] * dy[:, :, 0])
        cond = (tr + np.sqrt(np.maximum(0, tr * tr - 4 * det * det))) / np.maximum(
            0.003, 2 * det
        )
        quality = 1 - smooth((cond - 4) / 6) * smooth((gr - 0.018) / 0.04) * smooth(
            (ga - 0.10) / 0.63
        )
        m = m * quality * prior
        if c == "front" and angle > 20:
            m *= 1 - 0.86 * smooth((angle - 20) / 25)
        flow = (
            align(base, col, mask, m, 6.5) if mask.any() else np.zeros((H, L, 2), "f")
        )
        p = XY + flow
        m = remap(m, p)
        col = remap(col, p)
        w = (1 - mask) * m
        if w.sum() < 1:
            continue
        # Exposure-match overlapping low frequencies across lenses, with small bounded channel correction.
        both = (mask > 0.75) & (m > 0.75)
        gain = np.ones(3, "f")
        if c != "front" and both.sum() > 90:
            ratio = (base[both].astype("f") + 24) / (col[both].astype("f") + 24)
            gain = np.clip(np.median(ratio[::6], 0), 0.92, 1.08)
        for q in qs:
            share = remap(q["share"], p)
            deform(q, p)
            q["weight"] = share * w
            q["gain"] = gain
            allq.append(q)
        base = np.uint8(
            np.clip(
                (base * mask[:, :, None] + col * gain * w[:, :, None])
                / np.maximum(0.001, (mask + w)[:, :, None])
                + 0.5,
                0,
                255,
            )
        )
        mask += w
        uses.append((c, int(seq(c)[1][br[0]]), float(br[2]), float(w.mean())))
        flows.append(fl)
    if resolver is not None:
        return resolver(t,bg,allq,uses,ga,gr,flows)
    # Fill inter-lens blind angles by extending the *coordinate field* of the observed panorama, not
    # a rectangular patch of grey apron. The smooth extension is strongest at the far skyline; near
    # road holes keep the measured ground. A low-pass border gives stable ownership while moving.
    good = mask > 0.78
    if good.any():
        dist, nn = distance_transform_edt(~good, return_indices=True)
        reach = (1 - smooth((dist - 2) / 24)) * (1 - smooth((gr - 0.025) / 0.035))
        f = (1 - mask) * reach
        num = np.zeros((H, L, 3), "f")
        den = np.zeros_like(mask)
        for shift in [-12, -6, 0, 6, 12]:
            ix = np.clip(XY[:, :, 0].astype(int) + shift, 0, L - 1)
            iy = XY[:, :, 1].astype(int)
            yx = nn[:, iy, ix]
            wd = np.exp(
                -((yx[1] - XY[:, :, 0]) ** 2 + (yx[0] - XY[:, :, 1]) ** 2) / 145.0
            )
            num += base[yx[0], yx[1]] * wd[:, :, None]
            den += wd
        ext = np.where(
            (den > 0.003)[:, :, None],
            num / np.maximum(0.001, den[:, :, None]),
            base[nn[0], nn[1]],
        )
        bg = np.uint8(
            np.clip(
                bg.astype("f")
                + (resize(ext, OW, OH) - bg) * resize(reach, OW, OH)[:, :, None],
                0,
                255,
            )
        )
    out = bg.astype("f") * resize(np.clip(1 - mask, 0, 1), OW, OH)[:, :, None]
    for q in allq:
        weight = q["weight"]
        ys, xs = np.nonzero(weight > 0.0002)
        if not len(xs):
            continue
        x0 = max(xs.min() - 2, 0)
        x1 = min(xs.max() + 3, L)
        y0 = max(ys.min() - 2, 0)
        y1 = min(ys.max() + 3, H)
        roi = np.s_[y0:y1, x0:x1]
        img = out[y0 * 5 : y1 * 5, x0 * 5 : x1 * 5]
        size = (img.shape[1], img.shape[0])
        w = cv.resize(weight[roi], size)
        fg = cv.resize(q["fg"][roi], size)
        color = remap(q["tex"], cv.resize(q["p"][roi], size))
        if fg.max() > 0.005:
            color = (
                color.astype("f")
                + (remap(q["tex"], cv.resize(q["fp"][roi], size)).astype("f") - color)
                * fg[:, :, None]
            )
        img += color * (w[:, :, None] * q["gain"])
    # Determine inverse-map anisotropy only where real photo colour actually wins, not at viewport borders.
    stretch = []
    magnification = np.zeros((H, L), "f") if details == "map" else None
    if details:
        for q in allq:
            p = q["p"]
            dx = np.gradient(p, axis=1) / 5
            dy = np.gradient(p, axis=0) / 5
            det = abs(dx[:, :, 0] * dy[:, :, 1] - dx[:, :, 1] * dy[:, :, 0])
            tr = (dx * dx + dy * dy).sum(2)
            hi = np.sqrt((tr + np.sqrt(np.maximum(0, tr * tr - 4 * det * det))) * 0.5)
            ratio = hi * hi / np.maximum(0.001, det)
            ok = (q["weight"] > 0.35) & (q["fg"] < 0.3)
            stretch.extend(ratio[ok][::10].clip(0, 100).tolist())
            if magnification is not None:
                magnification[ok] = np.maximum(
                    magnification[ok], hi[ok] / np.maximum(0.001, det[ok])
                )
    if magnification is not None:
        np.save(W / f"magnification-{t:.2f}.npy", magnification)
    return np.uint8(np.clip(out + 0.5, 0, 255)), dict(
        t=float(t),
        baseArc=float(s),
        uses=uses,
        photo=float(mask.mean()),
        underlay=float((1 - mask).mean()),
        temporalFlowP90=max(flows, default=0),
        stretchP95=float(np.percentile(stretch, 95)) if stretch else None,
    )


def encode(f, w, h, mb=14):
    return subprocess.Popen(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-s",
            f"{w}x{h}",
            "-r",
            "30",
            "-pix_fmt",
            "bgr24",
            "-i",
            "-",
            "-an",
            "-c:v",
            "h264_videotoolbox",
            "-b:v",
            f"{mb}M",
            "-g",
            "15",
            "-bf",
            "0",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(f),
        ],
        stdin=subprocess.PIPE,
    )


def main(a, b, mode):
    ts = track(a, b)
    arc = choose(ts)
    out = W / f"{a}-{b}"
    out.mkdir(exist_ok=True)
    stats = []
    samples = []
    enc = encode(out / "chase.mp4", OW, OH) if mode == "film" else None
    tic = time.monotonic()
    for i, ((t, T), s) in enumerate(zip(ts, arc)):
        if not enc and i not in [0, 60, len(ts) // 2, len(ts) - 1]:
            continue
        im, d = render(
            t, T, s, i % 30 == 0 or not enc, (a, turn_plan(a, b)) if b <= 110 else None
        )
        stats.append(d)
        if enc:
            enc.stdin.write(im.tobytes())
        if i % 30 == 0 or not enc:
            path = out / f"{t:.2f}.jpg"
            cv.imwrite(str(path), im, [cv.IMWRITE_JPEG_QUALITY, 95])
            samples.append(cv.resize(im, (400, 250)))
            print(
                t,
                round(d["photo"], 3),
                d["stretchP95"],
                round(time.monotonic() - tic, 1),
                flush=True,
            )
    if enc:
        enc.stdin.close()
        assert enc.wait() == 0
    while len(samples) % 2:
        samples.append(np.zeros_like(samples[-1]))
    cv.imwrite(
        str(out / "sheet.jpg"),
        np.vstack([np.hstack(samples[i : i + 2]) for i in range(0, len(samples), 2)]),
    )
    savej(out / f"{mode}.json", stats)
    print("done", a, b, len(stats), flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
