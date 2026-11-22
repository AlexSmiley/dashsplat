"""Paired-outline audit on decoded video: compare both ends of a 3–40 px parallel-edge pair with each contributing photograph warped alone."""

from .morph_ownership import *
import zlib

OUT = W / "single-owner-v1"
OUT.mkdir(exist_ok=True)


def edges(im, weak=False):
    g = cv.GaussianBlur(cv.cvtColor(im, cv.COLOR_BGR2GRAY), (3, 3), 0.7)
    ed = cv.Canny(g, 22 if weak else 42, 55 if weak else 95, L2gradient=True)
    gx = cv.Sobel(g, cv.CV_32F, 1, 0)
    gy = cv.Sobel(g, cv.CV_32F, 0, 1)
    mag = np.maximum(1, np.hypot(gx, gy))
    return ed, gx / mag, gy / mag


def refmask(rgb):
    return cv.dilate(edges(rgb, True)[0], np.ones((3, 3), "B"))


def rect_for(lab, j):
    y, x = np.nonzero(lab == j)
    x0 = max(0, int(x.min()) - 44)
    x1 = min(OW, int(x.max()) + 45)
    y0 = max(0, int(y.min()) - 44)
    y1 = min(OH, int(y.max()) + 45)
    return x0, y0, x1 - x0, y1 - y0


def references(lab, patches=None, cards=None):
    refs = {}
    if patches is not None:
        for j, r, rgb in patches:
            refs[j] = (r, refmask(rgb))
    else:
        for j in np.unique(lab):
            r = rect_for(lab, j)
            refs[int(j)] = (r, refmask(full(*cards[j], r)))
    return refs


def pack(x):
    return cv.imencode(".png", x, [cv.IMWRITE_PNG_COMPRESSION, 1])[1].tobytes()


def unpack(x):
    return cv.imdecode(np.frombuffer(x, "B"), cv.IMREAD_GRAYSCALE)


def audit(im, lab, refs, detail=False):
    ed, nx, ny = edges(im)
    ey, ex = np.nonzero(ed)
    ix = ((ex + 3 * ey) % 4) == 0
    ey = ey[ix]
    ex = ex[ix]
    n = len(ex)
    owner = lab[ey, ex]
    xv = nx[ey, ex]
    yv = ny[ey, ex]
    match = np.zeros(n, bool)
    bad = np.zeros(n, bool)
    gap = np.zeros(n, "B")
    bx = np.zeros(n, int)
    by = np.zeros(n, int)
    Hh, Ww = ed.shape
    for j, (r, ref) in refs.items():
        k = np.flatnonzero(owner == j)
        x, y, w, h = r
        match[k] = ref[ey[k] - y, ex[k] - x] != 0
    # Normal-direction search, both polarities of displacement but same gradient polarity at the two
    # edges. A real pole has opposing sides; a translated copy produces same-facing parallel edges.
    for dist in range(3, 41):
        for sign in (1, -1):
            u = ex + np.rint(sign * dist * xv).astype(int)
            v = ey + np.rint(sign * dist * yv).astype(int)
            u = np.clip(u, 0, Ww - 1)
            v = np.clip(v, 0, Hh - 1)
            sel = np.flatnonzero(
                (~bad) & (ed[v, u] != 0) & (nx[v, u] * xv + ny[v, u] * yv > 0.945)
            )
            if not len(sel):
                continue
            A = owner[sel]
            B = lab[v[sel], u[sel]]
            pa = match[sel]
            pb = np.zeros_like(pa)
            qb = np.zeros_like(pa)
            qa = np.zeros_like(pa)
            for j in np.union1d(A, B):
                (x, y, w, h), rf = refs[j]
                z = np.flatnonzero(A == j)
                o = sel[z]
                pb[z] = rf[v[o] - y, u[o] - x] != 0
                z = np.flatnonzero(B == j)
                o = sel[z]
                qb[z] = rf[v[o] - y, u[o] - x] != 0
                qa[z] = rf[ey[o] - y, ex[o] - x] != 0
            k = sel[(pa | pb | qa | qb) & ~((pa & pb) | (qa & qb))]
            bad[k] = 1
            gap[k] = dist
            bx[k] = u[k]
            by[k] = v[k]
    # Dismiss chance crosses/isolated texture responses: a twin needs another flagged anchor within
    # 6 px along its tangent. Both endpoints must already meet the orientation and separation test.
    mark = np.zeros_like(ed)
    mark[ey[bad], ex[bad]] = 255
    support = np.zeros(n, bool)
    for ds in [-6, -4, 4, 6]:
        u = np.clip(ex + np.rint(-yv * ds).astype(int), 0, Ww - 1)
        v = np.clip(ey + np.rint(xv * ds).astype(int), 0, Hh - 1)
        support |= cv.dilate(mark, np.ones((3, 3), "B"))[v, u] > 0
    bad &= support
    hi = (ey > Hh * 0.045) & (ey < Hh * 0.66)
    mark[:] = 0
    mark[ey[bad], ex[bad]] = 255
    roi = hi & bad
    ys = ey[roi]
    xs = ex[roi]
    hh = np.histogram2d(
        ys, xs, bins=[np.arange(0, Hh + 100, 100), np.arange(0, Ww + 160, 160)]
    )[0]
    y, x = np.unravel_index(hh.argmax(), hh.shape)
    r = (
        min(max(0, x * 160 - 45), Ww - 250),
        min(max(0, y * 100 - 35), Hh - 170),
        250,
        170,
    )
    res = dict(
        edges=n,
        twins=int(bad.sum()),
        fraction=float(bad.sum() / max(1, n)),
        structureEdges=int(hi.sum()),
        structureTwins=int(roi.sum()),
        structureFraction=float(roi.sum() / max(1, hi.sum())),
        unmatchedFraction=float((~match).sum() / max(1, n)),
        worstBox=list(map(int, r)),
        worstHits=int(hh.max()),
        p90Gap=float(np.percentile(gap[bad], 90)) if bad.any() else 0,
    )
    if detail:
        res["points"] = (ex[roi], ey[roi], bx[roi], by[roi])
        res["mark"] = mark
        if detail == "all":
            res["allPoints"] = (ex[bad], ey[bad], bx[bad], by[bad])
    return res


def probe():
    ts = track(110, 125)
    arc, plan = clip_clock(110, 125)
    cap = cv.VideoCapture(str(W / "85-140/chase.mp4"))
    rows = []
    results = []
    for idx in [0, 60, 225, 450]:
        t, T = ts[idx]
        im, meta, lab, ps, ol, oc = render(t, T, arc[idx], plan=plan, resolver=Owner())
        ol = cv.resize(ol, (OW, OH), interpolation=cv.INTER_NEAREST)
        cap.set(cv.CAP_PROP_POS_FRAMES, round((t - 85) * 30))
        ok, base = cap.read()
        assert ok
        r = audit(im, lab, references(lab, ps))
        s = audit(base, ol, references(ol, cards=oc))
        solo = full(*oc[1], (0, 0, OW, OH))
        label = np.ones_like(lab)
        ref = {1: ((0, 0, OW, OH), refmask(solo))}
        pos = cv.addWeighted(solo, 0.5, np.roll(solo, 9, axis=1), 0.5, 0)
        control = audit(solo, label, ref)
        positive = audit(pos, label, ref)
        assert control["twins"] == 0
        results.append(dict(t=t, published=s, hard=r, solo=control, shift9=positive))
        print(
            t,
            "percent original, hard, solo, double9",
            [round(z["fraction"] * 100, 3) for z in [s, r, control, positive]],
            flush=True,
        )
    savej(OUT / "control.json", results)


if __name__ == "__main__":
    probe()
