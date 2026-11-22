"""Paired-outline diagnostic against both actual exposure owners, including cross-seam pairs.
A second unused photograph tests pose+appearance in audit.py; here isolate introduced twins.
"""

from .audit import owners, edges
from .proof import *


def pairs(im, ids, refs, valid):
    ed, nx, ny = edges(im, True)
    ys, xs = np.nonzero((ed > 0) & valid)
    sub = (xs + 3 * ys) % 4 == 0
    ys, xs = ys[sub], xs[sub]
    gx, gy = nx[ys, xs], ny[ys, xs]
    bad = np.zeros(len(xs), bool)
    H, W = valid.shape
    for ds in [*range(3, 31), *range(-30, -2)]:
        x = (xs + np.rint(gx * ds).astype(int)).clip(0, W - 1)
        y = (ys + np.rint(gy * ds).astype(int)).clip(0, H - 1)
        hit = np.flatnonzero(
            ~bad
            & valid[y, x]
            & (ed[y, x] > 0)
            & (nx[y, x] * gx + ny[y, x] * gy > 0.945)
        )
        if len(hit) == 0:
            continue
        supported = np.zeros(len(hit), bool)
        anyhit = np.zeros(len(hit), bool)
        aa = ids[ys[hit], xs[hit]]
        bb = ids[y[hit], x[hit]]
        for k in np.union1d(aa, bb):
            use = (aa == k) | (bb == k)
            jj = hit[use]
            a = refs[k][ys[jj], xs[jj]] > 0
            b = refs[k][y[jj], x[jj]] > 0
            supported[use] |= a & b
            anyhit[use] |= a | b
        bad[hit[anyhit & ~supported]] = 1
    m = np.zeros_like(ed)
    m[ys[bad], xs[bad]] = 1
    m = cv.dilate(m, np.ones((3, 3), "B"))
    adj = np.zeros(len(xs), bool)
    for ds in [-6, -4, 4, 6]:
        adj |= (
            m[
                (ys + np.rint(gx * ds).astype(int)).clip(0, H - 1),
                (xs - np.rint(gy * ds).astype(int)).clip(0, W - 1),
            ]
            > 0
        )
    return [int((adj & bad).sum()), len(xs)]


def check(lo=85, hi=140):
    at = Atlas()
    obs = read(W / "observations.json")
    q = read(RPT / f"audit-{lo}-{hi}.json")["rows"]
    rows = []
    for e in q:
        xy, _, _ = top_xy(e["t"], 10.5, 0, 680, 800, 0.01)
        im, a = at.rgba(xy)
        im = np.uint8(np.clip(im, 0, 255))
        ids = owners(at, xy)
        ids[a[:, :, 0] < 0.98] = 0
        witness, _ = groundphoto(dict(cam="front", t=e["t"], n=e["n"]), xy)
        lum = cv.cvtColor(witness, cv.COLOR_RGB2GRAY).astype("f")
        mark = cv.dilate(
            np.uint8(lum - cv.GaussianBlur(lum, (0, 0), 16) > 13),
            cv.getStructuringElement(cv.MORPH_ELLIPSE, (45, 45)),
        )
        valid = (mark > 0) & (ids > 0)
        refs = {}
        native = {}
        for k in np.unique(ids[valid]):
            row = obs[k - 1]
            rgb, ok = groundphoto(row, xy)
            if "field" in row:
                rgb = np.uint8(
                    np.clip(rgb * localfield(row, xy, "gain")[:, :, ::-1] + 0.5, 0, 255)
                )
            refs[k] = cv.dilate(edges(rgb, False)[0], np.ones((3, 3), "B"))
            native[k] = rgb
        co = [0, 0]
        inject = [0, 0]
        if refs:
            k = max(refs, key=lambda j: int(((ids == j) & valid).sum()))
            solo = native[k]
            v = valid & (ids == k)
            co = pairs(solo, ids, refs, v)
            inject = pairs(
                cv.addWeighted(solo, 0.5, np.roll(solo, 9, 1), 0.5, 0), ids, refs, v
            )
        rows.append(
            dict(
                t=e["t"],
                n=e["n"],
                atlas=pairs(im, ids, refs, valid),
                solo=co,
                injected=inject,
            )
        )
    totals = {
        k: np.sum([e[k] for e in rows], axis=0).tolist()
        for k in ["atlas", "solo", "injected"]
    }
    dump(
        RPT
        / (
            "sourcecheck.json"
            if (lo, hi) == (85, 140)
            else f"sourcecheck-{lo}-{hi}.json"
        ),
        dict(rows=rows, totals=totals),
    )
    print(totals)


if __name__ == "__main__":
    import sys

    check(*map(int, sys.argv[1:]))
