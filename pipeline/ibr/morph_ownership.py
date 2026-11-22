"""One image and one surface own each target pixel. Optimize hard seams around strong features, with advected ownership hysteresis."""

from .morph_clip import *
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow, breadth_first_order

CW, CH = L // 2, H // 2
shrink = lambda im: cv.resize(im, (CW, CH), interpolation=cv.INTER_AREA)


def grad(im):
    g = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    x = cv.Sobel(g, cv.CV_32F, 1, 0, ksize=3) / 8
    y = cv.Sobel(g, cv.CV_32F, 0, 1, ksize=3) / 8
    return np.hypot(x, y)


def cut(delta, barrier, legal, oldlegal):
    # Source-side nodes take the challenger. Integer capacities encode the excess unary cost
    # and the edge/disagreement cost of a seam. A forbidden projection never gets selected.
    h, w = delta.shape
    n = h * w
    S = n
    D = n + 1
    idx = np.arange(n).reshape(h, w)
    ii = idx.ravel()
    x = idx[:, :-1].ravel()
    y = idx[:-1].ravel()
    row = np.r_[np.full(n, S), ii, x, x + 1, y, y + w]
    col = np.r_[ii, np.full(n, D), x + 1, x, y + w, y]
    hor = np.maximum(barrier[:, :-1], barrier[:, 1:]).ravel()
    vert = np.maximum(barrier[:-1], barrier[1:]).ravel()
    src = np.where(~oldlegal & legal, 2e6, np.maximum(0, delta) * 22)
    sink = np.where(~legal, 2e6, np.maximum(0, -delta) * 22)
    cap = np.r_[src.ravel(), sink.ravel(), hor, hor, vert, vert].astype(np.int32)
    G = csr_matrix((cap, (row, col)), shape=(n + 2, n + 2))
    G.eliminate_zeros()
    flow = maximum_flow(G, S, D)
    res = G - flow.flow
    res.eliminate_zeros()
    order = breadth_first_order(res, S, directed=True, return_predecessors=False)
    labels = np.zeros(n + 2, "B")
    labels[order] = 1
    return labels[:n].reshape(h, w).astype(bool)


def anchor_rgb(q, k):
    src = q["tex"]
    p = q[k]
    return np.uint8(np.clip(remap(src, p) * q["gain"] + 0.5, 0, 255))


def full(q, k, rect):
    x, y, w, h = rect
    if k == "floor":
        return q[y : y + h, x : x + w]
    # Exact bilinear expansion of the source-coordinate field on this crop, including half-pixel offsets.
    left = max(0, x // 5 - 2)
    top = max(0, y // 5 - 2)
    right = min(L, (x + w + 4) // 5 + 2)
    bottom = min(H, (y + h + 4) // 5 + 2)
    field = q[k][top:bottom, left:right]
    uv = cv.resize(field, ((right - left) * 5, (bottom - top) * 5))
    uv = uv[y - top * 5 : y - top * 5 + h, x - left * 5 : x - left * 5 + w]
    rgb = remap(q["tex"], uv)
    return (
        np.uint8(np.clip(rgb * q["gain"] + 0.5, 0, 255))
        if np.any(q["gain"] != 1)
        else rgb
    )


class Owner:
    def __init__(self):
        self.history = None

    def __call__(self, t, bg, allq, uses, ga, gr, flows):
        cards = [(bg, "floor")]
        scores = []
        colors = []
        ids = [(0, -10000, 0)]
        total = np.zeros((H, L), "f")
        oldcards = [(bg, "floor")]
        oldweights = []
        for rank, ((c, n, f, amount), (a, b)) in enumerate(
            zip(uses, zip(allq[::2], allq[1::2]))
        ):
            group = a["weight"] + b["weight"]
            total += group
            for q in [a, b]:
                for k, m in [("p", 1 - q["fg"]), ("fp", q["fg"])]:
                    oldcards.append((q, k))
                    oldweights.append(q["weight"] * m)
            q = b if f > 0.5 else a
            A = source(c, q["i"])[1]
            ph, pw = q["tex"].shape[:2]
            lens = CAMS.index(c) + 1
            for k, m in [("p", 1 - q["fg"]), ("fp", q["fg"])]:
                uv = (q[k] + 0.5) * [GW / pw, GH / ph] - 0.5
                v = remap(A, uv)
                usable = (
                    (v[:, :, 3] < 0.25)
                    & (v[:, :, 6] > 0.55)
                    & (uv[:, :, 0] >= 0.4)
                    & (uv[:, :, 0] < GW - 1.4)
                    & (uv[:, :, 1] >= 0.4)
                    & (uv[:, :, 1] < GH - 1.4)
                )
                # The background chart cannot borrow a second copy of its own near silhouette.
                usable &= (v[:, :, 2] < 0.3) if k == "p" else (v[:, :, 2] > 0.27)
                # Do not import alpha-over weights into the hard-cut objective: a good alternate
                # can fully own a region even when the original mosaic gave it very little opacity.
                pri = (1.0 if rank == 0 else (0.78 if c == "front" else 0.60)) / (
                    1 + 0.07 * rank
                )
                score = shrink(
                    pri * m * q["mask"] * usable * (0.55 + 0.45 * np.sqrt(group))
                )
                score[shrink(usable.astype("f")) < 0.76] = 0
                if score.max() < 0.03:
                    continue
                cards.append((q, k))
                scores.append(score)
                colors.append(shrink(anchor_rgb(q, k)))
                ids.append((lens, q["n"], int(k == "fp")))
        scores = [
            0.012
            + 0.19 * shrink(1 - total) ** 2
            + 0.085 * shrink(ga) * shrink(smooth((gr - 0.07) / 0.06))
        ] + scores
        colors = [shrink(cv.resize(bg, (L, H), interpolation=cv.INTER_AREA))] + colors
        oldweights = [np.maximum(0, 1 - total)] + oldweights
        oldowner = np.argmax(np.stack(oldweights), axis=0).astype("B")
        score = np.log(np.maximum(0.00001, np.stack(scores)))
        guide = colors[1] if len(colors) > 1 else colors[0]
        G = cv.cvtColor(guide, cv.COLOR_BGR2GRAY)
        adv = None
        if self.history is not None:
            gt, prev, ls = self.history
            flow = cv.calcOpticalFlowFarneback(G, gt, None, 0.5, 3, 13, 3, 5, 1.1, 0)
            p = coord(CW, CH) + np.clip(flow, -8, 8)
            adv = cv.remap(
                prev.astype("f"),
                p,
                None,
                cv.INTER_NEAREST,
                borderMode=cv.BORDER_REPLICATE,
            ).astype(int)
            lookup = np.array(ls)[adv]
            for j, (lens, n, layer) in enumerate(ids):
                score[j] += 0.60 * (
                    (lookup[:, :, 0] == lens)
                    & (abs(lookup[:, :, 1] - n) < 9)
                    & (lookup[:, :, 2] == layer)
                )
        labels = np.zeros((CH, CW), "B")
        best = score[0].copy()
        oldc = colors[0].copy()
        for j in range(len(cards) - 1, 0, -1):
            # Slight front/earlier-pick preference; neighboring frames can only win when they really cover a hole.
            sc = score[j]
            legal = scores[j] > 0.009
            delta = sc - best
            if not (legal & (delta > 0.05)).any():
                continue
            newc = colors[j]
            eg = np.maximum(grad(newc), grad(oldc))
            halo = cv.dilate(np.clip(eg / 18, 0, 2), np.ones((5, 5), "B"))
            diff = abs(oldc.astype("f") - newc).mean(2)
            barrier = 3 + 85 * halo**2 + np.minimum(30, diff * 0.55)
            win = cut(delta, barrier, legal, np.ones_like(legal))
            labels[win] = j
            best[win] = sc[win]
            oldc[win] = newc[win]
        self.history = G, labels, ids
        # Smooth the placement of the boundary, not the colours. No alpha feather across different projections.
        hard = np.zeros((OH, OW), "B")
        high = np.full((OH, OW), -1, "f")
        for j in np.unique(labels):
            field = cv.resize(blur((labels == j).astype("f"), 0.82), (OW, OH))
            win = field > high
            hard[win] = j
            high[win] = field[win]
        # At half scale cluster real contours seen on either side of a seam. A single
        # observation must own the whole conflicting contour and its 40px exclusion band.
        # This prevents a cut between two displaced copies of a pole even though neither copy
        # was alpha-blended. Never extend a photo across a mover or out of the lens image.
        half = cv.resize(hard, (OW // 2, OH // 2), interpolation=cv.INTER_NEAREST)
        Y, X = half.shape
        boundary = cv.morphologyEx(half, cv.MORPH_GRADIENT, np.ones((3, 3), "B")) > 0
        zone = cv.dilate(boundary.astype("B"), np.ones((45, 45), "B")) > 0
        ed = np.zeros_like(half)
        legal2 = {}
        for j in np.unique(half):
            own = half == j
            near = cv.dilate(own.astype("B"), np.ones((43, 43), "B")) > 0
            patch = near & zone
            legal2[j] = cv.resize(scores[j], (X, Y)) > 0.022 if j else np.ones_like(own)
            if not patch.any():
                continue
            yy, xx = np.where(patch)
            x0 = max(0, xx.min() - 3)
            y0 = max(0, yy.min() - 3)
            x1 = min(X, xx.max() + 4)
            y1 = min(Y, yy.max() + 4)
            rc = (x0 * 2, y0 * 2, (x1 - x0) * 2, (y1 - y0) * 2)
            thumb = cv.resize(
                full(*cards[j], rc), (x1 - x0, y1 - y0), interpolation=cv.INTER_AREA
            )
            g = cv.cvtColor(thumb, cv.COLOR_BGR2GRAY)
            edge = cv.Canny(g, 43, 105, L2gradient=True)
            sl = np.s_[y0:y1, x0:x1]
            ed[sl] |= edge * (patch[sl] & legal2[j][sl])
        # Separate the upper structures from unrelated road texture: merge nearby displaced
        # contours in the seam corridor, not all foliage across the entire photograph.
        prot = cv.dilate(ed, cv.getStructuringElement(cv.MORPH_ELLIPSE, (23, 23)))
        n, cc, st, _ = cv.connectedComponentsWithStats(prot, 8)
        lock = np.zeros_like(half)
        locked = 0
        unresolved = 0
        for k in np.argsort(-st[1:, 4]) + 1:
            x, y, w, h, area = st[k]
            sl = np.s_[y : y + h, x : x + w]
            m = cc[sl] == k
            if area < 50 or not (m & boundary[sl]).any():
                continue
            local = half[sl]
            js, counts = np.unique(local[m], return_counts=True)
            candidates = [
                (n / area, j)
                for j, n in zip(js, counts)
                if j != 0 and legal2[j][sl][m].all()
            ]
            if not candidates:
                unresolved += 1
                continue
            _, j = max(candidates)
            lock[sl][m] = j
            locked += 1
        use = lock != 0
        half[use] = lock[use]
        # Upsample a categorical boundary with signed probabilities only to select a label.
        if use.any():
            high.fill(-1)
            for j in np.unique(half):
                v = cv.resize(blur((half == j).astype("f"), 0.9), (OW, OH))
                win = v > high
                hard[win] = j
                high[win] = v[win]
        out = np.empty((OH, OW, 3), "B")
        patches = []
        for j in np.unique(hard):
            ys, xs = np.where(hard == j)
            x = max(0, xs.min() - 44)
            y = max(0, ys.min() - 44)
            x1 = min(OW, xs.max() + 45)
            y1 = min(OH, ys.max() + 45)
            r = (int(x), int(y), int(x1 - x), int(y1 - y))
            rgb = full(*cards[j], r)
            sub = out[y:y1, x:x1]
            mask = hard[y:y1, x:x1] == j
            sub[mask] = rgb[mask]
            patches.append((int(j), r, rgb))
        meta = dict(
            t=t,
            photo=float((hard != 0).mean()),
            owners=len(patches),
            lockedContours=int(locked),
            unresolvedContours=int(unresolved),
            primary=uses[0][:3],
            uses=uses,
        )
        self.ids_for_report = {
            j: (("floor", -1, "floor") if k == "floor" else (q["c"], q["n"], k))
            for j, (q, k) in enumerate(cards)
        }
        return out, meta, hard, patches, oldowner, oldcards
