"""The v4 paired-outline detector with sampled anchor positions preserved for world-tile attribution."""

from pipeline.stitch.v4outline import edges, cv, np


def edge_locations(im, ref, valid):
    e, nx, ny = edges(im, True)
    H, W = valid.shape
    ey, ex = np.nonzero((e > 0) & valid)
    q = (ex + 3 * ey) % 4 == 0
    ey, ex = ey[q], ex[q]
    dx, dy = nx[ey, ex], ny[ey, ex]
    hit = np.zeros(len(ex), bool)
    found = ref[ey, ex] > 0
    for ds in list(range(3, 31)) + list(range(-30, -2)):
        x = (ex + np.rint(dx * ds).astype(int)).clip(0, W - 1)
        y = (ey + np.rint(dy * ds).astype(int)).clip(0, H - 1)
        hit |= (
            valid[y, x]
            & (e[y, x] > 0)
            & (nx[y, x] * dx + ny[y, x] * dy > 0.945)
            & (found != (ref[y, x] > 0))
        )
    mark = np.zeros_like(e)
    mark[ey[hit], ex[hit]] = 1
    mark = cv.dilate(mark, np.ones((3, 3), "B"))
    support = np.zeros_like(hit)
    for ds in [-6, -4, 4, 6]:
        support |= (
            mark[
                (ey + np.rint(dx * ds).astype(int)).clip(0, H - 1),
                (ex - np.rint(dy * ds).astype(int)).clip(0, W - 1),
            ]
            > 0
        )
    return ey, ex, (support & hit)
