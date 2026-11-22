"""Same-polarity outlines in confident held-out teacher pixels. Reject paired edges already present in the single-photo reference."""

from .v4teacher import OUT, split, decode, Path, savej, cv, np
from .v2eval import Wide
from concurrent.futures import ThreadPoolExecutor
import sys


def edges(a, strong):
    g = cv.GaussianBlur(
        cv.cvtColor(np.uint8(a.clip(0, 255)), cv.COLOR_RGB2GRAY), (3, 3), 0.7
    )
    e = cv.Canny(g, 42 if strong else 22, 95 if strong else 55, L2gradient=True)
    gx = cv.Sobel(g, cv.CV_32F, 1, 0)
    gy = cv.Sobel(g, cv.CV_32F, 0, 1)
    d = np.hypot(gx, gy).clip(1)
    return e, gx / d, gy / d


def count(im, ref, valid):
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
    return [int((support & hit).sum()), len(ex)]


def run(src, tag):
    _, held = split()
    es = [e for e in held if e["kind"] == "chase"][::3]
    r = Wide(src)

    def one(e):
        im, w, sky, d = decode(e)
        T = np.array(e["T_world_cam"])
        k = e["K"]
        valid = cv.erode((w >= 128).astype("B"), np.ones((7, 7), "B")).astype(bool)
        a = r.draw(
            e["t"],
            np.repeat(T[None], 33, axis=0),
            np.r_[k, np.zeros(4)],
            (1440, 900),
            0,
        )
        ref = cv.dilate(edges(im, False)[0], np.ones((3, 3), "B"))
        return dict(
            id=e["id"],
            render=count(a[:, :, :3], ref, valid),
            self=count(im, ref, valid),
            injected=count(
                cv.addWeighted(im, 0.5, np.roll(im, 9, 1), 0.5, 0), ref, valid
            ),
        )

    with ThreadPoolExecutor(3) as ex:
        rows = list(ex.map(one, es))
    stats = {
        k: np.sum([e[k] for e in rows], axis=0).tolist()
        for k in ["render", "self", "injected"]
    }
    savej(OUT / ("outlines-" + tag + ".json"), dict(rows=rows, totals=stats))
    print(tag, stats, flush=True)


if __name__ == "__main__":
    run(Path(sys.argv[1]), sys.argv[2])
