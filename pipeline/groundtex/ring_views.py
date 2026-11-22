"""Project published atlas provenance onto the actual browser chase camera. Exclude HUD in the gray-pixel test."""

from .outer import *
from .proof import Atlas, RPT


class Provenance(Atlas):
    @lru_cache(80)
    def data(self, k, l):
        a = np.asarray(Image.open(OUT / self.idx[k]["provenance"]))
        return np.dstack([a, a, a, np.full_like(a, 255)])


def proof_at(path):
    from . import proof

    old = proof.OUT
    proof.OUT = path

    class Before(Atlas):
        @lru_cache(64)
        def data(self, k, l):
            return np.asarray(Image.open(path / self.idx[k]["texture"]).convert("RGBA"))

    a = Before()
    proof.OUT = old
    return a


def render(view, folder):
    t = view["t"]
    before = np.array(Image.open(folder / f"before-{t}.jpg"))
    after = np.array(Image.open(folder / f"check-{t}.jpg"))
    H, W0 = after.shape[:2]
    w, h = W0 // 3, H // 3
    xx, yy = np.meshgrid(
        (np.arange(w) + 0.5) * 2 / w - 1, 1 - (np.arange(h) + 0.5) * 2 / h
    )
    proj = np.array(view["projection"]).reshape(4, 4).T
    world = np.array(view["world"]).reshape(4, 4).T
    ray = (
        np.stack([xx / proj[0, 0], yy / proj[1, 1], -np.ones_like(xx)], -1)
        @ world[:3, :3].T
    )
    ey = world[:3, 3]
    at = Atlas()
    pv = Provenance()
    z0 = at.z(ey[:2].reshape(1, 2), ey[2] - 5.3)[0]
    dep = np.clip((z0 - ey[2]) / np.minimum(-0.025, ray[:, :, 2]), 0, 160)
    xy = ey[:2] + dep[:, :, None] * ray[:, :, :2]
    for i in range(4):
        z = at.z(xy, z0)
        dep = np.clip((z - ey[2]) / np.minimum(-0.025, ray[:, :, 2]), 0, 160)
        xy = ey[:2] + dep[:, :, None] * ray[:, :, :2]
    label = pv.lookup(xy)[:, :, 0]
    aa = at.lookup(xy)[:, :, 3]
    r = height_model()[0].query(xy)[0]
    eligible = (ray[:, :, 2] < -0.025) & (dep < 150) & (r < 40) & (aa > 240)
    outer = (
        eligible
        & (r > 4)
        & (yy > -0.49)
        & (abs(xx) < 0.90)
        & ~((xx < -0.73) & (yy > 0.14))
    )
    previous = proof_at(B)
    v5alpha = previous.lookup(xy)[:, :, 3]
    outer &= v5alpha < 26
    a = cv.resize(after, (w, h), interpolation=cv.INTER_AREA)
    b = cv.resize(
        before, (w, h), interpolation=cv.INTER_AREA
    )  # Literal viewer apron: locally flat, near-neutral, middle-grey, rather than grass/pavement texture or white sky.

    def grey(im):
        y = im.mean(2)
        span = im.max(2) - im.min(2)
        var = np.maximum(0, cv.blur(y * y, (7, 7)) - cv.blur(y, (7, 7)) ** 2)
        return (
            (np.abs(im.astype(float) - np.array([137, 140, 141])).max(2) < 7)
            & (span < 9)
            & (var < 2.5)
        )

    ga, gb = grey(a) & outer, grey(b) & outer
    over = a.copy()
    m = outer & (label >= 230)
    over[m] = 0.64 * a[m] + 0.36 * np.array([60, 191, 112])
    m = outer & (label > 20) & (label < 200)
    over[m] = 0.64 * a[m] + 0.36 * np.array([242, 160, 68])
    over[ga] = [221, 51, 99]
    out = np.hstack([a, over])
    Image.fromarray(out).save(RPT / f"ground-rays-{t}.jpg", quality=90)
    return dict(
        t=t,
        eligible_rays=int(outer.sum()),
        near_grey_before=int(gb.sum()),
        near_grey_after=int(ga.sum()),
        potentially_observed=int((outer & (label > 200)).sum()),
        potentially_predicted=int((outer & (label > 20) & (label < 200)).sum()),
    )


if __name__ == "__main__":
    import sys

    p = Path(sys.argv[1])
    dump(RPT / "view-rays.json", [render(v, p) for v in read(p / "poses.json")])
