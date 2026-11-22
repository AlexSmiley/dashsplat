"""Before/after in exactly the same physical patch; owners, old hole coordinates, and distance-binned pavement strip."""

from collections import Counter
from . import proof, audit
from .proof import *
from .v5 import SLICES


def open_at(work, out):
    proof.OUT = Path(out)
    audit.W = Path(work)
    audit.state.cache_clear()
    a = Atlas()
    obs = read(Path(work) / "observations.json")
    return a, obs


def holes(v):
    bad = np.uint8(v < 0.1)
    n, lab, st, _ = cv.connectedComponentsWithStats(bad, 8)
    internal = [
        i
        for i in range(1, n)
        if st[i, 0] > 0
        and st[i, 1] > 0
        and st[i, 0] + st[i, 2] < bad.shape[1]
        and st[i, 1] + st[i, 3] < bad.shape[0]
        and st[i, 4] >= 4
    ]
    return np.isin(lab, internal), len(internal)


def strip(at, lo, hi):
    tr = inputs()[0]
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(tr.p[:, :2], axis=0), axis=1))]
    a, b = np.interp([lo, hi], tr.t, arc)
    t = np.interp(np.arange(a, b, 0.16), arc, tr.t)
    T = tr.T_world_ego(t)
    xy = T[:, :2, 3]
    f = T[:, :2, 0]
    f = f / np.linalg.norm(f, axis=1)[:, None]
    side = np.stack([f[:, 1], -f[:, 0]], 1)
    p = xy[None, :, :] + np.arange(-8, 8.01, 0.10)[:, None, None] * side[None, :, :]
    pix = at.lookup(p, np.full(p.shape[:2], 0.12))
    return np.uint8(pix[:, :, :3] + (1 - pix[:, :, 3:4] / 255) * 28)


def main():
    before = Path("work/groundtex-v5/before")
    newwork = W
    newout = OUT
    cfg = read("pipeline/groundtex/patches-v5.json")
    tab = {}
    stash = {}
    st = []

    def inspect(at, obs, zz):
        xy, _, _ = top_xy(zz["t"], zz["along"], zz["side"], zz["w"], zz["h"], 0.01)
        rgb, a = at.rgba(xy)
        a = a[:, :, 0]
        id = audit.owners(at, xy)
        cam = Counter(obs[i - 1]["cam"] for i in id[(id > 0) & (a > 0.98)].ravel())
        mask, count = holes(a)
        return (
            dict(
                camera_pixels=dict(cam),
                opaque_fraction=float((a > 0.98).mean()),
                interior_holes=count,
                interior_missing_pixels=int(mask.sum()),
            ),
            a,
            mask,
        )

    p0, obs = open_at("work/groundtex-final/state", "work/groundtex-final/ground")
    for zz in cfg:
        d, a, mask = inspect(p0, obs, zz)
        tab[zz["tag"]] = {"before": d}
        stash[zz["tag"]] = mask
    for lo, hi in SLICES:
        st.append(strip(p0, lo, hi))
    p1, obs = open_at(newwork, newout)
    for zz in cfg:
        tag = zz["tag"]
        d, a, mask = inspect(p1, obs, zz)
        d["original_hole_pixels_now_opaque"] = int((stash[tag] & (a > 0.98)).sum())
        tab[tag]["after"] = d
        old = np.asarray(Image.open(before / f"detail-{tag}.jpg"))
        new = np.asarray(Image.open(RPT / f"detail-{tag}.jpg"))
        w = zz["w"]
        save(
            f"compare-{tag}.jpg",
            np.hstack(
                [
                    label(old[28:, w : 2 * w], "Committed atlas"),
                    label(new[28:, w : 2 * w], "V5 atlas"),
                    new[:, 2 * w :],
                ]
            ),
        )
    for (lo, hi), old in zip(SLICES, st):
        new = strip(p1, lo, hi)
        W0 = 1200
        rows = []
        seg = round(120 / 0.16)
        for x in range(0, old.shape[1], seg):
            line = []
            for arr, txt in [(old, "old"), (new, "v5")]:
                v = arr[:, x : x + seg]
                blank = np.full((v.shape[0], seg, 3), 28, "B")
                blank[:, : v.shape[1]] = v
                line.append(
                    label(
                        cv.resize(blank, (W0 // 2, 100), interpolation=cv.INTER_AREA),
                        f"{txt}: {x * 0.16:.0f} m from slice start",
                    )
                )
            rows.append(np.hstack(line))
        save(f"road-strip-{lo}-{hi}.jpg", np.vstack(rows))
    dump(RPT / "patch-comparison.json", tab)


if __name__ == "__main__":
    main()
