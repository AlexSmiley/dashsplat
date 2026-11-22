"""Blind station-wise errors: full-resolution composited splats, fixed masks, frozen physical holdouts."""

from .v4teacher import OUT, split, decode
from .v2eval import *
from .v2data import Volume
import sys


def measure(src, tag, native):
    dst = OUT / "proof" / tag
    dst.mkdir(exist_ok=True, parents=True)
    r = Wide(src)
    vol = Volume()
    _, heldt = split()
    Xg, Yg = np.meshgrid(
        np.arange(1440, dtype="f") + 0.5, np.arange(900, dtype="f") + 0.5
    )

    def one(e):
        target, w, sky, z = decode(e)
        T = np.array(e["T_world_cam"], "d")
        K = e["K"]
        a = r.draw(
            e["t"], np.repeat(T[None], 33, 0), np.r_[K, np.zeros(4)], (1440, 900), 0
        )
        im = a[:, :, :3]
        conf = w >= 128
        weight = w.astype("f") / 255
        err = abs(im - target).mean(2)
        mse = ((im - target) ** 2).mean(2)
        y, x = np.nonzero(conf)
        q = z[y, x].astype("f") * 0.01
        P = (
            T[:3, 3]
            + np.c_[(Xg[y, x] - K[2]) / K[0] * q, (Yg[y, x] - K[3]) / K[1] * q, q]
            @ T[:3, :3].T
        )
        g = vol.lookup(P)[0]
        road = abs(P[:, 2] - g) < 0.34
        er = err[y, x]
        edge = (
            cv.Sobel(cv.cvtColor(target, cv.COLOR_RGB2GRAY).astype("f"), -1, 1, 0) / 8
        )
        edge_y = (
            cv.Sobel(cv.cvtColor(target, cv.COLOR_RGB2GRAY).astype("f"), -1, 0, 1) / 8
        )
        gs = np.hypot(edge, edge_y)
        l = cv.cvtColor(im, cv.COLOR_RGB2GRAY)
        gi = np.hypot(cv.Sobel(l, -1, 1, 0), cv.Sobel(l, -1, 0, 1)) / 8
        valid = cv.erode(conf.astype("B"), np.ones((5, 5), "B"))
        hard = (gs[y, x] > 10) & road & (valid[y, x] > 0)
        return dict(
            id=e["id"],
            station=e["station"],
            kind=e["kind"],
            pixels=len(x),
            mae=float(er.mean()),
            wmae=float((err * weight).sum() / max(1, weight.sum())),
            psnr=float(-10 * np.log10(max(0.001, mse[y, x].mean()) / 255**2)),
            road_mae=float(er[road].mean()) if road.any() else None,
            object_mae=float(er[~road].mean()) if (~road).sum() > 100 else None,
            edge_ratio=float(gi[y[hard], x[hard]].sum() / gs[y[hard], x[hard]].sum())
            if hard.sum() > 20
            else None,
            sky_alpha=float(a[:, :, 3][sky > 0].mean()) if sky.any() else None,
        )

    with ThreadPoolExecutor(4) as ex:
        rows = list(ex.map(one, heldt))
        savej(dst / "teachers.json", rows)
    for variant in ["chase", "offset"]:
        seq = [e for e in rows if (e["kind"] == "chase") == (variant == "chase")]
        print(
            tag,
            variant,
            len(seq),
            {
                k: round(float(np.mean([a[k] for a in seq if a[k] is not None])), 4)
                for k in [
                    "mae",
                    "wmae",
                    "psnr",
                    "road_mae",
                    "object_mae",
                    "edge_ratio",
                    "sky_alpha",
                ]
            },
            flush=True,
        )
    if not native:
        return

    def photo(e):
        ref, valid = reference(e)
        a = r.real(e)
        jpg(
            dst / "after/camera" / e["c"] / ("%06d.jpg" % e["n"]),
            np.uint8(a[:, :, :3].clip(0, 255)),
        )
        return dict(**e, **scores(ref, a, valid))

    with ThreadPoolExecutor(5) as ex:
        savej(dst / "native.json", list(ex.map(photo, held())))
    if not (dst / "before").exists():
        (dst / "before").symlink_to(
            RS / "work/stitch-v3/proof/after", target_is_directory=True
        )
    perceptual.REPORT = dst
    perceptual.run()
    verify.WORK = dst
    verify.run(src)


if __name__ == "__main__":
    measure(Path(sys.argv[1]), sys.argv[2], len(sys.argv) > 3)
