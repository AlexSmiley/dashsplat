"""Matched physical *held* object crops at native angular resolution, independent of the optimisation rasterizer."""

from .v5fit import physical_patch
from .v5world import *
from . import v2eval as E


def inspect(src, tag):
    E.RESULT = RS / "work/stitch-v3/proof"
    model = E.Wide(src)
    out = W5 / "proof" / tag
    out.mkdir(exist_ok=True, parents=True)
    res = []
    rng = np.random.default_rng(520)
    meta = []
    for c in CAMS:
        for e in [e for e in schedule()[c] if e["hold"] and semantic(c, e["n"])]:
            for k in [8, 2, 5, 7]:
                ex = physical_patch(e, rng, k)
                if not ex:
                    continue
                T, K, im, L = ex
                mask = L == k
                sky = L == 10
                hh, ww = L.shape
                if mask.sum() < 350:
                    continue
                a = model.draw(
                    e["t"],
                    np.repeat(T[None], 33, 0),
                    np.r_[K, np.zeros(4)],
                    (ww, hh),
                    0,
                )
                y, x = np.mgrid[:hh, :ww]
                ray = E.world_ray(T, K, x, y)
                pan = model.pan
                H, W = pan.shape[:2]
                skybg = cv.remap(
                    pan,
                    (
                        (0.5 - np.arctan2(ray[:, :, 1], ray[:, :, 0]) / (2 * np.pi)) * W
                        - 0.5
                    ).astype("f"),
                    ((0.5 - np.arcsin(ray[:, :, 2]) / np.pi) * H - 0.5).astype("f"),
                    cv.INTER_LINEAR,
                    borderMode=cv.BORDER_WRAP,
                )
                render = np.clip(
                    a[:, :, :3] + (1 - a[:, :, 3:4]) * (skybg - E.BG), 0, 255
                ).astype("B")
                grad = lambda a: np.hypot(
                    *[
                        cv.Sobel(
                            cv.cvtColor(a, cv.COLOR_RGB2GRAY).astype("f"), -1, i, 1 - i
                        )
                        / 8
                        for i in [0, 1]
                    ]
                )
                tg = grad(im)
                rg = grad(render)
                edge = (tg > 10) & mask
                rec = dict(
                    c=c,
                    n=e["n"],
                    t=e["t"],
                    k=k,
                    mae=float(abs(render.astype("f") - im)[mask].mean()),
                    contrast=float(rg[edge].sum() / tg[edge].sum())
                    if edge.sum() > 30
                    else None,
                    alpha=float(a[:, :, 5][mask].mean()),
                    sky_alpha=float(a[:, :, 5][sky].mean())
                    if sky.sum() > 150
                    else None,
                    edges=int(edge.sum()),
                    sky=int(sky.sum()),
                )
                res.append(rec)
                f = f"{c}-{e['n']}-{k}.jpg"
                cv.imwrite(str(out / f), np.hstack([im, render])[:, :, ::-1])
                meta.append(f)
    savej(out / "crops.json", res)
    print(
        tag,
        len(res),
        [(k, np.mean([e["mae"] for e in res if e["k"] == k])) for k in [8, 2, 5, 7]],
        flush=True,
    )


if __name__ == "__main__":
    inspect(Path(sys.argv[1]), sys.argv[2])
