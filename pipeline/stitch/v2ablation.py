"""Training-only ablation selection was fixed by the orbit; evaluate seed and full RGB fit afterwards."""

from .v2eval import *
from .v2data import V


def measure():
    data = {}
    for st, src in [
        ("dense_photo_prior", V / "photo-seed"),
        ("unfiltered_fit", V / "photo-guard"),
    ]:
        tic = time.monotonic()
        r = Wide(src)

        def one(e, renderer=r):
            im, valid = reference(e)
            a = renderer.real(e)
            return dict(**e, **scores(im, a, valid))

        with ThreadPoolExecutor(4) as ex:
            rows = list(ex.map(one, held()))
        savej(RESULT / (st + "-held.json"), rows)
        data[st] = {
            k: float(np.mean([a[k] for a in rows]))
            for k in ["psnr", "ssim", "coverage"]
        }
        data[st]["seconds"] = time.monotonic() - tic
        del one, r
    savej(RESULT / "camera-ablation.json", data)


if __name__ == "__main__":
    measure()
