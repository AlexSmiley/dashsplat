"""Use the producer's unmodified Cityscapes Segmenter; retain all semantic classes privately."""

from .pack import STAGE, RS, json, cv, np, Path
from .render import selection_frames

ROOT = RS / "work/ibr/popup-v2"
LENSES = ["front", "left_pillar", "right_pillar"]


def bank():
    return json.load(open(STAGE / "ibr.json"))["cameras"]


def bases(es):
    return [es[k] for k in selection_frames(es)]


def semfile(c, e):
    return ROOT / "sem" / c / f"{e['n']:06}.png"


def main():
    from pipeline.masks.seg import Segmenter

    seg = Segmenter()
    b = bank()
    for c in LENSES:
        es = bases(b[c]["keyframes"])
        es = [e for e in es if not semfile(c, e).exists()]
        (ROOT / "sem" / c).mkdir(parents=True, exist_ok=True)
        for i in range(0, len(es), 4):
            ee = es[i : i + 4]
            ims = [
                cv.resize(
                    cv.imread(str(RS / "data/drive/frames" / c / f"{e['n']:06}.jpg"))[
                        :, :, ::-1
                    ],
                    (724, 469),
                    interpolation=cv.INTER_AREA,
                )
                for e in ee
            ]
            out = seg(ims)
            for e, o in zip(ee, out):
                assert cv.imwrite(str(semfile(c, e)), o["semantic"])
        print(c, len(es), flush=True)


if __name__ == "__main__":
    main()
