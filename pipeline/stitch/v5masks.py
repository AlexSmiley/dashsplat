"""Supplement sparse cached panoptic observations, without consulting held physical exposures."""

from .v5world import *
from pipeline.masks.seg import Segmenter


def extra():
    model = Segmenter(fp16=True)
    todo = []
    for c in CAMS:
        last = -1e10
        for e in schedule()[c]:
            if e["hold"] or e["odo"] - last < 7.5:
                continue
            last = e["odo"]
            n = e["n"]
            dest = W5 / "sem" / c / f"{n:06}.png"
            if not semantic(c, n):
                todo.append((c, n, dest))
    for i in range(0, len(todo), 3):
        batch = todo[i : i + 3]
        imgs = [
            cv.resize(
                cv.imread(str(RS / "data/drive/frames" / c / f"{n:06}.jpg")),
                (724, 469),
                interpolation=cv.INTER_AREA,
            )[:, :, ::-1].copy()
            for c, n, p in batch
        ]
        pred = model(imgs)
        for (_, _, p), r in zip(batch, pred):
            p.parent.mkdir(parents=True, exist_ok=True)
            assert cv.imwrite(str(p), r["semantic"])
    print("supplemented", len(todo), flush=True)


if __name__ == "__main__":
    extra()
