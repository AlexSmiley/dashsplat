"""Retain the existing rig's full semantic labels. Road-only masks lose the crucial terrain-versus-hedge distinction."""

from .outer import *
from pipeline.masks.seg import Segmenter


def main():
    rows = read(W / "outer-rows.json")
    out = W / "classes"
    out.mkdir(exist_ok=True)
    seg = Segmenter(fp16=True)
    jobs = [
        (i + 1, r) for i, r in enumerate(rows) if not (out / f"{i + 1}.png").exists()
    ]
    start = time.monotonic()
    for j in range(0, len(jobs), 3):
        todo = jobs[j : j + 3]
        ims = [
            cv.resize(
                image(r["cam"], r["n"], True), (724, 469), interpolation=cv.INTER_AREA
            )[:, :, ::-1]
            for _, r in todo
        ]
        outmaps = seg(ims)
        for (id, r), v in zip(todo, outmaps):
            cv.imwrite(
                str(out / f"{id}.png"), v["semantic"], [cv.IMWRITE_PNG_COMPRESSION, 3]
            )
        if j % 240 == 0:
            print("classes", j, len(jobs), round(time.monotonic() - start), flush=True)
    print("classes ready", flush=True)


if __name__ == "__main__":
    main()
