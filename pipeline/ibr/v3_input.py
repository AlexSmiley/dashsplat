"""Task K source inventory. Keep the frozen Task I and J banks intact."""

from . import pack
from .popup_layers import *
from . import popup_layers as layers

W = RS / "work/ibr/popup-v3"
BUILD = W / "staging"
SCENE = RS / "scene/ibr-v1"
LO, HI = 93.0, 140.0
INP = W / "inputs.json"
C3 = ["front", "left_pillar", "right_pillar"]
ALL = list(pack.CAMS)


def load():
    return json.load(open(INP))


def sem(c, e):
    q = semfile(c, e)
    return q if q.exists() else W / "sem" / c / f"{e['n']:06d}.png"


def getsets():
    b = load()
    es = [e for e in b["front"] if 84 <= e["t"] <= 140.7]
    idx = selection_frames(es)
    sets = []
    for i in idx:
        ls = {c: min(b[c], key=lambda f: abs(f["t"] - es[i]["t"])) for c in ALL}
        sets.append({c: e for c, e in ls.items() if abs(e["t"] - es[i]["t"]) < 0.50})
    return sets


from .render import selection_frames


def inputs():
    W.mkdir(parents=True, exist_ok=True)
    tail = W / "tail"
    pack.STAGE = tail
    pack.T0 = 137.251
    pack.T1 = 140.8
    pack.build()
    b = {}
    for root in [STAGE, tail]:
        doc = json.load(open(root / "ibr.json"))
        for c, j in doc["cameras"].items():
            for e in j["keyframes"]:
                if e["t"] < 83:
                    continue
                b.setdefault(c, []).append(
                    dict(e, **{k: str(root / e[k]) for k in ["rgb", "mask", "z"]})
                )
    dump(INP, b)
    from pipeline.masks.seg import Segmenter

    model = Segmenter()
    S = getsets()
    for c in C3:
        es = {s[c]["n"]: s[c] for s in S if c in s}
        todo = [e for e in es.values() if not sem(c, e).exists()]
        (W / "sem" / c).mkdir(parents=True, exist_ok=True)
        for i in range(0, len(todo), 4):
            ee = todo[i : i + 4]
            out = model(
                [
                    cv.resize(
                        cv.imread(
                            str(RS / "data/drive/frames" / c / f"{e['n']:06}.jpg")
                        )[:, :, ::-1],
                        (B, H),
                        interpolation=cv.INTER_AREA,
                    )
                    for e in ee
                ]
            )
            for e, o in zip(ee, out):
                assert cv.imwrite(str(sem(c, e)), o["semantic"])
        print(c, "knots", len(es), "new semantics", len(todo), flush=True)


if __name__ == "__main__":
    inputs()
