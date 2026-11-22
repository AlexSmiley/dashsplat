"""Photographic category corroboration for mask false positives."""

import pickle, cv2, numpy as np
from .classifier import classify
from .online import camera
from .demo import R


def release(scores, cls):
    labels = {
        "person": ["people", "pedestrian", "person"],
        "cycle": ["bicycle", "motorcycle", "cyclist"],
    }.get(
        "person"
        if cls == "person"
        else "cycle"
        if cls in ("bicycle", "rider", "motorcycle")
        else "",
        ["automobile", "bus", "truck", "car", "van", "suv", "motorhome"],
    )
    for t, c, items in scores:
        if not items:
            continue
        lab = dict(items)
        v = max(lab.get(k, 0.0) for k in labels)
        if v >= 0.045 and v >= 0.42 * items[0][1] and v > 1.5 * lab.get("sign", 0.0):
            return t
    return None


def run():
    root = R / "semantics"
    root.mkdir(exist_ok=True)
    hs = pickle.load(open(R / "identities.pickle", "rb"))
    todo = []
    meta = []
    for h in hs:
        if h["confirm"] is None:
            continue
        bycam = {}
        for r in h["rows"]:
            photo = R.name.startswith("photo-")
            spacing = (
                0.21 if photo and r["release"] - h["rows"][0]["release"] < 1.3 else 0.58
            )
            if (
                r["score"] < (0.82 if photo else 0.94)
                or r["t"] - bycam.get(r["c"], -100) < spacing
            ):
                continue
            bycam[r["c"]] = r["t"]
            cam = camera(r["c"])
            l, t, bx, by = r["bb"]
            cx, cy = (l + bx) / 2, (t + by) / 2
            w = (bx - l) * 1.27 + 4
            hh = (by - t) * 1.3 + 4
            xx = np.clip(
                np.array([cx - w * 0.5, cx + w * 0.5]) * cam.width / 724, 0, cam.width
            ).astype(int)
            yy = np.clip(
                np.array([cy - hh * 0.5, cy + hh * 0.5]) * cam.height / 469,
                0,
                cam.height,
            ).astype(int)
            im = cv2.imread(f"data/drive/frames/{r['c']}/{r['n']:06}.jpg")
            crop = im[yy[0] : yy[1], xx[0] : xx[1]]
            ratio = 232 / max(crop.shape[:2])
            crop = cv2.resize(
                crop,
                (
                    max(1, int(crop.shape[1] * ratio)),
                    max(1, int(crop.shape[0] * ratio)),
                ),
            )
            dy = (232 - crop.shape[0]) // 2
            dx = (232 - crop.shape[1]) // 2
            crop = cv2.copyMakeBorder(
                crop,
                dy,
                232 - crop.shape[0] - dy,
                dx,
                232 - crop.shape[1] - dx,
                cv2.BORDER_REPLICATE,
            )
            file = root / f"{h['id']}-{r['c']}-{r['n']}.jpg"
            cv2.imwrite(str(file), crop)
            todo.append(str(file.absolute()))
            meta.append((h["id"], r["release"], r["c"]))
    scores = classify(todo, root)
    out = {}
    for (uid, t, c), items in zip(meta, scores, strict=True):
        out.setdefault(uid, []).append((t, c, items))
    pickle.dump(out, open(R / "semantics.pickle", "wb"))
    print(len(meta), "independent crop scores", flush=True)


if __name__ == "__main__":
    run()
