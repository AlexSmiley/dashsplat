"""Photo sky, conservative far-field cylindrical matte and a shared front-reference white balance."""

from .popup_layers import *
from .pack import photo


def finish_lens(c):
    doc = json.load(open(DST / (c + ".json")))
    b = bank()
    front = b["front"]["keyframes"]
    ts = np.array([a["t"] for a in front])
    raw = {a["n"]: a for a in b[c]["keyframes"]}
    p = photo("live")

    def enrich(e):
        im = cv.resize(
            cv.imread(str(STAGE / raw[e["n"]]["rgb"])),
            (B, H),
            interpolation=cv.INTER_AREA,
        )
        cat = cv.imread(str(STAGE / raw[e["n"]]["mask"]), -1)
        sem = cv.imread(
            str(semfile(c, e)), 0
        )  # Keep high-frequency sky pixels; extend low-frequency colour into masked holes.
        sky = cv.resize(skyframe(c, raw[e["n"]]), (B, H))
        skyMask = (cat == 3).astype("B")
        sg = np.clip(cv.distanceTransform(skyMask, cv.DIST_L2, 3) / 3, 0, 1)[:, :, None]
        sky = np.uint8(np.clip(im * sg + sky * (1 - sg), 0, 255))
        cv.imwrite(str(DST / e["sky"]), sky, [cv.IMWRITE_JPEG_QUALITY, 91])
        lab, ls = segment(c, raw[e["n"]], True)
        far = np.zeros(len(ls) + 1, "B")
        for a in ls:
            far[a["local"]] = a["source"] == "backdrop"
        mat = far[
            lab
        ]  # Unknown objects are allowed onto the common far shell only when the connected component is genuinely distant.
        unknown = (
            (cat == 1)
            & (lab == 0)
            & ~np.isin(sem, [0, 1, 9, 10, 11, 12, 13, 14, 15, 17, 18])
        )
        z = cv.imread(str(STAGE / raw[e["n"]]["z"]), -1) * 0.01
        num, idx, stats, _ = cv.connectedComponentsWithStats(unknown.astype("B"), 8)
        for n, (x, y, w, h, area) in enumerate(stats[1:], 1):
            if area < 30:
                continue
            view = np.s_[y : y + h, x : x + w]
            sel = idx[view] == n
            zz = z[view][sel]
            zz = zz[zz > 0.3]
            if len(zz) > 12 and np.median(zz) > 57:
                mat[view][sel] = 1
        # Keep sky/ground apart; no ground or instance pixel can enter the horizon atlas.
        mat &= cat == 1
        ys, xs = np.where(mat)
        e["horizon"] = None
        if len(xs):
            x, y, X, Y = (
                max(0, int(xs.min()) - 2),
                max(0, int(ys.min()) - 2),
                min(B, int(xs.max()) + 3),
                min(H, int(ys.max()) + 3),
            )
            st = "base/" + c + f"/{e['n']:06}-far"
            crop = np.s_[y:Y, x:X]
            alpha = np.uint8(
                np.clip((cv.distanceTransform(mat, cv.DIST_L2, 3) - 0.20) / 1.45, 0, 1)
                * 255
            )
            color = im[crop].copy()
            color[alpha[crop] == 0] = 0
            cv.imwrite(str(DST / (st + ".jpg")), color, [cv.IMWRITE_JPEG_QUALITY, 90])
            cv.imwrite(
                str(DST / (st + ".png")), alpha[crop], [cv.IMWRITE_PNG_COMPRESSION, 3]
            )
            e["horizon"] = dict(rgb=st + ".jpg", alpha=st + ".png", box=[x, y, X, Y])
        e["layers"] = [a for a in e["layers"] if a["anchor"] != "backdrop"]
        return e

    with ThreadPoolExecutor(4) as pool:
        doc["base"] = list(pool.map(enrich, doc["base"]))
    for f in doc["base"] + doc["movers"]:
        _, g, _ = p.params("front", front[np.argmin(abs(ts - f["t"]))]["n"])
        f["gain"] = np.exp(np.mean(g) - g).tolist()
    dump(DST / (c + "-final.json"), doc)
    print("horizon", c, len(doc["base"]), flush=True)


if __name__ == "__main__":
    for c in LENSES:
        finish_lens(c)
