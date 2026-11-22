"""Content-only pass: native-photo far texture and a free-space veto on the curtain matte."""

from .v3_pano import *
from .v3_publish import verify
from scipy.ndimage import gaussian_filter1d
from concurrent.futures import ThreadPoolExecutor
import shutil


def enhance():
    j = json.load(open(SCENE / "ibr.json"))
    assert j["contentRevision"] == 2
    sets = getsets()
    temp = W / "content-3"
    temp.mkdir(exist_ok=True)
    ww, hh = 4096, 448
    co = sphere(512, 56, BACK_EL)
    co /= np.linalg.norm(co[:, :, :2], axis=2)[:, :, None]
    els = BACK_EL[1] - (BACK_EL[1] - BACK_EL[0]) * (np.arange(hh) + 0.5) / hh

    def one(i):
        p = j["panoramas"][i]
        O = np.array(p["origin"])
        world = co * p["radius"] + O
        col = cv.resize(
            cv.imread(str(SCENE / p["rgb"])), (ww, hh), interpolation=cv.INTER_CUBIC
        ).astype("f")
        old = (
            cv.resize(cv.imread(str(SCENE / p["matte"]), 0), (ww, hh)).astype("f") / 255
        )
        best = np.zeros((hh, ww), "f")
        empty = np.zeros_like(best)
        truecol = np.zeros_like(col)
        seed = np.zeros_like(best)
        for c, e in sets[i].items():
            R, o = rowmat(e, np.array([0.5]))
            v = (world - o[0]) @ R[0]
            C = rig[c]
            uv, ok = C.project(v)
            uv = (
                np.nan_to_num(uv).astype("f") * np.float32([B / C.width, H / C.height])
                - 0.5
            )  # avoid blending through singular projections behind lenses
            uv = np.clip(uv, [-3 * B, -3 * H], [4 * B, 4 * H]).astype("f")
            edge = np.minimum.reduce(
                [uv[:, :, 0], B - 1 - uv[:, :, 0], uv[:, :, 1], H - 1 - uv[:, :, 1]]
            )
            taper = np.clip(edge / 36, 0, 1) * ok
            hi = cv.resize(uv, (ww, hh))
            w = cv.resize(taper, (ww, hh))
            cat = cv.imread(e["mask"], -1)
            Z = cv.imread(e["z"], -1) * 0.01
            r = rays(c)
            rr = (
                Z
                * np.linalg.norm((r @ R[0].T)[:, :, :2], axis=2)
                / np.maximum(0.06, r[:, :, 2])
            )
            good = (cat == 1) & (rr > 45) & (rr < 250)
            free = (cat == 3) | (cat == 2)
            sf = sem(c, e)
            if c in C3 and sf.exists():
                labels = cv.imread(str(sf), 0)
                ground = np.isin(labels, [0, 1, 9])
                good &= ~ground & ~np.isin(labels, [10, 11, 12, 13, 14, 15, 17, 18])
                free |= ground & (cat < 4)
            g = cv.remap(good.astype("f"), hi, None, cv.INTER_LINEAR) * w
            empty = np.maximum(
                empty, cv.remap(free.astype("f"), hi, None, cv.INTER_LINEAR) * w
            )
            rgb = cv.imread(e["rgb"])
            im = (
                cv.remap(rgb, 2 * hi + 0.5, None, cv.INTER_LINEAR).astype("f")
                * gain(c, e["t"])[::-1]
            )
            take = g > best
            best[take] = g[take]
            truecol[take] = im[take]
            seed = np.maximum(seed, g)
        # Use source RGB unchanged in reliable interiors. No averaging measured windows with misregistered neighbouring views.
        w = np.clip((best - 0.035) / 0.68, 0, 1)
        w = w * w * (3 - 2 * w)
        col += w[:, :, None] * (truecol - col)
        alpha = np.maximum(
            old, seed
        )  # True sky and ground are negative evidence, not disocclusions to fill.
        veto = np.clip((empty - seed - 0.08) / 0.75, 0, 1)
        alpha *= 1 - veto * veto * (3 - 2 * veto)
        # High, unsupported narrow structures (poles/signals) do not justify an opaque curtain under them.
        high = np.clip((els - 1.2) / 4, 0, 1)[:, None]
        solid = cv.GaussianBlur(seed, (0, 0), 7.0, sigmaY=3.0)
        alpha *= 1 - 0.68 * high * (
            1 - np.maximum(best, np.clip(solid * 4, 0, 1))
        ) * np.clip(empty * 2, 0, 1)
        colour = np.uint8(np.clip(col + 0.5, 0, 255))
        matte = np.uint8(np.clip(alpha * 255 + 0.5, 0, 255))
        name = f"{i:04}"
        assert cv.imwrite(
            str(temp / (name + ".jpg")), colour, [cv.IMWRITE_JPEG_QUALITY, 91]
        )
        assert cv.imwrite(
            str(temp / (name + ".png")), matte, [cv.IMWRITE_PNG_COMPRESSION, 4]
        )
        return [int((best > 0.7).sum()), int(((old > 0.5) & (alpha < 0.25)).sum())]

    with ThreadPoolExecutor(4) as pool:
        st = list(pool.map(one, range(len(sets))))
    os.rename(temp, SCENE / "backdrop-r3")
    backup = W / "ibr-revision2.json"
    shutil.copy2(SCENE / "ibr.json", backup)
    old = []
    for p in j["panoramas"]:
        old.extend([p["rgb"], p["matte"]])
        p["rgb"] = f"backdrop-r3/{p['id']:04}.jpg"
        p["matte"] = f"backdrop-r3/{p['id']:04}.png"
    j["contentRevision"] = 3
    dump(SCENE / "ibr.json", j)
    verify(SCENE)
    private = W / "backdrop-revision2"
    private.mkdir(exist_ok=True)
    for f in old:
        os.rename(SCENE / f, private / Path(f).name)
    result = dict(
        size=[ww, hh],
        nativeObservedTexels=int(np.sum(st, axis=0)[0]),
        freeSpaceTexelsCleared=int(np.sum(st, axis=0)[1]),
    )
    dump(W / "revision3.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    enhance()
