"""Same-format content revision: borrow actual measured far pixels across nearby viewpoints."""

from .v3_pano import *
from .v3_ground import RoadMask, CELL
from .v3_publish import verify
from scipy.ndimage import gaussian_filter
import shutil


def runsupported(top, ok, end):
    ii = np.flatnonzero(ok)
    u = np.arange(len(ok))
    w = len(ok)
    base = end - 5.5
    if not len(ii):
        return np.full(w, base)
    xp = np.r_[ii - w, ii, ii + w]
    fp = np.tile(top[ii], 3)
    y = np.interp(u, xp, fp)
    j = np.searchsorted(xp, u)
    gap = np.minimum(
        abs(u - xp[np.maximum(0, j - 1)]), abs(u - xp[j.clip(0, len(xp) - 1)])
    )
    coef = np.exp(-((gap / 27.0) ** 2))
    return base + (y - base) * coef


def improve():
    j = json.load(open(SCENE / "ibr.json"))
    assert j["contentRevision"] == 1
    ps = j["panoramas"]
    tmp = W / "content-2"
    tmp.mkdir(exist_ok=True)

    @lru_cache(50)
    def bank(i):
        p = ps[i]
        im = cv.imread(str(SCENE / p["rgb"])).astype("f")
        cf = cv.imread(str(W / "confidence" / f"{i:04}.png"), 0).astype("f") / 255
        sk = cv.imread(str(SCENE / p["sky"])).astype("f")
        return im, cf, sk

    t = np.array([p["t"] for p in ps])
    orig = np.array([p["origin"] for p in ps])
    rad = np.array([p["radius"] for p in ps])
    stats = []

    def one(i):
        p = ps[i]
        d = np.linalg.norm(orig[:, :2] - orig[i, :2], axis=1)
        near = np.flatnonzero(
            (d < 49) & (abs(t - t[i]) < 16)
        )  # Sample along both sides of the knot, bounded in world distance; never sample a rendered chase.
        ordered = sorted(near, key=lambda k: abs(t[k] - t[i]))
        keep = [i]
        for k in ordered:
            if (
                min(
                    abs(t[k] - t[l]) + np.linalg.norm(orig[k, :2] - orig[l, :2]) * 0.2
                    for l in keep
                )
                > 1.3
            ):
                keep.append(k)
            if len(keep) >= 10:
                break
        im, m, sk = bank(i)
        m = m.copy()
        v = im * m[:, :, None]
        w = m.copy()
        world = orig[i] + rad[i] * BA
        sw = 1.0
        skyL = cv.GaussianBlur(sk, (0, 0), 7)
        skySum = skyL.copy()
        for k in keep[1:]:
            vrel = world - orig[k]
            el = np.rad2deg(
                np.arctan2(vrel[:, :, 2], np.linalg.norm(vrel[:, :, :2], axis=2))
            )
            uv = np.stack(
                [
                    (0.5 + np.arctan2(vrel[:, :, 1], vrel[:, :, 0]) / (2 * np.pi)) * BW
                    - 0.5,
                    (BACK_EL[1] - el) / (BACK_EL[1] - BACK_EL[0]) * BH - 0.5,
                ],
                2,
            ).astype("f")
            q, conf, skn = bank(k)
            weight = np.exp(-d[k] / 23 - abs(t[k] - t[i]) / 7)
            mask = (
                cv.remap(conf, uv, None, cv.INTER_LINEAR, borderMode=cv.BORDER_WRAP)
                * (el > BACK_EL[0])
                * (el < BACK_EL[1])
            )
            ww = mask * weight * (1 - np.clip(m, 0, 1) * 0.95)
            v += (
                cv.remap(q, uv, None, cv.INTER_LINEAR, borderMode=cv.BORDER_WRAP)
                * ww[:, :, None]
            )
            w += ww
            coef = np.exp(-abs(t[k] - t[i]) / 0.85) * 0.5
            skySum += coef * cv.GaussianBlur(skn, (0, 0), 7)
            sw += coef
        sk = np.clip(sk + (skySum / sw - skyL) * 0.80, 0, 255)
        seen = w > 0.11
        colour = v / np.maximum(0.001, w[:, :, None])
        origH = 1.55
        end = (
            (BACK_EL[1] - np.rad2deg(np.arctan2(-origH - 0.40, rad[i])))
            / (BACK_EL[1] - BACK_EL[0])
            * BH
        )
        rows = np.arange(BH)[:, None]
        band = (rows < end + 3) & (rows > end - 40)
        area = cv.boxFilter(seen.astype("f"), -1, (11, 7))
        core = (area > 0.48) & seen
        cont = (core & band).sum(0) >= 5
        top = np.argmax(core, axis=0)
        skyline = runsupported(top, cont, end)
        skyline = np.minimum(
            end - 2,
            gaussian_filter1d(
                median_filter(skyline, size=13, mode="wrap"), 3.0, mode="wrap"
            ),
        )  # Only observed isolated fine structure may rise above the contiguous horizon.
        alpha = np.clip((rows - skyline + 1) / 3.0, 0, 1)
        nearcol = np.max(cv.boxFilter(seen.astype("f"), -1, (9, 5)) * band, axis=0)
        amp = 0.42 + 0.58 * np.minimum(1, nearcol * 2)
        alpha *= amp[None]
        alpha = np.maximum(alpha, np.minimum(1.0, w * 1.6))
        alpha *= np.clip((end + 6 - rows) / 8.0, 0, 1)
        bg = smear(
            v, w
        )  # Fill below the observed far surface with gentle local colours, never extrude the whole last row into a tall facade.
        hazecol = (
            np.median(colour[seen][::47], axis=0)
            if seen.any()
            else np.array([95, 105, 98])
        )
        den = cv.GaussianBlur(w, (0, 0), 6)
        fallback = np.clip(1 - den / 0.32, 0, 1)[:, :, None]
        bg += 0.44 * fallback * (hazecol - bg)
        skyfile = tmp / Path(p["sky"]).name
        rgbfile = tmp / Path(p["rgb"]).name
        mattefile = tmp / Path(p["matte"]).name
        for path, img in [(skyfile, sk), (rgbfile, bg), (mattefile, alpha * 255)]:
            assert cv.imwrite(
                str(path),
                np.uint8(np.clip(img + 0.5, 0, 255)),
                [cv.IMWRITE_JPEG_QUALITY, 89]
                if path.suffix == ".jpg"
                else [cv.IMWRITE_PNG_COMPRESSION, 4],
            )
        return [int((m > 0.11).sum()), int(seen.sum()), len(keep)]

    with ThreadPoolExecutor(4) as pool:
        stats = list(pool.map(one, range(len(ps))))
    # Immutable revision-specific pano path, atomic manifest change. Keep the original assets in private work afterwards.
    dest = SCENE / "pano-r2"
    assert not dest.exists()
    os.rename(tmp, dest)
    for p in ps:
        for key in ["sky", "rgb", "matte"]:
            p[key] = "pano-r2/" + Path(p[key]).name
    j["contentRevision"] = 2
    backup = W / "ibr-revision1.json"
    shutil.copy2(SCENE / "ibr.json", backup)
    dump(SCENE / "ibr.json", j)
    verify(SCENE)
    os.rename(SCENE / "pano", W / "pano-original")
    s = np.array(stats)
    result = dict(
        beforeMeasuredPixels=int(s[:, 0].sum()),
        afterMeasuredPixels=int(s[:, 1].sum()),
        medianNeighbourKnots=float(np.median(s[:, 2])),
        meanGain=float(s[:, 1].sum() / s[:, 0].sum()),
    )
    dump(W / "refinement.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    improve()
