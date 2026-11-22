"""Fit gently varying inter-lens white balance; measure a sky-only, seam-wrapped fallback."""

from .pack import *
from scipy.ndimage import gaussian_filter


def fit(bank=None, skyfile=None):
    bank = bank or STAGE / "ibr.json"
    skyfile = skyfile or STAGE / "sky.jpg"
    doc = json.load(open(bank))
    xx = []
    yy = []
    color = []
    for c, cs in doc["cameras"].items():
        ray = np.fromfile(STAGE / cs["rays"], "<f4").reshape(H, B, 3)[::8, ::8]
        es = cs["keyframes"]
        ts = []
        sg = []
        for e in es[::14]:
            img = cv.imread(str(STAGE / e["rgb"]))[:, :, ::-1][1::16, 1::16].astype("f")
            mask = cv.imread(str(STAGE / e["mask"]), -1)[::8, ::8]
            m = mask == 3
            r = mask == 2
            sky = np.median(img[m], 0) if m.sum() > 20 else None
            road = np.median(img[r], 0) if r.sum() > 20 else None
            if sky is None and road is None:
                continue
            neutral = lambda x: np.median(x) / np.maximum(15, x)
            g = (neutral(sky) if sky is not None else neutral(road)) * 0.63 + (
                neutral(road) if road is not None else neutral(sky)
            ) * 0.37
            g = np.clip(g, 0.64, 1.55)
            exp = (
                np.clip(179 / max(40, np.mean(sky * g)), 0.75, 1.36)
                if sky is not None
                else 1.0
            )
            sg.append(g * exp)
            ts.append(e["t"])
            if m.sum() < 20:
                continue
            R = Rotation.from_quat(e["rowPoses"][0][3:]).as_matrix()
            dirs = ray[m] @ R.T
            xx.append(
                ((0.5 - np.arctan2(dirs[:, 1], dirs[:, 0]) / (2 * np.pi)) * 128).astype(
                    int
                )
                % 128
            )
            yy.append(
                np.clip(
                    ((0.5 - np.arcsin(np.clip(dirs[:, 2], -1, 1)) / np.pi) * 64).astype(
                        int
                    ),
                    0,
                    63,
                )
            )
            color.append(img[m] * g * exp)
        sg = np.array(sg)
        ts = np.array(ts)
        for e in es:
            w = np.exp(-0.5 * ((ts - e["t"]) / 4) ** 2)
            g = (sg * w[:, None]).sum(0) / max(1e-8, w.sum())
            e["displayGain"] = g.tolist()
    x, y, rgb = np.concatenate(xx), np.concatenate(yy), np.concatenate(color)
    lum = rgb @ np.array([0.2126, 0.7152, 0.0722])
    idx = y * 128 + x
    ww = np.bincount(idx, minlength=8192).reshape(64, 128)
    sm = lambda a: gaussian_filter(a, (1.7, 3), mode=("nearest", "wrap"))
    val = sm(np.bincount(idx, weights=lum, minlength=8192).reshape(64, 128))
    den = sm(ww)
    l = (val + 6 * 179) / (den + 6)
    l[32:] = l[31]
    rgb = np.stack([l * 1.012, l, l * 0.993], -1)
    cv.imwrite(
        str(skyfile),
        cv.resize(
            np.uint8(np.clip(rgb[:, :, ::-1], 0, 255)),
            (1024, 512),
            interpolation=cv.INTER_CUBIC,
        ),
    )
    dump(bank, doc)


if __name__ == "__main__":
    fit()
