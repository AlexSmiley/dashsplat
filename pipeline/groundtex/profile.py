"""Transverse edge spread of the photographed 128 s dash at the native 1 cm atlas pitch."""

from .proof import *


def spread(rgb):
    lum = cv.cvtColor(rgb.astype("B"), cv.COLOR_RGB2GRAY).astype(float)
    widths = []
    contrast = []
    for y in range(30, 260, 2):
        v = np.median(lum[y - 2 : y + 3], axis=0)
        centre = np.argmax(np.convolve(v, np.ones(5) / 5, "same")[65:220]) + 65
        paint = np.median(v[centre - 2 : centre + 3])
        left = np.median(v[centre - 45 : centre - 25])
        right = np.median(v[centre + 25 : centre + 45])
        for sign, base in [(-1, left), (1, right)]:
            if paint - base < 8:
                continue
            signal = (v[centre + sign * np.arange(40)] - base) / (paint - base)
            cross = []
            for threshold in [0.8, 0.2]:
                j = np.flatnonzero(
                    (signal[:-1] >= threshold) & (signal[1:] < threshold)
                )
                if len(j):
                    i = int(j[0])
                    cross.append(
                        i + (signal[i] - threshold) / (signal[i] - signal[i + 1])
                    )
            if len(cross) == 2 and cross[1] > cross[0]:
                widths.append(cross[1] - cross[0])
                contrast.append(paint - base)
    return dict(
        edges=len(widths),
        median_20_80_cm=float(np.median(widths)),
        median_contrast=float(np.median(contrast)),
    )


def main():
    t = 128
    xy, c, yaw = top_xy(t, 11.4, -1.5, 280, 480, 0.01)
    s = Splats().draw(
        t, "top", centre=np.r_[c, 0], span=2.8, size=(280, 480), fixed_yaw=yaw
    )[0]
    im, a = Atlas().rgba(xy)
    assert np.min(a[28:262, 45:235]) > 0.98
    p = next(e for e in read(RPT / "patches.json") if e["time"] == 128)
    r = next(
        e
        for e in read(W / "observations.json")
        if e["n"] == p["source"] and e["cam"] == p["cam"]
    )
    src, _ = groundphoto(r, xy)
    stats = dict(
        splat=spread(s),
        atlas=spread(im),
        photo=spread(src),
        frame=r["n"],
        camera=r["cam"],
    )
    dump(RPT / "dash-profile.json", stats)
    print(stats)


if __name__ == "__main__":
    main()
