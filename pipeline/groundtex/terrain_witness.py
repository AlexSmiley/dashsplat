"""Show why accepting a flat-looking depth alone projected hedge/wall colour, and what pixel types own those cells now."""

from .outer import *
from .terrain_select import goodmask
from .ring_export import ctrl, rem
from .proof import RPT, label, save


def main():
    rows = read(W / "outer-rows.json")
    panels = []
    checks = []
    for cam, n in [
        ("left_repeater", 4251),
        ("left_repeater", 4316),
        ("right_pillar", 4247),
        ("right_repeater", 4317),
        ("right_pillar", 4596),
        ("front", 10695),
    ]:
        id, r = next(
            (i, r) for i, r in enumerate(rows, 1) if r["n"] == n and r["cam"] == cam
        )
        box = footprint6(r)
        xy = boxxy(box)
        z = surface().take(box)[:, :, 0]
        a, b, bid = (
            np.zeros(z.shape, bool),
            np.zeros(z.shape, bool),
            np.zeros(z.shape, "i"),
        )
        for k, s, v in cells(box):
            for directory, view in [("map", a), ("terrain-map", b)]:
                f = W / directory / f"{k[0]}_{k[1]}.npz"
                if f.exists():
                    data = np.load(f)["id"][s]
                    view[v] = data == id
                    if directory == "terrain-map":
                        bid[v] = data
        origin, delta, _ = ctrl(id)
        xy += rem(delta, (xy - origin) / (4 * L))
        uv, _, _, _ = project(r, xy, z)
        ref = image(cam, n, True)[:, :, ::-1].copy()
        h, w = ref.shape[:2]
        c = inputs()[1][cam]
        good, classes = goodmask(id)
        cl = sam(classes, uv, c, True)
        was = a & ~np.isin(cl, [0, 1, 9])
        kept = b & np.isin(cl, [0, 1, 9])
        for mask, col in [(was, [255, 96, 63]), (kept, [58, 241, 145])]:
            ij = (uv[mask] * [w / c.width, h / c.height]).astype(int)
            ref[ij[:, 1].clip(0, h - 1), ij[:, 0].clip(0, w - 1)] = col
        panels.append(
            label(
                cv.resize(ref, (480, 310)), f"{cam} {n} | rejected red / eligible green"
            )
        )
        checks.append(
            dict(
                cam=cam,
                n=n,
                depth_only_non_ground=int(was.sum()),
                reassigned_to_other_view=int((was & (bid > 0) & (bid != id)).sum()),
                removed_from_this_view=int((was & ~b).sum()),
            )
        )
    save(
        "native-witness.jpg",
        np.vstack([np.hstack(panels[i : i + 2]) for i in [0, 2, 4]]),
    )
    dump(RPT / "witness.json", checks)


if __name__ == "__main__":
    main()
