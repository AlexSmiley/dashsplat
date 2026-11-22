"""Native holdouts and chase-track proofs, separately from the four shipping-viewer screenshots."""

from .v3evidence import W as OUT, BASE
from . import v2eval as ve
from .v2eval import *


def run():
    dst = OUT / "proof"
    dst.mkdir(exist_ok=True)
    ve.RESULT = dst
    ve.TIMES = [93, 100, 101, 112, 128]
    shutil.copyfile(OUT / "viewer-cameras.json", dst / "viewer-cameras.json")
    ve.build()
    root = RS / "scene/stitched-opt-v3"
    r = ve.Wide(root)
    r.sky_path = root / "sky.jpg"
    r.pan = readjpg(r.sky_path)
    for t in ve.TIMES:
        for name, k in [
            ("default", 0),
            ("left", 80),
            ("right", 16),
            ("raised", "raised"),
        ]:
            a = r.viewpoint(t, k)
            im = r.display(t, k, a)
            jpg(dst / f"views/{t}/{name}.jpg", im, 92)
            if t == 112 and k == 0:
                g = np.load(RS / "work/stitch/report/v2/ground/road112.npz")
                Y, X = np.mgrid[:900, :1440]
                roi = (Y > 560) & (X > 270) & (X < 1120) & (g["alpha"] > 0.95)
                box = (slice(555, 735), slice(980, 1340))
                ref = np.load(RS / "work/stitch/report/v2/v2/road112.npz")
                savej(
                    dst / "occupancy112.json",
                    dict(
                        v2_apron_fraction=float((ref["alpha"][box] < 0.5).mean()),
                        v3_apron_fraction=float((a[:, :, 3][box] < 0.5).mean()),
                        original_road_rgb_mae=float(
                            abs(g["rgb"] - a[:, :, :3])[roi].mean()
                        ),
                        streetlight_gap_over_15pct=float(
                            (a[85:170, 490:840, 5] > 0.15).mean()
                        ),
                    ),
                )

    def physical(e):
        ref, valid = reference(e)
        a = r.real(e)
        jpg(
            dst / "after/camera" / e["c"] / ("%06d.jpg" % e["n"]),
            np.clip(a[:, :, :3], 0, 255).astype("B"),
        )
        return dict(**e, **scores(ref, a, valid))

    with ThreadPoolExecutor(5) as ex:
        rows = list(ex.map(physical, held()))
        savej(dst / "native.json", rows)
    if not (dst / "before").exists():
        (dst / "before").symlink_to(
            RS / "work/stitch/report/v2/v2", target_is_directory=True
        )
    perceptual.REPORT = dst
    perceptual.run()
    verify.WORK = dst
    verify.run(root)
    source = J(BASE / "map.json")
    current = J(root / "map.json")
    old = [i for i in source["tiles"] if i["layer"] == "ground"]
    assert all(
        i in current["tiles"] and (BASE / i["file"]).samefile(root / i["file"])
        for i in old
    )
    assert (root / "sky.jpg").read_bytes() == (
        RS / "scene/live-v5/sky.jpg"
    ).read_bytes()
    print("identity ok", len(old), flush=True)


if __name__ == "__main__":
    run()
