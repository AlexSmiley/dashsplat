"""Traffic tracking evidence in capture coordinates, with the stored transients fully z-composited."""

from .v4_proof import *


def inspect():
    _, s = variants()
    p = json.loads((V4 / "slice-plan.json").read_text())
    side = json.loads((V4 / "instances.json").read_text())["frames"]
    info = {}
    for name, c, nid, at in [
        ("crossing-car", "front", 46, 102.5),
        ("passing-bus", "left_pillar", 202, 131.5),
    ]:
        cols = []
        metrics = []
        last = None
        es = sorted(
            sorted(p[c], key=lambda e: abs(e["t"] - at))[:4], key=lambda e: e["t"]
        )
        for e in es:
            t = e["t"]
            n = e["n"]
            ids = cv.imread(
                str(RS / "masks/dyn" / c / f"{n:06d}.png"), cv.IMREAD_UNCHANGED
            )
            row = next(d for d in side if d["cam"] == c and d["frame"] == n)
            target = next((m for m in row["instances"] if m["id"] == nid), None)
            if not target:
                continue
            pt = s.points(t, traj("live").T_world_ego(t)[:3, 3], tag_transient=True)
            surf = s.image(t, c, size=(B, H), pts=pt)
            im = rgb(surf)
            ref = cv.resize(
                cv.imread(str(RS / "data/drive/frames" / c / f"{n:06d}.jpg"))[
                    :, :, ::-1
                ],
                (B, H),
            )
            box = target["bbox"]
            x, y, X, Y = box
            radius = max(15, int(0.4 * (X - x)))
            u = max(0, x - radius)
            U = min(B, X + radius)
            v = max(0, y - radius)
            V = min(H, Y + radius)
            crop = np.s_[v:V, u:U]
            alpha = surf[:, :, 5]
            drawn = alpha > 0.35
            mask = ids == nid
            anymask = cv.dilate((ids > 0).astype("B"), np.ones((5, 5), "B")) > 0
            trail = int((drawn & ~anymask)[crop].sum())
            count = int(drawn[crop].sum())
            delta = None
            if last is not None and target["centroid_world"] is not None:
                delta = float(np.linalg.norm(np.array(target["centroid_world"]) - last))
            last = target["centroid_world"]
            metrics.append(
                dict(
                    frame=n,
                    t=t,
                    tagged_fraction=float(drawn[mask].mean()),
                    fraction_outside_any_dilated_instance=trail / max(1, count),
                    centroid_step_m=delta,
                )
            )
            overlay = im.copy()
            overlay[drawn & ~anymask] = np.uint8(
                0.4 * overlay[drawn & ~anymask] + 0.6 * np.array([246, 65, 51])
            )
            overlay[mask & drawn] = np.uint8(
                0.4 * overlay[mask & drawn] + 0.6 * np.array([61, 208, 110])
            )
            imgs = []
            for img, caption in [
                (ref, "photograph"),
                (im, "v4 at capture time"),
                (overlay, "green match | red spill"),
            ]:
                imgs.append(
                    label(cv.resize(img[crop], (250, 160)), caption + f" {t:.2f}s")
                )
            cols.append(np.vstack(imgs))
        assert len(cols) >= 2
        save(OUT / f"traffic-handover-{name}.jpg", np.hstack(cols))
        info[name] = metrics
    (V4 / "slice-traffic.json").write_text(json.dumps(info, indent=2))
    print(info, flush=True)


if __name__ == "__main__":
    inspect()
