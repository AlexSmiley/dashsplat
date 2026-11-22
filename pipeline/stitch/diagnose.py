"""Geometry diagnostics on the four orbit sites and independent semantic observations."""

from .evaluate import *


def run():
    result = {}
    es = held()
    pairs = []
    for t in TIMES:
        for c in CAMS:
            ee = [e for e in es if e["c"] == c and e["clean"]]
            pairs.append(min(ee, key=lambda e: abs(e["t"] - t)))
    for st, src in [
        ("before", RS / "scene/stitched-v3"),
        ("after", RS / "scene/stitched-opt-v1"),
    ]:
        s = Render(src)
        r = []
        sky = []
        road = []
        for t in TIMES:
            a = s.orbit(t, 0)
            h, w = a.shape[:2]
            q = a[h // 8 : 3 * h // 4, w // 4 : 3 * w // 4]
            dep = q[:, :, 4] / np.maximum(0.001, q[:, :, 3])
            r.append(
                dict(
                    t=t,
                    near_object_fraction=float(((q[:, :, 5] > 0.5) & (dep < 6)).mean()),
                    object_fraction=float((q[:, :, 5] > 0.5).mean()),
                )
            )
        for e in pairs:
            a = s.real(e)
            c = e["c"]
            n = e["n"]
            out = []
            for k in ["sky", "road", "dyn"]:
                m = cv.imread(
                    str(RS / "masks" / k / c / ("%06d.png" % n)), cv.IMREAD_UNCHANGED
                )
                out.append(
                    cv.resize(
                        (m != 0).astype("B"), DIMS, interpolation=cv.INTER_NEAREST
                    )
                )
            sk, ro, dy = out
            ok = body(c) & (dy == 0)
            sk = ok & cv.erode(sk, np.ones((5, 5), "B")).astype(bool)
            ro = ok & cv.erode(ro, np.ones((5, 5), "B")).astype(bool)
            sky.extend(a[:, :, 3][sk].tolist())
            road.extend(a[:, :, 5][ro].tolist())
        result[st] = dict(
            chase=r,
            sky_opacity=float(np.mean(sky)),
            road_object_opacity=float(np.mean(road)),
            physical_frames=len(pairs),
        )
        s.local.cache_clear()
    # Displacement of sampled final ground centres from the corresponding initial grid.
    # Nearest matching is unambiguous well below the 5.68 cm minimum grid pitch.
    from scipy.spatial import cKDTree

    base = RS / "scene/stitched-v3"
    opt = RS / "scene/stitched-opt-v1"
    disp = []
    along = []
    ratio = []
    for tile in J(opt / "map.json")["tiles"]:
        if tile["layer"] != "ground":
            continue
        a = reader(base / tile["file"])
        b = reader(opt / tile["file"])[::100]
        d, ix = cKDTree(a[:, :3]).query(b[:, :3])
        x = b[:, :3] - a[ix, :3]
        nn = a[ix, 3:6]
        parallel = (x * nn).sum(1)
        along.extend(np.linalg.norm(x - parallel[:, None] * nn, axis=1))
        disp.extend(d)
        ratio.extend(np.max(np.exp(b[:, 10:12] - a[ix, 10:12]), axis=1))
    result["ground"] = dict(
        n=len(disp),
        displacement_mm=np.percentile(disp, [50, 90, 100]).tolist(),
        tangent_mm=np.percentile(along, [50, 90, 100]).tolist(),
        max_axis_scale_ratio_p90=float(np.percentile(ratio, 90)),
    )
    for k in ["displacement_mm", "tangent_mm"]:
        result["ground"][k] = [1000 * x for x in result["ground"][k]]
    # Compare actual imported covariances/DC through independent renderers, at identical chase poses.
    from .model import Map, torch

    cpu = Render(opt)
    metal = Map(opt, aa=True)
    diff = []
    for t in TIMES:
        ego = traj.T_world_ego(t)
        ang = np.arctan2(ego[1, 0], ego[0, 0])
        f = W * 0.8660254
        head = np.array([np.cos(ang), np.sin(ang), 0])
        up = np.array([0, 0, 1])
        fw = head * np.cos(0.215) - up * np.sin(0.215)
        right = np.cross(head, up)
        T = np.eye(4)
        T[:3, :3] = np.stack([right, np.cross(fw, right), fw], 1)
        T[:3, 3] = ego[:3, 3] - 6 * head + 3 * up
        with torch.no_grad():
            pred = metal.packet(T, [f, f, W / 2, H / 2], W, H)[-1].image.cpu().numpy()
        ref = cpu.orbit(t, 0, size=(W, H))
        ok = ref[:, :, 3] > 0.25
        d = np.abs(pred[ok] * 255 - ref[:, :, :3][ok])
        diff.append(
            dict(
                t=t,
                mean_rgb_levels=float(d.mean()),
                p90_rgb_levels=float(np.percentile(d, 90)),
            )
        )
    result["render_agreement"] = diff
    savej(REPORT / "geometry.json", result)
    print(result)


if __name__ == "__main__":
    run()
