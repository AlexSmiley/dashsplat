"""Construct a real Nerfstudio fixture for the stock msplat Python trainer."""

from .views import *
from pipeline.road.scene import reader


def run():
    import msplat

    base = WORK / "ms-bake"
    base.mkdir(exist_ok=True)
    frames = []
    es = [
        e
        for e in J(CACHE / "manifest.json")
        if e["c"] == "front" and not e["hold"] and 92 <= e["t"] < 95
    ]
    c = "front"
    K = design(c, 0)[0]
    for e in es:
        fn = "%06d.png" % e["n"]
        im = np.load(path(e))["rgb"]
        cv.imwrite(str(base / fn), im[:, :, ::-1])
        T = world(c, 0, e["t"], 3)
        T[:3, 1:3] *= -1
        frames.append(dict(file_path=fn, transform_matrix=T.tolist()))
    source = RS / "scene/road-stitched-v2"
    p = traj.T_world_ego(93)[:3, 3]
    chunks = []
    for tile in J(source / "map.json")["tiles"]:
        bb = np.array(tile["bbox"])
        d = np.maximum(np.maximum(bb[:2] - p[:2], p[:2] - bb[3:5]), 0)
        if np.linalg.norm(d) < 30:
            a = reader(source / tile["file"])[::8]
            chunks.append(a[np.linalg.norm(a[:, :2] - p[:2], axis=1) < 30])
    a = np.concatenate(chunks)
    typ = np.dtype([("pos", "<f4", 3), ("rgb", "u1", 3)])
    out = np.empty(len(a), typ)
    out["pos"] = a[:, :3]
    out["rgb"] = np.clip((0.5 + 0.28209479 * a[:, 6:9]) * 255, 0, 255)
    with open(base / "points3D.ply", "wb") as f:
        f.write(
            (
                "ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % len(a)
                + "property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
            ).encode()
        )
        out.tofile(f)
    savej(
        base / "transforms.json",
        dict(
            w=W,
            h=H,
            fl_x=K[0, 0],
            fl_y=K[1, 1],
            cx=W / 2,
            cy=H / 2,
            ply_file_path="points3D.ply",
            frames=frames,
        ),
    )
    d = msplat.Dataset(str(base))
    cfg = msplat.TrainingConfig(
        iterations=301,
        sh_degree=0,
        num_downscales=0,
        keep_crs=True,
        refine_every=100,
        bg_color=[22 / 255, 26 / 255, 30 / 255],
    )
    g = msplat.GaussianTrainer(d, cfg)
    g.step()
    msplat.sync()
    tic = time.monotonic()
    for i in range(300):
        g.step()
        if i % 30 == 29:
            float(g.render(i % len(es)).mean())
    msplat.sync()
    dt = time.monotonic() - tic
    pred = g.render(0)
    cv.imwrite(
        str(base / "render.png"), np.clip(pred[:, :, ::-1] * 255, 0, 255).astype("B")
    )
    # Same 41,691 records, eight full images and pose approximation. This is an import/API
    # and rendering benchmark, not transferable held-out reconstruction quality.
    from .model import gpu, Camera, render, SimpleNamespace, BG, torch

    cams = []
    refs = []
    masks = []
    for e in es:
        T = world(c, 0, e["t"], 3)
        R = T[:3, :3].T
        cams.append(
            Camera(gpu(R), gpu(-R @ T[:3, 3]), K[0, 0], K[1, 1], W / 2, H / 2, W, H)
        )
        z = np.load(path(e))
        refs.append(gpu(z["rgb"] / 255))
        masks.append(gpu(z["mask"] == 1))
    dc = gpu(a[:, 6:9]).requires_grad_(True)
    alpha = gpu(a[:, 9]).requires_grad_(True)
    means = gpu(a[:, :3])
    scales = gpu(np.exp(a[:, 10:13]))
    quats = gpu(a[:, 13:17])
    bg = gpu(BG)
    opt = torch.optim.Adam([dc, alpha], lr=0.012)

    def image(j):
        model = SimpleNamespace(
            means=means,
            scales=scales,
            quats=quats,
            sh_degree=0,
            colors=(0.5 + 0.28209479 * dc).clamp(0, 1),
            opacities=alpha.sigmoid(),
        )
        return render(model, cams[j], background=bg)

    def st(j):
        im = image(j)
        loss = ((im - refs[j]).abs().mean(2) * masks[j]).sum() / masks[j].sum()
        opt.zero_grad()
        loss.backward()
        opt.step()

    st(0)
    torch.mps.synchronize()
    t0 = time.monotonic()
    for j in range(300):
        st(j % len(es))
        if j % 30 == 29:
            with torch.no_grad():
                float(image(j % len(es)).mean().item())
    torch.mps.synchronize()
    metal_dt = time.monotonic() - t0
    with torch.no_grad():
        ss = {
            name: float(
                np.mean(
                    [
                        -10
                        * np.log10(
                            float(
                                (((im(j) - refs[j]) ** 2).mean(2) * masks[j])
                                .sum()
                                .item()
                                / masks[j].sum().item()
                            )
                        )
                        for j in range(len(es))
                    ]
                )
            )
            for name, im in [
                ("stock_msplat", lambda j: gpu(g.render(j))),
                ("masked_metalsplat", image),
            ]
        }
    result = dict(
        matched_metalsplat_seconds=metal_dt,
        matched_metalsplat_it_s=300 / metal_dt,
        in_sample_masked_psnr=ss,
        seed_points=len(a),
        splats=g.splat_count,
        frames=len(frames),
        seconds=dt,
        it_s=300 / dt,
        steps=300,
    )
    savej(WORK / "ms.json", result)
    print(result)


if __name__ == "__main__":
    run()
