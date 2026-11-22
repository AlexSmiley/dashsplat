"""Stand-in only: time gsplat antialiased forward, color backward and Adam on real tiles/camera/target."""

import json, os, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from plyfile import PlyData
from gsplat import rasterization

root = Path(os.environ["RS"]) / "bench"
m = json.loads((root / "frames.json").read_text())
f = m["frames"][0]
v = np.concatenate(
    [PlyData.read(p)["vertex"].data for p in sorted((root / "tiles").glob("*.ply"))]
)


def t(x):
    return torch.as_tensor(np.array(x), device="cuda", dtype=torch.float32).contiguous()


def cols(names):
    return t(np.stack([v[c] for c in names], 1))


xyz = cols(["x", "y", "z"])
quat = cols(["rot_" + str(i) for i in range(4)])
scale = cols(["scale_" + str(i) for i in range(3)]).exp()
opa = t(v["opacity"]).sigmoid()
color = (0.5 + 0.282095 * cols(["f_dc_" + str(i) for i in range(3)])).requires_grad_()
adam = torch.optim.Adam([color], lr=1e-4)
view = t(np.linalg.inv(f["T_world_cam"]))[None]
K = t(m["K"])[None]
K[:, :2, :] /= 2
target = (
    t(
        np.asarray(
            Image.open(root / "render" / f"{f['n']:04}.png")
            .convert("RGB")
            .resize((736, 480))
        )
    )
    / 255
)
for i in range(55):
    if i == 5:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
    adam.zero_grad()
    rgb, alpha, _ = rasterization(
        xyz, quat, scale, opa, color, view, K, 736, 480, rasterize_mode="antialiased"
    )
    (rgb[0] - target).abs().mean().backward()
    adam.step()
torch.cuda.synchronize()
result = dict(
    iter_s=50 / (time.perf_counter() - start),
    splats=len(v),
    crop=[736, 480],
    backend="gsplat, color-only Adam surrogate",
    allocated_GB=torch.cuda.max_memory_allocated() / 1e9,
)
print(result)
(root / "results/splat.json").write_text(json.dumps(result))
