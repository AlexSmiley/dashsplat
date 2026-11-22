"""Ground-contact inventory and fragment review, without relying on motion labels."""

import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from pipeline.common.rig import load_rig, Trajectory, frame_table

W = Path("work/movers")
W.mkdir(parents=True, exist_ok=True)
WIN = [(80, 140), (190, 215), (286, 330)]
SIZE = {
    "car": (4.65, 1.88, 1.6),
    "bus": (11.7, 2.55, 3.20),
    "truck": (7.1, 2.40, 3.0),
    "person": (0.48, 0.55, 1.70),
    "rider": (1.7, 0.65, 1.65),
    "motorcycle": (2, 0.75, 1.4),
    "bicycle": (1.75, 0.6, 1.45),
}


def lift(cam, tr, t, f):
    uv = f * np.array([cam.width / 724, cam.height / 469])
    rt = cam.row_time(t, uv[:, 1])
    E = tr.T_world_ego(rt)
    C = E @ cam.T_ego_cam
    ne, h = tr.ground_plane(rt)
    nw = (E[:, :3, :3] @ ne[:, :, None])[:, :, 0]
    ray = (C[:, :3, :3] @ cam.unproject(uv)[:, :, None])[:, :, 0]
    d = -((nw * (C[:, :3, 3] - E[:, :3, 3])).sum(1) + h) / np.minimum(
        -1e-5, (nw * ray).sum(1)
    )
    G = C[:, :3, 3] + ray * d[:, None]
    return G, d, C, nw, E, h


def inventory(root=Path("."), name=""):
    cs = load_rig()
    tr = Trajectory("poses/ego.npz")
    obs = []
    frags = []
    for c, cam in cs.items():
        tab = frame_table(cam)
        rows = []
        for p in sorted((root / "masks/dyn" / c).glob("[0-9]*.json")):
            n = int(p.stem)
            t = tab[n]["t"]
            if not any(a <= t <= b for a, b in WIN):
                continue
            for v in json.loads(p.read_text()):
                if v["class"] not in SIZE or v["area"] < 36:
                    continue
                rows.append([t, n, v["id"], v["class"], v["area"], *v["bbox"]])
        if not rows:
            continue
        t = np.array([x[0] for x in rows])
        B = np.array([x[5:] for x in rows])
        f = np.c_[(B[:, 0] + B[:, 2] + 1) * 0.5, B[:, 3] + 0.5]
        G, d, *_ = lift(cam, tr, t, f)
        dic = defaultdict(list)
        for x, g, r in zip(rows, G, d):
            if 1 < r < 140:
                v = [c, *x, *g.round(3).tolist(), round(r, 3)]
                dic[x[2]].append(v)
                obs.append(v)
        for i, rr in dic.items():
            arr = np.array([[x[1], *x[10:12], x[13], x[5]] for x in rr])
            t = arr[:, 0]
            dt = t[-1] - t[0]
            if len(t) < 3 or dt < 0.12:
                continue
            v = np.polyfit(t - t.mean(), arr[:, 1:3], 1)
            dev = np.linalg.norm(
                arr[:, 1:3] - ((t - t.mean())[:, None] * v[0] + v[1]), axis=1
            )
            frags.append(
                dict(
                    c=c,
                    id=i,
                    cls=rr[np.argmax(arr[:, -1])][4],
                    t0=round(t[0], 2),
                    t1=round(t[-1], 2),
                    n=len(t),
                    area=int(arr[:, -1].max()),
                    xy=v[1].round(1).tolist(),
                    vel=v[0].round(2).tolist(),
                    speed=round(np.linalg.norm(v[0]), 1),
                    mad=round(np.median(dev), 2),
                    ran=round(np.median(arr[:, -2]), 1),
                )
            )
    json.dump(obs, open(W / (name + "inventory.json"), "w"))
    json.dump(frags, open(W / (name + "fragments.json"), "w"), indent=1)


if __name__ == "__main__":
    inventory()
    if (W / "extra/masks/dyn").exists():
        inventory(W / "extra", "extra-")
