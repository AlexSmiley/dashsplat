"""Small independent later-window packets; never mutate the shared producer schedule/state."""

from pathlib import Path
import json, numpy as np
from .inventory import W
from pipeline.common.rig import load_rig, frame_table


def run():
    root = Path.cwd()
    out = W / "extra"
    (out / "depth").mkdir(parents=True, exist_ok=True)
    for d in ("data", "calib", "poses", "third_party"):
        if not (out / d).exists():
            (out / d).symlink_to(root / d, target_is_directory=True)
    k = json.load(open("depth/keyframes.json"))
    result = {}
    for name, cam in load_rig().items():
        tab = frame_table(cam)
        ts = np.array([r["t"] for r in tab])
        rows = []
        for t in np.arange(190, 215, 0.25):
            n = int(np.argmin(abs(ts - t)))
            rows.append(dict(n=n, t=ts[n], t_key=t, kind="time"))
        later = k["frames"][name]
        rows.extend(
            x for i, x in enumerate(later) if 286 <= x["t"] <= 330 and i % 15 == 0
        )
        result[name] = rows
    (out / "depth/keyframes.json").write_text(
        json.dumps({"version": 1, "frames": result})
    )


if __name__ == "__main__":
    run()
