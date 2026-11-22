"""Read the contract, not trainer state. Fail closed on count, geometry or chronology errors."""

from .views import *
from pipeline.road.scene import reader


def run(root):
    src = RS / "scene/stitched-v3"
    meta = J(root / "map.json")
    layers = {}
    seen = set()
    files = set()
    size = 0
    assert (root / "rig.json").read_bytes() == (
        RS / "calib/rig.json"
    ).read_bytes() and (root / "drive.json").read_bytes() == (
        src / "drive.json"
    ).read_bytes()
    assert J(root / "drive.json")["duration"] >= END
    for t in meta["tiles"]:
        assert (
            t["id"] not in seen
            and t["file"] not in files
            and t["layer"] in ["objects", "ground"]
            and t["kind"] == "static"
        )
        seen.add(t["id"])
        files.add(t["file"])
        f = root / t["file"]
        a = reader(f)
        size += f.stat().st_size
        assert (
            t["count"] == len(a)
            and np.isfinite(a).all()
            and np.allclose(
                t["bbox"], np.r_[a[:, :3].min(0), a[:, :3].max(0)], atol=0.0001
            )
        )
        assert (
            np.abs(np.linalg.norm(a[:, 13:17], axis=1) - 1).max() < 1e-4
            and np.abs(np.linalg.norm(a[:, 3:6], axis=1) - 1).max() < 1e-4
        )
        assert (np.abs(a[:, 6:9]) <= 0.5 / 0.28209479 + 0.001).all()
        b, d, e0, s1, e1, s2, e2 = a[:, 17:24].T
        assert (
            (b == 0)
            & (d == 1e9)
            & (b <= e0)
            & (e0 <= s1)
            & (s1 <= e1)
            & (e1 <= s2)
            & (s2 <= e2)
            & (e2 <= d)
            & ((s1 == 1e9) == (e1 == 1e9))
            & ((s2 == 1e9) == (e2 == 1e9))
        ).all()
        layers[t["layer"]] = layers.get(t["layer"], 0) + len(a)
    assert set(layers) == {"ground", "objects"}
    tl = np.load(RS / "data/drive/timeline.npz")
    odo = np.r_[0, np.cumsum(np.hypot(np.diff(tl["x"]), np.diff(tl["y"])))]
    dist = np.interp(END, tl["t"], odo) - np.interp(START, tl["t"], odo)
    ground = sum(
        t["count"] for t in J(src / "map.json")["tiles"] if t["layer"] == "ground"
    )
    res = dict(
        tiles=len(seen),
        layers=layers,
        ply_bytes=size,
        travel_m=dist,
        projected_whole_count=int(ground + layers["objects"] * 3300 / dist),
        ground_whole=ground,
    )
    savej(WORK / "verified.json", res)
    print(res)


if __name__ == "__main__":
    import sys

    run(Path(sys.argv[1]))
