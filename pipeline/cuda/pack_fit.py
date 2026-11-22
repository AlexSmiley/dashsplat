"""Run on the Studio after preparing the v5 native seed. Only training exposures/teachers leave it."""

import io, json, os, sys, tarfile
from pathlib import Path
import numpy as np
from PIL import Image
from pipeline.stitch.v5world import W5, V, RS, physical_schedule, semantic, MAN

root = "/root/rs"
native = W5 / "native"
meta = json.loads((native / "map.json").read_text())
assert meta["slice_s"] == [224.0, 248.0]
with tarfile.open(mode="w|", fileobj=sys.stdout.buffer) as tar:

    def add(path, target):
        tar.add(Path(path).resolve(strict=True), arcname=target, recursive=False)

    def buf(name, data):
        info = tarfile.TarInfo(name)
        info.size = len(data)
        info.mode = 0o600
        tar.addfile(info, io.BytesIO(data))

    def doc(name, data):
        buf(name, json.dumps(data).encode())

    def npz(name, **kw):
        b = io.BytesIO()
        np.savez(b, **kw)
        buf(name, b.getvalue())

    add(native / "map.json", "native/map.json")
    doc(
        "scene/road-stitched-v2/map.json",
        {"tiles": [t for t in meta["tiles"] if t["layer"] == "ground"]},
    )
    for t in meta["tiles"]:
        p = t["file"]
        assert Path(p).parent.as_posix() == "tiles"
        add(
            native / p,
            ("native/" if t["layer"] == "objects" else "scene/road-stitched-v2/") + p,
        )
    # assemble only copies these opaque envelope files. Do not upload the real sky/GPS trace.
    doc("metadata/drive.json", {"slice_s": [224, 248]})
    add(native / "rig.json", "metadata/rig.json")
    b = io.BytesIO()
    Image.new("RGB", (1, 1)).save(b, "JPEG")
    buf("metadata/sky.jpg", b.getvalue())
    add(V / "free-grid.npz", "v2/free-grid.npz")
    ex = physical_schedule()
    cams = list(ex)
    realns = {c: [] for c in cams}
    maskroot = Path(os.environ["STITCH_MASKS"])
    add(RS / "calib/rig.json", "calib/rig.json")
    for c, rows in ex.items():
        add(RS / f"calib/masks/{c}.png", f"calib/masks/{c}.png")
        doc(
            f"depth/{c}/index.json", {}
        )  # no depth is read by the object appearance optimizer
        for e in rows:
            assert 224 <= e["t"] <= 248
            n = e["n"]
            sem = semantic(c, n)
            if e["hold"] or not sem:
                continue
            realns[c].append(n)
            tail = f"{c}/{n:06}.png"
            add(sem, "v5/sem/" + tail)
            add(
                RS / f"data/drive/frames/{c}/{n:06}.jpg",
                f"data/drive/frames/{c}/{n:06}.jpg",
            )
            for k in ("sky", "road", "dyn"):
                add(maskroot / k / tail, "masks/" + k + "/" + tail)
            rem = RS / "maps/movers-v1/remove" / tail
            if rem.is_file():
                add(rem, "maps/movers-v1/remove/" + tail)
        for e in rows:
            e["depth"] = "prepared_on_studio" if e["depth"] else None
    doc("v2/exposures.json", ex)
    traj = np.load(RS / "poses/ego.npz")
    ts = traj["t"]
    inside = (ts > 224) & (ts < 248)
    from pipeline.stitch.views import traj as tv

    knots = np.r_[224.0, ts[inside], 248.0]
    npz("poses/ego.npz", t=knots, T_world_ego=tv.T_world_ego(knots))
    phot = np.load(RS / "photo/stitched-ground-v2.npz")
    ks = set()
    for i, c in enumerate(phot["names"].tolist()):
        for n in realns[c]:
            k = int(
                np.clip(
                    np.searchsorted(phot["ns"][:, i], n, "right") - 1,
                    0,
                    len(phot["ns"]) - 1,
                )
            )
            ks.update([k, min(k + 1, len(phot["ns"]) - 1)])
    ks = sorted(ks)
    npz(
        "photo/stitched-ground-v2.npz",
        **{
            k: phot[k][ks] if k in ("gain", "ns", "epoch") else phot[k]
            for k in ("gain", "ns", "epoch", "shade", "names", "sizes")
        },
    )
    teachers = json.loads(MAN.read_text())
    clean = []
    for e in teachers:
        assert all(
            224 <= e[k] <= 248
            for k in ("t", "base_t", "neighbor_t")
            if e.get(k) is not None
        )
        row = {
            k: e[k]
            for k in ("id", "station", "t", "K", "T_world_cam", "base_distance_m")
        }
        clean.append(row)
        for k in ("rgb", "sky", "weight", "depth"):
            name = f"teacher/{k}/{e['id']}.png"
            row[k] = root + "/" + name
            if e["station"] % 10:
                add(e[k], name)
    doc("teacher/manifest.json", clean)
