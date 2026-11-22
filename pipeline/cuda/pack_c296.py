"""Stream ONLY the 90 selected chase renders, published Studio repairs, and their two reference photographs."""

import io, json, tarfile, sys
from pathlib import Path
import cv2
from pipeline.genfill.views import photo

p = Path("work/genfill/clip/c296")
doc = json.loads((p / "frames.json").read_text())
size = tuple(doc["size"])
frames = doc["frames"][180:270]
assert len(frames) == 90 and doc["fps"] == 30 and size == (1472, 960)
refs = {}
rows = []
for i, f in enumerate(frames):
    n = f["n"]
    t = f["t"]
    r = f["ref"]
    assert n == 180 + i and abs(t - (302 + i / 30)) < 0.00006 and 300 <= r["t"] < 312
    key = f"{r['c']}-{r['n']}"
    refs[key] = r
    rows.append(dict(n=n, t=t, ref=key, ref_t=r["t"]))
with tarfile.open(mode="w|", fileobj=sys.stdout.buffer) as tar:

    def put(name, data):
        info = tarfile.TarInfo(name)
        info.size = len(data)
        info.mode = 0o600
        tar.addfile(info, io.BytesIO(data))

    for f in frames:
        for sub, out in [("render", "render"), ("b", "studio")]:
            n = f["n"]
            name = f"{n:04d}.png"
            tar.add(
                p / sub / name, arcname=f"bench/sequence/{out}/{name}", recursive=False
            )
    for key, r in refs.items():
        rgb, _ = photo(r["c"], r["n"], size)
        ok, png = cv2.imencode(".png", rgb[:, :, ::-1])
        assert ok
        put(f"bench/sequence/ref/{key}.png", png.tobytes())
    put(
        "bench/sequence/frames.json",
        json.dumps(dict(fps=30, size=size, frames=rows, studio_tag="b")).encode(),
    )
