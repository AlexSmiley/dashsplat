from pathlib import Path
import os, sys, tarfile, io, json, numpy as np

root = Path.home() / "dev/reality-splat/work/drives/hw4-raw-224-254/1361f01da6e9b73f"
w = root / "full"
chase = w / "work/run/chase"
data = root / "data/drive"
tar = tarfile.open(fileobj=sys.stdout.buffer, mode="w|gz")


def buf(p, b):
    t = tarfile.TarInfo(p)
    t.size = len(b)
    tar.addfile(t, io.BytesIO(b))


def j(p, o):
    buf(p, json.dumps(o).encode())


def add(p, rel):
    tar.add(str(p.resolve()), arcname=rel, recursive=False)


sel = {}
allids = {}
table = {}
tl = np.load(data / "timeline.npz")
odo = np.r_[0, np.cumsum(np.hypot(np.diff(tl["x"]), np.diff(tl["y"])))]
k = json.load(open(w / "depth/keyframes.json"))["frames"]
ts = [a["t_key"] for a in k["front"] if a["t_key"] >= 233][:12]
assert len(ts) == 12
for cam, entries in k.items():
    rows = [
        r
        for r in json.load(open(data / "cams" / f"{cam}.json"))
        if 224 <= r["t"] <= 254
    ]
    table[cam] = {r["n"]: r["t"] for r in rows}
    ns = np.array([r["n"] for r in rows])
    od = np.interp([r["t"] for r in rows], tl["t"], odo)
    ids = set()
    sel[cam] = [a for a in entries if a["t_key"] in ts]
    assert len(sel[cam]) == 12
    for a in sel[cam]:
        i = np.searchsorted(ns, a["n"])
        last = od[i]
        n = 1
        ids.add(int(ns[i]))
        while n < (3 if cam == "back" else 5) and i > 0:
            i -= 1
            if last - od[i] >= 1:
                ids.add(int(ns[i]))
                last = od[i]
                n += 1
    allids[cam] = ids
    j("data/drive/cams/" + cam + ".json", [{"n": r["n"], "t": r["t"]} for r in rows])
    add(root / "calib/masks" / f"{cam}.png", "calib/masks/" + cam + ".png")
add(root / "calib/rig.json", "calib/rig.json")
j("depth/keyframes.json", {"frames": sel, "selections": []})
for p, keys in [
    (data / "timeline.npz", ["t", "x", "y", "yaw", "speed"]),
    (root / "poses/ego.npz", ["t", "T_world_ego"]),
]:
    z = np.load(p)
    ix = (z["t"] >= 224) & (z["t"] <= 254)
    b = io.BytesIO()
    np.savez(b, **{k: z[k][ix] for k in keys})
    buf(
        "poses/ego.npz" if "poses" in str(p) else "data/drive/timeline.npz",
        b.getvalue(),
    )
f = json.load(open(chase / "frames.json"))
frames = []
for n in (0, 40, 80):
    a = f["frames"][n]
    r = a["ref"]
    assert 224 <= a["t"] <= 254 and 224 <= r["t"] <= 254 and r["n"] in table[r["c"]]
    allids[r["c"]].add(r["n"])
    frames.append(a)
    add(chase / "render" / f"{n:04}.png", f"bench/render/{n:04}.png")
    add(chase / "repaired" / f"{n:04}.png", f"bench/studio/{n:04}.png")
j("bench/frames.json", dict(K=f["K"], frames=frames))
for cam, ids in allids.items():
    for n in ids:
        assert 224 <= table[cam][n] <= 254
        add(
            data / "frames" / cam / f"{n:06}.jpg", f"data/drive/frames/{cam}/{n:06}.jpg"
        )
fit = w / "work/run/chunk-0000/fit"
m = json.load(open(fit / "map.json"))
T = np.array(frames[0]["T_world_cam"])
seltiles = []
for item in m["tiles"]:
    if "dense" not in item["file"]:
        continue
    bbox = np.array(item["bbox"]).reshape(2, 3)
    v = (bbox.mean(0) - T[:3, 3]) @ T[:3, :3]
    if 1 < v[2] < 65 and abs(v[0]) < v[2] + 12:
        seltiles.append(item)
for item in sorted(seltiles, key=lambda x: -x["count"])[:3]:
    add(fit / item["file"], "bench/" + item["file"])
# The same calibrated photo() path used by clip.fix, not a stretched sensor JPEG.
os.environ["RS"] = str(w)
from pipeline.genfill.views import photo
import cv2

for a in frames:
    r = a["ref"]
    pixels = photo(r["c"], r["n"], (1472, 960))[0]
    ok, png = cv2.imencode(".png", pixels[:, :, ::-1])
    assert ok
    buf(f"bench/ref/{a['n']:04}.png", png.tobytes())
tar.close()
