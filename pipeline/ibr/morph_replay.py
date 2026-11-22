"""Decode packaged texture clips and delta grids, recompute morph for a small orbit. Host contributes only fallback ground."""

from . import morph_chase as m, morph_sources as src
from .morph_clip import *
import zipfile, zlib, tempfile
from types import SimpleNamespace


def check(bank, out, t=118.0):
    with tempfile.TemporaryDirectory(dir=W) as tmp:
        root = Path(tmp)
        with zipfile.ZipFile(bank) as z:
            z.extractall(root)
        doc = J(root / "bank.json")
        record = min(doc["selection"], key=lambda s: abs(s["t"] - t))
        assert record["t"] == t
        lookup = {}
        P = {}
        for c in doc["rig"]:
            C = doc["rig"][c]
            src.rig[c] = SimpleNamespace(
                params=np.array(C["params"]), width=C["size"][0], height=C["size"][1]
            )
            nmax = max(r["records"][-1]["n"] for r in doc["sources"] if r["lens"] == c)
            P[c] = np.zeros((nmax + 1, 4, 4), "f")
        for r in doc["sources"]:
            for j, e in enumerate(r["records"]):
                c = r["lens"]
                n = e["n"]
                pose = np.eye(4, dtype="f")
                pose[:3] = np.array(e["camToWorld"]).reshape(3, 4)
                P[c][n] = pose
                lookup[c, n] = r, j

        @lru_cache(40)
        def decoded(c, n):
            run, j = lookup[c, n]
            recs = run["records"]
            start = j - j % 16
            f = (root / run["grid"]).open("rb")
            bits = None
            for k in range(start, j + 1):
                e = recs[k]
                f.seek(e["offset"])
                x = np.frombuffer(zlib.decompress(f.read(e["length"])), "B")
                bits = x if e["absolute"] else bits + x
            f.close()
            g = bits.reshape(GH, GW, 8)
            A = np.zeros((GH, GW, 7), "f")
            A[:, :, :2] = g[:, :, :4].copy().view("<u2").reshape(GH, GW, 2) / 65535
            A[:, :, [2, 3, 4, 6]] = g[:, :, 4:] / 255.0
            cap = cv.VideoCapture(str(root / run["texture"]))
            cap.set(cv.CAP_PROP_POS_FRAMES, j)
            ok, im = cap.read()
            cap.release()
            assert ok
            return P[c][n], A, im

        orig = {k: getattr(m, k) for k in ["source", "seq", "sky", "plane"]}
        m.source = decoded
        m.seq = lambda c: (None, np.arange(len(P[c])), P[c], None)
        m.sky = lambda: cv.imread(str(root / doc["sky"]["file"]))
        pl = np.array(record["groundPlane"])
        m.plane = lambda t, T, r: np.clip(
            -(r @ pl[:3]) / max(1, T[:3, 3] @ pl[:3] + pl[3]), 0, 1 / 2
        ).astype("f")
        T = np.eye(4, dtype="f")
        T[:3] = np.array(record["chaseCV"]).reshape(3, 4)
        pivot = T[:3, 3] + T[:3, 2] * 16.2
        fixed = [(c, (n, n + 1, float(f)), 1.0) for c, n, f in record["pairs"]]
        pics = []
        stats = []
        cap = cv.VideoCapture(str(out / "chase.mp4"))
        cap.set(cv.CAP_PROP_POS_FRAMES, round((t - 110) * 30))
        ok, reference = cap.read()
        assert ok
        cap.release()
        for yaw in [-3, 0, 3]:
            C = T.copy()
            R = Rotation.from_euler("z", yaw, degrees=True).as_matrix()
            C[:3, :3] = R @ T[:3, :3]
            C[:3, 3] = pivot + R @ (T[:3, 3] - pivot)
            im, res = m.render(t, C, record["baseArc"], fixed=fixed)
            small = cv.resize(im, (480, 300))
            cv.putText(
                small,
                f"{yaw:+d} deg / decoded source bank",
                (10, 22),
                0,
                0.48,
                (244, 244, 244),
                1,
                cv.LINE_AA,
            )
            pics.append(small)
            stats.append(
                dict(
                    yaw=yaw,
                    eyeDeltaM=float(np.linalg.norm(C[:3, 3] - T[:3, 3])),
                    photo=res["photo"],
                    referenceMAE=float(np.mean(abs(reference.astype("f") - im)))
                    if yaw == 0
                    else None,
                )
            )
        cv.imwrite(str(out / "small-orbit.jpg"), np.hstack(pics))
        savej(out / "small-orbit.json", stats)
        print(stats, flush=True)
        for k, v in orig.items():
            setattr(m, k, v)


if __name__ == "__main__":
    out = W / "110-125"
    check(out / "live-photo-bank.zip", out)
