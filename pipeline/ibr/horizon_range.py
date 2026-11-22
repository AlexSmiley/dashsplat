"""Compare host handover distances and an optional 32–48 m second curtain against held-out calibrated views."""

from .horizon_publish import stage
from .horizon_curtain import *
from .horizon_proof import Hybrid, SITES, state, save
from . import layer_player as lp
from functools import lru_cache

OPT = RS / "scene/stitched-opt-v2"
LIB = RS / "work/stitch/report/v2/wide.dylib"
TR = W / "stitched-chase.json"
REPORT = W / "proof"
REPORT.mkdir(exist_ok=True)


def holdouts():
    allframes = load(RS / "data/drive/cams/front.json")
    used = {s["front"]["n"] for s in rows()}
    return [
        min((e for e in allframes if e["n"] not in used), key=lambda e: abs(e["t"] - t))
        for t in SITES
    ]


def needs(h):
    ids = set()
    ft = holdouts()
    for t, e in zip(SITES, ft):
        T = h.view(t)[1]
        ids.update(h.selection(t, T)[0])
        tc = (
            traj("live").T_world_ego(
                rig["front"].row_time(e["t"], rig["front"].height * 0.5)
            )
            @ rig["front"].T_ego_cam
        )
        ids.update(h.selection(e["t"], tc)[0])
    return sorted({anchors()[k] for k in ids})


def band():
    p = stage()
    h = Hybrid(p, OPT, TR, LIB)
    need = needs(h)
    src = sorted(
        {j for i in need for j in range(max(0, i - 5), min(i + 6, len(rows())))}
    )
    folder = W / "mid-test"
    with ThreadPoolExecutor(3) as pool:
        list(pool.map(lambda k: primary(k, 32, 48), src))
        stats = list(pool.map(lambda k: braid(k, 32, 48, folder), need))
    dump(
        W / "mid.json", dict(knots=need, columns=[s["neighbourColumns"] for s in stats])
    )
    print("range trial baked", len(need), "mid-field curtains", flush=True)


def assess():
    h = Hybrid(stage(), OPT, TR, LIB)
    mid = {
        anchors().index(i): (W / "mid-test" / f"{i:04}", origins()[i])
        for i in load(W / "mid.json")["knots"]
    }
    results = []
    physical = holdouts()
    opts = [([34, 49], False), ([40, 60], False), ([50, 70], False), ([40, 60], True)]
    samples = []
    nativeView = h.view
    for t, e in zip(SITES, physical):
        C = rig["front"]
        et = e["t"]
        Td = traj("live").T_world_ego(C.row_time(et, C.height * 0.5)) @ C.T_ego_cam
        ds = state(h.dd, et)
        hs = state(h.hd, et)
        R = Rotation.from_euler("z", hs[3] - ds[3]).as_matrix()
        Th = Td.copy()
        Th[:3, :3] = R @ Td[:3, :3]
        Th[:3, 3] = hs[:3] + R @ (Td[:3, 3] - ds[:3])
        uv, okay = C.project(lp.RAYS)
        p = (uv * [1448 / C.width, 938 / C.height] - 0.5).astype("f")
        nearestRow = min(rows(), key=lambda s: abs(s["front"]["t"] - et))["front"]
        g = np.array(gains()[f"front:{nearestRow['n']}"], "f")
        gt = (
            cv.remap(pe("front", e["n"]), p, None, cv.INTER_LINEAR)[:, :, ::-1]
            * g[::-1]
        )
        gray = cv.cvtColor(gt.astype("f"), cv.COLOR_RGB2GRAY)
        gradient = lambda im: (
            np.hypot(cv.Sobel(im, cv.CV_32F, 1, 0), cv.Sobel(im, cv.CV_32F, 0, 1))
            * 0.125
        )
        edge = gradient(gray)
        ls = []
        pics = []
        h.view = nativeView
        for fade, m in opts:
            h.mid = mid if m else None
            im, _ = h.draw(t, fade=fade)
            crop = im[100:440]
            pics.append(cv.resize(crop, (520, 123)))
        h.view = lambda unused: (Th, Td)
        for fade, m in opts:
            h.mid = mid if m else None
            im, d = h.draw(et, fade=fade)
            ls.append((im, d))
        h.view = nativeView
        ym = np.argmax(ls[1][1]["alpha"] > 0.5, axis=0)
        yy = np.arange(lp.H)[:, None]
        mask = okay & (ym > 0) & (yy > ym - 18) & (yy < ym + 42)
        row = {}
        for (im, d), (fade, m) in zip(ls, opts):
            tag = "mid" if m else f"{fade[0]}-{fade[1]}"
            l = cv.cvtColor(im, cv.COLOR_RGB2GRAY).astype("f")
            row[tag] = dict(
                pixels=int(mask.sum()),
                mae=float(abs(l - gray)[mask].mean()),
                edgeError=float(abs(gradient(l) - edge)[mask].mean()),
            )
        results.append(dict(chase=t, heldOutTime=et, n=e["n"], metrics=row))
        samples.append(np.hstack(pics))
    save(REPORT / "range-comparison.jpg", np.vstack(samples))
    keys = list(results[0]["metrics"])
    mean = {
        key: {
            m: float(
                np.average(
                    [x["metrics"][key][m] for x in results],
                    weights=[x["metrics"][key]["pixels"] for x in results],
                )
            )
            for m in ["mae", "edgeError"]
        }
        for key in keys
    }
    dump(
        W / "range.json",
        dict(
            heldout=results, mean=mean, order=keys, sourceStartM=45, trialMid=[32, 48]
        ),
    )
    print(mean, flush=True)


if __name__ == "__main__":
    band() if sys.argv[-1] == "band" else assess()
