"""Require independent image motion before displaying a discovered object.

Track interior photographic corners between released exposures of one camera.
For each feature, a stationary point can be anywhere on its earlier sightline
inside a generous road-to-roof volume. Compare its later image against that
entire depth interval, instead of treating ego-relative optical flow as motion.
"""

from functools import lru_cache
from pathlib import Path
import cv2, numpy as np, pickle
from .online import camera
from .inventory import SIZE
from .observations import read
from .demo import R

cv2.setNumThreads(1)


@lru_cache(96)
def image(c, n):
    return cv2.cvtColor(read(c, n), cv2.COLOR_BGR2GRAY)


def evidence(a, b, cls):
    c = a["c"]
    cam = camera(c)
    ia = image(c, a["n"])
    ib = image(c, b["n"])
    raw = cv2.imread(str(Path(a["root"]) / "masks/dyn" / c / f"{a['n']:06}.png"), -1)
    mask = (raw == a["id"]).astype("uint8")
    mask = cv2.erode(mask, np.ones((3, 3), "uint8"))
    p = cv2.goodFeaturesToTrack(ia, 40, 0.012, 3, mask=mask, blockSize=3)
    if p is None or len(p) < 5:
        return None
    wa = np.maximum(4, a["bb"][2:] - a["bb"][:2])
    wb = np.maximum(4, b["bb"][2:] - b["bb"][:2])
    guess = (b["bb"][:2] + (p - a["bb"][:2]) * np.clip(wb / wa, 0.6, 1.6)).astype(
        "float32"
    )
    q, ok, err = cv2.calcOpticalFlowPyrLK(
        ia,
        ib,
        p,
        guess,
        winSize=(17, 17),
        maxLevel=3,
        flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
    )
    back, ok2, _ = cv2.calcOpticalFlowPyrLK(
        ib,
        ia,
        q,
        p.copy(),
        winSize=(17, 17),
        maxLevel=3,
        flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
    )
    m2 = cv2.imread(str(Path(b["root"]) / "masks/dyn" / c / f"{b['n']:06}.png"), -1)
    xy = np.rint(q[:, 0]).astype(int)
    inside = (xy[:, 0] >= 0) & (xy[:, 0] < 724) & (xy[:, 1] >= 0) & (xy[:, 1] < 469)
    ok = (
        ok.ravel().astype(bool)
        & ok2.ravel().astype(bool)
        & inside
        & (err.ravel() < 23)
        & (np.linalg.norm((back - p)[:, 0], axis=1) < 1.25)
        & (m2[np.clip(xy[:, 1], 0, 468), np.clip(xy[:, 0], 0, 723)] == b["id"])
    )
    p = p[ok, 0]
    q = q[ok, 0]
    if len(p) < 5:
        return None
    sf = np.array([cam.width / 724, cam.height / 469])
    rays = cam.unproject((p + 0.5) * sf) @ a["C"][:3, :3].T
    origin = a["C"][:3, 3]
    og = b["C"][:3, 3]
    ranges = np.geomspace(
        max(1.0, a["range"] * 0.38 - 4), min(220.0, a["range"] * 2.1 + 7), 90
    )
    world = origin + rays[:, None, :] * ranges[None, :, None]
    h = world[:, :, 2] - (world[:, :, :2] @ a["plane"][:2] + a["plane"][2])
    valid = (h > -0.4) & (h < SIZE[cls][2] * 1.32 + 0.38)
    u, ok = cam.project((world.reshape(-1, 3) - og) @ b["C"][:3, :3])
    u = (u / sf - 0.5).reshape(len(p), -1, 2)
    dist = np.linalg.norm(u - q[:, None], axis=2)
    dist[~(valid & ok.reshape(dist.shape))] = 1e6
    ix = np.argmin(dist, axis=1)
    keep = dist[np.arange(len(p)), ix] < 1e5
    if keep.sum() < 5:
        return None
    # Camera calibration error and parallax can make a painted sign appear to
    # translate. Measure the local background independently from the object.
    x0, y0, x1, y1 = np.rint(a["bb"]).astype(int)
    pad = int(np.clip(max(x1 - x0, y1 - y0) * 0.4, 12, 45))
    ring = np.zeros_like(mask)
    ring[
        max(0, y0 - pad) : min(469, y1 + pad), max(0, x0 - pad) : min(724, x1 + pad)
    ] = 1
    ring[max(0, y0 - 2) : min(469, y1 + 2), max(0, x0 - 2) : min(724, x1 + 2)] = 0
    bp = cv2.goodFeaturesToTrack(ia, 66, 0.012, 3, mask=ring, blockSize=3)
    contrast = None
    if bp is not None and len(bp) >= 8:
        guess = (bp + np.median(q - p, axis=0)).astype("float32")
        bq, bok, be = cv2.calcOpticalFlowPyrLK(
            ia,
            ib,
            bp,
            guess,
            winSize=(17, 17),
            maxLevel=3,
            flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
        )
        rb, rok, _ = cv2.calcOpticalFlowPyrLK(
            ib,
            ia,
            bq,
            bp.copy(),
            winSize=(17, 17),
            maxLevel=3,
            flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
        )
        bv = (
            bok.ravel().astype(bool)
            & rok.ravel().astype(bool)
            & (be.ravel() < 23)
            & (np.linalg.norm((rb - bp)[:, 0], axis=1) < 1.5)
        )
        pp = bp[bv, 0]
        qq = bq[bv, 0]
        if len(pp) >= 8:
            M, inlier = cv2.estimateAffinePartial2D(
                pp, qq, method=cv2.RANSAC, ransacReprojThreshold=2.1
            )
            if M is not None and inlier.sum() >= max(7, 0.45 * len(pp)):
                pred = p @ M[:, :2].T + M[:, 2]
                contrast = float(np.linalg.norm(np.median((q - pred)[keep], axis=0)))
    delta = q - u[np.arange(len(p)), ix]
    mag = np.linalg.norm(delta[keep], axis=1)
    coherent = np.linalg.norm(np.median(delta[keep], axis=0))
    return dict(
        t=b["release"],
        dt=b["rt"] - a["rt"],
        n=int(keep.sum()),
        p50=float(np.median(mag)),
        coherent=float(coherent),
        p25=float(np.percentile(mag, 25)),
        cam=c,
        background_difference_px=contrast,
    )


def release(evidence):
    strong = []
    for e in evidence:
        if (
            e.get("background_difference_px") is not None
            and e["background_difference_px"] < 2.4
        ):
            continue
        if e["p25"] > 1.2 and e["p50"] > 2.2 and e["coherent"] > 1.6:
            strong = [v for v in strong if e["t"] - v["t"] < 0.75] + [e]
            if len(strong) >= 2 or (
                e["n"] >= 10 and e["p25"] > 3 and e["coherent"] > 5
            ):
                return e["t"]
    return None


def run():
    hs = pickle.load(open(R / "identities.pickle", "rb"))
    res = {}
    for h in hs:
        if h["confirm"] is None:
            continue
        arr = []
        bycam = {}
        last = {}
        for b in h["rows"]:
            c = b["c"]
            history = bycam.setdefault(c, [])
            history[:] = [a for a in history if b["rt"] - a["rt"] < 0.78]
            if b["t"] - last.get(c, -1) > 0.12 and history:
                a = min(history, key=lambda a: abs(b["rt"] - a["rt"] - 0.35))
                dt = b["rt"] - a["rt"]
                if dt > 0.09:
                    last[c] = b["t"]
                    v = evidence(a, b, h["cls"])
                    if v is not None:
                        arr.append(v)
            history.append(b)
        res[h["id"]] = arr
    temp = R / "motion.next"
    pickle.dump(res, open(temp, "wb"))
    temp.replace(R / "motion.pickle")
    print("eligible motion pairs:", sum(map(len, res.values())), flush=True)


if __name__ == "__main__":
    run()
