"""Independent real-photo triangulation check in named regions, never an optimisation input."""

from .v2data import *
from .v2seed import unit, imask
from .evaluate import jpg, label

ROIS = [
    ("93 front roofline", 93, "front", [0.025, 0.28, 0.90, 0.54], 4.5),
    ("101 island canopy and trunk", 101, "left_pillar", [0.16, 0.02, 0.64, 0.72], 1.35),
    ("101 island kerb", 101, "left_pillar", [0.28, 0.56, 0.95, 0.81], 1.35),
    ("112 far office facade", 112, "front", [0.30, 0.35, 0.76, 0.54], 5.0),
    ("112 utility pole", 112, "front", [0.68, 0.04, 0.77, 0.55], 3.0),
    (
        "128 far office and foreground trees",
        128,
        "right_pillar",
        [0.27, 0.16, 0.80, 0.55],
        3.0,
    ),
]


def rayrow(e, q):
    c = e["c"]
    C = rig[c]
    uv = (q + 0.5) * [C.width / 724, C.height / 469]
    v = C.unproject(uv)
    T = T_world_cam(C, e["t"], uv[:, 1], traj)
    return T[:, :3, 3], np.einsum("nij,nj->ni", T[:, :3, :3], v)


def one(name, t, c, roi, base):
    rr = [e for e in schedule()[c] if not e["hold"]]
    e = min((e for e in rr if e["depth"]), key=lambda e: abs(e["t"] - t))
    ee = [min(rr, key=lambda a: abs(a["odo"] - e["odo"] - j)) for j in [-base, 0, base]]
    ee[1] = e
    ims = [rgb(c, a["n"]) for a in ee]
    gray = [cv.cvtColor(im, cv.COLOR_RGB2GRAY) for im in ims]
    lab = imask(c, e["n"])
    h, w = lab.shape
    y, x = np.mgrid[:h, :w]
    mask = (
        (x >= roi[0] * w)
        & (x < roi[2] * w)
        & (y >= roi[1] * h)
        & (y < roi[3] * h)
        & (lab == 1)
    )
    p = cv.goodFeaturesToTrack(
        gray[1], 800, 0.003, 4, mask=mask.astype("B"), blockSize=5
    )
    valid = np.ones(len(p), bool)
    uv = []
    args = dict(
        winSize=(23, 23),
        maxLevel=4,
        criteria=(cv.TERM_CRITERIA_COUNT | cv.TERM_CRITERIA_EPS, 36, 0.01),
    )
    for j in [0, 2]:
        q, st, err = cv.calcOpticalFlowPyrLK(gray[1], gray[j], p, None, **args)
        rev, sr, _ = cv.calcOpticalFlowPyrLK(gray[j], gray[1], q, None, **args)
        ij = q[:, 0].round().astype("i")
        xx, yy = np.clip(ij.T[0], 0, w - 1), np.clip(ij.T[1], 0, h - 1)
        valid &= (
            st[:, 0].astype(bool)
            & sr[:, 0].astype(bool)
            & (np.linalg.norm(p[:, 0] - rev[:, 0], axis=1) < 0.75)
            & (err[:, 0] < 17)
            & (imask(c, ee[j]["n"])[yy, xx] == 1)
        )
        uv.append(q[:, 0])
    views = [rayrow(e, u) for e, u in zip(ee, [uv[0], p[:, 0], uv[1]])]
    o0, d0 = views[0]
    o, d = views[1]
    o2, d2 = views[2]
    b = o2 - o0
    cross = np.cross(d0, d2)
    sine = np.linalg.norm(cross, axis=1)
    r0 = (np.cross(b, d2) * cross).sum(1) / np.maximum(sine**2, 1e-9)
    r2 = (np.cross(b, d0) * cross).sum(1) / np.maximum(sine**2, 1e-9)
    X0 = o0 + d0 * r0[:, None]
    X2 = o2 + d2 * r2[:, None]
    X = (X0 + X2) * 0.5
    r = np.linalg.norm(X - o, axis=1)
    f = w * rig[c].params[0] / rig[c].width
    passed = (
        valid
        & (sine > 0.0105)
        & (np.linalg.norm(unit(X - o) - d, axis=1) * f < 1.0)
        & (np.linalg.norm(X0 - X2, axis=1) / np.maximum(1, r) * f < 1.2)
        & (r0 > 1)
        & (r2 > 1)
        & (r < 210)
    )
    z, wt = cached_range(c, e["n"])
    xx, yy = p[:, 0].round().astype("i").T
    dd = z[yy, xx]
    scored = passed & (wt[yy, xx] > 0)
    rel = abs(dd[scored] - r[scored]) / r[scored]
    q = ims[1].copy()
    cv.rectangle(
        q,
        (int(w * roi[0]), int(h * roi[1])),
        (int(w * roi[2]), int(h * roi[3])),
        (242, 199, 51),
        2,
    )
    for (xx, yy), ok in zip(p[:, 0].round().astype("i"), passed):
        cv.circle(q, (xx, yy), 2, (85, 208, 118) if ok else (235, 106, 78), -1)
    return dict(
        frame=int(e["n"]),
        time=e["t"],
        lens=c,
        roi=roi,
        distinct_physical_images=len(set(v["n"] for v in ee)),
        baselines_m=[abs(v["odo"] - e["odo"]) for v in ee[::2]],
        features=len(p),
        tracked=int(valid.sum()),
        triangulated=int(passed.sum()),
        with_depth=int(scored.sum()),
        relative_depth_error_median=float(np.median(rel)) if len(rel) else None,
        within_15pct=int((rel <= 0.15).sum()),
        median_range_m=float(np.median(r[passed])) if passed.any() else None,
    ), label(cv.resize(q, (480, 311)), name)


def main():
    res = {}
    ims = []
    for roi in ROIS:
        stats, im = one(*roi)
        res[roi[0]] = stats
        ims.append(im)
    for name, t, c, roi, base in ROIS:
        if t == 101:
            alt, _ = one(name, t, c, roi, 0.44)
            res[name + " (shorter baseline)"] = alt
    dest = RS / "work/stitch/report/v2"
    savej(dest / "optical-tracks.json", res)
    jpg(
        dest / "optical-tracks.jpg",
        np.vstack([np.hstack(ims[i : i + 2]) for i in range(0, len(ims), 2)]),
    )
    print(res)


if __name__ == "__main__":
    main()
