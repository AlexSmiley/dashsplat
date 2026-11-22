"""Payload, angular support and cross-interval readback using the final published assets and official poses only."""

from .horizon_proof import *


def check(h, out):
    doc = h.doc
    ps = doc["panoramas"]
    programs = [
        (0, 85),
        (85, 140),
        (190, 215),
        (286, 330),
        (105, 120),
        (0, doc["interval"][1]),
    ]
    assets = {p[k] for p in ps for k in ["sky", "rgb", "matte"]}
    sizes = {p: h.asset(p).stat().st_size for p in assets}
    budget = {}
    assert len({p["sky"] for p in ps}) == 1
    for a, b in programs:
        seen = set()
        loads = []
        for i in range(round(30 * a), min(len(h.poses), int(np.ceil(30 * b)))):
            t = i / 30
            ids, _ = h.selection(t, h.poses[i])
            now = {ps[k][key] for k in ids for key in ["sky", "rgb", "matte"]}
            loads.append(sum(sizes[x] for x in now - seen))
            seen |= now
        budget[f"{a}-{b:.2f}"] = dict(
            uniqueBytes=sum(loads),
            meanMBs=sum(loads) / 1e6 / (b - a),
            peakRollingMB=float(np.max(np.convolve(loads, np.ones(30), "valid")) / 1e6),
        )
    info = []
    span = []
    missing = []
    for p in ps:
        m = cv.imread(str(h.asset(p["matte"])), 0)
        y = np.argmax(m > 0.50 * 255, axis=0)
        x = np.arange(m.shape[1])
        ok = (y >= 5) & (y < m.shape[0] - 6) & (m.max(0) > 229)
        y = y[ok]
        x = x[ok]
        narrow = np.arange(-5, 7)[:, None]
        seq = m[(y + narrow), x]
        lo = np.where((narrow <= 0) & (seq < 26), narrow, -6).max(0)
        hi = np.where((narrow >= 0) & (seq > 229), narrow, 7).min(0)
        valid = (lo > -6) & (hi < 7)
        span += (hi[valid][::8] - lo[valid][::8] - 1).tolist()
        missing.append(int((m.max(0) <= 127).sum()))
    pics = []
    datas = []
    for t in [0.0, 85.0, 140.0, 190.0, 203.0, 215.0, 286.0, 310.0, 330.0, 361.4]:
        T = h.poses[round(t * 30)]
        w, _ = h.selection(t, T)
        bg, z, m, pr = h.background(t, T, w)
        im = np.uint8(np.clip(bg + 0.5, 0, 255))
        p = cv.resize(im, (400, 250), interpolation=cv.INTER_AREA)
        cv.putText(
            p,
            f"{t:.1f} s  |  photographs only",
            (6, 20),
            0,
            0.47,
            (245, 245, 245),
            1,
            cv.LINE_AA,
        )
        pics.append(p)
        datas.append(
            dict(
                time=t,
                sourceTimes=[ps[k]["t"] for k in w],
                backdropPixels=int((m > 0.5).sum()),
            )
        )
    save(
        out / "drive-coverage.jpg",
        np.vstack([np.hstack(pics[i : i + 2]) for i in range(0, len(pics), 2)]),
    )
    audit = dict(
        revision=doc["contentRevision"],
        sha256=hashlib.sha256(h.asset("ibr.json").read_bytes()).hexdigest(),
        range=doc["backdropRange"],
        photoBytes=sum(sizes.values()),
        numPanoramas=len(ps),
        skyFiles=1,
        baseRecords=sum(map(len, h.cams.values())),
        cutouts=len(h.cards),
        sumUncoveredColumns=sum(missing),
        maxUncoveredColumns=max(missing),
        mattePartialTexelsP50=float(np.median(span)),
        mattePartialTexelsP90=float(np.percentile(span, 90)),
        budget=budget,
        samples=datas,
    )
    (out / "asset-audit.json").write_text(json.dumps(audit, indent=2))
    print(audit, flush=True)


if __name__ == "__main__":
    root, host, track, library, path = sys.argv[1:]
    h = Hybrid(root, host, track, library)
    check(h, Path(path))
