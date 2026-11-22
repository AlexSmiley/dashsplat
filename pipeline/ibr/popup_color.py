"""Measure pillar residual gain against the front in the SAME sky directions across the turn."""

from .popup_layers import *
from collections import defaultdict


def fit():
    docs = {c: json.load(open(DST / (c + "-final.json"))) for c in LENSES}
    bins = {}
    for c in LENSES:
        v = defaultdict(list)
        es = bank()[c]["keyframes"]
        lookup = {e["n"]: e for e in es}
        ray = rays(c)[6::12, 6::12]
        for f in docs[c]["base"]:
            e = lookup[f["n"]]
            mask = cv.imread(str(STAGE / e["mask"]), -1)[6::12, 6::12] == 3
            im = cv.imread(str(STAGE / e["rgb"]))[12::24, 12::24, ::-1]
            R, _ = rowmat(e, np.array([0.5]))
            w = ray @ R[0].T
            ang = np.floor(
                (np.rad2deg(np.arctan2(w[:, :, 1], w[:, :, 0])) + 180) / 4
            ).astype(int)
            el = np.floor(np.rad2deg(np.arcsin(np.clip(w[:, :, 2], -1, 1))) / 4).astype(
                int
            )
            keys = ang * 24 + el
            sel = mask & (el >= 1) & (im.min(2) > 25) & (im.max(2) < 241)
            keys = keys[sel]
            colors = im[sel]
            for k in np.unique(keys):
                co = colors[keys == k]
                if len(co) >= 5:
                    v[int(k)].append(np.r_[f["t"], np.median(co, 0)])
        bins[c] = {k: np.array(ls) for k, ls in v.items()}
    report = {}
    for c in LENSES:
        if c == "front":
            continue
        samples = []
        for k, A in bins[c].items():
            if k not in bins["front"]:
                continue
            B0 = bins["front"][k]
            ii = np.argmin(abs(B0[:, 0, None] - A[:, 0]), axis=0)
            dt = abs(B0[ii, 0] - A[:, 0])
            good = dt < 16
            samples.extend(
                np.column_stack(
                    [A[good, 0], dt[good], np.log(B0[ii[good], 1:] / A[good, 1:])]
                )
            )
        A = np.array(samples)
        assert len(A) > 100
        glob = np.median(A[:, 2:], 0)
        report[c] = dict(matchedCells=len(A), medianRGBGain=np.exp(glob).tolist())
        for f in docs[c]["base"] + docs[c]["movers"]:
            wt = np.exp(
                -0.5 * ((A[:, 0] - f["t"]) / 5.5) ** 2 - 0.5 * (A[:, 1] / 8) ** 2
            )
            sg = []
            for j in range(3):
                i = np.argsort(A[:, j + 2])
                p = np.cumsum(wt[i])
                local = A[i[np.searchsorted(p, p[-1] * 0.5)], j + 2]
                ratio = np.clip(wt.sum() / 35, 0, 1)
                sg.append(np.clip(ratio * local + (1 - ratio) * glob[j], -0.50, 0.50))
            f["gain"] = (
                np.array(f.setdefault("frontGauge", f["gain"])) * np.exp(sg)
            ).tolist()
        dump(DST / (c + "-final.json"), docs[c])
        print(c, report[c], flush=True)
    dump(ROOT / "lens-match.json", report)


if __name__ == "__main__":
    fit()
