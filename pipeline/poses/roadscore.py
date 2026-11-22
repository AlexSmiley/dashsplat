"""Score and visually audit independent appearance correspondences on road surfaces."""

from .validate import *
import cv2 as cv
from collections import Counter
from scipy.optimize import least_squares


def refine_far(s, X, c, R, O, uv):
    # Only source pixels: optimize depth and bearing in the actual lens metric.
    def residual(v):
        p = np.empty_like(uv)
        q = np.einsum("nji,nj->ni", R, v - O)
        for i in np.unique(c):
            rows = c == i
            p[rows] = s.C[i].project(q[rows])[0]
        return (p - uv).ravel()

    v = least_squares(residual, X, loss="huber", f_scale=2.5, max_nfev=35).x
    return v


def score():
    s = Fit().load()
    tr = Trajectory(OUT / "ego.npz")
    live = Trajectory(OUT / "ego_causal.npz")
    planar = Trajectory(
        W / "no-refined-poses.npz", timeline=RS / "data/drive/timeline.npz"
    )
    rows = json.load(open(W / "road-tracks.json"))
    stats = defaultdict(list)
    counts = defaultdict(set)
    allcomp = defaultdict(list)
    examples = defaultdict(list)
    funnel = Counter()
    oracle = defaultdict(list)
    for j, rec in enumerate(rows):
        reg = rec["region"]
        v = np.array(rec["obs"])
        k, c = v[:, :2].astype(int).T
        uv = v[:, 2:4]
        n = len(k)
        r, O, R, P, L = exact(s, tr, k, c, uv)
        tiny = np.arange(n - n // 3, n)
        Xn, an = solve_ray(r[tiny], O[tiny])
        range_all = np.linalg.norm(Xn - O, axis=1)
        funnel["appearance"] += 1
        if (
            range_all[0] <= 25
            or not 0.8 < range_all[-1] < 8
            or an < 0.000015
            or (((Xn - O[tiny]) * r[tiny]).sum(1) <= 0.2).any()
        ):
            continue
        last = tiny[range_all[tiny] < 8]
        first = np.arange(n // 3)
        first = first[range_all[first] > 25]
        if len(last) < 2 or len(first) < 3:
            continue
        funnel["range"] += 1
        Xf, af = solve_ray(r[first], O[first])
        e0 = s.project(
            c[first],
            np.broadcast_to(Xf, (len(first), 3)),
            R[first],
            O[first],
            uv[first],
        )[0]
        en = s.project(
            c[tiny], np.broadcast_to(Xn, (len(tiny), 3)), R[tiny], O[tiny], uv[tiny]
        )[0]
        if af < 0.000008 or (((Xf - O[first]) * r[first]).sum(1) < 0.5).any():
            continue
        funnel["parallax"] += 1
        if (
            np.median(e0) > 3.5
            or np.percentile(e0, 85) > 7
            or np.median(en) > 4
            or np.percentile(en, 85) > 8
        ):
            continue
        funnel["separately_consistent"] += 1
        Xf = refine_far(s, Xf, c[first], R[first], O[first], uv[first])
        out = s.project(
            c[last], np.broadcast_to(Xf, (len(last), 3)), R[last], O[last], uv[last]
        )[0]
        # Diagnostic only: change depth along the middle far ray using near-only
        # information. This is deliberately not counted as a held-out prediction.
        q = first[len(first) // 2]
        Xoracle = O[q] + r[q] * ((Xn - O[q]) @ r[q])
        ee = s.project(
            c[last],
            np.broadcast_to(Xoracle, (len(last), 3)),
            R[last],
            O[last],
            uv[last],
        )[0]
        alt = []
        for tr0 in (planar, live):
            ra, oo, aa, *_ = exact(s, tr0, k, c, uv)
            xp, _ = solve_ray(ra[first], oo[first])
            xp = refine_far(s, xp, c[first], aa[first], oo[first], uv[first])
            alt.append(
                s.project(
                    c[last],
                    np.broadcast_to(xp, (len(last), 3)),
                    aa[last],
                    oo[last],
                    uv[last],
                )[0]
            )
        for qi, m in enumerate(last):
            for label in ["whole", reg]:
                oracle[label].append(ee[qi])
                stats[label, c[m]].append(out[qi])
                counts[label, c[m]].add(j)
                allcomp[label].append([out[qi], alt[0][qi], alt[1][qi]])
        examples[reg].append(
            (
                float(np.median(out)),
                j,
                int(first[len(first) // 2]),
                int(last[-1]),
                Xf.tolist(),
            )
        )
    fmt = {
        g: [
            (
                len(counts[g, i]),
                len(stats[g, i]),
                *map(float, np.percentile(stats[g, i], [50, 90])),
            )
            if stats[g, i]
            else (0, 0, None, None)
            for i in range(6)
        ]
        for g in ["whole", "car park", "90° turn", "exit turn", "fast straight"]
    }
    comp = {g: np.percentile(v, [50, 90], axis=0).tolist() for g, v in allcomp.items()}
    res = dict(
        table=fmt,
        paired=comp,
        funnel=dict(funnel),
        depth_oracle={
            g: np.percentile(v, [50, 90]).tolist() for g, v in oracle.items()
        },
    )
    json.dump(res, open(W / "roadmetrics.json", "w"), indent=2)
    print("road geometry validation", res, flush=True)
    # Raw-image contexts and enlarged raw pixel crops. Select by rank rather than smallest errors.
    panels = []
    for reg, group in examples.items():
        group.sort()
        sel = [
            group[len(group) // 2],
            group[min(len(group) - 1, int(0.9 * len(group)))],
        ]
        for err, j, f, l, Xf in sel:
            rec = rows[j]
            v = np.array(rec["obs"])
            k, c = v[:, :2].astype(int).T
            uv = v[:, 2:4]
            cs = []
            for label, ia in [("far", f), ("near", l)]:
                i = ia
                cam = s.C[c[i]]
                img = cv.imread(
                    str(
                        RS
                        / "data/drive/frames"
                        / NAMES[c[i]]
                        / f"{s.s['ns'][k[i], c[i]]:06d}.jpg"
                    )
                )
                small = cv.resize(img, (430, 279))
                pts = np.rint(
                    (uv[i] - 0.5) * [430 / cam.width, 279 / cam.height]
                ).astype(int)
                tile = cv.getRectSubPix(
                    img, (94, 94), tuple((uv[i] - 0.5).astype(float))
                )
                tile = cv.resize(tile, (164, 164))
                cv.drawMarker(tile, (82, 82), (0, 220, 255), cv.MARKER_CROSS, 12, 1)
                cv.circle(small, tuple(pts), 8, (0, 220, 255), 2)
                if label == "near":
                    _, oo, aa, *_ = exact(
                        s, tr, k[i : i + 1], c[i : i + 1], uv[i : i + 1]
                    )
                    pred = cam.project(aa[0].T @ (np.asarray(Xf) - oo[0]))[0]
                    px = np.rint(
                        (pred - 0.5) * [430 / cam.width, 279 / cam.height]
                    ).astype(int)
                    cv.line(small, tuple(pts), tuple(px), (225, 70, 225), 1, cv.LINE_AA)
                    cv.drawMarker(
                        small, tuple(px), (225, 70, 225), cv.MARKER_CROSS, 9, 2
                    )
                cn = np.zeros((279, 164, 3), np.uint8)
                cn[55:219] = tile
                cs.append(np.hstack((small, cn)))
            frame = np.vstack((np.zeros((37, 1188, 3), np.uint8), np.hstack(cs)))
            cv.putText(
                frame,
                f"{reg}: {NAMES[c[f]]} {s.kt[k[f]]:.2f}s to {NAMES[c[l]]} {s.kt[k[l]]:.2f}s; near error {err:.1f}px",
                (9, 26),
                0,
                0.65,
                (245, 245, 245),
                1,
                cv.LINE_AA,
            )
            panels.append(frame)
    if panels:
        cv.imwrite(str(OUT / "debug/road-transfer-audit.jpg"), np.vstack(panels))


if __name__ == "__main__":
    score()
