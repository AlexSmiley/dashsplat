"""Re-triangulate raw LK track prefixes. No BA xyz, offline poses, or future support."""

import os
from .geometry import *
from .audit import BANDS, fit


def run(c):
    C = rig[c]
    ci = NAMES.index(c)
    plan = frames()[c]
    hold = {e["n"] for e in plan if e["hold"]}
    lk = np.load(RS / "work/poses" / f"lk-{c}.npz")
    uv, id, ptr = [lk[z] for z in ("uv", "id", "ptr")]
    s = np.load(RS / "work/poses/schedule.npz")
    ts = s["tc"][:, ci]
    ns = s["ns"][:, ci]
    tl, release = release_table()
    win = 0.5 * abs(C.rs_readout_s or 0)
    ready0 = np.array(
        [
            release[
                max(0, np.searchsorted(tl, t + C.time_offset_s - win) - 1) : min(
                    len(tl), np.searchsorted(tl, t + C.time_offset_s + win) + 1
                )
            ].max()
            for t in ts
        ]
    )
    tcam = traj("live")
    road = np.load(W / "live" / (c + "-anchors.npz"))["road"]
    hist = []
    qual = {}
    prior = []
    allanchors = []
    allroad = []
    frozen = np.cos(np.radians(62 if c == "back" else 88))

    @lru_cache(maxsize=64)
    def rays(k):
        i = slice(ptr[k], ptr[k + 1])
        u = uv[i]
        cw = tcam.T_world_ego(C.row_time(ts[k], u[:, 1])) @ C.T_ego_cam
        d = np.einsum("nij,nj->ni", cw[:, :3, :3], C.unproject(u))
        o = cw[:, :3, 3]
        return id[i], d, o

    ray, cal = lattice(c)
    angles = np.degrees(np.arccos(np.clip(ray[:, :, 2][cal], -1, 1)))
    rim_start = 62.0 if c == "back" else float(np.percentile(angles, 85))
    for e in sorted(plan, key=lambda e: ready("live", c, e)):
        t = e["t"]
        n = e["n"]
        now = ready("live", c, e)
        if now > float(os.environ.get("LIVEMAP_CUTOFF", np.inf)):
            break
        path = W / "live/depth" / c / f"{n:06d}.npy"
        if not path.exists() or not (e["train"] or e["hold"]):
            continue
        # Coefficients and range admission come only from previously released train data.
        l = np.vstack(
            [x for stamp, x in hist if stamp < now and stamp > now - 18]
            + [np.empty((0, 8))]
        )
        r = np.vstack(
            [x for stamp, x in prior if stamp < now and stamp > now - 12]
            + [np.empty((0, 6))]
        )
        q = (
            fit(l[(l[:, 6].astype(int) % 4) != 0], r)
            if len(l) > 25 or len(r) > 12
            else [1.0, 0.0]
        )
        bands = []
        conditional = []
        test = l[l[:, 6].astype(int) % 4 == 0]
        predicted = test[:, 1] / (q[0] + q[1] * test[:, 1])
        predroad = r[:, 1] / (q[0] + q[1] * r[:, 1])
        for lo, hi in BANDS:
            z = test[(predicted >= lo) & (predicted < hi)]
            if len(z) > 0:
                z = z[np.unique(z[:, 6], return_index=True)[1]]
            err = (
                z[:, 1] / (q[0] + q[1] * z[:, 1]) / z[:, 2] - 1
                if len(z)
                else np.array([])
            )
            if len(err) >= 4 and (
                abs(np.median(err)) > 0.17 or np.median(abs(err)) > 0.24
            ):
                continue
            g = r[(predroad >= lo) & (predroad < hi)]
            pass0 = (
                len(z) >= 14
                and abs(np.median(err)) < 0.13
                and np.percentile(abs(err), 90) < 0.32
            )
            if len(z) < 14 and hi <= 15 and len(g) >= 20:
                pass0 = (
                    np.median(abs(g[:, 1] / (q[0] + q[1] * g[:, 1]) / g[:, 2] - 1))
                    < 0.14
                )
            # Unknown is different from failed. Allow proposals (no direct birth) below
            # engineering caps in untested bands; only separated depth agreement can promote them.
            if (
                not pass0
                and len(z) < 14
                and hi <= ({"front": 70, "back": 15}.get(c, 40))
                and (
                    not len(g)
                    or np.median(abs(g[:, 1] / (q[0] + q[1] * g[:, 1]) / g[:, 2] - 1))
                    < 0.23
                )
            ):
                conditional.append([lo, hi])
                bands.append([lo, hi])
            if hi == 70 and c != "front":
                pass0 = False
            if pass0:
                bands.append([lo, hi])
        rim = test[test[:, 3] >= rim_start]
        rim = rim[np.unique(rim[:, 6], return_index=True)[1]]
        re = rim[:, 1] / (q[0] + q[1] * rim[:, 1]) / rim[:, 2] - 1
        capangle = (
            rim_start
            if c == "back"
            or len(re) < 14
            or np.percentile(abs(re), 90) > 0.32
            or abs(np.median(re)) > 0.13
            else float(angles.max() + 0.01)
        )
        qual[str(n)] = dict(
            a=float(q[0]),
            b=float(q[1]),
            bands_admitted=bands,
            conditional=conditional,
            max_angle=capangle,
            max_range=max([r[1] for r in bands] + [1]),
            past_anchor_n=len(l),
        )
        if e["hold"]:
            continue
        prior.append((now, road[road[:, 0] == e["ordinal"]]))
        ks = np.flatnonzero((ts <= t) & (ts >= t - 3.2) & (ready0 <= now))
        if len(ks) < 5 or t - ts[ks[-1]] > 0.2:
            continue
        ks = ks[-30:][::-2][::-1]
        ks = [k for k in ks if ns[k] not in hold]
        if len(ks) < 4:
            continue
        parts = [rays(k) for k in ks]
        ii, rr, oo = [np.concatenate([v[j] for v in parts]) for j in range(3)]
        order = np.argsort(ii, kind="stable")
        ii = ii[order]
        rr = rr[order]
        oo = oo[order]
        uniq, st, num = np.unique(ii, return_index=True, return_counts=True)
        last = st + num - 1
        M = np.eye(3) - rr[:, :, None] * rr[:, None, :]
        A = np.add.reduceat(M, st)
        rhs = np.add.reduceat(np.einsum("nij,nj->ni", M, oo), st)
        xyz = np.linalg.solve(A + 1e-8 * np.eye(3), rhs[..., None])[..., 0]
        vec = np.repeat(xyz, num, axis=0) - oo
        sg = np.linalg.norm(np.cross(vec, rr), axis=1) / np.maximum(
            1, np.linalg.norm(vec, axis=1)
        )
        par = np.sqrt(np.maximum(0, np.linalg.eigvalsh(A)[:, 0] / num))
        good = (
            (num >= 4)
            & (par > 0.018)
            & (np.maximum.reduceat(sg, st) < 0.003)
            & np.isin(uniq, parts[-1][0])
        )
        xyz = xyz[good]
        uniq = uniq[good]
        par = par[good]
        dist, safe, _, _ = maps("live", c, n)
        cw = rows("live", c, t)[0]
        pu, vis, v, _ = projected(c, xyz, cw)
        x = np.clip(pu[:, 0] * B / C.width, 0, B - 1).astype(int)
        y = np.clip(pu[:, 1] * H / C.height, 0, H - 1).astype(int)
        ref = np.linalg.norm(v, axis=1)
        ok = vis & safe[y, x] & (ref > 2) & (ref < 70) & (v[:, 2] > ref * frozen)
        wt = np.clip(par * 60 / (np.maximum(4, ref) / 10), 0.1, 4)
        a = np.column_stack(
            (
                np.full(len(ref), e["ordinal"]),
                dist[y, x],
                ref,
                np.degrees(np.arccos(np.clip(v[:, 2] / np.maximum(1, ref), -1, 1))),
                wt,
                np.zeros(len(ref)),
                uniq,
                np.zeros(len(ref)),
            )
        )[ok]
        hist.append((now, a))
        allanchors.append(a)
        allowed = []
        for lo, hi in bands:
            pred = a[:, 1] / (q[0] + q[1] * a[:, 1])
            zz = a[(pred >= lo * 0.82) & (pred < hi * 1.25)]
            err = zz[:, 1] / (q[0] + q[1] * zz[:, 1]) / zz[:, 2] - 1
            if len(zz) >= 2 and (
                np.mean(err > 0.33) >= 0.66 or np.mean(err < -0.30) >= 0.66
            ):
                continue
            if len(zz) >= 4 and (
                abs(np.median(err)) > 0.17 or np.median(abs(err)) > 0.24
            ):
                continue
            allowed.append([lo, hi])
        qual[str(n)]["bands_admitted"] = allowed
    path = W / "live" / (c + "-causal-policy.json")
    if "LIVEMAP_CUTOFF" in os.environ:
        full = json.loads(path.read_text())
        assert all(v == full[k] for k, v in qual.items()), c
        path = W / "live" / (c + "-prefix-policy.json")
    path.write_text(json.dumps(qual))
    print(
        c,
        len(qual),
        sum(len(x) for x in allanchors),
        "chronological anchors",
        flush=True,
    )


if __name__ == "__main__":
    import sys

    for c in sys.argv[1:] or CAMS:
        run(c)
