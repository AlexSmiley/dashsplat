"""Evidence only. Compare frozen pixels, never use reviewed IDs for discovery."""

from collections import Counter, defaultdict
from pathlib import Path
import json, pickle, numpy as np, cv2
from PIL import Image
from .demo import W, R
from .shutters import clock, correct
from .model import projection
from .shape import silhouette
from .packets import iou
from pipeline.common.rig import load_rig, Trajectory


def truth():
    return [pickle.load(open(p, "rb"))[:2] for p in sorted(W.glob("stitched-*.pickle"))]


def timing(run, scene, native):
    r = W / run
    sc = json.load(open(r / "report/online-evaluation.json"))
    oo = {
        o["id"]: o
        for o in json.load(open(Path("scene") / scene / "objects.json"))["objects"]
    }
    hs = pickle.load(open(r / "identities.pickle", "rb"))
    by = defaultdict(list)
    for h in hs:
        for row in h["rows"]:
            by[row["c"], row["id"]].append(
                (row["rt"], row["n"], h["id"], row["release"])
            )
    delays = []
    report = {}
    all_causes = Counter()
    details = []
    for sp, rows in truth():
        name = sp["name"]
        v = sc["objects_benchmark"][name]
        uid = v["primary"][0] if v["primary"] else None
        o = oo.get(uid)
        rr = [correct(q) if native else q for q in rows if q["key"]]
        first = min(q["rt"] for q in rr)
        right_obs = [
            t2
            for q in rr
            if not q.get("held")
            for rt, n, u, t2 in by[q["c"], q["id"]]
            if u == uid and n == q["n"]
        ]
        birth = max(min(right_obs), o["t_first"]) if right_obs and o else None
        d = (
            max(0, birth - first)
            if birth is not None and birth <= o["t_last"]
            else None
        )
        if d is not None:
            delays.append(d)
        causes = Counter()
        before = after = 0
        for q in rr:
            if not q.get("held"):
                continue
            t = q["rt"]
            if o and o["t_first"] <= t <= o["t_last"]:
                continue
            early = o and t < o["t_first"]
            before += int(bool(early))
            after += int(bool(o and not early))
            recent = [h for h in by[q["c"], q["id"]] if abs(h[0] - t) < 0.34]
            ids = [x[2] for x in recent]
            small = np.prod(np.subtract(q["bb"][2:], q["bb"][:2])) < 160
            cause = (
                "unmatched identity"
                if not o
                else "split / wrong persistent ID"
                if recent and uid not in ids
                else "late birth"
                if early
                else "premature retirement"
                if t < max(h[0] for pairs in by.values() for h in pairs if h[2] == uid)
                else "track ended / handover"
            )
            causes[cause] += 1
            details.append(
                dict(
                    name=name,
                    cam=q["c"],
                    n=q["n"],
                    t=round(t, 4),
                    cause=cause,
                    small=bool(small),
                    rear=q["c"] == "back",
                )
            )
        all_causes.update(causes)
        report[name] = dict(
            primary=uid,
            first_detection_s=round(first, 4),
            already_renderable_other_assignment=bool(
                right_obs and o and o["t_first"] < min(right_obs) - 0.05
            ),
            first_assigned_s=round(min(right_obs), 4) if right_obs else None,
            scene_birth_s=o["t_first"] if o else None,
            first_renderable_correct_assignment_delay_s=round(d, 3)
            if d is not None
            else None,
            before=before,
            after=after,
            causes=dict(causes),
            score=v["strict"],
        )
    return dict(
        median_s=float(np.median(delays)),
        p90_s=float(np.percentile(delays, 90)),
        episodes_timed=len(delays),
        causes=dict(all_causes),
        details=details,
        episodes=report,
    )


def hind():
    root = W / "photo-demo-native"
    oo = {
        o["id"]: o for o in json.load(open(root / "hindsight/objects.json"))["objects"]
    }
    cs = load_rig()
    tr = Trajectory("poses/ego.npz")
    out = [0, 0, 0]
    for sp, rr in truth():
        o = oo.get(sp["id"])
        for r0 in rr:
            if not r0.get("held"):
                continue
            r = correct(r0)
            out[0] += 1
            if not o or not o["t_first"] <= r["rt"] <= o["t_last"]:
                out[1] += 1
                continue
            p = silhouette(o["class"], o["size"])
            uv, ok, *_ = projection(cs[r["c"]], tr, r["t"], o["track"], p, p * 0)
            b = np.clip(np.r_[uv.min(0), uv.max(0)], [0, 0, 0, 0], [724, 469, 724, 469])
            out[2] += int(iou(b, r["bb"]) < 0.5)
    return out


def masks():
    root = Path("maps/movers-v1")
    ms = json.load(open(root / "remove-index.json"))
    D = json.load(open(root / "decisions.json"))["rows"]
    by = {(r[0], r[1], r[2]): r for r in D}
    views = {}
    cs = load_rig()
    coverage = {}
    for a, b in [(105, 140), (286, 330), (190, 215)]:
        active = 0
        produced = 0
        bad = []
        stats = ms["windows"][f"{a}-{b}"]
        for c, cam in cs.items():
            for n, t in clock(c).items():
                if a <= t < b:
                    active += 1
                    p = root / "remove" / c / f"{n:06}.png"
                    if not p.is_file():
                        bad.append([c, n])
                        continue
                    with Image.open(p) as im:
                        assert im.size == (cam.width, cam.height) and im.mode == "1"
                    produced += 1
        coverage[f"{a}-{b}"] = dict(
            native_exposures=active,
            files=produced,
            missing=bad,
            nonempty=sum(v["counts"]["nonempty"] for v in stats.values()),
            observed=sum(v["counts"]["key"] for v in stats.values()),
            propagated=sum(v["counts"]["propagated"] for v in stats.values()),
        )
    for sp, rr in [*truth(), pickle.load(open(W / "live-18.pickle", "rb"))[:2]]:
        ratios = []
        states = Counter()
        for q in rr:
            if not q["key"]:
                continue
            c, n, i = q["c"], q["n"], q["id"]
            p = root / "remove" / c / f"{n:06}.png"
            base = W / "extra" if sp.get("input") == "extra" else Path(".")
            obs = cv2.imread(str(base / "masks/dyn" / c / f"{n:06}.png"), -1) == i
            if not p.exists() or not obs.any():
                continue
            m = cv2.resize(
                cv2.imread(str(p), 0), (724, 469), interpolation=cv2.INTER_NEAREST
            )
            ratios.append(float(np.mean(m[obs] > 0)))
            v = by.get((c, n, i))
            states[
                v[4] if v and (len(v) < 11 or v[10] == str(base)) else "unassigned"
            ] += 1
        views[sp["name"]] = dict(
            exposures=len(ratios),
            over90pct=sum(q > 0.90 for q in ratios),
            under10pct=sum(q < 0.1 for q in ratios),
            states=dict(states),
        )
    return dict(
        windows=coverage,
        observed_instance_overlap=views,
        decisions=dict(Counter(q[4] for q in D)),
        unknown_removed=sum(q[4] == "unknown" and q[7] for q in D),
    )


def shutter_holdout():
    """Spatial propagation diagnostic: ignore the evaluated shutter's mask;
    neighbouring source decisions still come from the delayed production run.
    This is not an independent test of the moving/parked classifier.
    """
    from .removal import GRID, matte, warp

    root = Path("maps/movers-v1")
    rows = json.load(open(root / "decisions.json"))["rows"]
    keep = defaultdict(set)
    for c, n, mid, uid, state, conf, ctx, d, *_ in rows:
        if d:
            keep[c, n].add(mid)
    sources = {}
    ledger = root / "sources.json"
    ledger = ledger if ledger.is_file() else W / "photo-full-native/mask-sources.json"
    for c, v in json.load(open(ledger)).items():
        roots = {int(n): base for n, base in v.items()}
        sources[c] = (roots, np.array(sorted(roots)))
    out = {}
    for sp, rr in [*truth(), pickle.load(open(W / "live-18.pickle", "rb"))[:2]]:
        cov = []
        skips = 0
        gaps = []
        for r in rr:
            if not r.get("held"):
                continue
            c, n = r["c"], r["n"]
            ts = clock(c)
            roots, nn = sources[c]
            j = np.searchsorted(nn, n)
            near = [
                int(k)
                for k in nn[max(0, j - 1) : j + 2]
                if k != n and abs(ts[k] - ts[n]) < 0.8
            ]
            if not near or n not in roots:
                skips += 1
                continue
            mass = np.zeros(GRID, np.uint8)
            for k in near:
                blob = matte(c, k, roots[k], tuple(sorted(keep[c, k])))
                prev = k
                mid = (
                    sorted((u for u in ts if min(n, k) < u < max(n, k)), reverse=k > n)[
                        3::4
                    ]
                    if abs(ts[k] - ts[n]) > 0.17
                    else []
                )
                for u in [*mid, n]:
                    blob = warp(c, u, prev, blob)
                    prev = u
                mass |= blob
            mass = cv2.dilate(
                mass, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            )
            obs = (
                cv2.imread(
                    str(
                        (W / "extra" if sp.get("input") == "extra" else Path("."))
                        / "masks/dyn"
                        / c
                        / f"{n:06}.png"
                    ),
                    -1,
                )
                == r["id"]
            )
            if obs.any():
                cov.append(float(np.mean(mass[obs] > 0)))
                gaps.append(min(abs(ts[k] - ts[n]) for k in near))
        out[sp["name"]] = dict(
            n=len(cov),
            over90pct=sum(v > 0.90 for v in cov),
            under10pct=sum(v < 0.10 for v in cov),
            median_coverage=float(np.median(cov)) if cov else None,
            skipped=skips,
            nearest_anchor_p50_s=float(np.median(gaps)) if gaps else None,
        )
    (R / "report/mask-propagation.json").write_text(json.dumps(out, indent=2))
    return out


def reviewed_native():
    from .tracking import Factors, make_track, released_rows, playback
    from .model import bake
    from tempfile import TemporaryDirectory

    cs = load_rig()
    tr = Trajectory("poses/ego_causal.npz")
    result = {}
    # Independent baseline run: same frozen image panel and corrected shutter
    # clock. Refit reviewed trajectories and require a released photographed PLY.
    with TemporaryDirectory(dir=R) as tmp:
        for sp, rr in truth():
            rr = released_rows([correct(v) for v in rr if v["n"] in clock(v["c"])])
            f = Factors([q for q in rr if not q.get("held")], sp["class"], "live")
            fit = make_track(f, "live")
            poses = None
            lo = 1e10
            hi = -1e10
            if fit:
                poses, sc, st, scan = fit
                size = f.size * sc
                data = sp, rr, poses, size, st, scan
                try:
                    m = bake(data, "live", Path(tmp))
                    poses = playback(poses, "live")
                    lo = max(float(poses["t"][0]), m["first_photo"])
                    hi = poses["t"][-1]
                    p = silhouette(sp["class"], size)
                except ValueError as e:
                    if "no photographs" not in str(e):
                        raise
            counts = [0, 0, 0]
            for r in rr:
                if not r.get("held"):
                    continue
                counts[0] += 1
                if r["rt"] < lo or r["rt"] > hi:
                    counts[1] += 1
                    continue
                uv = projection(cs[r["c"]], tr, r["t"], poses, p, p * 0)[0]
                b = np.clip(
                    np.r_[uv.min(0), uv.max(0)], [0, 0, 0, 0], [724, 469, 724, 469]
                )
                counts[2] += int(iou(b, r["bb"]) < 0.5)
            result[sp["name"]] = counts
    out = dict(total=np.sum(list(result.values()), axis=0).tolist(), episodes=result)
    (R / "report/reviewed-refit.json").write_text(json.dumps(out, indent=2))
    print("reviewed native refit", out["total"])


def prefix():
    from .handoff import associate, robust_class
    from .filter import refine
    from .motion import release
    from .semantics import release as semrelease
    from .footprint import classify
    from .image_lock import steer

    packets = pickle.load(open(R / "packets.pickle", "rb"))
    allh = pickle.load(open(R / "identities.pickle", "rb"))
    ds = {d[0]["id"]: d for d in pickle.load(open(R / "online.pickle", "rb"))}
    flow = pickle.load(open(R / "motion.pickle", "rb"))
    sem = pickle.load(open(R / "semantics.pickle", "rb"))
    out = {}
    sig = lambda r: (r["c"], r["n"], r["id"], r["release"])
    for cut, token in [
        (131.85, ("left_pillar", 202)),
        (134.48, ("left_repeater", 565)),
        (322.82, ("left_repeater", 1466)),
    ]:
        partial = associate([p for p in packets if p[0] <= cut], photo=True)
        expected = {
            h["id"]: [sig(r) for r in h["rows"] if r["release"] <= cut] for h in allh
        }
        count = 0
        for h in partial:
            assert [sig(r) for r in h["rows"]] == expected[h["id"]]
            count += len(h["rows"])
        h = next(
            (
                h
                for h in partial
                if any((r["c"], r["id"]) == token for r in h["rows"]) and h["id"] in ds
            ),
            None,
        )
        v = dict(assignments=count, token=token)
        out[str(cut)] = v
        if h is None or h["confirm"] is None:
            continue
        uid = h["id"]
        verified = semrelease([x for x in sem.get(uid, []) if x[0] <= cut], h["cls"])
        v["primary_id_at_cut"] = uid
        if verified is None:
            continue
        t = max(verified, h["confirm"])
        hh = dict(
            h, confirm=t, cls=robust_class([r for r in h["rows"] if r["release"] <= t])
        )
        flow_at = release(flow.get(uid, []))
        trial = refine(
            hh,
            True,
            flow_at if flow_at is not None and flow_at <= cut else None,
            classify(h),
        )
        if trial is None:
            continue
        a = steer(trial)[2]
        b = ds[uid][2]
        chosen = a["t"] <= cut + 1 / 36
        ids = np.searchsorted(b["t"], a["t"][chosen])
        assert np.array_equal(a["t"][chosen], b["t"][ids])
        err = max(
            float(np.max(abs(a[key][chosen] - b[key][ids])))
            for key in ("x", "y", "z", "yaw")
        )
        assert err < 1e-5
        v.update(pose_knots=len(ids), max_numeric_error=err)
    (R / "report/native-prefix.json").write_text(json.dumps(out, indent=2))
    print("prefix", out)


def scenes():
    from .model import FIELDS

    out = {}
    for scene in [
        "live-v5-movers-v4",
        "live-v5-movers-v4-held",
        "stitched-opt-v3-movers-photo-v1",
        "live-v5-traffic-photo-hindsight",
    ]:
        root = Path("scene") / scene
        idx = json.load(open(root / "rsb/index.json"))
        mp = json.load(open(root / "map.json"))
        objs = json.load(open(root / "objects.json"))["objects"]
        assert idx["objects"] and all(
            (root / t["file"]).exists()
            and (root / "rsb" / idx["tiles"][t["id"]]["file"]).exists()
            for t in mp["tiles"]
        )
        maxs = {}
        ply = 0
        dense = 0
        for o in objs:
            tr = o["track"]
            X = np.stack([tr[k] for k in ("x", "y", "z")], 1)
            maxs[o["id"]] = float(np.linalg.norm(np.diff(X, axis=0), axis=1).max())
            assert np.max(abs(np.diff(tr["t"]) - 1 / 36)) < 2e-6
            assert o["contact_shadow"] and o["photometry"]
            for k in ("file", "predicted_file"):
                header, buf = (
                    (root / o[k]).read_bytes().split(b"end_header" + bytes([10]), 1)
                )
                n = int(header.split(b"element vertex ")[1].splitlines()[0])
                a = np.frombuffer(buf, "<f4").reshape(-1, len(FIELDS))
                assert (
                    len(a) == n
                    and np.isfinite(a).all()
                    and (a[:, 18] >= a[:, 17]).all()
                )
                ply += 1
                dense += int((a[:, 9] > 4.1).sum()) if k == "file" else 0
        if "movers-v4" in scene:
            assert max(maxs.values()) < 1.1
        clips = (
            json.load(open(root / "generated.json"))["clips"]
            if (root / "generated.json").is_file()
            else []
        )
        assert all((root / q["file"]).is_file() for q in clips)
        out[scene] = dict(
            objects=len(objs),
            ply=ply,
            dense=dense,
            tiles=len(mp["tiles"]),
            generated=clips,
            photo=idx.get("photo"),
            max_step_m=max(maxs.values()),
            max_step_by_id=maxs,
        )
        print(scene, len(objs), len(mp["tiles"]), max(maxs.values()), dense, flush=True)
    (R / "report/scene-audit.json").write_text(json.dumps(out, indent=2))


def run():
    out = R / "report"
    out.mkdir(exist_ok=True)
    prior = timing("component", "live-v5-movers-v3", False)
    new = timing("photo-held-native", "live-v5-movers-v4-held", True)
    result = dict(previous=prior, new=new, hindsight_native=hind(), removal=masks())
    (out / "task3-audit.json").write_text(json.dumps(result, indent=2))
    print(
        "birth latency",
        prior["median_s"],
        new["median_s"],
        new["p90_s"],
        "hindsight",
        result["hindsight_native"],
        "masks",
        result["removal"]["windows"],
    )


if __name__ == "__main__":
    run()
