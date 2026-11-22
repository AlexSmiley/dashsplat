"""Reproducible evidence tables for the exported scenes."""

from .proof import *
from .audit import stats, BANDS


def prefix_compare(cut=101):
    # Rerun the same native fusion after physically withholding the tail from its
    # input queue. Compare the geometry, colour, opacity, and all births before cut.
    def records(root):
        a = []
        for tile in json.loads((root / "map.json").read_text())["tiles"]:
            if tile["layer"] != "objects":
                continue
            x = reader(root / tile["file"])
            x = x[x[:, 17] < cut, :].copy()
            a.append(x)
        a = np.concatenate(a)
        i = np.lexsort(
            tuple(
                a[:, j]
                for j in [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 3, 4, 5, 2, 1, 0]
            )
        )
        return a[i]

    a = records(RS / "scene/live-v3")
    b = records(W / "stages/live-prefix")
    assert a.shape == b.shape and np.array_equal(a[:, :18], b[:, :18]), (len(a), len(b))
    edge = cut - 0.5
    keep = a[:, 17] < edge
    tim = np.minimum(a[keep, 18:], edge)
    pref = np.minimum(b[keep, 18:], edge)
    assert np.array_equal(tim, pref), np.max(abs(tim - pref))
    return dict(
        cut=cut,
        records=len(b),
        max_difference=0,
        timeline_records=int(keep.sum()),
        timeline_until=edge,
    )


def fmt(z):
    return "—" if not z["n"] else f"{z['median']:+.2f} / {z['scatter']:.2f} ({z['n']})"


def track_transfer(mode, c, policy, anc):
    # Lift a distant sample and predict a later, nearer native image coordinate.
    # Target is the actual measured LK pixel, never DA3 or an in-sample render.
    h = np.load(RS / "work/poses/hold.npz")
    off = len(np.load(RS / "work/poses/offline-fit.npz")["n"])
    schedule = np.load(RS / "work/poses/schedule.npz")
    ids = np.repeat(np.arange(len(h["num"])) + off, h["num"])
    ci = NAMES.index(c)
    tt = schedule["tc"][h["k"], h["c"]]
    ix = (h["c"] == ci) & (tt >= 80) & (tt <= 141) & (ids % 3 == 0)
    ids, tt, uv = ids[ix], tt[ix], h["uv"][ix]
    C = rig[c]
    tr = traj(mode)
    lt = anc[(anc[:, 5] == 1) & (anc[:, 6].astype(int) % 3 == 0)]
    p = frames()[c]
    obs = landmarks_cache()
    out = {}
    for zone, t0, t1 in [("turn", 85, 110), ("straight", 130, 140)]:
        by = {k: [] for k in BANDS}
        done = set()
        for k in sorted(np.unique(lt[:, 0]).astype(int)):
            e = p[k]
            t = e["t"]
            n = e["n"]
            if not t0 <= t < t1:
                continue
            sub = lt[lt[:, 0] == k]
            cw = rows(mode, c, t)[0]
            refcw = rows("stitched", c, t)[0]
            pol = policy[str(n)]
            for z in sub:
                ident = int(z[6])
                band = next((band for band in BANDS if band[0] <= z[2] < band[1]), None)
                if band is None or (ident, band) in done:
                    continue
                oo = obs.get(ident)
                ii = np.flatnonzero(
                    (ids == ident) & (tt > t + 0.20) & (tt < min(t1, t + 5.0))
                )
                if oo is None or not len(ii):
                    continue
                pu, ok, v, j = projected(c, oo[None, :], refcw)
                ray = C.unproject(pu)[0]
                src = cw[j[0]]
                pred = z[1] / (pol["a"] + pol["b"] * z[1])
                point = src[:3, 3] + src[:3, :3] @ ray * pred
                target_t = C.row_time(tt[ii], uv[ii, 1])
                T = tr.T_world_ego(target_t) @ C.T_ego_cam
                Tref = traj("stitched").T_world_ego(target_t) @ C.T_ego_cam
                rs = np.linalg.norm(oo - Tref[:, :3, 3], axis=1)
                v2 = np.einsum("nji,nj->ni", T[:, :3, :3], point - T[:, :3, 3])
                predpix, visible = C.project(v2)
                keep = (
                    (rs < 0.85 * z[2])
                    & (rs > 2)
                    & visible
                    & (v2[:, 2] > 0)
                    & (np.linalg.norm(T[:, :3, 3] - src[:3, 3], axis=1) > 1)
                )
                if not np.any(keep):
                    continue
                i = np.flatnonzero(keep)[np.argmin(rs[keep])]
                res = np.linalg.norm(predpix[i] - uv[ii[i]])
                by[band].append(float(res))
                done.add((ident, band))
        out[zone] = {f"{a}-{b}": stats(np.asarray(v)) for (a, b), v in by.items()}
        out[zone]["all"] = stats(np.array([z for v in by.values() for z in v]))
    return out


@lru_cache(None)
def landmarks_cache():
    a = np.load(W / "bundle-observations-v2.npz")["obs"]
    a = a[(a[:, 7] == 1) & (a[:, 2].astype(int) % 3 == 0)]
    return {int(x[2]): x[3:6] for x in a}


def feature_probes():
    # Manually marked, single-surface interiors on actual photos; separate from scoring.
    examples = [
        ("turn wall", "front", 3314, (0.64, 0.78, 0.408, 0.46)),
        ("straight wall", "right_pillar", 4875, (0.222, 0.315, 0.301, 0.365)),
        ("turn antenna", "front", 3314, (0.253, 0.262, 0.176, 0.310)),
        ("straight post", "right_pillar", 4875, (0.175, 0.187, 0.140, 0.268)),
    ]
    out = {}
    for mode in ("live", "stitched"):
        scene0 = Scene(mode)
        raw = np.memmap(W / mode / "final.vertices", "f", mode="r").reshape(-1, 23)
        meta = np.memmap(W / mode / "final.vertices.meta", "B", mode="r").reshape(-1, 4)
        for name, c, n, roi in examples:
            e = next(e for e in frames()[c] if e["n"] == n)
            t = e["t"]
            p = traj(mode).T_world_ego(t)[:3, 3]
            allp = scene0.local(t, p)
            obj = allp[allp[:, -1] == 1]
            cam = rig[c]
            cw = rows(mode, c, t)[0]

            def rays(xyz):
                uv, ok, v, _ = projected(c, xyz, cw)
                x, y = (uv / [cam.width, cam.height]).T
                lo, hi, top, bot = roi
                select = ok & (x >= lo) & (x < hi) & (y >= top) & (y < bot)
                bins = (
                    np.floor((x[select] - lo) / (hi - lo) * 3)
                    + 3 * np.floor((y[select] - top) / (bot - top) * 3)
                ).astype(int)
                return bins, np.linalg.norm(v[select], axis=1)

            b, r = rays(obj[:, :3])
            cells = double = 0
            for j in range(9):
                z = np.sort(r[b == j])
                cells += len(z) >= 4
                if len(z) >= 8:
                    gaps = np.diff(z) / np.maximum(z[:-1], 1)
                    double += any(gaps[i] > 0.20 for i in range(2, len(gaps) - 2))
            count = None
            if mode == "live" and "wall" not in name:
                sel = (
                    (raw[:, 17] <= t)
                    & (raw[:, 17] > t - 6)
                    & ((meta[:, 3] >> 1) == 1)
                    & (np.sum((raw[:, :2] - p[:2]) ** 2, axis=1) < 80**2)
                )
                _, ret = rays(raw[sel, :3])
                dep = maps(mode, c, n)[0]
                lo, hi, top, bot = roi
                dr = dep[int(H * top) : int(H * bot), int(B * lo) : int(B * hi)]
                dr = dr[dr > 2]
                count = (
                    int(np.sum(abs(ret / np.median(dr) - 1) < 0.20))
                    if len(dr)
                    else None
                )
            out[mode + " " + name] = dict(
                frame=n,
                lens=c,
                normalised_roi=roi,
                occupied_cells=int(cells),
                separated_cells=int(double),
                matching_recent_contradiction_retirements=count,
            )
    return out


def write():
    d = {
        m: json.loads((W / m / "final-summary.json").read_text())
        for m in ("live", "stitched")
    }
    proof = json.loads((W / "proof.json").read_text())
    prefix = prefix_compare()
    full = {}
    tables = []
    gates = []
    pars = []
    rims = []
    transfer = []
    plan = frames()
    for mode in d:
        audit = json.loads((W / mode / "audit.json").read_text())
        full[mode] = {}
        bundle = []
        held = []
        road = []
        for c in CAMS:
            policy = json.loads(
                (
                    W
                    / mode
                    / (
                        c
                        + ("-causal-policy.json" if mode == "live" else "-policy.json")
                    )
                ).read_text()
            )
            anc = np.load(W / mode / (c + "-anchors.npz"))
            l = anc["landmarks"]
            r = anc["road"]
            p = plan[c]
            test = l[(l[:, 5] == 1) & (l[:, 6].astype(int) % 3 == 0)]
            bu = l[(l[:, 5] == 0) & (l[:, 7] == 1)]
            rr = r[r[:, 5] == 1]
            measure = {}

            def corr(arr):
                coeff = np.array(
                    [
                        [policy[str(p[int(i)]["n"])][x] for x in ("a", "b")]
                        for i in arr[:, 0]
                    ]
                ).reshape(-1, 2)
                return arr[:, 1] / (coeff[:, 0] + coeff[:, 1] * arr[:, 1])

            pc, bc, rc = map(corr, (test, bu, rr))
            bands = []
            bvals = []
            hvals = []
            rvals = []
            for lo, hi in BANDS:
                ix = (test[:, 2] >= lo) & (test[:, 2] < hi)
                bx = (bu[:, 2] >= lo) & (bu[:, 2] < hi)
                gx = (rr[:, 2] >= lo) & (rr[:, 2] < hi)
                item = dict(
                    bundle_raw=stats((bu[:, 1] - bu[:, 2])[bx]),
                    bundle=stats((bc - bu[:, 2])[bx]),
                    held_raw=stats((test[:, 1] - test[:, 2])[ix]),
                    held=stats((pc - test[:, 2])[ix]),
                    road_raw=stats(rr[gx, 3]),
                    road=stats((rr[:, 3] + (rc - rr[:, 1]) * rr[:, 4])[gx]),
                )
                measure[f"{lo}-{hi}"] = item
                bvals.append(fmt(item["bundle"]))
                hvals.append(fmt(item["held"]))
                rvals.append(fmt(item["road"]))
            tn = track_transfer(mode, c, policy, l)
            measure["far_to_near_px"] = tn
            transfer.append(
                "| "
                + mode
                + " "
                + c
                + " | "
                + " | ".join(fmt(tn[k]["all"]) for k in ["turn", "straight"])
                + " |"
            )
            rs = audit[c].get("rim_start", 65.0)
            ix = test[:, 3] >= rs
            z = stats((pc / test[:, 2] - 1)[ix])
            measure["outer_rim_frac"] = z
            measure["rim_start_deg"] = rs
            rims.append(f"| {mode} {c} | {rs:.1f}° | {fmt(z)} |")
            full[mode][c] = measure
            row = lambda vals: "| " + c + " | " + " | ".join(vals) + " |"
            bundle.append(row(bvals))
            held.append(row(hvals))
            road.append(row(rvals))
            ranges = [f"{lo}–{hi}" for lo, hi in audit[c]["bands_admitted"]]
            ap = [
                p0
                for key, p0 in policy.items()
                if any(
                    x["n"] == int(key) and x["train"] and x["t_key"] >= 85 for x in p
                )
            ]
            fraction = [
                sum([lo, hi] in v["bands_admitted"] for v in ap) / len(ap)
                for lo, hi in BANDS
            ]
            gates.append(
                "| "
                + mode
                + " "
                + c
                + " | "
                + " / ".join(f"{100 * f:.0f}%" for f in fraction)
                + f" | {round(float(np.median([v['max_angle'] for v in ap])), 1)}° |"
            )
            pars.append(
                "| "
                + mode
                + " "
                + c
                + " | "
                + " / ".join(f"{np.median([v[k] for v in ap]):.5f}" for k in ["a", "b"])
                + " |"
            )
        for label, entries in [
            ("BA landmarks on reserved captures", bundle),
            ("Tracks never used for pose fitting, correction or band selection", held),
            ("Road: signed height above fitted surface", road),
        ]:
            tables += [
                "",
                f"### {mode.capitalize()}: {label}",
                "",
                "| Camera | 2–8 m | 8–15 m | 15–25 m | 25–40 m | 40–70 m |",
                "|---|---:|---:|---:|---:|---:|",
                *entries,
            ]
    summary = [
        "# Static object layer: 85–140 s",
        "",
        "Both scenes retain the whole-drive ground v2. **Objects are a sampled slice, not whole-drive coverage.** Dark openings in non-sky regions are gaps in the stored geometry; sky itself is intentionally blank. The first and last scheduled reference packet times are %.3f and %.3f s."
        % tuple(d["live"]["actual"]),
        "",
        "| Variant | Object records | Tiles | Object PLY bytes | Fusion (s) | Camera frames/s | Distance-scaled whole drive |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for m, x in d.items():
        summary.append(
            f"| {m} | {x['splats']:,} | {x['tiles']} | {x['object_ply_bytes']:,} | {x['fusion_seconds']:.2f} | {x['fps']:.1f} | {x['projected_whole_drive']:,.0f} |"
        )
    summary += [
        "",
        "All 838 selected training packets per variant were processed. Live used released evidence from 85.701–139.808 s (maximum capture-to-release lag 0.389 s). The whole-drive object budgets are 40M live / 20M stitched. Distance extrapolation is ×%.2f, not a whole-drive benchmark. Timings include JPEG/depth/mask I/O, row poses, radiometry, footprint preparation and native fusion; they exclude depth inference, audit, PLY export and proof. Ground is hard-linked and occupies no additional disk blocks."
        % d["live"]["travel_factor"],
        "",
        "## Held-out photographs",
        "",
        "Each plate is the unmodified native photo, live render, stitched render. Every scored physical frame was excluded as a fusion target, from all same-lens training depth windows and from the frozen ground schedule. Fixed pre-existing camera, trajectory and radiometry calibration are shared. Every lens has %d scored frames; three are shown per plate. The additional storefront row is an object-only holdout that ground v2 saw; it is labelled and excluded from scores. No sky, car model or image backplate is composited. Geometry is evaluated at physical row times. The raw-photo PSNR penalises missing static non-road pixels as background; matched MAE and coverage are reported together so a sparse layer cannot win by staying blank."
        % proof["cameras"]["front"]["live"]["frames"],
        "",
        "| Camera | Ground → v3 PSNR gain dB L/S | Non-road PSNR dB L/S | Non-road object coverage L/S | Covered MAE L/S (RGB levels) | Depth too near L/S | Sky overlap L/S |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for c in CAMS:
        m = proof["cameras"][c]
        pair = lambda key, k=1: " / ".join(
            f"{k * m[n][key]:.1f}" for n in ("live", "stitched")
        )
        summary += [
            f"| [{c}](report/camera-{c}.jpg) | {' / '.join(format(m[q]['nonground_psnr'] - m[q]['ground_only_nonground_psnr'], '.2f') for q in ('live', 'stitched'))} | {pair('nonground_psnr')} | {pair('nonground_object_coverage', 100)}% | {pair('matched_object_mae')} | {pair('render_too_near', 100)}% | {pair('sky_silhouette_overlap', 100)}% |"
        ]
    summary += [
        "",
        "“Too near” compares the depth-sorted alpha-weighted range to **held-out DA3**, only where an object was drawn and valid reference range was 2–40 m. A violation is >20% and >0.5 m too near. It is a symptom, not independent truth. Sky overlap is rendered object coverage on held-out sky labels; sky was never a depth or free-space anchor. Raw and photometrically corrected MAEs, all-static PSNR and both depth tails are in the measurements.",
        "",
        "## Depth qualification",
        "",
        "Depth is physical optical-axis z. Divide by the original distorted grid’s unit ray z **before** comparing or fitting. The 59,665-point BA is distinct from DA3; its poses were inputs to DA3, its landmark positions were not. Transform original BA xyz through its original row camera into the v2 road gauge. For the second check, triangulate reserved tracks from their row-timed rays; exclude weak perpendicular baselines, high angular residual, dynamic/sky/edge pixels and extrapolation without a nearby observed track. Road checks intersect the separately fitted suspension/road tangent. Counts are correlated observations, not independent trials. All bands below use **reference range**; birth gates use measured/corrected range. Table entries: signed median error / robust scatter (1.4826 × MAD), metres, followed by N. Raw uncorrected tables and p90 absolute errors are in [measurements](measurements.json).",
        "",
        "| Camera | Median applied a / b (1/m for b) |",
        "|---|---:|",
        *pars,
        "",
        "Live uses only earlier measured road and raw LK prefixes triangulated in the causal gauge. Its gates also use the current released prefix; an untested band can make *proposals* but needs independent confirmation. Stitched fits a regularised affine inverse-range model, retains identity if validation gain is under 1.5%, and quarantines local packets where separate bundle anchors disagree. Rejected calibration candidates are not used.",
        "",
        "| Lens | Admitted packets by measured band: 2–8 / 8–15 / 15–25 / 25–40 / 40–70 m | Angular cap |",
        "|---|---:|---:|",
        *gates,
        "",
        "### Lens periphery",
        "",
        "Outer 15% of the calibrated angular grid (rear: ≥65°); signed fractional range error, scatter and N on the separate test tracks. Includes unsupported distances. An empty cell is unmeasured, not a zero error.",
        "",
        "| Lens | Rim starts | Fractional bias / scatter (N) |",
        "|---|---:|---:|",
        *rims,
        "",
        "### Far-to-near reprojection",
        "",
        "A distant DA3 observation on a reserved track is lifted with the applied calibration, then projected onto a later *measured* feature in the same physical camera (native pixels). Require ≥1 m baseline and ≥15% nearer range; select at most one pair per track and source band. Turn: 85–110 s. Fast straight: 130–140 s, ≈21 m/s. A dash means that lens had no qualifying pair. Per-band values, counts and p90 are in measurements.json.",
        "",
        "| Lens | Turn median / scatter (N), px | Straight median / scatter (N), px |",
        "|---|---:|---:|",
        *transfer,
        *tables,
        "",
        "## Playback and visual evidence",
        "",
        f"Prefix replay: {prefix['records']:,} live records born before 101 s reproduced float-for-float in XYZ, normal, colour, opacity, covariance and birth after independently refitting the prefix calibration and stopping fusion at 101 s. The clipped death/visibility endpoints agree through 100.5 s (outside the visibility-run join tolerance). Both manifests and every object PLY were read back; the exact ground tiles, counts, bounds, unit normals/quaternions and visibility ordering passed.",
        "",
        "[Exit chase](report/chase-exit.jpg) · [first road](report/chase-first-road.jpg) · [arterial](report/chase-arterial.jpg). Stitched 360° orbit contact sheets: [exit](report/orbit-exit.jpg), [first road](report/orbit-first-road.jpg), [arterial](report/orbit-arterial.jpg). The corresponding 12 s, 216-frame side-by-side live/stitched videos are in $RS/maps/object-v3/orbits/.",
    ]
    quality = [
        "## What the pictures show",
        "",
        "The [93.7 s exit orbit](report/orbit-exit.jpg) is the strongest result: the planting island, shop columns, red sign and opposite wall retain their positions as the camera turns. The sign shape and tint survive; its lettering does not. On the [135 s arterial](report/chase-arterial.jpg) the stitched right verge has trees and posts, but the forward horizon is mostly missing. Live on the fast road is still essentially a ground reconstruction with scattered roadside objects. The slice demonstrates static memory, not complete street geometry.",
        "",
        "- **Façades and poles.** At 92.7 s the roofline has gaps, and some of the opposite wall occupies two ranges. At 136.1 s the right-pillar photo has one thick pole; stitched has a displaced second outline. The 117.5 s utility pole is missing almost entirely in both. The spot checks below distinguish holes from duplicated depth and found no matching recent live contradiction retirement at the two sampled pole cores. Thin features lose their valid interiors under silhouette erosion; remaining estimates move laterally when range is wrong.",
        "- **Foliage and terrain.** Canopies have thickness and survive edge-on, but the 60° exit and 240° first-road orbit show smeared branches and vegetation floating above gaps at the kerb. Thickness helps visibility, not positional accuracy. Grass and coarse terrain are deliberately sparse where neither detector nor ground fit supports them. Sky input pixels were excluded; splats drawn over held-out sky are silhouette errors (21% on stitched right pillar).",
        "- **Lens decisions.** The stitched right repeater had zero independently supported bands, hence zero source records; other lenses still render in that viewpoint. Stitched rear births are limited to locally qualified 2–8 m packets inside 65°. On separate test tracks the outer rear under-ranges by 58% median (live: 61%). Short-range evidence and safe subjects are preferable to filling these gaps with wrongly placed geometry.",
        "- **Scaling decision.** Object counts and CPU fusion speed leave ample headroom. The limiting factor is independent depth agreement. Prioritise side-camera/edge geometry and fast-road capture density before expanding this configuration to the whole drive. The frozen v2 ground and its existing exit stripe are unchanged.",
        "",
    ]
    at = summary.index("## Held-out photographs")
    summary[at:at] = quality
    coverage = [
        "",
        "## What is actually present",
        "",
        "| Source lens | Train packets L/S | First–last physical time (s) | Longest gap (s) | Exported birth records L/S |",
        "|---|---:|---:|---:|---:|",
    ]
    for c in CAMS:
        pts = [e for e in plan[c] if e["train"] and 85 <= e["t_key"] <= 140]
        times = np.asarray([e["t"] for e in pts])
        assert len(pts) == d["live"]["frames"][c] == d["stitched"]["frames"][c]
        coverage.append(
            f"| {c} | {len(pts)} / {len(pts)} | {times.min():.3f}–{times.max():.3f} | {np.diff(times).max():.2f} | {d['live']['source_records'][c]:,} / {d['stitched']['source_records'][c]:,} |"
        )
    for m in d:
        v = d[m]
        coverage.append(
            f"\n{m.capitalize()}: {v['terrain']:,} coarse terrain records; {v['volumetric']:,} thick foliage/rough patches; {v['events']['superseded']:,} supersessions, {v['events']['false_retired']:,} contradiction retirements, {v['hull_pruned']:,} removals by prior hull occupancy, {v['events']['continuations']:,} visibility continuations. Supersession and contradiction counts are fusion events, not human-labelled errors. {v['ground_ply_bytes']:,} bytes of original ground are shared by hard link."
        )
    summary[
        summary.index("## Held-out photographs") : summary.index(
            "## Held-out photographs"
        )
    ] = coverage + [""]
    probes = feature_probes()
    summary += [
        "",
        "### Two wall and two pole spot checks",
        "",
        "For each wall, nine fixed image patches lie inside the photographed façade. A double-depth flag requires at least three surfel centres per group with a >20% range gap; missing patches are not counted as correct. This identifies potential doubled surfaces, not physically surveyed walls. For poles, count live see-through retirements during the preceding 6 s whose centres reproject into the photographed pole core and agree with its nonzero reference depth within 20%; supersessions and visibility continuations are excluded. These are candidates for wrong retirement, not proof that an entire pole was retired.",
        "",
    ]
    for name in ("turn wall", "straight wall"):
        msg = [
            f"{mode}: {probes[mode + ' ' + name]['separated_cells']}/{probes[mode + ' ' + name]['occupied_cells']} populated patch(es) flagged"
            for mode in ("live", "stitched")
        ]
        summary += ["- " + name.capitalize() + " — " + "; ".join(msg) + "."]
    for name in ("turn antenna", "straight post"):
        v = probes["live " + name]["matching_recent_contradiction_retirements"]
        summary += [
            "- "
            + name.capitalize()
            + (
                " — no usable metric depth on the thin core."
                if v is None
                else f" — {v} recent compatible see-through retirement(s)."
            )
        ]
    out = RS / "pipeline/livemap"
    (out / "REPORT.md").write_text(chr(10).join(summary) + chr(10))
    (out / "measurements.json").write_text(
        json.dumps(
            dict(
                build=d, validation=proof, depth=full, prefix=prefix, spot_checks=probes
            ),
            indent=2,
        )
        + chr(10)
    )


if __name__ == "__main__":
    write()
