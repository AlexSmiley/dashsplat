"""Build the report from measured products; missing observations never count as passes."""

from .evaluate import *
from shutil import copy2


def run():
    data = json.load(open(WORK / "results.json"))
    checks = json.load(open(WORK / "checks.json"))
    hyp = json.load(open(WORK / "cal-diagnostic.json"))
    q = data["summary"]
    A = data["attitude"]
    study = PRODUCT / "debug"
    dest = RS / "pipeline/road/report"
    dest.mkdir(parents=True, exist_ok=True)
    s, d, t, B, stop = inputs()
    v = np.load(WORK / "ego.npz")
    l = np.load(WORK / "ego_causal.npz")
    spr = np.load(WORK / "spring.npz")
    windows = [(name, int(((s["t"] >= a) & (s["t"] <= b)).sum())) for name, a, b in SEG]
    fmt = lambda v, dec=2: " / ".join("—" if x is None else f"{x:.{dec}f}" for x in v)
    cell = lambda z: "—" if not z[0] else f"{fmt(z[1:], 1)} ({z[0]})"
    passed = []
    for seg, a, b in SEG:
        for n, j, k, md, p9 in data["detail"]["offline"][seg]:
            if k:
                passed.append((md < 3 and p9 < 8) if n == j else md < 5)
    S = [
        "# Road-relative attitude and the bird’s-eye test",
        "",
        f"**{sum(passed)} of {len(passed)} supported segment–camera-pair cells meet their requested thresholds** ({63 - len(passed)} unsupported). For same-camera cells both median <3 cm and p90 <8 cm must hold; for cross-camera cells median <5 cm. A singleton is reported, not treated as strong evidence. Raster resolution is 2 cm, not an accuracy claim.",
        "",
        "## Ground registration",
        "",
        f"Every scheduled keyframe of all six cameras entered the three mosaics: {', '.join(f'{n}: {k} packets' for n, k in windows)}. There are {data['matches'][0]:,} accepted photographic pair observations over the drive; {data['matches'][1]:,} were reserved by a deterministic hash of 1 m ground cells before fitting either trajectory. Neighbouring patches may share texture; the observations are not independent statistical trials. Same-camera repeats must be ≥1 s **or** ≥5 m apart; cross-camera pairs can be simultaneous. Both keyframes must be inside the labelled segment. Statistics below use both observations inside 10 m on the planar reference and connected pavement in that baseline. Measurements use the same native pixel correspondences under every candidate, without refitting target shifts to that candidate or gating on transfer error.",
        "",
        "Cells: **median / p90 cm (number of patch matches)**. All lenses included. V1 has no road model: the default flat-world test uses z=0. The extra V1/local-level rows give it a horizontal tangent at the instantaneous axle elevation, removing vertical datum error. The final V2 rows use its estimated road plane.",
        "",
        "|Trajectory|Park same|Park cross|Turn same|Turn cross|Straight same|Straight cross|",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, regs in q.items():
        S.append(
            "|"
            + name
            + "|"
            + "|".join(cell(z) for seg, _, _ in SEG for z in regs[seg])
            + "|"
        )
    S += [
        "",
        "### Final offline, by physical lens pair",
        "",
        "|Lenses|Park 84–105 s|Turn 316–326 s|Straight 293–310 s|",
        "|---|---:|---:|---:|",
    ]
    dd = data["detail"]["offline"]
    for j, z in enumerate(dd["park"]):
        S.append(
            "|"
            + z[0]
            + " / "
            + z[1]
            + "|"
            + "|".join(cell(dd[g][j][2:]) for g, _, _ in SEG)
            + "|"
        )
    S += [
        "",
        "The [full pair × distance × segment table](report/patch-shifts.csv) includes both versions, planar telemetry, 0–4, 4–7, 7–10 and 10–12 m bins. A lens looking away from the patch cannot supply a match. Some same-camera forward-looking fields are very narrow inside 10 m. Zero matches after the repeat-baseline, pavement, usable footprint and correlation checks means unmeasured, not perfect alignment. The fixed ±0.60 m photographic search, NCC >0.78, peak margin >0.065 and reverse error <2.4 cm select recognisable texture and can under-count larger failures. Broad untextured asphalt is not measured. Gain-normalised patches are 0.675 m across, and subpixel correlation peaks are converted back to the two original source rows. Camera/time-dependent exposure and elongated edges remain failure modes.",
        "",
        "### Failure evidence and calibration",
        "",
        "The [pair diagnostics](report/pair-diagnostics.csv) give median signed shifts in the observation heading frame, NCC and the larger endpoint’s ground displacement per native pixel. A broad aperture, foreshortened paint, raised curbs/vehicles and crown or ramps within one tangent footprint can look like a mount error. The following distributions use an independent diagnostic subsample and the deployed geometry (including unsegmented texture), so are not denominators of the table above.",
        "",
        "|Diagnostic|Park|Turn|Straight|",
        "|---|---:|---:|---:|",
    ]
    for key in ["0–4 m", "4–7 m", "7–10 m", "centre", "edge"]:
        S.append(
            "|"
            + key
            + "|"
            + "|".join(cell(hyp["distribution"][g][key]) for g, _, _ in SEG)
            + "|"
        )
    S += [
        "",
        "Camera-specific small roll/pitch/yaw, clock and row-readout shifts, and an image-radius term were fitted to training pairs, then transferred to geographically or dynamically different observations. These perturbations compete with local pavement shape and one another. This is a differential hypothesis test, not a new calibration. Native-feature RMS below uses the same fixed 3D scene, without retriangulating to disguise a change.",
        "",
        "|Trained on → evaluated on complement|Road median / p90 before (cm)|Road median / p90 after (cm)|Largest lens native RMS before → after (px)|",
        "|---|---:|---:|---:|",
    ]
    for key, vv in hyp["transfer"].items():
        nn = np.array(hyp["native"][key])
        i = np.argmax(nn[:, 1] - nn[:, 0])
        S.append(
            f"|{key} ({vv[1]} eval)|{fmt(vv[2], 1)}|{fmt(vv[3], 1)}|{NAMES[i]}: {nn[i, 0]:.2f} → {nn[i, 1]:.2f}|"
        )
    S += [
        "",
        "[Single-parameter checks](report/single-parameter-check.csv) separately test clock, readout, each mounting rotation and a radius-dependent warp for each lens on an unseen half of the drive. No calibration file was rewritten. A smaller road-only residual is insufficient evidence for a physical change that damages the static scene. The two independent road halves suggest +5.8 and +6.6 ms at the front; that single correction reduces transferred road median by at least 3.8 cm but raises its static-feature RMS by as much as 4.0 px. Rear pitch changes of +0.16° / +0.17° gain about 2.1 cm but cost up to 0.34 px. Among one-parameter changes costing <0.10 px, the best replicated median improvement is just 0.12 cm. Neither the road-only clock nor these mount/optics alternatives are an identified calibration fix.",
        "",
        "## Suspension and world gauge",
        "",
        "The offline solve robustly minimises 2-D ground disagreement of normalised image patches, with weak roll/pitch/heave and XY/yaw regularisation; the acceleration intercepts and gains are nuisance variables fitted jointly. The 6-DoF static-feature Ceres solve then retains all original accepted observations, physical odometry and baselines, with a road-state prior on each knot and the same training patch centres as explicit row-timed ray/plane factors. All landmarks are initialised in the new local rigid frame and are free to adjust. The Cauchy scale for the retained scene observations is 4.5 px; road factors use 1.8 cm scale with robust tails. The live solve updates suspension and planar corrections against current and earlier patches only; it learns its gain coefficients forward in time, above the V1 causal landmark filter. Both solutions share the same camera geometry and body mask. No offline trajectory or future-route gain is fed into the live filter.",
        "",
        "Road shape and absolute vertical datum are weakly observable. The offline estimator filters V1 scene height and attitude-minus-suspension over 31–37 m, integrates longitudinal grade with a low-frequency height anchor and centres the distance-weighted mean world normal. These priors choose a gauge; pavement texture independently measures the **relative** rotation/heave, not a surveyed crossfall. The causal estimator integrates a 29–46 m forward-only grade/bank estimate with a running mean normal. It cannot know the exact future whole-route normal. The plane routine returns n,h for n·X_ego+h=0. Height is normal to the road, all angles are in heading-aligned coordinates.",
        "",
        "|Motion statistic|Offline|Causal|",
        "|---|---:|---:|",
        f"|Gain: pitch / roll (° per m/s²)|{fmt(data['gains'], 3)}|{fmt(data['live_gain'], 3)} (last online fit)|",
        f"|Correlation: pitch with −ay / roll with −ax|{fmt(A['offline']['cor'], 3)}|{fmt(A['causal']['cor'], 3)}|",
        f"|p95 absolute world pitch / roll (°)|{fmt(A['offline']['worldp95'])}|{fmt(A['causal']['worldp95'])}|",
        f"|World surface height min / max (m)|{fmt(checks['ego']['surface_z_range'])}|{fmt(checks['ego_causal']['surface_z_range'])}|",
        f"|Path-weighted mean normal tilt (°)|{checks['ego']['mean_normal_tilt_deg']:.4f}|{checks['ego_causal']['mean_normal_tilt_deg']:.4f}|",
        f"|p95 height vs integrated grade discrepancy over 100 m (cm)|{checks['ego']['grade_integral_100m_p95_cm']:.2f}|{checks['ego_causal']['grade_integral_100m_p95_cm']:.2f}|",
    ]
    for i, key in enumerate(
        [
            "Suspension pitch (°)",
            "Suspension roll (°)",
            "Suspension heave (cm)",
            "Road grade (°)",
            "Road crossfall (°)",
        ]
    ):
        S.append(
            "|"
            + key
            + " p05 / p50 / p95|"
            + "|".join(
                fmt(np.array(A[n]["amplitudes"])[:, i]) for n in ["offline", "causal"]
            )
            + "|"
        )
    Dlt = np.array(data["delta"])
    S += [
        "",
        f"Moving-sample offline–causal absolute differences (median / p95): suspension pitch {fmt(Dlt[[0, 2], 0])}°, roll {fmt(Dlt[[0, 2], 1])}°, heave {fmt(Dlt[[0, 2], 2])} cm; total world pitch {fmt(np.array(A['world_angle_diff'])[:, 0])}°, roll {fmt(np.array(A['world_angle_diff'])[:, 1])}°. Correlations use the same 0.20 s causal-filtered acceleration as the fit, on moving samples, and are not an independent validation of the physical gains. [Updated attitude plot](report/road-attitude.png).",
        "",
        f"Both deliver 12,927 master samples and exact stationary poses and decompositions (5.67–81.83, 198.82–202.65, 257.70–286.67 s). Maximum causal lookahead including the V1 interpolation dependency and the last shutter row: {checks['lag_max_all_s']:.3f} s. Native row projection round-trip on fast/turn road pixels: {max(x for c in ['ego', 'ego_causal'] for x in checks[c]['roundtrip_max_px'].values() if x is not None):.4f} px maximum. These are internal consistency checks, not ground truth.",
        "",
        "## Whole-route scene consistency",
        "",
        "Same 649,384 native observations, same IDs, no new pruning after the change.",
        "",
        "|Camera|Observations|V1 RMS px|V2 RMS px|V2 p95 px|",
        "|---|---:|---:|---:|---:|",
    ]
    v1 = json.load(open(PW / "metrics.json"))["repro"]
    for (n, count, rms, p9), old in zip(data["feature"], v1):
        S.append(f"|{n}|{count:,}|{old[2]:.3f}|{rms:.3f}|{p9:.3f}|")
    S += [
        "",
        "## Visual comparisons",
        "",
        "Each five-column crop uses the same time and ego-relative offset (accounts for different global XY gauges), identical metric scale and no image-based final registration. At 101.19 s the car-park exit arrow no longer breaks into two displaced pieces as it does under offline V1. At 318.62 s the ONLY letters regain their expected size and a recognisable outline. The arrow at 87.86 s and the stop bar at 297.67 s still show camera-boundary steps and exposure bands; not every marking is single-edged. Inspection also found square holes cut out of observed paint by the colour classifier. Small enclosed neutral-colour islands in the display now retain their original best-view pixels; nothing is inpainted or registered after the fact. The stricter road-mask.png is retained for all evaluation. The best-view raster can hide disagreement between views; use the held-out shifts alongside it. The first/last maps and full mosaics remain on Studio.",
    ]
    S.insert(
        S.index("## Visual comparisons"),
        "The front RMS increase is material: 146 observations exceed 10 px (V1: four), and 80% of the increase in squared error comes from the worst 1% of the V2 front observations. The 100–110 s exit accounts for 56% of the net front increase; 150–160 and 320–330 s also contribute. No observation was removed to improve this number.",
    )
    S.insert(
        S.index("## Whole-route scene consistency"),
        "Across non-overlapping 100 m chords, absolute modelled road elevation change has median/p95 9/56 cm offline and 15/46 cm causal; the largest offline chord is 86 cm. The absolute road profile is not surveyed.",
    )
    # Specific parked-roll and heading-change sanity checks requested by the coordinator.
    jj = S.index("## Visual comparisons")
    v1pose = Trajectory(OUT / "ego_v1.npz").poses
    rr1 = angles(v1pose[:, :3, :3])[2]
    rr2 = angles(v["T_world_ego"][:, :3, :3])[2]
    rrl = angles(l["T_world_ego"][:, :3, :3])[2]
    lines = [
        "## Parked roll and the heading change near 150 s",
        "",
        "Median by interval, degrees. A change in absolute bank is not evidence of suspension motion while stationary.",
        "",
        "|Window (s)|V1 world roll|V2 world roll|Causal V2 world roll|V2 road crossfall|V2 suspension roll|",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for lo, hi in [(6, 81), (137, 144), (153, 160), (199, 202), (258, 286)]:
        ii = (t >= lo) & (t <= hi)
        lines.append(
            f"|{lo}–{hi}|"
            + "|".join(
                f"{np.rad2deg(np.median(z[ii])):+.2f}"
                for z in [rr1, rr2, rrl, v["road_roll"], v["susp_roll"]]
            )
            + "|"
        )
    S[jj:jj] = lines + [""]
    S.insert(
        S.index("## Whole-route scene consistency"),
        "Road support is weakest at 0–1.64, 81.86–84.16 and 286.70–287.84 s (departures), and 334.34–335.34 s. Bit 8 marks 455 of 9,028 moving master samples with fewer than four training pairs in the preceding 350 ms. The visible roll sign change across the heading change near 150 s is smaller than V1 but not eliminated; its division into road bank and body roll is prior-dependent.",
    )
    for p in sorted(study.glob("compare-*.jpg")):
        S.append(f"- [{p.stem}](report/{p.name})")
    for p in list(study.glob("compare-*.jpg")) + [
        study / "road-attitude.png",
        study / "patch-shifts.csv",
        study / "pair-diagnostics.csv",
        study / "single-parameter-check.csv",
    ]:
        copy2(p, dest / p.name)
    text = chr(10).join(S) + chr(10)
    for header in (
        "## Whole-route scene consistency",
        "## Parked roll and the heading change near 150 s",
    ):
        text = text.replace(chr(10) + header, chr(10) * 2 + header)
    (RS / "pipeline/road/REPORT.md").write_text(text)
    (PRODUCT / "REPORT.md").write_text(text.replace("](report/", "](debug/"))
    print("report ready", flush=True)


if __name__ == "__main__":
    run()
