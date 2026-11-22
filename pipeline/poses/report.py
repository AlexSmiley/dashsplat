"""Assemble the report from measured outputs. Missing support is explicit."""

from .motion import *
from shutil import copy2


def main():
    m = json.load(open(W / "metrics.json"))
    road = json.load(open(W / "roadmetrics.json"))
    tim = json.load(open(W / "timing.json"))
    D = m["length"]
    fmt = lambda v: " / ".join(f"{x:.2f}" for x in v)
    st = m["stereo"]["whole"]
    lines = [
        "# Reality Splat — whole-drive ego poses",
        "",
        "**Both trajectory products are available on all 12,927 master timestamps.** They are locally image-constrained and globally tied to the fused GPS/wheel solution. Small inlier RMS does not guarantee that a point triangulated far ahead will be within a few pixels on approach. Use the separate transfer checks below.",
        "",
        "## Estimation and coordinate frame",
        "",
        f"{m['n_vision']:,} distance/time selected six-view packets, {m['n_knots']:,} independent 6-DoF offline knots, {m['n_landmarks']:,} retained static-consistent landmarks including {m['cross_inliers']:,} seen in multiple physical cameras. Bidirectional LK checks and masked GFTT corners provide within-camera continuity. RootSIFT tracks span camera and time at 888 more widely spaced rig frames. Ceres fits rolling-row individual reprojections, wheel distance and heading increments, weak absolute fused GPS and elevation, and smooth pose corrections. Robust pixel losses and independent multiview consistency reject moving tracks and mismatches; very slow traffic cannot always be distinguished geometrically. After the first solve, observations above 5 px are removed and tracks must retain ≥4 observations, >75% support and a clipped mean below 4 px. The final scores below use the exported master-clock poses and source-row timestamps, not just the solver knots.",
        "",
        "Ego is rear-axle centre on the ground, x forward/y left/z up; world is ENU with zero translation at the first sample. Inter-knot pose corrections retain high-frequency wheel shape. The shared geometry module then interpolates master poses with a constant body twist. The same pose variable represents both endpoints of each stopped plateau. Tiny encoded speeds around −2.48×10⁻⁷ m/s count as zero. No external altitude or gravity channel was available.",
        "",
        "## Whole-drive bundle fit",
        "",
        "Native-resolution Euclidean pixel residuals on retained observations. Track counts are distinct per lens; cross-camera landmarks may appear in several rows.",
        "",
        "|Camera|Tracks|Observations|RMS px|p95 px|Fast residual equivalent clock shift (ms)|",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for n, (tracks, obs, rms, p95) in zip(NAMES, m["repro"]):
        lines.append(f"|{n}|{tracks:,}|{obs:,}|{rms:.3f}|{p95:.3f}|{tim[n][1]:+.3f}|")
    lines += [
        "",
        "Clocks and readouts are unchanged. The fast-road test is the signed inlier residual along measured image velocity for speed >14 m/s. It is conditional on the joint trajectory/landmark fit and is not an independent shutter calibration. No coherent multi-millisecond shift was identified.",
        "",
        "## Long-range transfer — held out from pose estimation",
        "",
        "The natural-track split reserves IDs before either live resection or offline BA. Very few features persist from >25 m to <8 m. Supported lens results (landmarks, observations, median / p90 pixels): "
        + "; ".join(
            f"{n}: {q[0]}, {q[1]}, {q[2]:.1f} / {q[3]:.1f}"
            for n, q in zip(NAMES, m["hold"]["whole"])
            if q[0]
        )
        + ". This sparse check is not a six-camera accuracy estimate.",
        "",
        "A separate photographic check targets static road paint and pavement texture. It is independent of the offline solve: no road-check pixels or landmarks enter any pose factor. A causal-pose plane warp proposes matching places; actual positions come from image correlation with scale search, a uniqueness margin, reverse matching and a continuous sequence of accepted image matches. A search proposal is not a ground control point. No flat-road assumption is used to triangulate the final landmark. Near-only triangulation checks the range label; far-only triangulation uses ≥3 observations from the first third, each >25 m. Refine the far point against only those native source pixels (Huber loss), then score the last-third observations inside 8 m. Each end must be separately self-consistent and have positive depth. There is **no far-to-near target error gate**. Interpolation, far angular resolution, repeated paint patterns and finite patch-centering accuracy contribute to the numbers; they are not a pose-only error. Inspection of the raw audits shows repeated stripe/curb edges and far marks only a few pixels across: some associations are ambiguous, so these are end-to-end correspondence + depth + pose errors, not surveyed static-point errors.",
        "",
        "Each cell is median / p90 pixels **(landmarks, observations)**. A dash means no supported measurement; broad roadside texture is not evidence for a missing lens or segment.",
        "",
        "|Near camera|Whole sampled set|Car park 88–105 s|90° turn 317.5–325.5 s|Exit turn 105–111.5 s|Fast straight 293–310 s|",
        "|---|---|---|---|---|---|",
    ]
    regions = ["whole", "car park", "90° turn", "exit turn", "fast straight"]
    for i, n in enumerate(NAMES):
        row = [n]
        for g in regions:
            p, q, med, p90 = road["table"][g][i]
            row.append(f"{med:.1f} / {p90:.1f} ({p}, {q})" if p else "—")
        lines.append("|" + "|".join(row) + "|")
    ff = road["funnel"]
    lines += [
        "",
        f"{ff.get('appearance', 0)} deduplicated appearance tracks; {ff.get('range', 0)} met the range and endpoint-count checks, {ff.get('parallax', 0)} retained sufficient far parallax, {ff.get('separately_consistent', 0)} passed separate endpoint fits. See the [raw-image correspondence audit](report/road-transfer-audit.jpg).",
        "",
        "Paired comparison on exactly the same marks, under each trajectory. Far points are re-triangulated for each product. The causal trajectory was used to propose photographic matches; this comparison is not blind to the causal model.",
        "",
        "|Region|Offline median / p90|Planar median / p90|Causal median / p90|",
        "|---|---:|---:|---:|",
    ]
    for key in regions:
        x = road["paired"].get(key)
        lines.append(
            "|"
            + key
            + "|"
            + ("|".join(fmt([x[0][j], x[1][j]]) for j in range(3)) if x else "—|—|—")
            + "|"
        )
    lines += [
        "",
        "Do not interpret unsupported cells or good short-track RMS as a pass for few-pixel near reprojection everywhere. In particular, forward motion gives weak depth from distant, nearly collinear rays. Delay fusion or update depth after stronger parallax or another camera sees the surface.",
        "",
        (
            "Depth diagnostic on the road set: choosing the depth along a middle far ray using near observations reduces whole-set near median/p90 to "
            + fmt(road["depth_oracle"]["whole"])
            + " px. This oracle uses the answers and is not a held-out result; it separates along-ray depth weakness from other errors."
        )
        if road.get("depth_oracle", {}).get("whole")
        else "No depth-oracle statistic available.",
        "",
        "## Metric scale and global anchoring",
        "",
        f"Path length: offline {D['offline3d']:.2f} m (horizontal {D['offlineXY']:.2f}); causal {D['causal3d']:.2f} m; integrated wheel {D['wheel']:.2f} m; raw fused GPS polyline {D['gpsXY']:.2f} m. Over 33 non-overlapping ~100 m chords, wheel minus GPS chord length (p10 / median / p90) was {fmt(m['gpswheel'])}%; offline minus GPS {fmt(m['gpsoff'])}%. GPS is a fused sensor reference, not survey control. The +2.14% result on the earlier 15.64 m departure is not a global wheel-speed correction.",
        "",
        f"Independent scale sensitivity on {st[1]:,} reserved cross-camera landmarks: hold visual orientations and rigid baselines fixed, vary only ego displacement, re-triangulate each point, and minimize robust bearing error. The fitted multiplier on the delivered displacement is {st[0]:.4f} ({(st[0] - 1) * 100:+.2f}%). Each landmark must have a well-conditioned scale direction and a consistent local fit, not agree with unit scale. No post-hoc multiplier was applied: whole-route GPS and the baselines support roughly sub-percent scale, not a uniform 2% stretch.",
        "",
        f"Offline horizontal difference from dead reckoning, median / p95 / max: {fmt(m['dead_delta'])} m. Versus fused GPS: {fmt(m['gps_delta'])} m. [Difference over time](report/trajectory-differences.png).",
        "",
        "## Attitude, elevation and live latency",
        "",
        f"[Pitch, roll, height and acceleration](report/attitude-height.png). Absolute pitch p95 {m['pitch'][1]:.2f}°, absolute roll p95 {m['roll'][1]:.2f}°, height {m['height'][0]:+.2f} to {m['height'][1]:+.2f} m. On moving samples, estimated pitch vs telemetry forward acceleration (−ay) r={m['pitch_forward_accel']:+.2f}; roll vs left acceleration (−ax) r={m['roll_left_accel']:+.2f}. Versus wheel-derived dv/dt and v·yaw-rate, r={m['pitch_dv']:+.2f} and {m['roll_vyaw']:+.2f}. The axis assignment was checked against smoothed wheel kinematics (both |r| ≈0.95–0.99). Turn bank/grade and suspension response cannot be separated without independent inertial attitude. Height is weakly constrained, not surveyed.",
        "",
        f"Both products are bit-identical throughout the three stopped intervals: "
        + ", ".join(f"{a:.2f}–{b:.2f} s" for a, b in m["stop"])
        + ". The initial 5.67 seconds include real movement.",
        "",
        f"The causal filter triangulates from prior rays and robustly resects each new packet. It uses a low-frequency GPS-minus-wheel offset, wheel increments, and gentle attitude/height anchoring. This prevents correlated monocular depths from accumulating unbounded drift. The filter has no backward smoothing; master interpolation needs at most {m['max_lag']:.3f} s of future exposure on moving samples. Live minus offline over all frames (median / p95): {fmt(m['causal_delta'][:2])} m and {fmt(m['causal_angle'][:2])}°. Moving-only: {fmt(m['causal_moving'][:2])} m and {fmt(m['causal_ang_moving'][:2])}°.",
        "",
        "## Coverage and downstream quality",
        "",
        "Both NPZ files carry sigma_pos (metres), sigma_rot (radians), n_tracks and quality_flag. Sigmas are local support/error-based quality proxies, not absolute GNSS covariance or independently calibrated coverage intervals. Flag bit 1 = stationary; bit 2 = low moving-frame visual support; bit 4 = just after a recorder timestamp gap. The weakest moving starts are 0–1.6, 81.9–83.8, 202.7–203.9 and 286.7–288.7 s; live depth support takes longer to recover after stops. Look also around 120.1, 180.4, 240.7 and 301.0 s minute boundaries and 152.7–155.5 s (relatively weak fast-road support). Parked bins with zero active tracks are intentional.",
        "",
        "[Per-camera track counts for every 10 s](report/tracks-per-10s.csv). Unique retained tracks by 10 s bin (a track spanning bins counts once in each):",
        "",
        "|Start (s)|Tracks|Start (s)|Tracks|Start (s)|Tracks|Start (s)|Tracks|",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    ws = m["windows"]
    for j in range(0, len(ws), 4):
        lines.append(
            "|"
            + "|".join(
                str(z)
                for v in (ws[j : j + 4] + [["", ""]] * max(0, 4 - len(ws[j : j + 4])))
                for z in v[:2]
            )
            + "|"
        )
    text = chr(10).join(lines) + chr(10)
    (OUT / "report.md").write_text(text.replace("](report/", "](debug/"))
    dest = RS / "pipeline/poses/report"
    dest.mkdir(exist_ok=True, parents=True)
    for p in (OUT / "debug").iterdir():
        if p.name in {
            "attitude-height.png",
            "trajectory-differences.png",
            "tracks-per-10s.csv",
            "road-transfer-audit.jpg",
        }:
            copy2(p, dest / p.name)
    (RS / "pipeline/poses/REPORT.md").write_text(text)
    print("report written", flush=True)


if __name__ == "__main__":
    main()
