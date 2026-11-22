# Reality Splat — whole-drive ego poses

**Both trajectory products are available on all 12,927 master timestamps.** They are locally image-constrained and globally tied to the fused GPS/wheel solution. Small inlier RMS does not guarantee that a point triangulated far ahead will be within a few pixels on approach. Use the separate transfer checks below.

## Estimation and coordinate frame

5,527 distance/time selected six-view packets, 1,921 independent 6-DoF offline knots, 59,665 retained static-consistent landmarks including 9,427 seen in multiple physical cameras. Bidirectional LK checks and masked GFTT corners provide within-camera continuity. RootSIFT tracks span camera and time at 888 more widely spaced rig frames. Ceres fits rolling-row individual reprojections, wheel distance and heading increments, weak absolute fused GPS and elevation, and smooth pose corrections. Robust pixel losses and independent multiview consistency reject moving tracks and mismatches; very slow traffic cannot always be distinguished geometrically. After the first solve, observations above 5 px are removed and tracks must retain ≥4 observations, >75% support and a clipped mean below 4 px. The final scores below use the exported master-clock poses and source-row timestamps, not just the solver knots.

Ego is rear-axle centre on the ground, x forward/y left/z up; world is ENU with zero translation at the first sample. Inter-knot pose corrections retain high-frequency wheel shape. The shared geometry module then interpolates master poses with a constant body twist. The same pose variable represents both endpoints of each stopped plateau. Tiny encoded speeds around −2.48×10⁻⁷ m/s count as zero. No external altitude or gravity channel was available.

## Whole-drive bundle fit

Native-resolution Euclidean pixel residuals on retained observations. Track counts are distinct per lens; cross-camera landmarks may appear in several rows.

|Camera|Tracks|Observations|RMS px|p95 px|Fast residual equivalent clock shift (ms)|
|---|---:|---:|---:|---:|---:|
|front|7,192|88,334|1.796|3.767|-0.013|
|back|13,340|92,183|1.696|3.698|+0.008|
|left_pillar|12,437|127,710|1.495|3.283|-0.001|
|right_pillar|13,020|133,709|1.532|3.334|-0.002|
|left_repeater|11,955|100,633|1.693|3.714|-0.019|
|right_repeater|11,841|106,815|1.778|3.846|+0.018|

Clocks and readouts are unchanged. The fast-road test is the signed inlier residual along measured image velocity for speed >14 m/s. It is conditional on the joint trajectory/landmark fit and is not an independent shutter calibration. No coherent multi-millisecond shift was identified.

## Long-range transfer — held out from pose estimation

The natural-track split reserves IDs before either live resection or offline BA. Very few features persist from >25 m to <8 m. Supported lens results (landmarks, observations, median / p90 pixels): front: 1, 4, 126.6 / 159.9; left_pillar: 1, 2, 53.5 / 65.5. This sparse check is not a six-camera accuracy estimate.

A separate photographic check targets static road paint and pavement texture. It is independent of the offline solve: no road-check pixels or landmarks enter any pose factor. A causal-pose plane warp proposes matching places; actual positions come from image correlation with scale search, a uniqueness margin, reverse matching and a continuous sequence of accepted image matches. A search proposal is not a ground control point. No flat-road assumption is used to triangulate the final landmark. Near-only triangulation checks the range label; far-only triangulation uses ≥3 observations from the first third, each >25 m. Refine the far point against only those native source pixels (Huber loss), then score the last-third observations inside 8 m. Each end must be separately self-consistent and have positive depth. There is **no far-to-near target error gate**. Interpolation, far angular resolution, repeated paint patterns and finite patch-centering accuracy contribute to the numbers; they are not a pose-only error. Inspection of the raw audits shows repeated stripe/curb edges and far marks only a few pixels across: some associations are ambiguous, so these are end-to-end correspondence + depth + pose errors, not surveyed static-point errors.

Each cell is median / p90 pixels **(landmarks, observations)**. A dash means no supported measurement; broad roadside texture is not evidence for a missing lens or segment.

|Near camera|Whole sampled set|Car park 88–105 s|90° turn 317.5–325.5 s|Exit turn 105–111.5 s|Fast straight 293–310 s|
|---|---|---|---|---|---|
|front|88.9 / 192.4 (4, 11)|—|87.2 / 101.2 (2, 4)|—|142.2 / 206.2 (2, 7)|
|back|29.9 / 87.4 (26, 86)|88.5 / 91.2 (1, 3)|63.8 / 65.9 (1, 3)|—|29.1 / 85.7 (24, 80)|
|left_pillar|75.6 / 147.8 (24, 73)|—|101.9 / 142.1 (5, 15)|—|66.7 / 151.0 (19, 58)|
|right_pillar|75.5 / 220.4 (24, 78)|170.4 / 217.9 (3, 10)|—|—|68.8 / 216.4 (21, 68)|
|left_repeater|94.5 / 346.7 (21, 59)|—|284.0 / 515.8 (2, 9)|—|82.7 / 225.2 (19, 50)|
|right_repeater|72.2 / 411.1 (35, 109)|431.1 / 509.1 (3, 9)|—|—|64.5 / 279.4 (32, 100)|

217 deduplicated appearance tracks; 151 met the range and endpoint-count checks, 147 retained sufficient far parallax, 88 passed separate endpoint fits. See the [raw-image correspondence audit](report/road-transfer-audit.jpg).

Paired comparison on exactly the same marks, under each trajectory. Far points are re-triangulated for each product. The causal trajectory was used to propose photographic matches; this comparison is not blind to the causal model.

|Region|Offline median / p90|Planar median / p90|Causal median / p90|
|---|---:|---:|---:|
|whole|66.27 / 227.72|132.84 / 446.86|201.73 / 559.28|
|car park|194.44 / 486.42|275.01 / 405.22|196.67 / 556.65|
|90° turn|105.69 / 388.96|106.86 / 463.91|393.53 / 1102.22|
|exit turn|—|—|—|
|fast straight|55.52 / 195.73|130.11 / 458.06|192.70 / 505.35|

Do not interpret unsupported cells or good short-track RMS as a pass for few-pixel near reprojection everywhere. In particular, forward motion gives weak depth from distant, nearly collinear rays. Delay fusion or update depth after stronger parallax or another camera sees the surface.

Depth diagnostic on the road set: choosing the depth along a middle far ray using near observations reduces whole-set near median/p90 to 8.90 / 21.54 px. This oracle uses the answers and is not a held-out result; it separates along-ray depth weakness from other errors.

## Metric scale and global anchoring

Path length: offline 3312.85 m (horizontal 3312.77); causal 3315.17 m; integrated wheel 3323.66 m; raw fused GPS polyline 3319.26 m. Over 33 non-overlapping ~100 m chords, wheel minus GPS chord length (p10 / median / p90) was -0.26 / 0.40 / 0.76%; offline minus GPS -0.44 / 0.05 / 0.29%. GPS is a fused sensor reference, not survey control. The +2.14% result on the earlier 15.64 m departure is not a global wheel-speed correction.

Independent scale sensitivity on 1,285 reserved cross-camera landmarks: hold visual orientations and rigid baselines fixed, vary only ego displacement, re-triangulate each point, and minimize robust bearing error. The fitted multiplier on the delivered displacement is 0.9935 (-0.65%). Each landmark must have a well-conditioned scale direction and a consistent local fit, not agree with unit scale. No post-hoc multiplier was applied: whole-route GPS and the baselines support roughly sub-percent scale, not a uniform 2% stretch.

Offline horizontal difference from dead reckoning, median / p95 / max: 6.10 / 9.15 / 9.35 m. Versus fused GPS: 0.39 / 0.69 / 1.04 m. [Difference over time](report/trajectory-differences.png).

## Attitude, elevation and live latency

[Pitch, roll, height and acceleration](report/attitude-height.png). Absolute pitch p95 0.85°, absolute roll p95 2.15°, height -1.06 to +1.04 m. On moving samples, estimated pitch vs telemetry forward acceleration (−ay) r=+0.35; roll vs left acceleration (−ax) r=-0.15. Versus wheel-derived dv/dt and v·yaw-rate, r=+0.28 and -0.43. The axis assignment was checked against smoothed wheel kinematics (both |r| ≈0.95–0.99). Turn bank/grade and suspension response cannot be separated without independent inertial attitude. Height is weakly constrained, not surveyed.

Both products are bit-identical throughout the three stopped intervals: 5.67–81.83 s, 198.82–202.65 s, 257.70–286.67 s. The initial 5.67 seconds include real movement.

The causal filter triangulates from prior rays and robustly resects each new packet. It uses a low-frequency GPS-minus-wheel offset, wheel increments, and gentle attitude/height anchoring. This prevents correlated monocular depths from accumulating unbounded drift. The filter has no backward smoothing; master interpolation needs at most 0.324 s of future exposure on moving samples. Live minus offline over all frames (median / p95): 0.64 / 1.49 m and 1.06 / 1.91°. Moving-only: 0.81 / 1.52 m and 1.22 / 2.00°.

## Coverage and downstream quality

Both NPZ files carry sigma_pos (metres), sigma_rot (radians), n_tracks and quality_flag. Sigmas are local support/error-based quality proxies, not absolute GNSS covariance or independently calibrated coverage intervals. Flag bit 1 = stationary; bit 2 = low moving-frame visual support; bit 4 = just after a recorder timestamp gap. The weakest moving starts are 0–1.6, 81.9–83.8, 202.7–203.9 and 286.7–288.7 s; live depth support takes longer to recover after stops. Look also around 120.1, 180.4, 240.7 and 301.0 s minute boundaries and 152.7–155.5 s (relatively weak fast-road support). Parked bins with zero active tracks are intentional.

[Per-camera track counts for every 10 s](report/tracks-per-10s.csv). Unique retained tracks by 10 s bin (a track spanning bins counts once in each):

|Start (s)|Tracks|Start (s)|Tracks|Start (s)|Tracks|Start (s)|Tracks|
|---:|---:|---:|---:|---:|---:|---:|---:|
|0|442|10|0|20|0|30|0|
|40|0|50|0|60|0|70|0|
|80|1226|90|2002|100|2149|110|2811|
|120|3465|130|3766|140|3014|150|2883|
|160|3342|170|3424|180|3159|190|1355|
|200|1477|210|3344|220|2358|230|2295|
|240|2776|250|1381|260|0|270|0|
|280|435|290|3015|300|3368|310|2913|
|320|2428|330|1627|340|2355|350|3843|
|360|663|||||||
