# Six-camera rig calibration — single HW4 drive

**Result:** six footage-fitted lenses and poses. Subpixel RMS is an inlier fit statistic. Absolute axle position and independent metric scale are less certain; see below.

## Method

RootSIFT matches, robust pair filtering, conflict-free temporal tracks and later cross-camera joins. Joint CPU Ceres sparse-Schur BA fits per-frame 6-DoF ego corrections, metric landmarks, lens focal lengths/centres, two equidistant distortion terms (one for front), camera poses, effective exposure offsets and rolling shutter. k3/k4 are fixed at zero: high-order terms cannot be reliably separated from this drive. Pixel centres follow COLMAP: the top-left centre is (0.5, 0.5). World is ENU; camera axes right/down/forward; the exported transform maps camera to ego. No learned models or factory rig parameters. Physical mount locations and the published front focal length are initial/soft priors, not ground truth.

The fit uses 164 synchronized rig frames from five windows (0–4, 82–94, 95–104, 104.1–111.4, 318.5–325 s). Speed-integrated displacement fixes metric scale. Heading and pose soft priors plus a no-lateral-slip and along-track increment residual define the vehicle frame. Dark borders and bodywork are manual polygons from the temporal median of moving footage. Triangulation requires ≥3 consistent observations and rejects behind-camera points. Cauchy BA is followed by 4 px and 3 px geometric outlier gates; the final timing fit does not rethreshold the observations. RMS is sqrt(mean(dx²+dy²)) in native-resolution pixels.

## Rig and final bundle

HFOV/VFOV first pair is the usable span along the principal axes after the body mask; next is the model span to the sensor edges. Yaw positive left, pitch positive up, roll about the forward optical direction after yaw/pitch. Positions in metres, ground origin at rear axle.

|Camera|Usable H × V (°)|Sensor H × V (°)|x, y, z (m)|yaw, pitch, roll (°)|RMS / p95 (px)|inlier obs / tracks|offset / readout (ms)|
|---|---:|---:|---|---|---:|---:|---:|
|front|46.2, 29.4|46.2, 29.6|1.721, 0.038, 1.302|-0.38, 0.01, -0.38|0.974 / 2.09|25270 / 4602|0.00 / 14.04|
|back|176.5, 94.7|180.3, 113.6|-0.977, 0.052, 0.836|179.24, -33.32, -1.03|0.915 / 1.95|21039 / 5399|-8.70 / 17.80|
|left_pillar|82.6, 52.0|83.0, 52.3|0.944, 0.697, 1.260|70.32, -4.13, 0.14|0.815 / 1.78|15403 / 3797|1.24 / 16.30|
|right_pillar|83.2, 52.4|83.6, 52.7|0.948, -0.709, 1.263|-68.76, -4.40, -0.23|0.823 / 1.78|12451 / 3195|-6.83 / 14.98|
|left_repeater|77.3, 52.2|83.3, 52.5|2.389, 0.911, 0.733|142.84, -4.81, 0.35|0.815 / 1.74|21526 / 5236|-3.09 / 17.08|
|right_repeater|78.2, 52.2|83.2, 52.5|2.402, -0.898, 0.725|-141.83, -3.11, 0.43|0.813 / 1.74|18859 / 4720|1.13 / 15.69|

Offsets are relative to the front clock (front fixed at zero). Effective readout is top-to-bottom, referenced at the middle row. For row v, use t + offset + (v/height − 0.5) × readout. The timing model linearizes world velocity and yaw rate over an exposure; no gyro or hardware timing measurement exists. Rolling shutter and clock offsets can trade off with lens parameters and pose.

## Gauge and symmetry

The rear tyre corridor photometric road-plane sweep accepted 142/159 frame pairs. Its median height in the preliminary frame was -0.020 m; 16th–84th percentiles -0.060, 0.079 m. Grades and suspension motion limit absolute height. Yaw comes from robust mid-arc travel direction; lateral zero is the mean of the two left/right mount midpoints. No mirror rotations or pair-specific coordinate equality were imposed.

The no-slip rear-axle fit alone puts the back lens at x=-1.078 m, outside the car’s −0.977 m rear extent. The delivered common longitudinal origin is the nearest feasible one under that hard body-envelope bound. The 10.1 cm change applies to all six; it is not a measurement of the rear lens recess. Trust relative baselines more than absolute x. The lateral residual at the unconstrained solution is 5.3 mm per increment (already informed by the no-slip prior).

- pillar mirror residual: 1.3 cm (-0.4, -1.3, -0.3 cm XYZ), 1.59° SO(3).
- repeater mirror residual: 2.0 cm (-1.4, 1.3, 0.7 cm XYZ), 2.10° SO(3).

## Independent frame splits

Alternating keyframes from each route window, no shared images; features, matches, track assembly, triangulation, BA, timing and ground alignment are fit separately from identical starting priors. The scenes are shared, so this is same-drive repeatability, not between-drive accuracy. Distortion differences are absolute coefficient units (percentage is meaningless near zero). The body bound anchors back x in each run.

|Camera|max Δf/f|Δ principal point (max axis px)|max Δk|Δ position (cm)|Δ orientation (°)|Δ offset / readout (ms)|
|---|---:|---:|---:|---:|---:|---:|
|front|0.165%|2.35|0.0002|2.76|0.091|0.00 / 1.88|
|back|0.218%|0.31|0.0015|1.44|0.146|0.29 / 0.67|
|left_pillar|0.210%|0.28|0.0018|2.25|0.071|0.21 / 0.17|
|right_pillar|0.233%|0.75|0.0010|2.70|0.134|0.11 / 1.46|
|left_repeater|0.286%|1.08|0.0030|1.67|0.101|0.26 / 0.18|
|right_repeater|0.276%|1.34|0.0022|1.70|0.146|0.06 / 1.26|

## Held-out scale

231.623–235.317 s, excluded from both rig and ground fits. Freeze the rig and solve 18 held-out poses and landmarks, with one fixed gauge pose, no velocity/odometric measurements, and only visual velocities for exposure compensation. Arbitrary forward spacings of 0.60 and 1.30 m lead to 15.9701 and 15.9683 m, with 826 / 833 multi-camera landmarks. Integrated telemetry: 15.6360 m. Difference **+2.14%**. This misses the ±1% goal; the rig was not rescaled to the holdout. This short departure is not a whole-drive metric accuracy claim.

## Cross-camera checks

Triangulate from ≥2 images in a single other lens strictly before or strictly after the target frame. Exclude all target-lens observations from point fitting; require source-only angular baseline, positive range and source RMS <2.5 px. No target residual gate. Poses/intrinsics were fit on this scene, so this is leave-target-lens-out point transfer, not an unseen-scene test. Gold circles mark measured features; magenta crosses mark predictions. Each figure uses the image pair with the most eligible transfers, independent of its errors. Large tails arise from narrow source triangulation angles, viewpoint changes and mismatches; the transfer statistic is stricter than fitted-point RMS.

|Target|transfers|median / p90 (px)|shown count / median / max (px)|
|---|---:|---:|---:|
|front|1993|2.81 / 7.79|15 / 3.11 / 6.45|
|back|7454|0.73 / 2.02|33 / 0.58 / 3.08|
|left_pillar|2483|1.42 / 5.10|15 / 0.97 / 1.72|
|right_pillar|1697|1.37 / 5.06|13 / 0.78 / 2.20|
|left_repeater|6239|1.36 / 3.81|27 / 1.29 / 6.88|
|right_repeater|5255|1.34 / 3.79|24 / 1.09 / 4.64|

The rear is retained. Its masked rim and bumper are not extrapolated into the rectified view. Confidence in the usable optical core is much higher than in the extreme rim or its absolute longitudinal mount. Rectified images use a forward pinhole crop, not a full-width fill of a nearly 180° fisheye. Inspection of the rectified drive-through images showed straight masonry joints, posts and lane edges. All six paired overlays place projections on the same identifiable landmarks (traffic signal supports, façade edges, signs and trees); the front has the largest transfer errors.
