# Road-relative attitude and the bird’s-eye test

**0 of 41 supported segment–camera-pair cells meet their requested thresholds** (22 unsupported). For same-camera cells both median <3 cm and p90 <8 cm must hold; for cross-camera cells median <5 cm. A singleton is reported, not treated as strong evidence. Raster resolution is 2 cm, not an accuracy claim.

## Ground registration

Every scheduled keyframe of all six cameras entered the three mosaics: park: 172 packets, turn: 166 packets, straight: 533 packets. There are 61,479 accepted photographic pair observations over the drive; 12,359 were reserved by a deterministic hash of 1 m ground cells before fitting either trajectory. Neighbouring patches may share texture; the observations are not independent statistical trials. Same-camera repeats must be ≥1 s **or** ≥5 m apart; cross-camera pairs can be simultaneous. Both keyframes must be inside the labelled segment. Statistics below use both observations inside 10 m on the planar reference and connected pavement in that baseline. Measurements use the same native pixel correspondences under every candidate, without refitting target shifts to that candidate or gating on transfer error.

Cells: **median / p90 cm (number of patch matches)**. All lenses included. V1 has no road model: the default flat-world test uses z=0. The extra V1/local-level rows give it a horizontal tangent at the instantaneous axle elevation, removing vertical datum error. The final V2 rows use its estimated road plane.

|Trajectory|Park same|Park cross|Turn same|Turn cross|Straight same|Straight cross|
|---|---:|---:|---:|---:|---:|---:|
|planar|46.4 / 59.4 (19)|17.5 / 46.1 (214)|33.2 / 51.1 (49)|25.9 / 50.7 (450)|20.7 / 40.6 (219)|17.4 / 38.2 (1416)|
|offline-v1|28.7 / 60.3 (19)|23.9 / 96.4 (214)|665.3 / 803.6 (49)|455.8 / 912.3 (450)|79.1 / 153.2 (219)|49.1 / 161.1 (1416)|
|causal-v1|16.2 / 25.5 (19)|18.0 / 51.7 (214)|50.6 / 132.4 (49)|37.5 / 154.9 (450)|25.3 / 120.3 (219)|26.5 / 117.3 (1416)|
|offline-v1/local-level|58.1 / 84.5 (19)|31.6 / 73.7 (214)|63.9 / 193.1 (49)|39.6 / 119.1 (450)|58.8 / 126.6 (219)|51.6 / 151.0 (1416)|
|causal-v1/local-level|41.3 / 49.0 (19)|22.6 / 48.3 (214)|51.4 / 132.6 (49)|36.7 / 102.8 (450)|19.8 / 43.5 (219)|18.1 / 41.5 (1416)|
|offline|14.0 / 29.3 (19)|10.6 / 55.3 (214)|9.1 / 32.1 (49)|13.2 / 36.6 (450)|9.8 / 28.5 (219)|12.0 / 28.4 (1416)|
|causal|8.4 / 27.8 (19)|9.3 / 29.3 (214)|9.9 / 30.9 (49)|15.9 / 38.3 (450)|10.7 / 28.4 (219)|18.0 / 44.3 (1416)|

### Final offline, by physical lens pair

|Lenses|Park 84–105 s|Turn 316–326 s|Straight 293–310 s|
|---|---:|---:|---:|
|front / front|—|—|—|
|front / back|46.3 / 56.8 (17)|36.6 / 55.0 (27)|17.3 / 62.1 (39)|
|front / left_pillar|—|12.0 / 12.0 (1)|—|
|front / right_pillar|—|37.1 / 37.1 (1)|26.9 / 33.2 (5)|
|front / left_repeater|19.0 / 19.0 (1)|25.4 / 50.5 (24)|9.0 / 55.9 (35)|
|front / right_repeater|7.2 / 55.9 (4)|123.4 / 150.4 (2)|23.5 / 55.5 (44)|
|back / back|19.7 / 27.4 (7)|20.2 / 34.5 (18)|7.0 / 32.2 (48)|
|back / left_pillar|—|14.6 / 31.0 (57)|9.1 / 13.3 (62)|
|back / right_pillar|10.5 / 52.2 (41)|—|12.3 / 31.5 (91)|
|back / left_repeater|31.7 / 65.0 (13)|14.1 / 28.1 (182)|13.8 / 26.5 (362)|
|back / right_repeater|9.2 / 21.1 (98)|8.9 / 56.8 (66)|12.1 / 25.9 (431)|
|left_pillar / left_pillar|63.5 / 63.5 (1)|3.9 / 6.3 (7)|6.8 / 17.0 (29)|
|left_pillar / right_pillar|—|—|—|
|left_pillar / left_repeater|42.0 / 61.4 (5)|5.5 / 25.1 (88)|7.5 / 19.7 (162)|
|left_pillar / right_repeater|—|—|—|
|right_pillar / right_pillar|14.6 / 24.7 (4)|—|10.1 / 24.2 (17)|
|right_pillar / left_repeater|—|—|—|
|right_pillar / right_repeater|6.2 / 65.9 (35)|28.0 / 35.2 (2)|10.1 / 29.7 (185)|
|left_repeater / left_repeater|—|6.5 / 18.0 (15)|14.8 / 28.2 (46)|
|left_repeater / right_repeater|—|—|—|
|right_repeater / right_repeater|3.4 / 13.7 (7)|10.8 / 77.8 (9)|10.5 / 29.0 (79)|

The [full pair × distance × segment table](report/patch-shifts.csv) includes both versions, planar telemetry, 0–4, 4–7, 7–10 and 10–12 m bins. A lens looking away from the patch cannot supply a match. Some same-camera forward-looking fields are very narrow inside 10 m. Zero matches after the repeat-baseline, pavement, usable footprint and correlation checks means unmeasured, not perfect alignment. The fixed ±0.60 m photographic search, NCC >0.78, peak margin >0.065 and reverse error <2.4 cm select recognisable texture and can under-count larger failures. Broad untextured asphalt is not measured. Gain-normalised patches are 0.675 m across, and subpixel correlation peaks are converted back to the two original source rows. Camera/time-dependent exposure and elongated edges remain failure modes.

### Failure evidence and calibration

The [pair diagnostics](report/pair-diagnostics.csv) give median signed shifts in the observation heading frame, NCC and the larger endpoint’s ground displacement per native pixel. A broad aperture, foreshortened paint, raised curbs/vehicles and crown or ramps within one tangent footprint can look like a mount error. The following distributions use an independent diagnostic subsample and the deployed geometry (including unsegmented texture), so are not denominators of the table above.

|Diagnostic|Park|Turn|Straight|
|---|---:|---:|---:|
|0–4 m|6.8 / 26.2 (19)|7.0 / 19.5 (39)|7.5 / 14.5 (120)|
|4–7 m|8.6 / 20.9 (112)|9.0 / 23.9 (189)|9.3 / 26.0 (569)|
|7–10 m|17.6 / 57.5 (75)|16.2 / 39.4 (186)|11.5 / 34.1 (658)|
|centre|9.2 / 24.8 (94)|11.7 / 30.5 (175)|9.3 / 25.1 (587)|
|edge|9.0 / 34.9 (29)|12.3 / 43.1 (71)|11.2 / 43.2 (201)|

Camera-specific small roll/pitch/yaw, clock and row-readout shifts, and an image-radius term were fitted to training pairs, then transferred to geographically or dynamically different observations. These perturbations compete with local pavement shape and one another. This is a differential hypothesis test, not a new calibration. Native-feature RMS below uses the same fixed 3D scene, without retriangulating to disguise a change.

|Trained on → evaluated on complement|Road median / p90 before (cm)|Road median / p90 after (cm)|Largest lens native RMS before → after (px)|
|---|---:|---:|---:|
|early (3583 eval)|9.6 / 28.2|6.6 / 25.4|right_pillar: 1.57 → 8.39|
|late (1848 eval)|9.2 / 31.6|7.1 / 29.6|front: 2.20 → 12.11|
|slow (4524 eval)|9.7 / 29.9|7.6 / 27.9|right_pillar: 1.57 → 6.79|
|fast (1774 eval)|9.0 / 28.7|6.5 / 26.3|front: 2.20 → 12.43|

[Single-parameter checks](report/single-parameter-check.csv) separately test clock, readout, each mounting rotation and a radius-dependent warp for each lens on an unseen half of the drive. No calibration file was rewritten. A smaller road-only residual is insufficient evidence for a physical change that damages the static scene. The two independent road halves suggest +5.8 and +6.6 ms at the front; that single correction reduces transferred road median by at least 3.8 cm but raises its static-feature RMS by as much as 4.0 px. Rear pitch changes of +0.16° / +0.17° gain about 2.1 cm but cost up to 0.34 px. Among one-parameter changes costing <0.10 px, the best replicated median improvement is just 0.12 cm. Neither the road-only clock nor these mount/optics alternatives are an identified calibration fix.

## Suspension and world gauge

The offline solve robustly minimises 2-D ground disagreement of normalised image patches, with weak roll/pitch/heave and XY/yaw regularisation; the acceleration intercepts and gains are nuisance variables fitted jointly. The 6-DoF static-feature Ceres solve then retains all original accepted observations, physical odometry and baselines, with a road-state prior on each knot and the same training patch centres as explicit row-timed ray/plane factors. All landmarks are initialised in the new local rigid frame and are free to adjust. The Cauchy scale for the retained scene observations is 4.5 px; road factors use 1.8 cm scale with robust tails. The live solve updates suspension and planar corrections against current and earlier patches only; it learns its gain coefficients forward in time, above the V1 causal landmark filter. Both solutions share the same camera geometry and body mask. No offline trajectory or future-route gain is fed into the live filter.

Road shape and absolute vertical datum are weakly observable. The offline estimator filters V1 scene height and attitude-minus-suspension over 31–37 m, integrates longitudinal grade with a low-frequency height anchor and centres the distance-weighted mean world normal. These priors choose a gauge; pavement texture independently measures the **relative** rotation/heave, not a surveyed crossfall. The causal estimator integrates a 29–46 m forward-only grade/bank estimate with a running mean normal. It cannot know the exact future whole-route normal. The plane routine returns n,h for n·X_ego+h=0. Height is normal to the road, all angles are in heading-aligned coordinates.

|Motion statistic|Offline|Causal|
|---|---:|---:|
|Gain: pitch / roll (° per m/s²)|0.149 / 0.156|0.133 / 0.172 (last online fit)|
|Correlation: pitch with −ay / roll with −ax|0.562 / 0.296|0.548 / 0.422|
|p95 absolute world pitch / roll (°)|0.76 / 1.20|0.61 / 0.78|
|World surface height min / max (m)|-0.61 / 0.60|-0.29 / 0.72|
|Path-weighted mean normal tilt (°)|0.0005|0.0873|
|p95 height vs integrated grade discrepancy over 100 m (cm)|0.48|0.57|
|Suspension pitch (°) p05 / p50 / p95|-0.58 / -0.09 / 0.28|-0.43 / -0.06 / 0.34|
|Suspension roll (°) p05 / p50 / p95|-0.77 / -0.22 / 0.51|-0.57 / -0.14 / 0.33|
|Suspension heave (cm) p05 / p50 / p95|-4.87 / -1.41 / 2.48|-3.07 / 0.09 / 4.46|
|Road grade (°) p05 / p50 / p95|-0.19 / 0.03 / 0.26|-0.25 / 0.01 / 0.21|
|Road crossfall (°) p05 / p50 / p95|-0.63 / 0.11 / 0.74|-0.31 / -0.00 / 0.41|

Moving-sample offline–causal absolute differences (median / p95): suspension pitch 0.15 / 0.48°, roll 0.18 / 0.62°, heave 2.57 / 7.37 cm; total world pitch 0.21 / 0.61°, roll 0.34 / 0.86°. Correlations use the same 0.20 s causal-filtered acceleration as the fit, on moving samples, and are not an independent validation of the physical gains. [Updated attitude plot](report/road-attitude.png).

Both deliver 12,927 master samples and exact stationary poses and decompositions (5.67–81.83, 198.82–202.65, 257.70–286.67 s). Maximum causal lookahead including the V1 interpolation dependency and the last shutter row: 0.484 s. Native row projection round-trip on fast/turn road pixels: 0.0029 px maximum. These are internal consistency checks, not ground truth.

Across non-overlapping 100 m chords, absolute modelled road elevation change has median/p95 9/56 cm offline and 15/46 cm causal; the largest offline chord is 86 cm. The absolute road profile is not surveyed.
Road support is weakest at 0–1.64, 81.86–84.16 and 286.70–287.84 s (departures), and 334.34–335.34 s. Bit 8 marks 455 of 9,028 moving master samples with fewer than four training pairs in the preceding 350 ms. The visible roll sign change across the heading change near 150 s is smaller than V1 but not eliminated; its division into road bank and body roll is prior-dependent.

## Whole-route scene consistency

Same 649,384 native observations, same IDs, no new pruning after the change.

|Camera|Observations|V1 RMS px|V2 RMS px|V2 p95 px|
|---|---:|---:|---:|---:|
|front|88,334|1.796|2.337|3.961|
|back|92,183|1.696|1.695|3.528|
|left_pillar|127,710|1.495|1.632|3.274|
|right_pillar|133,709|1.532|1.629|3.263|
|left_repeater|100,633|1.693|1.794|3.685|
|right_repeater|106,815|1.778|1.896|3.816|

The front RMS increase is material: 146 observations exceed 10 px (V1: four), and 80% of the increase in squared error comes from the worst 1% of the V2 front observations. The 100–110 s exit accounts for 56% of the net front increase; 150–160 and 320–330 s also contribute. No observation was removed to improve this number.

## Parked roll and the heading change near 150 s

Median by interval, degrees. A change in absolute bank is not evidence of suspension motion while stationary.

|Window (s)|V1 world roll|V2 world roll|Causal V2 world roll|V2 road crossfall|V2 suspension roll|
|---|---:|---:|---:|---:|---:|
|6–81|+0.48|-0.35|-0.33|+0.11|-0.47|
|137–144|-1.93|-0.81|+0.18|-0.54|-0.33|
|153–160|+0.47|+0.41|+0.36|+0.14|-0.06|
|199–202|+1.39|+0.14|-0.11|+0.46|-0.32|
|258–286|+2.00|+0.80|+0.31|+0.29|+0.51|

## Visual comparisons

Each five-column crop uses the same time and ego-relative offset (accounts for different global XY gauges), identical metric scale and no image-based final registration. At 101.19 s the car-park exit arrow no longer breaks into two displaced pieces as it does under offline V1. At 318.62 s the ONLY letters regain their expected size and a recognisable outline. The arrow at 87.86 s and the stop bar at 297.67 s still show camera-boundary steps and exposure bands; not every marking is single-edged. Inspection also found square holes cut out of observed paint by the colour classifier. Small enclosed neutral-colour islands in the display now retain their original best-view pixels; nothing is inpainted or registered after the fact. The stricter road-mask.png is retained for all evaluation. The best-view raster can hide disagreement between views; use the held-out shifts alongside it. The first/last maps and full mosaics remain on Studio.
- [compare-park-entrance-arrow](report/compare-park-entrance-arrow.jpg)
- [compare-park-exit-arrows](report/compare-park-exit-arrows.jpg)
- [compare-park-turn-arrow](report/compare-park-turn-arrow.jpg)
- [compare-straight-lane-crossing](report/compare-straight-lane-crossing.jpg)
- [compare-straight-stop-bar](report/compare-straight-stop-bar.jpg)
- [compare-turn-only-approach](report/compare-turn-only-approach.jpg)
- [compare-turn-only-turn](report/compare-turn-only-turn.jpg)
- [compare-turn-turn-pointer](report/compare-turn-turn-pointer.jpg)
