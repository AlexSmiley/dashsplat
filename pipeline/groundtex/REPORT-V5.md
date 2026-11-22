# Photographic ground in the two v5 display scenes

**The hatched median is continuous, its black dot is gone, and the 128 s left road edge no longer steps into dark gaps.** The front camera still supplies both arrows and the dash. The remaining weaknesses are visible camera joins on the entrance apron, some verges and outside lanes, and a residual paint-registration error above the single-photo control. This is the published offline ground, not a claim about browser rendering.

## Production

[Regular v5](http://10.0.50.51:8090/scenes/stitched-opt-v5/ground/ground.json) and [movers v1](http://10.0.50.51:8090/scenes/stitched-opt-v5-movers-v1/ground/ground.json) share **1,160 immutable texture files and one manifest via hard links**. Both served manifests match SHA-256 `bd16b475c21503ce3bb0d3b10f03c47e0d87342c43af6675f527fedc8fa5c161`; an actual texture was fetched and checked from both URLs. All three drive files (v4, v5, v5 movers) were byte-for-byte equal: 12,927 poses, same orientation, timing and ENU origin, **0 cm** maximum positional difference. Cache rebuild: 924 splat tiles already current in each v5 scene. [Publication receipt](report/v5/publication.json) · [Viewer format: orientation, shared edges and premultiplied mip chain](FORMAT.md).

**43,509,915 bytes active**, including the 1,005,797-byte manifest; full 361.42 s / 3,312.21 m, **1.31 MB per 100 m**. 1,054 tiles at 1 cm pitch and 106 at 2 cm. 17,217 shared height comparisons matched exactly. These are active bytes, not the disk usage of historical content-addressed payloads. [Decode/geometry receipt](report/v5/verified.json).

## Compare the photographs

| Feature | Change and remaining limit |
|---|---|
| [Hatched median](report/v5/compare-median.jpg), visible ahead at 118 s, crop centred at 122 s | The committed crop mixed pillar and front exposures across the right arm. All 307,200 native crop samples now come from **one left-pillar photograph (4360)**. All 382 interior missing samples are now opaque measured pavement. The only long-range front exposure that covered ≥85% of the patch failed the cross-photo shape match (normalized correlation 0.46). |
| [Bicycle](report/v5/compare-bike.jpg) and [kerb](report/v5/compare-bike-kerb.jpg), 110 s | The bicycle and road side of the kerb are continuous in rear photo 3950. The complete long-range front candidates did not reproduce the symbol (best correlation 0.11 on the kerb crop). Grass falls back to splats. Six enclosed holes became zero; the edge is deliberately softer than the original hard, jagged pavement/grass transition. |
| [Left verge](report/v5/compare-left-verge.jpg), 128 s | One front exposure (4557) registers to the side witness with a 4 cm correction. Seven enclosed holes became zero, and all 710 originally missing crop samples are now opaque. |
| [Left arrow](report/v5/compare-left-arrow.jpg), [right arrow](report/v5/compare-right-arrow.jpg), [dash](report/v5/compare-lane.jpg) | Both arrows remain entirely front-camera sourced. The 128 s dash has a **1.26 cm** median 20–80% edge transition, versus **1.04 cm** in the source photograph and **4.18 cm** in v5 splats. 230 transverse edges per panel; [profile](report/v5/dash-profile.json). |
| [Entrance apron](report/v5/entrance-clean.jpg), about 91 s | The lane overview caught a red bollard from an 83 s stationary front image projected onto the apron. Replaced 28,033 occluded 4 cm cells with photographed semantic road, leaving 326 unsupported cells to fallback. The pole is gone; tone joins and small unsupported areas remain here. |

[100 s chase](report/v5/chase-100.jpg) · [112 s](report/v5/chase-112.jpg) · [118 s](report/v5/chase-118.jpg) · [128 s](report/v5/chase-128.jpg). Identical 1440×900 viewer camera transforms; left is v5 ground splats, right is the published textured mesh over the same fallback. Native comparison crops use 1 cm samples. Measured patch coverage and camera ownership are in [patch-comparison.json](report/v5/patch-comparison.json).

## Registration and independent audit

All **841** originally flagged paint anchors had a front-camera owner. Camera priority alone could not fix them. We registered source patches against a front-weighted photographic geometric reference, matched source sharpness by lens/exposure selection, then ran a separate correction in the 32 baseline tiles with at least four flags. All windows in each such tile were eligible: 6,235 matches against **training** front exposures produced 1,411 consensus locations. Median correction **0.66 cm**, maximum smoothed correction **3.90 cm**. No held-out pixels went into a source warp. The marking-owner selection uses only source image/mask evidence; smooth gains adjust tone, never texture sharpness. [Fit data](report/v5/patch-registration.json).

Fixed audit: a flag is a same-facing edge pair 3–30 cm apart not independently supported by a held-out photograph. Floor is a different unused photograph at nearly the same time. Fresh audit: the **other alternating held-out frames**, not consulted to select tiles or tune repairs. Different coverage changes some denominators and floor estimates. RGB errors are median per-frame mean absolute channel errors in 0–255 code values, measured on mutually visible static road 5–15 m ahead.

| Slice / tile map | Fixed poses | Fixed flags, committed → v5 | Fixed photo control | Fresh flags, committed → v5 | V5 RGB / photo floor |
|---|---:|---:|---:|---:|---:|
| [85-140 s](report/v5/flags-85-140.jpg) | 33 | 1.71 → **0.83%** (107/12911) | 0.03% | 1.47 → 1.85% | 4.45 / 2.95 |
| [190-215 s](report/v5/flags-190-215.jpg) | 15 | 0.29 → **0.29%** (22/7679) | 0.00% | 1.45 → 0.97% | 6.11 / 3.15 |
| [286-330 s](report/v5/flags-286-330.jpg) | 26 | 4.03 → **2.13%** (289/13557) | 1.62% | 3.21 → 2.16% | 4.60 / 3.35 |

The independent floor is **not reached**. The 190–215 s fixed count is unchanged; the first slice gets worse on the fresh split. Fresh photo controls on v5 were **1.43%, 0%, 7.04%** respectively; the last one is particularly sensitive to viewpoint. Checking against the *actual contributing photographs* instead found **3/13,917**, **1/8,261** and **8/14,769** suspicious outline anchors, versus zero for unmixed controls. These are distinct tests. Per-tile JSONs include physical ENU locations, held frame numbers and actual owners: [85–140](report/v5/tiles-85-140.json) · [190–215](report/v5/tiles-190-215.json) · [286–330](report/v5/tiles-286-330.json).

## Tone and limits of coverage

The distance-straightened road overviews compare the committed atlas and v5 with no splat background: [85–140](report/v5/road-strip-85-140.jpg) · [190–215](report/v5/road-strip-190-215.jpg) · [286–330](report/v5/road-strip-286-330.jpg). Black is transparent. Actual crossroads, paint, shadows and road materials can cause changes in brightness. The centre-lane stripe statistic measures deviation from a 1.5 m smooth profile, not the physical cause of each stripe. P90 amplitude (0–255) was **1.56 → 1.52**, **1.51 → 1.52**, **1.88 → 1.86**: broadly steady, not a claim of zero banding. The low-gradient verge-to-v5-splat mean RGB difference improved in all three slices: **11.98 → 8.87**, **8.45 → 7.52**, **13.73 → 8.53**. This checks the intended colour reference, not an independent photographic accuracy floor. Farther outside the lane some camera colour/texture steps remain. [Surface measurements](report/v5/surface.json).

Opacity in the 3.2 m driven corridor by **distance**: **99.86%, 100%, 100%** in the display slices; **99.92%** whole drive. By **time**, **99.67%, 100%, 100%**; whole drive **93.13%**, down from 97.54% in the committed atlas. The stricter masks leave more fallback at the nearly stationary initial 0–85 s: 71.07% time-weighted there. The data do not imply full-width pavement or coverage across raised verges. [Coverage by window](report/v5/coverage.json).

## Moving vehicles

A decoded-output audit checked **every nontransparent interior texel** of the 611 tiles within the display footprints (tile centres within 43 m of the slice trajectory), including partial alpha. It recovered the actual member photograph and tested all four native bilinear source neighbours. Results: **136,216,395 / 49,679,407 / 147,683,510** texels by slice; **zero** intersected an instance or removal mask, had an absent removal mask, were ownerless or sampled outside the rig mask. [Decoded mask audit](report/v5/slice-masks.json). That exhaustively excludes **labelled** vehicle pixels; it does not independently prove the segmenter missed none. There is no temporal median or shadow-removal operation.

At [198–206 s](report/v5/risk-198-206.jpg), the red vehicle first appears about 34–40 m away, outside the near-camera source stencil. The exclusion audit extended to 80 m, recovered 8,672 cells from before/after frames, retained 3,076 already from other times, and left 4,660 unsupported. At [286–292 s](report/v5/risk-286-292.jpg) it replaced 27,724 cells, retained 40,044 already drawn from other times, and left 32,928 unsupported. **Zero remaining owners from either risky interval on those projected vehicle footprints.** Shadows/stains in the admitted photographs remain; where no clear photograph survived, alpha reveals the scene, not invented road. [Temporal ownership receipt](report/v5/risk-ownership.json).

[Studio build commands](README.md) · [Original v4 report](REPORT.md).
