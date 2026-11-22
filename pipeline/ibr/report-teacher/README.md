# Front-camera chase teachers · 85–140 s

On **anomastudio**, use `--pseudo "$RS/work/ibr/teacher-85-140/manifest.json"`. Paths in the JSON array are absolute Studio paths. The optimizer was not run. The teacher set has no viewer manifest and is not a Generative clip.

## Measurements

**1,498 usable views:** all 682 default stations plus 816 perturbations. All six offsets at the first station fell below the threshold. Confidence ≥0.5 covers a median **26.3%** of the default view (range **1.8–34.6%**); across all variants the median is **26.5%**. The base-to-target 3D distance has a median of **4.61 m** for the default chase. At 85 s it is 13.09 m, so almost the entire image is masked.

| Confident-pixel after-image check | Suspected twin anchors / sampled strong edges | Rate |
|---|---:|---:|
| All delivered views | 5 / 1,996,939 | **0.00025%** |
| Matched, unmixed PNG control · 20 poses | 0 / 20,782 | 0% |
| Injected 9 px double image · same 20 masks | 1,388 / 11,575 | 11.99% |

Worst individual audit: **0515-chase**, 2 / 2,279 sampled edges (0.088%). The five flags occurred in 3 views. No double outlines or camera joins were apparent in the 20 comparison plates. The capture is more informative than the stitched render on lane paint, tree crowns and the portions of the office with supported depth. It does **not** teach an uninterrupted skyline, narrow wires, or the entire building.

All **20/20** comparisons had higher fine detail on independently textured pixels and smaller color error against a third real front exposure. Median per-view RGB absolute error: **2.47/255 teacher vs 8.62/255 splat**. This held separately on ground (20/20) and uprights wherever enough pixels survived (15/15). Detail is strongest on the pavement. At 140 s, foliage still has a small high-error tail: 5.8% of tested confident upright pixels differed from the check exposure by >20/255. Treat the weight as confidence, not a per-pixel guarantee.

The existing optimizer's actual PNG decoder and pseudo crop ran on **26 poses**, covering the 20 comparison stations and all six perturbation types. All 5,992 manifest-referenced files exist. [Studio manifest snapshot](manifest.json).

## What the optimizer receives

| Field | Meaning |
|---|---|
| `rgb` | Lossless 1440 × 900 PNG, photographed by the front camera. Unsupported pixels are exactly black. **Use the weight**, not RGB black, as the mask. |
| `weight` | Single-channel uint8 PNG, scaled by 1/255. Zero means no color loss. “Confident” in the CSV, audit and plates means ≥128/255. |
| `sky` | Separate uint8 flag: 255 = confidently observed sky for a no-opacity loss, 0 = no claim about sky. All flagged pixels have zero RGB weight. Thin dark wires and uncertain silhouette pixels are excluded. |
| `depth` | uint16 PNG in **centimeters of target optical Z**, not Euclidean range. Zero = no depth prior (including sky). Use the RGB weight as a weak-depth validity/strength mask. |
| `T_world_cam`, `K` | Camera-to-world OpenCV axes (+X right, +Y down, +Z forward); pinhole [965.028114, 965.028114, 720, 450], 50° vertical. |
| Provenance | Station, distance along slice, pose variant, target time, base RGB frame/time and effective time, 3D and horizontal source–eye distances, neighbour frame/time, maximum temporal mix, and depth keys. |

Current `v2fit --pseudo` reads RGB, weight, pose and intrinsics. It **does not consume** the extra sky/depth fields or automatically scale by source distance. Those fields are for a separate no-opacity loss and weak depth prior if the optimizer engineer enables them. No synthetic ground, lidar-like guesses, pillar imagery, history fills or scene colors are embedded in the targets.

## Construction and limits

1. Join the four recorded 30 Hz viewer chase tracks and interpolate translation and rotation at each meter of *vehicle* travel. Include the endpoints (last spacing 0.934 m). At station 0 and every fifth station, try ±2 m lateral, ±1.5 m vertical and ±8° world-vertical yaw, using the same base pair. Offset views require ≥2.5% confident pixels. Keep low-coverage default poses with any nonzero RGB evidence.
2. Choose the front exposure closest in station to the delayed eye and its bracketing front neighbour. Read full-resolution **2896 × 1876** photographs and calibrated photometry. Ground projects through the local measured plane with the source-row exposure pose. For structures, independently rasterize the two bracketing, measured front depth sheets; inverse-Z consensus supplies the warp, not splat depth. Exclude the stitch exposure hold set from source RGB and the stitch-rejected depth windows.
3. Confidence accounts for calibrated lens/mover masks, their margin, ground/sky semantic agreement, insufficient depth, depth disagreement in projected pixels, close surfaces and depth cliffs, Jacobian singular values/folds, and disagreement with the other real exposure. Ground gets high weight only inside observed road. Temporal blending is allowed only where the projections already agree in a 9 × 9 neighborhood. Non-ground alignment is bounded at 0.75 target pixels; ground never uses flow. The second frame cannot fill an unsupported base pixel.
4. Audit **decoded PNGs**, counting strong edges only inside the eroded confident mask and requiring **both** candidate endpoints inside it. Use the unmixed, warped base photo as independent reference for the 3–40 px same-facing parallel-edge test. The matched lossless solo and injected 9 px double-image controls use the same masks. The 0.15% floor from the earlier H.264 study used different encoding and full frames; it is not this set’s control.
5. Render the actual static `scene/stitched-opt-v3` tiles and their sky at the exact same time, transform and intrinsics for 20 stations spaced in time. Each plate has the masked teacher, RGB weight and splat, with the orange box repeated at native resolution. An unused third front exposure checks colors in mutually visible confident pixels. High-pass detail is measured where that photograph has texture. This is corroboration from another low camera exposure, **not independent ground truth at chase height**; large unsupported regions are intentionally absent.

## Evidence

[Per-view coverage and audit](views.csv) · [20 comparison measurements](comparison.json) · [Aggregate audit](result.json) · [Optimizer read/crop receipt](ingest.json)

**[Overview 1](sheet-1.jpg)** · [85.00 s](0000-chase.jpg) · [87.88 s](0009-chase.jpg) · [90.76 s](0021-chase.jpg) · [93.62 s](0035-chase.jpg) · [96.50 s](0042-chase.jpg)

**[Overview 2](sheet-2.jpg)** · [99.46 s](0054-chase.jpg) · [102.37 s](0073-chase.jpg) · [105.25 s](0094-chase.jpg) · [108.07 s](0106-chase.jpg) · [111.05 s](0126-chase.jpg)

**[Overview 3](sheet-3.jpg)** · [113.93 s](0160-chase.jpg) · [116.87 s](0208-chase.jpg) · [119.75 s](0262-chase.jpg) · [122.66 s](0320-chase.jpg) · [125.52 s](0379-chase.jpg)

**[Overview 4](sheet-4.jpg)** · [128.44 s](0440-chase.jpg) · [131.34 s](0501-chase.jpg) · [134.20 s](0561-chase.jpg) · [137.13 s](0622-chase.jpg) · [140.00 s](0681-chase.jpg)
