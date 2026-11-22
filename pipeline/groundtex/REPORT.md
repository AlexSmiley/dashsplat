# Photographic ground · offline drive, 0–361.42 s

**The near road now looks photographic in the chase view. The claim holds best for lane dashes and turn arrows; it does not hold uniformly at kerbs and camera joins.** The 128 s dash has a **1.23 cm** transverse edge spread versus **1.01 cm** in the front photograph and **4.50 cm** in the optimized v4 splats. The atlas missed both the independent photograph-noise target and the single-photograph doubled-edge floor. It is a measured improvement, not a seamless ground-truth reconstruction.

**Published on anomastudio:** `$RS/scene/stitched-opt-v4/ground/ground.json`. SHA-256 `3834d29607c2e9845ff28b8f7acf6e1d44cd49202bb13b18eb0aaffc1c6e1bad` matched the verified candidate after installation and the [manifest served by the published scene host](http://10.0.50.51:8090/scenes/stitched-opt-v4/ground/ground.json). The atlas spans the full drive even though v4’s optimized object scene covers 85–140 s. [Viewer contract](FORMAT.md) · [Reproduction](README.md).

## Look at the same ground

Native crops are sampled at **1 cm**, left to right: exported v4 ground splats, decoded photographic atlas over the splat fallback, and one native photograph reprojected through its registered ground map. The comparison photograph is chosen by visible fraction and worst-direction source pixel density. Its ordinary calibrated camera colour is shown, before the atlas's smooth residual illumination fit.

| Feature | What the crop shows |
|---|---|
| [Left turn arrow](report/detail-left-arrow.jpg) · [right arrow](report/detail-right-arrow.jpg), 93 s | The tip, inner elbow and stroke end resolve, without the splats’ broad halo. The first atlas cut the left arrow; connected-mark ownership removed that cut. The right arrow's head still has a softer edge than the excellent side-camera exposure. |
| [Bicycle](report/detail-bike.jpg), 110 s | Both wheels and the rider read separately; the clearest comparison exposure is from the back camera. The bright kerb-side surface and lower-right texture still show a camera join. |
| [Hatched median](report/detail-median.jpg), 122 s | The V and its sides are recognizable. There is a texture/colour break through the right arm and a small unsupported hole near the outside stripe. |
| [Lane dash](report/detail-lane.jpg), 128 s | The lateral step present in the first atlas is gone. Across 230 transverse profiles, median 20–80% edge width is **4.50 / 1.23 / 1.01 cm** (splat / atlas / front exposure 4590). Median local luminance contrast is **30 / 52 / 44** JPEG levels. Each profile measures its own paint and shoulder levels; this one stripe is not a route-wide MTF claim. [Raw profile](report/dash-profile.json) |

Exact viewer chase poses: [100 s](report/chase-100.jpg) · [112 s](report/chase-112.jpg) · [118 s](report/chase-118.jpg) · [128 s](report/chase-128.jpg). Each panel is 1440×900 before its title strip. Both sides share the same v4 splat fallback; the right side replaces supported road with the decoded textured height mesh. These are ground-only offline renders. A missing apron by the island at 100 s, the broken verge at 118 s, and mixed terrain at 128 s remain visible. No viewer colour grade, car or HUD was baked into the textures.

## Independent checks

**Real camera, 5–15 m ahead of the axle.** Reserve every exposure whose frame number modulo 29 equals 7, plus the existing stitch physical holdout. For evaluation, take alternating reserved front exposures. Project the decoded atlas into the original row-timed camera, using only mutually observed, opaque, static road pixels. A second unused front photograph, nearest in *timestamp* across ingest gaps, supplies the physical comparison floor. Sites with less than 300 valid samples are not scored; a disagreement over 55 levels between the two real photographs is also excluded. No fitted test-image warp, gain or sharpening. Results are the median of per-frame mean absolute RGB errors in 0–255 code values.

| Interval | Independent front frames / pixel samples | Atlas | Adjacent-photo floor |
|---|---:|---:|---:|
| [85–140 s](report/heldout.json) | 33 / 471,663 | **4.82** | **2.94** |
| [286–330 s](report/heldout-286-330.json) | 26 / 370,213 | **5.03** | **3.35** |

**Doubled paint edges.** On the 33 first-slice poses, same-facing edge pairs 3–30 native atlas pixels apart must also have tangent support. Test both endpoints in a confident, paint-selected region. The independent test asks whether the pair is supported by a different held-out front exposure. The ownership test asks whether either *actual contributing photograph*, reprojected alone, already contains both edges. These isolate different failures; their denominators are not interchangeable. Both have a matched photographic control and a deliberately doubled, nine-pixel-shifted positive control. A flag is a suspected outline, not a count of proven ghost markings.

| Check | Atlas | Single-photograph control | Injected twin |
|---|---:|---:|---:|
| [Independent held photograph](report/audit-85-140.json) | 243 / 14,185 (**1.71%**) | adjacent photo: 4 / 12,804 (**0.031%**); self: 0 / 12,887 | 10,292 / 13,728 |
| [Actual source owners](report/sourcecheck.json) | 40 / 15,378 (**0.26%**) | unmixed owner: 0 / 4,046 | 4,007 / 5,012 |

**Seam visibility.** On 91,820 horizontal source-label crossings over low-gradient, non-paint pavement in those held poses, median per-view mean RGB step was **1.69 levels**, versus **0.73** in adjacent same-source pavement; median per-view p90 at seams was **3.33**. This is a code-value jump, not a perceptual threshold. Texture direction and residual misalignment remain visible in selected close-ups even when tone matches. All interior RGB is from a single exposure; no multiview detail blending or edge sharpening.

## Coverage and streaming

The 3.2 m wide probe runs from 2 m behind to 10 m ahead of the axle, sampled every 20 cm. Opaque means decoded alpha ≥250/255. Time-weighted measurements use 0.25 s stations; distance-weighted measurements use one station per travelled metre. They measure **availability, not classification precision**. [Coverage data](report/coverage.json)

| Window | Distance | Opaque by time | Opaque by distance |
|---|---:|---:|---:|
| Full drive | 3,312.21 m | 97.54% | **99.95%** |
| 85–140 s | 680.93 m | 99.83% | 99.93% |
| 286–330 s | 630.05 m | 99.97% | 99.98% |
| Initial 0–85 s, mostly stationary | 8.24 m | 89.70% | 93.13% |
| Terminal 330–361.42 s | 410.35 m | 100.00% | 100.00% |

**1,118 tiles**: 822 at 1 cm and 296 at 2 cm, each covering 10.24 m square. Of 57,755 m² opaque, 55,071 m² is on 1 cm tiles. All texture dimensions, encoded payload hashes, world bounds and mesh heights were checked. **16,569 shared-edge height comparisons matched exactly**. 56,114,010 texture bytes plus 970,333 manifest bytes = **57,084,343 bytes**, or **1.72 MB per 100 m**. This is the active streamable export; the Studio directory also retains immutable first-pass payloads for previously fetched manifests (112 MiB allocated in total). [Verification receipt](report/verified.json)

**Registration and mask limits.** 6,911 physical exposures contributed candidates, including 2,189 from the front. In 1,245 front strips with enough unique paired corners, median matched-point disagreement fell from **10.07 to 1.38 cm** after fitting the strip similarity. That is an inlier fit statistic, not a held-out pose guarantee. Side-camera similarities were less often constrained; local normal-to-stripe refinements and mark ownership reduce the consequences but do not identify a true vehicle pose. All untimestamped display textures are static. Per-exposure fits, 32 cm smooth displacement/gain lattices and ownership remain in `$RS/work/groundtex-final/state/`; raw captures and large intermediate rasters stay on Studio.
