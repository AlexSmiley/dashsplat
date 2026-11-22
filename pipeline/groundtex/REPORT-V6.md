# Ground beyond the kerb

The [published default chase view](http://10.0.50.51:8090/?t=118&paused=1) now has continuous ground on both sides at 118 s, on the right at 128 and 300 s, and on the left of the junction at 322 s. The road retains its measured paint sharpness. The low-detail areas beyond the road are not all photographs: **53.1% of the covered surface inside 40 m is predicted**, and every such texel has an explicit label.

## Published viewer evidence

| Default chase | Before | Published | Formerly empty ground: grey samples before → after |
|---|---|---|---:|
| 118 s, either side | [before](report/v6/before-118.jpg) | [after](report/v6/check-118.jpg) · [ground labels](report/v6/ground-rays-118.jpg) | 1,355 → 0 |
| 128 s, right verge | [before](report/v6/before-128.jpg) | [after](report/v6/check-128.jpg) · [ground labels](report/v6/ground-rays-128.jpg) | 20 → 0 |
| 300 s, right carriageway edge | [before](report/v6/before-300.jpg) | [after](report/v6/check-300.jpg) · [ground labels](report/v6/ground-rays-300.jpg) | 4,121 → 0 |
| 322 s, left of junction | [before](report/v6/before-322.jpg) | [after](report/v6/check-322.jpg) · [ground labels](report/v6/ground-rays-322.jpg) | 2,262 → 0 |

Both sets are real browser screenshots after ground and splat streaming settled, with no scene or camera override. The [ray test](report/v6/view-rays.json) samples a central HUD-excluded region at one-third resolution, intersects the [actual camera](report/v6/poses.json) with the mesh and counts flat pixels close to the viewer’s neutral background on newly covered ground inside 40 m. This narrowly detects the grey void, not all visually smooth pavement. Overlays show the **underlying** ground provenance despite foreground splats: green observed, amber predicted. A blue/grey paved surface is not automatically a gap.

Played the default scene at 1× from 110 through 125.02 s: [112 s](report/v6/play-112.jpg) · [118 s](report/v6/play-118.jpg) · [near 125 s](report/v6/play-125.jpg). The [61 playback samples](report/v6/playback.json) had 129–182 resident ground tiles; streaming continued in motion (up to 44 pending). No broad grey apron reappeared in those captured frames. Shadows, soft distant verges, some source-boundary steps and pre-existing floating or irregular tree splats are still visible. Predicted ground does not reconstruct buildings, trees or detailed car parks.

## What the pixels represent

The original 1,160 tiles keep their 1/2 cm pitch; **2,052 new tiles use 4 cm**. A mixed tile uses its existing finer raster. There is no overlapping outer mesh. From 5,174 non-held native exposures, metric depth and rolling-shutter projection provide surface candidates; [41,165 local image matches](report/v6/registration.json) constrain cross-view source coordinates where texture permits. Final admissions require flat depth, separated viewing positions, local colour agreement, road/pavement/low-terrain labels, and no sky, vehicle or removal mask. A winning photograph supplies each observed texel. The coarse voting and geometric guide do not supply RGB detail. [Native witness images](report/v6/native-witness.jpg) show where using depth alone would have projected hedge/wall pixels; all 732 sampled non-ground selections in those six views were removed from their original owners. These are model-based screens, not proof that every semantic classification was correct.

| Area-weighted provenance | Observed | Predicted | Predicted share |
|---|---:|---:|---:|
| Covered ground ≤40 m | 125,328 m² | 142,001 m² | 53.1% |
| Entire atlas including taper | 127,304 m² | 166,455 m² | 56.7% |
| Newly sourced/filled area, excluding retained v5 photographs | 64,522 m² | 166,455 m² | 72.1% |

The display-footprint tile sets have predicted shares of **54.9%, 54.5%, 56.2%** for 85–140, 190–215 and 286–330 s respectively. These count entire nearby tiles, not just pixels visible in a particular frame. Standing-object footprints cut out 2,387 m² inside 40 m; the export found no other uncovered interior in that band. Beyond 40 m coverage tapers to zero at 44 m. An independent path-distance check found zero covered texels in 1,496,600 samples beyond 44.2 m. [Decoded area, colour, opacity and shared-height checks](report/v6/verified.json).

Prediction extends only low-frequency adjacent asphalt, concrete or grass colour. Bright, dark and high-contrast seeds are excluded; it does not copy a marking, invent a kerb or letter, or generate a missing building. The conservative fallback is why the predicted share is substantial. Every tile has an aligned, lossless label PNG: **0 unsupported / 127 predicted / 255 observed**. Alpha remains coverage. **No viewer change is needed for LOD or normal display**; [FORMAT.md](FORMAT.md) specifies how a viewer can tint or fade predictions while keeping depth and splat suppression consistent.

## Road, apron and holdouts

The 128 s dash has a 20–80% median edge width of **1.26 cm in both exports** (230 transverse edges each). At the old road-to-ring boundary, the average adjacent RGB step is 5.72/255; nearby within-surface steps average 1.29/255, so the join remains measurable. On 71,541 observed low-terrain cells overlapping splat grass, mean RGB error against the splat reference fell **10.43 → 7.80/255** with tone matching. [Surface measurements](report/v6/surface-check.json).

The [91.4 s entrance apron](report/v6/apron.jpg) now uses a single left-repeater exposure (3333, median range about 10 m) where the independent angle in frame 3507 also confirms flat pavement. That removes the strongest rectangular exposure joins without blending photographic texture. The second photograph checks the patch; it is not mixed into it. A shallow viewing angle still stretches texture, and the upright-object footprint stays open. In the 9.3 × 11.55 m close-up, nearly transparent samples fell from 114,228 to 24,529. [Apron receipt](report/v6/apron.json).

The three worst first-slice fresh hotspots were owned by **front** exposures at about 8–11 m, rather than by a switch to pillar imagery. Restoring jointly supported earlier front mappings on paint interiors reduced fresh flags at those tiles from 129 → 52, 27 → 9, and 45 → 23. This isolates a local registration/ownership problem; it does not establish a global range threshold. Neither held split supplied source pixels. Because the first-slice fresh flag map was used to locate repairs, that split is now a regression check, not a blind validation set.

| Slice | Fixed v5 → v6 | Other alternating holdout v5 → v6 |
|---|---:|---:|
| 85–140 s | 0.83 → 1.02% | 1.84 → 0.97% |
| 190–215 s | 0.29 → 0.36% | 0.97 → 1.10% |
| 286–330 s | 2.13 → 2.26% | 2.16 → 1.97% |

Paired decoded-image test on common opaque support, using an adjacent unused photo as a control. Counts and denominators vary with the number of detected candidate edges. The first-slice fresh regression is reversed, with a small fixed-split cost; the other slices did not all improve. [Per-frame and per-tile counts](report/v6/paint.json).

## Publication

The regular v5 and default movers scenes share **6,425 hard links**: 3,212 textures, 3,212 masks and the manifest. **64,449,098 active bytes**, including 6,318,222 mask bytes. All 54,207 shared height comparisons were identical. The Studio cache was rebuilt; both scene manifests served SHA-256 `0db3452a28e6a5da61db099ec2795a8e7d64ab7d846d11149b5dba9171d748a9`. [Publication receipt](report/v6/publication.json) · [Rebuild commands](README.md). Full source maps, class masks and tiles stay on Studio; the repository contains only implementation and compact proofs.
