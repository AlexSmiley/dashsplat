# Viewer-ready splat v3 · 85–140 s

[Open the published 112 s chase](http://10.0.50.51:8090/?scene=stitched-opt-v3&t=112&paused=1). **Use v3 for the default chase on this slice.** V2 already had rooted trees occupying real depth; the most convincing v3 change is seeing those trees stand in photographed grass instead of a pale, speckled apron. V2 remains available for comparison. V3 is still soft at very close orbit distances, and the distant office is incomplete.

## Published viewer, not the offline renderer

Captured the shipping viewer in Chrome after running the Studio tile-cache builder. Each URL was paused at its requested drive time, with ~15 s for tiles. All seven final screenshots are unretouched 1440×900 captures, with the viewer’s natural grade, haze, AA, apron, path, car and HUD. The source for the separate numerical camera checks is the official 1440×900 `chase-track.ts` output and the shared chase pose function.

| Site | What is visible |
|---|---|
| [93 s](report-v3/viewer-93.jpg) | Two planted islands, clean arrows, and the tan commercial frontage with dark window openings. The far roofline is still broken by foliage and uncertain depth. |
| [100 s](report-v3/viewer-100.jpg) | Individual trees retain trunks and separated crowns; the grass at right is more continuous. The near island’s kerb is still incomplete. At [101 s, ego-right](report-v3/viewer-101-right.jpg) a near crown still blurs into a sheet; in the left foreground test crop the alpha-weighted median surface range is **3.16 m**. [Absolute 40°](report-v3/viewer-101-raised.jpg) shows the near kerb but also unobserved terrain beyond the plantings. |
| [112 s](report-v3/viewer-112.jpg) | The right verge reads as grass and the row of trunks still meets it. The clear sky between the lamps remains clear. The distant office and all of the lamp-head/shaft detail are not independently resolved. |
| [128 s](report-v3/viewer-128.jpg) | The near median is green rather than frosted. [Ego-right 60°](report-v3/viewer-128-right.jpg) shows the foreground trunks and the second row behind them: the selected object-dominated ROI spans 39–75 m (10th–90th percentiles of composited range). Some crown texture remains smoky. The white office does not emerge cleanly through the haze. |

## What changed, and why

1. **Frost was both missing ground and weak objects.** In the 112 s right-verge ROI (x 980–1339, y 555–734), 54.2% of v2 pixels had original ground alpha below .5 and 30.7% had *total* alpha below .5. The real viewer shows pale apron through these holes. Of 254,716 object centres in the first metre, 39.8% had alpha below .18. V3 removes 118,382 first-metre leftovers using opacity, measured surface support and overlap with the frozen road. In missing verge cells only, it adds 29,299 shallow, depth-supported turf/kerb surfels, then 78,756 more supported by displaced photos on the ground plane (median 13 views for the latter). They are ground tiles, not a hovering grass cloud. In the same ROI the mostly uncovered fraction falls to **0.0%**. This is a local measurement, not a claim that all verge holes are closed.
2. **Crown darkness was largely in the photographs; translucency was introduced.** Among 470,392 height/colour-selected crown candidates with substantial photographic support, stored DC was a median 3.9 RGB levels *brighter* than the corresponding calibrated photograph. Across 525,733 candidates the median alpha was only .146. V2’s repeated visibility thinning and averaging of edge colours were more important than a global exposure deficit. V3 preserves the dawn exposure, selects a local high-resolution colour witness bounded by the multi-image mean, prunes detached weak fringes and contracts broad boundary kernels. The median core opacity rises .211 → .688. Core positions, existing trunks and road are not globally displaced or brightened. These are colour/height proxies for foliage, not semantic tree counts.
3. **Distant coverage comes from real images.** Across 1,094 real-photo triplets, local triangles with consistent optical depth and static colour supplied 81,725 repeat-confirmed surface patches. The same tracks support piecewise vertical facade planes, with 26,872 texels passing displaced-colour and sky checks. We also restored optical thickness on 224,558 supported far candidates from v2. The scene ships the 1024×512 live-v5 sky byte-for-byte. The viewer itself both recolours distant splats into the sky and fades their opacity at the far limit. Geometry beyond that limit cannot be made visible merely by adding more splats; no scene content was moved nearer to defeat the haze.
4. **Poles needed the actual viewer feedback.** The first pass triangulated upright ridges and admitted 300 photometrically supported segments, but the viewer showed dark sticks hanging above trees at 100, 112 and 128 s. A ridge-search window was too wide to use as a physical shaft diameter, and partial high-elevation tracks did not prove a pole touched ground. V3 measures half-contrast widths at multiple resolutions and requires a contiguous base-to-shaft run. It retains **28 uprights**, inserted after opacity pruning and subject to the swept-body/road guard. These are measured upright structures, not 28 semantically identified signs. Flat sign surfaces use the same photo-track patches as facades; sign text and missing lamp heads were not fabricated.
5. **Original road remains identical.** All 36 original fused ground tile files remain hard links and their complete metadata records match v2. The additional 108,055 verge surfels have separate tile IDs. In the 261,823-pixel 112 s road ROI the final RGB MAE against the frozen mosaic is **0.120 levels out of 255** (v2: .108). The lamp-gap ROI still has zero pixels above .15 object opacity. The verified scene has **3,845,068 ground + 1,188,730 objects = 5,033,798** across 419 tiles.

## Held-out real photographs

Same six-lens, fixed physical split, native row timing, static valid mask, background and reference-selected LPIPS crop as v2. The original 36 ground tiles keep their previously documented provenance; the new stages used held-in images only. Values are frame means. Lower LPIPS is better.

| Set | PSNR (dB) v2 → v3 | SSIM v2 → v3 | LPIPS v2 → v3 | Coverage v2 → v3 |
|---|---:|---:|---:|---:|
| All 852 | 19.716 → **19.959** | .7003 → .6952 | .1742 → **.1677** | 89.93% → **93.67%** |
| Common clean 361 | 19.910 → **20.154** | .7158 → .7099 | .1627 → **.1560** | 90.42% → **94.02%** |

All-frame LPIPS by lens: front .1187 → .1205, rear .1757 → .1757, left pillar .2542 → .2295, right pillar .2153 → .2000, left repeater .1363 → .1366, right repeater .1447 → .1440. Most of the perceptual gain is on the two pillars. SSIM slipped, and the front and left repeater did not improve in LPIPS.

## Reproduce on Studio

With the v2 immutable training caches, held-out split, `scene/stitched-opt-v2`, calibrated photographs and live-v5 sky present, run from the Reality Splat root with the environment in the [v2 README](README.md#dense-optimisation-v2):

```sh
python -m pipeline.stitch.v3evidence
python -m pipeline.stitch.v3terrain && python -m pipeline.stitch.v3grass
python -m pipeline.stitch.v3tracks && python -m pipeline.stitch.v3facades
python -m pipeline.stitch.v3lines && python -m pipeline.stitch.v3shafts
python -m pipeline.stitch.v3build scene/stitched-opt-v3 candidate-B  # fresh destination
```

On basebook, from `viewer/`, run `bun tools/chase-track.ts ../scenes/road-stitched-v2 85 140 30 1440 900 > /path/to/track.json`. Then `bun ../pipeline/stitch/v2camera.ts ../scenes/road-stitched-v2/drive.json /path/to/track.json /path/to/viewer-cameras.json 93,100,101,112,128`. Copy the matrices to Studio’s `work/stitch-v3/viewer-cameras.json`, run `python -m pipeline.stitch.v3proof` on Studio, then `bun run cache:studio` on basebook. Studio evidence is in `work/stitch-v3/proof/`; first-pass scene and diagnostics are under `work/stitch-v3/`.

**Extension status:** At the post-viewer check, 286–330 s had 968–971 scheduled exposures per lens, but no posed depth arrays yet in either indexed depth location. The producer was segmenting that window. 190–215 s had no scheduled exposures in this snapshot. The published v3 covers 85–140 s only.
