# Viewpoint-selected RGB-D · road and car-park exit

**Result: do not export this version.** Photograph textures restore fine detail that v5 cannot carry. They also make incorrect depth much more conspicuous: a crisp piece of a tree or a crossing lane is still wrong when it projects across the street. At 112 s there is no old overhead slab, but a dark band of misregistered roadside texture now cuts across the distant road. The moving exit and the 60° views fail the grounding test.

[Road film: 105–120 s](report/road-105-120.mp4) · [Exit film: 93–104 s](report/exit-93-104.mp4) · [Road contact sheet](report/road-sequence.jpg) · [Exit contact sheet](report/exit-sequence.jpg) · [Private bank and renderer spec](FORMAT-mesh-prototype.md)

## What improved, and where it failed

On the straight, lane boundaries, arrows, traffic lights, utility lines and windows read as parts of the actual street. At 135 s, the real green lights appear above the intersection; in v5 that area was largely empty. Grass, nearby foliage and sky boundaries still show cuts between cameras. Even after residual exposure balancing, road from different sensors has visible pink/grey transitions. On the exit, photo-projected kerbs turn through the frame while the underlying street and distant buildings disagree. Orbiting shows parts of buildings missed by v5, but exposes a jagged boundary between the photographed road and the v5 apron. The ego box is 4.5 × 1.85 × 1.48 m, solely a framing witness.

| Chase | Primary front exposure | Arc behind car | Direct RGB-D or source sky | P90 edge strength: source / IBR / v5 |
|---|---:|---:|---:|---:|
| 100 s | 96.80 s | 13.9 m | 78.4% | 21.8 / 21.6 / 5.0 |
| 112 s | 109.74 s | 20.2 m | 82.1% | 22.0 / 18.6 / 4.4 |
| 128 s | 126.99 s | 21.5 m | 79.8% | 22.7 / 19.6 / 3.9 |
| 135 s | 133.88 s | 23.5 m | 83.1% | 21.8 / 17.7 / 4.1 |

Edge measurement: Sobel luminance magnitude ÷8, code values/pixel. The target samples are inside same-source static non-ground patches, away from borders. For each sample the winning world point is projected back into its actual source with row timing. Source gradients are at native texture sampling density (1448 wide), output gradients at 1440×900; these are not equal-angular-resolution MTF or held-out photometric scores. Holes are excluded from sharpness and counted separately. Sharp edges can still be misplaced.

Each full-width proof below has four panels: **v5, IBR, the spatially selected front photograph, and the physical front photograph nearest playback time**. Both photos use the acquisition photometry, before the IBR display adjustment. They are different viewpoints, not target ground truth. Each world render is 1440 × 900; the 28 px legend is outside it. The percentages are the fraction of pixels with a direct mesh contribution (including source sky), **before** fallback. Sky can additionally use the panorama and v5 near objects.

| Time | −60° | −30° | Chase | +30° | +60° |
|---|---:|---:|---:|---:|---:|
| 100 s | [78.3%](report/100--60.jpg) | [88.7%](report/100--30.jpg) | [78.4%](report/100-+0.jpg) | [51.5%](report/100-+30.jpg) | [50.6%](report/100-+60.jpg) |
| 112 s | [50.9%](report/112--60.jpg) | [75.1%](report/112--30.jpg) | [82.1%](report/112-+0.jpg) | [77.5%](report/112-+30.jpg) | [65.1%](report/112-+60.jpg) |
| 128 s | [58.3%](report/128--60.jpg) | [77.3%](report/128--30.jpg) | [79.8%](report/128-+0.jpg) | [72.5%](report/128-+30.jpg) | [52.4%](report/128-+60.jpg) |
| 135 s | [56.9%](report/135--60.jpg) | [84.1%](report/135--30.jpg) | [83.1%](report/135-+0.jpg) | [80.2%](report/135-+30.jpg) | [58.7%](report/135-+60.jpg) |

At 100 s the chosen front heading differs from the target by 54.4°: being near in XY on a turn does not make the front photograph a matching view. The differences at 112, 128 and 135 s are 7.4°, 4.1° and 1.9°. This also limits the horizontal-baseline approximation: for front-facing distant static surfaces the 4.5 m rise is relatively benign, but it reveals backs and undersides of near objects that the road-level photos never saw. At 100 s/+60°, the bottom third is 95% fallback; at 112 s/−60° it is 88%. Some of the upper-third misses are between the source sky mask and a lifted tree silhouette. Filling them with a panorama hides the hole, not the placement error.

## What changes during playback

| Clip | Frames | Mean fallback | P95 frame fallback | Matched luma change: same / new source | Change >20: same / new source |
|---|---:|---:|---:|---:|---:|
| road | 450 | 20.3% | 27.6% | 2.7 / 11.7 | 2.4% / 17.1% |
| exit | 330 | 18.4% | 31.7% | 2.7 / 13.4 | 2.2% / 19.2% |

Both clips are continuous, 30 fps, 1440×900. For temporal comparison, sample static non-ground every eighth pixel; reproject the current winning depth into the previous chase pose, keeping static points with axial depth agreement within max(0.35 m, 5.5%). Compare display luma on a 0–255 scale. “New source” means a different camera **or** keyframe won at that target point. This measures hand-over appearance on supported regions, not stability inside holes; disocclusions and rejected depths are excluded. Cross-fading does not remove jumps when the winner changes. Around 99.27 s the kerb and road exposure move independently of the trees. [Road per-frame data](report/road.json) · [Exit per-frame data](report/exit.json).

## Movers and stopped control

[Actual traffic observations and mover contribution](report/traffic.jpg). All old instance-labelled pixels, including parked/unknown tracks, are excluded from static projection. At 102.83 s the source actually sees the white crossing SUV. Its current front mask has 2,891 depth samples and produced 258 triangles; they project to 54–64 m axial distance. Within their projected bounding rectangle, the winning static mesh is 20–23 m away (p10–p90). **Zero** direct instance pixels survive in the chase. There is no ground truth image at the chase position that proves the correct occlusion; this is nevertheless a failure to demonstrate faithful traffic. At 131.77 s, 350 direct instance pixels survive, 25 labelled confirmed-moving, all from the current front exposure 2.09 ms from T. No claim of accurate rigid lead-car tracking.

At the stopped light (200.0 s, 0 m/s), the actual front exposure nearest the chase ground position is at **191.598 s**, **19.218 m** back along the path, with a **0.104 m** horizontal residual. There are no production RGB-D keyframes within ±2 s of 200.0 s; the selection witness is telemetry, not a rendered light. An [actual 0 m/s render at 81.5 s](report/stopped-81.5.jpg) is a negative control: the available route has not yet traversed the chase location, so only 18.0% has direct support. No fictitious earlier position was synthesized.

## Payload and implementation

The verified private bank has **5,408** timestamped camera records across 80.0–137.25 s. The extra lead-in is necessary for the 93 s target. All 546 scheduled lens keyframes in 93–104 s and all 1,642 in 105–120 s are in the audited bank. Front and rear use DA3; both pillars use MapAnything. The two repeaters use 1,025 MapAnything and 192 archived DA3 frames. Their later coverage is sparse (82/441 and 77/441 frames in 125–137.25 s); missing producer depths were not invented. The films froze the earlier bank; it lacked one 101.48 s rear keyframe which the still/audit bank recovered. Photographs and physical depth are on their original distorted image axes. The proof favours the road photographed onto the measured local surface; v5's live ground, fused objects, and its apron remain the fallback.

Payload: **1,359,689,439 bytes** of JPEG / optical-z PNG16 / mask PNG16, plus **28,254,219 bytes** of manifest, ray grids and panorama. That is 23.75 MB/s of drive for all available keyframes, excluding the existing fallback scene. With each selected frame fetched once and kept for the clip, the road requests 1,210 distinct keys (292.7 MB, **19.51 MB/s**, 31.50 MB peak rolling second); the exit requests 397 (108.2 MB, **9.84 MB/s**, 19.60 MB peak). The median simultaneous counts are 32 and 29. These are payload estimates, not a measured network player; there is no viewer export and this is not yet a light streaming budget. [Payload, stop and traffic audit](report/audit.json) · [pose-by-pose measurements](report/stills.json).
