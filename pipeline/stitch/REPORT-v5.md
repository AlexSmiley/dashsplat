# Standing-world splats · v5

**Published, but not ready for a side-by-side with Tesla’s screen.** At 112 s the pale strip that crossed the driving lane in the first candidate is gone. The near-right tree has a visible trunk and a less uniform crown. At 135 s several traffic lights are easy to see precisely because their supporting arms are missing. Neither named office reads as a building on the default drive.

[Open the photo-off viewer at 112 s](http://10.0.50.51:8090/?scene=stitched-opt-v5&photo=off&t=112&paused=1) · [Drive 110–125 s](report-v5/drive.mp4) · [Traffic 300–315 s](report-v5/traffic.mp4) · [Measurements and playback receipts](report-v5/evidence.json)

## What the actual drive shows

| Matched pose | v4 → v5, photos off | Reading |
|---|---|---|
| 93 s | [v4](report-v4/v4-93.jpg) / [v5](report-v5/93.jpg) | The kerbed islands and paired yellow lines are clear. Several new straight posts terminate in midair without fixtures. |
| 100 s | [v4](report-v4/v4-100.jpg) / [v5](report-v5/100.jpg) | The turn markings and edge of the island read well. The nearest tree breaks into flecks instead of individual branches. |
| 112 s | [v4](report-v4/v4-112.jpg) / [v5](report-v5/112.jpg) | The lane is open; the nearest right trunk meets the verge. The crown has dark clumps separated by pale smears. The median pole reads as a blunt vertical stroke. |
| 118 s | [v4](report-v4/v4-118.jpg) / [v5](report-v5/118.jpg) | The fork and adjacent painted island stay in place. Thin upright fragments stand out against an otherwise empty horizon. |
| 128 s | [v4](report-v4/v4-128.jpg) / [v5](report-v5/128.jpg) | The left crown has more local contrast, but the foreground still smears and the distant white office is absent. |
| 135 s | [v4](report-v4/v4-135.jpg) / [v5](report-v5/135.jpg) | Dark signal heads and a green light are visible across the intersection. Several appear to hang on nothing. |
| +60° orbit | [101 s: v4](report-v4/v4-101-orbit.jpg) / [v5](report-v5/101-orbit.jpg); [128 s: v4](report-v4/v4-128-orbit.jpg) / [v5](report-v5/128-orbit.jpg) | These are different views of the same world. At 101 s the foreground crown still fills the view with a wash of colour. At 128 s trunks remain rooted, but neighbouring canopies look like one continuous hedge. |

On the extensions, [199 s](report-v5/199.jpg) has a recognisable crossroad and an actual horizontal signal support. [290 s](report-v5/290.jpg) shows the overpass at the correct place; the deck edge is scalloped and the road beneath has streaks. At [308 s](report-v5/308.jpg) the through lanes remain open, but a blunt roadside post and floating fragments under the right frontage are conspicuous. At [324 s](report-v5/324.jpg) rows of trees track the bend; their bases and outer silhouettes are incomplete. In traffic playback the right curb and verge stay aligned as the palms go past. Individual branches, storefront bottoms and distant traffic infrastructure do not hold together.

Both clips are actual Chrome playback with photos off, 15.00 s each, 450 frames, H.264 at 960×600. The clips retain captured frames selected against the drive clock, without motion interpolation (445 and 439 distinct frames; maximum clock errors 95 and 99 ms). The earlier v4 whole-scene flash did not recur in these recordings; this does not establish a viewer fix. The stills are 1440×900. Each capture checked the **18,211,768**-splat scene key. The highest settled still count was 4.46 M at 118 s, against the 4.5 M desktop budget.

## Trees, surface contacts and capacity

The 85–140 s field has **4,081,062 objects** versus 1,207,602 in v4 (3.38×). Median largest Gaussian axis fell from **15.2 to 9.8 cm** (90th percentile: 34.2 to 20.3 cm). **78.4%** of released objects exceed opacity .8, versus 10.9% in v4. The native witness pass pruned 338,305 candidates. Angular-resolution physical crops use the calibrated lens at its native focal length; confident teacher supervision starts above the estimated ground surface. The first slice used 1,933 physical, 1,182 teacher and 485 agreement-gated Difix updates. The extensions used physical/teacher updates only: 750/750 and 1,110/990. All appearance is static world-space colour, with no photographic cut-outs or per-view colour.

562 physical-camera triplets contributed 201,634 triangulated canopy samples; 30,742 nearby old centres were removed. [Native photos / v4 / v5](report-v5/native-object-crops.jpg) show why density and opacity are not the same as resolution. In the first crop the photo has visible crossing branches; v5 still merges them. The third crop gains contrast but produces pale chunks where the photo has leaves and sky.

| Held native crops, 85–140 s | n | RGB error ↓ | Strong-edge contrast / photo (1 = match) | Object alpha on object | Object alpha on labelled sky ↓ |
|---|---:|---:|---:|---:|---:|
| Crown | 110 | 24.12 → 24.27 | .111 → .168 | .863 → .891 | .324 → .340 |
| Building | 100 | 31.72 → 31.64 | .120 → .166 | .748 → .776 | .223 → .258 |
| Pole | 85 | 26.76 → 24.73 | .177 → .171 | .691 → .848 | .166 → .201 |
| Sign | 22 | 28.37 → 25.78 | .116 → .205 | .858 → .948 | .237 → .229 |

Crown contrast improved, **sky-gap separation did not**. The 317 crops use the same held exposures, deterministic locations, reference edge masks and independent compositor for v4/v5. The compositor reaches 215 m, short of the office distances in the identifying photos. The isolated office comparison below intentionally removes that limit and is not a viewer still.

Ground never entered the optimiser. The 85–140 s slice reuses v3 ground/verges; each extension uses the original tagged mosaic. These remain separate `layer: ground` tiles for the atlas renderer. The standing-world scene contains no atlas or baked road-sharpening pass. Over owned ground the low horizontal-centre check found **zero** base-skirt candidates among 3,497,808 first-slice dense object samples (height <10 cm, |normal.z| >.76). This checks geometry, not every contact in the image: the 128 s orbit shows grounded trunks, but some traffic verge and canopy edges visibly float.

## Uprights: which assemblies actually survived?

| Slice | Mask components | Track proposals | Continuous shafts | Panels on continuous shafts | Panels without verified mount | Partial (including broken shafts) | Duplicate / unverified / removed | Unassigned sightings |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| [85–140](report-v5/uprights-85-140.json) | 5,331 | 554 | 23 | 3 | 31 | 74 | 10 / 413 / 0 | 1,120 |
| [190–215](report-v5/uprights-190-215.json) | 1,159 | 97 | 2 | 3 | 13 | 12 | 2 / 60 / 5 | 401 |
| [286–330](report-v5/uprights-286-330.json) | 7,426 | 751 | 9 | 1 | 51 | 47 | 15 / 545 / 83 | 1,812 |

The linked catalogues list **every proposal and every unmatched mask component**, with camera/frame/box witnesses. A proposal is within 40 m of the driven path; an unmatched 2D sighting has no verified range. A shaft requires independent support down its height and surviving splats at 67 sampled levels from the base to the observed top. A mounted panel must connect to a shaft that still passes after pruning. This is not an exhaustive count of physical poles: masks can miss them and unassigned sightings can repeat one object.

Mounted IDs: **85–140:** signal heads 6-508→5-063 and 6-509→5-001; sign 7-537→5-091. **190–215:** signal heads 6-081→5-041, 6-087 and 6-088→5-018 (two heads share one shaft). **286–330:** sign 7-711→5-389. The first slice also has nine supported arm segments on seven proposed shafts. A complete photographed lamp or mast needs its arm and head; neither a continuous shaft nor a mounted head proves the whole assembly. No complete-lamp count is claimed.

## Offices and the traffic visibility correction

The brick office contributes 25,760 + 124,412 splats; the white office 79,683 + 60,802. The four physical planes were fitted from displaced front images and reside in 13 object tiles. The [held photograph and isolated plane render](report-v5/office-reprojection.jpg) show the brick window grid and roof step, and the white window bands and corner. Hidden lower storeys are missing, not filled. Fitted ranges in the identifying exposures: **305–336 m** (brick), **384–474 m** (white).

The published viewer actually used a 172 m streaming radius and 50–140 m distance haze (zero splat alpha by 150 m). At 112 s it loaded **0/13** office tiles; at 128 s it loaded **6/13**. At both poses all tiles inside the streaming radius had loaded, with zero failures. At 128 s the nearer brick pieces were outside the forward field of view and the nearest white piece was 284 m from the car. No named office centre was both in-frame and within 150 m at any of the six standard chase times. **Higher opacity cannot make these windows appear in this drive.**

An early 112 s viewer render showed a storefront in the traffic lane. The 600 near-camera centres in the offending patch all came from the generic facade fit. A whole-plane road-clearance and displaced-range check kept 12 of 226 generic planes and removed 1,506,316 splats; the named offices were not affected. Both the final still and motion recording show the lane open. In the extensions, a six-lens visibility pass removed unsupported depth sheets, leaving 1,114,482 of 3,322,672 objects at 190–215 s and 3,224,078 of 9,363,787 at 286–330 s. The released extension scores are reported below.

The mover-removal masks excluded moving observations without treating parked instances as moving. No parked decision in the audited physical keyframes was marked for removal. [Parked-track audit](report-v5/parked.json): 8/10 traffic tracks and 1/3 eligible 190–215 s tracks had nearby static samples in at least two tested views (median distance <45 cm per view). This is positional support, not a complete vehicle outline. Parked cars are not all visibly recovered; the absence of the moving vehicles from the static scene is intentional.

## Held-out measurements

Six physical lenses, same 852 exposures and perceptual crops as the rooted v4 baseline. Per-camera image metrics use the existing static-pixel evaluation; the object-detail table above uses native angular-resolution patches. Lower LPIPS is better.

| Exposures | n | PSNR dB | SSIM | LPIPS | Static coverage |
|---|---:|---:|---:|---:|---:|
| all | 852 | 19.74 → 19.80 | 0.691 → 0.683 | 0.1714 → 0.1616 | 93.0 → 93.9% |
| clean | 361 | 19.90 → 20.00 | 0.706 → 0.699 | 0.1601 → 0.1500 | 93.3 → 94.1% |
| front | 142 | 19.77 → 19.82 | 0.744 → 0.721 | 0.1057 → 0.1165 | 92.6 → 93.7% |
| back | 142 | 20.43 → 20.57 | 0.799 → 0.788 | 0.1727 → 0.1757 | 94.0 → 94.6% |
| left pillar | 142 | 20.12 → 20.13 | 0.636 → 0.636 | 0.2325 → 0.2152 | 93.0 → 93.2% |
| right pillar | 142 | 18.32 → 18.70 | 0.575 → 0.589 | 0.2198 → 0.1892 | 90.5 → 92.4% |
| left repeater | 142 | 19.92 → 19.88 | 0.715 → 0.703 | 0.1456 → 0.1331 | 94.8 → 95.2% |
| right repeater | 142 | 19.88 → 19.73 | 0.679 → 0.662 | 0.1521 → 0.1398 | 92.9 → 94.0% |

All-camera LPIPS improved but front-camera LPIPS and overall SSIM worsened. Held elevated teachers use confidence ≥.5 and hold out every tenth station with all its offsets. They are pose references, not independent chase-camera photographs. RGB MAE is on a 0–255 scale:

| Slice / camera | Held default / offset | Default RGB / above-ground MAE | Offset RGB / above-ground MAE |
|---|---:|---:|---:|
| 85–140, rooted v4 | 69 / 408 | 6.32 / 15.31 | 6.11 / 14.98 |
| 85–140, published v5 | 69 / 408 | 7.89 / 16.92 | 7.60 / 16.66 |
| 190–215, v5 | 18 / 105 | 7.72 / 20.52 | 7.55 / 20.55 |
| 286–330, v5 | 64 / 380 | 7.95 / 20.56 | 8.01 / 20.96 |

## Release identity

The promoted field is **8,419,622 object + 9,792,146 ground splats**, 924 static tiles, 601,217,620 compact payload bytes. Payload counts, bboxes, normals, quaternions, lifetimes, rig, drive, separate layer tags and cache entries were verified against the promotion. The combined tile SHA-256 digest is `77151be1b92e64c80ef71cc8faa949ab08f66ad124907135d17e48ed4ff47036`. The review used the shipping viewer bundle `index-DCk1CRgK.js`, not the concurrent viewer working tree.
