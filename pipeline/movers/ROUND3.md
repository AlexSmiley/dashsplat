# Photographic movers: Studio handoff

The removal bank is production (unheld). The live benchmark independently omits the same 136 image masks across 17 reviewed episodes. All three use explicit camera-table frame numbers, including gaps; array position is not a shutter ID.

## Persistent identities and projection

| Run on 136 exposures | Absent | Additional box IoU < 0.5 | Aligned |
|---|---:|---:|---:|
| Published reviewed live, old clock | 22 | 34 | 80 |
| Published v3 unseeded, old clock | 52 | 30 | 54 |
| Published hindsight, old clock | 0 | 10 | 126 |
| Existing unseeded v3, scored at native shutters | 57 | 38 | 41 |
| Reviewed live, refit and rebaked at native shutters | 25 | 34 | 77 |
| Unseeded v4, native shutters, one persistent ID | 15 | 41 | 80 |
| Same v4, allowing associated fragments | 15 | 39 | 82 |
| Reviewed hindsight, refit at native shutters | 0 | 6 | 130 |

The original 52 were 36 before the matched identity rendered and 16 after it retired; the black oncoming SUV accounted for 14 after retirement. Twenty-eight had another nearby fragment of the same camera token. Only three were back-camera exposures; none had a box below 160 grid pixels in area. The counts point to confirmation and handover, not principally a tiny-box threshold.

Median first detection → available photographed primary ID: 0.790 s previously, 0.438 s now (P90 1.639 s; 17/17 timed). For a pre-existing ID the clock stops only after a released observation of this vehicle was assigned to it. Availability does not imply correct position. The table marks IDs already renderable on another assignment with an asterisk.

| Episode | ID | Absent / low IoU | First renderable delay (s) | Largest live step (m) |
|---|---:|---:|---:|---:|
| crossing-suv | 40 | 0 / 2 | 0.394 | 1.0911 |
| passing-bus | 228 | 2 / 7 | 1.147 | 1.0911 |
| black-oncoming-suv | 238 | 4 / 9 | 0.981 | 1.0911 |
| red-oncoming-sedan | 265* | 0 / 2 | 0.064 | 1.0865 |
| junction-white-suv | 299 | 2 / 0 | 1.04 | 1.0833 |
| junction-silver-crossover | 317 | 2 / 1 | 4.29 | 1.0775 |
| junction-red-suv | 326 | 0 / 0 | 0.541 | 1.0833 |
| junction-approach-suv | 298 | 0 / 1 | 0.79 | 1.0839 |
| departing-coupe | 352 | 1 / 1 | 2.377 | 1.0843 |
| oncoming-pickup | 630 | 1 / 0 | 0.242 | 1.089 |
| oncoming-minivan | 639 | 0 / 3 | 0.104 | 1.0886 |
| oncoming-black-sedan | 617* | 0 / 1 | 0.064 | 1.0911 |
| oncoming-grey-suv | 707 | 2 / 0 | 0.937 | 1.0911 |
| turning-black-coupe | 553* | 0 / 3 | 0.012 | 1.0911 |
| turning-white-sedan | 739 | 1 / 2 | 0.438 | 1.0871 |
| turning-silver-crossover | 799 | 0 / 3 | 0.215 | 1.089 |
| junction-white-sedan | 803* | 0 / 6 | 0.092 | 1.0911 |

All 57 production and 56 held object episodes stayed below 1.1 m per 36 Hz step (max 1.0911 m). [Full identities and steps](report3/online-evaluation.json) · [latencies, causes and controls](report3/task3-audit.json) · [release-prefix replay](report3/native-prefix.json).

## Native removal masks

On Studio: `$RS/maps/movers-v1/remove/<cam>/<n:06d>.png`. One-bit PNG, white=remove, black=keep, 2896×1876 front / 1448×938 surrounds. Dilate by 8 native pixels front / 4 surround; add dark road immediately adjacent and below a silhouette only where an observed road mask supports it. The shadow heuristic extended 83/136 reviewed silhouettes before dilation; it does not measure true shadow boundaries. An all-black PNG is a supported no-removal decision. **A missing PNG means unsupported, not empty.**

| Window | Native frames / files | Mask anchors | Propagated | Nonempty union |
|---|---:|---:|---:|---:|
| 105-140 s | 7469 / 7469 | 5871 | 1598 | 804 |
| 286-330 s | 9403 / 9403 | 5815 | 3588 | 4473 |
| 190-215 s | 5396 / 5396 | 592 | 4804 | 668 |

Verified snapshot: 27,506 supported of 77,593 native exposures; 50,087 unsupported. For reviewed moving vehicles 1087/1246 actual observed instance masks had >90% pixel coverage. Ignoring the target shutter’s mask, neighbouring production decisions/flow covered >90% on 119/136 frozen movers. This tests spatial propagation, not classification independently; unlabeled footage has no scene-wide precision estimate.

Production labelled 6114 moving, 909 parked, 16632 unknown usable detections; 1166 unknowns removed. Estimate a stationary versus translating world footprint over ≥1 s; corroborate independent cameras and matched ego/lane motion. Default unknown ahead in lane to remove; unknown at the roadside to keep. Scores are evidence strengths, not probabilities. decisions.json distinguishes at-shutter footprint, later joint-camera evidence and context-only decisions, with available_at_s timestamps; do not feed delayed labels back into live discovery.

Blue SUV control: parked on all 149 delayed decisions; 0/149 mostly removed, 146/149 under 10% overlap. Live: 2 IDs still consumed its observations. The three-camera fit estimated 0.33 m/s over 8.42 s (parked, strength 0.85, available at 324.433 s); the repeater alone instead suggested 4.32 m/s. Neither sign control joined a rendered live ID; their 17 and 24 native masks had zero removal overlap. [Confidence evidence](report3/confidence.json). [Controls](report3/stationary-controls.json) · [spatial propagation vs actual masks](report3/mask-propagation.json) · [shadow evidence](report3/shadow-support.json).

## Photographic overlay

[Crossing SUV](report3/crossing-suv-raised.jpg) · [passing bus](report3/passing-bus-raised.jpg) · [black oncoming SUV](report3/black-oncoming-suv-raised.jpg). Each left pane is a real native front crop. Middle is the actual published 85–140 s photographic chase movie. Right is the same decoded frame plus released PLY records at the attached scene’s world pose. The chase camera is 4.3–4.6 m higher in world z. Rendered offline, without an IBR foreground depth buffer; this does not verify viewer playback or prove the existing movie used the new mask bank.

Native measured paint, glass highlights and observed lamps share pipeline/photo’s exposure/white-balance reference. Unseen faces are inferred in the separate predicted PLY; visibility generations stay in object coordinates. The compositor reads the elliptical contact-shadow metadata. The bus side still streaks, vehicle contours can look doubled and the photographic background sometimes puts vegetation at the reconstructed tyre position. Contact and foreground occlusion are not photographically convincing in every proof.

Cached scenes: stitched-opt-v3-movers-photo-v1 has the four 85–140 s episodes and 419 parent tiles; live-v5-traffic-photo-hindsight has 13 later episodes on 1,411 live-v5 tiles. Later optimized static parents were absent at handoff. [Scene/PLY/cache verification](report3/scene-audit.json) · [exact image times and poses](report3/raised-proofs.json). The mask writer pins paired input shutters and namespaces per packet generation. The [snapshot receipt](report3/removal-contract.json) identifies the checked input and decision ledgers. Bounded Studio watchers refresh completed input mask pairs and attach later optimized parents if published during their watch.

Production: MOVERS_RUN=photo-full-native, modules packets --all → handoff → footprint → motion → semantics → filter → landmarks → removal → online_scene publish → skin. Held benchmark: MOVERS_RUN=photo-held-native, packets --holdout, then the same discovery/filter/scene stages and online_scene score. Reviewed photographic assets use MOVERS_RUN=photo-demo-native, photo_demo → skin → photo_proof. Use the existing Studio environment from the repository root and a single BLAS thread.
