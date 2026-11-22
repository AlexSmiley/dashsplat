# Capture requests from the film (09:15)

The cut picks captures up automatically by id (`film/render/src/shots.rs`, `captured()` and `evolution()`), so
please keep these ids exactly. Everything is HUD-off, 3840x2160. Your keep.txt order is right; in priority order:

1. `hero-gen-100-122-60fps` — Final1 uses 105-116.5 s, Close uses 113-122 s.
2. `hero-gen-redlight-190-206` — Final3 uses 196.5-203.5 s (stopped at the light).
3. `ladder-r2-live-v4-110-today`, `ladder-r3-stitched-opt-v2-110-today`, `ladder-r3-stitched-opt-v3-110-today` — the optimiser wipe (all three needed; registered, same start frame at 110.00 s).
4. `ladder-r5-stitched-opt-v4-110-today` — Teacher wipe v3 -> v4 (with the v3 above).
5. `ladder-r4-stitched-opt-v3-110-auth-photo` — CutOuts (photographic layers as they looked).
6. `ladder-r5-stitched-opt-v5-110-today-splatground` + `ladder-r6-stitched-opt-v5-110-today-atlas` — the ground wipe (both needed).
7. `ladder-r7-stitched-opt-v5-movers-v1-284-today-scanned` + `...-284-today-clean` — Movers wipe (the film matches the
   prefixes `...-284-today-scan` and `...-284-today-clean`).
8. `ladder-r7-stitched-opt-v5-movers-v1-312-today-clean` + `ladder-r8-stitched-opt-v5-movers-v1-312-today-generative` — a
   split "splats | Generative" after the image-model chapter (both needed, registered).
9. Any further `ladder-*-110-*` rung (today's build preferred): the Evolution montage uses every 110 s rung, one per
   scene, in rung order.

Not needed by the cut: HUD-on variants, the 312 s ladder except item 8, `dir-tracks-junction-top-304-330` (the film
draws its own top-down tracks from objects.json). Already in and used: the palms junction hero (Open, Final2),
`dir-points-to-splats-110-122` (Geometry), `ladder-r1-road-live-v1-110-auth` (First ground), `ladder-r2-live-v4-110-auth`
(First world), r1/r2 rungs (Evolution).
