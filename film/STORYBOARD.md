# Reality Splat: the journey (storyboard)

3:59.5 at 3840x2160, 30 fps. Thirteen chapters that each stand alone (each is also rendered on its own into
`film/out/chapters/`, opening and closing cleanly). No voice, no music: Ryan's voiceover sits on top, so on-screen
text is limited to small labels and real numbers (every number is traced in `film/notes/facts.md`). One accent
colour, the viewer's blue path ribbon. Lettering: Berkeley Mono, small, letterspaced. A quiet commit thread
(53 ticks, 15:45 → 04:15) runs along the bottom edge; its playhead glides to the commits each shot covers, passed
commits stay lit and each crossing gets a soft tick.

Sources: **CAP** camera department capture (pending, see README), **VIZ** my real-data renders (`film/viz/`),
**REP** report media in the repo, **SCR** `scratch/` recordings (viewer screen recordings with the HUD cropped
away), **RAW** dashcam frames and project data on the Studio. Sound: beds are the car cabin, the machine room and a
soft drone pad, each shot sets their level; effects are listed. **P** marks a placeholder that a capture replaces.

| # | Cut | On screen | Source | Sound |
|---|---|---|---|---|
| **0 Cold open** | | | | |
| Open | 0:00.0 | Finished generated view, junction with traffic (314-319 s), "ONLY" under the car, slow push | P SCR night3/g296 | cabin up, car pass-by |
| Title | 0:05.5 | Black, "REALITY SPLAT", "one drive · six cameras · one night" | type | sub, tick, pad enters |
| **1 Input** | | | | |
| Unfold | 0:09.1 | The car (viewer model as points) on a polar grid; the six real frames at 111 s unfold from their lenses along their calibrated rays (fisheye back becomes a cap); frustum lines; name and resolution of each | VIZ unfold.py | six unfold hisses panned to their cameras, shutter |
| Cameras | 0:17.5 | All six cameras in sync from 108 s, 3×2 grid | RAW cams108 | three shutters, cabin |
| Drive | 0:24.5 | The refined trajectory draws itself, clock 0:00 → 6:01 (the 80 s parked hurried), speed trace; "3.3 KM" | VIZ drive.json | sweep, tick |
| **2 Geometry** | | | | |
| Calib | 0:34.5 | Calibration evidence (gold measured, magenta projected) beside six bars 0.81-0.97 px against a 1 px line | REP calib/debug + data | shutter, blips |
| Depth | 0:41.5 | The 111 s front photo becomes its depth map, lifts along its own rays into points, the other five cameras join around the car, then 191 keyframes add a corridor | VIZ depthfly.py | sweep, three shimmers |
| Masks | 0:54.5 | Front camera 300-306 s; scan reveals sky, moving cars (with speeds from the tracker) and parked cars | RAW masks300 | sweep, blips |
| **3 First ground** | | | | |
| Mosaic | 1:00.5 | Bird's-eye road at 2 cm (the 318 s turn), push into "ONLY" | REP road/mosaics offline/turn | sweep |
| FirstDemo | 1:08.5 | The first demo as sent (16:51): ground-only chase beside the six cameras | SCR demo/ground-live | cabin |
| **4 First world** | | | | |
| FirstWorld | 1:17.0 | Live world v4 ("newest view wins"), chase and orbit, in panels | REP livemap v4 (P CAP ladder) | shimmer |
| **5 Optimiser** | | | | |
| Ladder | 1:27.0 | Registered wipes at 112 s: live v4 → optimiser v2 (19.72 dB) → v3 (19.96 dB) | P stills at 112 s (CAP ladder) | sweep per wipe |
| SecondDemo | 1:37.5 | The second demo as sent (18:17): optimiser v3 beside the six cameras | SCR demo/world-stitched | cabin |
| **6 Experiments** | | | | |
| CutOuts | 1:45.5 | Photographic cut-outs, labelled with their time | SCR viewer/run-a-ibr-v1 (P CAP) | dissonant drone |
| NVS | 1:52.5 | Video diffusion (StreetNVS) at its native 512×320, labelled with its time | REP genvideo | blips |
| Morph | 1:56.5 | Re-projected photograph, labelled with its time | REP ibr/report-m (P CAP) | dissonant drone |
| Teacher | 2:03.5 | The photograph as a teacher: v3 \| v4 side by side | REP stitch/report-v4 (P CAP) | sweep |
| **7 Road atlas** | | | | |
| Atlas | 2:09.5 | The published ground atlas appears along the whole drive (registered to the drive map), then a continuous zoom from 0.2 m/px to 1 cm/px at 128 s | VIZ ground atlas (Studio) | shimmer, sweep |
| Lane | 2:21.5 | The 128 s lane dash at 1 cm: splats 4.50 cm · atlas 1.23 cm · photograph 1.01 cm | REP groundtex detail-lane | three focus clicks |
| Wide | 2:29.5 | Wipe from ground v5 to +40 m each side (21:14); 21:19 power cut: picture drops, flickers, returns | REP groundtex v6 | sweep, power down/up |
| **8 Vehicles** | | | | |
| Tracks | 2:36.5 | Top-down junction 304-330 s: the car's blue path, every tracked vehicle as a box with its track ribbon, predicted states dashed, parked ones outlined | VIZ objects.json + atlas | two pass-bys, blips |
| Movers | 2:45.5 | Scanned vehicle models (20:11) wipe to clean models (00:26) under the overpass | REP movers/report4 + SCR night2 | pass-bys, sweep |
| **9 Image model** | | | | |
| ImageModel | 2:54.0 | Splat render in; nearest photograph inset; registered wipe to Difix-ref out; provenance map (observed / repaired / filled); stabilised result | RAW work/genfill/clip/c093 (110-129 s) | sweep, shutter, shimmer, blips |
| **10 Finished view** | | | | |
| Final1 | 3:11.0 | Finished generated view 105-115 s | P SCR night3/g093 | bed lifts, sub |
| Final2 | 3:21.0 | The six cameras beside the reconstruction, same moment, 306-318 s | RAW cams306 + P SCR g296 | shutter |
| Final3 | 3:33.5 | Stopped at the red light (197 s): the picture holds still | P SCR night3/g190 | cabin |
| **11 One command** | | | | |
| Pipeline | 3:40.5 | The whole drive's atlas again; the runner's 14 stages light in turn; "30 s of footage → 35 min 52 s", "one command · no manual step" | VIZ + pipeline/run/README | ticks |
| **12 Close** | | | | |
| Close | 3:51.5 | Last finished shot into the distance, "15:45 → 04:15 · 53 commits", fade | P SCR night3/g093 | sub, pad resolves |
