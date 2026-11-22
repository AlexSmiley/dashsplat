# dashsplat viewer

**On your machine.** This README records the viewer as built on the author's machines: "basebook" is his working Mac (an
Apple M5 Max, where the performance figures below were measured), "the Studio" his Mac Studio that hosted the
published scenes on his LAN, and `ferry` his copy tool. Without them:

- No scene ships with this repository. `bun run dev` serves the repo-root `scenes/` folder, while the pipeline
  publishes to `scene/` under its installation root: symlink one to the other, or `bun run build` and serve with
  `bun ../deploy/serve.ts dist ../scene 8090 127.0.0.1` from `viewer/`.
- `bun run cache <scene dir>` builds a scene's compact tile cache locally. `bun run cache:studio` and
  `tools/offload.ts` are the author's Studio helpers.
- `tools/capture.ts` proxies scenes from `RS_STUDIO` (default `http://127.0.0.1:8090`) and expects Google Chrome at
  its macOS path unless `RS_CHROME` is set.
- The HUD imitates Tesla's screen on purpose; see [../docs/NAMING.md](../docs/NAMING.md).

Plays back a recorded Model 3 drive as a Gaussian-splat world, styled after Tesla's Reality View screen.
TypeScript, Vite, three.js r187 and Spark 2.4 (WebGL2). Reads scenes in the viewer contract of `../PLAN.md`.

```sh
cd viewer && bun install && bun run dev      # http://127.0.0.1:5188 (127.0.0.1 only)
bun run build                                # type-check + production build
```

The dev server serves the repo-root `scenes/` folder at `/scenes` and lists scenes at `/scenes/index.json`.
The default scene is the newest `stitched-opt-vN`, its movers edition when there is one (`stitched-opt-v5-movers-v1`),
with the textured ground on and photo layers off, starting at 109 s; then the newest `*live*` scene that has tiles,
then `*stitch*`, then any with tiles.

URL parameters: `scene=road-live-v2`, `t=200` (start time, default 90 s: leaving the car park), `paused=1`, `mode=memory|stitched|generative`,
`cam=map|cams|free`, `orbit=az,el,zoom` (fixed orbit in degrees, for checks), `night=1`, `colour=camera`,
`paint=white`, `stats=1`, `dpr=2` (fixed resolution instead of adaptive), `overlay=auto|car|path|none` (Generative),
`bare=1` (no car, shadow or path: for recording reference renders of the chase view), `photo=ibr-v1|<url of an ibr.json>|off`,
`layers=sky,backdrop,cutouts,movers` (or `all`), `cutouts=5,6,7|all` (static cut-out kinds to draw), `sanity=0` (no
cut-out checks), `audit=photo` (run every cut-out through the checks; result in `document.body.dataset.photoAudit`),
`objFade=38,72`, `far=80,240,0.6,0.06` (optimised scenes' haze), `wbk=`, `g2=`, `clouds=0` (grade, see Look), `predicted=1`, `generated=<url>` (Generative mode), `ground=0` (the splat ground instead of the
textured one), `moverShadows=0`.

Keys: space play/pause, ←/→ 5 s (shift 30 s), ↑/↓ speed, 1 follow, 2 map, 3 car cameras (press again to cycle),
4 free fly (WASD/QE, drag to look, Esc), M Memory, L Live, K Stitched, N night, F stats.

## What is on screen

- **Scene**: splat tiles streamed around the car, the Model 3 Highland at the ego pose (wheels steer and spin
  from `steer` and arc length, tail lamps from `brake`, indicators from `blink_l/r`), soft contact shadow,
  Tesla-blue path ribbon from under the nose fading out ahead (about 14-75 m depending on speed), upright
  chevrons before turns, sky panorama from `map.json` or a procedural dawn-overcast sky, fog matched to the
  horizon. Night (moon button) grades splats dark blue, lights the road with two soft low beams and a red brake
  glow, and turns on head and tail lamps.
- **Cameras**: follow camera (see "Chase pose" below: 16.2 m from the car centre, 16.9° up, pitched down 11.6°,
  50° vertical field of view, car centred, horizon 28 % from the top). Drag orbits, wheel or
  pinch zooms, it eases back 1.4 s after release. Map view (heading-up, drag pans, zoom 35-1600 m, whole route
  drawn). Car cameras from `rig.json` (fisheye fields of view fitted to the window, rear capped at 128°). Free fly.
- **HUD**, element by element after `reference/ref_15.5.jpg`: gear row and battery, thin speed numeral,
  Self-Driving state from `ap` (Tesla dashcam enum: 1 self-driving, 2 autosteer, 3 cruise) with mode line,
  streak (distance since engagement), speed-limit sign, green light icons, next turn with distance and street,
  lock, clock from the recording (starts 2026-11-21 07:15:40), weather, airbag badge. Bottom: Map and moon
  buttons, the drive timeline in the media card's place (route thumbnail as "album art", scene and mode as title
  and subtitle, the scrubber as the track-progress line with self-driving stretches in blue, transport icons
  plus playback rate, Memory and camera), and the trip card (arrival time, minutes and miles left, destination,
  battery, End Trip jumps to the end). The "…" opens a menu with Live / Stitched / Memory (and Generative where a scene has it), view, colour, paint
  (Quicksilver, Pearl White, Stealth Grey, Diamond Black, Deep Blue, Ultra Red) and scene. Phones and tablets get
  the same language scaled, with the cards spanning the width. Font: Inter Variable (OFL), the closest free face
  to Tesla's Gotham-like UI type at these weights.
- **Not in the telemetry, shown the way the car would**: battery (78 % minus 0.21 % per km), 58 °F and cloud,
  passenger airbag badge, speed limit (85th-percentile speed within 250 m, rounded down to 5 mph). Gear enum:
  0 P, 1 D, 2 R, 3 N. Street names come from `public/streets.json` (`bun tools/streets.ts`, OpenStreetMap
  Nominatim, data © OpenStreetMap contributors, ODbL), matched to a scene's drive by position; other drives
  fall back to "Turn left / right".
- **Legibility over a bright sky**: as on the car's daytime screen, a very soft dark pool behind the top-left
  block (speed, Self-Driving, streak, limit; at most 32 % right behind the text, fading to nothing), a smaller one
  behind the clock and badge, a faint band along the top, and a tighter shadow on the thin speed numerals.

## Look (realism pass)

- **Grade** (splats and the sky panorama): "Natural" (default) measures the camera cast on the road surface
  (median colour of low-saturation, mid-grey splats within 2.6 m of the driven path) and corrects it hue-safely:
  near-neutral colours (asphalt, concrete, cloud) get the full white balance, saturated colours and deep shadow
  much less (partial chromatic adaptation), the gains never change brightness, and the correction may not push
  green below both red and blue. Muted violet and lavender (sky light bled into foliage; blue above green with
  red level with or above green) are drawn back toward grey; sky blue and strongly saturated paint are spared.
  Contrast (black point, S-curve) works on luminance only, keeping channel ratios, because a per-channel curve
  saturates dark colours and turned shaded foliage purple; the saturation lift skips shadows and colours that
  are already saturated. "Camera" shows the footage colours. Gains come from `rsb/index.json` (written by the
  cache tool), else from an earlier visit (localStorage), else they are estimated live. Night grades on top.
- **One grade for both modes** (second night pass): the splats and the generated clips go through the same
  Natural grade. Its neutral point comes from the footage: dry asphalt is neutral grey, and in the clip and the
  splats (both in the camera's corrected colours) it reads (0.38, 0.47, 0.46), needing the same gains as the
  road-measured white balance (1.16, 0.94, 0.94, brightness kept); the raw dashcam frames are tealer still
  (asphalt 0.32, 0.48, 0.50). So: the full white balance (`?wbk=` for less), exposure 1.02, S-curve 0.28,
  saturation 1.15 in the midtones, no black point (`?g2=`). Asphalt reads neutral, the overcast sky soft
  white-grey, grass green. The car's environment is built from the graded sky and the graded frame, so it
  stays consistent. The haze of optimised scenes is lighter (80 → 240 m, 60 %), and the sky has a procedural
  cloud deck (`?clouds=0` to remove): fbm noise on a plane 1.5 km up, darker grey undersides, thinning towards
  the horizon, sliding overhead as the car drives, also in the car's reflections.
- **Sky panorama** (`sky.jpg`): measured when it loads (mean luminance of the upper sky and how much each row
  varies across its width) and shown with weight smoothstep(0.45, 0.68, luminance) × (1 − smoothstep(0.035,
  0.09, blotchiness)), the rest being the procedural overcast sky; the haze colour follows the same mix. The
  car's reflections take at most half that weight and are never darker than the procedural sky. live-v4's
  panorama (luminance 0.56, blotchiness 0.072, vertical colour streaks) is shown at 12 %.
- **Haze and the map's limits**: splats fade into the sky's horizon colour with horizontal distance from the
  car (50 → 140 m in the follow view; from the camera in free view, from the map centre in the map view) and
  are gone beyond it, so the coarse far field never shows as streaks. The sky's horizon band is exactly that
  colour. A ground apron (graded road colour part-way into the haze) lies under everything, so holes read as
  hazy ground, not sky. Distant splats have their elongation capped. Sky and apron are dithered.
- **Car**: see "The ego car". Night: emissive lamps with soft additive halos, inverse-square headlight pools and
  a red brake glow on the road, warm light on lit splats, a faint hemisphere fill. The car writes depth, so
  splats in front of it cover it.
- **Ribbon**: lies on the road (drive height and cross-slope), luminous blue brighter near the car, soft edges.
  Drawn before the splats with a depth pulled 0.7 m towards the camera: ground splats cannot cut into it,
  anything really in front (a car, a pole) still covers it.
- **Camera**: see "Chase pose".

## Chase pose

The default follow camera is a pure function of drive time T and the view's aspect ratio
(`src/camera/chasePose.ts`), so it is identical at any playback rate, after any seek, and in an offline render:

- Pivot: the car centre, 1.45 m ahead of the ego origin along a smoothed heading and 0.8 m above the road.
  Heading = the drive's (unwrapped) yaw, Gaussian-smoothed (σ 0.25 s) around T + 0.05 s: no pose jitter, no lag in
  turns. Height = drive z, Gaussian σ 0.2 s around T. Position x, y: the car's, unsmoothed (the car stays put in
  frame).
- Camera: `dist` 16.2 m from the pivot (× 1 + 0.004 per m/s of speed, speed being a trailing 1.2 s exponential
  mean, capped at 25 m/s), `elev` 16.9° above it, looking at it, then tilted up 5.3° (`tilt`): pitch −11.6°.
  About 5.5 m above the road and 15.5 m behind the car centre at rest. No sideways pan.
- Field of view: 50° vertical when the view is 16:10 or wider (at least 64° horizontal for narrower landscape
  views); portrait views go up to 74° and 0.78 × the distance.
- Measured on Tesla's frames (`ref_15.5`, `ref_24`, `ref_31`): horizon 25-32 % from the top (we put it at 28 %),
  the car as seen 61-63 % down and centred (ours 62-63 %), its body 6.3-8 % of the view's width (ours 7.7 % at
  rest, 7.1 % at 47 mph). The previous pose (19.5 m, 15° up, pitch −7.8°, nudged 1.1° left) put the horizon at
  35 %, so it showed about a quarter more sky and read as lower and farther than Tesla's.

## Generative mode

For a pre-rendered, generated video of the default chase view. If a scene has `generated.json`, a fourth map
mode, "Generative", appears in the menu (and `mode=generative`). In it, in the chase view by day and with T
inside a clip, the clip is shown full-frame in place of the splat render, with the car model, its contact shadow
(multiplied onto the video) and the path ribbon drawn over it ("Over the generated video": Auto, Car + path,
Path, Nothing; Auto leaves the car out when the clip says it already has one). The car's reflections then come
from the video frame (see "The ego car").

- **Automatic**: until the user picks a mode (`?mode=` or the menu), playback and scrubbing move into
  Generative wherever a clip covers the current time and back to the splat view where none does. The page opens
  on the first clip when it starts within 6 s of the default start (`stitched-opt-v5-movers-v1`: 110.05 s).
- **Hand-over**: a 0.25 s cross-fade in, starting on the clip's first frame; where a clip ends and the splat
  view takes over, a 1 s fade that ends on its last frame (the video still playing). Adjacent clips (one starts
  within a frame of where the other ends) join seamlessly: the next is cued in a second video element and set
  rolling 0.1 s before the join, then takes over in the same frame (no fade; measured within a frame of sync
  across the join). Orbiting fades the clip with the angle (fully gone by about 2.3 degrees of orbit or a 9 %
  zoom), so it never slides against the splats, and it comes back the same way.
- **Grade**: the clip goes through the same Natural grade as the splats (one grade for both modes, see "Look").
- **Vehicles**: a clip marked `"vehicles": "overlay"` has none of its own; the viewer draws every vehicle over
  it as a clean model (see "Clean vehicles"). Clips without the mark are shown as they are, without them.
- **Time sync**, checked with a test clip carrying its frame number as a bit code, read back from the canvas
  on every frame: paused and while scrubbing exactly the right frame; playing at 0.5x-2x the right frame or
  its neighbour (1x: 73 % exact), 4x within one frame either side. The video is steered by the frames actually
  presented (`requestVideoFrameCallback`), seeks on jumps and nudges its rate by up to 15 % for drift.
- **Predicted** (menu, `?predicted=1`): shows the clip's provenance mask as a quiet violet wash, stronger on
  filled pixels than on repaired ones. A clip without a mask says so in the status line. The same switch tints
  the textured ground's predicted texels (see "Textured ground").
- `?generated=<url of a generated.json>` tries a clip on any scene before it is attached (its `file` relative
  to that json).
- Outside the clips the status line lists them.

```json
{ "clips": [ { "file": "generated/clip-0.mp4", "t0": 104.0, "t1": 126.0, "fps": 30,
               "pose": "chase", "vfov": 50, "ego": "free" } ] }
```

- `file`: H.264 mp4 (yuv420p, no audio), relative to the scene folder. Short GOPs (keyframe every 0.5 s) keep
  scrubbing quick. The viewer downloads a clip whole when T comes within 20 s of it, so it needs no byte
  ranges from the server; keep clips to a few tens of MB (split long stretches into several clips).
- `t0`, `t1`: drive time of the first and last frame; frame n shows T = t0 + n / fps.
- `pose`: `"chase"`, rendered from exactly the default chase pose; anything else is ignored for now.
  `bun tools/chase-track.ts <scene dir> <t0> <t1> [fps] [width] [height]` prints that camera for every frame
  (position in the scene frame, z up; three.js camera quaternion, looking down local −z with +y up; vertical
  field of view) for rendering or for conditioning a generator.
- `vfov`: vertical field of view in degrees (default 50, the chase camera's for 16:10 and wider). The clip is
  drawn at that field of view, centred; 16:10 or wider covers a 16:10 view, a clip narrower than the view
  blends into the splats at its sides.
- `ego`: `"free"` (default) when the ego car's region is left for the viewer's car model, `"generated"` when the
  video already shows the car (Auto then draws only the path).
- `vehicles` (optional): `"overlay"` when the clip shows no vehicles and the viewer should draw them.
- `provenance` (optional): `"below"` when every frame carries its provenance mask under the picture: the
  video is twice the picture's height (picture rows on top, mask rows below, same width and pixel grid), and
  the mask is grey with these codes: **255 observed** (a camera pixel kept), **170 repaired** (corrected by the
  model where the splat render had content), **85 filled** (made up where nothing was observed), **0 none**.
  One file keeps the two in exact frame sync; levels 85 apart survive H.264 (use `-pix_fmt yuv420p`, CRF 18 or
  better). Example: `{"file": "generated/genfill-chase-110-125.mp4", "t0": 110, "t1": 125, "fps": 30,
  "pose": "chase", "vfov": 50, "ego": "free", "provenance": "below"}`.
- Colours are taken as display-referred video and drawn as is (no grade, no tone mapping); the splat view's
  Natural grade is matched to the clip instead (see "Look").
- After adding `generated.json` to a scene that has an `rsb` cache, rerun the cache tool (`bun run
  cache:studio` for the Studio): its index records whether the file exists, so scenes without it do not probe.

Stand-in: `scenes/live-v4/generated.json` (local and on the Studio) points at
`generated/standin-splat-chase-104-126.mp4`, the viewer's own splat chase view from 104 to 126 s recorded with
`bare=1` (canvas capture at 1×, 30 fps) and warm-tinted so it is easy to tell apart. Delete both to remove it.

## The ego car

- **Lighting** (`src/render/environment.ts`): the car is lit only by a prefiltered (PMREM) environment of the
  scene it is in: the sky dome's own shader (panorama, procedural clouds, haze band) over a road-coloured
  ground, plus the surroundings. Four times a second the frame on screen (splats or the generated clip) is
  copied at 640x400 (`blitFramebuffer`, then a mip chain) and projected back onto the directions it shows:
  each direction's vanishing point through the capture camera, the nearest frame edge for directions outside
  it, the rear hemisphere mirrored forward. So the glass roof and windscreen show sky and clouds, the flanks
  the roadside, the boot the road behind, and the reflections move as the car drives. The cube is re-rendered
  and prefiltered two frames after each capture (reading it at once stalls on the frame still being drawn);
  about 0.5 ms of CPU, no measurable frame time. Radiance is linear (display code values decoded), so a
  mirror shows the scene as bright as it is on screen; three quarters of its colour, so the silver reads
  silver under the teal morning sky.
- **Materials** (`src/render/car.ts`): clear-coated metallic paint (Quicksilver: metalness 0.9, roughness
  0.22, clear coat 0.04), tinted glass with a coated reflection (index 2.0 for the reflection), satin black trim,
  gloss black pillars, machined alloy, darkened rubber with the sidewall texture. The body darkens towards the
  road (sills, bumpers, arches, underbody: the model has no baked occlusion), by height and by facing down.
- **Contact shadow** (`src/render/contactShadow.ts`): the footage runs 07:15:40-07:21:42 EST on 21 Nov 2026 at
  30.25 N, 81.52 W: the sun is 2.9 to 4.1 degrees up at azimuth 115 (east-south-east; solar position from date,
  time and place), so a sun shadow would be 20-30 m long and fall west-north-west across the road. The
  photographs show full overcast and no cast shadows at all (hardness 0), so there is no sun shadow; the shadow is the sky's occlusion, computed once by casting 112 rays per texel from the road
  against a box model of the car (body, shoulders, glasshouse, wheels) under a CIE overcast sky: up to 90 %
  under the body, about half at the sills, fading out within a metre or two, darkest at the tyres. One soft
  texture under the car, multiplied onto whatever is there (splats, the textured road, the generated video).
- **Wheels and lamps**: wheels roll with the distance driven (capped at 24 degrees a frame, so a five-spoke
  wheel turns steadily instead of strobing) and steer with the telemetry; brake lights on the `brake` flag, a
  deceleration over 1.3 m/s^2 or at a standstill (the telemetry's brake flag ends after the car park), with
  their red light on the road behind; dawn running lights on.

## Clean vehicles

`src/render/vehicles.ts`, posed by `src/scene/objects.ts`. Menu "Vehicles: Clean / Scanned" (`?vehicles=scanned`),
default Clean; the scanned splat models stay available.

- **Models**: one per kind, from the track's name and class (the producer names them: "red SUV at the roadside",
  "oncoming-pickup", "passing-bus"), else from its height. Saloons, coupes and Teslas use the Highland Model 3
  (cloned, uniformly scaled by length so the wheels stay round). SUVs, off-roaders, minivans, pickups, buses and
  lorries are procedural bodies in the same language: a lofted rounded section (superellipse, tumblehome above
  the beltline, rounded ends) with glasshouse, black trim, wheel arches and lamps as per-vertex material regions,
  on the Model 3's wheels scaled to the kind. No third-party models.
- **Size**: the track's measured box, clamped per axis to what the kind can be (scanned boxes are often cut short:
  a 3.3 m "SUV" becomes a 4.4 m one).
- **Paint**: the colour in the name (white, black, silver, grey, red, burgundy, blue, ...), else the median colour
  of the scan's opaque splats on the sides at door height (graded), else mid-grey. Coloured paints keep most of
  their diffuse colour (a strongly metallic red reads black side-on).
- **Look**: the ego car's: scene environment light and reflections, clear coat, dark glass, ground occlusion, the
  overcast contact shadow stretched to the footprint, wheels rolling with the distance (capped against strobing)
  and steering with the yaw rate, brake lamps when decelerating or stopped in traffic (not when parked), dawn
  running lights.
- **Pose**: the hindsight track smoothed once on load: positions Gaussian-smoothed (0.15 s) and, at a
  standstill, taken from a 0.7 s window (no creep); heading from the direction of travel once moving faster than
  about 2 m/s (the track's own yaw when slow or parked, never flipped backwards); speed, acceleration, yaw rate
  and distance from the smoothed track. Wheels on the textured ground where it is there. Fading in and out over
  0.45 s at the track's ends and towards 55 % where the track is predicted, as a dither (depth stays right).
- **Leftovers**: a car that stood long enough is baked into the static splats; in Clean mode the static splats
  give way inside each vehicle's box (above 0.15 m, soft edge), so the model stands alone.
- **Over a generated clip** (no depth in a video): at half resolution the vehicles' depth is drawn, then every
  splat against it; what lands in front of a vehicle leaves alpha, read back softly (five taps) as how much of the
  vehicle (and its shadow) is hidden. Approximate (splats test at their centres) and soft-edged.

## Textured ground

`src/render/groundAtlas.ts` implements `pipeline/groundtex/FORMAT.md` ("reality-ground" v1): the road surface as
a 1-2 cm per texel texture on a height-grid mesh, 10.24 m tiles in `ground/ground.json`.

- **Streaming**: tiles inside a 164 m square around the streaming focus (35 m ahead of the car), nearest first, six
  fetches at a time, one texture upload every other frame (with the outer ring about 170 tiles are resident and
  a recentre brings in a few dozen). The coverage map is rewritten incrementally (only tiles that came or went),
  at most four times a second; coverage is read from quarter-size decodes only, never from a full texture. Within 45 m of the camera a tile has its full texture, beyond
  that a quarter-size decode (the GPU would pick those mips anyway); the compressed file is kept for the upgrade.
- **Filtering**: decoded off-thread with alpha premultiplied, so the mip chain filters premultiplied colour (no
  halos at alpha edges); trilinear mipmaps and the device's maximum anisotropy (16 on this Mac).
- **Colour**: the splats' path, once: Natural grade, movers' contact shadows, night lights, haze.
- **Depth**: drawn after the apron and before the ribbon and the splats, in two passes: the opaque interior
  (alpha >= 0.98) with depth, its soft edges blended without depth. Splats standing on it (kerbs, grass relief,
  objects, movers) depth-test against it; the car, its contact shadow and the ribbon draw over it.
- **Ground-layer splats** give way where the atlas is opaque: a 512 x 512 coverage map (0.32 m cells: the share of
  opaque texels and the mesh height, from the resident tiles) is read by the splat shader, and a ground-layer
  splat within 0.2 m of the mesh fades out with the bilinear coverage. They stay through transparent texels,
  while tiles stream and outside the atlas, and come back everywhere in Memory mode and with `?ground=0` (menu:
  Ground, Textured / Splats).
- **Gaps**: where the atlas has no texels (e.g. left of the road at 287 s) the splat ground, and its gaps, remain.
- **Predicted** (menu, `?predicted=1`): with the manifest's `provenance` masks, the predicted texels get the same
  quiet violet wash as the generated clip's. Per FORMAT.md the codes are read unfiltered (nearest, at a quarter of
  the texture size), turned into the predicted weight times coverage, then filtered with mips; the shader divides
  by the filtered coverage, so edges are neither haloed nor averaged codes. Masks load only while the switch is on.
- **FORMAT.md check** (night pass): the texture's GPU levels 0, 2 and 4 were read back: every zero-alpha texel has
  RGB 0 and every partial one RGB <= alpha, so the bitmap is premultiplied before the mip chain is built (decoded
  with `premultiplyAlpha: "premultiply"`, uploaded without another unpack premultiply, unpremultiplied before
  grading). Tile lookups use half-open ownership (`floor(x / 10.24)`), an absent neighbour contributes zero
  coverage, and edges clamp (no wrap). Not done: neighbour-aware mips where the filter footprint exceeds the 8-texel
  gutter (beyond mip level 3), which can leave a faint seam on far tiles.
- **Phones** (`?groundFilter=1` / `0` forces it): ground-layer splats under fully opaque atlas cells are left out
  when their tile is decoded (compacted in the tile worker): they are hidden there anyway (identical frame, checked
  by difference), and they were most of the budget. The cells come from the atlas tiles' coverage, which is read
  for the filter even where the atlas is not drawn (quarter-size decodes, kept for 1,600 tiles); a ground tile
  waits up to 5 s for it. With the filter, ground tiles load within 120 m of the focus and unload beyond 180 m
  (farther ground is hazed out), so the budget goes to the scenery: at 121 s on a phone, 191 tiles and the far
  trees instead of 61 tiles and an empty horizon (1.8M splats either way; 1.55M ground splats left out).
  Memory mode and the splat ground switch it off and reload the ground whole.
- **The car stands on it**: the mesh lies within a few centimetres of the drive's height; the car follows the
  mesh under it (smoothed), so the tyres neither sink nor float and the contact shadow is not cut by the road.

## Movers over the optimised scene

Movers are sorted with the static splats (one sort), so a vehicle behind a splat tree is hidden by it and one in
front covers it (the bus at 129-134 s behind the median trees, the car at 132.6 s in front of them). Each mover
has the ego car's contact shadow scaled to its box: inside the footprint the profile is stretched to the box,
outside it the falloff keeps metres times the height ratio (a bus's shadow reaches further than a hatchback's),
at the producer's `contact_shadow` offset when given (its peak of 0.32 read too faintly and is not used). It is
drawn by the splat, ground-atlas and apron shaders, so it lies on whatever ground is there and whatever stands in
front of the mover covers its shadow too.

## Missing frame (fixed)

One frame in a played recording showed the page background with only the HUD: the adaptive resolution step
resized the canvas after the frame had been drawn, and resizing clears a WebGL canvas. Size and resolution
changes now apply at the start of the next frame, before it renders. Checked with a frame hook that reads every
presented frame back: before the fix one blank frame at the first resolution step (3200x2000 to 2720x1700 at
118.5 s), after it none in three 105-125 s runs that each crossed a step.

## Photographic layers

`src/render/photo.ts` implements `pipeline/ibr/FORMAT.md` ("photo-layers" v1): a photographed sky, one blended
backdrop curtain per selected knot, canonical static cut-outs and per-exposure movers, chosen by which real
exposure was taken nearest the virtual camera.

- **Where they come from**: `?photo=<scene or URL of an ibr.json>` (`off` to disable), else `"photo"` in the scene's
  `map.json` (a scene name or a URL relative to the scene folder), else the scene's own `ibr.json`, else, for
  `stitched-opt-*` and `live-*` scenes, the newest `ibr-*` scene recorded on the same drive (same duration and
  origin) whose interval lies inside it. The cache tool's index records whether a scene has its own.
- **Switches**: menu ("Photo layers") and `?layers=`: Sky, Backdrop, Cut-outs, Movers, each easing in and out. A
  scene's own layers start with all four on and every cut-out kind. Borrowed layers start off (the display is one
  coherent splat scene); `?layers=sky,backdrop,cutouts,movers` is the former composition, with cut-outs of
  poles, lights and signs only (kinds 5, 6, 7). The photographed sky alone stays off too: across 93-140 s it
  carries ghost pole tops and a dark seam. `?cutouts=` picks kinds (2 building/wall, 4 fence, 5 pole, 6 light,
  7 sign, 8 vegetation, or `all`).
- **Cut-out checks** (borrowed layers only): a card is dropped, and its id logged and listed in the stats, if
  its base is within 3.6 m of the driven path (the car's lane and half the next; overhead cards above 4.2 m are
  kept), if it is too wide for its kind (a pole over 1.5 m and over a quarter of its height, a light over 2 m and
  half its height, a sign over 3.5 m, a fence over 25 m), or if its matte is mostly blur (over half of its
  pixels soft and more than 2 px from any solid pixel, which separates a smear from the anti-aliased rim of a
  thin pole) or has under 24 solid pixels. In `ibr-v1` this drops 7 of the pole, light and sign cards: 74, 320,
  1077, 1250 (too wide), 230 (a sign plate standing on the road edge at 109-111 s), 285 (a pole 3.4 m from the
  path), 1099 (blurred).
- **Selection** exactly as the format gives it (path proximity, camera direction, lens weights, knot shares,
  duplicates summed), per frame from the live camera, so an orbit picks the cameras that looked that way. It
  matches the reference's examples (100 s: left pillar, knots 17/18 at 0.366/0.634; 112 s: front, 45/46 at
  0.697/0.303). Neighbouring knots are fetched ahead; a knot whose images are not in yet borrows the nearest
  one that is, so the sky never shows a hole. JPEG and matte are decoded off-thread without colour management
  and premultiplied on the GPU before any filtering; cut-out atlases get mipmaps of premultiplied colour.
- **Drawing order**: sky dome (photographed sky blended across the knots, in the dome shader); apron; the
  backdrop (fullscreen pass: analytic exit intersection per curtain, Cfar / afar / Zfar sums as specified,
  drawn only where nearer than the road plane + 12 cm, writing its depth where afar > 0.5); the ribbon; cut-outs
  and movers (a depth prepass of alpha > 0.5, then colour far to near, plane-depth and edge-on rejection as
  specified, double-sided, atlas lookups clamped to their rectangle); the splats, depth-tested against all of
  it; the car (opaque, depth-tested) and its overlays.
- **Differences from `layer_player.py`**, chosen to look the same at the default chase pose: the ground depth
  for the backdrop rule is the road plane (drive z − 0.08 m), not the opacity-weighted depth of the ground
  splats (no extra float target); cut-outs are drawn before the splats rather than over them, so a splat tree
  in front of a card covers it (needed when the splats have real trees), and splats behind a card's soft edge
  (alpha ≤ 0.5) show through that edge. The apron keeps the scene's road colour instead of the reference's grey.
- **Borrowing across frames**: `ibr-v1` is in the live drive's frame; `stitched-*` scenes use the stitched drive,
  which differs by up to 1.3 m in 93-140 s. The layers are placed with a per-time rigid transform that maps the
  photo drive's car pose onto the scene drive's (offset and yaw, Gaussian-smoothed over 0.5 s), so they line up
  around the car; identity when both are the same drive.
- **Grade**: the Natural grade's white balance and saturation, once, on unpremultiplied colour (sky, backdrop,
  cut-outs, movers), without its black point and S-curve: those make up for the soft contrast of fused splats,
  and on photographs they turned the sky into a storm sky. Exposure 0.98 (the splat road comes out about 10 %
  under the photographs' road after its curve; this keeps the sky's brightness and its ratio to the road both
  within a few percent). The photographed sky takes the full white balance (its tint is the camera's cast) and
  about half its chroma: a pale overcast, not teal.
- **Sky brightness** (the sky band 3-12° above the horizon and the road beside the car, against the front
  camera's photographs at the same times; `screenshots/photo/sky-fix/`): before, the sky was 24-26 % darker
  than the photographs and darker than the road (ratio 0.88 against 1.09-1.11). Two causes: the HUD's top scrim
  (34 % black at the top, fading over 46 % of the height) and the grade's black point and S-curve. Now, Natural:
  sky −4 % / −2 %, sky-to-road ratio +5 % / +2 % (112 / 128 s); Camera colours: sky −1 % / +2 %, ratio −3 % / −5 %.
  The brick office's facade matches the photograph (59 against 59, was 38). The scrim is now 10 % over the top 16 % (since the second night pass: soft pools behind the HUD's text blocks, see "What is on screen"). The car's image-based lighting uses the photographed sky (rebuilt when the leading knot has moved a
  dozen on, at most every 4 s; a rebuild costs about 14 ms), and the haze colour is its horizon.
- **Haze (optimised scenes)**: with photo layers off, `stitched-opt-*` scenes haze from 80 to 240 m, at most 60 % of
  the way into the sky, stream to 260 m and use a slim sky horizon band (0.06 rad), so distant trees and buildings
  stay readable instead of ending in a grey wall.
- **Haze**: over a photographed sky or backdrop the splats fade out between 60 and 118 m (showing the
  photographs) instead of hazing into grey. Over the backdrop, splats that are not ground (object tiles, or more
  than 2.5 m above the road, as ground tiles carry poles too) hand over sooner, from 38 to 72 m: far trees and
  poles are blobs where the backdrop has the photograph.
- **Validity**: 93-140 s, easing in and out over 0.4 s at the ends, back to the normal sky and haze outside.
  Off at night and in Memory mode, backdrop off in the map view. Sky and backdrop stay when orbiting; cut-outs
  and movers fade out between 18 and 30° of azimuth and 7 and 15° of extra height, leaving the splat world.
- **Measured** (basebook, 3200×2000, playing from 105 s): `ibr-v1` all layers 60 fps (1.06 M splats drawn),
  `stitched-opt-v2` + sky and backdrop 60 fps (1.83 M), + all layers 60 fps with more upload spikes (p99 30 ms,
  14 frames over 25 ms in 820), plain `stitched-opt-v2` 60 fps (2.4 M).
- Screenshots: `screenshots/photo/` (`a-` ibr-v1 alone, `b-` stitched-opt-v2 + sky and backdrop, `c-` + all
  layers, `d-` + sky, backdrop, poles, lights and signs, `o-` stitched-opt-v2 plain, `ref-` Camera colours next
  to the reference stills).

## Movers and predicted content

`objects.json` (PLAN.md): each object's PLY (and optional `predicted_file`) is decoded by its own worker into
a Spark splat mesh in the object frame and moved every frame along the interpolated track (yaw on the
circle, eased in and out at `t_first` / `t_last`), with the same time shader, so its model fills in as it
is observed. Movers sort with the static splats (one global sort), the ego car (depth) and the ribbon (depth).
Predicted content (`predicted_file`, and tiles whose `layer` ends in `-predicted`) is drawn softer, and
violet in Memory mode with its own legend entry; a mover whose pose is predicted (`state` 1) reads as a ghost
in Memory mode. Transient tiles are prefetched 2.5 s + 2 s × playback rate ahead.

The synthetic scene has four objects: one overtaking on the right that is predicted for about 1.8 s while
alongside (inside the side cameras' body-mask zone), an oncoming car, one crossing at the second light and
a lead car pulling away. It stays local (not on the Studio).

## Modes

- **Live**: a splat exists while `t_birth <= T < t_death`; it fades in over 0.12 s after birth and out over the
  last 0.12 s before death (scale 0.85 → 1 while fading in). The contract's 0.25 s overlap therefore holds the
  finer generation fully opaque before the coarser one starts to go, so hand-overs never dip in opacity
  (two 0.2 s fades would overlap and visibly breathe).
- **Stitched**: the final state of the map: static splats that never die, no transients.
- **Memory**: Live plus age. `seen = min(T, e0); if (T >= s1) seen = max(seen, min(T, e1)); if (T >= s2) ...`,
  `age = T - seen`. Seen now keeps its colour; remembered splats desaturate and cool with age (half way at about
  10 s, fully grey-blue after a minute). Thin outlines of the six camera frusta (from `rig.json`, fisheye model)
  show what is seen right now. The legend card explains the ramp.

## Per-splat time on the GPU


Each tile's splats go into a Spark `ExtSplats` (float32 centres relative to the tile centre, so 3 km of world
does not band) and a per-tile RGBA32UI texture with one texel per splat, indexed by the splat index:

    word0 = birth | death << 16   word1 = e0 | s1 << 16   word2 = e1 | s2 << 16   word3 = e2

Each value is a uint16 step of `quantum` seconds after the tile's earliest `t_birth` (10 ms unless a tile
spans more than 655 s); 65535 = never (1e9). A Spark dyno world modifier (`src/render/splatShader.ts`) fetches
the texel, decodes the seven times and applies existence, fades, Memory colouring, fog, the loaded-area edge
fade and the night grade; dead or unborn splats lose Spark's active flag so the sorter drops them. Every input
is a uniform shared by all tiles (one compiled program), so **scrubbing costs nothing per splat on the CPU**:
changing T updates a float and Spark regenerates its splat buffer on the GPU (it does that on every camera
move anyway) and re-sorts asynchronously; measured scrubbing stays at the display rate. A jump to a far time
also streams the tiles around the new position (transient tiles are picked by their `t_min..t_max`).
Limits: 10 ms time resolution; the three seen-runs are what the contract carries (a fourth sighting is folded
into the third by the pipeline); every loaded splat is still generated each frame, alive or not, so memory
and generation cost scale with loaded splats, not visible ones.

## Streaming and the tile cache

`src/scene/tiles.ts` loads static tiles whose footprint is within about 235 m of a point 45 m ahead of the car
(map view: proportional to the zoom), nearest first, through a pool of 3 workers (2 on phones), unloads beyond
+70 m, keeps a splat budget (4.5 M loaded on desktop, 1.8 M on phones) and keeps recently unloaded tiles decoded
in an LRU (450 MB / 80 MB) so scrubbing back does not refetch. Tiles fade in over 0.45 s.

Contract PLY is 96 bytes per splat (1.4-2.7 GB per scene). `bun run cache <scene dir>`
(`tools/build-tile-cache.ts`) writes a compact `.rsb` copy, 33 bytes per splat, inside the scene
(`scenes/<name>/rsb/`, removed with the scene): positions as uint16 in the tile box (0.8 mm on a 50 m tile),
log scales as uint8 per tile range, Spark's 32-bit rotation, RGBA8, and the time texels ready to upload. The
viewer uses it when `rsb/index.json` lists a tile with the same splat count as `map.json`, else the PLY.
road-live-v2: 2689 MB → 924 MB in 5 s; road-stitched-v2: 1500 → 516 MB. The index also stores the scene's
white balance and whether `objects.json` exists (so scenes without movers do not log a 404).
`bun run cache:studio` (`tools/cache-on-studio.sh`) pushes the tool to `~/dev/reality-splat/viewer-cache-tool/`
on anomastudio and builds every scene in `~/dev/reality-splat/scene/` in place; unchanged tiles are skipped.

Streaming installs at most one decoded tile per frame and uploads its textures that frame. The loaded radius
follows the haze (172 m around a point 25 m ahead in the follow view), which halved loaded splats and cut
frames over 25 ms during a 2× run from 32 to 3-7 per 1200.

## Synthetic scene

`bun run synth` (`tools/make-synthetic-scene.ts`) writes `scenes/synth/` from `scenes/dev/drive.json` and
`rig.json`: lanes and markings, curbs, sidewalks, verges, a car park with a drive-through, intersections with
crosswalks and signals, street lights, trees, buildings, parked cars and moving cars (18 Hz transient snapshots),
3.5 M splats in 189 tiles of 64 m plus a sky panorama. Every surface is a quadtree / octree of up to four
generations (1.6, 0.8, 0.4, 0.2 m cells) born when a camera first sees it and refined when seen from within
40, 16 and 6 m, each superseding its parent with 0.25 s overlap; seen-runs come from the real rig (fisheye
fields of view, ranges 45-120 m, a body-occlusion box, no scene occlusion) sampled at 5 Hz. Runs in about 8 s.

## Car model

`public/models/model3.glb` comes from **"2024 Tesla Model 3" by RBLXSupercars (the credit embedded in the file; the project's notes also recorded "Tesla model 3 2024" by brandonleong28)**, Sketchfab, licensed
**CC BY 4.0** (https://sketchfab.com/3d-models/tesla-model-3-2024-36c52f3f89f6439c90310f14e8ff33f2), a Model 3
Highland. `tools/prepare-car.ts` bakes it into the ego frame (rear-axle origin, wheelbase 2.875 m, 4.72 m long),
splits the four wheels into hub-pivoted nodes, renames paint and lamp materials, resizes textures and
meshopt-compresses it (1.6 MB). The cabin has no roof glass, so the interior is darkened to read as tinted glass.

## Performance (basebook, Chrome-family browser)

| Scene | Viewport | fps | Splats drawn / loaded |
|---|---|---|---|
| road-live-v2, playing from 105 s | 1600×1000 @2 (3200×2000) | 60 | 1.1 M / 3.1 M |
| road-stitched-v2, playing from 105 s | 1600×1000 @2 | 60 | 2.3 M / 2.4 M |
| road-live-v2, playing from 300 s | 1600×1000 @2 | 53-61 | 1.0 M / 3.8 M |
| road-live-v2, phone emulation | 390×844 @1.5 (585×1266) | 60 | 0.8 M / 1.75 M |

After the realism pass (same machine, after a few seconds' warm-up): road-live-v2 60 fps (1.2 M / 3.5 M),
road-stitched-v2 60 (1.8 M / 2.2 M), road-live-v2 from 300 s 60 (0.8 M / 3.4 M), phone emulation 60
(0.9 M / 1.9 M), live-v3 on the published host 60 (1.1 M / 4.5 M).

The phone row is a desktop GPU with the phone budget and resolution, not an iPhone. Resolution adapts down
when frames run long (`dpr=` pins it). The page needs no secure-context APIs.

## Capture (film shots)

`tools/capture.ts` renders shots offline, frame by frame, for the film (`film/captures/`, not committed). Playback
speed does not matter: every frame is fully settled before it is grabbed, so the same shot renders the same
every time, at any size.

    bun tools/capture.ts tools/shots/ladder.json [--only id,id] [--out ../film/captures] [--force]

A shot list is a JSON array (`tools/shots/ladder.json`, `hero.json`, `director.json`):

| Field | |
|---|---|
| `id`, `description` | file name (`<id>.mp4`, `<id>-hud.mp4`) and the manifest's one-liner |
| `build` | `"head"`: the working tree, built into its own folder for the run (`<HEAD>+wt-<hash>` of the viewer's diff and new files, the patch kept in `film/captures/builds/`); a commit: that commit's viewer, built once in a temporary git worktree (`$TMPDIR/rs-capture-worktrees`) |
| `scene`, `params` | the scene and any URL flags (`mode`, `ground`, `vehicles`, `ribbon`, `car`, `director`, …) |
| `t0`, `t1`, `fps`, `speed` | drive-time range, output rate (30 or 60), drive seconds per output second (timelapse) |
| `size` | output pixels; 3840x2160 is a 1920x1080 page at device pixel ratio 2 (the HUD is laid out as on a 1080p screen) |
| `hud` | `["off"]`, `["on"]` or both: each a file of its own, from two screenshots of the same settled frame |
| `campath` | a camera path (below); none = the build's chase camera |
| `keys` | values eased between keys over drive time: `maskAmt` (the clip's Predicted overlay), `dir` (the director look's amount) |
| `preroll` | seconds stepped before `t0` so eases and the early builds' damped cameras are in their steady state |

How it works: headless Chrome (real WebGL on Metal, `ANGLE Metal Renderer: Apple M5 Max`) driven over CDP from
Bun with no dependencies; the build is served on 127.0.0.1 with `/scenes/*` proxied to the Studio's published
host (`RS_STUDIO`, Range requests passed through), so nothing is copied locally. Today's build has a capture
mode (`?capture=1`, off on the normal page): no clock, `window.__capture.step(T, dt)` advances eases and
smoothing by exactly one output frame, then the page renders with dt 0 until `state()` is quiet: no tile
loading, queued or fading in, the ground atlas's requests, uploads, coverage map and provenance all landed,
movers loaded and faded in, the vehicle kit ready, the generated clip seeked to the exact frame
(`floor(T·fps)`, texture re-uploaded) and a splat sort that started after the camera was in place has
finished (Spark sorts continuously, so its own flags never go quiet; `driveSort` is wrapped to timestamp
sorts). Then the car's environment is re-captured from that settled frame and the frame is settled again, so
reflections show the frame they belong to. The clip's fade follows drive time (in from the first frame of a
chain of clips, out over its last second) instead of wall time. Frames go to ffmpeg as PNG and come out HEVC
CRF 15, 4:2:0, a keyframe a second (visually lossless). Files are written as `.part.mp4` and renamed when
done; the manifest (`film/captures/manifest.json`: id, file, scene, viewer commit, camera, flags, keys, drive
range, fps, speed, size, frames, bytes, description, frames not settled) gains each shot as it finishes. A run
stops before a shot when under 6 GB are free (`RS_CAPTURE_MIN_FREE_GB`).

The library lives mostly on the Studio: `bun tools/offload.ts` mirrors every finished shot to
`anomastudio:~/dev/reality-splat/film/captures/` (same names, with the manifest, keep list and build patches)
and removes it here unless `film/captures/keep.txt` names it (what the film's storyboard uses and what wipes
are built from) or it is among the newest 3 GB (`--keep-gb`) of HUD-off shots; each manifest entry's
`location` says where its file is. `--fetch <id>` brings a shot back. It runs every 10 minutes while
captures run.

Early builds (`e7f92d6`, `e8e1ada`, `844ad5d`) have no capture mode. They are driven through `window.__rs`:
paused, `seek(T)`, and their `frame(dt)` wrapped so dt is the output step once per frame and 0 while it
settles, so the damped follow cameras before `d083540` lag behind the car exactly as at 1x. Settled there
means tiles loaded and faded in, a fresh sort, and no fetch, XHR or image in flight for 250 ms (photo layers).

Speed at 3840x2160 (one shot at a time): today's build 1.2 s per output frame with both HUD variants, about
0.6 s per frame with one; the first viewer about 0.6 s with both; a 12 s ladder rung is 5-8 minutes.

**Camera paths** (`src/camera/paths.ts`, also `?campath=<json>` on the page): pure functions of drive time on
the chase pose's own smoothed inputs, so nothing judders.

- `chase` (or none): the default chase camera.
- `orbit`: `radius`, `height`, `deg0` (relative to the heading, 180 = behind, 90 = left), `degPerSec`, `t0`.
- `crane`: from the chase pose up an arc to straight down `height` m above the car (heading up) and back:
  `t0`, `up`, `hold`, `down` seconds, smootherstep eases; position, orientation (slerp) and fov blend.
- `spline`: Catmull-Rom through `keys` (`t`, `pos`, `look`, `fov`) in the car's frame (x forward, y left, z
  up, from the rear axle on the ground) or the world's; starts and stops softly (velocity ramps over the first
  and last 15 %), constant speed between; `reach` streams and hazes round the look point instead of the car.
- `map`: locked top-down `height` m above the car, heading up or `northUp`.

Off the chase pose the generated clip fades with the camera's angle and distance from it, as when orbiting by
hand (a crane dissolves the clip into the splats as it lifts and back as it lands), and the car's reflections
come from a chase-pose frame rendered for the purpose (close to the car the frame on screen is mostly the car).

**URL flags added for shots**: `car=0` (no ego car), `ribbon=0` (no path ribbon), `campath=`, `capture=1`,
`director=` (below), `dreach=` (director arrival distance), `dpoint=` (point size, radians per metre).

**Director looks** (`src/render/director.ts`, `?director=<look>[,<look>]`, off by default; each a function of
drive time; the key `dir` blends it in and out):

| Look | |
|---|---|
| `points` | every splat as its centre, an opaque dot about 3 px across at 4K (size proportional to distance); `dir` 1 = dots, 0 = full splats, interpolated in log scale so small and large splats arrive together. Ground splats instead of the atlas |
| `depth` | colour by distance from the camera: warm white near, the accent blue, navy far, on a dark sky; the splat's own brightness kept as texture |
| `height` | colour by height above the road: slate road, blue kerbs, pale cyan at roof height, warm white crowns, amber above |
| `tiles` | each splat tile's bounds (accent-blue lines) light up and decay to a faint outline, and its splats fade in, when the car first comes within `dreach` (100 m) of it: drive order, not network order, so it is repeatable |
| `atlas` | the ground atlas alone on a dark ground and sky (no splats, no vehicles), its tiles fading in as the car comes within `dreach` (55 m); with `tracks` the clean vehicles stay |
| `atlasmask` | the atlas's observed / predicted mask (its Predicted overlay), `dir` = strength |
| `tracks` | every tracked vehicle's box (moving blue, parked white, predicted dimmer) and the last 6 s of each moving track as a ribbon on the road, predicted stretches dashed |

The clip provenance overlay is the existing Predicted overlay keyed with `maskAmt`. Wipes and split screens are
composited from registered captures (same camera path, drive times and size; every `today` ladder shot is
registered with every other): `bun tools/wipe.ts --out f.mp4 --in a.mp4 --in b.mp4 [--in c.mp4] --wipe t0:t1
[--wipe …] [--soft 0.03] [--line] [--rtl] [--desc "…"]` or `--split 0.5` (`--desc` adds it to the manifest with its
sources); the edge is vertical, smoothstep-eased, hard
(`--soft 0`) or soft, with an optional thin light line. The six camera frustums with their frames on the
image planes are not built.

Choices: the accent blue for everything a director look adds; dark backgrounds only where colour carries data;
north-up for the whole-drive map (heading-up spins the world at 5x), heading-up for short top-down shots;
close passes without the path ribbon (it reads as a blue slab at that distance); HEVC rather than ProRes to
keep the library under 15 GB on a nearly full disk.

## Files

`src/main.ts` app and render loop · `src/scene/` drive, rig, tile streaming, PLY/RSB codec, decode worker ·
`src/render/` splat time shader, car, path ribbon and chevrons, sky, camera frusta · `src/camera/controller.ts`
· `src/camera/paths.ts` scripted camera paths · `src/render/director.ts` director looks · `src/ui/` HUD and icons ·
`tools/` synthetic scene, tile cache, car preparation, street names, shot capture (`capture.ts`, `shots/`) and
wipes (`wipe.ts`).

## Known gaps

- The real scenes are ground only so far: no buildings, trees, traffic or sky panorama, so the scene reads
  emptier than Tesla's frames until the object layers land.
- The map view shows the splat world from above with the route; Tesla's map is a vector map.
- Street names exist only for this drive; the weather, battery and airbag badge are fixed placeholders.
- The car's reflections of its surroundings come from the frame on screen: directions behind the camera are
  mirrored from the view ahead, and in the map view the last chase capture is kept (`?envcap=0` turns the
  capture off, leaving sky, clouds and road).
- On phones, ground splats under the atlas are left out while it is shown; orbiting far behind the car can show
  the ground missing where the atlas has already been dropped (beyond about 50 m behind).
