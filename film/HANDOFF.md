# Journey film: handoff

Written 22 Nov 2026, about 09:30 EST, by the session that ran the Reality Splat project overnight. This hands the
**journey film** to a new primary session that works with Ryan directly on his YouTube video. Everything the film
needs is listed here with its location. The reconstruction project itself (pipeline, GPU port, Hugging Face study,
whole-drive run) stays with the original session.

## What Ryan asked for

A film of the project's progress for a YouTube video he is making, in his words: "showing all of the iterations...
starting from the beginning, showing the dash cam footage, and then the first version and the second version, and
then some cool visualizations to see it evolve"; "visualization heavy, and not a ton of text, because I'll probably
do a voiceover"; "ambient background sounds and some sound effects"; "super cool and realistic" with "nice VFX".
He records his own voiceover. He is making the YouTube video now and needs this film to cut against.

## What exists right now

- **A complete rough cut (3:59.5, 4K, with sound), rendered 08:11**, with stand-in footage in several chapters:
  - 1080p: http://10.0.50.51:8090/scenes/_preview/journey/journey-rough-0811-1080p.mp4
  - 4K: http://10.0.50.51:8090/scenes/_preview/journey/journey-rough-0811-4k.mp4
  - contact sheet: `scratch/film/journey-rough-0811-sheet.jpg`
- **The film's source**: `film/` in this repo. Read `film/README.md` (deliverables, running order, source of every
  shot, decisions, how it is built and re-rendered), `film/STORYBOARD.md` (every shot, its source and sound; "P"
  marks a stand-in waiting for a capture) and `film/notes/facts.md` (every on-screen number with its source).
  It is an fframes project (`film/render/`, Rust, Skia) plus Python visualisation and sound scripts.
- **In flight when this was written**: the film engineer (a helper of the original session) is rendering "step 1",
  a complete cut using the best footage available, into `~/dev/reality-splat/scene/_preview/journey/` on the
  Studio, and will then stop and record its final state at the top of `film/README.md`. Before you change anything
  under `film/` or start a render, check that it has finished: `ferry sh anomastudio 'pgrep -fl "journey|deliver.sh"'`
  should show nothing, and `film/README.md` should carry a "State at handoff" section. Until then, read, watch
  and plan with Ryan.
- **The capture library** (frame-exact 4K shots of the viewer, the footage that replaces the stand-ins) is still
  being shot on the Studio by another helper of the original session: `~/dev/reality-splat/film/captures/` with
  `manifest.json` (16 of about 69 shots at 09:10; the film's missing shots are first in the queue). The film
  picks a capture up automatically when it lands (see "Shots switch from placeholder to capture" in the README):
  re-run the ingest and the render.

## Where everything is

**This Mac (basebook), repo `/Users/vogel/dev/experiments/2026-11-21-tesla-reality-splat/`**
- `AGENTS.md`: the rule that matters most. **basebook must stay free**: no rendering, encoding, captures or model
  runs here. All of that runs on the Mac Studio.
- `PLAN.md`: the project's architecture. `git log --reverse --format='%h %ad %s' --date=format:%H:%M`: the
  timeline (the film's skeleton: 15:45 on 21 Nov to the morning of 22 Nov).
- Reports with the measurements and stills behind each chapter: `pipeline/calib/REPORT.md`,
  `pipeline/road/REPORT.md`, `pipeline/depth/REPORT.md`, `pipeline/stitch/REPORT-v4.md` and `REPORT-v5.md`,
  `pipeline/ibr/MORPH.md`, `pipeline/movers/` (`ROUND3.md`, `report4/`), `pipeline/groundtex/`
  (`REPORT.md`, `REPORT-V5.md`, `REPORT-V6.md`, `report/`), `pipeline/genfill/` (`REPORT.md`, `report/`,
  `report-clip/`), `pipeline/run/README.md`, `research/05-diffusion-route.md`, `research/06-hugging-face.md`,
  `viewer/report-night/`.
- `scratch/` (not in git): everything shown to Ryan along the way. The demos as sent: `scratch/demo/` (first two,
  21 Nov) and `scratch/demo2/` (`reality-splat-traffic-first-47s.mp4`, `reality-splat-junction-15s.mp4`);
  viewer recordings `scratch/night*/`; stills in `scratch/opt*`, `v4`, `v5`, `movers`, `genfill`, `now`.
- `viewer/`: the web viewer. `viewer/README.md`, section "Capture (film shots)": the capture tool
  (`viewer/tools/capture.ts`), camera paths, "director" render modes (points, depth, height, tiles, atlas, atlas
  mask, tracks), registered wipes (`viewer/tools/wipe.ts`), shot lists in `viewer/tools/shots/`.
- `reference/`: Tesla's own promo frames and video, for comparison only. Do not put Tesla footage in the film.

**Mac Studio (`anomastudio`), folder `~/dev/reality-splat`** (reach it with the `ferry` skill:
`ferry sh anomastudio '...'`, `ferry cp`; long jobs with `--detach`)
- `film/`: the film project as rendered there (`render/` with the built binary, `work/` media and audio,
  `out/`, `out-placeholder-0811/` with the rough cut and stems, `captures/`, `capture/` with the capture tool
  and its Chrome). `sh film/studio.sh push|build|run|fetch` from basebook wraps all of it.
- Raw footage: `data/clips` (the Tesla's saved clips), frames at 36 fps in `data/drive/frames/<cam>/`, tables in
  `data/drive/cams/`, poses, telemetry.
- Project data: `depth/`, `masks/`, `road/` (bird's-eye mosaics), `work/ibr/teacher-*` (teacher views),
  `work/genfill/clip/` (the image model's per-frame input, output and provenance), `maps/`, `work/drives/`
  (the one-command pipeline's runs).
- Every published scene: `scene/` (from `road-live-v1` to `stitched-opt-v5-movers-v1`, the pipeline-built
  `hw4-raw-224-254` and `hw4-drive`), served with the viewer at **http://10.0.50.51:8090/** (LAN, plain http;
  `?scene=<name>&t=<seconds>`; the default page opens the finished generated view). `scene/_preview/` is a
  folder for sharing files by link: `http://10.0.50.51:8090/scenes/_preview/<file>`.

## The story, for orientation (verify against the reports before putting a number on screen)

Six cameras and telemetry in (36 fps, calibrated to under a pixel) → the road first, as a bird's-eye mosaic turned
into ground splats (first demo, 16:51) → depth from two models and masks, a first blocky world → an optimiser
fitting the splats to the photographs (v1-v3; second demo 18:17) → two looks Ryan rejected (photographic cut-outs
with after-images; re-projected photographs) → the re-projection kept only as a teacher (v4) and a standing-world
pass (v5) → other vehicles as tracked models (scanned, later clean) → the road rebuilt as a 1 cm texture atlas (a
lane-line edge from 4.5 cm to 1.2 cm, against 1.0 cm in the photograph), extended 40 m to the sides → a one-step
diffusion model (Difix) redrawing every chase frame from the splat render plus a real photograph, stabilised in
time → the car lit by its surroundings, clean vehicle models, one grade → a one-command pipeline (36 minutes for
30 seconds of driving on the Studio) → the honest distance to real time (about 70x). A power cut hit the Studio at
21:19 and everything resumed. Known flaws worth being honest about: invented clouds, a stand-in model for the black
coupe, no generated clip under the overpass, the generated view exists only from the chase camera.

## Rules that apply to you

- basebook stays free (see `AGENTS.md`); render on the Studio, one job at a time at low priority, and check what
  is running there first (captures, and later the paused whole-drive reconstruction, share it).
- Sound: the current beds and effects are synthesised (`film/sound/synth.py`). The film engineer also generated
  takes in Ryan's Suno library ("reality-splat ..." clips) but did not download them, because each download uses
  one of his credits: ask him before using them.
- Fonts: Berkeley Mono instances in `film/render/media` are licensed and not committed.
- Nothing shown as ours may be Tesla's footage or a mock-up; every number on screen is a real measurement.
- Files over 5 MB stay out of git; deliverables go to the Studio's `scene/_preview/journey/` and are linked from
  there. Taildrop to Ryan's Air (`airbook`) was failing this morning (502 while it was offline).
- If you need new viewer shots: the capture tool runs on the Studio (README section above). One capture at a time;
  if a capture queue is already running there, add your shots after it rather than in parallel.
