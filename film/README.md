# Reality Splat: the journey (film)

## State at handoff (22 Nov 2026, 09:45 EST)

**Last render: the step-1 cut, 4:05.7, 3840x2160 / 30 fps, master rendered on the Studio 09:13-09:20, chapters
09:20-09:40** (chapters 04, 09, 11 and 12 re-rendered after a fix: a chapter whose first shot cross-faded from a
missing previous shot came out black; `Journey::new` now gives a lone first shot a 0.4 s fade instead). Files in
`anomastudio:~/dev/reality-splat/scene/_preview/journey/`, served on the LAN:

- film with mix: http://10.0.50.51:8090/scenes/_preview/journey/reality-splat-journey.mp4
- 1080p proxy: http://10.0.50.51:8090/scenes/_preview/journey/reality-splat-journey-1080p.mp4
- silent master `reality-splat-journey-silent.mp4`, stems `stems/{mix,ambience,effects}.wav` (24-bit; the mix
  measures -20.0 LUFS integrated, LRA 9.9 LU, true peak -3.1 dBTP on the delivered AAC), chapters `chapters/00-cold-open.mp4` … `12-close.mp4`, `contact-sheet.png`.
  Local copy here: `film/out/contact-sheet.png` and the proxy only. Older 08:11 rough cut: `journey-rough-0811-*`
  in the same folder and `film/out-placeholder-0811/` on the Studio.

**Captures already in the cut:** Open and the big panel of Final2 (`hero-gen-palms-junction-299-329-60fps`, cut to
314-321 s and 306-321 s), Splats (`dir-points-to-splats-110-122`), GroundLive (`ladder-r1-road-live-v1-110-auth`),
FirstWorld (`ladder-r2-live-v4-110-auth`), and the Evolution montage (every 110 s rung present: r1 v1, r1 v2, r2 v3,
r2 v4, 6.2 s; it grows by itself, max 11.8 s).

**Still stand-ins, with the capture each waits for** (`notes/capture-requests.md` has the same list; the
capture engineer has it):

| Shot | Stand-in now | Waits for |
|---|---|---|
| Ladder (ch 5) | viewer stills at 112 s, upscaled | `ladder-r2-live-v4-110-today`, `ladder-r3-stitched-opt-v2-110-today`, `ladder-r3-stitched-opt-v3-110-today` (all three) |
| CutOuts (ch 6) | `viewer/screenshots/photo/run-a-ibr-v1-105-120.mp4` | `ladder-r4-stitched-opt-v3-110-auth-photo` |
| Teacher (ch 6) | report clip v3 \| v4 (1280x426) | `ladder-r3-stitched-opt-v3-110-today` + `ladder-r5-stitched-opt-v4-110-today` |
| Wide (ch 7) | v6 before/after stills at 118 s | `ladder-r5-stitched-opt-v5-110-today-splatground` + `ladder-r6-stitched-opt-v5-110-today-atlas` |
| Movers (ch 8) | report clip + `scratch/night2` | `ladder-r7-stitched-opt-v5-movers-v1-284-today-scan…` + `…-284-today-clean…` |
| Final1, Close (ch 10, 12) | `scratch/night3/g093.mp4` (HUD cropped) | `hero-gen-100-122-60fps` (cut to 105-116.5 s and 113-122 s) |
| Final3 (ch 10) | `scratch/night3/g190.mp4` | `hero-gen-redlight-190-206` (cut to 196.5-203.5 s) |
| Split (new, after ch 9) | not in the cut yet | `ladder-r7-…-312-today-clean` + `ladder-r8-…-312-today-generative` |

FirstDemo and SecondDemo are the demos exactly as sent; Morph and NVS are report media; they are final.

**To pick up a capture:** on the Studio `python3 film/viz/ingest_captures.py` (links new shots into
`work/media/cap-<id>.mp4`, cuts the hero stretches listed in its `DERIVED` table), then re-render: a chapter alone
with `CHAPTER=<n> ./target/release/journey render -o …`, or everything with `OUT=<dir> sh deliver.sh`. No code change
is needed: `render/src/shots.rs` (`captured()`, `evolution()`) switches each shot when its file exists.

**Known issues and half-done things**
- Length: 4:05.7 now; with the full Evolution (≈12 s) and the Split (7 s) it becomes ≈4:18. Trim first: Unfold
  8.4 → 7.5, Cameras 7 → 6, Morph and NVS together, Final2 11.5 → 10, Pipeline 11 → 10.
- The Evolution's first rungs come from the first viewer build with its own follow camera (`-auth`), so wipes
  between those and today's rungs are not pixel-registered; today's (`-today`) rungs register with each other.
- Depth shot: the last 2 s (keyframes accumulating, `viz/depthfly.py`) are noisy; the first 10 s are the strong part.
- The sound is synthesised (`sound/synth.py`) and has only been checked by measurement (loudness, peaks,
  spectrograms), never by ear. The beds are deliberately plain.
- `inspect` reports one harmless warning on the very last (black) frame.
- basebook no longer has `render/target` or `work/media`; build and render only on the Studio (`film/studio.sh`).

**What I would do next, in order**
1. As each capture lands: ingest, re-render the changed chapter, check it with `journey strip <Shot>` (fetch the PNG).
2. When the Ladder, Teacher, Wide and Movers captures are in, trim to about 4:00 and re-run `deliver.sh`.
3. Listen to the mix with Ryan; swap beds or effects for real recordings or the Suno takes if he wants.
4. Adjust label timing to his voiceover once it exists (labels are in `render/src/shots.rs`, one line each).

**Tooling notes**
- Full `deliver.sh` on the Studio: about 6 minutes for the 4K master (350 s of rendering at nice 15 beside the
  capture job), 1-2 minutes for audio, stems and contact sheet, about 8 minutes for the 13 chapters. It runs at
  `nice` by itself; run one at a time. `--skip-master`, `--skip-chapters` and `--only-chapters "4 9"` re-do parts;
  `OUT=<dir>` picks the output folder (the cut above used `OUT=~/dev/reality-splat/scene/_preview/journey`). The 1080p
  proxy is a plain `ffmpeg -vf scale=1920:1080 -crf 21` of the master, not part of `deliver.sh`.
- Fonts: `render/media/BerkeleyMono-{Regular,Bold,Black}.ttf` are static instances cut from Ryan's licensed
  variable font with `uv run ~/.agents/skills/glitch-sizzle-reel/scripts/berkeley_fonts.py render/media`; they are not
  in git, so a fresh checkout needs that step before `cargo build`.
- Toolchain on the Studio: `export PATH=$HOME/.cargo/bin:/opt/homebrew/bin:$PATH` (cargo is in the home folder, not on
  the default PATH of a ferry shell). fframes 1.2.0 with Skia on Metal; the first build there took under a minute.
- Suno takes: 14 sounds × 2 takes were generated in Ryan's Suno library (titles "reality-splat amb_cabin",
  "… amb_room", "… pad", "… shutter", "… sweep", "… shimmer", "… tick", "… power_down", "… power_up", "… unfold",
  "… passby", "… sub", "… blips", "… dissonant"). None was downloaded: each download costs one of his monthly
  download credits (57 left at 08:00) and needs his OK. To fetch one: `tools.suno["audio-download"]({ id, format:
  "wav", confirm: true })` with the clip id from `tools.suno["library-search"]`, then drop the file in `work/audio/`
  under the name the film uses (e.g. `amb_cabin.wav`) and re-render.
- The point-cloud shots (`viz/depthfly.py`, `viz/unfold.py`) run on any Mac with torch/MPS; their inputs are on the
  Studio under `film/work/` (`points-111.npz`, `hero111/`, `car-points.bin`). Rendered results:
  `film/work/media/{depth,unfold}.mp4` there, originals in `film/work-local/`.


A four-minute documentary film of one day and night (15:45 on 21 Nov to 04:15 on 22 Nov 2026) of rebuilding Tesla's
Reality View from one Model 3 drive in Jacksonville. Made for Ryan to cut into a YouTube video with his own
voiceover: visualisation first, very little text, an ambient bed and effects, no voice and no music track.

Everything on screen is project footage, project data or a recording of the project's own viewer; nothing is
Tesla's footage and nothing is mocked up. Every number on screen is a measurement from the project's reports;
[notes/facts.md](notes/facts.md) lists each with its source (path:line or commit).

## Deliverables (on the Studio in `~/dev/reality-splat/scene/_preview/journey/`, not committed)

| File | What |
|---|---|
| `reality-splat-journey.mp4` | 3840x2160, 30 fps, H.264 (crf 19) + full mix (AAC 320k, 48 kHz stereo) |
| `reality-splat-journey-silent.mp4` | the same picture with no audio, the master to cut against |
| `stems/mix.wav`, `stems/ambience.wav`, `stems/effects.wav` | 48 kHz stereo 24-bit; ambience + effects ≈ mix (the effects stem carries the same peak limiter as the mix) |
| `chapters/NN-name.mp4` | each of the 13 chapters rendered on its own (opens and closes cleanly), picture + mix at the film's gain |
| `contact-sheet.png` | 60 frames across the whole film |

Loudness: the full mix is at **-20 LUFS integrated, true peak -3.0 dBTP** (measured on the delivered AAC), so a
voiceover at about -16 LUFS sits clearly on top. The stems and chapters carry the same gain as the mix.

## Running order (the 09:15 cut, 4:05.7)

| Ch | Cut | Chapter | Shots |
|---|---|---|---|
| 0 | 0:00.0 | Cold open | Open, Title |
| 1 | 0:09.1 | Input | Unfold, Cameras, Drive |
| 2 | 0:33.5 | Geometry | Calib, Depth, Splats, Masks |
| 3 | 1:05.0 | First ground | Mosaic, GroundLive, FirstDemo |
| 4 | 1:25.5 | First world | FirstWorld |
| 5 | 1:33.5 | Optimiser | Ladder, SecondDemo |
| 6 | 1:51.0 | Experiments | CutOuts, NVS, Morph, Teacher |
| 7 | 2:14.0 | Road atlas | Atlas, Lane, Wide (with the 21:19 power cut) |
| 8 | 2:41.0 | Vehicles | Tracks, Movers |
| 9 | 2:57.5 | Image model | ImageModel (+ Split once its captures land) |
| 10 | 3:15.5 | Finished view | Evolution, Final1, Final2 (six cameras alongside), Final3 |
| 11 | 3:46.7 | One command | Pipeline |
| 12 | 3:57.7 | Close | Close (ends 4:05.7) |

Every shot with its source and sound is in [STORYBOARD.md](STORYBOARD.md) (written for the 3:59.5 cut; the shots
added since are Splats, GroundLive and Evolution, all captures, and `timeline` gives the current times).

## Sources of the shots

- **Real-data visualisations (made for the film):** the six-camera unfold (each pixel of the real 111 s frames
  placed along its calibrated ray, `calib/rig.json` via `pipeline/common/rig.py`; car from
  `viewer/public/models/model3.glb`, point-sampled); the depth flight (the project's DA3 / MapAnything depth maps
  at the 111 s keyframe and every third keyframe 104-120 s, unprojected with the rig and the refined trajectory
  `poses/ego.npz`); the drive map (`poses/ego.npz`, `data/drive/timeline.npz`); the ground atlas map and the zoom
  to 1 cm (the published `scene/stitched-opt-v5/ground` tiles, composed at 0.2 m, 4 cm and 1 cm per pixel); the
  junction tracks (`scene/stitched-opt-v5-movers-v1/objects.json`); the mask overlay (`masks/sky`,
  `masks/dyn` and their JSON, Mask2Former); the image-model panel (`work/genfill/clip/c093`: `render/`, `b/`,
  `b-prov/`, reference photos from `frames.json`); the six-camera clips (`data/drive/frames`).
- **Report media:** calibration evidence (`scratch/calib/debug/front-transfer.jpg`), bird's-eye mosaic
  (`$RS/road/mosaics/offline/turn/best.jpg`), live world v4 (`pipeline/livemap/report/`), StreetNVS
  (`pipeline/genvideo/report/generated.mp4`), re-projected photograph (`pipeline/ibr/report-m/110-125/chase.mp4`),
  v3 | v4 (`pipeline/stitch/report-v4/chase-110-125.mp4`), lane dash (`pipeline/groundtex/report/detail-lane.jpg`),
  ground v6 before/after (`pipeline/groundtex/report/v6/`), scanned movers (`pipeline/movers/report4/pull-away.mp4`).
- **Viewer recordings (HUD cropped away):** `scratch/demo` (first and second demo as sent), `scratch/night2`,
  `scratch/night3` (Generative mode), `viewer/screenshots/photo/run-a-ibr-v1-105-120.mp4`, the 112 s stills in
  `scratch/opt2`, `opt3`, `v4`.
- Car model in the unfold and depth shots: "2024 Tesla Model 3" by RBLXSupercars (the credit embedded in the file; the project's notes also recorded "Tesla model 3 2024" by brandonleong28), CC BY 4.0 (as in the viewer).

## Decisions taken (no questions asked)

- **4K (3840x2160).** The film renders at about 16 fps on this Mac (7.5 min for the whole film), so 4K was practical.
  Most viewer recordings are 1440x900, so they are upscaled; captures from the camera department replace them.
- **Sound is synthesised** (`sound/synth.py`: shaped noise, tones and grains; deterministic). I generated takes
  in Ryan's Suno library (14 sounds, 28 clips titled "reality-splat ...", 14 credits), but every download
  needs one of his monthly download credits and the tool asks for his approval per download, so none were taken.
  The beds are deliberately plain so they can be swapped.
- **Commit thread stops at 04:15** (commit 3b7a42b), as briefed; the 06:34 commit is outside the film.
- **"First world" uses live world v4 (17:20).** The `world-stitched` demo in `scratch/demo` is optimiser v3 (its
  HUD reads STITCHED-OPT-V3, saved 18:17), so it is shown as the second demo at the end of the optimiser chapter.
- **Provenance colours:** observed = accent blue, repaired = warm white, filled = grey, sky/haze = near black.

## How it is built

```
film/
  STORYBOARD.md, README.md, notes/facts.md
  conform.json / conform.py   cut every source clip/still into film/work/media (30 fps H.264, short GOP)
  viz/                        real-data visualisations
    pack_points.py              (Studio) unproject depth keyframes into coloured world points
    depthfly.py, unfold.py      point-cloud shots (torch on MPS), pointcloud.py = shared z-buffered splatter
    car_points.ts               point-sample the viewer's car model (bun, viewer's node_modules)
    ground_mosaic.py            (Studio) compose the published ground atlas into top-down images
    studio_clips.py             (Studio) camera clips, mask overlay, image-model clips
    export_data.py              JSON tables compiled into the renderer (path, commits, vehicles, atlas placement)
    peek.py                     (Studio) contact sheet of camera frames at given times
  sound/synth.py, master.py   every sound; loudness and peak delivery
  render/                     the fframes project (Rust, Skia/Metal): shots.rs (running order), kinds.rs (shot
                              types, commit thread, power cut, vignette/grain), sound.rs (beds and cues), lib.rs
  deliver.sh                  renders master, stems, chapters and contact sheet into film/out
  work/ out/                  (ignored) media, audio, data, reviews / deliverables
```

## Re-render (on the Studio)

Rendering, encoding and the point-cloud passes run on anomastudio (`~/dev/reality-splat/film/`, low priority,
one job at a time); basebook only edits source and fetches review images and deliverables. `film/studio.sh`
wraps it:

1. `sh film/studio.sh push` (source, sound, conform tables, small data) and, once, `sh film/studio.sh push-media`.
2. Sources: on the Studio run the scripts in `film/viz` (each documents its arguments; e.g.
   `.venv/bin/python film/viz/studio_clips.py cams 108 8 cams108`), `python3 film/conform.py`,
   `python3 film/viz/ingest_captures.py` (links the camera department's library from `film/captures` and cuts the
   hero stretches the film uses), `uv run --with numpy --with scipy python film/sound/synth.py`. The two point-cloud
   shots (`viz/depthfly.py`, `viz/unfold.py`, torch on MPS) need `work/points-111.npz` (`viz/pack_points.py`),
   `work/hero111` and `work/car-points.bin` (`viz/car_points.ts`).
3. `sh film/studio.sh build` (fframes 1.2.0, Skia on Metal; fonts in `render/media` are Berkeley Mono instances
   cut by `~/.agents/skills/glitch-sizzle-reel/scripts/berkeley_fonts.py`, licensed, not committed).
4. Review: `sh film/studio.sh run 'journey strip Depth -n 8 -o ../work/review/s.png'`, then
   `sh film/studio.sh fetch work/review/s.png`. Also `journey timeline | inspect | frame | audio analyze`.
5. `ferry sh anomastudio --detach 'cd ~/dev/reality-splat/film && sh deliver.sh'` writes everything into
   `~/dev/reality-splat/film/out/` there (about 30 minutes); fetch the deliverables into `film/out/`.

Shots switch from placeholder to capture on their own: `render/src/shots.rs` (`captured()`, `evolution()`)
uses a capture whenever `work/media/cap-<id>.mp4` exists, so re-running ingest and the render picks up new
captures without code changes.
