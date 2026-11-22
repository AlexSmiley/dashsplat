# Naming and Tesla's marks

## The name

The public name is **dashsplat** (lower case, one word; repository `R44VC0RP/dashsplat`): Gaussian splats from
dashcam footage. It replaces two working names:

- **Reality Splat**, used while the project was built (21 and 22 November 2026). It is close to **Tesla Reality
  View**, which became the official name of Tesla's FSD v15 visualisation on 21 November 2026
  ([research/06-hugging-face.md](../research/06-hugging-face.md), section 5.11).
- **Reality Recapture**, proposed for the repository. It is close to **RealityCapture**, Epic Games'
  photogrammetry software.

No trademark search was done for "dashsplat".

## What this pass changed

Only places where the project names itself:

- [README.md](../README.md), [LICENSE](../LICENSE), [NOTICE](../NOTICE) and the other new files use dashsplat.
- Titles of [PLAN.md](../PLAN.md) and of the entry-point READMEs (`pipeline/run`, `pipeline/cuda`, `viewer`).
- The viewer's page title (`viewer/index.html`), package name (`viewer/package.json`, `viewer/bun.lock`) and Vite
  plugin name (`viewer/vite.config.ts`).
- Two research reports called the project by Tesla's feature name; that phrase now says "this project (then called
  Reality Splat)": `research/02-feedforward-and-online.md` line 3 (was 'the "Tesla Reality View" project') and
  `research/04-compute-and-viewer.md` line 3 (was 'a "Tesla Reality View" clone').

## Where "Reality Splat" remains

The stage reports, research reports, film documents, code comments and docstrings keep the working name: they
are records. It also remains in names the author's machines depend on: the installation path
`~/dev/reality-splat` in his helper scripts and in records, the launchd job
`deploy/dev.vogel.reality-splat-viewer.plist`, the `RS_` prefix of environment variables, the film's output
files (`film/deliver.sh`: `reality-splat-journey*.mp4`) and **the film's title card**
(`film/render/src/shots.rs` line 246, `"REALITY SPLAT"`; the delivered cut shows it). Changing the title card
is a one-line edit if the owner wants the film to match.

## Uses of Tesla's names and imitation of its screen (for the owner to decide)

Nothing below was changed. Following research/06 section 5.11, describing compatibility ("works with Tesla
dashcam footage") is fine; imitating Tesla's screen and using its product names in the product is the part to
decide on.

### Tesla's feature name "Reality View"

| Where | Use |
|---|---|
| [PLAN.md](../PLAN.md) line 5 | "Our own version of Tesla's Reality View": describes the goal |
| [viewer/README.md](../viewer/README.md) line 16 | "styled after Tesla's Reality View screen" |
| `viewer/src/style.css` line 1, `viewer/src/ui/hud.ts` line 1, `viewer/src/camera/chasePose.ts` line 10 | Code comments saying what the HUD and camera were modelled on |
| `film/OUTLINE.md` | The video's working title, "I rebuilt Tesla's Reality View from my dashcam", and talking points |
| `film/README.md`, `film/render/src/lib.rs` line 1 | Describe the film's subject |
| `research/01-tesla-cameras-and-reality-view.md` (file name and text), `research/05`, `research/06` | Research about Tesla's feature |
| [TIMELINE.md](../TIMELINE.md) | Commit messages `e7f92d6` "Add the Reality View viewer" and `c7f10a6`; history, kept verbatim |

### The viewer imitates Tesla's screen

| Where | What |
|---|---|
| `viewer/src/ui/hud.ts`, `viewer/src/style.css` | HUD modelled element by element on Tesla's screen: P R N D gear row, battery percentage, thin speed numeral with MPH, "Self-Driving" / "Autosteer" / "Cruise" with a mode line, "Streak", speed-limit sign, light icons, next-turn card, lock, clock, weather, "PASSENGER AIRBAG ON" badge, a media card used as the timeline, a trip card with "End Trip" |
| `viewer/src/ui/icons.ts` | Icons drawn "in the visual language of the Tesla UI" |
| `viewer/src/render/path.ts`, `viewer/src/ui/hud.ts` line 593, viewer README | "Tesla-blue" path ribbon and self-driving stretches |
| `viewer/src/style.css` lines 144 and 162 | Legend "styled as a Tesla card"; "Tesla's layout, shrunk" on narrow windows |
| `viewer/src/camera/chasePose.ts`, viewer README "Chase pose", `viewer/tools/shots/*.json` | Chase camera measured on Tesla's promo frames ("Tesla-matched chase pose") |
| viewer README "What is on screen" | Inter chosen as the closest free face to Tesla's Gotham-like UI type; battery, weather and airbag badge "shown the way the car would" |

### Tesla product names and designs

| Where | What |
|---|---|
| `viewer/public/models/model3.glb`, `viewer/src/render/car.ts`, `viewer/src/render/vehicles.ts`, `viewer/tools/prepare-car.ts` (node `tesla_model_3_highland`), `film/viz/car_points.ts` | A Model 3 Highland model as the ego car and as every saloon in "Clean vehicles". Its CC BY 4.0 licence covers the model, not Tesla's design |
| `viewer/src/render/car.ts` lines 21 to 26 | Paint menu with Tesla's paint names: Quicksilver, Pearl White, Stealth Grey, Diamond Black, Deep Blue, Ultra Red |
| Scene names `hw4-drive`, `hw4-raw-224-254`, `hw4-cuda-224-254`; "Model 3" and "HW4" across documents | Describe the source car and its camera hardware |
| `pipeline/ingest/ingest.py`, `pipeline/run/raw.py`, `pipeline/cuda/window_clips.py` | "Tesla SEI", "Tesla dashcam clips": describe the input format |

### Images

Most viewer screenshots and recordings in the reports show the Tesla-style HUD and the Model 3 model (for example
`viewer/report-night/`, `viewer/screenshots/`, `pipeline/genfill/report-clip/viewer-playback-*.jpg`, and the
README's first image). Tesla's own promo frames (`reference/`) were removed in the privacy pass.

### Suggested direction (research/06 section 5.11, not done)

Keep the information (speed, gear, path ribbon, blinkers, drive state, time) in the project's own layout and colours;
replace Tesla's wording ("Self-Driving", "Streak", "End Trip", the airbag badge) and paint names; use a neutral car
body, optionally in the driver's paint colour; keep labelling generated and predicted content.
