# genfill: generative repair of novel views (Difix-ref on Apple Silicon)

Renders the map from novel viewpoints, repairs each render with a single-step diffusion fixer
conditioned on a nearby real photograph, and writes the repaired image together with a
per-pixel record of what the fixer changed, so the optimiser can distil the detail into the
splats as weak supervision and label fixer-only content **predicted**. Results and the decision
are in [REPORT.md](REPORT.md).

## Model, licence, settings

| | |
|---|---|
| Model | **Difix** (NVIDIA, Difix3D+, CVPR 2025): SD-Turbo UNet with cross-view attention to one reference image, LoRA'd VAE decoder with encoder skips. Weights `nvidia/difix_ref` (render + reference) and `nvidia/difix` (render only), Hugging Face, ungated |
| Licence | **NVIDIA License, research and non-commercial use only**; inherits the Stability AI SD-Turbo community licence. Fine for this personal project, not for a product. Code: `third_party/Difix3D` (same licence) |
| Settings | one step, prompt "remove degradation", timestep **199** (the training value; 99 is the gentle option), guidance 0, float16 on MPS, seed 0, one reference (the top-ranked photo), no blending |
| Speed, memory (M5 Ultra, MPS) | 1.6 s per 1024x576 image uncontended, 2.1-3.0 s with depth inference sharing the GPU; see REPORT.md for 1408x896. Peak Metal driver memory about 14 GB (float16; float32 is no faster and takes 24 GB) |

## Run on the Studio

From `~/dev/reality-splat` (push this folder first: `ferry cp ./pipeline/genfill anomastudio:dev/reality-splat/pipeline/`):

~~~sh
export RS=$PWD
bash pipeline/genfill/setup.sh                 # once: envs/genfill (uv, Python 3.12), clones, weights
# chase poses come from the viewer: on basebook, cd viewer && bun tools/chase-track.ts ../scenes/<scene> 84 141 100 1408 896 > track.json,
# copied to work/genfill/chase-track-<scene>.json and passed as --track
# 1. Render sets (shared .venv: it has the rig, photometry and raster modules' dependencies)
.venv/bin/python -m pipeline.genfill.views chase --scene scene/live-v4 --out work/genfill/sets/chase-v4
.venv/bin/python -m pipeline.genfill.views real  --scene scene/live-v4 --out work/genfill/sets/real-v4 --ref-away 2
# 2. Repair (resumable; one model per process)
envs/genfill/bin/python -m pipeline.genfill.fix_views work/genfill/sets/chase-v4 \
    --out work/genfill/fix/chase-v4/ref-t199 --timestep 199
# 3. Cross-view agreement (demotes inconsistent repaired pixels), then the optimiser manifest
envs/genfill/bin/python -m pipeline.genfill.consistency work/genfill/sets/chase-v4 --fix work/genfill/fix/chase-v4/ref-t199
.venv/bin/python -m pipeline.genfill.distil pseudo work/genfill/sets/chase-v4 work/genfill/fix/chase-v4/ref-t199 work/genfill/distil/pseudo.json
.venv/bin/python -m pipeline.genfill.distil fit --source scene/stitched-opt-v3 --out scene/stitched-opt-v3-fix1 --pseudo work/genfill/distil/pseudo.json
.venv/bin/python -m pipeline.genfill.distil check work/genfill/distil/pseudo.json scene/stitched-opt-v3 scene/stitched-opt-v3-fix1  # did the splats move towards the repairs?
# 4. Checks
envs/genfill/bin/python -m pipeline.genfill.evaluate real  work/genfill/sets/real-v4 --fix ref-t199=work/genfill/fix/real-v4/ref-t199 --out work/genfill/eval/real.json
envs/genfill/bin/python -m pipeline.genfill.evaluate chase work/genfill/sets/chase-v4 --fix ref-t199=work/genfill/fix/chase-v4/ref-t199 --out work/genfill/eval/chase.json
~~~

`views.py chase` options: `--t0/--t1` slice, `--step` metres between stations (2), `--kinds`
(`mid,orb-60,orb-30,orb+30,orb+60`), `--t-map` to render every view at one playback time instead
of its own. `fix_views.py` options: `--model ref|noref`, `--timestep`, `--blend` (output =
render + blend x (fix - render)), `--ids` (comma-separated substrings), `--every`.

## Inputs

A render set is a folder with `views.json` (one entry per view: `id`, `T_world_cam` 4x4 with
OpenCV camera axes in the ENU world, `K`, `size`, playback time `t_map`, and `refs`: ranked
photographs `{c, n, t, score, path}`), `render/<id>.png` and `render/<id>.npz` (`alpha` splat
coverage, `z` depth in metres, 0 = none). Any renderer can produce one; sizes must be multiples
of 64. If a reference's `path` is missing, `fix_views.py` rebuilds it from
`data/drive/frames/<c>/<n>.jpg` (undistorted to that camera's pinhole design, ground-v2
photometry applied).

`views.py` makes two kinds of set from a contract scene: **chase** (the viewer's default chase
camera from `viewer/src/camera/controller.ts`, every 2 m of travel, plus 30 and 60 degree orbits
either side, 1408x896, which is 1440x900 rounded to Difix's 64-pixel grid) and **real** (held-out
photographs' poses, 1024x576, the photo as ground truth, references at least `--ref-away` metres
away). Static tiles only (moving-object tiles are skipped), the sky panorama behind the splats as
in the viewer, no distance haze. References are scored on 1,500 rendered surface points: seen
inside the photo's valid area, from a similar direction (cos^4) and distance (min/max ratio).

## Outputs (per view, in `--out`)

| File | Meaning |
|---|---|
| `fixed/<id>.png` | repaired RGB, same size as the render |
| `change/<id>.png` | 255 x *changed*: CIELAB distance between render and repair after a 1.5 px blur, mapped 4 -> 0 and 20 -> 1 (smoothstep); 1 where the render has no surface (alpha < 0.5). Pure sharpening scores low; moved, recoloured or new structure scores high |
| `class/<id>.png` | 0 **kept** (changed < 0.25), 1 **repaired** (0.25-0.75), 2 **fixer-only** (no rendered surface below the horizon, changed >= 0.75, new road paint the render lacks within 3 px, or, after `consistency.py`, disagreement with neighbouring views), 3 **sky** (no surface, ray above the horizon). `class-single/` keeps the pre-consistency classes |
| `agree/<id>.png` | from `consistency.py`: 255 x clip(1 - excess / 20 levels), where excess is the median over warped neighbours (previous/next station, and default pose <-> 30-degree orbit) of the repaired views' disagreement minus the renders' own; 128 where no neighbour sees the pixel |
| `fixed.json` | settings and per-view statistics (seconds, reference, class fractions) |

## Pseudo-views for the stitch optimiser

`distil.py pseudo` writes the manifest that `pipeline/stitch/v2fit.py --pseudo` already reads: a JSON
array of `{"rgb": fixed PNG, "weight": uint8 PNG, "K": [fx, fy, cx, cy], "T_world_cam": 4x4
camera-to-world (OpenCV axes, ENU world)}` (plus `id`, `t`, `mean_weight`). Weight is 0 on
fixer-only and sky pixels and otherwise the agreement value (0.25 where unchecked), so only content
the fixer reproduces in neighbouring views gets weight. `distil.py fit` runs v2fit unmodified with
`STITCH_V2_WORK` pointed at a symlinked mirror (its checkpoints stay in `work/genfill/distil/`) and
then restores the source scene's ground, sky and map keys in the exported scene.

How to use them in the optimiser: supervise splat colour with `fixed` at a weight well below a
real photograph's, only on classes 0-1; never let class 2 pixels create or move observed splats
(new geometry from them goes to a `...-predicted` layer); ignore class 3 for splats. REPORT.md
gives the measured reasons and the recommended weights.

## Generated chase clips (`clip.py`)

For the viewer's Generative mode. On basebook: `cd viewer && bun tools/chase-track.ts ../scenes/road-stitched-v2 110 125 30 1440 900 > track.json`, copied to `work/genfill/clip/track-110-125.json`. On the Studio:

~~~sh
.venv/bin/python -m pipeline.genfill.clip render scene/stitched-opt-v5 work/genfill/clip/track-110-125.json work/genfill/clip/c110v6 --workers 8   # add --movers for objects.json
envs/genfill/bin/python -m pipeline.genfill.clip fix work/genfill/clip/c110v6 final4 --timestep 199 --ff-in 0.5 --ff-out 0.5
envs/genfill/bin/python -m pipeline.genfill.clip encode work/genfill/clip/c110v6 final4 genfill-chase-110-125 scene/stitched-opt-v5 scene/stitched-opt-v5-movers-v1
envs/genfill/bin/python -m pipeline.genfill.clip sheets work/genfill/clip/c110v6 final4 clip-110-125
envs/genfill/bin/python -m pipeline.genfill.clip flicker work/genfill/clip/c110v6 final4 --real
# then on basebook: cd viewer && bun run cache:studio
~~~

Every stage skips frames whose outputs exist and decode, and writes atomically, so a crash or reboot resumes. `render` first drops, in its own copy of the splats, object splats the photographs contradict (sky behind them, or a vehicle instance; whole smeared columns too; cached in `clean.npz`) and fades splats near the chase camera (gone inside 5.5 m, full at 9 m, nothing between camera and car); tracked vehicles are not drawn (the viewer overlays them; clips say `"vehicles": "overlay"`). `encode --n0 --n1` cuts a long continuous run into clips; `unpublish NAME SCENE...` withdraws one. Targeted repairs: `render --frames n0:n1 --refs keep` (or `--refs n0:n1 --far-bias 3`), `--hand=x,y,r;...` to drop single objects, `--float-drop`; then `fix --n0 --n1 --seed b` into a new tag (carry continues from the published frame), `splice OUT b NEWTAG n0 n1` (0.5 s cross-fades, originals kept in `b-orig/`), and re-encode only the affected clips. `fix --haze-beyond M` fills holes out to M metres (plain colour plus fine grain beyond 40 m). `render` draws sky, ground splats, the ground atlas (with its observed/predicted label), object splats and optional tracked vehicles on a 1472x960 canvas (the 1440x900 track's focal length; Difix needs multiples of 64), and holds a reference photo (later photos only, at most every 2 s, 1.25x better). `fix` repairs in order with feed-forward of the previous frame through the render's depth, fills holes and atlas-predicted ground, keeps the observed atlas road exactly, and writes `<tag>-prov/` (0 observed, 1 repaired, 2 filled, 3 sky or haze) plus `stats.jsonl`. `encode` stacks the provenance codes under the picture (255 observed, 170 repaired, 85 filled, 0 none) and writes `generated.json` entries with `"provenance": "below"` (viewer/README.md).
