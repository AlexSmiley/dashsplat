# Environment map experiment, 300-316 s

Variant scene: [stitched-opt-v5-movers-v1-env1, Generative at 300 s](http://10.0.50.51:8090/?scene=stitched-opt-v5-movers-v1-env1&mode=generative&t=300). The base scene is untouched: every file is hard-linked except `map.json`, `generated.json` and `rsb/index.json`, which are copies. The variant adds `environment/` and two re-encoded clips, 296-311 and 311-326. Its `generated.json` lists the same clips as the base scene.

## What was built

| Step | Result |
|---|---|
| Observed map (`build.py`) | Six cameras, photometry `stitched-ground-v2`, median over **306.5-309.5 s** (every 3rd frame, 36 per camera). Sky masks eroded by 3 px at mask resolution, minus movers, the body mask, dark thin strokes per frame and 580 px of strokes in the result. No taps off the frame edge. Residual gains come from overlaps; the front camera overlaps no other camera, so it is matched row by row to the pillars across the gaps. 4096x2048, coverage in `coverage.png`. |
| Why 3 s | Phase correlation of the cloud texture between windows: 0.27-0.32 for windows 6 s apart (shift 0.2-0.3 deg), 0.01 for windows 12 s apart. Mean change from 302 to 314 s: 8.5 levels. A longer window would average different skies. |
| Reach (`reach.py`) | The chase camera travels 300 m over the stretch. Content frozen at 1 km would slide up to 95 px (median 56), at 10 km up to 9 px. Threshold 1 px. **No far-horizon band**; only sky is in the map. |
| Fill (`fill.py`) | Kandinsky 2.2 decoder-inpaint, **Apache-2.0**, conditioned on CLIP image embeddings of real sky patches. 8 ring tiles plus 1 zenith tile, 768 px, differential diffusion (holes 0.95, real pixels next to holes 0.3 tapering to 0 over 12 px), fixed seed, about 7 s per tile, 100 s per map. |
| Viewer | `sky.ts`: an `environment` in map.json is shown whole and sharp, without the cloud deck, and its horizon is the haze colour. `environment.ts`: the car reflects it fully. `main.ts` and `tiles.ts`: one line each. Deployed build = the live commit c9cf385 plus this patch. For the old scene, the orbit view at 308 s is pixel-identical; the chase view at 308 s differs in 0.05 % of pixels (car reflections from live frame capture, far points). |
| Clips (`clip.py --env`) | The render shows the map behind the splats. The fixer's output decides the matte: its sky is the map plus a slow offset, and anything 10+ levels off that is kept as structure. At edges only the sky part of a pixel is swapped (no halo). Spliced 300-316 s with 0.5 s cross-fades and encoded into the variant only. |

## Real vs filled

- Map, upper hemisphere: 11.9 % observed, 10.3 % observed but harmonised next to a fill, 77.8 % filled. The cameras never see above about 24 deg.
- Map, 0-13 deg (what the chase view can show): 25 % observed, 23 % harmonised, 51 % filled.
- Clip frames, 300.5-315.5 s, share of the frame: observed sky 4.9 %, harmonised 7.4 %, filled 10.3 %, fixer-drawn structure in sky areas 1.2 %. In the published clip, 22.0 % of the frame is sky the fixer made up.

## Model choice (`bakeoff.jpg`)

Held-out real sky, 2 tiles: the top 11 deg of the observed band and a 4 deg join removed. Pairs are tile 1 / tile 2.

| | Colour error (levels) | Mid-scale structure vs real | Seam vs real | Time per tile |
|---|---|---|---|---|
| Push-pull only | 7.1 / 13.7 | 0.32 / 0.15 | 1.1 / 1.3 | 0 s |
| Kandinsky 2.2 (Apache-2.0) | 6.8 / 13.3 | 0.42 / 0.38 | 2.0 / 1.6 | 8-9 s |
| SDXL inpaint 0.1 (OpenRAIL++-M) | 7.4 / 13.6 | 0.67 / 0.35 | 1.8 / 2.1 | 10-13 s |

Neither model rebuilds the real cloud structure, and neither beats push-pull on colour. Kandinsky was picked as the permissive one; on quality the two are effectively tied. Starting from grey holes, both only painted a flat wash of an invented colour. Unguarded, Kandinsky wrote text, drew dotted wires and left specks and blobs. The guards (pre-fill, a cap on detail in holes, a stroke filter, a colour lock) remove most of this.

## Measurements: frame-to-frame change after warping by pose and depth, 300.5-315.5 s

| | Sky (21.8 % of frame) | Crown edges (1.7 %) |
|---|---|---|
| Published | 1.81 levels (p95 7.1) | 10.0 (p95 15.4) |
| Environment map | **0.14** (p95 0.33) | **4.8** (p95 6.9) |

## How it looks (`old-vs-env1.mp4`, `viewer-stills.jpg`, `clip-*.jpg`)

- **Better:** real dark clouds in the forward view instead of a flat cyan sky. The sky no longer crawls. Crown edges are clean, with no pink or pale rim, and the fixer's ghost vertical strokes in open sky are gone.
- **Wrong:**
  - The sky is a patchwork: the crisp real band ends in hard edges against a soft hazy fill. In the forward view the two camera joins show as smooth vertical strips, and the top of the front camera's view (about 15 deg) is just inside the frame.
  - Looking up (orbit, reflections), the zenith is a plain haze with faint tile arcs and one dark blob.
  - In the splat view, the crowns' pale fringes read as halos against the darker real sky.
  - Fixer-drawn pole strokes stronger than 10 levels survive.
  - The sky switches at 300 s and 316 s, where the regenerated range is spliced in.
  - The clouds stand still. Within 16 s that is hard to see; across stretches it would not hold.
- Strokes left in the map: 0.18 % of observed pixels, 1.9 % of harmonised pixels (crown fringes next to masked holes), 0.15 % of filled pixels.

## Verdict

Don't roll it out to the other stretches. The observed part survives contact with the footage: the cameras give a stable, tone-matched real sky up to about 24 deg, and putting it back after the fixer halves the edge flicker and removes the fixer's sky. The fill does not: the cameras see under a quarter of the sky, and neither inpainting model continues real cloud texture, so the joins and the zenith read as haze patches. The forward chase view shows them.

## Notes

- All heavy work ran on the Studio under `work/locks/reconstruction.lock`. One wrapper waiting on that lock had `pipeline.genfill.clip fix` in its command line. The whole-drive runner pauses its clip worker for any such process, so the two deadlocked for about 1 h until I killed my waiter. Later wrappers run a script file instead.
- Modified `pipeline/genfill` ran from a copy (`work/envmap/code`). The Studio's live `pipeline/genfill`, which the runner versions as its clip producer, is unchanged.
- The cache tool ran on the variant only, so the runner's live release was not touched.
- Files on the Studio: `work/envmap/300-316/` (map, provenance, coverage, build, fill, reach, flicker JSON), `work/genfill/clip/c296-env1/` (render, `e1` with `e1-raw`, the fixer's frames before the map went back in), `work/envmap/rec/` (viewer recordings).
