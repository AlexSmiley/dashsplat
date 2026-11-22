# M2 · Same-instant pillar chase, 110–125 s

**Review candidate.** The original street is recognizable in the center. The complete frame does not hold together as one photograph in motion. The break is inside the view, between the front and pillar footprints, not just at the outermost edges. See the [paint comparison](paint-112.jpg) and [50°/44° pair](fov-112.jpg).

## Release verdict

**Withheld.** On all 451 decoded frames, suspected twin strong edges fell from **1.47% to 1.44%** overall and from **1.45% to 1.25%** in the structure band. The prior hard-cut candidate was 1.78% / 1.70%. The matched M2 solo control was **0.143% / 0.158%**, so M2 is roughly ten times the whole-frame floor; 426/451 frames exceeded twice their own solo control. Worst whole frame: **4.70% at 115.367 s**. Of 46,229 flagged pairs, 26,336 crossed an ownership boundary. These are detector suspicions, not 46,229 independent objects. [Per-second audit](per-second.csv) · [receipt and encoded SHA-256](result.json) · [worst structural crops](audit-crops.jpg).

In the [actual published viewer](http://10.0.50.51:8090/?scene=/scenes/stitched-opt-v3/generated/m2-review/&t=110&mode=generative&paused=1), the ego, ribbon and HUD composited over the exact candidate. [Playback recording](viewer-play.mp4) · [motion sheet](viewer-motion.jpg) · [near-mask comparison](near-right.jpg). At 110 s the median pole folds diagonally across the road. At 112 s the central office and utility pole are recognizable, but wires step across the left seam; the right sign, hedge and trunk stretch below their contact. At 115.37 s wires break at the upper-left cut. At 117.5 s the right-hand RGB seam visibly splits the office facade even though that frame's whole-image number is low. The underlay is visible through the trees/verge approaching 125 s. **No extension to the turn or the next 15 s.**

[Pillar timing](same-instant.json): P95 effective time separation from the selected front was 1.75/2.28 ms left/right; at a missing exposure the nearest sample was 57/58 ms away. Across the clip, pillars owned 25.1% of pixels, older front bridges 4.5%, underlay without confident stitched ground 4.7%. None of 2,572 bridge hypotheses furnished consistent registration tracks on **both** sides. Every selected exposure had the newer mover-removal mask; this does not prove every moving object was detected.

The review delivery is [1440×900 H.264, 30 fps, no baked ego](http://10.0.50.51:8090/scenes/stitched-opt-v3/generated/m2-review/pillar-photo-110-125.mp4). The review manifest lives below the existing scene and uses the same cached tiles and drive. Rebuilt the cache (419/419 tiles current). The default `generated.json` still selects the original 85–140 s clip.

## Source and geometry

The delayed front clock is still the projection of the chase eye onto the physical camera path. Sample the nearest front exposure and both pillars at the same **effective exposure time**, including clock offsets. In the gap only, try front frames behind that instant in nearest-first distance bands. A frame whose source depth indicates an occluder cannot supply a distant building. Outside 36 m, the bridge is eligible only for masked sky above the horizon. This prevents an earlier drive-through wall from appearing in place of the street, at the price of real holes. A frustum hit alone is not visibility.

The existing near-static and background depth fields feed a forward-projected relief grid. Raw per-pixel displacement badly deformed treetops and the roof; the candidate fits separate near (<25 m) and far upright ranges across azimuth, with a conservative near contribution, then samples a single continuous coordinate chart. The nearer measured road plane wins up to its contact with the profile. This preserves a single silhouette within a photograph but underestimates parallax on close trunks and signs, especially the right verge. It does not reconstruct the hidden side of a tree.

Every photographed ground sample uses the tangent plane and the physical rolling-shutter row. Real front/pillar samples win where available, otherwise the measured stitched road supplies the strip. No image flow, RGB feathering or older front frame is used on ground. The [independent per-ray check](ground-lookup.json) compares the lookup to full calibrated projection and three iterations of original-row pose: at 112, 117.5 and 125 s, front P95 was below 0.010 **native** px, pillars below 0.002 px. This checks the implementation of the tangent plane, not the accuracy of the fitted road or its semantic boundary. The 112 s paint comparison shows why they differ: the driven lane is sharp; across the left gap the stitched paint is wider and displaced from the front stripe.

Correct each exposure with stitched-ground-v2 photometry. Fit a residual gain on corresponding low-gradient overlap; if the lens lacks direct overlap, use observed pavement in the front gauge. A smooth sensor-height gain uses contemporaneous masked sky for the disconnected pillar color reference. The correction does not fix all brightness differences at the rim. Binary cuts prefer lower disagreement and lower gradients. Reproject previous lens ownership through the current geometry for temporal hysteresis. Nearest-frame selection avoids temporal double exposures. On non-ground only, a bridge can receive a single translation of at most 2.5 output px when both neighbors supply consistent round-trip feature tracks; otherwise use the unmodified geometric warp. Unknown agreement is never a pass.

## Where the evidence runs out

At 112 s, the same-instant calibrated lenses cannot observe **210 horizontal pixels** at the world horizon (x=90–224 and 1070–1144 at y≈252). On a ray near the center of the right gap, the closest historical front frustum that could cover a hypothetical point 20–80 m away is **58–67 m back along the path**; at infinity it is 76 m back. The equivalent left-gap ray needs only 1–3 m. These are lower bounds from every preceding exposure within 95 m, before testing visibility. [Ray-by-ray data](bridge-frusta.json) · [horizon intervals](blind-angles.json).

Narrowing from 50° to 44° vertical (73.45° to 65.76° horizontal) increases the chase arm by 15.4% to preserve projected car size at the pivot. Recompute the eye, delayed clock, near/far geometry and stitched underlay, rather than cropping the wide photograph. The horizon gap falls from 210 to 145 px at 112 s, but grows from 200 to 235 px at 125 s. Neither one-frame result supports narrowing the viewer camera. Pixel counts are in 1440-wide output; the horizon probe is quantized to 5 pixels.

## Reproduce on the Studio

From `$RS`, with the existing chase tracks, depth, masks, RGB, photometry and stitched scene:

```sh
mkdir -p work/ibr/m2
clang++ -O3 -std=c++17 -dynamiclib pipeline/ibr/m2_surface.cpp -o work/ibr/m2/m2_surface.dylib
export RS="$PWD" VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1
.venv/bin/python -m pipeline.ibr.morph_m2  # wide and narrower probes
.venv/bin/python -m pipeline.ibr.m2_gate   # 451 frames; matched separately encoded solo control
```

Full video, frame-level source numbers, mask provenance and standalone control parts stay in `$RS/work/ibr/m2/110-125/`. The existing [after-image detector](../morph_ghost.py) is unchanged. The concatenated delivery is verified by decoded per-frame hashes against the audited H.264 parts. The old default and interval files are preserved in `$RS/work/ibr/m2/previous/`.
