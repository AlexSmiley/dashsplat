# Night pass report (viewer)

All stills are from the published build (http://10.0.50.51:8090/, scene `stitched-opt-v5-movers-v1`) unless noted.

- `car-before-after.jpg`: the ego car before this pass, after it (splat view and over the generated clip), beside
  Tesla's V15 promo frames and the real Model 3 that passes at 287 s in this drive's left repeater camera (same
  morning light); then the car in an orbit view, at night with its tail and brake light on the road, and a mover's
  new contact shadow.
- `grade-match.jpg`: the splat view (tuned Natural grade, cloud deck) beside the generated clip at 112 and 122 s
  (local dev build, HUD hidden). Region means land within about 0.03 (road, verges, trees).
- `predicted.jpg`: the "Predicted" overlay on a test clip whose mask has observed / repaired / filled bands
  (local), and on the textured ground's predicted texels (outer ring, top view, published).
- `phone-budget.jpg`: phone emulation (844 x 390 @3) at 121 s before and after leaving ground splats under the
  opaque atlas out of the budget (local).
- `stills/`: 112 and 118 s open on the generated clip (Generative mode is the default where a clip covers the start);
  `112-splat` is the splat view at the same time; 128, 300 and 322 s are splat views.

Played passes on the published build (1600 x 1000 @2): 110-132 s from the default page (clip, then the hand-over
at 125 s) 60 fps, p99 21.5-23 ms; phone emulation 60 fps; no blank frames in 110-128 s (Generative) or 118-136 s
(splats), checked by reading every presented frame back.

Clip sync was measured with a test clip carrying its frame number as an 11-bit code along the top (made with
`ffmpeg -f lavfi -i color=c=gray:s=800x1000:r=30:d=22 -vf "format=yuv420p,geq=lum='if(lt(Y,500), if(lt(Y,60),
if(between(X,32,735), 255*mod(floor(N/pow(2,floor((X-32)/64))),2), 128), 90+40*sin(X/50+N/10)),
if(lt(X,266),255,if(lt(X,533),170,85)))':cb=128:cr=128" -c:v libx264 -crf 16 -g 30 sync.mp4`, tried with
`?generated=`): paused and scrubbing exact, 1x and 2x the right frame or its neighbour, 4x within one frame.

## Second night pass (clean vehicles, automatic Generative, one grade)

- `vehicles-sheet.jpg`: clean vehicles at the stop (287 s), the coupe pulling away (290.5 s), traffic ahead and
  oncoming (312.5 s), the junction turn (319.5 s), beside Tesla promo traffic frames; the scanned models they
  replace; every kind side by side (local dev build); vehicles over a clip with splat-depth occlusion (local
  stand-in clip: the published scene has no vehicle-free clip yet).
- `grade-one.jpg`: the clip before (camera teal) and after the shared grade, the splat view with the same grade,
  a Tesla frame. `hud-legibility.jpg`: the HUD over the generated sky before and after.
- `played-110-126-sheet.jpg`, `played-286-300-sheet.jpg`: one frame a second from canvas recordings of the
  published page playing (`scratch/night2/pub-110-126.mp4`, `pub-286-300.mp4`, not committed): into the clip at
  110 s by itself, the one-second hand-over at 124-125 s; the clean vehicles at the stop and pulling away.
- `stills2/`: published stills at 112, 118 (Generative), 198, 203 (cross traffic), 287.3, 290.5, 312.5, 319.5, 322.

Published checks (1600 x 1000 @2): no blank frames 110-127 s or 285.6-300.6 s; p99 21.4 ms over 286-300 s and
27 ms over 110-127 s (the hand-over draws both). Automatic mode: Generative from 110.05 s, 0.25 s in, 1 s out
ending at 125.0 s. Adjacent clips (tested with a stand-in cut in two at 293 s): no fade, within a frame of sync.
