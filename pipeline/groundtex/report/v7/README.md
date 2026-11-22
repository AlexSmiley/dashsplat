# Standstill paint check — no atlas change

**Changed tiles: none. Affected time ranges: none.** The three transverse stripes at Burnt Mill Road are photographed road markings: one broad stop line, followed by the two narrower crosswalk boundaries at the kerb ramp. Reducing them to one would remove real paint.

**Decisive evidence:** [left and right pillar photographs](199-stationary-pillars.jpg) taken at 198.99/199.00 s each capture all three at once. These are crops of the native JPEGs: no atlas lookup, projection, warping or compositing. [Two approach front photographs versus the atlas](199-atlas-vs-photos.jpg) independently agree. The stopped **front** camera sees just the farthest stripe; the other two fall below its field of view. It is not a suitable single source for the entire stopped footprint. [Published default and Stitched chase, both at 199 s](199-published.jpg) reproduce the recording sheet in both modes.

At the ego-lane centre, the decoded atlas has one **0.58 m** stripe centred 4.32 m ahead, then **0.30 m** and **0.29 m** stripes at 5.93 m and 7.92 m. A single front exposure (6919 at 193.32 s) yields widths of 0.58, 0.32 and 0.28 m. These estimates use a 21 cm centre strip at 1 cm sample pitch, median luminance across the strip and half contrast between local asphalt and paint. The raw-photo projection uses the rig calibration only; its positions need not match registered atlas coordinates exactly. [Transverse profiles](bar-profiles.json).

The footprint spans tiles `-139_-21`, `-139_-20`, `-139_-19`, `-138_-20` and `-138_-19`. [Source ownership](199-owners.json) shows most bright samples come from approach photographs, with a few departure samples. Ownership boundaries are visible at native scale, but all three stripes occur in individual unmodified photographs. The [lane-detail crops](lane-checks.jpg) distinguish short reflector and arrow features from a second continuous stripe. The two longitudinal boundaries of the bike lane are also present in the native photographs.

## Other slow locations

All six contiguous speed <1 m/s intervals were identified from the published 36 Hz drive and inspected by location, together with the requested car park exit. [Other-stop comparisons](other-stops.jpg) show the atlas and a single front photograph. A dark region in a photo panel means it was outside that camera's view, not an extra texture sample.

| Drive interval (s) | Finding |
|---|---|
| 0.00–0.17; 3.33–83.66 | Initial crawl and parking standstill. No repeated stop bar in the driven lane. |
| 85–96, requested exit | Exit has one transverse bar; the two adjacent yellow centre lines are present in the front images. |
| 196.18–203.15, checked within the 197–206 s junction sequence | One stop line and two crosswalk boundaries. |
| 231.15–232.23 | Three transverse stripes in the ego lane in both the atlas and front exposure 8173. |
| 254.95–287.26, including the overpass stop | One stop bar in both atlas and front exposure 8987. |
| 337.89–339.67 | One broad bar and two narrow ones in both atlas and front exposure 11995. |

**Publication:** the verified publisher reported `Already published 3212` for both v5 scenes. The Studio cache was rebuilt. All 3,212 textures, 3,212 masks and the manifest are byte-identical to the verified v6 payload and hard-linked between both scene roots. Both HTTP manifest hashes match commit `5350095`. [Verification receipt](verified.json). There is no replacement or synthetic “after” image: the published crops show the unchanged result.
