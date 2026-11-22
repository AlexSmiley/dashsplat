# Live world v5 · 85–140 s

**Whole-drive gate: fail.** The streetlight slab is gone at 112 s, including from both displaced views. The 128 s side view still turns detached-looking groups of foliage into upright rectangular panels. Their lower edges are most conspicuous where road texture gives way to the apron. Soft trees would be acceptable; this reads as the wrong shape and placement. The slice is published for inspection. No whole-drive job was started.

[Published slice at 112 s (Studio LAN)](http://10.0.50.51:8090/?scene=live-v5&t=112&paused=1&cam=chase&colour=natural) · [108–118 s chase](report/v5/chase-v5-108-118.mp4) · [movie contact sheet](report/v5/sequence.jpg)

## V4 / v5 at the controller pose

Each link shows v4 on the left, v5 on the right, each at **1440 × 900**. Signed angles are the offsets passed to `setOrbit`. The software proof draws the published PLY geometry, scene panorama, Natural grade, distance haze and apron. It omits the car, route ribbon, HUD, asynchronous tile uploads and cache quantisation. It is a reconstruction of the viewer camera, not a screen recording.

| Time | Orbit −60° | Chase | Orbit +60° |
|---:|---|---|---|
| 93 s | [−60°](report/v5/93--60.jpg) | [straight](report/v5/93-+0.jpg) | [+60°](report/v5/93-+60.jpg) |
| 100 s | [−60°](report/v5/100--60.jpg) | [straight](report/v5/100-+0.jpg) | [+60°](report/v5/100-+60.jpg) |
| 112 s | [−60°](report/v5/112--60.jpg) | [straight](report/v5/112-+0.jpg) | [+60°](report/v5/112-+60.jpg) |
| 128 s | [−60°](report/v5/128--60.jpg) | [straight](report/v5/128-+0.jpg) | [+60°](report/v5/128-+60.jpg) |
| 135 s | [−60°](report/v5/135--60.jpg) | [straight](report/v5/135-+0.jpg) | [+60°](report/v5/135-+60.jpg) |

At 93 s the store frontage and island survive the change, but the orbit passes very close to softened vegetation. At 100 s the sky is clear and the old purple canopy streaks are shorter. At 112 s the straight-ahead overhead window (x 485–1084, y 85–174) goes from **72.4% to 0%** of pixels with composited object contribution above 0.35. At 128 s the right roadside shows the flat-patch failure; at 135 s the missing signal arms leave the intersection more open. [Elevated 112 s orbit](report/v5/112-high.jpg).

The chase uses the damped drive position (0.05 s), 0.3 s yaw look-ahead damped over 0.28 s, and speed damped over 1.2 s, replayed at 30 Hz. Its forward/up pivot is 1.45/0.8 m; distance is 19.5 m multiplied by the controller's speed pull-back. Elevation is 15°, followed by local +7.2° tilt and +1.1° yaw. At 1440 × 900 the 50° vertical FOV wins over the 64° horizontal minimum.

The 300-frame chase confirms the overhead road stays clear through the 112 s regression. Around 114–116 s the roadside crowns thin out abruptly as profiles retire; a bright, empty horizon replaces some of them. The lane markings remain visible through the transition.

## What was built

1. **Ground support.** Each patch starts at a cluster of at least four non-sky depth samples near an already-released ground height or bounded local road plane. Height/range bins have to connect back to the bottom; a canopy can connect sideways to its trunk. The members share a lifetime so the trunk cannot retire independently. This geometric constraint does not certify the position of a root against another camera.
2. **Shape and colour.** More erosion is applied to higher, more distant sky boundaries. Near-ground fragments flatten; walls face the observation point; foliage gets near-round footprints and interior-weighted colour; thin poles are radius-capped and unsupported high branches or distant masts are dropped. Far profiles are upright, have lower opacity, and receive the viewer's range haze. 1,469 causal packets produced **1,759,192** object records, of which **1,652,984** have finite lifetimes. There is no semantic classifier.
3. **Road and sky.** Ground is fully weighted to 9.5 m laterally, tapers to zero at 13.5 m, and leaves the rest to the viewer's apron. The export retains 26,093,605 of 28,012,974 v2 ground records across the drive. Outside the requested slice this is road, not a complete object world. The 1024 × 512 panorama uses 127,311 interior sky measurements, filtered by direction and compressed around the upper sky exposure; it is bright, smooth, and slightly warm under the v5 grade. The existing viewer uses this same panorama for car lighting. The offline proof does not measure the rendered car paint.
4. **Time-aware loading.** Short finite generations are grouped by 1.5 s death windows; longer finite generations by birth and death; open-ended memory by birth. Only open-ended chunks use the viewer's static selection rule. All 1,411 PLY extents, lifetimes and bounds were checked, along with all 7,065 instance row references. All 1,411 published RSB headers, counts and byte sizes agree with the manifest: 919,500,724 bytes cached.

## Tile-selection replay

1× playback, 200 ms planning cadence from 84 s, 172 m forward-biased radius, viewer prefetch and retain margins. **Eligible** is the count in wanted or retained tiles, including unborn records. It is a manifest estimate before runtime scheduling, budget eviction and network delay, not a GPU-residency measurement.

| Time | v4 eligible | v5 eligible | v5 alive in eligible tiles | v4 dead >3 s | v5 dead >3 s |
|---:|---:|---:|---:|---:|---:|
| 93 | 4,820,165 | 1,161,890 | 383,137 | 992,208 | 0 |
| 100 | 5,933,983 | 1,273,153 | 507,272 | 1,936,827 | 39,946 |
| 112 | 7,950,379 | 2,117,765 | 1,034,095 | 3,951,847 | 43,576 |
| 128 | 5,960,792 | 2,725,637 | 1,241,295 | 1,063,512 | 37,292 |
| 135 | 5,323,799 | 2,420,379 | 1,071,803 | 1,352,992 | 0 |

Published on Studio: `scene/live-v5/` under `$RS`. The instance references, build totals and pose matrices are in `$RS/maps/live-v5/slice-{instances,summary,proof}.json`.
