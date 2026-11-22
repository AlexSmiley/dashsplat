# I rebuilt Tesla's Reality View from my dashcam: outline

Talking points for Ryan to say in his own words on camera: no written lines. Thirteen beats in the order of the
journey film's chapters. Each beat has the point to land, the facts worth hitting and what is on screen. Every
number is from `film/notes/facts.md`; every post is linked in the table at the end. The outline covers the
techniques only and leaves out how the code was written.

Recording: one beat at a time, two seconds of silence between beats, say a line again if it goes wrong. The cut is
made from the recording, so order and length can change afterwards.

## 0. Cold open

- On screen: finished view at the junction, 5 seconds, no talking.

## 1. The hook (on camera)

- Point: Tesla shipped FSD v15, this is not a review, I rebuilt the headline feature myself.
- Facts:
  - Tesla announced v15 on Saturday 21 Nov, 2:24 pm Eastern. Their post is at about 7 million views.
  - The headline feature is Reality View: a realistic live 3D view of the world around the car, from a camera floating behind it, replacing the grey cartoon cars.
  - Everyone is posting first-drive videos (Sawyer Merritt, AI DRIVR, Dirty Tesla, Whole Mars Catalog). Musk's line on it: "It's like GTA".
  - I don't have it: so far it is a small early-access group, and the cars logged with it are all Model Ys. I drive a Model 3. Wide release is supposed to be December.
  - 3:22 pm, my post: "i wonder if I can recreate this....." The car already has the cameras and already records everything.
- Promise to the viewer: how something like this actually works, kept simple. Then show the result from the next morning.
- On screen: Tesla's post, 2 to 3 seconds each of other people's videos, my 3:22 pm post, 5 seconds of the finished view.

## 2. What you start with (over film chapter 1)

- Point: the whole input is six videos and some numbers.
- Facts:
  - A saved dashcam clip has six cameras: front, back, two on each side.
  - One drive: 6 minutes, about 2 miles (3.3 km), Jacksonville, dawn, overcast.
  - Every frame also carries speed, steering, heading and GPS, 36 times a second.
  - No lidar, no radar.
- On screen: the six cameras unfolding from the car, the synced grid, the drive map.

## 3. Where exactly are the cameras? (over chapter 2, calibration)

- Point: to merge six cameras into one world you must know exactly where each sits and points. Tesla knows that from the factory; I had to work it out from the footage.
- How, in plain words: find the same thing in two cameras (a sign corner, a crack in the road) and ask where the cameras must be for those to line up. Repeat until it locks in.
- Facts:
  - Result: all six cameras accurate to under one pixel (0.81 to 0.97).
  - Worked out from the footage alone, no factory numbers.
  - It also found the cameras are a few milliseconds out of sync with each other (up to 8.7 ms).
- On screen: my 3:55 pm post "all of the Model 3 cameras have been position mapped", the calibration shot.

## 4. How far away is everything? (over chapter 2, depth and masks)

- Point: a photo is flat; 3D needs a distance for every pixel.
- Facts:
  - Neural networks can guess depth from a single picture.
  - On this drive every one of them got the scale of the world wrong, by 16 to 69 percent. Shapes right, size wrong.
  - Fix: the car knows its own speed, so the distance it drove pins the scale down.
  - Then a labelling pass: sky, road, parked cars, moving cars.
- Tease: the moving cars cause trouble later.
- On screen: photo turning into a depth map and lifting into 3D points, then the mask overlay.

## 5. Splats, and the road first (over chapter 3)

- Point: explain what a splat is, then show the first real result.
- Splat in plain words: a tiny, soft, see-through blob of colour in 3D. One is nothing; millions blend into a picture from any angle. Not triangles like a video game.
- Why the road first: it is flat and you drive right over it.
- Facts:
  - About an hour and a half after the first post there was a drivable road.
  - The road alone is 28 million splats.
  - You can read the paint: the word ONLY passes under the car at the same moment as in the real camera.
- On screen: my 4:33 pm post "we now have a splat of the road", my 4:53 pm post "the ground layer splatting is done" (147 K views), the first demo.

## 6. The rest of the world (over chapters 4 and 5)

- Point: the first 3D world is blobby; a compare-and-correct loop fixes it.
- Facts:
  - First attempt: throw all the depth into 3D. Trees smear, poles float.
  - The loop: render the splats from exactly where a real camera was, compare with the real photo, nudge the splats to shrink the difference, repeat.
  - Each round: trees fill in, grass appears under them, poles stand up straight.
  - About three hours in: a full 3D world to fly around. Later versions reach 18 million splats.
- On screen: the blobby first world, the wipes between rounds, my 6:23 pm post "MASSIVE update", the second demo.

## 7. Experiments along the way (over chapter 6)

- Point: with a 3D world working, I tried three other routes to make it look more real, and kept the best idea from them.
- The three:
  - Building the view straight from cut-outs of the real photos.
  - Letting a video model generate the whole view.
  - Re-projecting the front camera's photo to the new angle.
- What came out of it: the re-projected photos became a teacher for the splats. The splats get graded against them and come out sharper.
- On screen: each experiment as it looked, then the before and after with the teacher.

## 8. The road, properly (over chapter 7)

- Point: lane lines were soft up close, so the road was rebuilt as one giant flat photograph.
- Facts:
  - Stitched from the best view of every patch of asphalt, at one pixel per centimetre.
  - Lane-line edge: 4.5 cm of blur before, 1.2 cm after, 1.0 cm in the original photo. About as sharp as the camera allows.
  - The whole 2 miles of road is 57 megabytes.
  - 9:19 pm: the power went out. Everything resumed where it left off.
- On screen: the atlas along the whole drive, the zoom to 1 cm, the lane-dash comparison, the picture dropping out and returning.

## 9. Other cars (over chapter 8)

- Point: anything that moves smears when you build a still world from video.
- Facts:
  - Moving cars are cut out completely: not one pixel of the road comes from a vehicle.
  - Each car is tracked on its own (position and heading).
  - Each is replaced by a clean 3D model that drives the same path.
- On screen: the top-down junction with the tracks, scanned models wiping to clean ones, my 6:27 am post "INSANE overnight update".

## 10. The last ten percent (over chapter 9)

- Point: the 3D world is accurate; an image model adds the photo-real finish.
- Facts:
  - Tesla's own release notes call Reality View "generative". This is the generative step.
  - In: the 3D render of a frame plus the closest real photograph. Out: same layout, real-looking detail.
  - About 3 seconds per frame.
  - Every pixel is traceable: seen by a camera, repaired, or filled in. The road actually driven stays exactly as measured.
- On screen: render in, photo inset, the wipe to the redrawn frame, the coloured map.

## 11. The result (first line on camera, then chapter 10)

- Point: here it is, next to the six cameras it came from.
- Let it play 8 to 10 seconds without talking.
- Detail worth one line: the camera angle is matched to Tesla's, about 50 feet back, looking down.

## 12. How Tesla does it, and where this goes (on camera, chapter 11 as cutaway)

- Point: what I learned about how Tesla's version works, and what mine can do now.
- Facts:
  - Mine is now one command: point it at a new drive and it builds the scene, with no manual steps.
  - Tesla's runs live in the car. A year ago they showed splats like these being generated in about a fifth of a second.
  - Their head of AI software says the same network that drives the car also draws this view.
- Land on: one model that drives the car and draws the world. Having built one from scratch, that is the exciting part.

## 13. Close (on camera)

- From the first post to this: one afternoon and one night (3:22 pm to about 6:30 am).
- When v15 reaches my car: a side-by-side. That is the sequel.
- Ask for the subscribe.
- On screen: my post "@TeslaSI how did i do?", then the last shot into the distance.

## Posts used on screen (times are Eastern)

| When | Post | Beat |
|---|---|---|
| 21 Nov, 2:24 pm | Tesla: "FSD v15 now starting to roll out" (video, 6.9 M views) https://x.com/Tesla/status/2124222447855743019 | 1 |
| 21 Nov, 2:37 pm | You, replying to Tesla: "ho lee shit" https://x.com/ryanvogel/status/2124225798194290836 | 1, optional |
| 21 Nov, 3:19 pm | DirtyTesLa: "Tesla's new Reality View / Generates the world around you, viewable from any angle you please" (4 photos) https://x.com/DirtyTesLa/status/2124236337096040778 | 1 |
| 21 Nov, 3:22 pm | "i wonder if I can recreate this....." quoting that post https://x.com/ryanvogel/status/2124237027428430321 | 1 |
| 21 Nov, 3:55 pm | "all of the Model 3 cameras have been position mapped" https://x.com/ryanvogel/status/2124245254731927843 | 3 |
| 21 Nov, 4:33 pm | "we now have a splat of the road" https://x.com/ryanvogel/status/2124254845960097888 | 5 |
| 21 Nov, 4:53 pm | "the ground layer splatting is done" (video, 147 K views) https://x.com/ryanvogel/status/2124259810711335049 | 5 |
| 21 Nov, 6:23 pm | "MASSIVE update": higher resolution ground and 3D static objects (video) https://x.com/ryanvogel/status/2124282699510857746 | 6 |
| 22 Nov, 6:27 am | "INSANE overnight update": moving cars and buses, "@TeslaSI how did i do?" (video) https://x.com/ryanvogel/status/2124464850550292638 | 9, 13 |
| 22 Nov, 7:00 am | "so @elonmusk how did i do recreating tesla reality view?" https://x.com/ryanvogel/status/2124473050393727033 | 13, optional |

Other sources for points in the outline: Tesla's release notes (https://x.com/TeslaSI/status/2124223658680061958,
"real-time generative driving visualization"), Ashok Elluswamy on Reality View
(https://x.com/aelluswamy/status/2124229978560835880, "The same end-to-end driving model also renders a realistic
view of the world around the vehicle") and his October 2025 article with the ~220 ms figure for Tesla's Generative
Gaussian Splatting (https://x.com/aelluswamy/status/1981644831790379245). Tesla has not said that Reality View is
built on that splatting work; the outline only uses the two things Tesla did say.

## Other people's v15 videos for beat 1

Collected by a research helper on 22 Nov; view counts are snapshots from that day. Use two or three seconds of
each with the creator's name on screen. No timestamps for the Reality View screen were found, so each clip needs
scrubbing by eye.

| Use | Creator | Title | Link | Views then |
|---|---|---|---|---|
| first | Tesla | Introducing Tesla Self-Driving V15 (0:35, the official promo) | https://www.youtube.com/watch?v=eNBqkc7V1nU | 707 K |
| first | Sawyer Merritt (X) | "Here are the new FSD visualizations in FSD V15…" (the clip Musk quoted with "It's like GTA") | https://x.com/SawyerMerritt/status/2124226258196103492 | 558 K |
| first | AI DRIVR | Tesla FSD V15: Reality View is INSANE (8:09) | https://www.youtube.com/watch?v=iAqFI3cnBJY | 57 K to 131 K |
| first | Sawyer Merritt | First drive on Tesla FSD V15 (new visualizations are mind-blowing!) (25:15, Model Y) | https://www.youtube.com/watch?v=TSpBfh7Ubw8 | about 100 K |
| spare | Dirty Tesla | Tesla FSD V15 Update First Drive and Initial Impressions (30:35) | https://www.youtube.com/watch?v=5rDGMNjDPIA | 169 K |
| spare | Whole Mars Catalog | Tesla Self-Driving 15 First Ride with my Dad (38:07) | https://www.youtube.com/watch?v=qHg4Fjh4Jww | 132 K |
| spare | Mar Maldonado | Tesla FSD v15.0.1 First Drive: Reality View, Grok Reroute, and Self-Parking | https://www.youtube.com/watch?v=K-1zLqfzyo8 | 33 K |
| spare | TruthTom (X) | Reality View renders a large branch in the road and the car goes around it | https://x.com/TruthTom/status/2124320271712624845 | 4 K |

## Who has v15 (behind the point in beat 1)

- Tesla and Ashok Elluswamy both called it a first wave and named no cars (https://x.com/aelluswamy/status/2124224924512641512).
  Press reports say it went to Early Access members, as software 2026.38.6.11 with FSD v15.0.1.
- The cars logged with it so far are Model Ys with HW4; trackers had not seen it on a Model 3 by 22 Nov
  (https://teslant.com/en/changelog/tesla-fsd-v15-hw4-reality-view-first-wave-2026-11).
- Nothing says a Model 3 with HW4 is excluded, so the outline does not say "because I have a Model 3". Musk said
  wide release is "next month", meaning December.
