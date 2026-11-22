//! The running order. Times are seconds from each shot's cut; thread values are minutes after 15:45
//! (the first commit); every number on screen is in film/notes/facts.md with its source.
use crate::Shot;
use crate::draw::*;
use crate::kinds::{Kind, Panel, Step, TitleLine};
#[allow(unused_imports)]
use crate::sound::{Cue, cue};

fn shot(id: &'static str, chapter: usize, dur: f32, kind: Kind) -> Shot {
    Shot { id, chapter, dur, overlap: 0.5, kind, labels: vec![], thread: None, fade_in: 0.0, fade_out: 0.0, sfx: vec![], power: None, mood: None, need: None, unless: None }
}
impl Shot {
    fn ov(mut self, o: f32) -> Self { self.overlap = o; self }
    fn lab(mut self, l: Vec<Label>) -> Self { self.labels = l; self }
    fn th(mut self, a: f32, b: f32) -> Self { self.thread = Some((a, b)); self }
    fn fin(mut self, f: f32) -> Self { self.fade_in = f; self }
    fn fout(mut self, f: f32) -> Self { self.fade_out = f; self }
    fn sfx(mut self, c: Vec<Cue>) -> Self { self.sfx = c; self }
    fn power(mut self, t: f32) -> Self { self.power = Some(t); self }
    fn mood(mut self, m: [f32; 3]) -> Self { self.mood = Some(m); self }
    fn need(mut self, id: &'static str) -> Self { self.need = Some(id); self }
    fn unless(mut self, id: &'static str) -> Self { self.unless = Some(id); self }
}

fn plate(src: Src, size: (f32, f32), crop0: [f32; 4], crop1: [f32; 4]) -> Kind {
    Kind::Plate { src, size, crop0, crop1, start: 0.0 }
}
/// a 1440x900 viewer recording with its HUD cropped away
const VIEW: (f32, f32) = (1440.0, 900.0);
const VIEW_IN: [f32; 4] = [140.0, 100.0, 1160.0, 652.5];
const VIEW_IN2: [f32; 4] = [176.0, 120.0, 1088.0, 612.0];
/// a capture from the camera department's library (film/captures, linked into work/media by ingest_captures.py)
fn cap(id: &str) -> Src {
    Src::V(Box::leak(format!("cap-{id}.mp4").into_boxed_str()))
}
const C4K: (f32, f32) = (3840.0, 2160.0);
/// the hero junction capture (generated picture, palms, traffic; 299-329 s at 60 fps)
const HERO_J: &str = "hero-gen-palms-junction-299-329-60fps";
const F4K: [f32; 4] = [0.0, 0.0, 3840.0, 2160.0];
/// a slow push of about 4 % on a 4K capture
const P4K: [f32; 4] = [77.0, 43.0, 3686.0, 2074.0];

/// short on-screen names for the viewer scenes the ladder captures show, with the commit time each appeared
fn rung_label(scene: &str) -> String {
    let known = [
        ("road-live-v1", "GROUND V1 · 16:25"), ("road-live-v2", "GROUND V2 · 16:38"), ("road-stitched-v2", "STITCHED GROUND V2 · 16:38"),
        ("live-v3", "FIRST OBJECTS · 17:02"), ("live-v4", "LIVE WORLD V4 · 17:20"), ("live-v5", "LIVE WORLD V5 · 17:35"),
        ("stitched-opt-v1", "OPTIMISER V1 · 17:06"), ("stitched-opt-v2", "OPTIMISER V2 · 18:02"), ("stitched-opt-v3", "OPTIMISER V3 · 18:14"),
        ("ibr-v1", "PHOTOGRAPHIC CUT-OUTS · 18:07"), ("stitched-opt-v4", "TEACHER V4 · 19:46"), ("stitched-opt-v5", "STANDING WORLD V5 · 20:01"),
        ("stitched-opt-v5-movers-v1", "V5 + VEHICLES · 20:11"),
    ];
    known.iter().find(|(k, _)| *k == scene).map(|(_, v)| v.to_string()).unwrap_or_else(|| scene.to_uppercase().replace('-', " "))
}

#[derive(serde::Deserialize)]
struct CapInfo {
    id: String,
    scene: Option<String>,
    hud: Option<bool>,
    t: Option<Vec<f32>>,
    description: Option<String>,
}

/// the library summary written by viz/ingest_captures.py (empty when there is none yet)
fn library() -> Vec<CapInfo> {
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/../work/data/captures.json");
    std::fs::read_to_string(p).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default()
}

/// The evolution: every ladder rung at 110-122 s, registered, one after another through quick wipes.
fn evolution() -> Option<Shot> {
    let lib = library();
    let mut rungs: Vec<(u32, String, String)> = Vec::new();
    for (k, c) in lib.iter().enumerate().filter(|(_, c)| c.id.starts_with("ladder-") && c.hud != Some(true) && c.t.as_ref().is_some_and(|t| t[0] < 200.0)) {
        // "Rung N: ..." in the description gives the order; otherwise the library order
        let n: u32 = c.description.as_deref().and_then(|d| d.strip_prefix("Rung ")).and_then(|d| d.split(|ch: char| !ch.is_ascii_digit()).next())
            .and_then(|x| x.parse().ok()).map(|x: u32| x * 1000).unwrap_or(1_000_000 + k as u32);
        if !crate::has_capture(&c.id) { continue; }
        let scene = c.scene.clone().unwrap_or_default();
        // one shot per scene; prefer the version at the matched chase pose (no "-auth") so the wipes register
        if let Some(r) = rungs.iter_mut().find(|r| r.2 == scene) {
            if r.1.ends_with("-auth") && !c.id.ends_with("-auth") { r.1 = c.id.clone(); }
            continue;
        }
        rungs.push((n + k as u32, c.id.clone(), scene));
    }
    if rungs.len() < 3 { return None; }
    rungs.sort_by_key(|r| r.0);
    let step_t = 1.25;
    let steps: Vec<Step> = rungs.iter().enumerate()
        .map(|(i, r)| Step { src: cap(&r.1), size: C4K, crop: P4K, at: i as f32 * step_t, start: 0.0 })
        .collect();
    let dur = (rungs.len() as f32 * step_t + 1.2).min(11.8);
    let labels: Vec<Label> = rungs.iter().enumerate()
        .map(|(i, r)| lab(i as f32 * step_t + 0.25, (i as f32 + 1.0) * step_t + 0.2, MX, H - 250.0, &rung_label(&r.2)).size(40.0))
        .collect();
    let sfx: Vec<Cue> = (1..rungs.len()).map(|i| cue(i as f32 * step_t - 0.1, "sweep.wav", -19.0)).collect();
    Some(shot("Evolution", 10, dur, Kind::Ladder { steps, wipe: 0.55 }).th(53.0, 482.0)
        .lab(with(chap("10", "EVOLUTION · 110–122 S"), labels)).sfx(sfx))
}

/// the drive runner's stages in order (pipeline/run/README.md, stage inventory)
const STAGES: [&str; 14] = ["INGEST", "MOTION", "RIG CHECK", "POSES", "GROUND", "APPEARANCE", "ATLAS", "MASKS", "MOVERS",
    "DEPTH", "TEACHER", "OPTIMISER", "CHASE CLIP", "PUBLISH"];
/// the large 16:10 panel used by the image-model chapter
const BIG: [f32; 4] = [420.0, 250.0, 3000.0, 1500.0];

fn step(src: Src, size: (f32, f32), crop: [f32; 4], at: f32) -> Step {
    Step { src, size, crop, at, start: at - 0.2 }
}
fn panel(src: Src, size: (f32, f32), crop: [f32; 4], dest: [f32; 4], t0: f32, t1: f32) -> Panel {
    Panel { src, size, crop0: crop, crop1: crop, dest, t0, t1, start: t0 - 0.3, frame: true, alpha: 1.0 }
}

/// chapter titles (small, top-left, first seconds of a chapter's first shot)
fn chap(n: &str, name: &str) -> Vec<Label> {
    vec![lab(0.3, 3.2, MX, MY, &format!("{n}  {name}")).size(44.0).dim()]
}
fn with(mut a: Vec<Label>, b: Vec<Label>) -> Vec<Label> {
    a.extend(b);
    a
}

/// the first capture in work/media whose id starts with `prefix` (derived cuts, with '@', only when asked for)
fn capid(prefix: &str) -> Option<&'static str> {
    let mut ids: Vec<String> = std::fs::read_dir(crate::MEDIA_DIR).ok()?
        .filter_map(|e| e.ok()).map(|e| e.file_name().to_string_lossy().to_string())
        .filter(|n| n.starts_with("cap-") && n.ends_with(".mp4"))
        .map(|n| n[4..n.len() - 4].to_string())
        .filter(|id| id.starts_with(prefix) && (prefix.contains('@') || !id.contains('@')))
        .filter(|id| crate::has_capture(id))
        .collect();
    ids.sort();
    ids.into_iter().next().map(|s| &*Box::leak(s.into_boxed_str()))
}

fn replace(v: &mut [Shot], id: &str, f: impl FnOnce(&Shot) -> Shot) {
    if let Some(i) = v.iter().position(|s| s.id == id) {
        v[i] = f(&v[i]);
    }
}

fn insert_after(v: &mut Vec<Shot>, id: &str, s: Shot) {
    let i = v.iter().position(|x| x.id == id).map(|i| i + 1).unwrap_or(v.len());
    v.insert(i, s);
}

/// Swap placeholders for the camera department's captures as they land (film/captures/keep.txt names them).
fn captured(v: &mut Vec<Shot>) {
    let by = H - 250.0;
    let lx = MX;
    if let Some(c) = capid("hero-gen-100-122-60fps@final1") {
        replace(v, "Final1", |o| shot("Final1", 10, o.dur, plate(cap(c), C4K, F4K, P4K)).ov(o.overlap).th(482.0, 521.0)
            .lab(chap("10", "FINISHED VIEW")).sfx(o.sfx.clone()));
    }
    if let Some(c) = capid("hero-gen-100-122-60fps@close") {
        replace(v, "Close", |o| shot("Close", 12, o.dur, plate(cap(c), C4K, F4K, P4K)).ov(o.overlap).fout(o.fade_out).th(663.0, 750.0)
            .lab(o.labels.clone()).sfx(o.sfx.clone()));
    }
    if let Some(c) = capid("hero-gen-redlight-190-206@final3") {
        replace(v, "Final3", |o| shot("Final3", 10, o.dur, plate(cap(c), C4K, F4K, P4K)).ov(o.overlap).th(521.0, 521.0)
            .lab(o.labels.clone()));
    }
    if let Some(c) = capid("ladder-r2-live-v4-110-auth") {
        replace(v, "FirstWorld", |o| shot("FirstWorld", 4, o.dur, plate(cap(c), C4K, F4K, P4K)).th(77.0, 97.0)
            .lab(with(chap("04", "FIRST WORLD"), vec![
                lab(1.0, 9.8, lx, by, "LIVE WORLD V4 · 17:20 · AS IT LOOKED").size(40.0),
                lab(1.6, 9.8, lx, by + 60.0, "NEWEST VIEW WINS · COVERS 92–96 % OF EACH CAMERA'S VIEW").size(34.0).dim(),
            ])).sfx(o.sfx.clone()));
    }
    if let (Some(a), Some(b), Some(c)) = (capid("ladder-r2-live-v4-110-today"), capid("ladder-r3-stitched-opt-v2-110-today"), capid("ladder-r3-stitched-opt-v3-110-today")) {
        replace(v, "Ladder", |o| shot("Ladder", 5, o.dur, Kind::Ladder { wipe: 1.1, steps: vec![
            Step { src: cap(a), size: C4K, crop: P4K, at: 0.0, start: 0.0 },
            Step { src: cap(b), size: C4K, crop: P4K, at: 3.4, start: 0.0 },
            Step { src: cap(c), size: C4K, crop: P4K, at: 7.0, start: 0.0 },
        ] }).th(95.0, 149.0).lab(o.labels.clone()).sfx(o.sfx.clone()));
    }
    if let (Some(a), Some(b)) = (capid("ladder-r3-stitched-opt-v3-110-today"), capid("ladder-r5-stitched-opt-v4-110-today")) {
        replace(v, "Teacher", |o| shot("Teacher", 6, o.dur, Kind::Ladder { wipe: 1.2, steps: vec![
            Step { src: cap(a), size: C4K, crop: P4K, at: 0.0, start: 0.0 },
            Step { src: cap(b), size: C4K, crop: P4K, at: 2.0, start: 0.0 },
        ] }).th(214.0, 241.0).sfx(o.sfx.clone()).lab(vec![
            lab(0.4, 2.0, lx, by, "OPTIMISER V3").size(40.0),
            lab(3.0, 5.9, lx, by, "V4 · TRAINED ON TEACHER PHOTOGRAPHS AT THE CHASE CAMERA · 19:46").size(40.0),
        ]));
    }
    if let Some(c) = capid("ladder-r4-stitched-opt-v3-110-auth-photo") {
        replace(v, "CutOuts", |o| shot("CutOuts", 6, o.dur, plate(cap(c), C4K, F4K, P4K)).th(142.0, 165.0)
            .lab(o.labels.clone()).sfx(o.sfx.clone()));
    }
    if let (Some(a), Some(b)) = (capid("ladder-r5-stitched-opt-v5-110-today-splatground"), capid("ladder-r6-stitched-opt-v5-110-today-atlas")) {
        replace(v, "Wide", |o| shot("Wide", 7, o.dur, Kind::Ladder { wipe: 1.4, steps: vec![
            Step { src: cap(a), size: C4K, crop: P4K, at: 0.0, start: 0.0 },
            Step { src: cap(b), size: C4K, crop: P4K, at: 1.4, start: 0.0 },
        ] }).th(295.0, 336.0).power(o.power.unwrap_or(4.6)).sfx(o.sfx.clone()).lab(vec![
            lab(0.3, 1.5, lx, by, "GROUND AS SPLATS").size(40.0),
            lab(2.6, 4.5, lx, by, "ROAD ATLAS · CONTINUOUS TO 40 M EACH SIDE · 21:14").size(40.0),
        ]));
    }
    let movers: Vec<&'static str> = ["scan", "clean"].iter()
        .filter_map(|k| capid(&format!("ladder-r7-stitched-opt-v5-movers-v1-284-today-{k}"))).collect();
    if movers.len() == 2 {
        replace(v, "Movers", |o| shot("Movers", 8, o.dur, Kind::Ladder { wipe: 1.2, steps: vec![
            Step { src: cap(movers[0]), size: C4K, crop: P4K, at: 0.0, start: 0.0 },
            Step { src: cap(movers[1]), size: C4K, crop: P4K, at: 4.2, start: 0.0 },
        ] }).th(o.thread.map(|t| t.0).unwrap_or(266.0), o.thread.map(|t| t.1).unwrap_or(521.0)).lab(o.labels.clone()).sfx(o.sfx.clone()));
    }
    if let (Some(a), Some(b)) = (capid("ladder-r7-stitched-opt-v5-movers-v1-312-today-clean"), capid("ladder-r8-stitched-opt-v5-movers-v1-312-today-generative")) {
        insert_after(v, "ImageModel", shot("Split", 9, 7.0, Kind::Split {
            a: Step { src: cap(a), size: C4K, crop: F4K, at: 0.0, start: 0.0 },
            b: Step { src: cap(b), size: C4K, crop: F4K, at: 0.0, start: 0.0 },
            x0: 0.62, x1: 0.38,
        }).th(482.0, 482.0).lab(vec![
            lab(0.6, 6.8, lx, MY, "SPLATS").size(44.0).dim(),
            lab(0.6, 6.8, W - MX, MY, "GENERATIVE").end().size(44.0).dim(),
            lab(1.0, 6.8, lx, by, "THE SAME MOMENT · 312 S").size(40.0),
        ]).sfx(vec![cue(0.0, "sweep.wav", -20.0)]));
    }
    if let Some(c) = capid("dir-points-to-splats-110-122") {
        insert_after(v, "Depth", shot("Splats", 2, 6.5, plate(cap(c), C4K, F4K, P4K)).th(131.0, 256.0)
            .lab(vec![lab(0.8, 6.3, lx, by, "POINTS BECOME SPLATS · SHOWN ON THE V5 SCENE").size(40.0)])
            .sfx(vec![cue(0.3, "shimmer.wav", -17.0)]));
    }
}

pub fn all() -> Vec<Shot> {
    let mut v = base();
    captured(&mut v);
    if let Some(e) = evolution() {
        let i = v.iter().position(|s| s.id == "Final1").unwrap_or(v.len());
        v.insert(i, e);
    }
    v
}

fn base() -> Vec<Shot> {
    let lx = MX;
    let by = H - 250.0; // label line above the commit thread
    vec![
        // 0 cold open
        shot("Open", 0, 5.5, plate(Src::V("junction.mp4"), VIEW, VIEW_IN, VIEW_IN2)).ov(0.0).unless(HERO_J)
            .sfx(vec![cue(1.6, "passby.wav", -17.0).pan(0.2)]),
        shot("Open", 0, 5.5, plate(cap(&format!("{HERO_J}@open")), C4K, F4K, P4K)).ov(0.0).need(HERO_J)
            .sfx(vec![cue(1.6, "passby.wav", -17.0).pan(0.2)]),
        shot("Title", 0, 3.6, Kind::Title(vec![
            TitleLine { text: "REALITY SPLAT", size: 132.0, y: 1080.0, t0: 0.25, color: FG, weight: 400, spacing: 40.0 },
            TitleLine { text: "one drive · six cameras · one night", size: 48.0, y: 1190.0, t0: 1.0, color: DIM, weight: 400, spacing: 4.0 },
        ])).ov(0.0).fout(0.7).mood([-30.0, -70.0, -27.0])
            .sfx(vec![cue(0.0, "sub.wav", -18.5), cue(0.25, "tick.wav", -16.0)]),
        // 1 input
        shot("Unfold", 1, 8.4, plate(Src::V("unfold.mp4"), (W, H), [0.0, 0.0, W, H], [0.0, 0.0, W, H])).ov(0.6)
            .lab(chap("01", "INPUT"))
            .th(0.0, 0.0)
            .sfx(vec![cue(0.6, "unfold.wav", -17.0), cue(1.25, "unfold.wav", -19.0).pan(-0.5), cue(1.45, "unfold.wav", -19.0).pan(0.5),
                cue(2.05, "unfold.wav", -20.0).pan(-0.7), cue(2.25, "unfold.wav", -20.0).pan(0.7), cue(2.9, "unfold.wav", -18.0),
                cue(0.62, "shutter.wav", -21.0)]),
        shot("Cameras", 1, 7.0, Kind::Grid6 { tag: "cams108" }).th(0.0, 0.0)
            .sfx(vec![cue(0.1, "shutter.wav", -17.0), cue(0.35, "shutter.wav", -20.0).pan(-0.4), cue(0.5, "shutter.wav", -20.0).pan(0.4)])
            .mood([-21.0, -40.0, -30.0])
            .lab(vec![lab(0.8, 6.6, lx, MY, "SIX CAMERAS · 36 FPS").size(44.0).dim()]),
        shot("Drive", 1, 9.0, Kind::DriveMap).th(0.0, 4.0)
            .sfx(vec![cue(0.5, "sweep.wav", -22.0), cue(6.8, "tick.wav", -15.0)])
            .lab(vec![lab(6.8, 8.9, W - MX, 300.0, "3.3 KM").end().size(96.0), lab(7.1, 8.9, W - MX, 370.0, "ONE CONTINUOUS DRIVE").end().size(38.0).dim()]),
        // 2 geometry
        shot("Calib", 2, 6.5, Kind::Stack(vec![
            Kind::Panels(vec![panel(Src::I("calib-transfer.jpg"), (1800.0, 627.0), [900.0, 42.0, 900.0, 585.0], [MX, 560.0, 1700.0, 1105.0], 0.0, 7.5)]),
            Kind::Bars { rows: vec![("FRONT", 0.974), ("BACK", 0.915), ("LEFT PILLAR", 0.815), ("RIGHT PILLAR", 0.823), ("LEFT REPEATER", 0.815), ("RIGHT REPEATER", 0.813)],
                max: 1.35, refv: 1.0, refl: "1 PX", unit: "PX", x: MX + 1880.0, y: 640.0, w: 1000.0, t0: 1.0 },
        ])).th(5.0, 8.0)
            .sfx(vec![cue(0.0, "shutter.wav", -17.0), cue(1.2, "blips.wav", -24.0)])
            .lab(with(chap("02", "GEOMETRY"), vec![
                lab(0.8, 7.3, MX, 1740.0, "GOLD MEASURED · MAGENTA PROJECTED FROM ANOTHER CAMERA").size(34.0).dim(),
                lab(1.2, 7.3, MX + 1880.0, 1340.0, "REPROJECTION ERROR · CALIBRATED FROM THE FOOTAGE ALONE").size(34.0).dim(),
            ])),
        shot("Depth", 2, 13.0, plate(Src::V("depth.mp4"), (W, H), [0.0, 0.0, W, H], [0.0, 0.0, W, H])).th(125.0, 131.0)
            .sfx(vec![cue(0.9, "sweep.wav", -22.0), cue(2.3, "shimmer.wav", -15.0), cue(4.1, "shimmer.wav", -17.0).pan(-0.3), cue(8.6, "shimmer.wav", -18.0).pan(0.3)])
            .lab(vec![
                lab(1.0, 4.0, lx, MY, "DEPTH · FRONT CAMERA · 111 S").size(44.0).dim(),
                lab(6.6, 10.0, lx, MY, "SIX CAMERAS · ONE INSTANT").size(44.0).dim(),
                lab(10.3, 13.3, lx, MY, "+ 191 KEYFRAMES").size(44.0).dim(),
            ]),
        shot("Masks", 2, 5.5, plate(Src::V("masks300.mp4"), (W, H), [0.0, 0.0, W, H], [100.0, 56.0, 3640.0, 2048.0])).th(131.0, 135.0)
            .sfx(vec![cue(0.0, "sweep.wav", -21.0), cue(0.1, "blips.wav", -19.0)])
            .lab(vec![lab(1.0, 6.2, lx, MY, "SKY · MOVING · PARKED").size(44.0).dim()]),
        // 3 first ground
        shot("Mosaic", 3, 8.0, plate(Src::I("road-turn.jpg"), (5540.0, 2752.0), [0.0, 0.0, 5540.0, 2752.0], [560.0, 760.0, 1900.0, 1069.0])).th(31.0, 53.0)
            .sfx(vec![cue(0.0, "sweep.wav", -21.0)])
            .lab(with(chap("03", "FIRST GROUND"), vec![lab(1.5, 7.8, lx, by, "BIRD'S-EYE ROAD · 2 CM · 16:16").size(40.0)])),
        shot("GroundLive", 3, 5.0, plate(cap("ladder-r1-road-live-v1-110-auth"), C4K, F4K, P4K)).need("ladder-r1-road-live-v1-110-auth").th(53.0, 58.0)
            .lab(vec![lab(0.6, 5.8, lx, by, "GROUND-ONLY LIVE MAP · FIRST VIEWER · 16:53").size(40.0)]),
        shot("FirstDemo", 3, 7.5, plate(Src::V("demo-ground.mp4"), (1920.0, 1080.0), [0.0, 0.0, 1920.0, 1080.0], [40.0, 22.0, 1840.0, 1035.0])).th(58.0, 66.0).mood([-22.0, -36.0, -30.0])
            .lab(vec![lab(0.8, 9.6, lx, by, "FIRST DEMO · AS SENT").size(40.0).dim()]),
        // 4 first world
        shot("FirstWorld", 4, 8.0, Kind::Panels(vec![
            panel(Src::V("live-v4-chase.mp4"), (768.0, 432.0), [0.0, 0.0, 768.0, 432.0], [MX, 560.0, 1880.0, 1057.5], 0.0, 10.0),
            panel(Src::V("live-v4-orbit.mp4"), (512.0, 288.0), [0.0, 0.0, 512.0, 288.0], [MX + 1960.0, 560.0, 1280.0, 720.0], 0.6, 10.0),
        ])).th(77.0, 97.0)
            .lab(with(chap("04", "FIRST WORLD"), vec![
                lab(1.0, 9.8, MX, 1720.0, "LIVE WORLD V4 · 17:20").size(40.0),
                lab(1.6, 9.8, MX, 1780.0, "NEWEST VIEW WINS · COVERS 92–96 % OF EACH CAMERA'S VIEW").size(34.0).dim(),
                lab(2.0, 9.8, MX + 1960.0, 1390.0, "ORBIT · ARTERIAL").size(34.0).dim(),
            ]))
            .sfx(vec![cue(0.4, "shimmer.wav", -19.0)]),
        // 5 optimiser
        shot("Ladder", 5, 10.5, Kind::Ladder { wipe: 1.1, steps: vec![
            step(Src::I("lad-v4.jpg"), VIEW, VIEW_IN, 0.0),
            step(Src::I("lad-opt2.jpg"), (720.0, 450.0), [55.0, 35.0, 610.0, 343.0], 3.4),
            step(Src::I("lad-opt3.jpg"), VIEW, VIEW_IN, 7.0),
        ] }).th(95.0, 149.0)
            .sfx(vec![cue(3.25, "sweep.wav", -15.0), cue(6.85, "sweep.wav", -15.0)])
            .lab(with(chap("05", "OPTIMISER"), vec![
                lab(0.6, 3.4, lx, by, "LIVE WORLD V4 · 17:20").size(40.0),
                lab(4.3, 7.0, lx, by, "OPTIMISER V2 · 18:02 · 19.72 DB").size(40.0),
                lab(7.9, 10.4, lx, by, "OPTIMISER V3 · 18:14 · 19.96 DB").size(40.0),
                lab(4.6, 10.4, lx, by + 60.0, "PSNR ON 852 HELD-OUT PHOTOGRAPHS").size(34.0).dim(),
            ])),
        shot("SecondDemo", 5, 7.0, plate(Src::V("demo-world.mp4"), (1920.0, 1080.0), [0.0, 0.0, 1920.0, 1080.0], [40.0, 22.0, 1840.0, 1035.0])).th(149.0, 152.0)
            .lab(vec![lab(0.8, 8.8, lx, by, "SECOND DEMO · AS SENT · 18:17").size(40.0).dim()]).mood([-23.0, -36.0, -28.0]),
        // 6 experiments
        shot("CutOuts", 6, 7.0, plate(Src::V("ibr.mp4"), VIEW, VIEW_IN, VIEW_IN2)).th(142.0, 142.0)
            .sfx(vec![cue(0.0, "dissonant.wav", -17.0)])
            .lab(with(chap("06", "EXPERIMENTS"), vec![
                lab(1.0, 6.8, lx, by, "PHOTOGRAPHIC CUT-OUTS · 18:07").size(40.0),
            ])),
        shot("NVS", 6, 3.5, Kind::Panels(vec![panel(Src::V("streetnvs.mp4"), (512.0, 320.0), [0.0, 0.0, 512.0, 320.0], [1920.0 - 768.0, 1080.0 - 560.0, 1536.0, 960.0], 0.0, 4.0)])).th(158.0, 158.0)
            .sfx(vec![cue(0.2, "blips.wav", -23.0)])
            .lab(vec![
                lab(0.5, 3.8, lx, by, "VIDEO DIFFUSION · 18:23").size(40.0),
            ]),
        shot("Morph", 6, 6.5, plate(Src::V("morph.mp4"), VIEW, [0.0, 0.0, 1440.0, 810.0], [80.0, 45.0, 1280.0, 720.0])).th(199.0, 199.0)
            .sfx(vec![cue(0.0, "dissonant.wav", -17.0)])
            .lab(vec![
                lab(0.6, 6.8, lx, by, "RE-PROJECTED PHOTOGRAPH · 19:04").size(40.0),
            ]),
        shot("Teacher", 6, 6.0, Kind::Panels(vec![panel(Src::V("v4v3.mp4"), (1280.0, 426.0), [0.0, 0.0, 1280.0, 426.0], [MX, 520.0, 3240.0, 1078.3], 0.0, 6.0)])).th(214.0, 256.0)
            .sfx(vec![cue(0.0, "sweep.wav", -17.0)]).mood([-32.0, -27.0, -27.0])
            .lab(vec![lab(0.6, 6.8, lx, by, "THE PHOTOGRAPH AS TEACHER · V3 | V4").size(40.0)]),
        // 7 road atlas
        shot("Atlas", 7, 12.0, Kind::Atlas { zoom: Some((6.6, 5.0)), reveal: 6.0 }).th(267.0, 295.0)
            .sfx(vec![cue(0.3, "shimmer.wav", -17.0), cue(0.2, "sweep.wav", -22.0)])
            .lab(with(chap("07", "ROAD ATLAS"), vec![
                lab(3.6, 6.8, W - MX, 300.0, "3,312 M OF ROAD").end().size(60.0),
                lab(3.9, 6.8, W - MX, 370.0, "3,212 TILES · 40 M EACH SIDE").end().size(38.0).dim(),
                lab(10.6, 11.9, MX, by, "1 CM PER PIXEL · 128 S").size(40.0),
            ])),
        shot("Lane", 7, 8.0, Kind::Panels(vec![
            panel(Src::I("lane-splat.png"), (1112.0, 1944.0), [0.0, 0.0, 1112.0, 1944.0], [MX, 330.0, 1000.0, 1350.0], 0.0, 8.0),
            panel(Src::I("lane-atlas.png"), (1112.0, 1944.0), [0.0, 0.0, 1112.0, 1944.0], [MX + 1120.0, 330.0, 1000.0, 1350.0], 0.35, 8.0),
            panel(Src::I("lane-photo.png"), (1112.0, 1944.0), [0.0, 0.0, 1112.0, 1944.0], [MX + 2240.0, 330.0, 1000.0, 1350.0], 0.7, 8.0),
        ])).th(267.0, 267.0)
            .sfx(vec![cue(0.0, "shutter.wav", -21.0).pan(-0.5), cue(0.4, "shutter.wav", -21.0), cue(0.8, "shutter.wav", -21.0).pan(0.5)])
            .lab(vec![
                lab(1.2, 7.8, MX, 1770.0, "SPLATS").size(40.0).dim(), lab(1.2, 7.8, MX, 1850.0, "4.50 CM").size(72.0),
                lab(1.6, 7.8, MX + 1120.0, 1770.0, "ATLAS").size(40.0).dim(), lab(1.6, 7.8, MX + 1120.0, 1850.0, "1.23 CM").size(72.0).accent(),
                lab(2.0, 7.8, MX + 2240.0, 1770.0, "PHOTOGRAPH").size(40.0).dim(), lab(2.0, 7.8, MX + 2240.0, 1850.0, "1.01 CM").size(72.0),
                lab(2.6, 7.8, W - MX, 1940.0, "LANE-LINE EDGE WIDTH · 128 S").end().size(34.0).dim(),
            ]),
        shot("Wide", 7, 7.0, Kind::Ladder { wipe: 1.2, steps: vec![
            step(Src::I("v6-before.jpg"), (1952.0, 1250.0), [210.0, 140.0, 1540.0, 866.0], 0.0),
            step(Src::I("v6-check.jpg"), (1952.0, 1250.0), [210.0, 140.0, 1540.0, 866.0], 1.6),
        ] }).th(329.0, 336.0).power(4.6)
            .sfx(vec![cue(1.45, "sweep.wav", -15.0), cue(4.6, "power_down.wav", -13.5), cue(6.0, "power_up.wav", -15.0)])
            .lab(vec![
                lab(0.4, 1.6, lx, by, "GROUND V5").size(40.0),
                lab(2.6, 4.5, lx, by, "+ 40 M EACH SIDE · 21:14").size(40.0),
            ]),
        // 8 vehicles
        shot("Tracks", 8, 8.0, Kind::Movers { t0: 304.0, t1: 330.0 }).th(163.0, 217.0)
            .lab(with(chap("08", "VEHICLES"), vec![
                lab(1.0, 9.8, MX, by, "23 MOVING · 9 PARKED · ON THE V5 WORLD").size(40.0),
                lab(1.4, 9.8, MX, by + 60.0, "SHOWN 304–330 S · RIBBONS ARE TRACKS").size(34.0).dim(),
            ]))
            .sfx(vec![cue(1.5, "passby.wav", -20.0).pan(-0.3), cue(5.0, "passby.wav", -21.0).pan(0.4), cue(0.2, "blips.wav", -23.0)]),
        shot("Movers", 8, 8.5, Kind::LadderIn { wipe: 1.2, dest: [480.0, 200.0, 2880.0, 1620.0], steps: vec![
            step(Src::V("movers-pull.mp4"), (960.0, 600.0), [60.0, 38.0, 840.0, 525.0], 0.0),
            step(Src::V("clean-cars.mp4"), VIEW, VIEW_IN, 4.2),
        ] }).th(266.0, 521.0)
            .sfx(vec![cue(1.2, "passby.wav", -18.0).pan(-0.2), cue(4.05, "sweep.wav", -16.0), cue(5.8, "passby.wav", -17.0).pan(0.3)])
            .lab(with(vec![], vec![
                lab(0.6, 4.1, lx, by, "SCANNED MODELS · 20:11").size(40.0),
                lab(5.1, 8.4, lx, by, "CLEAN MODELS · 00:26").size(40.0),
            ])),
        // 9 image model
        shot("ImageModel", 9, 18.0, Kind::Stack(vec![
            Kind::LadderIn { wipe: 2.6, dest: BIG, steps: vec![
                Step { src: Src::V("gen110-render.mp4"), size: VIEW, crop: [0.0, 0.0, 1440.0, 900.0], at: 0.0, start: 0.0 },
                Step { src: Src::V("gen110-result.mp4"), size: VIEW, crop: [0.0, 0.0, 1440.0, 900.0], at: 4.6, start: 0.0 },
            ] },
            Kind::Panels(vec![
                Panel { src: Src::V("gen110-change.mp4"), size: VIEW, crop0: [0.0, 0.0, 1440.0, 900.0], crop1: [0.0, 0.0, 1440.0, 900.0],
                    dest: BIG, t0: 7.8, t1: 10.9, start: 0.0, frame: false, alpha: 0.92 },
                Panel { src: Src::V("gen110-prov.mp4"), size: VIEW, crop0: [0.0, 0.0, 1440.0, 900.0], crop1: [0.0, 0.0, 1440.0, 900.0],
                    dest: BIG, t0: 11.1, t1: 14.2, start: 0.0, frame: false, alpha: 0.82 },
                Panel { src: Src::V("gen110-ref.mp4"), size: (1448.0, 938.0), crop0: [0.0, 0.0, 1448.0, 938.0], crop1: [0.0, 0.0, 1448.0, 938.0],
                    dest: [BIG[0] + BIG[2] - 860.0 - 40.0, BIG[1] + 40.0, 860.0, 557.1], t0: 2.2, t1: 9.0, start: 0.0, frame: true, alpha: 1.0 },
            ]),
        ])).th(273.0, 482.0)
            .sfx(vec![cue(0.0, "sweep.wav", -21.0), cue(0.8, "shutter.wav", -18.0), cue(2.2, "shimmer.wav", -17.5), cue(7.7, "blips.wav", -22.0), cue(11.0, "blips.wav", -22.0)])
            .lab(with(chap("09", "IMAGE MODEL"), vec![
                lab(0.6, 4.6, BIG[0], BIG[1] + BIG[3] + 70.0, "IN · SPLAT RENDER AT THE CHASE CAMERA").size(40.0),
                lab(2.6, 8.8, BIG[0] + BIG[2] - 900.0, BIG[1] + 650.0, "+ NEAREST PHOTOGRAPH").size(34.0),
                lab(4.8, 7.7, BIG[0], BIG[1] + BIG[3] + 70.0, "OUT · DIFIX-REF, ONE STEP · 1.4–2.4 S PER FRAME").size(40.0),
                lab(8.0, 10.9, BIG[0], BIG[1] + BIG[3] + 70.0, "CHANGED PIXELS · THE ROAD IS KEPT").size(40.0),
                lab(11.3, 14.2, BIG[0], BIG[1] + BIG[3] + 70.0, "OBSERVED").size(40.0).accent(),
                Label { color: "#ede6d6", ..lab(11.5, 14.2, BIG[0] + 12.0 * 28.8, BIG[1] + BIG[3] + 70.0, "REPAIRED").size(40.0) },
                Label { color: "#8a8f96", ..lab(11.7, 14.2, BIG[0] + 24.0 * 28.8, BIG[1] + BIG[3] + 70.0, "FILLED").size(40.0) },
                lab(14.4, 17.8, BIG[0], BIG[1] + BIG[3] + 70.0, "STABILISED · FLICKER 0.81–1.16 LEVELS · REAL CAMERA 1.36–1.59").size(40.0),
            ])),
        // 10 finished view
        shot("Final1", 10, 8.0, plate(Src::V("final093.mp4"), VIEW, VIEW_IN, VIEW_IN2)).th(482.0, 521.0).ov(0.8)
            .sfx(vec![cue(-0.4, "sub.wav", -21.0), cue(4.0, "passby.wav", -20.0)])
            .lab(chap("10", "FINISHED VIEW")),
        shot("Final2", 10, 11.5, Kind::Panels(vec![
            if crate::has_capture(HERO_J) { panel(cap(&format!("{HERO_J}@final2")), C4K, F4K, [MX + 860.0, 300.0, 2380.0, 1338.0], 0.0, 14.0) }
            else { panel(Src::V("final296.mp4"), VIEW, VIEW_IN, [MX + 860.0, 300.0, 2380.0, 1338.0], 0.0, 14.0) },
            panel(Src::V("cams306-front.mp4"), (1448.0, 938.0), [0.0, 0.0, 1448.0, 938.0], [MX, 300.0, 800.0, 518.0], 0.3, 14.0),
            panel(Src::V("cams306-left_pillar.mp4"), (1448.0, 938.0), [0.0, 0.0, 1448.0, 938.0], [MX, 850.0, 390.0, 252.6], 0.5, 14.0),
            panel(Src::V("cams306-right_pillar.mp4"), (1448.0, 938.0), [0.0, 0.0, 1448.0, 938.0], [MX + 410.0, 850.0, 390.0, 252.6], 0.6, 14.0),
            panel(Src::V("cams306-left_repeater.mp4"), (1448.0, 938.0), [0.0, 0.0, 1448.0, 938.0], [MX, 1135.0, 390.0, 252.6], 0.7, 14.0),
            panel(Src::V("cams306-right_repeater.mp4"), (1448.0, 938.0), [0.0, 0.0, 1448.0, 938.0], [MX + 410.0, 1135.0, 390.0, 252.6], 0.8, 14.0),
            panel(Src::V("cams306-back.mp4"), (1448.0, 938.0), [0.0, 0.0, 1448.0, 938.0], [MX, 1420.0, 800.0, 518.0], 0.9, 14.0),
        ])).th(521.0, 521.0)
            .lab(vec![
                lab(0.6, 12.3, MX, 260.0, "DASHCAM").size(34.0).dim(),
                lab(0.9, 12.3, MX + 860.0, 260.0, "RECONSTRUCTED · SAME MOMENT · 306 S").size(34.0).dim(),
            ])
            .sfx(vec![cue(0.3, "shutter.wav", -20.0)]),
        shot("Final3", 10, 5.5, plate(Src::V("final190.mp4"), VIEW, VIEW_IN, VIEW_IN2)).th(521.0, 521.0).ov(0.7)
            .lab(vec![lab(1.0, 6.6, lx, by, "STOPPED AT THE LIGHT · FRAMES HOLD STILL").size(36.0).dim()]),
        // 11 pipeline
        shot("Pipeline", 11, 11.0, Kind::Stack(vec![
            Kind::Atlas { zoom: None, reveal: 8.5 },
            Kind::Stages { names: STAGES.to_vec(), t0: 0.8, step: 0.55, x: W - MX - 640.0, y: 560.0 },
        ])).th(471.0, 663.0)
            .sfx({ let mut v = vec![cue(0.3, "shimmer.wav", -21.0)]; for i in 0..STAGES.len() { v.push(cue(0.8 + i as f32 * 0.55, "tick.wav", -19.0).pan(0.35)); } v })
            .lab(with(chap("11", "ONE COMMAND"), vec![
                lab(8.6, 10.9, W - MX, 300.0, "30 S OF FOOTAGE → 35 MIN 52 S").end().size(56.0),
                lab(9.0, 10.9, W - MX, 380.0, "ONE COMMAND · NO MANUAL STEP").end().size(40.0).dim(),
            ])),
        // 12 close
        shot("Close", 12, 8.0, plate(Src::V("close.mp4"), VIEW, VIEW_IN, VIEW_IN2)).ov(0.8).fout(2.4).th(663.0, 750.0)
            .sfx(vec![cue(-0.3, "sub.wav", -19.0)])
            .lab(vec![lab(2.0, 7.0, 1920.0, 1100.0, "15:45 → 04:15 · 53 COMMITS").mid().size(52.0)]),
    ]
}
