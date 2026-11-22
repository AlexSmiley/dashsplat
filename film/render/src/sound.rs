//! Sound: an ambient bed (cabin + machine room) under the whole film, and effects placed from the same shot
//! timings that drive the picture. Files live in film/work/audio (loaded at runtime, stereo kept).
//! STEM=amb or STEM=sfx renders one stem; unset renders both.
use crate::Journey;
use fframes::{AudioMap, AudioTimestamp, AudioTrack, FadeCurve};

pub const AUDIO_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../work/audio");

#[derive(Clone)]
pub struct Cue {
    pub at: f32,
    pub file: &'static str,
    pub gain: f32,
    /// seconds to play (0 = to the end of the file)
    pub len: f32,
    pub offset: f32,
    pub fade_in: f32,
    pub fade_out: f32,
    pub pan: f32,
    pub amb: bool,
}

pub fn cue(at: f32, file: &'static str, gain: f32) -> Cue {
    Cue { at, file, gain, len: 0.0, offset: 0.0, fade_in: 0.0, fade_out: 0.0, pan: 0.0, amb: false }
}
impl Cue {
    pub fn len(mut self, l: f32) -> Self { self.len = l; self }
    pub fn off(mut self, o: f32) -> Self { self.offset = o; self }
    pub fn fades(mut self, i: f32, o: f32) -> Self { self.fade_in = i; self.fade_out = o; self }
    pub fn pan(mut self, p: f32) -> Self { self.pan = p; self }
    pub fn amb(mut self) -> Self { self.amb = true; self }
}

fn exists(file: &str) -> bool {
    std::path::Path::new(AUDIO_DIR).join(file).exists()
}

fn track(c: &Cue, t0: f32) -> AudioTrack<'static> {
    let start = AudioTimestamp::Second((t0 + c.at).max(0.0));
    let end = if c.len > 0.0 { AudioTimestamp::Second(t0 + c.at + c.len) } else { AudioTimestamp::Eof };
    let mut tr = AudioTrack::new(c.file, start..end).gain_db(c.gain).fade_curve(FadeCurve::EqualPower);
    if c.offset > 0.0 { tr = tr.offset(c.offset); }
    if c.fade_in > 0.0 { tr = tr.fade_in(c.fade_in); }
    if c.fade_out > 0.0 { tr = tr.fade_out(c.fade_out); }
    if c.pan != 0.0 { tr = tr.pan(c.pan); }
    tr
}

/// overall bed level against the effects
pub const BED_LIFT: f32 = 8.0;

/// bed levels per chapter in dB: (car cabin, machine room, drone pad)
pub fn mood(chapter: usize) -> [f32; 3] {
    match chapter {
        0 => [-15.0, -70.0, -70.0],
        1 => [-25.0, -33.0, -29.0],
        2 => [-32.0, -26.0, -28.0],
        3 => [-26.0, -31.0, -28.0],
        4 => [-29.0, -28.0, -28.0],
        5 => [-32.0, -26.0, -27.0],
        6 => [-36.0, -27.0, -34.0],
        7 => [-32.0, -27.0, -26.0],
        8 => [-26.0, -31.0, -27.0],
        9 => [-34.0, -25.0, -25.0],
        10 => [-22.0, -42.0, -19.5],
        11 => [-32.0, -26.0, -24.0],
        _ => [-28.0, -44.0, -21.0],
    }
}

pub fn map(j: &Journey) -> AudioMap<'static> {
    let stem = std::env::var("STEM").unwrap_or_default();
    let want = |amb: bool| stem.is_empty() || (stem == "amb") == amb;
    let total = j.spans.last().map(|s| s.1).unwrap_or(0.0);
    let mut v: Vec<AudioTrack> = Vec::new();
    // beds: one segment per shot at that shot's level, sample-aligned to film time so overlapping segments
    // (linear 0.5 s fades) only move the level and never restart or phase the bed
    if want(true) {
        for (i, (s, &(t0, t1))) in j.shots.iter().zip(&j.spans).enumerate() {
            let m = s.mood.unwrap_or_else(|| mood(s.chapter));
            for (k, file) in ["amb_cabin.wav", "amb_room.wav", "pad.wav"].iter().enumerate() {
                if !exists(file) || m[k] < -60.0 { continue; }
                let a = if i == 0 { 0.0 } else { (t0 - 0.5).max(0.0) };
                let b = if i + 1 == j.shots.len() { t1 } else { t1 + 0.5 };
                let fi = if i == 0 { 0.25 } else { 0.5 };
                let fo = if i + 1 == j.shots.len() { s.fade_out.max(1.5) } else { 0.5 };
                // the power cut takes the room with it: the beds drop out with the picture and come back after
                let segs = match s.power {
                    Some(pc) => vec![(a, t0 + pc + 0.12, fi, 0.1), (t0 + pc + 1.45, b, 0.9, fo)],
                    None => vec![(a, b, fi, fo)],
                };
                for (sa, sb, sfi, sfo) in segs {
                    v.push(AudioTrack::new(*file, AudioTimestamp::Second(sa)..AudioTimestamp::Second(sb))
                        .offset(sa + 3.0)
                        .gain_db(m[k] + BED_LIFT)
                        .fade_in(sfi)
                        .fade_out(sfo)
                        .fade_curve(FadeCurve::Linear));
                }
            }
        }
    }
    for (s, &(t0, _)) in j.shots.iter().zip(&j.spans) {
        for c in &s.sfx {
            if want(c.amb) && exists(c.file) && t0 + c.at < total {
                v.push(track(c, t0));
            }
        }
    }
    // commit ticks: a soft tick each time the thread's playhead passes a commit
    if want(false) && exists("tick.wav") {
        let commits = &crate::data::data().commits;
        for (s, &(t0, _)) in j.shots.iter().zip(&j.spans) {
            if let Some((m0, m1)) = s.thread {
                if m1 <= m0 { continue; }
                for c in commits.iter().filter(|c| c.minute > m0 && c.minute <= m1) {
                    let at = t0 + (c.minute - m0) / (m1 - m0) * s.dur;
                    v.push(AudioTrack::new("tick.wav", AudioTimestamp::Second(at)..AudioTimestamp::Eof).gain_db(-21.0).pan(-0.15 + 0.3 * (c.minute / 750.0)));
                }
            }
        }
    }
    AudioMap::from(v)
}
