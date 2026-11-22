//! Reality Splat: the journey. A documentary film of one day and night of rebuilding Tesla's Reality View from
//! one drive's dashcam footage. Every shot is project footage, project data or a capture of the project's viewer;
//! every number is a measurement from the project's reports (film/notes/facts.md).
pub mod data;
pub mod draw;
pub mod kinds;
pub mod shots;
pub mod sound;

use draw::*;
use fframes::{AudioMap, Color, Duration, FFramesContext, Frame, Overlap, Scene, Scenes, Svgr, Video, include_media_dir};

include_media_dir!(pub struct FilmFonts, "media");

pub const WIDTH: usize = 3840;
pub const HEIGHT: usize = 2160;

pub struct Shot {
    pub id: &'static str,
    pub chapter: usize,
    pub dur: f32,
    pub overlap: f32,
    pub kind: kinds::Kind,
    pub labels: Vec<Label>,
    /// commit-thread playhead in minutes after 15:45 at the start and end of the shot; None hides the thread
    pub thread: Option<(f32, f32)>,
    pub fade_in: f32,
    pub fade_out: f32,
    pub sfx: Vec<sound::Cue>,
    /// scene second of the 21:19 power cut, if this shot carries it
    pub power: Option<f32>,
    /// bed levels for this shot (cabin, room, pad) in dB; None uses the chapter's
    pub mood: Option<[f32; 3]>,
    /// keep this shot only when the capture with this id is in film/work/media (cap-<id>.mp4)
    pub need: Option<&'static str>,
    /// keep this shot only when that capture is missing (the placeholder it stands in for)
    pub unless: Option<&'static str>,
}

/// CLEAN=1 renders the picture with no lettering (labels, chapter titles, commit thread, title card, chart text):
/// source footage to cut with; captions can be added again in the edit.
pub fn clean() -> bool {
    static C: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *C.get_or_init(|| std::env::var("CLEAN").is_ok_and(|v| !v.is_empty() && v != "0"))
}

pub const MEDIA_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../work/media");

/// whether a capture from the camera department's library is available
pub fn has_capture(id: &str) -> bool {
    std::path::Path::new(MEDIA_DIR).join(format!("cap-{id}.mp4")).exists()
}

impl std::fmt::Debug for Shot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.id)
    }
}

impl Scene for Shot {
    fn name(&self) -> &'static str {
        self.id
    }
    fn duration(&self) -> Duration<'_> {
        Duration::Seconds(self.dur)
    }
    fn overlap(&self) -> Overlap {
        if self.overlap > 0.0 { Overlap::Previous(self.overlap) } else { Overlap::None }
    }
    fn render_frame<'a>(&'a self, frame: Frame, ctx: &FFramesContext<'a, '_>) -> Svgr<'a> {
        // scene time measured from the cut (the end of the previous shot); negative during the cross-fade
        let t = frame.seconds() - self.overlap;
        let body = kinds::render(self, &frame, ctx, t);
        let mut a = 1.0;
        if self.overlap > 0.0 {
            a *= io(ramp(t, -self.overlap, self.overlap));
        }
        if self.fade_in > 0.0 {
            a *= io(ramp(t, 0.0, self.fade_in));
        }
        if self.fade_out > 0.0 {
            a *= 1.0 - io(ramp(t, self.dur - self.fade_out, self.fade_out));
        }
        let (lab, scrim) = if clean() { (Svgr::empty(), 0.0) } else { (labels(t, &self.labels), 1.0) };
        // soft scrims keep the small lettering and the commit thread legible over bright footage
        fframes::svgr!(<g opacity={a}>
            <rect x="0" y="0" width={W} height={H} fill={BG} />
            {body}
            <defs>
                <linearGradient id="scrimb" x1="0" x2="0" y1="0" y2="1">
                    <stop offset="0" stop-color="#000000" stop-opacity="0" />
                    <stop offset="1" stop-color="#000000" stop-opacity="0.62" />
                </linearGradient>
                <linearGradient id="scrimt" x1="0" x2="0" y1="0" y2="1">
                    <stop offset="0" stop-color="#000000" stop-opacity="0.42" />
                    <stop offset="1" stop-color="#000000" stop-opacity="0" />
                </linearGradient>
            </defs>
            <rect x="0" y="1560" width={W} height="600" fill="url(#scrimb)" opacity={scrim} />
            <rect x="0" y="0" width={W} height="380" fill="url(#scrimt)" opacity={scrim} />
            {lab}
        </g>)
    }
}

pub struct Journey {
    pub shots: Vec<Shot>,
    /// (cut time, end time) of every shot in film seconds
    pub spans: Vec<(f32, f32)>,
    pub only_chapter: Option<usize>,
}

impl std::fmt::Debug for Journey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Journey")
    }
}

impl Journey {
    pub fn new(only_chapter: Option<usize>) -> Self {
        let mut shots: Vec<Shot> = shots::all()
            .into_iter()
            .filter(|s| s.need.is_none_or(has_capture) && s.unless.is_none_or(|id| !has_capture(id)))
            .filter(|s| only_chapter.is_none_or(|c| s.chapter == c))
            .filter(|s| !(clean() && matches!(s.kind, kinds::Kind::Title(_))))
            .collect();
        // a chapter rendered on its own opens with a short fade instead of a cross-fade from a shot that is not
        // there (fframes does not draw a lone first scene that overlaps a previous one)
        if let Some(first) = shots.first_mut() {
            if first.overlap > 0.0 {
                first.overlap = 0.0;
                first.fade_in = first.fade_in.max(0.4);
            }
        }
        let mut spans = Vec::new();
        let mut end = 0.0;
        for s in &shots {
            spans.push((end, end + s.dur));
            end += s.dur;
        }
        Self { shots, spans, only_chapter }
    }

    fn active(&self, t: f32) -> Option<(usize, f32)> {
        self.spans.iter().position(|&(a, b)| t >= a && t < b).map(|i| (i, t - self.spans[i].0))
    }
}

impl Video for Journey {
    const FPS: usize = 30;
    const WIDTH: usize = WIDTH;
    const HEIGHT: usize = HEIGHT;
    const BACKGROUND_COLOR: Color = Color::BLACK;

    fn duration(&self) -> Duration<'_> {
        Duration::Seconds(self.spans.last().map(|s| s.1).unwrap_or(1.0))
    }

    fn audio(&self) -> AudioMap<'_> {
        sound::map(self)
    }

    fn define_scenes(&self) -> Scenes<'_> {
        Scenes::from(self.shots.iter().map(|s| s as &dyn Scene).collect::<Vec<_>>())
    }

    fn render_frame<'a>(&'a self, frame: Frame, ctx: &FFramesContext<'a, '_>) -> Svgr<'a> {
        let t = frame.global_index as f32 / Self::FPS as f32;
        let thread = match self.active(t).filter(|_| !clean()) {
            Some((i, st)) => kinds::thread(&self.shots[i], st, if i > 0 { self.shots[i - 1].thread.map(|t| t.1) } else { None }),
            None => Svgr::empty(),
        };
        let power = match self.active(t) {
            Some((i, st)) => kinds::power_cut(&self.shots[i], st),
            None => Svgr::empty(),
        };
        let finish = kinds::finish(ctx, frame.global_index);
        fframes::svgr!(<svg xmlns="http://www.w3.org/2000/svg" width={W} height={H} viewBox="0 0 3840 2160">
            <defs>
                <filter id="bloom" x="0" y="0" width="100%" height="100%" color-interpolation-filters="sRGB">
                    <feComponentTransfer in="SourceGraphic" result="hi">
                        <feFuncR type="linear" slope="4.0" intercept="-3.25" />
                        <feFuncG type="linear" slope="4.0" intercept="-3.25" />
                        <feFuncB type="linear" slope="4.0" intercept="-3.25" />
                    </feComponentTransfer>
                    <feGaussianBlur in="hi" stdDeviation="26" result="glow" />
                    <feComposite in="SourceGraphic" in2="glow" operator="arithmetic" k1="0" k2="1" k3="0.28" k4="0" />
                </filter>
            </defs>
            <rect width={W} height={H} fill={BG} />
            <g filter="url(#bloom)">{ctx.render_scenes(&frame)}</g>
            {thread}
            {power}
            {finish}
        </svg>)
    }
}
