//! Drawing helpers shared by every shot: easing, media plates, labels, the commit thread, finish layers.
use fframes::{FFramesContext, FFramesSyncedVideoFrame, Frame, Svgr, SyncVideoFrameInput};

pub const W: f32 = 3840.0;
pub const H: f32 = 2160.0;
pub const BG: &str = "#0b0c0e";
pub const FG: &str = "#ebe8e1";
pub const DIM: &str = "#8e939b";
pub const FAINT: &str = "#30343a";
pub const ACCENT: &str = "#3d7bff";
pub const MONO: &str = "Berkeley Mono";
/// left/right safe margin and the label baseline grid
pub const MX: f32 = 300.0;
pub const MY: f32 = 210.0;

pub fn clamp01(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}
/// cubic in-out
pub fn io(x: f32) -> f32 {
    let x = clamp01(x);
    if x < 0.5 { 4.0 * x * x * x } else { 1.0 - (-2.0 * x + 2.0).powi(3) / 2.0 }
}
/// expo-ish out (fast then settles)
pub fn out(x: f32) -> f32 {
    let x = clamp01(x);
    1.0 - (1.0 - x).powi(4)
}
pub fn inn(x: f32) -> f32 {
    clamp01(x).powi(3)
}
/// 0 -> 1 over [a, a + d]
pub fn ramp(t: f32, a: f32, d: f32) -> f32 {
    clamp01((t - a) / d.max(1e-4))
}
/// fade in over [a, a+fi], out over [b-fo, b]
pub fn window(t: f32, a: f32, b: f32, fi: f32, fo: f32) -> f32 {
    io(ramp(t, a, fi)) * (1.0 - io(ramp(t, b - fo, fo)))
}
pub fn lerp(a: f32, b: f32, s: f32) -> f32 {
    a + (b - a) * s
}
pub fn lerp4(a: [f32; 4], b: [f32; 4], s: f32) -> [f32; 4] {
    [lerp(a[0], b[0], s), lerp(a[1], b[1], s), lerp(a[2], b[2], s), lerp(a[3], b[3], s)]
}

#[derive(Clone, Copy)]
pub enum Src {
    V(&'static str),
    I(&'static str),
}

/// A source (video or still) of pixel size `size`, showing `crop` (x, y, w, h in source pixels) inside `dest`.
/// The crop is widened or heightened about its centre to match the destination aspect, so nothing distorts.
pub fn plate<'a>(frame: &Frame, ctx: &FFramesContext<'a, '_>, src: Src, size: (f32, f32), crop: [f32; 4], dest: [f32; 4], start: f32, id: &str) -> Svgr<'a> {
    let [dx, dy, dw, dh] = dest;
    let [mut cx, mut cy, mut cw, mut ch] = crop;
    let want = dw / dh;
    if cw / ch > want {
        let nw = ch * want;
        cx += (cw - nw) / 2.0;
        cw = nw;
    } else {
        let nh = cw / want;
        cy += (ch - nh) / 2.0;
        ch = nh;
    }
    let k = dw / cw;
    let (x, y, w, h) = (dx - cx * k, dy - cy * k, size.0 * k, size.1 * k);
    let href = match src {
        Src::V(file) => {
            let input = SyncVideoFrameInput { start_from: start.max(0.0), looping: false, editor_fallback_image: None };
            match frame.get_synced_video_frame(ctx, file, &input) {
                Some(v) => v.into_image().href(),
                None => return Svgr::empty(),
            }
        }
        Src::I(file) => match ctx.get_image(file) {
            Some(i) => i.href(),
            None => return Svgr::empty(),
        },
    };
    let clip = format!("clip-{id}");
    let url = format!("url(#{clip})");
    fframes::svgr!(<g>
        <defs><clipPath id={clip}><rect x={dx} y={dy} width={dw} height={dh} /></clipPath></defs>
        <g clip-path={url}><image href={href} x={x} y={y} width={w} height={h} preserveAspectRatio="none" /></g>
    </g>)
}

#[derive(Clone)]
pub struct Label {
    pub t0: f32,
    pub t1: f32,
    pub x: f32,
    pub y: f32,
    pub text: String,
    pub size: f32,
    pub anchor: &'static str,
    pub color: &'static str,
    pub weight: u16,
}

pub fn lab(t0: f32, t1: f32, x: f32, y: f32, text: &str) -> Label {
    Label { t0, t1, x, y, text: text.to_string(), size: 50.0, anchor: "start", color: FG, weight: 400 }
}
impl Label {
    pub fn size(mut self, s: f32) -> Self { self.size = s; self }
    pub fn end(mut self) -> Self { self.anchor = "end"; self }
    pub fn mid(mut self) -> Self { self.anchor = "middle"; self }
    pub fn dim(mut self) -> Self { self.color = DIM; self }
    pub fn accent(mut self) -> Self { self.color = ACCENT; self }
    pub fn bold(mut self) -> Self { self.weight = 700; self }
}

/// Small letterspaced technical label that rises 18 px and fades in, then fades out.
pub fn label<'a>(t: f32, l: &Label) -> Svgr<'a> {
    if t < l.t0 || t > l.t1 {
        return Svgr::empty();
    }
    let a = window(t, l.t0, l.t1, 0.45, 0.3);
    let dy = 18.0 * (1.0 - out(ramp(t, l.t0, 0.6)));
    let spacing = l.size * 0.12;
    let text = l.text.clone();
    fframes::svgr!(<text x={l.x} y={l.y + dy} font-family={MONO} font-size={l.size} font-weight={l.weight} fill={l.color}
        text-anchor={l.anchor} letter-spacing={spacing} opacity={a}>{text}</text>)
}

pub fn labels<'a>(t: f32, ls: &[Label]) -> Svgr<'a> {
    let v: Vec<Svgr> = ls.iter().map(|l| label(t, l)).collect();
    fframes::svgr!(<g>{v}</g>)
}

/// Thin corner brackets around a rectangle (used to mark a region of interest without boxing it in).
pub fn brackets<'a>(r: [f32; 4], len: f32, color: &'static str, opacity: f32, width: f32) -> Svgr<'a> {
    let [x, y, w, h] = r;
    let d = format!(
        "M{} {}h{}M{} {}v{}M{} {}h{}M{} {}v{}M{} {}h{}M{} {}v{}M{} {}h{}M{} {}v{}",
        x, y, len, x, y, len, x + w, y, -len, x + w, y, len, x, y + h, len, x, y + h, -len, x + w, y + h, -len, x + w, y + h, -len
    );
    fframes::svgr!(<path d={d} stroke={color} stroke-width={width} fill="none" opacity={opacity} />)
}
