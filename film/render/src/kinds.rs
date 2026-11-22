//! The shot types and the overlays drawn over every frame (commit thread, power cut, vignette and grain).
use crate::Shot;
use crate::data::data;
use crate::draw::*;
use fframes::{FFramesContext, Frame, Svgr};

pub struct TitleLine {
    pub text: &'static str,
    pub size: f32,
    pub y: f32,
    pub t0: f32,
    pub color: &'static str,
    pub weight: u16,
    pub spacing: f32,
}

pub struct Step {
    pub src: Src,
    pub size: (f32, f32),
    pub crop: [f32; 4],
    /// scene second the wipe that reveals this step starts (the first step is shown from the start)
    pub at: f32,
    /// scene second the step's video starts playing
    pub start: f32,
}

pub struct Panel {
    pub src: Src,
    pub size: (f32, f32),
    pub crop0: [f32; 4],
    pub crop1: [f32; 4],
    pub dest: [f32; 4],
    pub t0: f32,
    pub t1: f32,
    pub start: f32,
    pub frame: bool,
    pub alpha: f32,
}

pub enum Kind {
    Black,
    Title(Vec<TitleLine>),
    Plate { src: Src, size: (f32, f32), crop0: [f32; 4], crop1: [f32; 4], start: f32 },
    Grid6 { tag: &'static str },
    DriveMap,
    /// the ground atlas revealed along the whole drive; `zoom` dives into the 1 cm crop at 128 s
    Atlas { zoom: Option<(f32, f32)>, reveal: f32 },
    Ladder { steps: Vec<Step>, wipe: f32 },
    /// a ladder confined to a rectangle (registered wipes inside a panel)
    LadderIn { steps: Vec<Step>, wipe: f32, dest: [f32; 4] },
    /// several kinds drawn in order
    Stack(Vec<Kind>),
    /// two registered sources split by a vertical divider that drifts from x0 to x1 (fractions of the width)
    Split { a: Step, b: Step, x0: f32, x1: f32 },
    /// horizontal bars (name, value) against a reference line, growing in turn
    Bars { rows: Vec<(&'static str, f32)>, max: f32, refv: f32, refl: &'static str, unit: &'static str, x: f32, y: f32, w: f32, t0: f32 },
    /// a column of pipeline stage names that light up one after another
    Stages { names: Vec<&'static str>, t0: f32, step: f32, x: f32, y: f32 },
    /// top-down junction: the car's path and every tracked vehicle as a box with its track ribbon (drive time t0..t1)
    Movers { t0: f32, t1: f32 },
    Panels(Vec<Panel>),
}

/// opacity of lettering drawn inside a shot's picture: hidden in a clean render
fn text_alpha() -> f32 {
    if crate::clean() { 0.0 } else { 1.0 }
}

pub fn full() -> [f32; 4] {
    [0.0, 0.0, W, H]
}

pub fn render<'a>(s: &'a Shot, frame: &Frame, ctx: &FFramesContext<'a, '_>, t: f32) -> Svgr<'a> {
    render_kind(s, &s.kind, frame, ctx, t)
}

fn render_kind<'a>(s: &'a Shot, kind: &'a Kind, frame: &Frame, ctx: &FFramesContext<'a, '_>, t: f32) -> Svgr<'a> {
    match kind {
        Kind::Black => Svgr::empty(),
        Kind::Title(lines) => title(t, lines),
        Kind::Plate { src, size, crop0, crop1, start } => {
            let p = clamp01((t + s.overlap) / (s.dur + s.overlap));
            let k = 0.5 * p + 0.5 * io(p);
            plate(frame, ctx, *src, *size, lerp4(*crop0, *crop1, k), full(), *start, s.id)
        }
        Kind::Grid6 { tag } => grid6(frame, ctx, t, tag, s.id),
        Kind::DriveMap => drive_map(ctx, t, s.dur),
        Kind::Atlas { zoom, reveal } => atlas(ctx, t, *zoom, *reveal),
        Kind::Ladder { steps, wipe } => ladder(frame, ctx, t, steps, *wipe, s.id, full()),
        Kind::LadderIn { steps, wipe, dest } => ladder(frame, ctx, t, steps, *wipe, s.id, *dest),
        Kind::Bars { rows, max, refv, refl, unit, x, y, w, t0 } => {
            if crate::clean() {
                return Svgr::empty();
            }
            let rowh = 96.0;
            let v: Vec<Svgr> = rows
                .iter()
                .enumerate()
                .map(|(i, (name, val))| {
                    let g = out(ramp(t, t0 + 0.12 * i as f32, 0.9));
                    let yy = y + i as f32 * rowh;
                    let bw = (w * val / max * g).max(0.5);
                    let txt = format!("{:.2} {unit}", val);
                    let tx = x + w + 40.0;
                    fframes::svgr!(<g opacity={io(ramp(t, t0 - 0.2 + 0.12 * i as f32, 0.4))}>
                        <text x={*x} y={yy + 30.0} font-family={MONO} font-size="36" fill={DIM} letter-spacing="4">{*name}</text>
                        <rect x={*x} y={yy + 46.0} width={bw} height="14" fill={FG} opacity="0.85" />
                        <text x={tx} y={yy + 60.0} font-family={MONO} font-size="36" fill={FG} letter-spacing="2" opacity={g}>{txt}</text>
                    </g>)
                })
                .collect();
            let rx = x + w * refv / max;
            let ra = io(ramp(t, t0 + 0.9, 0.6));
            let y1 = y + rows.len() as f32 * rowh;
            fframes::svgr!(<g>
                {v}
                <line x1={rx} y1={y - 30.0} x2={rx} y2={y1} stroke={ACCENT} stroke-width="3" stroke-dasharray="10 10" opacity={ra} />
                <text x={rx} y={y - 50.0} font-family={MONO} font-size="36" fill={ACCENT} text-anchor="middle" letter-spacing="4" opacity={ra}>{*refl}</text>
            </g>)
        }
        Kind::Stages { names, t0, step, x, y } => {
            if crate::clean() {
                return Svgr::empty();
            }
            let v: Vec<Svgr> = names
                .iter()
                .enumerate()
                .map(|(i, n)| {
                    let on = io(ramp(t, t0 + i as f32 * step, 0.25));
                    let yy = y + i as f32 * 62.0;
                    let a = 0.28 + 0.72 * on;
                    fframes::svgr!(<g opacity={a}>
                        <circle cx={*x} cy={yy - 13.0} r="7" fill={if on > 0.5 { ACCENT } else { "#ffffff" }} opacity={0.3 + 0.7 * on} />
                        <text x={x + 34.0} y={yy} font-family={MONO} font-size="38" fill={FG} letter-spacing="5">{*n}</text>
                    </g>)
                })
                .collect();
            let appear = io(ramp(t, 0.2, 0.8));
            fframes::svgr!(<g opacity={appear}>{v}</g>)
        }
        Kind::Split { a, b, x0, x1 } => {
            let p = clamp01((t + s.overlap) / (s.dur + s.overlap));
            let x = W * lerp(*x0, *x1, io(p));
            let pa = plate(frame, ctx, a.src, a.size, a.crop, full(), a.start, &format!("{}sa", s.id));
            let pb = plate(frame, ctx, b.src, b.size, b.crop, full(), b.start, &format!("{}sb", s.id));
            let cid = format!("{}split", s.id);
            let url = format!("url(#{cid})");
            fframes::svgr!(<g>
                {pa}
                <defs><clipPath id={cid}><rect x={x} y="0" width={(W - x).max(0.5)} height={H} /></clipPath></defs>
                <g clip-path={url}>{pb}</g>
                <rect x={x - 1.5} y="0" width="3" height={H} fill="#ffffff" opacity="0.85" />
            </g>)
        }
        Kind::Stack(ks) => {
            let v: Vec<Svgr> = ks.iter().map(|k| render_kind(s, k, frame, ctx, t)).collect();
            fframes::svgr!(<g>{v}</g>)
        }
        Kind::Movers { t0, t1 } => movers(ctx, t, s.dur, *t0, *t1),
        Kind::Panels(ps) => panels(frame, ctx, t, ps, s.id),
    }
}

fn title<'a>(t: f32, lines: &[TitleLine]) -> Svgr<'a> {
    let v: Vec<Svgr> = lines
        .iter()
        .map(|l| {
            let a = io(ramp(t, l.t0, 0.9));
            let dy = 14.0 * (1.0 - out(ramp(t, l.t0, 1.2)));
            fframes::svgr!(<text x="1920" y={l.y + dy} font-family={MONO} font-size={l.size} font-weight={l.weight} fill={l.color}
                text-anchor="middle" letter-spacing={l.spacing} opacity={a}>{l.text}</text>)
        })
        .collect();
    fframes::svgr!(<g>{v}</g>)
}

const CAM6: [(&str, &str); 6] = [
    ("left_pillar", "LEFT PILLAR"),
    ("front", "FRONT"),
    ("right_pillar", "RIGHT PILLAR"),
    ("left_repeater", "LEFT REPEATER"),
    ("back", "BACK"),
    ("right_repeater", "RIGHT REPEATER"),
];

pub fn grid6<'a>(frame: &Frame, ctx: &FFramesContext<'a, '_>, t: f32, tag: &str, id: &str) -> Svgr<'a> {
    let gap = 28.0;
    let vgap = 110.0;
    let tw = (W - 2.0 * MX - 2.0 * gap) / 3.0;
    let th = tw * 938.0 / 1448.0;
    let y0 = (H - (2.0 * th + vgap)) / 2.0 - 40.0;
    let order = [1usize, 0, 2, 3, 5, 4];
    let txt = text_alpha();
    let v: Vec<Svgr> = CAM6
        .iter()
        .enumerate()
        .map(|(i, (cam, name))| {
            let (col, row) = ((i % 3) as f32, (i / 3) as f32);
            let r = [MX + col * (tw + gap), y0 + row * (th + vgap), tw, th];
            let k = order.iter().position(|&o| o == i).unwrap_or(0) as f32;
            let a = io(ramp(t, -0.45 + 0.09 * k, 0.5));
            let file: &'static str = Box::leak(format!("{tag}-{cam}.mp4").into_boxed_str());
            let pid = format!("{id}{i}");
            let p = plate(frame, ctx, Src::V(file), (1448.0, 938.0), [0.0, 0.0, 1448.0, 938.0], r, 0.0, &pid);
            let ly = r[1] + th + 58.0;
            fframes::svgr!(<g opacity={a}>
                {p}
                <rect x={r[0]} y={r[1]} width={r[2]} height={r[3]} fill="none" stroke="#ffffff" stroke-opacity="0.10" stroke-width="2" />
                <text x={r[0]} y={ly} font-family={MONO} font-size="40" fill={DIM} letter-spacing="5" opacity={txt}>{*name}</text>
            </g>)
        })
        .collect();
    fframes::svgr!(<g>{v}</g>)
}

/// world metres -> screen for the whole-drive map (north up), shared by the drive and atlas shots so they register
pub struct MapView {
    pub k: f32,
    pub wx: f32,
    pub wy: f32,
    pub cx: f32,
    pub cy: f32,
}
impl MapView {
    pub fn base() -> Self {
        // bbox of the refined trajectory, padded by the 44 m ground band
        MapView { k: 1.12, wx: -753.0, wy: 348.0, cx: 1920.0, cy: 960.0 }
    }
    pub fn p(&self, x: f32, y: f32) -> (f32, f32) {
        (self.cx + (x - self.wx) * self.k, self.cy - (y - self.wy) * self.k)
    }
}

fn path_d(view: &MapView, t_end: f32) -> (String, (f32, f32)) {
    let d = &data().drive;
    let n = d.index_at(t_end);
    let mut s = String::with_capacity(n * 16);
    let mut last = (0.0, 0.0);
    let mut prev = (f32::NAN, f32::NAN);
    for i in 0..=n {
        let (x, y) = view.p(d.x[i], d.y[i]);
        if i > 0 && (x - prev.0).abs() + (y - prev.1).abs() < 1.5 {
            continue;
        }
        s.push_str(if i == 0 { "M" } else { "L" });
        s.push_str(&format!("{x:.1} {y:.1}"));
        prev = (x, y);
    }
    let (hx, hy, _, _) = d.at(t_end);
    let (x, y) = view.p(hx, hy);
    s.push_str(&format!("L{x:.1} {y:.1}"));
    last.0 = x;
    last.1 = y;
    (s, last)
}

fn drive_map<'a>(_ctx: &FFramesContext<'a, '_>, t: f32, dur: f32) -> Svgr<'a> {
    let d = &data().drive;
    let view = MapView::base();
    let p = io(ramp(t, 0.4, dur - 2.0));
    // the first 81 s are parked at the drive-through window: spend only 12 % of the reveal there (the clock still shows real time)
    let tt = if p < 0.12 { p / 0.12 * 81.0 } else { 81.0 + (p - 0.12) / 0.88 * (d.duration - 81.0) };
    let (path, head) = path_d(&view, tt.max(0.05));
    let grid: Vec<Svgr> = (0..40)
        .map(|i| {
            let x = view.p(-1700.0 + i as f32 * 100.0, 0.0).0;
            let y = view.p(0.0, -400.0 + i as f32 * 100.0).1;
            fframes::svgr!(<g>
                <line x1={x} y1="0" x2={x} y2={H} stroke="#ffffff" stroke-opacity="0.035" stroke-width="1.5" />
                <line x1="0" y1={y} x2={W} y2={y} stroke="#ffffff" stroke-opacity="0.035" stroke-width="1.5" />
            </g>)
        })
        .collect();
    // speed trace along the bottom
    let gx0 = MX;
    let gx1 = W - MX;
    let gy = 1880.0;
    let gh = 120.0;
    let n = d.index_at(tt);
    let mut sp = String::new();
    let step = (n / 700).max(1);
    for i in (0..=n).step_by(step) {
        let x = gx0 + (gx1 - gx0) * d.t[i] / d.duration;
        let y = gy - gh * d.speed[i] / 22.5;
        sp.push_str(&format!("{}{x:.1} {y:.1}", if i == 0 { "M" } else { "L" }));
    }
    let (_, _, _, v) = d.at(tt);
    let hx = gx0 + (gx1 - gx0) * tt / d.duration;
    let hy = gy - gh * v / 22.5;
    sp.push_str(&format!("L{hx:.1} {hy:.1}"));
    let mins = (tt / 60.0).floor();
    let secs = tt - mins * 60.0;
    let clock = format!("{}:{:02}", mins as i32, secs.floor() as i32);
    let mph = format!("{:.0} km/h", v * 3.6);
    let start = view.p(d.x[0], d.y[0]);
    let a = io(ramp(t, 0.0, 0.8));
    let txt = text_alpha();
    fframes::svgr!(<g opacity={a}>
        <defs>
            <filter id="glow" x="-20%" y="-20%" width="140%" height="140%"><feGaussianBlur stdDeviation="9" /></filter>
        </defs>
        {grid}
        <path d={path.clone()} stroke={ACCENT} stroke-width="22" fill="none" stroke-linecap="round" stroke-linejoin="round" opacity="0.45" filter={if tt > 82.0 { "url(#glow)" } else { "none" }} />
        <path d={path} stroke={ACCENT} stroke-width="7" fill="none" stroke-linecap="round" stroke-linejoin="round" />
        <circle cx={start.0} cy={start.1} r="9" fill="none" stroke={DIM} stroke-width="3" />
        <circle cx={head.0} cy={head.1} r="13" fill={FG} />
        <circle cx={head.0} cy={head.1} r="30" fill="none" stroke={FG} stroke-opacity="0.35" stroke-width="2" />
        <line x1={gx0} y1={gy} x2={gx1} y2={gy} stroke="#ffffff" stroke-opacity="0.12" stroke-width="2" />
        <path d={sp} stroke={FG} stroke-opacity="0.75" stroke-width="3" fill="none" stroke-linejoin="round" />
        <circle cx={hx} cy={hy} r="7" fill={FG} />
        <text x={gx0} y={gy + 58.0} font-family={MONO} font-size="38" fill={DIM} letter-spacing="5" opacity={txt}>"SPEED"</text>
        <text x={gx1} y={gy + 58.0} font-family={MONO} font-size="38" fill={DIM} letter-spacing="5" text-anchor="end" opacity={txt}>{mph}</text>
        <text x={MX} y="300" font-family={MONO} font-size="96" fill={FG} letter-spacing="4" opacity={txt}>{clock}</text>
        <text x={MX} y="370" font-family={MONO} font-size="38" fill={DIM} letter-spacing="5" opacity={txt}>"07:15:40 · JACKSONVILLE"</text>
    </g>)
}

/// a world-registered image (north up) placed through a map view
fn world_image<'a>(ctx: &FFramesContext<'a, '_>, view: &MapView, file: &str, bounds: [f32; 4], opacity: f32, mask: Option<&str>) -> Svgr<'a> {
    if opacity <= 0.001 {
        return Svgr::empty();
    }
    let Some(img) = ctx.get_image(file) else { return Svgr::empty() };
    let (x, y) = view.p(bounds[0], bounds[3]);
    let w = (bounds[2] - bounds[0]) * view.k;
    let h = (bounds[3] - bounds[1]) * view.k;
    match mask {
        Some(m) => fframes::svgr!(<g mask={m.to_string()} opacity={opacity}><image href={img.href()} x={x} y={y} width={w} height={h} preserveAspectRatio="none" /></g>),
        None => fframes::svgr!(<g opacity={opacity}><image href={img.href()} x={x} y={y} width={w} height={h} preserveAspectRatio="none" /></g>),
    }
}

const ATLAS_4CM: [f32; 4] = [-436.8, 122.8, -283.2, 209.2];
const ATLAS_1CM: [f32; 4] = [-381.2, 155.2, -342.8, 176.8];
const ZOOM_TARGET: (f32, f32) = (-362.0, 166.0);

fn atlas<'a>(ctx: &FFramesContext<'a, '_>, t: f32, zoom: Option<(f32, f32)>, reveal: f32) -> Svgr<'a> {
    let d = &data().drive;
    let g = &data().ground;
    let mut view = MapView::base();
    let mut zs = 0.0;
    if let Some((z0, zd)) = zoom {
        zs = io(ramp(t, z0, zd));
        let k1 = 100.0;
        view.k = view.k * (k1 / view.k).powf(zs);
        let c = 1.0 - (1.0 - ramp(t, z0, zd * 0.75)).powi(3);
        view.wx = lerp(view.wx, ZOOM_TARGET.0, c);
        view.wy = lerp(view.wy, ZOOM_TARGET.1, c);
        view.cy = lerp(view.cy, 1080.0, c);
    }
    let p = io(ramp(t, 0.3, reveal));
    let tt = d.t_at_dist(p * d.length_m).max(0.05);
    let (path, head) = path_d(&view, tt);
    let band = 100.0 * view.k;
    let blur = (14.0 * view.k / 1.18).min(400.0);
    let base = world_image(ctx, &view, "ground-v5-0.2.png", g.bounds, 1.0, Some("url(#reveal)"));
    let mid = world_image(ctx, &view, "atlas128-4cm.png", ATLAS_4CM, ramp(view.k, 4.0, 3.0), None);
    let fine = world_image(ctx, &view, "atlas128-1cm.png", ATLAS_1CM, ramp(view.k, 18.0, 10.0), None);
    let line_a = 0.85 * (1.0 - ramp(view.k, 3.0, 4.0));
    let a = io(ramp(t, 0.0, 0.6));
    fframes::svgr!(<g opacity={a}>
        <defs>
            <filter id="soft" x="-10%" y="-10%" width="120%" height="120%"><feGaussianBlur stdDeviation={blur} /></filter>
            <mask id="reveal" maskUnits="userSpaceOnUse" x="-20000" y="-20000" width="60000" height="60000">
                <rect x="-20000" y="-20000" width="60000" height="60000" fill="black" />
                <path d={path.clone()} stroke="white" stroke-width={band} fill="none" stroke-linecap="round" stroke-linejoin="round" filter={if tt > 82.0 { "url(#soft)" } else { "none" }} />
            </mask>
        </defs>
        {base}
        {mid}
        {fine}
        <path d={path} stroke={ACCENT} stroke-width="4" fill="none" stroke-opacity={line_a} stroke-linecap="round" stroke-linejoin="round" />
        <circle cx={head.0} cy={head.1} r="11" fill={FG} opacity={1.0 - zs} />
    </g>)
}

const JUNCTION: [f32; 4] = [-1680.0, 780.0, -1364.0, 957.75];

fn boxpoly(view: &MapView, x: f32, y: f32, yaw: f32, l0: f32, l1: f32, w: f32) -> String {
    let (c, s) = (yaw.cos(), yaw.sin());
    let pts = [(l1, w / 2.0), (l1, -w / 2.0), (l0, -w / 2.0), (l0, w / 2.0)];
    pts.iter()
        .map(|&(a, b)| {
            let (px, py) = view.p(x + a * c - b * s, y + a * s + b * c);
            format!("{px:.1},{py:.1}")
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn movers<'a>(ctx: &FFramesContext<'a, '_>, t: f32, dur: f32, t0: f32, t1: f32) -> Svgr<'a> {
    let dd = data();
    let view = MapView { k: W / (JUNCTION[2] - JUNCTION[0]) * 1.02, wx: -1522.0, wy: 868.0, cx: 1920.0, cy: 1080.0 };
    let tt = lerp(t0, t1, clamp01(t / dur));
    let img = world_image(ctx, &view, "junction-8cm.png", JUNCTION, 1.0, None);
    // ego trail and car
    let d = &dd.drive;
    let (i0, i1) = (d.index_at(t0), d.index_at(tt));
    let mut trail = String::new();
    for i in i0..=i1 {
        let (x, y) = view.p(d.x[i], d.y[i]);
        trail.push_str(&format!("{}{x:.1} {y:.1}", if i == i0 { "M" } else { "L" }));
    }
    let (ex, ey, eyaw, _) = d.at(tt);
    let (hx, hy) = view.p(ex, ey);
    trail.push_str(&format!("L{hx:.1} {hy:.1}"));
    let ego = boxpoly(&view, ex, ey, eyaw, -0.97, 3.75, 1.85);
    let mut els: Vec<Svgr> = Vec::new();
    for m in &dd.movers {
        let parked = m.t.len() < 2 || m.t[m.t.len() - 1] - m.t[0] < 0.2;
        if parked {
            let (x, y, yaw) = (m.x[0], m.y[0], m.yaw[0]);
            let (px, py) = view.p(x, y);
            if !(0.0..W).contains(&px) || !(0.0..H).contains(&py) { continue; }
            let poly = boxpoly(&view, x, y, yaw, -m.size[0] / 2.0, m.size[0] / 2.0, m.size[1]);
            els.push(fframes::svgr!(<polygon points={poly} fill="#ffffff" fill-opacity="0.06" stroke="#ffffff" stroke-opacity="0.35" stroke-width="2" />));
            continue;
        }
        if tt < m.t[0] - 0.01 { continue; }
        let tail0 = m.t[0].max(tt - 7.0);
        let tail1 = tt.min(*m.t.last().unwrap());
        if tail1 <= tail0 { continue; }
        let mut s = String::new();
        let mut n = 0;
        for k in 0..m.t.len() {
            if m.t[k] < tail0 || m.t[k] > tail1 { continue; }
            let (px, py) = view.p(m.x[k], m.y[k]);
            s.push_str(&format!("{}{px:.1} {py:.1}", if n == 0 { "M" } else { "L" }));
            n += 1;
        }
        let fade = 1.0 - ramp(tt, *m.t.last().unwrap(), 2.5);
        if n > 1 {
            els.push(fframes::svgr!(<path d={s} stroke="#ffffff" stroke-opacity={0.55 * fade} stroke-width="4" fill="none" stroke-linecap="round" stroke-linejoin="round" />));
        }
        if let Some((x, y, yaw, pred)) = m.at(tt) {
            let poly = boxpoly(&view, x, y, yaw, -m.size[0] / 2.0, m.size[0] / 2.0, m.size[1]);
            let (px, py) = view.p(x, y);
            let label = if !crate::clean() && px > 200.0 && px < W - 500.0 && py > 250.0 && py < H - 300.0 { format!("{} {}", m.cls.to_uppercase(), m.id) } else { String::new() };
            let dash = if pred { "10 8" } else { "none" };
            els.push(fframes::svgr!(<g>
                <polygon points={poly} fill="#ffffff" fill-opacity="0.22" stroke="#ffffff" stroke-width="3" stroke-dasharray={dash} />
                <text x={px + 40.0} y={py - 30.0} font-family={MONO} font-size="30" fill={FG} letter-spacing="3" opacity="0.85">{label}</text>
            </g>));
        }
    }
    let clock = format!("T + {:.1} S", tt);
    let a = io(ramp(t, 0.0, 0.6));
    fframes::svgr!(<g opacity={a}>
        <defs><filter id="mglow" x="-20%" y="-20%" width="140%" height="140%"><feGaussianBlur stdDeviation="8" /></filter></defs>
        {img}
        <rect x="0" y="0" width={W} height={H} fill="#000000" opacity="0.38" />
        <path d={trail.clone()} stroke={ACCENT} stroke-width="26" fill="none" opacity="0.4" filter="url(#mglow)" stroke-linecap="round" />
        <path d={trail} stroke={ACCENT} stroke-width="9" fill="none" stroke-linecap="round" stroke-linejoin="round" />
        {els}
        <polygon points={ego} fill={FG} stroke={FG} stroke-width="2" />
        <text x={W - MX} y="300" font-family={MONO} font-size="60" fill={FG} letter-spacing="6" text-anchor="end" opacity={text_alpha()}>{clock}</text>
    </g>)
}

fn ladder<'a>(frame: &Frame, ctx: &FFramesContext<'a, '_>, t: f32, steps: &[Step], wipe: f32, id: &str, dest: [f32; 4]) -> Svgr<'a> {
    let cur = steps.iter().rposition(|s| s.at <= t).unwrap_or(0);
    let base_i = if cur > 0 && t < steps[cur].at + wipe { cur - 1 } else { cur };
    let b = &steps[base_i];
    let base = plate(frame, ctx, b.src, b.size, b.crop, dest, b.start, &format!("{id}b{base_i}"));
    if base_i == cur {
        return base;
    }
    let s = &steps[cur];
    let p = io(ramp(t, s.at, wipe));
    let [dx, dy, dw, dh] = dest;
    let x = dx + (dw * p).max(0.5);
    let inc = plate(frame, ctx, s.src, s.size, s.crop, dest, s.start, &format!("{id}i{cur}"));
    let cid = format!("{id}wipe{cur}");
    let url = format!("url(#{cid})");
    let gid = format!("{id}glow{cur}");
    let gurl = format!("url(#{gid})");
    let gw = (dw * 0.04).min(140.0);
    fframes::svgr!(<g>
        {base}
        <defs><clipPath id={cid}><rect x={dx} y={dy} width={x - dx} height={dh} /></clipPath>
            <linearGradient id={gid} x1="0" x2="1" y1="0" y2="0">
                <stop offset="0" stop-color="#ffffff" stop-opacity="0" />
                <stop offset="1" stop-color="#ffffff" stop-opacity="0.22" />
            </linearGradient></defs>
        <g clip-path={url}>{inc}</g>
        <rect x={(x - gw).max(dx)} y={dy} width={gw.min(x - dx).max(0.5)} height={dh} fill={gurl} />
        <rect x={x - 1.5} y={dy} width="3" height={dh} fill="#ffffff" opacity={0.9 * (1.0 - ramp(p, 0.96, 0.04))} />
    </g>)
}

#[allow(dead_code)]
fn ladder_old<'a>(frame: &Frame, ctx: &FFramesContext<'a, '_>, t: f32, steps: &[Step], wipe: f32, id: &str) -> Svgr<'a> {
    let cur = steps.iter().rposition(|s| s.at <= t).unwrap_or(0);
    let base_i = if cur > 0 && t < steps[cur].at + wipe { cur - 1 } else { cur };
    let b = &steps[base_i];
    let base = plate(frame, ctx, b.src, b.size, b.crop, full(), b.start, &format!("{id}b{base_i}"));
    if base_i == cur {
        return base;
    }
    let s = &steps[cur];
    let p = io(ramp(t, s.at, wipe));
    let x = -60.0 + (W + 120.0) * p;
    let inc = plate(frame, ctx, s.src, s.size, s.crop, full(), s.start, &format!("{id}i{cur}"));
    let cid = format!("{id}wipe{cur}");
    let url = format!("url(#{cid})");
    fframes::svgr!(<g>
        {base}
        <defs><clipPath id={cid}><rect x="0" y="0" width={x.max(0.5)} height={H} /></clipPath>
            <linearGradient id="wipeglow" x1="0" x2="1" y1="0" y2="0">
                <stop offset="0" stop-color="#ffffff" stop-opacity="0" />
                <stop offset="1" stop-color="#ffffff" stop-opacity="0.22" />
            </linearGradient></defs>
        <g clip-path={url}>{inc}</g>
        <rect x={x - 140.0} y="0" width="140" height={H} fill="url(#wipeglow)" />
        <rect x={x - 1.5} y="0" width="3" height={H} fill="#ffffff" opacity="0.9" />
    </g>)
}

fn panels<'a>(frame: &Frame, ctx: &FFramesContext<'a, '_>, t: f32, ps: &[Panel], id: &str) -> Svgr<'a> {
    let v: Vec<Svgr> = ps
        .iter()
        .enumerate()
        .map(|(i, p)| {
            // a panel that starts with the shot is already up during the incoming cross-fade
            let t0 = if p.t0 <= 0.01 { -2.0 } else { p.t0 };
            if t < t0 - 0.01 || t > p.t1 + 0.01 {
                return Svgr::empty();
            }
            let a = window(t, t0, p.t1, 0.5, 0.35) * p.alpha;
            let k = clamp01((t - p.t0) / (p.t1 - p.t0).max(0.1));
            let crop = lerp4(p.crop0, p.crop1, 0.5 * k + 0.5 * io(k));
            let body = plate(frame, ctx, p.src, p.size, crop, p.dest, p.start, &format!("{id}p{i}"));
            let [x, y, w, h] = p.dest;
            let border = if p.frame {
                fframes::svgr!(<rect x={x} y={y} width={w} height={h} fill="none" stroke="#ffffff" stroke-opacity="0.12" stroke-width="2" />)
            } else {
                Svgr::empty()
            };
            fframes::svgr!(<g opacity={a}>{body}{border}</g>)
        })
        .collect();
    fframes::svgr!(<g>{v}</g>)
}

/// Quiet commit timeline along the bottom edge: 53 ticks from 15:45 to 04:15, passed commits lit.
pub fn thread<'a>(s: &Shot, st: f32, prev: Option<f32>) -> Svgr<'a> {
    let Some((m0, m1)) = s.thread else { return Svgr::empty() };
    // glide from where the previous shot left the playhead; fade in when the previous shot had no thread
    let glide = io(ramp(st, 0.0, 0.7));
    let fade = if prev.is_none() { io(ramp(st, 0.0, 0.6)) } else { 1.0 };
    let commits = &data().commits;
    let span = 750.0;
    let x0 = MX;
    let x1 = W - MX;
    let y = H - 118.0;
    let mut ph = m0 + (m1 - m0) * clamp01(st / s.dur);
    if let Some(p) = prev {
        ph = lerp(p, ph, glide);
    }
    let xp = x0 + (x1 - x0) * ph / span;
    let ticks: Vec<Svgr> = commits
        .iter()
        .map(|c| {
            let x = x0 + (x1 - x0) * c.minute / span;
            let passed = c.minute <= ph + 0.01;
            let fresh = passed && ph - c.minute < 6.0 && c.minute >= m0 - 0.01;
            let (col, h, op) = if fresh { (ACCENT, 26.0, 1.0) } else if passed { (FG, 16.0, 0.55) } else { ("#ffffff", 12.0, 0.16) };
            fframes::svgr!(<rect x={x - 1.5} y={y - h} width="3" height={h} fill={col} opacity={op} />)
        })
        .collect();
    let mins = (945.0 + ph).round() as i32;
    let now = format!("{:02}:{:02}", (mins / 60) % 24, mins % 60);
    fframes::svgr!(<g opacity={0.9 * fade}>
        <line x1={x0} y1={y} x2={x1} y2={y} stroke="#ffffff" stroke-opacity="0.14" stroke-width="2" />
        <line x1={x0} y1={y} x2={xp} y2={y} stroke="#ffffff" stroke-opacity="0.38" stroke-width="2" />
        {ticks}
        <circle cx={xp} cy={y} r="7" fill={ACCENT} />
        <text x={xp} y={y + 52.0} font-family={MONO} font-size="34" fill={DIM} text-anchor="middle" letter-spacing="3">{now}</text>
    </g>)
}

/// The 21:19 power cut: the frame drops out, flickers and comes back.
pub fn power_cut<'a>(s: &Shot, st: f32) -> Svgr<'a> {
    let Some(pc) = s.power else { return Svgr::empty() };
    let u = st - pc;
    if !(0.0..2.2).contains(&u) {
        return Svgr::empty();
    }
    let dark = if u < 0.07 {
        u / 0.07 * 0.92
    } else if u < 0.42 {
        if ((u * 30.0) as i32) % 3 == 0 { 0.55 } else { 0.92 }
    } else if u < 1.5 {
        0.96
    } else {
        0.96 * (1.0 - io((u - 1.5) / 0.7))
    };
    let la = window(u, 0.45, 1.75, 0.15, 0.3) * text_alpha();
    fframes::svgr!(<g>
        <rect x="0" y="0" width={W} height={H} fill="#000000" opacity={dark} />
        <text x="1920" y="1100" font-family={MONO} font-size="52" fill={FG} text-anchor="middle" letter-spacing="8" opacity={la}>"21:19 · POWER CUT"</text>
    </g>)
}

/// Finish over every frame: soft vignette and fine moving grain.
pub fn finish<'a>(ctx: &FFramesContext<'a, '_>, index: usize) -> Svgr<'a> {
    let Some(grain) = ctx.get_image("grain.png") else { return Svgr::empty() };
    let href = grain.href();
    let h = (index as u64).wrapping_mul(2654435761) ^ 0x9e3779b9;
    let ox = -((h % 1024) as f32);
    let oy = -(((h >> 12) % 1024) as f32);
    let tiles: Vec<Svgr> = (0..5)
        .flat_map(|i| (0..4).map(move |j| (i, j)))
        .map(|(i, j)| {
            let x = ox + i as f32 * 1024.0;
            let y = oy + j as f32 * 1024.0;
            fframes::svgr!(<image href={href.clone()} x={x} y={y} width="1024" height="1024" />)
        })
        .collect();
    fframes::svgr!(<g>
        <defs>
            <radialGradient id="vig" cx="50%" cy="50%" r="75%">
                <stop offset="0.55" stop-color="#000000" stop-opacity="0" />
                <stop offset="1" stop-color="#000000" stop-opacity="0.55" />
            </radialGradient>
        </defs>
        <rect x="0" y="0" width="3840" height="2160" fill="url(#vig)" />
        <g opacity="0.07" style="mix-blend-mode:overlay">{tiles}</g>
    </g>)
}
