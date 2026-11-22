#!/usr/bin/env bun
// Writes a synthetic Reality Splat scene that follows the viewer scene contract in PLAN.md:
//   scenes/synth/drive.json   copy of the real drive
//   scenes/synth/rig.json     copy of the real rig (scenes/dev/rig.json), also used for the visibility model
//   scenes/synth/map.json     {t_end, tiles: [{id, kind, file, count, bbox, t_min, t_max}], sky}
//   scenes/synth/sky.png      equirectangular overcast panorama (centre column faces east, up is +z)
//   scenes/synth/tiles/*.ply  3DGS PLY + t_birth t_death e0 s1 e1 s2 e2 (seconds, unused = 1e9)
//
// World: a road band along the real trajectory (lanes, markings, curbs, sidewalks, verges), a car park with a
// drive-through at the start, intersections with crosswalks and signals, street lights, trees, buildings,
// parked cars, and moving cars as short-lived transient splats.
//
// Time model, mimicking the live map: every surface is a quadtree / octree of up to four generations whose
// footprints halve (about 1.6, 0.8, 0.4, 0.2 m cells). Generation 0 is born when a camera first sees the place;
// generation g+1 is born when a camera sees it from within RADII[g+1] metres, and the parent then dies 0.25 s
// later (cross-fade overlap). Seen-runs come from the six cameras of rig.json (range, fisheye field of view, a
// body-occlusion box; no scene occlusion), sampled at 5 Hz.
//
// Usage: bun tools/make-synthetic-scene.ts [--src ../scenes/dev] [--out ../scenes/synth] [--density 1]

import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import path from "node:path";
import { deflateSync } from "node:zlib";
import { Drive, type DriveJson, wrapAngle } from "../src/scene/drive";
import { cameraGeometry, defaultRig, inImage, type RigJson } from "../src/scene/rig";

const args = process.argv.slice(2);
const arg = (k: string, d: string) => {
  const i = args.indexOf(k);
  return i >= 0 && args[i + 1] ? args[i + 1] : d;
};
const ROOT = path.resolve(import.meta.dir, "../..");
const SRC = path.resolve(arg("--src", path.join(ROOT, "scenes/dev")));
const OUT = path.resolve(arg("--out", path.join(ROOT, "scenes/synth")));
const DENSITY = Number(arg("--density", "1"));
const TILE = 64;
const TRANSIENT_BUCKET_S = 30;
const NEVER = 1e9;
const OVERLAP = 0.25;
const GEN0 = 1.6 / Math.sqrt(DENSITY);
// a generation is born when seen from within this distance of a camera (index = generation; 4 = crisp markings)
const RADII = [Infinity, 40, 16, 6, 10];
const T0 = performance.now();
const log = (...a: unknown[]) => console.log("[" + ((performance.now() - T0) / 1000).toFixed(1) + "s]", ...a);

// ---------------------------------------------------------------- random + noise
let seed = 1234567;
function rand() {
  seed |= 0;
  seed = (seed + 0x6d2b79f5) | 0;
  let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}
const rr = (a: number, b: number) => a + (b - a) * rand();
const gauss = () => (rand() + rand() + rand() - 1.5) * 1.15;
function hash2(ix: number, iy: number) {
  let h = Math.imul(ix, 374761393) + Math.imul(iy, 668265263);
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}
function vnoise(x: number, y: number) {
  const ix = Math.floor(x), iy = Math.floor(y);
  const fx = x - ix, fy = y - iy;
  const sx = fx * fx * (3 - 2 * fx), sy = fy * fy * (3 - 2 * fy);
  const a = hash2(ix, iy), b = hash2(ix + 1, iy), c = hash2(ix, iy + 1), d = hash2(ix + 1, iy + 1);
  return a + (b - a) * sx + (c - a) * sy + (a - b - c + d) * sx * sy;
}
const fbm = (x: number, y: number) => 0.55 * vnoise(x, y) + 0.3 * vnoise(x * 2.1 + 17, y * 2.1 + 9) + 0.15 * vnoise(x * 4.3 + 3, y * 4.3 + 41);
type RGB = [number, number, number];

// ---------------------------------------------------------------- splat store
// x y z | sx sy sz (sigma, m) | qw qx qy qz | opacity | r g b (display 0-1) | t_birth t_death e0 s1 e1 s2 e2
const STRIDE = 21;
class Splats {
  data = new Float32Array(STRIDE * 2_000_000);
  n = 0;
  alloc() {
    if ((this.n + 1) * STRIDE > this.data.length) {
      const d = new Float32Array(this.data.length * 2);
      d.set(this.data);
      this.data = d;
    }
    return this.n++ * STRIDE;
  }
}
const S = new Splats();
const TINT: RGB = [0.95, 1.0, 1.035]; // dawn overcast, teal ISP tint
const clamp01 = (v: number) => Math.min(1, Math.max(0, v));

// quaternion (w,x,y,z) from orthonormal axes (columns)
function quatFromAxes(ax: number[], ay: number[], az: number[]): [number, number, number, number] {
  const m00 = ax[0], m10 = ax[1], m20 = ax[2], m01 = ay[0], m11 = ay[1], m21 = ay[2], m02 = az[0], m12 = az[1], m22 = az[2];
  const tr = m00 + m11 + m22;
  if (tr > 0) { const s = Math.sqrt(tr + 1) * 2; return [0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s]; }
  if (m00 > m11 && m00 > m22) { const s = Math.sqrt(1 + m00 - m11 - m22) * 2; return [(m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s]; }
  if (m11 > m22) { const s = Math.sqrt(1 + m11 - m00 - m22) * 2; return [(m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s]; }
  const s = Math.sqrt(1 + m22 - m00 - m11) * 2;
  return [(m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s];
}
const quatYaw = (a: number): [number, number, number, number] => [Math.cos(a / 2), 0, 0, Math.sin(a / 2)];
function quatRandom(): [number, number, number, number] {
  const u1 = rand(), u2 = rand() * 2 * Math.PI, u3 = rand() * 2 * Math.PI;
  const a = Math.sqrt(1 - u1), b = Math.sqrt(u1);
  return [a * Math.sin(u2), a * Math.cos(u2), b * Math.sin(u3), b * Math.cos(u3)];
}

// ---------------------------------------------------------------- drive, rig and path
const driveJson = JSON.parse(readFileSync(path.join(SRC, "drive.json"), "utf8")) as DriveJson;
const drive = new Drive(driveJson);
const rigPath = path.join(SRC, "rig.json");
const rig: RigJson = existsSync(rigPath) ? JSON.parse(readFileSync(rigPath, "utf8")) : defaultRig();
const cams = cameraGeometry(rig);
log("drive: " + drive.n + " samples, " + drive.duration.toFixed(1) + " s, " + drive.length.toFixed(0) + " m; rig: " + (existsSync(rigPath) ? rigPath : "default") + " (" + cams.map((c) => c.name + " " + c.hfovDeg.toFixed(0) + "x" + c.vfovDeg.toFixed(0)).join(", ") + ")");

const DS = 0.5;
const NP = Math.floor(drive.length / DS) + 1;
const px = new Float64Array(NP), py = new Float64Array(NP), ph = new Float64Array(NP);
for (let i = 0; i < NP; i++) {
  const p = drive.atS(i * DS);
  px[i] = p.x; py[i] = p.y;
}
for (let i = 0; i < NP; i++) {
  const a = Math.max(0, i - 2), b = Math.min(NP - 1, i + 2);
  ph[i] = Math.atan2(py[b] - py[a], px[b] - px[a]);
}
const turnSig = new Float64Array(NP);
const W = 24;
for (let i = 0; i < NP; i++) turnSig[i] = wrapAngle(ph[Math.min(NP - 1, i + W)] - ph[Math.max(0, i - W)]);
const LOT_END_S = drive.state(108).s;
type Turn = { i: number; s: number; hin: number; hout: number };
const turns: Turn[] = [];
for (let i = W; i < NP - W; i++) {
  if (i * DS < LOT_END_S + 10) continue;
  const v = Math.abs(turnSig[i]);
  if (v < 0.7) continue;
  let isMax = true;
  for (let k = Math.max(0, i - 40); k <= Math.min(NP - 1, i + 40); k++) if (Math.abs(turnSig[k]) > v) { isMax = false; break; }
  if (!isMax || (turns.length && i - turns[turns.length - 1].i < 60)) continue;
  turns.push({ i, s: i * DS, hin: ph[Math.max(0, i - W)], hout: ph[Math.min(NP - 1, i + W)] });
}
type Stop = { s: number; t0: number; t1: number };
const stops: Stop[] = [];
{
  let start = -1;
  for (let i = 0; i < drive.n; i++) {
    const still = drive.speed[i] < 0.3;
    if (still && start < 0) start = i;
    if ((!still || i === drive.n - 1) && start >= 0) {
      const t0 = drive.t[start], t1 = drive.t[i];
      if (t1 - t0 > 4 && t0 > 110) stops.push({ s: drive.s[start], t0, t1 });
      start = -1;
    }
  }
}
type Inter = { s: number; h: number; hc: number; turn: boolean };
const inters: Inter[] = turns.map((t) => ({ s: t.s, h: t.hin, hc: t.hout, turn: true }));
for (const st of stops) {
  const sI = st.s + 22;
  if (inters.some((q) => Math.abs(q.s - sI) < 40)) continue;
  const h = ph[Math.min(NP - 1, Math.round(sI / DS))];
  inters.push({ s: sI, h, hc: h + Math.PI / 2, turn: false });
}
inters.sort((a, b) => a.s - b.s);
const nearInter = (s: number, r: number) => inters.some((q) => Math.abs(q.s - s) < r);
const interAt = (s: number) => inters.find((q) => Math.abs(q.s - s) < 30);
log("path: " + NP + " samples, " + turns.length + " turns, " + stops.length + " stops, " + inters.length + " intersections, lot ends at s=" + LOT_END_S.toFixed(0) + " m");

// ---------------------------------------------------------------- raster: nearest path point for every 0.5 m cell
const CELL = 0.5;
const BAND = 42;
let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
for (let i = 0; i < NP; i++) {
  minX = Math.min(minX, px[i]); maxX = Math.max(maxX, px[i]);
  minY = Math.min(minY, py[i]); maxY = Math.max(maxY, py[i]);
}
minX -= BAND + 6; minY -= BAND + 6; maxX += BAND + 6; maxY += BAND + 6;
const GW = Math.ceil((maxX - minX) / CELL), GH = Math.ceil((maxY - minY) / CELL);
const gDist = new Float32Array(GW * GH).fill(1e18);
const gIdx = new Int32Array(GW * GH).fill(-1);
{
  const R = Math.ceil((BAND + 4) / CELL);
  for (let i = 0; i < NP; i++) {
    const cx = Math.floor((px[i] - minX) / CELL), cy = Math.floor((py[i] - minY) / CELL);
    for (let gy = Math.max(0, cy - R); gy <= Math.min(GH - 1, cy + R); gy++) {
      const wy = minY + (gy + 0.5) * CELL - py[i];
      const row = gy * GW;
      for (let gx = Math.max(0, cx - R); gx <= Math.min(GW - 1, cx + R); gx++) {
        const wx = minX + (gx + 0.5) * CELL - px[i];
        const d2 = wx * wx + wy * wy;
        if (d2 < gDist[row + gx]) { gDist[row + gx] = d2; gIdx[row + gx] = i; }
      }
    }
  }
  for (let k = 0; k < gDist.length; k++) gDist[k] = Math.sqrt(gDist[k]);
}
// cross roads at intersections (straight continuation of the incoming road and the crossing road)
type Seg = { x0: number; y0: number; x1: number; y1: number };
const crossSegs: Seg[] = [];
for (const q of inters) {
  const L = 46;
  const c = drive.atS(q.s);
  crossSegs.push({ x0: c.x, y0: c.y, x1: c.x + Math.cos(q.h) * L, y1: c.y + Math.sin(q.h) * L });
  crossSegs.push({ x0: c.x - Math.cos(q.hc) * L, y0: c.y - Math.sin(q.hc) * L, x1: c.x + Math.cos(q.hc) * L, y1: c.y + Math.sin(q.hc) * L });
}
const gCross = new Float32Array(GW * GH).fill(1e9);
for (const s of crossSegs) {
  const dx = s.x1 - s.x0, dy = s.y1 - s.y0, l2 = dx * dx + dy * dy;
  const xa = Math.min(s.x0, s.x1) - 20, xb = Math.max(s.x0, s.x1) + 20, ya = Math.min(s.y0, s.y1) - 20, yb = Math.max(s.y0, s.y1) + 20;
  for (let gy = Math.max(0, Math.floor((ya - minY) / CELL)); gy <= Math.min(GH - 1, Math.floor((yb - minY) / CELL)); gy++)
    for (let gx = Math.max(0, Math.floor((xa - minX) / CELL)); gx <= Math.min(GW - 1, Math.floor((xb - minX) / CELL)); gx++) {
      const x = minX + (gx + 0.5) * CELL, y = minY + (gy + 0.5) * CELL;
      const u = Math.max(0, Math.min(1, ((x - s.x0) * dx + (y - s.y0) * dy) / l2));
      const d = Math.hypot(s.x0 + dx * u - x, s.y0 + dy * u - y);
      const k = gy * GW + gx;
      if (d < gCross[k]) gCross[k] = d;
    }
}
log("raster " + GW + "x" + GH);
function cellAt(x: number, y: number) {
  const gx = Math.floor((x - minX) / CELL), gy = Math.floor((y - minY) / CELL);
  if (gx < 0 || gy < 0 || gx >= GW || gy >= GH) return -1;
  return gy * GW + gx;
}

// ---------------------------------------------------------------- road layout and ground appearance
// lateral offsets in metres, + = left of travel; the car drives the inner lane of its direction at 0
const PAVE_R = -5.4, PAVE_L = 9.0, CURB_R = -5.6, CURB_L = 9.2, SW_R0 = -8.0, SW_R1 = -9.8, SW_L0 = 11.6, SW_L1 = 13.4;
const LOT_HALF = 17;
const enum C { None, Asphalt, Lot, Curb, Grass, Sidewalk, Land, Mulch }
type G = { c: C; lat: number; s: number; idx: number; lot: boolean; cross: number };
const gtmp: G = { c: C.None, lat: 0, s: 0, idx: 0, lot: false, cross: 0 };
function ground(x: number, y: number): G {
  const k = cellAt(x, y);
  gtmp.c = C.None;
  if (k < 0 || gIdx[k] < 0) return gtmp;
  const i = gIdx[k];
  const fx = Math.cos(ph[i]), fy = Math.sin(ph[i]);
  const wx = x - px[i], wy = y - py[i];
  const lat = -wx * fy + wy * fx, s = i * DS + wx * fx + wy * fy, d = Math.hypot(wx, wy);
  const cross = gCross[k];
  const lot = s < LOT_END_S;
  gtmp.lat = lat; gtmp.s = s; gtmp.idx = i; gtmp.lot = lot; gtmp.cross = cross;
  if (d > BAND) return gtmp;
  const al = Math.abs(lat);
  if (lot) gtmp.c = al < LOT_HALF ? C.Lot : al < LOT_HALF + 0.3 ? C.Curb : al < LOT_HALF + 3 ? C.Mulch : C.Land;
  else if (cross < 7.5 || (lat > PAVE_R && lat < PAVE_L)) gtmp.c = C.Asphalt;
  else if (cross < 7.8 || (lat <= PAVE_R && lat > CURB_R) || (lat >= PAVE_L && lat < CURB_L)) gtmp.c = C.Curb;
  else if ((lat <= SW_R0 && lat > SW_R1) || (lat >= SW_L0 && lat < SW_L1) || cross < 10.5) gtmp.c = C.Sidewalk;
  else if ((lat <= CURB_R && lat > SW_R0) || (lat >= CURB_L && lat < SW_L0)) gtmp.c = C.Grass;
  else gtmp.c = C.Land;
  return gtmp;
}
// paint: 0 none, 1 white, 2 yellow
function paintAt(lat: number, s: number, lot: boolean): number {
  if (lot) {
    const al = Math.abs(lat);
    if (al > 7 && al < 12.5 && ((s % 2.8) + 2.8) % 2.8 < 0.12) return 1;
    return Math.abs(al - 7) < 0.06 ? 1 : 0;
  }
  const q = interAt(s);
  if (q) {
    const ds = s - q.s;
    if (lat > PAVE_R && lat < PAVE_L && Math.abs(Math.abs(ds) - 15.5) < 1.6) return ((lat + 100) % 1.2) < 0.6 ? 1 : 0;
    if (ds < 0 && Math.abs(ds + 18.2) < 0.25 && lat > PAVE_R + 0.2 && lat < 1.8) return 1;
    if (Math.abs(ds) < 17.2) return 0;
  }
  const dash = ((s % 12) + 12) % 12 < 3;
  if (Math.abs(lat - (PAVE_R + 0.25)) < 0.075 || Math.abs(lat - (PAVE_L - 0.25)) < 0.075) return 1;
  if ((Math.abs(lat + 1.8) < 0.075 || Math.abs(lat - 5.4) < 0.075) && dash) return 1;
  if (Math.abs(lat - 1.62) < 0.06 || Math.abs(lat - 1.98) < 0.06) return 2;
  return 0;
}
const COL: Record<number, RGB> = {
  [C.Asphalt]: [0.355, 0.37, 0.38], [C.Lot]: [0.4, 0.405, 0.41], [C.Curb]: [0.66, 0.66, 0.64], [C.Grass]: [0.36, 0.41, 0.28],
  [C.Sidewalk]: [0.63, 0.635, 0.615], [C.Land]: [0.33, 0.38, 0.26], [C.Mulch]: [0.36, 0.29, 0.22],
};
const WHITE: RGB = [0.84, 0.86, 0.84], YELLOW: RGB = [0.86, 0.68, 0.26];
const gcol: RGB = [0, 0, 0];
let gz = 0;
function groundColour(x: number, y: number): C {
  const g = ground(x, y);
  if (g.c === C.None) return C.None;
  const base = COL[g.c];
  let r = base[0], gg = base[1], b = base[2];
  const n1 = fbm(x * 0.08, y * 0.08);
  gz = 0;
  if (g.c === C.Asphalt || g.c === C.Lot) {
    const v = 0.86 + 0.22 * n1 + 0.08 * (vnoise(x * 3.1, y * 3.1) - 0.5);
    r *= v; gg *= v; b *= v;
    if (!g.lot) for (const lc of [-3.6, 0, 3.6, 7.2]) for (const w of [-0.85, 0.85]) {
      const dd = Math.abs(g.lat - lc - w);
      if (dd < 0.35) { const f = 1 - 0.08 * (1 - dd / 0.35); r *= f; gg *= f; b *= f; }
    }
    const p = paintAt(g.lat, g.s, g.lot);
    if (p) {
      const pc = p === 1 ? WHITE : YELLOW;
      const wear = 0.8 + 0.2 * vnoise(x * 1.7, y * 1.7);
      r += (pc[0] - r) * wear; gg += (pc[1] - gg) * wear; b += (pc[2] - b) * wear;
    }
  } else if (g.c === C.Sidewalk) {
    const v = 0.92 + 0.12 * n1;
    r *= v; gg *= v; b *= v;
    if (((g.s % 1.5) + 1.5) % 1.5 < 0.05) { r *= 0.8; gg *= 0.8; b *= 0.8; }
    gz = 0.13;
  } else if (g.c === C.Curb) {
    gz = 0.13;
  } else if (g.c === C.Grass || g.c === C.Land) {
    const v = 0.8 + 0.32 * n1 + 0.1 * (vnoise(x * 2.3, y * 2.3) - 0.5);
    r *= v; gg *= v; b *= v;
    const dry = Math.max(0, fbm(x * 0.05 + 7, y * 0.05 + 3) - 0.55) * 2.2;
    r += (0.46 - r) * dry; gg += (0.42 - gg) * dry; b += (0.3 - b) * dry;
    gz = 0.06 + 0.06 * n1;
  } else if (g.c === C.Mulch) {
    const v = 0.85 + 0.25 * vnoise(x * 2, y * 2);
    r *= v; gg *= v; b *= v;
    gz = 0.1;
  }
  gcol[0] = r; gcol[1] = gg; gcol[2] = b;
  return g.c;
}

// ---------------------------------------------------------------- visibility model
const RANGE: Record<string, number> = { front: 120, back: 45, left_pillar: 55, right_pillar: 55, left_repeater: 60, right_repeater: 60 };
const NC = cams.length;
const camRange2 = cams.map((c) => (RANGE[c.name] ?? 60) ** 2);
const camWide = cams.map((c) => c.wide);
const camTanH = cams.map((c) => c.tanH), camTanV = cams.map((c) => c.tanV), camThH = cams.map((c) => c.thH), camThV = cams.map((c) => c.thV);
type Pose = { x: number; y: number; yaw: number; t0: number; t1: number };
const poses: Pose[] = [];
for (let t = 0; t <= drive.duration + 1e-6; t += 0.2) {
  const st = drive.state(Math.min(t, drive.duration));
  const last = poses[poses.length - 1];
  if (last && Math.hypot(st.x - last.x, st.y - last.y) < 0.05 && Math.abs(st.yaw - last.yaw) < 0.003) { last.t1 = t; continue; }
  poses.push({ x: st.x, y: st.y, yaw: st.yaw, t0: t, t1: t });
}
const NPOSE = poses.length;
const camW = new Float64Array(NPOSE * NC * 12);
for (let k = 0; k < NPOSE; k++) {
  const p = poses[k], c = Math.cos(p.yaw), s = Math.sin(p.yaw);
  const rot = (v: number[]) => [v[0] * c - v[1] * s, v[0] * s + v[1] * c, v[2]];
  for (let j = 0; j < NC; j++) {
    const g = cams[j], o = (k * NC + j) * 12;
    const pp = rot(g.pos);
    camW[o] = p.x + pp[0]; camW[o + 1] = p.y + pp[1]; camW[o + 2] = pp[2];
    camW.set(rot(g.right), o + 3); camW.set(rot(g.down), o + 6); camW.set(rot(g.fwd), o + 9);
  }
}
const VG = 32;
const maxRange = Math.sqrt(Math.max(...camRange2)) + 6;
const vgrid = new Map<number, number[]>();
const vkey = (ix: number, iy: number) => ix * 100003 + iy;
for (let k = 0; k < NPOSE; k++) {
  const key = vkey(Math.floor(poses[k].x / VG), Math.floor(poses[k].y / VG));
  let a = vgrid.get(key);
  if (!a) vgrid.set(key, (a = []));
  a.push(k);
}
const candCache = new Map<number, Int32Array>();
function candidates(x: number, y: number) {
  const ix = Math.floor(x / VG), iy = Math.floor(y / VG), key = vkey(ix, iy);
  let c = candCache.get(key);
  if (c) return c;
  const R = Math.ceil(maxRange / VG) + 1;
  const list: number[] = [];
  for (let dy = -R; dy <= R; dy++) for (let dx = -R; dx <= R; dx++) {
    const a = vgrid.get(vkey(ix + dx, iy + dy));
    if (a) for (const k of a) list.push(k);
  }
  list.sort((a, b) => a - b);
  c = new Int32Array(list);
  candCache.set(key, c);
  return c;
}
/** squared distance from the nearest camera that sees the point at pose k, or -1 */
function visDist2(k: number, x: number, y: number, z: number) {
  const p = poses[k];
  const c = Math.cos(p.yaw), s = Math.sin(p.yaw);
  const ex = (x - p.x) * c + (y - p.y) * s, ey = -(x - p.x) * s + (y - p.y) * c;
  if (z < 0.7 && ex > -2.0 && ex < 4.0 && Math.abs(ey) < 1.8) return -1; // under and right around the car
  let best = -1;
  for (let j = 0; j < NC; j++) {
    const o = (k * NC + j) * 12;
    const dx = x - camW[o], dy = y - camW[o + 1], dz = z - camW[o + 2];
    const d2 = dx * dx + dy * dy + dz * dz;
    if (d2 > camRange2[j] || (best >= 0 && d2 >= best)) continue;
    const zc = dx * camW[o + 9] + dy * camW[o + 10] + dz * camW[o + 11];
    const xc = dx * camW[o + 3] + dy * camW[o + 4] + dz * camW[o + 5];
    const yc = dx * camW[o + 6] + dy * camW[o + 7] + dz * camW[o + 8];
    if (!camWide[j]) {
      if (zc <= 0.05 || Math.abs(xc) > camTanH[j] * zc || Math.abs(yc) > camTanV[j] * zc) continue;
    } else {
      const rho = Math.hypot(xc, yc);
      if (rho > 1e-6) {
        const th = Math.atan2(rho, zc);
        if ((th * Math.abs(xc)) / rho > camThH[j] || (th * Math.abs(yc)) / rho > camThV[j]) continue;
      } else if (zc <= 0) continue;
    }
    best = d2;
  }
  return best;
}
// timeline of one place: seen-runs (merged across gaps < 0.45 s) and the first time it was seen within each radius
type TL = { runs: Float32Array; n: number; ref: Float64Array };
const R2 = RADII.map((r) => r * r);
let tlCalls = 0;
function timeline(x: number, y: number, z: number): TL | null {
  tlCalls++;
  const cand = candidates(x, y);
  const runs: number[] = [];
  const ref = new Float64Array(RADII.length).fill(Infinity);
  const mr2 = maxRange * maxRange;
  for (let i = 0; i < cand.length; i++) {
    const k = cand[i];
    const p = poses[k];
    const dx = p.x - x, dy = p.y - y;
    if (dx * dx + dy * dy > mr2) continue;
    const d2 = visDist2(k, x, y, z);
    if (d2 < 0) continue;
    for (let g = 0; g < RADII.length; g++) if (ref[g] === Infinity && d2 <= R2[g]) ref[g] = p.t0;
    const n = runs.length;
    if (n && p.t0 - runs[n - 1] <= 0.45) runs[n - 1] = p.t1;
    else runs.push(p.t0, p.t1);
  }
  if (!runs.length) return null;
  return { runs: new Float32Array(runs), n: runs.length / 2, ref };
}
/** [e0, s1, e1, s2, e2] for a splat living [birth, death) */
const rtmp = new Float64Array(5);
function seenRuns(tl: TL, birth: number, death: number) {
  const R = tl.runs;
  let i = 0;
  while (i < tl.n && R[2 * i + 1] < birth) i++;
  let e0 = birth;
  if (i < tl.n && R[2 * i] <= birth + 0.21) { e0 = Math.max(birth, Math.min(R[2 * i + 1], death)); i++; }
  rtmp.fill(NEVER);
  rtmp[0] = e0;
  let slot = 1;
  for (; i < tl.n && R[2 * i] < death; i++) {
    const s = R[2 * i], e = Math.min(R[2 * i + 1], death);
    if (slot <= 2) { rtmp[2 * slot - 1] = s; rtmp[2 * slot] = e; slot++; }
    else rtmp[4] = e; // more than three runs: fold the rest into the third
  }
  return rtmp;
}
function emit(x: number, y: number, z: number, sx: number, sy: number, sz: number, q: number[], a: number, c: RGB, birth: number, death: number, tl: TL | null) {
  const o = S.alloc(), D = S.data;
  D[o] = x; D[o + 1] = y; D[o + 2] = z;
  D[o + 3] = Math.max(0.004, sx); D[o + 4] = Math.max(0.004, sy); D[o + 5] = Math.max(0.004, sz);
  D[o + 6] = q[0]; D[o + 7] = q[1]; D[o + 8] = q[2]; D[o + 9] = q[3];
  D[o + 10] = a;
  D[o + 11] = clamp01(c[0] * TINT[0]); D[o + 12] = clamp01(c[1] * TINT[1]); D[o + 13] = clamp01(c[2] * TINT[2]);
  D[o + 14] = birth; D[o + 15] = death;
  if (tl) {
    const r = seenRuns(tl, birth, death);
    D[o + 16] = r[0]; D[o + 17] = r[1]; D[o + 18] = r[2]; D[o + 19] = r[3]; D[o + 20] = r[4];
  } else {
    D[o + 16] = Math.min(death, NEVER); D[o + 17] = NEVER; D[o + 18] = NEVER; D[o + 19] = NEVER; D[o + 20] = NEVER;
  }
}
/** time the next generation replaces this node, or Infinity */
function refineTime(tl: TL, gen: number, maxGen: number, birth: number) {
  if (gen >= maxGen) return Infinity;
  const t = tl.ref[gen + 1];
  return t === Infinity ? Infinity : Math.max(t, birth + 0.3);
}

// ---------------------------------------------------------------- ground quadtree
const gcount = [0, 0, 0, 0];
const ROAD_STRETCH = [1.0, 1.35, 1.8, 2.4];
const sum: RGB = [0, 0, 0];
function groundNode(gen: number, x0: number, y0: number, c: number, birth: number, tl: TL) {
  const ns = gen === 0 ? 4 : gen === 1 ? 3 : 2;
  sum[0] = sum[1] = sum[2] = 0;
  let cnt = 0, zs = 0, road = 0, fine = 0;
  for (let i = 0; i < ns; i++) for (let j = 0; j < ns; j++) {
    const cl = groundColour(x0 + ((i + 0.5) / ns) * c, y0 + ((j + 0.5) / ns) * c);
    if (cl === C.None) continue;
    sum[0] += gcol[0]; sum[1] += gcol[1]; sum[2] += gcol[2]; zs += gz; cnt++;
    if (cl === C.Asphalt || cl === C.Lot) road++;
    if (cl === C.Asphalt || cl === C.Lot || cl === C.Sidewalk || cl === C.Curb) fine++;
  }
  if (cnt * 2 < ns * ns) return;
  const cx = x0 + c * (0.5 + rr(-0.18, 0.18)), cy = y0 + c * (0.5 + rr(-0.18, 0.18));
  const maxGen = fine * 2 >= cnt ? 3 : 2;
  const tR = refineTime(tl, gen, maxGen, birth);
  const death = tR === Infinity ? NEVER : tR + OVERLAP;
  const isRoad = road * 2 >= cnt;
  const g = ground(cx, cy);
  const heading = isRoad && g.c !== C.None ? ph[g.idx] : rand() * Math.PI;
  const sa = 0.6 * c * rr(0.9, 1.1);
  const stretch = isRoad ? ROAD_STRETCH[gen] * rr(0.85, 1.15) : rr(1, 1.25);
  const col: RGB = [sum[0] / cnt, sum[1] / cnt, sum[2] / cnt];
  emit(cx, cy, zs / cnt + 0.003 * gauss(), sa * stretch, sa, 0.012, quatYaw(heading), rr(0.62, 0.78), col, birth, death, tl);
  gcount[gen]++;
  if (death >= NEVER) return;
  const h = c / 2;
  for (const [dx, dy] of [[0, 0], [h, 0], [0, h], [h, h]]) {
    const ctl = gen + 1 <= 2 ? timeline(x0 + dx + h / 2, y0 + dy + h / 2, 0.05) ?? tl : tl;
    groundNode(gen + 1, x0 + dx, y0 + dy, h, tR, ctl);
  }
}
{
  const nx = Math.ceil((maxX - minX) / GEN0), ny = Math.ceil((maxY - minY) / GEN0);
  for (let iy = 0; iy < ny; iy++) for (let ix = 0; ix < nx; ix++) {
    const x0 = minX + ix * GEN0, y0 = minY + iy * GEN0;
    const k = cellAt(x0 + GEN0 / 2, y0 + GEN0 / 2);
    if (k < 0 || gDist[k] > BAND + 1) continue;
    const tl = timeline(x0 + GEN0 / 2, y0 + GEN0 / 2, 0.05);
    if (!tl) continue;
    groundNode(0, x0, y0, GEN0, tl.ref[0], tl);
  }
}
log("ground splats per generation: " + gcount.join(" / ") + " (timelines " + tlCalls + ")");

// crisp lane-marking splats near the driven lane, born when seen from within 10 m
let markCount = 0;
for (let i = 0; i < NP - 1; i++) {
  const s0 = i * DS;
  if (s0 < LOT_END_S) continue;
  const hd = ph[i], fx = Math.cos(hd), fy = Math.sin(hd);
  for (let sub = 0; sub < 2; sub++) {
    const s = s0 + sub * 0.25;
    const bx = px[i] + fx * (s - s0), by = py[i] + fy * (s - s0);
    for (const L of [PAVE_R + 0.25, -1.8, 1.62, 1.98, 5.4]) {
      const x = bx - fy * L, y = by + fx * L;
      const k = cellAt(x, y);
      if (k < 0 || Math.abs(gIdx[k] - i) > 8) continue;
      const p = paintAt(L, s, false);
      if (!p) continue;
      const tl = timeline(x, y, 0.05);
      if (!tl || tl.ref[4] === Infinity) continue;
      const pc = p === 1 ? WHITE : YELLOW;
      const wear = 0.8 + 0.2 * vnoise(x * 1.3, y * 1.3);
      emit(x, y, 0.01, 0.13, 0.035, 0.006, quatYaw(hd), 0.7 * wear, pc, tl.ref[4], NEVER, tl);
      markCount++;
    }
  }
}
log("marking splats: " + markCount);

// ---------------------------------------------------------------- generic surfaces (rectangles) with generations
// o: corner, u/v: unit in-plane axes, n: normal; lu/lv: extents in metres
type Surf = { o: number[]; u: number[]; v: number[]; n: number[]; lu: number; lv: number; maxGen: number; col: (u: number, v: number) => RGB; alpha: number };
const ocount = [0, 0, 0, 0];
function surfNode(sf: Surf, gen: number, u0: number, v0: number, cu: number, cv: number, birth: number, tl: TL, q: number[]) {
  const du = Math.min(cu, sf.lu - u0), dv = Math.min(cv, sf.lv - v0);
  if (du <= 0.01 || dv <= 0.01) return;
  const uc = u0 + du * (0.5 + rr(-0.12, 0.12)), vc = v0 + dv * (0.5 + rr(-0.12, 0.12));
  const ns = gen <= 1 ? 3 : 2;
  sum[0] = sum[1] = sum[2] = 0;
  for (let i = 0; i < ns; i++) for (let j = 0; j < ns; j++) {
    const c = sf.col(u0 + ((i + 0.5) / ns) * du, v0 + ((j + 0.5) / ns) * dv);
    sum[0] += c[0]; sum[1] += c[1]; sum[2] += c[2];
  }
  const m = ns * ns;
  const x = sf.o[0] + sf.u[0] * uc + sf.v[0] * vc, y = sf.o[1] + sf.u[1] * uc + sf.v[1] * vc, z = sf.o[2] + sf.u[2] * uc + sf.v[2] * vc;
  const tR = refineTime(tl, gen, sf.maxGen, birth);
  const death = tR === Infinity ? NEVER : tR + OVERLAP;
  emit(x, y, z, 0.6 * du * rr(0.92, 1.08), 0.6 * dv * rr(0.92, 1.08), 0.012, q, sf.alpha * rr(0.9, 1.05), [sum[0] / m, sum[1] / m, sum[2] / m], birth, death, tl);
  ocount[gen]++;
  if (death >= NEVER) return;
  const hu = cu / 2, hv = cv / 2;
  for (const [a, b] of [[0, 0], [hu, 0], [0, hv], [hu, hv]]) {
    if (u0 + a >= sf.lu || v0 + b >= sf.lv) continue;
    const mu = Math.min(u0 + a + hu / 2, sf.lu), mv = Math.min(v0 + b + hv / 2, sf.lv);
    const ctl = gen + 1 <= 2 ? timeline(sf.o[0] + sf.u[0] * mu + sf.v[0] * mv, sf.o[1] + sf.u[1] * mu + sf.v[1] * mv, sf.o[2] + sf.u[2] * mu + sf.v[2] * mv) ?? tl : tl;
    surfNode(sf, gen + 1, u0 + a, v0 + b, hu, hv, tR, ctl, q);
  }
}
function surface(sf: Surf) {
  const q = quatFromAxes(sf.u, sf.v, sf.n);
  for (let u0 = 0; u0 < sf.lu - 0.01; u0 += GEN0) for (let v0 = 0; v0 < sf.lv - 0.01; v0 += GEN0) {
    const mu = Math.min(u0 + GEN0 / 2, sf.lu), mv = Math.min(v0 + GEN0 / 2, sf.lv);
    const tl = timeline(sf.o[0] + sf.u[0] * mu + sf.v[0] * mv, sf.o[1] + sf.u[1] * mu + sf.v[1] * mv, sf.o[2] + sf.u[2] * mu + sf.v[2] * mv);
    if (!tl) continue;
    surfNode(sf, 0, u0, v0, GEN0, GEN0, tl.ref[0], tl, q);
  }
}
/** oriented box: centre (x, y) at ground, yaw, length L (local x), width Wd, height H from z0 */
function boxSurfaces(x: number, y: number, z0: number, yaw: number, L: number, Wd: number, H: number, maxGen: number, col: (face: number, u: number, v: number) => RGB, faces = 0b11111, alpha = 0.85) {
  const c = Math.cos(yaw), s = Math.sin(yaw);
  const ex = [c, s, 0], ey = [-s, c, 0], ez = [0, 0, 1];
  const P = (lx: number, ly: number, lz: number) => [x + lx * c - ly * s, y + lx * s + ly * c, z0 + lz];
  const neg = (a: number[]) => [-a[0], -a[1], -a[2]];
  // faces: 0 front(+x) 1 back(-x) 2 left(+y) 3 right(-y) 4 top
  if (faces & 1) surface({ o: P(L / 2, -Wd / 2, 0), u: ey, v: ez, n: ex, lu: Wd, lv: H, maxGen, col: (u, v) => col(0, u / Wd, v), alpha });
  if (faces & 2) surface({ o: P(-L / 2, Wd / 2, 0), u: neg(ey), v: ez, n: neg(ex), lu: Wd, lv: H, maxGen, col: (u, v) => col(1, u / Wd, v), alpha });
  if (faces & 4) surface({ o: P(L / 2, Wd / 2, 0), u: neg(ex), v: ez, n: ey, lu: L, lv: H, maxGen, col: (u, v) => col(2, u / L, v), alpha });
  if (faces & 8) surface({ o: P(-L / 2, -Wd / 2, 0), u: ex, v: ez, n: neg(ey), lu: L, lv: H, maxGen, col: (u, v) => col(3, u / L, v), alpha });
  if (faces & 16) surface({ o: P(-L / 2, -Wd / 2, H), u: ex, v: ey, n: ez, lu: L, lv: Wd, maxGen, col: (u, v) => col(4, u / L, v / Wd), alpha });
}

// ---------------------------------------------------------------- volumes (tree canopies) and lines (trunks, poles)
function volNode(gen: number, x: number, y: number, z: number, c: number, maxGen: number, inside: (x: number, y: number, z: number) => boolean, col: (x: number, y: number, z: number) => RGB, birth: number, tl: TL, alpha: number) {
  const tR = refineTime(tl, gen, maxGen, birth);
  const death = tR === Infinity ? NEVER : tR + OVERLAP;
  const s = 0.55 * c;
  emit(x + rr(-0.15, 0.15) * c, y + rr(-0.15, 0.15) * c, z + rr(-0.15, 0.15) * c, s * rr(0.8, 1.25), s * rr(0.8, 1.25), s * rr(0.5, 0.9), quatRandom(), alpha * rr(0.9, 1.05), col(x, y, z), birth, death, tl);
  ocount[gen]++;
  if (death >= NEVER) return;
  const h = c / 4;
  for (const dx of [-h, h]) for (const dy of [-h, h]) for (const dz of [-h, h]) {
    if (!inside(x + dx, y + dy, z + dz) || rand() < 0.12) continue;
    const ctl = gen + 1 <= 2 ? timeline(x + dx, y + dy, z + dz) ?? tl : tl;
    volNode(gen + 1, x + dx, y + dy, z + dz, c / 2, maxGen, inside, col, tR, ctl, alpha);
  }
}
function volume(bb: number[], maxGen: number, inside: (x: number, y: number, z: number) => boolean, col: (x: number, y: number, z: number) => RGB, alpha = 0.8) {
  for (let x = bb[0] + GEN0 / 2; x < bb[3]; x += GEN0) for (let y = bb[1] + GEN0 / 2; y < bb[4]; y += GEN0) for (let z = bb[2] + GEN0 / 2; z < bb[5]; z += GEN0) {
    if (!inside(x, y, z)) continue;
    const tl = timeline(x, y, z);
    if (!tl) continue;
    volNode(0, x, y, z, GEN0, maxGen, inside, col, tl.ref[0], tl, alpha);
  }
}
function lineNode(gen: number, a: number[], b: number[], radius: number, maxGen: number, col: RGB, birth: number, tl: TL) {
  const dx = b[0] - a[0], dy = b[1] - a[1], dz = b[2] - a[2];
  const len = Math.hypot(dx, dy, dz);
  const dir = [dx / len, dy / len, dz / len];
  const ref = Math.abs(dir[2]) > 0.9 ? [1, 0, 0] : [0, 0, 1];
  let p1 = [dir[1] * ref[2] - dir[2] * ref[1], dir[2] * ref[0] - dir[0] * ref[2], dir[0] * ref[1] - dir[1] * ref[0]];
  const pl = Math.hypot(p1[0], p1[1], p1[2]);
  p1 = [p1[0] / pl, p1[1] / pl, p1[2] / pl];
  const p2 = [dir[1] * p1[2] - dir[2] * p1[1], dir[2] * p1[0] - dir[0] * p1[2], dir[0] * p1[1] - dir[1] * p1[0]];
  const tR = refineTime(tl, gen, maxGen, birth);
  const death = tR === Infinity ? NEVER : tR + OVERLAP;
  const shade = 0.92 + 0.12 * rand();
  emit((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2, 0.55 * len, radius * (gen === 0 ? 1.4 : 1), radius * (gen === 0 ? 1.4 : 1), quatFromAxes(dir, p1, p2), 0.9, [col[0] * shade, col[1] * shade, col[2] * shade], birth, death, tl);
  ocount[gen]++;
  if (death >= NEVER) return;
  const m = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2];
  for (const [p, q] of [[a, m], [m, b]]) {
    const ctl = gen + 1 <= 2 ? timeline((p[0] + q[0]) / 2, (p[1] + q[1]) / 2, (p[2] + q[2]) / 2) ?? tl : tl;
    lineNode(gen + 1, p, q, radius, maxGen, col, tR, ctl);
  }
}
function line(a: number[], b: number[], radius: number, maxGen: number, col: RGB) {
  const len = Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
  const n = Math.max(1, Math.round(len / GEN0));
  for (let i = 0; i < n; i++) {
    const p = [0, 1, 2].map((k) => a[k] + ((b[k] - a[k]) * i) / n), q = [0, 1, 2].map((k) => a[k] + ((b[k] - a[k]) * (i + 1)) / n);
    const tl = timeline((p[0] + q[0]) / 2, (p[1] + q[1]) / 2, (p[2] + q[2]) / 2);
    if (tl) lineNode(0, p, q, radius, maxGen, col, tl.ref[0], tl);
  }
}

// ---------------------------------------------------------------- placement helpers
function isFree(x: number, y: number, margin = 0) {
  const k = cellAt(x, y);
  if (k < 0 || gIdx[k] < 0) return false;
  if (gCross[k] < 11 + margin) return false;
  const g = ground(x, y);
  if (g.lot) return Math.abs(g.lat) > LOT_HALF + 1 + margin;
  return g.lat < SW_R1 - margin || g.lat > SW_L1 + margin;
}
const occupied: { x: number; y: number; r: number }[] = [];
function claim(x: number, y: number, r: number) {
  for (const o of occupied) if (Math.hypot(o.x - x, o.y - y) < o.r + r) return false;
  occupied.push({ x, y, r });
  return true;
}

// curb faces along both curb lines
for (let i = 0; i < NP - 8; i += 8) {
  if (i * DS < LOT_END_S) continue;
  const hd = ph[i], fx = Math.cos(hd), fy = Math.sin(hd);
  for (const [L, side] of [[CURB_R + 0.04, 1], [CURB_L - 0.04, -1]] as const) {
    const x = px[i] - fy * L, y = py[i] + fx * L;
    const k = cellAt(x, y);
    if (k < 0 || Math.abs(gIdx[k] - i) > 4 || gCross[k] < 8) continue;
    const u = [fx * side, fy * side, 0];
    const o = side > 0 ? [x, y, 0] : [x + fx * 4, y + fy * 4, 0];
    surface({ o, u, v: [0, 0, 1], n: [-fy * side, fx * side, 0], lu: 4, lv: 0.13, maxGen: 3, col: () => [0.6, 0.6, 0.58], alpha: 0.85 });
  }
}

// --- buildings
const BUILD_COLS: RGB[] = [[0.74, 0.68, 0.58], [0.66, 0.66, 0.64], [0.58, 0.4, 0.33], [0.8, 0.79, 0.75], [0.7, 0.62, 0.5], [0.52, 0.55, 0.56]];
let buildingCount = 0;
function building(x: number, y: number, yaw: number, L: number, Wd: number, H: number, frontFace: number, kind: number, maxGen = 2) {
  const base = BUILD_COLS[kind % BUILD_COLS.length];
  const roof: RGB = rand() < 0.5 ? [0.46, 0.46, 0.45] : [0.3, 0.3, 0.31];
  const sign = ([[0.62, 0.16, 0.14], [0.14, 0.3, 0.55], [0.86, 0.84, 0.8], [0.2, 0.42, 0.3]] as RGB[])[Math.floor(rand() * 4)];
  boxSurfaces(x, y, 0, yaw, L, Wd, H, maxGen, (face, u, v) => {
    if (face === 4) { const n = 0.9 + 0.15 * vnoise(u * 9, v * 9); return [roof[0] * n, roof[1] * n, roof[2] * n]; }
    const ao = 0.72 + 0.28 * Math.min(1, v / 2.5);
    let c = base;
    const span = face <= 1 ? Wd : L;
    if (v > H - 0.6) c = [base[0] * 0.75, base[1] * 0.75, base[2] * 0.75];
    if (face === frontFace) {
      if (v > 0.4 && v < 3.0 && (u * span) % 4.2 > 0.35) c = [0.17, 0.22, 0.25];
      if (v > 3.3 && v < 4.1 && u > 0.2 && u < 0.8) c = sign;
    } else if (H > 6 && v > 3.5 && v < H - 1 && v % 3.4 > 1.4 && (u * span) % 3 > 1.2) c = [0.24, 0.29, 0.31];
    const n = 0.95 + 0.08 * vnoise(u * 30, v * 3);
    return [c[0] * ao * n, c[1] * ao * n, c[2] * ao * n];
  }, 0b11111, 0.88);
  buildingCount++;
}
{
  // drive-through restaurant just left of where the car waits at the window
  const p0 = drive.state(20);
  const h = p0.yaw, fx = Math.cos(h), fy = Math.sin(h);
  const L = 20, Wd = 13, H = 4.6;
  const cx = p0.x + fx * 1.5 - fy * (2.35 + Wd / 2), cy = p0.y + fy * 1.5 + fx * (2.35 + Wd / 2);
  building(cx, cy, h, L, Wd, H, 3, 0, 3);
  occupied.push({ x: cx, y: cy, r: 12 });
  boxSurfaces(p0.x + fx * 1.6 - fy * 1.95, p0.y + fy * 1.6 + fx * 1.95, 2.6, h, 3.2, 1.1, 0.3, 3, () => [0.5, 0.18, 0.16]);
  boxSurfaces(p0.x - fx * 9 - fy * 2.4, p0.y - fy * 9 + fx * 2.4, 0.3, h, 2.0, 0.25, 1.9, 3, (f, _u, v) => (f === 3 && v > 0.3 ? [0.2, 0.22, 0.26] : [0.3, 0.3, 0.3]));
}
for (const side of [-1, 1]) {
  let s = LOT_END_S + rr(20, 60);
  while (s < drive.length - 20) {
    const i = Math.min(NP - 1, Math.round(s / DS));
    const hd = ph[i], fx = Math.cos(hd), fy = Math.sin(hd);
    const L = rr(16, 38), Wd = rr(12, 22), H = rand() < 0.15 ? rr(8, 13) : rr(4.2, 6.5);
    const setback = side > 0 ? rr(22, 30) : rr(17, 25);
    const lat = side * (setback + Wd / 2);
    const cx = px[i] - fy * lat, cy = py[i] + fx * lat;
    let ok = !nearInter(s, 32);
    if (ok) for (const [u, v] of [[-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0, 0]]) {
      if (!isFree(cx + fx * u * L - fy * v * Wd, cy + fy * u * L + fx * v * Wd, 1.5)) { ok = false; break; }
    }
    // the face looking at the road: +y (2) for buildings on the right, -y (3) on the left
    if (ok && claim(cx, cy, Math.max(L, Wd) * 0.55)) building(cx, cy, hd, L, Wd, H, side < 0 ? 2 : 3, Math.floor(rand() * 6));
    s += L + rr(8, 40);
  }
}
log("buildings: " + buildingCount);

// --- trees
let treeCount = 0;
function tree(x: number, y: number, kind: number) {
  const bark: RGB = [0.29, 0.26, 0.22];
  if (kind === 0) {
    const H = rr(6, 9.5), R = rr(3.2, 5.2), trunkH = H * 0.42, cz = trunkH + R * 0.35;
    line([x, y, 0], [x, y, trunkH + 0.4], 0.16, 2, bark);
    const moss = rand();
    volume([x - R, y - R, cz - R * 0.6, x + R, y + R, cz + R * 0.6], 2,
      (a, b, c) => { const dx = (a - x) / R, dy = (b - y) / R, dz = (c - cz) / (R * 0.55); return dx * dx + dy * dy + dz * dz < 1 + 0.25 * (vnoise(a, b + c) - 0.5); },
      (a, b, c) => {
        const top = clamp01((c - trunkH) / (R * 0.9));
        const sh = 0.6 + 0.55 * top + 0.12 * (vnoise(a * 2, b * 2 + c) - 0.5);
        if (hash2(Math.floor(a * 3), Math.floor(b * 3 + c * 5)) < 0.1 + 0.1 * moss) return [0.44 * sh, 0.47 * sh, 0.4 * sh];
        return [0.2 * sh, 0.26 * sh, 0.16 * sh];
      }, 0.82);
  } else if (kind === 1) {
    const H = rr(13, 19), R = rr(2, 3.2), cz = H - R;
    line([x, y, 0], [x, y, cz], 0.14, 2, [0.33, 0.27, 0.22]);
    volume([x - R, y - R, cz - R, x + R, y + R, cz + R], 2,
      (a, b, c) => { const dx = (a - x) / R, dy = (b - y) / R, dz = (c - cz) / (R * 0.8); return dx * dx + dy * dy + dz * dz < 1; },
      (a, b, c) => { const sh = 0.7 + 0.35 * clamp01((c - cz + R) / (2 * R)) + 0.1 * (vnoise(a * 2, b * 2) - 0.5); return [0.2 * sh, 0.25 * sh, 0.15 * sh]; }, 0.8);
  } else {
    const H = rr(6, 9);
    line([x, y, 0], [x, y, H], 0.15, 2, [0.42, 0.38, 0.32]);
    const R = 2.6;
    volume([x - R, y - R, H - 0.8, x + R, y + R, H + 1.0], 2,
      (a, b, c) => { const d = Math.hypot(a - x, b - y) / R; return d < 1 && Math.abs(c - (H + 0.4 - d * d * 1.1)) < 0.45; },
      () => { const sh = 0.8 + 0.3 * rand(); return [0.25 * sh, 0.33 * sh, 0.2 * sh]; }, 0.8);
  }
  treeCount++;
}
for (const side of [-1, 1]) {
  let s = rr(0, 10);
  while (s < drive.length) {
    const i = Math.min(NP - 1, Math.round(s / DS));
    const hd = ph[i], fx = Math.cos(hd), fy = Math.sin(hd);
    const lot = s < LOT_END_S;
    const lat = side * (lot ? rr(LOT_HALF + 3, LOT_HALF + 12) : rand() < 0.35 ? (side > 0 ? rr(14.5, 19) : rr(10.8, 15)) : rr(16, 38));
    const x = px[i] - fy * lat, y = py[i] + fx * lat;
    if (isFree(x, y, 0.5) && claim(x, y, 3.5)) tree(x, y, lot ? 2 : rand() < 0.55 ? 0 : rand() < 0.6 ? 1 : 2);
    s += rr(8, 28);
  }
}
log("trees: " + treeCount);

// --- street lights, signals, signs
const GALV: RGB = [0.52, 0.53, 0.53];
let poleCount = 0;
for (let s = LOT_END_S + 15; s < drive.length; s += rr(42, 55)) {
  const i = Math.min(NP - 1, Math.round(s / DS));
  const hd = ph[i], fx = Math.cos(hd), fy = Math.sin(hd);
  const lat = CURB_R - 1.1;
  const x = px[i] - fy * lat, y = py[i] + fx * lat;
  if (nearInter(s, 20) || !claim(x, y, 0.6)) continue;
  const H = 9, arm = 2.6;
  line([x, y, 0], [x, y, H], 0.09, 2, GALV);
  const ax = x - fy * arm, ay = y + fx * arm;
  line([x, y, H - 0.1], [ax, ay, H + 0.1], 0.06, 2, GALV);
  boxSurfaces(ax - fy * 0.3, ay + fx * 0.3, H - 0.05, hd + Math.PI / 2, 0.7, 0.3, 0.12, 2, (f) => (f === 4 ? [0.4, 0.4, 0.4] : [0.78, 0.78, 0.74]));
  poleCount++;
}
let signalCount = 0;
for (const q of inters) {
  if (q.turn) continue;
  const j = Math.min(NP - 1, Math.round((q.s + 17) / DS));
  const hd = ph[j], fx = Math.cos(hd), fy = Math.sin(hd);
  const lat = CURB_R - 1.2;
  const x = px[j] - fy * lat, y = py[j] + fx * lat;
  line([x, y, 0], [x, y, 6.2], 0.12, 2, GALV);
  line([x, y, 6.0], [x - fy * 9.5, y + fx * 9.5, 6.0], 0.08, 2, GALV);
  for (const off of [2.8, 6.4]) {
    const hx = x - fy * off, hy = y + fx * off;
    boxSurfaces(hx, hy, 4.9, hd, 0.3, 0.38, 1.05, 3, (f, _u, v) => {
      if (f === 1) { if (v < 0.3) return [0.3, 0.85, 0.5]; if (v > 0.75) return [0.25, 0.1, 0.08]; return [0.22, 0.18, 0.08]; }
      return [0.4, 0.34, 0.1];
    });
  }
  signalCount++;
}
for (let s = LOT_END_S + 60; s < drive.length; s += rr(260, 420)) {
  const i = Math.min(NP - 1, Math.round(s / DS));
  const hd = ph[i], fx = Math.cos(hd), fy = Math.sin(hd);
  const lat = CURB_R - 1.4;
  const x = px[i] - fy * lat, y = py[i] + fx * lat;
  if (nearInter(s, 25) || !claim(x, y, 0.5)) continue;
  line([x, y, 0], [x, y, 2.2], 0.05, 2, [0.5, 0.5, 0.5]);
  surface({ o: [x - fx * 0.3, y - fy * 0.3, 2.2], u: [fx, fy, 0], v: [0, 0, 1], n: [-fy, fx, 0], lu: 0.6, lv: 0.76, maxGen: 3, alpha: 0.95,
    col: (u, v) => ((v > 0.5 && v < 0.62) || (v > 0.12 && v < 0.42 && Math.abs(u - 0.3) < 0.18) ? [0.12, 0.12, 0.12] : [0.86, 0.87, 0.85]) });
}
log("poles: " + poleCount + ", signals: " + signalCount);

// --- cars: parked cars are static surfaces with generations; moving cars use a fixed single-resolution cloud
const CAR_COLS: RGB[] = [[0.85, 0.85, 0.84], [0.62, 0.63, 0.64], [0.13, 0.13, 0.14], [0.2, 0.27, 0.42], [0.55, 0.1, 0.1], [0.38, 0.39, 0.4], [0.75, 0.74, 0.7]];
const GLASS: RGB = [0.12, 0.15, 0.17];
function carColour(color: RGB) {
  return (part: number) => (face: number, u: number, v: number): RGB => {
    if (part === 1) return face === 4 ? [color[0] * 1.08, color[1] * 1.08, color[2] * 1.08] : GLASS;
    if (face === 1 && v > 0.5 && v < 0.62 && Math.abs(u - 0.5) > 0.2) return [0.62, 0.08, 0.07];
    if (face === 0 && v > 0.35 && v < 0.47 && Math.abs(u - 0.5) > 0.24) return [0.86, 0.87, 0.85];
    if ((face === 2 || face === 3) && v < 0.32 && (Math.abs(u - 0.2) < 0.08 || Math.abs(u - 0.8) < 0.08)) return [0.07, 0.07, 0.08];
    const sh = face === 4 ? 1.1 : 0.78 + 0.3 * v;
    return [color[0] * sh, color[1] * sh, color[2] * sh];
  };
}
let parked = 0;
for (let s = 6; s < LOT_END_S - 4; s += 2.8) {
  const i = Math.min(NP - 1, Math.round(s / DS));
  const hd = ph[i], fx = Math.cos(hd), fy = Math.sin(hd);
  for (const side of [-1, 1]) {
    if (rand() < 0.45) continue;
    const lat = side * 9.7;
    const x = px[i] - fy * lat + fx * 1.4, y = py[i] + fx * lat + fy * 1.4;
    const g = ground(x, y);
    if (g.c !== C.Lot || !claim(x, y, 1.3)) continue;
    const yaw = hd + (side > 0 ? -Math.PI / 2 : Math.PI / 2) + rr(-0.05, 0.05);
    const cc = carColour(CAR_COLS[Math.floor(rand() * CAR_COLS.length)]);
    boxSurfaces(x, y, 0.25, yaw, 4.6, 1.84, 0.75, 3, cc(0));
    boxSurfaces(x - Math.cos(yaw) * 0.2, y - Math.sin(yaw) * 0.2, 1.0, yaw, 2.5, 1.5, 0.45, 3, cc(1));
    parked++;
  }
}
log("parked cars: " + parked);
const N_STATIC = S.n;
log("static splats: " + N_STATIC + " (objects per generation " + ocount.join(" / ") + ", timelines " + tlCalls + ")");

// ---------------------------------------------------------------- moving cars (transient splats)
type Tpl = { p: number[]; n: number };
function carTemplate(color: RGB): Tpl {
  const out: number[] = [];
  const cc = carColour(color);
  const step = 0.3 / Math.sqrt(DENSITY);
  const addBox = (x0: number, z0: number, L: number, Wd: number, H: number, col: (f: number, u: number, v: number) => RGB) => {
    const faces: [number[], number[], number[], number, number, number][] = [
      [[x0 + L / 2, -Wd / 2, z0], [0, 1, 0], [0, 0, 1], Wd, H, 0],
      [[x0 - L / 2, Wd / 2, z0], [0, -1, 0], [0, 0, 1], Wd, H, 1],
      [[x0 + L / 2, Wd / 2, z0], [-1, 0, 0], [0, 0, 1], L, H, 2],
      [[x0 - L / 2, -Wd / 2, z0], [1, 0, 0], [0, 0, 1], L, H, 3],
      [[x0 - L / 2, -Wd / 2, z0 + H], [1, 0, 0], [0, 1, 0], L, Wd, 4],
    ];
    for (const [o, u, v, lu, lv, f] of faces) {
      const n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
      const q = quatFromAxes(u, v, n);
      for (let a = step / 2; a < lu; a += step) for (let b = step / 2; b < lv; b += step) {
        const c = col(f, a / lu, f === 4 ? b / lv : b);
        out.push(o[0] + u[0] * a + v[0] * b, o[1] + u[1] * a + v[1] * b, o[2] + u[2] * a + v[2] * b, step * 0.62, step * 0.62, 0.015, q[0], q[1], q[2], q[3], c[0], c[1], c[2]);
      }
    }
  };
  addBox(1.4, 0.25, 4.6, 1.84, 0.75, cc(0));
  addBox(1.2, 1.0, 2.5, 1.5, 0.45, cc(1));
  return { p: out, n: out.length / 13 };
}
const HZ = 18;
type Mover = { name: string; t0: number; t1: number; pose: (t: number) => { x: number; y: number; yaw: number } | null; color: RGB };
const movers: Mover[] = [];
for (const [a, b] of [[168, 200], [302, 324]]) {
  movers.push({ name: "lead" + a, t0: a, t1: b, color: [0.85, 0.85, 0.84], pose: (t) => {
    const st = drive.state(t);
    const p = drive.atS(Math.min(drive.length, st.s + 8 + 1.4 * st.speed));
    return { x: p.x, y: p.y, yaw: p.yaw };
  } });
}
for (let m = 0; m < 16; m++) {
  const tMeet = 125 + m * 14 + rr(-4, 4);
  if (tMeet > drive.duration - 8) break;
  if (stops.some((st) => tMeet > st.t0 - 3 && tMeet < st.t1 + 3)) continue;
  const sMeet = drive.state(tMeet).s, v = rr(13, 19), lane = rand() < 0.5 ? 3.6 : 7.2;
  movers.push({ name: "oncoming" + m, t0: tMeet - 7, t1: tMeet + 6, color: CAR_COLS[Math.floor(rand() * CAR_COLS.length)], pose: (t) => {
    const sv = sMeet - v * (t - tMeet);
    if (sv < LOT_END_S + 5 || sv > drive.length - 2 || nearInter(sv, 20)) return null;
    const i = Math.min(NP - 1, Math.max(0, Math.round(sv / DS)));
    return { x: px[i] - Math.sin(ph[i]) * lane, y: py[i] + Math.cos(ph[i]) * lane, yaw: ph[i] + Math.PI };
  } });
}
for (const st of stops) {
  const q = inters.find((qq) => qq.s > st.s && qq.s - st.s < 45);
  if (!q) continue;
  const c = drive.atS(q.s);
  for (let m = 0; m * 6.5 < st.t1 - st.t0 - 4; m++) {
    const dir = m % 2 === 0 ? 1 : -1, tc = st.t0 + 2 + m * 6.5 + rr(0, 2), v = rr(11, 15), off = dir > 0 ? -1.8 : 1.8;
    movers.push({ name: "cross" + Math.round(st.t0) + "_" + m, t0: tc - 4, t1: tc + 4, color: CAR_COLS[Math.floor(rand() * CAR_COLS.length)], pose: (t) => {
      const d = dir * v * (t - tc);
      return { x: c.x + Math.cos(q.hc) * d - Math.sin(q.hc) * off, y: c.y + Math.sin(q.hc) * d + Math.cos(q.hc) * off, yaw: q.hc + (dir > 0 ? 0 : Math.PI) };
    } });
  }
}
let snapshots = 0;
for (const mv of movers) {
  const tpl = carTemplate(mv.color);
  for (let t = Math.ceil(mv.t0 * HZ) / HZ; t < mv.t1; t += 1 / HZ) {
    if (t < 0 || t > drive.duration) continue;
    const p = mv.pose(t);
    if (!p) continue;
    const st = drive.state(t);
    const cx = p.x + Math.cos(p.yaw) * 1.4, cy = p.y + Math.sin(p.yaw) * 1.4;
    const c = Math.cos(st.yaw), s = Math.sin(st.yaw);
    const ex = (cx - st.x) * c + (cy - st.y) * s, ey = -(cx - st.x) * s + (cy - st.y) * c;
    let vis = false;
    for (let j = 0; j < NC && !vis; j++) {
      const g = cams[j];
      const dx = ex - g.pos[0], dy = ey - g.pos[1], dz = 0.8 - g.pos[2];
      if (dx * dx + dy * dy + dz * dz > camRange2[j]) continue;
      const zc = dx * g.fwd[0] + dy * g.fwd[1] + dz * g.fwd[2];
      const xc = dx * g.right[0] + dy * g.right[1] + dz * g.right[2];
      const yc = dx * g.down[0] + dy * g.down[1] + dz * g.down[2];
      if (!g.wide) vis = zc > 0.3 && Math.abs(xc) < g.tanH * zc * 1.15 && Math.abs(yc) < g.tanV * zc * 1.15;
      else { const rho = Math.hypot(xc, yc), th = Math.atan2(rho, zc); vis = rho < 1e-6 || ((th * Math.abs(xc)) / rho <= g.thH && (th * Math.abs(yc)) / rho <= g.thV); }
    }
    if (!vis) continue;
    const death = t + 1 / HZ + 0.025;
    const cy2 = Math.cos(p.yaw), sy2 = Math.sin(p.yaw), qa = quatYaw(p.yaw);
    for (let j = 0; j < tpl.n; j++) {
      const o = j * 13, P = tpl.p;
      const bw = P[o + 6], bx = P[o + 7], by = P[o + 8], bz = P[o + 9];
      const q = [qa[0] * bw - qa[3] * bz, qa[0] * bx - qa[3] * by, qa[0] * by + qa[3] * bx, qa[0] * bz + qa[3] * bw];
      emit(p.x + P[o] * cy2 - P[o + 1] * sy2, p.y + P[o] * sy2 + P[o + 1] * cy2, P[o + 2], P[o + 3], P[o + 4], P[o + 5], q, 0.85, [P[o + 10], P[o + 11], P[o + 12]], t, death, null);
    }
    snapshots++;
  }
}
log("movers: " + movers.length + ", snapshots: " + snapshots + ", transient splats: " + (S.n - N_STATIC));


// ---------------------------------------------------------------- tracked movers (objects.json)
// Rigid car models in their own frame (x forward, y left, z up, origin on the ground under the box centre),
// built like a photo scan: each patch of each face is born when a camera first sees that face, and refined
// (0.6 -> 0.3 -> 0.15 m) when seen from within 20 m and 9 m. Faces never seen get a predicted mirror of the
// other side, superseded where an observation arrives. Tracks are sampled at 10 Hz; state 1 while the car is
// not seen (outside every camera, or alongside within the body-mask zone of the side cameras).
type TrackPose = { x: number; y: number; yaw: number };
type ObjDef = { id: number; color: RGB; t0: number; t1: number; pose: (t: number) => TrackPose };
function egoCams(t: number) {
  const st = drive.state(t);
  const c = Math.cos(st.yaw), s = Math.sin(st.yaw);
  const rot = (v: number[]) => [v[0] * c - v[1] * s, v[0] * s + v[1] * c, v[2]];
  return { st, cams: cams.map((g) => ({ g, p: rot(g.pos).map((v, i) => v + (i === 0 ? st.x : i === 1 ? st.y : 0)), r: rot(g.right), d: rot(g.down), f: rot(g.fwd), range2: camRange2[cams.indexOf(g)] })) };
}
/** distance from the nearest camera that sees world point p with outward normal n at time t, or -1 */
function seenFrom(E: ReturnType<typeof egoCams>, p: number[], n: number[]) {
  // body-mask zone: right alongside the car the side cameras see their own body
  const c = Math.cos(E.st.yaw), s = Math.sin(E.st.yaw);
  const ex = (p[0] - E.st.x) * c + (p[1] - E.st.y) * s, ey = -(p[0] - E.st.x) * s + (p[1] - E.st.y) * c;
  if (ex > -1.2 && ex < 2.6 && Math.abs(ey) < 4.6) return -1;
  let best = -1;
  for (const k of E.cams) {
    const dx = p[0] - k.p[0], dy = p[1] - k.p[1], dz = p[2] - k.p[2];
    const d2 = dx * dx + dy * dy + dz * dz;
    if (d2 > k.range2) continue;
    if (n[0] * dx + n[1] * dy + n[2] * dz > 0) continue; // facing away
    const zc = dx * k.f[0] + dy * k.f[1] + dz * k.f[2];
    const xc = dx * k.r[0] + dy * k.r[1] + dz * k.r[2];
    const yc = dx * k.d[0] + dy * k.d[1] + dz * k.d[2];
    if (!inImage(k.g, xc, yc, zc)) continue;
    const d = Math.sqrt(d2);
    if (best < 0 || d < best) best = d;
  }
  return best;
}
const objDefs: ObjDef[] = [];
{
  // overtaking on the right lane at +4 m/s relative, passing through the body-mask zone alongside
  const t0 = 128;
  objDefs.push({ id: 1, color: [0.2, 0.27, 0.42], t0, t1: t0 + 16, pose: (t) => {
    const st = drive.state(t);
    const sObj = st.s - 32 + 4 * (t - t0);
    const i = Math.min(NP - 1, Math.max(0, Math.round(sObj / DS)));
    return { x: px[i] + Math.sin(ph[i]) * 3.6, y: py[i] - Math.cos(ph[i]) * 3.6, yaw: ph[i] };
  } });
  // oncoming in the far lane
  const tm = 150, sm = drive.state(tm).s;
  objDefs.push({ id: 2, color: [0.82, 0.82, 0.8], t0: tm - 6, t1: tm + 5, pose: (t) => {
    const sv = sm - 17 * (t - tm);
    const i = Math.min(NP - 1, Math.max(0, Math.round(sv / DS)));
    return { x: px[i] - Math.sin(ph[i]) * 7.2, y: py[i] + Math.cos(ph[i]) * 7.2, yaw: ph[i] + Math.PI };
  } });
  // crossing in front of us while we wait at the second light
  const stop = stops[stops.length - 1];
  if (stop) {
    const q = inters.find((qq) => qq.s > stop.s && qq.s - stop.s < 45);
    if (q) {
      const c0 = drive.atS(q.s);
      const tc = stop.t0 + 6;
      objDefs.push({ id: 3, color: [0.55, 0.1, 0.1], t0: tc - 4, t1: tc + 4, pose: (t) => {
        const d = 12 * (t - tc);
        return { x: c0.x + Math.cos(q.hc) * d - Math.sin(q.hc) * -1.8, y: c0.y + Math.sin(q.hc) * d + Math.cos(q.hc) * -1.8, yaw: q.hc };
      } });
    }
    // a lead car pulling away when the light turns green
    const tg = stop.t1;
    objDefs.push({ id: 4, color: [0.13, 0.13, 0.14], t0: tg - 5, t1: tg + 12, pose: (t) => {
      const st = drive.state(t);
      const extra = t < tg ? 0 : 0.9 * (t - tg) * (t - tg);
      const p = drive.atS(Math.min(drive.length, st.s + 9.5 + extra));
      return { x: p.x, y: p.y, yaw: p.yaw };
    } });
  }
}
type Cell = { o: number[]; n: number[]; u: number[]; v: number[]; lu: number; lv: number; col: (u: number, v: number) => RGB; face: number };
function carFaces(color: RGB): Cell[] {
  const cc = carColour(color);
  const out: Cell[] = [];
  const box = (x0: number, z0: number, L: number, Wd: number, H: number, part: number) => {
    const col = cc(part);
    out.push({ o: [x0 + L / 2, -Wd / 2, z0], u: [0, 1, 0], v: [0, 0, 1], n: [1, 0, 0], lu: Wd, lv: H, col: (u, v) => col(0, u / Wd, v), face: 0 });
    out.push({ o: [x0 - L / 2, Wd / 2, z0], u: [0, -1, 0], v: [0, 0, 1], n: [-1, 0, 0], lu: Wd, lv: H, col: (u, v) => col(1, u / Wd, v), face: 1 });
    out.push({ o: [x0 + L / 2, Wd / 2, z0], u: [-1, 0, 0], v: [0, 0, 1], n: [0, 1, 0], lu: L, lv: H, col: (u, v) => col(2, u / L, v), face: 2 });
    out.push({ o: [x0 - L / 2, -Wd / 2, z0], u: [1, 0, 0], v: [0, 0, 1], n: [0, -1, 0], lu: L, lv: H, col: (u, v) => col(3, u / L, v), face: 3 });
    out.push({ o: [x0 - L / 2, -Wd / 2, z0 + H], u: [1, 0, 0], v: [0, 1, 0], n: [0, 0, 1], lu: L, lv: Wd, col: (u, v) => col(4, u / L, v / Wd), face: 4 });
  };
  box(0, 0.25, 4.6, 1.84, 0.75, 0);
  box(-0.2, 1.0, 2.5, 1.5, 0.45, 1);
  return out;
}
const OBJ_PROPS = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3", "t_birth", "t_death", "e0", "s1", "e1", "s2", "e2"];
function writeSplatPly(rows: number[][]) {
  const header = "ply\nformat binary_little_endian 1.0\ncomment Reality Splat synthetic mover\nelement vertex " + rows.length + "\n" + OBJ_PROPS.map((p) => "property float " + p).join("\n") + "\nend_header\n";
  const hb = new TextEncoder().encode(header);
  const body = new Float32Array(rows.length * OBJ_PROPS.length);
  rows.forEach((r, i) => body.set(r, i * OBJ_PROPS.length));
  const buf = new Uint8Array(hb.length + body.byteLength);
  buf.set(hb, 0);
  buf.set(new Uint8Array(body.buffer), hb.length);
  return buf;
}
const SH = 0.28209479177387814;
function splatRow(p: number[], sx: number, sy: number, q: number[], a: number, c: RGB, times: number[]) {
  const al = Math.min(0.995, Math.max(0.005, a));
  return [p[0], p[1], p[2], (clamp01(c[0] * TINT[0]) - 0.5) / SH, (clamp01(c[1] * TINT[1]) - 0.5) / SH, (clamp01(c[2] * TINT[2]) - 0.5) / SH, Math.log(al / (1 - al)), Math.log(Math.max(0.004, sx)), Math.log(Math.max(0.004, sy)), Math.log(0.012), q[0], q[1], q[2], q[3], ...times];
}
type ObjOut = { id: number; class: string; size: number[]; file: string; predicted_file: string | null; t_first: number; t_last: number; track: { t: number[]; x: number[]; y: number[]; z: number[]; yaw: number[]; state: number[] } };
const objectsOut: ObjOut[] = [];
const objFiles: { file: string; data: Uint8Array }[] = [];
for (const od of objDefs) {
  const ts: number[] = [];
  for (let t = Math.max(0, od.t0); t <= Math.min(drive.duration, od.t1) + 1e-6; t += 0.1) ts.push(Math.round(t * 1000) / 1000);
  const E = ts.map((t) => egoCams(t));
  const P = ts.map((t) => od.pose(t));
  const toW = (k: number, p: number[]) => { const c = Math.cos(P[k].yaw), s = Math.sin(P[k].yaw); return [P[k].x + p[0] * c - p[1] * s, P[k].y + p[0] * s + p[1] * c, p[2]]; };
  const dirW = (k: number, n: number[]) => { const c = Math.cos(P[k].yaw), s = Math.sin(P[k].yaw); return [n[0] * c - n[1] * s, n[0] * s + n[1] * c, n[2]]; };
  // predicted while its centre is outside every camera or inside the side cameras' body-mask zone
  const state = ts.map((_, k) => (seenFrom(E[k], toW(k, [0, 0, 0.8]), [0, 0, 0]) >= 0 ? 0 : 1));
  const seenAny = state.some((v) => v === 0);
  if (!seenAny) continue;
  // timeline of a patch: seen-runs and first times within 20 / 9 m
  const patchTL = (p: number[], n: number[]) => {
    const runs: number[] = [];
    const ref = [Infinity, Infinity, Infinity];
    for (let k = 0; k < ts.length; k++) {
      const d = seenFrom(E[k], toW(k, p), dirW(k, n));
      if (d < 0) continue;
      if (ref[0] === Infinity) ref[0] = ts[k];
      if (ref[1] === Infinity && d < 20) ref[1] = ts[k];
      if (ref[2] === Infinity && d < 9) ref[2] = ts[k];
      const m = runs.length;
      if (m && ts[k] - runs[m - 1] <= 0.25) runs[m - 1] = ts[k];
      else runs.push(ts[k], ts[k]);
    }
    return runs.length ? { runs, ref } : null;
  };
  const times7 = (tl: { runs: number[] }, birth: number, death: number) => {
    const R = tl.runs;
    let i = 0;
    while (i < R.length / 2 && R[2 * i + 1] < birth) i++;
    let e0 = birth;
    if (i < R.length / 2 && R[2 * i] <= birth + 0.15) { e0 = Math.min(Math.max(birth, R[2 * i + 1]), death); i++; }
    const out = [birth, death, e0, NEVER, NEVER, NEVER, NEVER];
    let slot = 1;
    for (; i < R.length / 2 && R[2 * i] < death; i++) {
      if (slot <= 2) { out[1 + 2 * slot] = R[2 * i]; out[2 + 2 * slot] = Math.min(R[2 * i + 1], death); slot++; } else out[6] = Math.min(R[2 * i + 1], death);
    }
    return out;
  };
  const rows: number[][] = [];
  const pred: number[][] = [];
  const firstSeenOfFace = new Map<string, number>();
  const sizes = [0.6, 0.3, 0.15];
  const faces = carFaces(od.color);
  for (const f of faces) {
    const q = quatFromAxes(f.u, f.v, f.n);
    const node = (gen: number, u0: number, v0: number, birth: number, tl: { runs: number[]; ref: number[] }) => {
      const c = sizes[gen];
      const du = Math.min(c, f.lu - u0), dv = Math.min(c, f.lv - v0);
      if (du <= 0.01 || dv <= 0.01) return;
      const uc = u0 + du / 2, vc = v0 + dv / 2;
      const p = [0, 1, 2].map((k) => f.o[k] + f.u[k] * uc + f.v[k] * vc);
      const next = gen < 2 ? tl.ref[gen + 1] : Infinity;
      const tR = next === Infinity ? Infinity : Math.max(next, birth + 0.3);
      const death = tR === Infinity ? NEVER : tR + OVERLAP;
      rows.push(splatRow(p, 0.6 * du, 0.6 * dv, q, 0.92, f.col(uc, vc), times7(tl, birth, death)));
      if (death >= NEVER) return;
      for (const [a, b] of [[0, 0], [c / 2, 0], [0, c / 2], [c / 2, c / 2]]) {
        if (u0 + a >= f.lu || v0 + b >= f.lv) continue;
        const pc = [0, 1, 2].map((k) => f.o[k] + f.u[k] * Math.min(u0 + a + c / 4, f.lu) + f.v[k] * Math.min(v0 + b + c / 4, f.lv));
        const ctl = patchTL(pc, f.n) ?? tl;
        node(gen + 1, u0 + a, v0 + b, tR, ctl);
      }
    };
    for (let u0 = 0; u0 < f.lu - 0.01; u0 += sizes[0]) for (let v0 = 0; v0 < f.lv - 0.01; v0 += sizes[0]) {
      const p = [0, 1, 2].map((k) => f.o[k] + f.u[k] * Math.min(u0 + 0.3, f.lu) + f.v[k] * Math.min(v0 + 0.3, f.lv));
      const tl = patchTL(p, f.n);
      const key = f.face + ":" + Math.round(p[0] * 10) + ":" + Math.round(Math.abs(p[1]) * 10) + ":" + Math.round(p[2] * 10);
      if (tl) {
        node(0, u0, v0, tl.ref[0], tl);
        const prev = firstSeenOfFace.get(key);
        firstSeenOfFace.set(key, Math.min(prev ?? Infinity, tl.ref[0]));
      }
    }
  }
  // predicted: every 0.3 m patch of the sides is mirrored from the other side once that is seen, until seen itself
  for (const f of faces) {
    if (f.face !== 2 && f.face !== 3) continue;
    const q = quatFromAxes(f.u, f.v, f.n);
    const mirror = faces.find((m2) => m2.face === (f.face === 2 ? 3 : 2) && Math.abs(m2.o[2] - f.o[2]) < 1e-6)!;
    for (let u0 = 0; u0 < f.lu - 0.01; u0 += 0.3) for (let v0 = 0; v0 < f.lv - 0.01; v0 += 0.3) {
      const du = Math.min(0.3, f.lu - u0), dv = Math.min(0.3, f.lv - v0);
      const p = [0, 1, 2].map((k) => f.o[k] + f.u[k] * (u0 + du / 2) + f.v[k] * (v0 + dv / 2));
      const mp = [p[0], -p[1], p[2]];
      const own = patchTL(p, f.n);
      const mtl = patchTL(mp, mirror.n);
      if (!mtl) continue;
      const birth = mtl.ref[0];
      const death = own ? Math.max(birth + 0.3, own.ref[0]) + OVERLAP : NEVER;
      if (death <= birth + 0.05) continue;
      pred.push(splatRow(p, 0.6 * du, 0.6 * dv, q, 0.8, f.col(u0 + du / 2, v0 + dv / 2), [birth, death, birth, NEVER, NEVER, NEVER, NEVER]));
    }
  }
  if (!rows.length) continue;
  const tFirst = Math.min(...rows.map((r) => r[14]), ...(pred.length ? pred.map((r) => r[14]) : [Infinity]));
  let lastSeen = 0;
  state.forEach((v, k) => { if (v === 0) lastSeen = ts[k]; });
  const tLast = Math.min(ts[ts.length - 1], lastSeen + 1.5);
  const file = "objects/" + od.id + ".ply";
  objFiles.push({ file, data: writeSplatPly(rows) });
  let predFile: string | null = null;
  if (pred.length) {
    predFile = "objects/" + od.id + ".pred.ply";
    objFiles.push({ file: predFile, data: writeSplatPly(pred) });
  }
  const keep = ts.map((_t, k) => k).filter((k) => ts[k] >= tFirst - 0.2 && ts[k] <= tLast + 0.2);
  const r4 = (v: number) => Math.round(v * 1e4) / 1e4;
  objectsOut.push({
    id: od.id, class: "car", size: [4.6, 1.84, 1.45], file, predicted_file: predFile, t_first: r4(tFirst), t_last: r4(tLast),
    track: { t: keep.map((k) => ts[k]), x: keep.map((k) => r4(P[k].x)), y: keep.map((k) => r4(P[k].y)), z: keep.map(() => 0), yaw: keep.map((k) => r4(P[k].yaw)), state: keep.map((k) => state[k]) },
  });
  log("mover " + od.id + ": " + rows.length + " splats, " + pred.length + " predicted, t " + tFirst.toFixed(1) + "-" + tLast.toFixed(1) + ", predicted poses " + state.filter((v) => v === 1).length + "/" + state.length);
}

// ---------------------------------------------------------------- tiles and map.json
type TileAcc = { id: string; idx: number[]; kind: "static" | "transient" };
const tiles = new Map<string, TileAcc>();
{
  const D = S.data;
  for (let j = 0; j < S.n; j++) {
    const o = j * STRIDE;
    const transient = j >= N_STATIC;
    const tx = Math.floor(D[o] / TILE), ty = Math.floor(D[o + 1] / TILE);
    const id = transient ? "d_" + tx + "_" + ty + "_" + Math.floor(D[o + 14] / TRANSIENT_BUCKET_S) : "s_" + tx + "_" + ty;
    let t = tiles.get(id);
    if (!t) tiles.set(id, (t = { id, idx: [], kind: transient ? "transient" : "static" }));
    t.idx.push(j);
  }
}
if (existsSync(OUT)) rmSync(OUT, { recursive: true, force: true });
mkdirSync(path.join(OUT, "tiles"), { recursive: true });
writeFileSync(path.join(OUT, "drive.json"), readFileSync(path.join(SRC, "drive.json")));
writeFileSync(path.join(OUT, "rig.json"), JSON.stringify(rig, null, 1));
mkdirSync(path.join(OUT, "objects"), { recursive: true });
for (const f of objFiles) writeFileSync(path.join(OUT, f.file), f.data);
writeFileSync(path.join(OUT, "objects.json"), JSON.stringify({ objects: objectsOut }));

const SH_C0 = 0.28209479177387814;
const PROPS = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3", "t_birth", "t_death", "e0", "s1", "e1", "s2", "e2"];
const mapTiles: { id: string; kind: string; file: string; count: number; bbox: number[]; t_min: number; t_max: number }[] = [];
let totalBytes = 0;
const r3 = (v: number) => Math.round(v * 1000) / 1000;
for (const t of [...tiles.values()].sort((a, b) => a.id.localeCompare(b.id))) {
  const D = S.data, idx = t.idx;
  for (let i = idx.length - 1; i > 0; i--) { const j = Math.floor(rand() * (i + 1)); const tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp; }
  const header = "ply\nformat binary_little_endian 1.0\ncomment Reality Splat synthetic scene\nelement vertex " + idx.length + "\n" + PROPS.map((p) => "property float " + p).join("\n") + "\nend_header\n";
  const hb = new TextEncoder().encode(header);
  const NPR = PROPS.length;
  const body = new Float32Array(idx.length * NPR);
  const bb = [Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity];
  let tmin = Infinity, tmax = -Infinity;
  for (let i = 0; i < idx.length; i++) {
    const o = idx[i] * STRIDE, b = i * NPR;
    body[b] = D[o]; body[b + 1] = D[o + 1]; body[b + 2] = D[o + 2];
    body[b + 3] = (D[o + 11] - 0.5) / SH_C0; body[b + 4] = (D[o + 12] - 0.5) / SH_C0; body[b + 5] = (D[o + 13] - 0.5) / SH_C0;
    const a = Math.min(0.995, Math.max(0.005, D[o + 10]));
    body[b + 6] = Math.log(a / (1 - a));
    body[b + 7] = Math.log(D[o + 3]); body[b + 8] = Math.log(D[o + 4]); body[b + 9] = Math.log(D[o + 5]);
    body[b + 10] = D[o + 6]; body[b + 11] = D[o + 7]; body[b + 12] = D[o + 8]; body[b + 13] = D[o + 9];
    for (let k = 0; k < 7; k++) body[b + 14 + k] = D[o + 14 + k];
    for (let k = 0; k < 3; k++) { bb[k] = Math.min(bb[k], D[o + k]); bb[3 + k] = Math.max(bb[3 + k], D[o + k]); }
    tmin = Math.min(tmin, D[o + 14]);
    tmax = Math.max(tmax, D[o + 15] >= NEVER ? drive.duration : D[o + 15]);
  }
  const file = "tiles/" + t.id + ".ply";
  const buf = new Uint8Array(hb.length + body.byteLength);
  buf.set(hb, 0);
  buf.set(new Uint8Array(body.buffer), hb.length);
  writeFileSync(path.join(OUT, file), buf);
  totalBytes += buf.length;
  mapTiles.push({ id: t.id, kind: t.kind, file, count: idx.length, bbox: bb.map(r3), t_min: r3(tmin), t_max: r3(tmax) });
}

// ---------------------------------------------------------------- sky panorama (PNG, equirectangular, centre column faces east)
function writePng(file: string, w: number, h: number, rgb: Uint8Array) {
  const raw = Buffer.alloc((w * 3 + 1) * h);
  for (let y = 0; y < h; y++) { raw[y * (w * 3 + 1)] = 0; Buffer.from(rgb.buffer, y * w * 3, w * 3).copy(raw, y * (w * 3 + 1) + 1); }
  const crcTable = new Int32Array(256).map((_, n) => { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; return c; });
  const crc = (b: Buffer) => { let c = -1; for (const x of b) c = crcTable[(c ^ x) & 255] ^ (c >>> 8); return (c ^ -1) >>> 0; };
  const chunk = (type: string, data: Buffer) => { const len = Buffer.alloc(4); len.writeUInt32BE(data.length); const td = Buffer.concat([Buffer.from(type), data]); const c = Buffer.alloc(4); c.writeUInt32BE(crc(td)); return Buffer.concat([len, td, c]); };
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(w, 0); ihdr.writeUInt32BE(h, 4); ihdr[8] = 8; ihdr[9] = 2; ihdr[10] = 0; ihdr[11] = 0; ihdr[12] = 0;
  writeFileSync(file, Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk("IHDR", ihdr), chunk("IDAT", deflateSync(raw, { level: 9 })), chunk("IEND", Buffer.alloc(0))]));
}
{
  const w = 2048, h = 1024;
  const img = new Uint8Array(w * h * 3);
  for (let y = 0; y < h; y++) {
    const el = (0.5 - (y + 0.5) / h) * Math.PI; // elevation
    for (let x = 0; x < w; x++) {
      const az = (0.5 - (x + 0.5) / w) * 2 * Math.PI; // CCW from east
      const dx = Math.cos(el) * Math.cos(az), dy = Math.cos(el) * Math.sin(az), dz = Math.sin(el);
      let r: number, g: number, b: number;
      if (el > -0.02) {
        const t = clamp01(el / (Math.PI / 2));
        // overcast dawn: bright milky horizon, slightly darker blue-grey zenith, soft cloud texture
        r = 0.8 - 0.2 * t; g = 0.83 - 0.18 * t; b = 0.85 - 0.13 * t;
        const k = 1 / Math.max(0.08, dz + 0.08);
        const cl = fbm(dx * k * 1.6 + 11, dy * k * 1.6 + 5) * 0.7 + 0.3 * fbm(dx * k * 5 + 1, dy * k * 5 + 3);
        const cv = (cl - 0.5) * 0.16 * (0.4 + 0.6 * t);
        r += cv; g += cv; b += cv * 0.9;
        // faint warm glow low in the east-south-east (sunrise behind clouds)
        const sun = Math.max(0, dx * 0.93 + dy * -0.37) ** 6 * Math.exp(-el * 6);
        r += 0.07 * sun; g += 0.04 * sun; b += 0.0 * sun;
      } else {
        // below the horizon: distant hazy ground, normally hidden behind the splats
        const t = clamp01(-el / 0.4);
        r = 0.62 - 0.2 * t; g = 0.65 - 0.2 * t; b = 0.64 - 0.2 * t;
      }
      const o = (y * w + x) * 3;
      img[o] = Math.round(clamp01(r) * 255); img[o + 1] = Math.round(clamp01(g) * 255); img[o + 2] = Math.round(clamp01(b) * 255);
    }
  }
  writePng(path.join(OUT, "sky.png"), w, h, img);
}
writeFileSync(path.join(OUT, "map.json"), JSON.stringify({ t_end: r3(drive.duration), tiles: mapTiles, sky: "sky.png" }, null, 1));
const total = mapTiles.reduce((a, t) => a + t.count, 0);
log("wrote " + mapTiles.length + " tiles (" + mapTiles.filter((t) => t.kind === "static").length + " static), " + total + " splats, " + (totalBytes / 1e6).toFixed(0) + " MB to " + OUT);
