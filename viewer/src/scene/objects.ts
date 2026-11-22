// Moving objects from objects.json (PLAN.md): one rigid splat model per tracked object in its own frame
// (x forward, y left, z up, origin on the ground under the box centre), moved every frame along its smoothed
// track. The model's splats carry the usual seven drive times, so it fills in like a photo scan as it is
// observed; an optional predicted_file (never-observed parts) is drawn softer. Same time shader as the map.
// "Clean" vehicles (the default) draw each one as a smooth 3D model instead (render/vehicles.ts), posed from a
// smoothed copy of the track: positions Gaussian-smoothed (0.15 s), held still at a standstill, heading from the
// direction of travel once moving, wheels on the textured ground where it is there.

import * as THREE from "three";
import { ExtSplats, SplatMesh } from "@sparkjsdev/spark";
import TileWorker from "./tileWorker?worker";
import type { TileRequest, TileResult } from "./tileWorker";
import { AO_HALF_X, AO_HALF_Y, EGO } from "../render/contactShadow";
import { makeTileUniforms, makeTimeModifier, makeTimeTexture, MAX_MOVER_SHADOWS, TIME_TEX_WIDTH, type SplatGlobals, type TileUniforms } from "../render/splatShader";
import { wrapAngle } from "./drive";
import { fromHalf } from "./whiteBalance";
import { gradeColour } from "../render/splatShader";
import { GREY_PAINT, namedPaint, vehicleDims, vehicleKind, type Paint, type Vehicle, type VehicleKit } from "../render/vehicles";

export type TrackJson = { t: number[]; x: number[]; y: number[]; z?: number[]; yaw: number[]; state?: number[] };
export type ContactShadow = { shape?: string; offset_xyz?: number[]; sigma_xy_m?: number[]; peak_opacity?: number };
export type ObjectJson = { id: number | string; name?: string; class?: string; size?: number[]; file: string; predicted_file?: string | null; t_first: number; t_last: number; track: TrackJson; contact_shadow?: ContactShadow };
export type ObjectsJson = { objects: ObjectJson[] };

type Part = { mesh: SplatMesh; ext: ExtSplats; tex: THREE.DataTexture; U: TileUniforms; loadedAt: number };
/** the track smoothed for clean vehicles, at the track's own samples */
type Smooth = { t: Float64Array; x: Float64Array; y: Float64Array; z: Float64Array; yaw: Float64Array; speed: Float64Array; accel: Float64Array; yawRate: Float64Array; dist: Float64Array; ghost: Float64Array; moving: boolean };
type Mover = { o: ObjectJson; yaw: Float64Array; parts: Part[]; state: "idle" | "loading" | "live"; sm: Smooth; vehicle?: Vehicle; scanPaint?: Paint };

function gauss(t: Float64Array, v: ArrayLike<number>, sigma: number) {
  const n = t.length, out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0, w = 0;
    for (let j = i; j >= 0 && t[i] - t[j] < 3 * sigma; j--) { const k = Math.exp(-0.5 * ((t[i] - t[j]) / sigma) ** 2); s += k * v[j]; w += k; }
    for (let j = i + 1; j < n && t[j] - t[i] < 3 * sigma; j++) { const k = Math.exp(-0.5 * ((t[j] - t[i]) / sigma) ** 2); s += k * v[j]; w += k; }
    out[i] = s / w;
  }
  return out;
}
function deriv(t: Float64Array, v: Float64Array) {
  const n = t.length, out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const a = Math.max(0, i - 1), b = Math.min(n - 1, i + 1);
    out[i] = t[b] > t[a] ? (v[b] - v[a]) / (t[b] - t[a]) : 0;
  }
  return out;
}

function smoothTrack(tr: TrackJson, yawU: Float64Array): Smooth {
  const t = Float64Array.from(tr.t);
  const n = t.length;
  const z0 = tr.z ?? new Array(n).fill(0);
  const ghost = Float64Array.from(tr.state ?? new Array(n).fill(0));
  if (n < 3) {
    const c = (v: number) => new Float64Array(n).fill(v);
    return { t, x: Float64Array.from(tr.x), y: Float64Array.from(tr.y), z: Float64Array.from(z0), yaw: Float64Array.from(yawU), speed: c(0), accel: c(0), yawRate: c(0), dist: c(0), ghost, moving: false };
  }
  const xs = gauss(t, tr.x, 0.15), ys = gauss(t, tr.y, 0.15), zs = gauss(t, z0, 0.3);
  // at a standstill the track still wanders by centimetres: there the position comes from a much longer window
  const xh = gauss(t, tr.x, 0.7), yh = gauss(t, tr.y, 0.7);
  const vx = deriv(t, xs), vy = deriv(t, ys);
  const speed = gauss(t, Array.from(vx, (v, i) => Math.hypot(v, vy[i])), 0.2);
  const yawT = gauss(t, yawU, 0.3);
  const yaw = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const still = 1 - THREE.MathUtils.smoothstep(speed[i], 0.25, 1.2);
    xs[i] += (xh[i] - xs[i]) * still;
    ys[i] += (yh[i] - ys[i]) * still;
    // heading: the direction of travel once it is moving (unless that is backwards to the track's yaw), else the
    // track's own yaw
    const hv = Math.atan2(vy[i], vx[i]);
    const d = wrapAngle(hv - yawT[i]);
    const w = Math.abs(d) < Math.PI / 2 ? THREE.MathUtils.smoothstep(speed[i], 1.0, 3.0) : 0;
    yaw[i] = yawT[i] + d * w;
  }
  for (let i = 1; i < n; i++) yaw[i] = yaw[i - 1] + wrapAngle(yaw[i] - yaw[i - 1]);
  const yawS = gauss(t, yaw, 0.12);
  const accel = gauss(t, deriv(t, speed), 0.25);
  const yawRate = gauss(t, deriv(t, yawS), 0.2);
  const dist = new Float64Array(n);
  for (let i = 1; i < n; i++) dist[i] = dist[i - 1] + 0.5 * (speed[i] + speed[i - 1]) * (t[i] - t[i - 1]);
  let vmax = 0;
  for (let i = 0; i < n; i++) vmax = Math.max(vmax, speed[i]);
  return { t, x: xs, y: ys, z: zs, yaw: yawS, speed, accel, yawRate, dist, ghost: gauss(t, ghost, 0.3), moving: vmax > 0.8 };
}

export type Pose = { x: number; y: number; z: number; yaw: number; ghost: number };

const EASE = 0.45;

export class ObjectManager {
  readonly group = new THREE.Group();
  private movers: Mover[] = [];
  private worker = new TileWorker();
  private pending = new Map<string, (r: TileResult | null) => void>();
  private seq = 0;
  loaded = 0;
  /** contact shadows under the movers (?moverShadows=0 to compare) */
  shadows = new URLSearchParams(location.search).get("moverShadows") !== "0";
  /** clean vehicles (smooth models) instead of the scanned splat models */
  clean = true;
  /** the clean vehicles, drawn over a generated clip separately */
  readonly vehicles = new THREE.Group();
  kit: VehicleKit | null = null;
  /** the road height at a point (the textured ground), for the clean vehicles' wheels */
  groundAt?: (x: number, y: number) => number | null;
  night = 0;
  /** capture: the simulated step instead of the wall clock's */
  fixedDt: number | null = null;
  private lastNow = 0;

  constructor(private readonly baseUrl: string, json: ObjectsJson, private readonly G: SplatGlobals) {
    this.group.name = "movers";
    this.vehicles.name = "vehicles";
    for (const o of json.objects ?? []) {
      if (!o.track?.t?.length) continue;
      // unwrap yaw so interpolation never takes the long way round
      const yaw = Float64Array.from(o.track.yaw);
      for (let i = 1; i < yaw.length; i++) yaw[i] = yaw[i - 1] + wrapAngle(yaw[i] - yaw[i - 1]);
      this.movers.push({ o, yaw, parts: [], state: "idle", sm: smoothTrack(o.track, yaw) });
    }
    this.worker.onmessage = (ev: MessageEvent<{ ok: boolean; result?: TileResult; id?: string }>) => {
      const id = ev.data.result?.id ?? ev.data.id ?? "";
      this.pending.get(id)?.(ev.data.ok ? ev.data.result! : null);
      this.pending.delete(id);
    };
  }

  get count() {
    return this.movers.length;
  }

  private decode(file: string): Promise<TileResult | null> {
    const id = `o${this.seq++}`;
    const req: TileRequest = { id, url: new URL(file, new URL(this.baseUrl, location.href)).href, texWidth: TIME_TEX_WIDTH };
    return new Promise((res) => {
      this.pending.set(id, res);
      this.worker.postMessage(req);
    });
  }

  /** interpolated pose at T (clamped to the track), with the predicted-state amount */
  pose(m: Mover, T: number): Pose {
    const tr = m.o.track;
    const t = tr.t;
    let i = 0;
    if (T >= t[t.length - 1]) i = t.length - 1;
    else if (T > t[0]) {
      let lo = 0, hi = t.length - 1;
      while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (t[mid] <= T) lo = mid; else hi = mid; }
      i = lo;
    }
    const j = Math.min(t.length - 1, i + 1);
    const a = j > i && t[j] > t[i] ? THREE.MathUtils.clamp((T - t[i]) / (t[j] - t[i]), 0, 1) : 0;
    const L = (arr: ArrayLike<number> | undefined, d = 0) => (arr ? arr[i] + (arr[j] - arr[i]) * a : d);
    return { x: L(tr.x), y: L(tr.y), z: L(tr.z), yaw: L(m.yaw), ghost: tr.state ? L(tr.state) : 0 };
  }

  /** the smoothed state at T (clamped to the track) */
  /** director looks: every mover around T with its smoothed pose, its box size and its track */
  *tracks(T: number, before = 3, after = 1) {
    for (const m of this.movers) {
      const { t_first, t_last } = m.o;
      if (T < t_first - after || T > t_last + before) continue;
      const presence = THREE.MathUtils.smoothstep(T, t_first, t_first + EASE) * (1 - THREE.MathUtils.smoothstep(T, t_last - EASE, t_last));
      const size = m.vehicle?.spec.dims ?? (m.o.size as [number, number, number] | undefined) ?? [4.5, 1.85, 1.5];
      yield { id: String(m.o.id), pose: this.smoothAt(m, Math.min(Math.max(T, t_first), t_last)), at: (t: number) => this.smoothAt(m, t), t_first, t_last, presence, size, moving: m.sm.moving };
    }
  }

  private smoothAt(m: Mover, T: number) {
    const s = m.sm, t = s.t;
    let i = 0;
    if (T >= t[t.length - 1]) i = t.length - 1;
    else if (T > t[0]) {
      let lo = 0, hi = t.length - 1;
      while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (t[mid] <= T) lo = mid; else hi = mid; }
      i = lo;
    }
    const j = Math.min(t.length - 1, i + 1);
    const a = j > i && t[j] > t[i] ? THREE.MathUtils.clamp((T - t[i]) / (t[j] - t[i]), 0, 1) : 0;
    const L = (arr: Float64Array) => arr[i] + (arr[j] - arr[i]) * a;
    return { x: L(s.x), y: L(s.y), z: L(s.z), yaw: L(s.yaw), speed: L(s.speed), accel: L(s.accel), yawRate: L(s.yawRate), dist: L(s.dist), ghost: L(s.ghost) };
  }

  /** the body colour of a scan: the median of its opaque splats on the sides at door height, graded */
  private scanColour(m: Mover, r: TileResult) {
    const H = m.o.size?.[2] ?? 1.5, W = m.o.size?.[1] ?? 1.85;
    const fa = new Float32Array(r.extA.buffer, r.extA.byteOffset, r.extA.length);
    const rs: number[] = [], gs: number[] = [], bs: number[] = [];
    const zMin = Math.min(...Array.from({ length: Math.min(r.count, 4000) }, (_, k) => fa[k * Math.max(1, Math.floor(r.count / 4000)) * 4 + 2] + r.origin[2]));
    for (let i = 0; i < r.count; i++) {
      const i4 = i * 4;
      const z = fa[i4 + 2] + r.origin[2] - zMin, y = fa[i4 + 1] + r.origin[1];
      if (z < 0.35 * H || z > 0.62 * H || Math.abs(y) < 0.3 * W) continue;
      if (fromHalf(r.extA[i4 + 3] & 65535) < 0.4) continue;
      const b0 = r.extB[i4], b1 = r.extB[i4 + 1];
      rs.push(fromHalf(b0 & 65535));
      gs.push(fromHalf(b0 >>> 16));
      bs.push(fromHalf(b1 & 65535));
    }
    if (rs.length < 50) return;
    const med = (a: number[]) => a.sort((p, q) => p - q)[a.length >> 1];
    const c = gradeColour(new THREE.Color(med(rs), med(gs), med(bs)), this.G);
    const lum = 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;
    m.scanPaint = { color: new THREE.Color().setRGB(c.r, c.g, c.b, THREE.SRGBColorSpace), metal: lum > 0.6 ? 0.08 : 0.35, rough: 0.32 };
  }

  private async load(m: Mover) {
    m.state = "loading";
    const files: [string, boolean][] = [[m.o.file, false]];
    if (m.o.predicted_file) files.push([m.o.predicted_file, true]);
    for (const [file, predicted] of files) {
      const r = await this.decode(file);
      if (!r || m.state !== "loading") continue;
      if (!predicted && !m.scanPaint) this.scanColour(m, r);
      const ext = new ExtSplats({ extArrays: [r.extA, r.extB], numSplats: r.count });
      const tex = makeTimeTexture(r.times, r.count);
      const U = makeTileUniforms(tex, r.base, r.quantum, false, predicted);
      U.obj.value.w = 1;
      const mesh = new SplatMesh({ extSplats: ext, worldModifier: makeTimeModifier(this.G, U), editable: false, raycastable: false });
      mesh.name = `mover-${m.o.id}${predicted ? "-pred" : ""}`;
      // the decoder centres the splats on their box; put that offset back inside the object frame
      const child = new THREE.Group();
      mesh.position.set(r.origin[0], r.origin[1], r.origin[2]);
      child.add(mesh);
      this.group.add(child);
      m.parts.push({ mesh, ext, tex, U, loadedAt: performance.now() });
    }
    if (m.state === "loading") m.state = "live";
  }

  private unload(m: Mover) {
    for (const p of m.parts) {
      p.mesh.parent && this.group.remove(p.mesh.parent);
      p.mesh.dispose();
      p.tex.dispose();
    }
    m.parts = [];
    m.state = "idle";
  }

  private dropVehicle(m: Mover) {
    if (!m.vehicle) return;
    this.vehicles.remove(m.vehicle.root);
    m.vehicle.dispose();
    m.vehicle = undefined;
  }

  /** switch between clean and scanned vehicles */
  setClean(on: boolean) {
    this.clean = on;
    if (!on) for (const m of this.movers) this.dropVehicle(m);
  }

  /** load what is near in time and space, move every live model to its pose */
  update(T: number, fx: number, fy: number, radius: number, stitched: boolean, now: number) {
    let n = 0;
    let ns = 0;
    const D = this.G.moverData;
    const dt = this.fixedDt ?? Math.min(0.1, Math.max(0, (now - (this.lastNow || now)) / 1000));
    this.lastNow = now;
    for (const m of this.movers) {
      const { t_first, t_last } = m.o;
      const near = !stitched && T >= t_first - 3 && T <= t_last + 1;
      const p = near ? this.pose(m, T) : null;
      const inRange = !!p && Math.hypot(p.x - fx, p.y - fy) < radius;
      const presence = THREE.MathUtils.smoothstep(T, t_first, t_first + EASE) * (1 - THREE.MathUtils.smoothstep(T, t_last - EASE, t_last));
      // clean: the scan is only fetched for its colour when the producer did not name one
      const needScan = !this.clean || !namedPaint(m.o.name ?? "");
      if (m.state === "idle" && near && inRange && needScan) void this.load(m);
      if (m.state !== "idle" && (stitched || T < t_first - 6 || T > t_last + 4)) this.unload(m);
      if (this.clean && this.kit && near && inRange && presence > 0.001) {
        if (!m.vehicle) {
          const kind = vehicleKind(m.o.name ?? "", m.o.class ?? "car", m.o.size);
          const paint = namedPaint(m.o.name ?? "") ?? m.scanPaint ?? GREY_PAINT;
          m.vehicle = this.kit.make({ kind, dims: vehicleDims(kind, m.o.size), paint, parked: !m.sm.moving });
          this.vehicles.add(m.vehicle.root);
        }
        if (m.scanPaint && m.vehicle.spec.paint === GREY_PAINT) m.vehicle.setPaint(m.scanPaint);
        const s = this.smoothAt(m, T);
        const g = this.groundAt?.(s.x, s.y);
        if (g != null && Math.abs(g - s.z) < 0.4) s.z = g;
        m.vehicle.update({ ...s, fade: presence * (1 - 0.45 * s.ghost) }, this.night, dt);
        n++;
        // its box for the static splats' leftovers to give way (the shadow is the model's own, strength 0 here)
        if (ns < MAX_MOVER_SHADOWS && presence > 0.05) {
          const [L, W, H] = m.vehicle.spec.dims;
          const k = ns * 12;
          D[k] = s.x;
          D[k + 1] = s.y;
          D[k + 2] = s.z;
          D[k + 3] = (Math.hypot(L / 2, W / 2) + 0.5) ** 2;
          D[k + 4] = Math.cos(s.yaw);
          D[k + 5] = Math.sin(s.yaw);
          D[k + 6] = L / 2;
          D[k + 7] = W / 2;
          D[k + 8] = 0;
          D[k + 9] = 1;
          D[k + 10] = H;
          ns++;
        }
      } else if (m.vehicle && (!near || !inRange || presence <= 0.001 || !this.clean)) this.dropVehicle(m);
      for (const part of m.parts) part.mesh.parent!.visible = !this.clean && presence > 0.001;
      if (this.clean || !m.parts.length || !p) continue;
      n++;
      // contact shadow: the ego car's overcast-sky occlusion (render/contactShadow.ts) scaled to the box, at the
      // producer's offset when it gives one; its footprint from the box size (the producer's sigma is a third of
      // the length, the same footprint). Drawn by the splat, ground-atlas and apron shaders (MOVER_SHADOW_GLSL), so
      // whatever is in front of the mover covers its shadow as well
      const fade = presence * Math.min(1, (now - (m.parts[0]?.loadedAt ?? now)) / 300);
      const cs = m.o.contact_shadow;
      const L = m.o.size?.[0] ?? 4.5, W = m.o.size?.[1] ?? 1.85, H = m.o.size?.[2] ?? 1.5;
      const off = cs?.offset_xyz ?? [0, 0, 0];
      const strength = fade * (1 - 0.6 * p.ghost);
      const hr = EGO.height / Math.max(0.8, H);
      const reach = Math.hypot(L / 2 + (AO_HALF_X - EGO.halfL) / hr, W / 2 + (AO_HALF_Y - EGO.halfW) / hr);
      if (this.shadows && ns < MAX_MOVER_SHADOWS && fade > 0.01) {
        const c = Math.cos(p.yaw), s = Math.sin(p.yaw);
        const k = ns * 12;
        D[k] = p.x + off[0] * c - off[1] * s;
        D[k + 1] = p.y + off[0] * s + off[1] * c;
        D[k + 2] = p.z;
        D[k + 3] = strength;
        D[k + 4] = c;
        D[k + 5] = s;
        D[k + 6] = L / 2;
        D[k + 7] = W / 2;
        D[k + 8] = reach * reach;
        D[k + 9] = hr;
        ns++;
      }
      for (const part of m.parts) {
        const holder = part.mesh.parent!;
        holder.position.set(p.x, p.y, p.z);
        holder.rotation.set(0, 0, p.yaw);
        holder.visible = presence > 0.001;
        part.U.tile.value.z = presence * Math.min(1, (now - part.loadedAt) / 300);
        part.U.obj.value.y = p.ghost;
        part.mesh.updateVersion();
      }
    }
    this.loaded = n;
    if (ns !== this.G.moverInfo.value.x || ns > 0) {
      this.G.moverInfo.value.x = ns;
      this.G.moverTexture.needsUpdate = true;
    }
    this.G.moverInfo.value.y = this.clean ? 1 : 0;
  }

  /** no shadows (scene change) */
  clearShadows() {
    this.G.moverInfo.value.x = 0;
  }

  dispose() {
    for (const m of this.movers) {
      this.unload(m);
      this.dropVehicle(m);
    }
    this.worker.terminate();
  }
}
