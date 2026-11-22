// Drive telemetry from drive.json (viewer scene contract), with interpolation by time and arc length.

export type DriveJson = {
  fps: number;
  duration: number;
  origin: { lat: number; lon: number };
  t: number[];
  x: number[];
  y: number[];
  z?: number[];
  yaw: number[];
  pitch?: number[];
  roll?: number[];
  speed: number[];
  steer?: number[];
  ap?: number[];
  brake?: number[];
  blink_l?: number[];
  blink_r?: number[];
  gear?: number[];
  lat?: number[];
  lon?: number[];
};

export type DriveState = {
  t: number;
  x: number;
  y: number;
  z: number;
  yaw: number;
  pitch: number;
  roll: number;
  speed: number;
  steer: number;
  ap: number;
  brake: boolean;
  blinkL: boolean;
  blinkR: boolean;
  gear: number;
  s: number;
};

const f32 = (a: number[] | undefined, n: number) => {
  const out = new Float32Array(n);
  if (a) for (let i = 0; i < n; i++) out[i] = a[i] ?? 0;
  return out;
};
const f64 = (a: number[] | undefined, n: number) => {
  const out = new Float64Array(n);
  if (a) for (let i = 0; i < n; i++) out[i] = a[i] ?? 0;
  return out;
};

export function wrapAngle(a: number) {
  while (a > Math.PI) a -= 2 * Math.PI;
  while (a < -Math.PI) a += 2 * Math.PI;
  return a;
}

export class Drive {
  readonly n: number;
  readonly duration: number;
  readonly fps: number;
  readonly origin: { lat: number; lon: number };
  readonly t: Float64Array;
  readonly x: Float64Array;
  readonly y: Float64Array;
  readonly z: Float64Array;
  readonly yaw: Float64Array;
  readonly pitch: Float32Array;
  readonly roll: Float32Array;
  readonly speed: Float32Array;
  readonly steer: Float32Array;
  readonly ap: Float32Array;
  readonly brake: Float32Array;
  readonly blinkL: Float32Array;
  readonly blinkR: Float32Array;
  readonly gear: Float32Array;
  readonly lat?: Float64Array;
  readonly lon?: Float64Array;
  /** cumulative arc length in metres */
  readonly s: Float64Array;
  readonly length: number;

  constructor(j: DriveJson) {
    const n = j.t.length;
    this.n = n;
    this.fps = j.fps;
    this.origin = j.origin;
    this.t = f64(j.t, n);
    this.duration = j.duration ?? this.t[n - 1];
    this.x = f64(j.x, n);
    this.y = f64(j.y, n);
    this.z = f64(j.z, n);
    // unwrap yaw so interpolation never crosses the +-pi seam
    this.yaw = f64(j.yaw, n);
    for (let i = 1; i < n; i++) this.yaw[i] = this.yaw[i - 1] + wrapAngle(this.yaw[i] - this.yaw[i - 1]);
    this.pitch = f32(j.pitch, n);
    this.roll = f32(j.roll, n);
    this.speed = f32(j.speed, n);
    this.steer = f32(j.steer, n);
    this.ap = f32(j.ap, n);
    this.brake = f32(j.brake, n);
    this.blinkL = f32(j.blink_l, n);
    this.blinkR = f32(j.blink_r, n);
    this.gear = j.gear ? f32(j.gear, n) : new Float32Array(n).fill(1);
    if (j.lat && j.lon) {
      this.lat = f64(j.lat, n);
      this.lon = f64(j.lon, n);
    }
    this.s = new Float64Array(n);
    for (let i = 1; i < n; i++) {
      const dx = this.x[i] - this.x[i - 1];
      const dy = this.y[i] - this.y[i - 1];
      this.s[i] = this.s[i - 1] + Math.hypot(dx, dy);
    }
    this.length = this.s[n - 1];
  }

  /** index i such that t[i] <= time < t[i+1], clamped */
  index(time: number) {
    const t = this.t;
    if (time <= t[0]) return 0;
    if (time >= t[this.n - 1]) return this.n - 2;
    let lo = 0;
    let hi = this.n - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (t[mid] <= time) lo = mid;
      else hi = mid;
    }
    return lo;
  }

  /** index i such that s[i] <= dist < s[i+1], clamped */
  indexAtS(dist: number) {
    const s = this.s;
    if (dist <= s[0]) return 0;
    if (dist >= s[this.n - 1]) return this.n - 2;
    let lo = 0;
    let hi = this.n - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (s[mid] <= dist) lo = mid;
      else hi = mid;
    }
    return lo;
  }

  state(time: number, out?: DriveState): DriveState {
    const i = this.index(time);
    const t0 = this.t[i];
    const t1 = this.t[i + 1];
    const a = t1 > t0 ? Math.min(1, Math.max(0, (time - t0) / (t1 - t0))) : 0;
    const L = (arr: ArrayLike<number>) => arr[i] + (arr[i + 1] - arr[i]) * a;
    const near = a < 0.5 ? i : i + 1;
    const o = out ?? ({} as DriveState);
    o.t = time;
    o.x = L(this.x);
    o.y = L(this.y);
    o.z = L(this.z);
    o.yaw = L(this.yaw);
    o.pitch = L(this.pitch);
    o.roll = L(this.roll);
    o.speed = Math.max(0, L(this.speed));
    o.steer = L(this.steer);
    o.ap = this.ap[near];
    o.brake = this.brake[near] > 0.5;
    o.blinkL = this.blinkL[near] > 0.5;
    o.blinkR = this.blinkR[near] > 0.5;
    o.gear = this.gear[near];
    o.s = L(this.s);
    return o;
  }

  /** position and heading at arc length */
  atS(dist: number) {
    const i = this.indexAtS(dist);
    const s0 = this.s[i];
    const s1 = this.s[i + 1];
    const a = s1 > s0 ? Math.min(1, Math.max(0, (dist - s0) / (s1 - s0))) : 0;
    const x = this.x[i] + (this.x[i + 1] - this.x[i]) * a;
    const y = this.y[i] + (this.y[i + 1] - this.y[i]) * a;
    const z = this.z[i] + (this.z[i + 1] - this.z[i]) * a;
    const yaw = this.yaw[i] + (this.yaw[i + 1] - this.yaw[i]) * a;
    const t = this.t[i] + (this.t[i + 1] - this.t[i]) * a;
    return { x, y, z, yaw, t };
  }

  /** latitude / longitude at arc length (from the drive's GPS columns), or null */
  latLonAtS(dist: number): [number, number] | null {
    if (!this.lat || !this.lon) return null;
    const i = this.indexAtS(dist);
    return [this.lat[i], this.lon[i]];
  }

  /** roll (left side up positive) at arc length */
  rollAtS(dist: number) {
    const i = this.indexAtS(dist);
    return this.roll[i];
  }

  /** time at which the drive first reaches arc length dist */
  timeAtS(dist: number) {
    return this.atS(dist).t;
  }
}

export async function loadDrive(url: string): Promise<Drive> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`drive.json: ${res.status} ${res.statusText}`);
  return new Drive((await res.json()) as DriveJson);
}
