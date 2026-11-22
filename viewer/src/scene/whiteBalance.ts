// Per-scene white balance for the "Natural" grade: the camera footage has a teal/green cast, so find the
// colour of the road surface (asphalt next to the driven path, excluding bright paint) and compute channel
// gains that make it neutral grey at the same brightness. Shared by the viewer (live estimate from loaded
// tiles, cached in localStorage) and tools/build-tile-cache.ts (stored in rsb/index.json).

export type WhiteBalance = { gains: [number, number, number]; road: [number, number, number]; samples: number };

/** spatial index of drive path points, for "is this splat on the driven road?" */
export class PathIndex {
  private cells = new Map<number, number[]>();
  constructor(xs: ArrayLike<number>, ys: ArrayLike<number>, zs: ArrayLike<number>, private readonly cell = 4) {
    let lx = Infinity, ly = Infinity;
    for (let i = 0; i < xs.length; i++) {
      if (Math.hypot(xs[i] - lx, ys[i] - ly) < 1.5) continue;
      lx = xs[i];
      ly = ys[i];
      const k = this.key(Math.floor(lx / cell), Math.floor(ly / cell));
      let a = this.cells.get(k);
      if (!a) this.cells.set(k, (a = []));
      a.push(lx, ly, zs[i] ?? 0);
    }
  }
  private key(ix: number, iy: number) {
    return ix * 73856093 + iy * 19349663;
  }
  /** horizontal distance to the nearest path point and the height difference to it (within 2 cells) */
  near(x: number, y: number, z: number, maxD: number) {
    const ix = Math.floor(x / this.cell), iy = Math.floor(y / this.cell);
    let best = Infinity, dz = 0;
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      const a = this.cells.get(this.key(ix + dx, iy + dy));
      if (!a) continue;
      for (let j = 0; j < a.length; j += 3) {
        const d = Math.hypot(a[j] - x, a[j + 1] - y);
        if (d < best) { best = d; dz = z - a[j + 2]; }
      }
    }
    return best <= maxD ? { d: best, dz } : null;
  }
}

/** collects candidate road colours; feed it splats (display rgb 0..1, opacity, world position) */
export class RoadColour {
  private r: number[] = [];
  private g: number[] = [];
  private b: number[] = [];
  constructor(private readonly path: PathIndex, private readonly max = 120000) {}
  get count() {
    return this.r.length;
  }
  add(r: number, g: number, b: number, a: number, x: number, y: number, z: number) {
    if (this.r.length >= this.max || a < 0.35) return;
    const L = (r + g + b) / 3;
    if (L < 0.06 || L > 0.6) return; // shadows and paint
    if (Math.max(r, g, b) - Math.min(r, g, b) > 0.22) return; // vegetation, signs
    const n = this.path.near(x, y, z, 2.6);
    if (!n || Math.abs(n.dz) > 0.6) return;
    this.r.push(r);
    this.g.push(g);
    this.b.push(b);
  }
  result(): WhiteBalance | null {
    if (this.r.length < 2000) return null;
    const med = (a: number[]) => {
      const s = Float32Array.from(a).sort();
      return s[s.length >> 1];
    };
    const road: [number, number, number] = [med(this.r), med(this.g), med(this.b)];
    const L = (road[0] + road[1] + road[2]) / 3;
    const clamp = (v: number) => Math.min(1.4, Math.max(0.72, v));
    return { gains: [clamp(L / road[0]), clamp(L / road[1]), clamp(L / road[2])], road, samples: this.r.length };
  }
}

const h16 = new Uint16Array(1);
const hf = new Float32Array(1);
const hu = new Uint32Array(hf.buffer);
/** IEEE half bits -> float */
export function fromHalf(h: number) {
  h16[0] = h;
  const s = (h >> 15) & 1, e = (h >> 10) & 31, f = h & 1023;
  if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
  if (e === 31) return f ? NaN : s ? -Infinity : Infinity;
  hu[0] = (s << 31) | ((e - 15 + 127) << 23) | (f << 13);
  return hf[0];
}
