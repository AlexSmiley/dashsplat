// Director looks for captured shots (?director=<look>[,<look>], off by default; README "Capture"). Each is a
// function of drive time, so a capture is repeatable frame for frame; "amount" (0..1, keyed by tools/capture.ts as
// "dir") blends a look in and out.
//
//   points     every splat's centre as an opaque dot a few pixels across; amount 1 = dots, 0 = the full splats
//   depth      splats coloured by distance from the camera (warm white near, the accent blue, navy far), dark sky
//   height     splats coloured by height above the road (slate, blue, pale cyan, warm white, amber), dark sky
//   tiles      the splat tiles' bounds light up as the car comes within reach of them (drive order, not network
//              order) and each tile's splats fade in with its box
//   atlas      the ground atlas alone on a dark ground: no splats, no sky; its tiles appear in driving order
//   atlasmask  the atlas's observed / predicted mask (its "Predicted" overlay), amount = overlay strength
//   tracks     every tracked vehicle as a box with its track behind it as a ribbon on the road (moving ones
//              blue, parked ones outlined white, predicted stretches dashed)
// The wipe between two configurations is done on registered captures (tools/wipe.ts), not in the page.

import * as THREE from "three";
import { LineMaterial } from "three/examples/jsm/lines/LineMaterial.js";
import { LineSegments2 } from "three/examples/jsm/lines/LineSegments2.js";
import { LineSegmentsGeometry } from "three/examples/jsm/lines/LineSegmentsGeometry.js";
import type { Drive } from "../scene/drive";
import type { MapTile } from "../scene/tiles";

export type DirectorLook = "points" | "depth" | "height" | "tiles" | "atlas" | "atlasmask" | "tracks";
const LOOKS: DirectorLook[] = ["points", "depth", "height", "tiles", "atlas", "atlasmask", "tracks"];

const ACCENT = new THREE.Color(0.3, 0.56, 1.0);
const ease = (x: number) => {
  const t = THREE.MathUtils.clamp(x, 0, 1);
  return t * t * (3 - 2 * t);
};

export type Track = {
  id: string;
  pose: { x: number; y: number; z: number; yaw: number; ghost: number };
  at: (t: number) => { x: number; y: number; z: number; yaw: number; ghost: number };
  t_first: number;
  t_last: number;
  presence: number;
  size: number[];
  moving: boolean;
};

export class Director {
  readonly looks: Set<DirectorLook>;
  /** how much of the look (0..1; capture key "dir") */
  amount = 1;
  readonly group = new THREE.Group();
  /** metres from the car within which a splat tile / an atlas tile arrives */
  readonly reach: number;
  readonly atlasReach: number;
  /** a dark backdrop inside the sky, for the looks that want one */
  readonly backdrop: THREE.Mesh;
  readonly dark = new THREE.Color(0.02, 0.026, 0.04);
  private arrivals = new Map<string, number>();
  private boxes?: { lines: LineSegments2; tiles: MapTile[]; colors: Float32Array };
  private trackLines?: LineSegments2;
  private ribbons?: THREE.Mesh;
  private readonly lineMats: { m: LineMaterial; w: number }[] = [];

  constructor(spec: string, private readonly drive: Drive, params: URLSearchParams) {
    this.looks = new Set(spec.split(",").filter((s): s is DirectorLook => (LOOKS as string[]).includes(s)));
    this.reach = Number(params.get("dreach") ?? 100);
    this.atlasReach = Number(params.get("dreach") ?? 55);
    this.backdrop = new THREE.Mesh(
      new THREE.SphereGeometry(380, 32, 16),
      new THREE.MeshBasicMaterial({ color: this.dark, side: THREE.BackSide, transparent: true, opacity: 0, depthWrite: false, depthTest: false, fog: false }),
    );
    this.backdrop.renderOrder = -999;
    this.backdrop.frustumCulled = false;
    this.backdrop.visible = false;
    this.group.add(this.backdrop);
    this.group.name = "director";
  }

  get on() {
    return this.looks.size > 0;
  }
  has(l: DirectorLook) {
    return this.looks.has(l);
  }
  /** the splat shader's look: 1 points, 2 depth, 3 height */
  get splatLook() {
    return this.has("points") ? 1 : this.has("depth") ? 2 : this.has("height") ? 3 : 0;
  }
  /** the looks drawn without the ground atlas (the ground splats carry the look) */
  get hidesAtlas() {
    return this.splatLook > 0;
  }
  /** how dark the background goes (depth and height looks, scaled by the amount; the atlas look fully) */
  get darkness() {
    if (this.has("atlas")) return 1;
    return this.splatLook >= 2 ? this.amount : 0;
  }

  /** the first drive time the car comes within r of a rectangle (from the drive's own samples) */
  private arrival(key: string, x0: number, y0: number, x1: number, y1: number, r: number) {
    const k = key + ":" + r;
    let a = this.arrivals.get(k);
    if (a !== undefined) return a;
    const d = this.drive;
    a = Infinity;
    for (let i = 0; i < d.n; i += 2) {
      const dx = Math.max(x0 - d.x[i], 0, d.x[i] - x1), dy = Math.max(y0 - d.y[i], 0, d.y[i] - y1);
      if (dx * dx + dy * dy < r * r) {
        a = d.t[i];
        break;
      }
    }
    this.arrivals.set(k, a);
    return a;
  }

  /** a splat tile's presence at T (the tiles look) */
  tilePresence(t: MapTile, T: number) {
    const b = t.bbox as number[] | undefined;
    if (!b) return 1;
    return ease((T - this.arrival(t.id, b[0], b[1], b[3], b[4], this.reach)) / 0.7);
  }
  /** an atlas tile's presence at T (the atlas look) */
  atlasPresence(bounds: [number, number, number, number], T: number) {
    return ease((T - this.arrival(`a${bounds[0].toFixed(2)},${bounds[1].toFixed(2)}`, bounds[0], bounds[1], bounds[2], bounds[3], this.atlasReach)) / 0.6);
  }

  private lineMat(width: number) {
    const m = new LineMaterial({ vertexColors: true, linewidth: width, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, worldUnits: false });
    this.lineMats.push({ m, w: width });
    return m;
  }

  /** per frame */
  update(T: number, cam: THREE.Camera, size: THREE.Vector2, tiles: MapTile[] | null, tracks: Iterable<Track> | null) {
    // line widths in pixels of a 1080-line frame, whatever the capture size
    for (const { m, w } of this.lineMats) {
      m.resolution.copy(size);
      m.linewidth = (w * size.y) / 1080;
    }
    this.backdrop.position.copy(cam.position);
    const dk = this.darkness;
    (this.backdrop.material as THREE.MeshBasicMaterial).opacity = dk;
    this.backdrop.visible = dk > 0.001;
    if (this.has("tiles") && tiles) this.updateBoxes(T, tiles);
    if (this.has("tracks") && tracks) this.updateTracks(T, tracks);
  }

  /** the tiles look: every tile's bounds, lit as it arrives and glowing down to a faint outline */
  private updateBoxes(T: number, tiles: MapTile[]) {
    if (!this.boxes) {
      const list = tiles.filter((t) => t.bbox && t.kind !== "transient");
      const pos: number[] = [];
      for (const t of list) {
        const [x0, y0, z0, x1, y1, z1raw] = t.bbox as number[];
        // flat ground tiles as a rectangle just above the road; others as boxes
        const z1 = z1raw - z0 < 0.6 ? z0 + 0.05 : z1raw;
        const za = z1raw - z0 < 0.6 ? z0 + 0.05 : z0;
        const c = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
        for (let k = 0; k < 4; k++) {
          const [ax, ay] = c[k], [bx, by] = c[(k + 1) % 4];
          pos.push(ax, ay, za, bx, by, za);
          if (z1 > za + 0.1) pos.push(ax, ay, z1, bx, by, z1, ax, ay, za, ax, ay, z1);
        }
      }
      const g = new LineSegmentsGeometry();
      g.setPositions(pos);
      const colors = new Float32Array(pos.length);
      g.setColors(colors);
      const lines = new LineSegments2(g, this.lineMat(1.6));
      lines.renderOrder = 30;
      lines.frustumCulled = false;
      this.group.add(lines);
      this.boxes = { lines, tiles: list, colors };
    }
    const { tiles: list, colors, lines } = this.boxes;
    let o = 0;
    for (const t of list) {
      const b = t.bbox as number[];
      const n = (b[5] - b[2] < 0.6 ? 4 : 12) * 2;
      const a = this.arrival(t.id, b[0], b[1], b[3], b[4], this.reach);
      const dt = T - a;
      const k = dt < 0 ? 0 : (0.16 + 0.84 * Math.exp(-dt / 0.9)) * ease(dt / 0.15) * this.amount;
      for (let v = 0; v < n; v++, o += 3) {
        colors[o] = ACCENT.r * k;
        colors[o + 1] = ACCENT.g * k;
        colors[o + 2] = ACCENT.b * k;
      }
    }
    (lines.geometry as LineSegmentsGeometry).setColors(colors);
  }

  /** the tracks look: boxes and the track behind each vehicle (the last 6 s) as a ribbon on the road */
  private updateTracks(T: number, tracks: Iterable<Track>) {
    if (!this.trackLines) {
      this.trackLines = new LineSegments2(new LineSegmentsGeometry(), this.lineMat(2.2));
      this.trackLines.renderOrder = 31;
      this.trackLines.frustumCulled = false;
      const mat = new THREE.ShaderMaterial({
        transparent: true,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
        polygonOffset: true,
        polygonOffsetFactor: -4,
        vertexShader: /* glsl */ `
          attribute vec4 rgba;
          attribute float dist;
          varying vec4 vC;
          varying float vD;
          void main() { vC = rgba; vD = dist; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
        fragmentShader: /* glsl */ `
          varying vec4 vC;
          varying float vD;
          void main() {
            // predicted stretches (alpha marked negative) dashed every 1.6 m
            float a = abs(vC.a);
            if (vC.a < 0.0 && fract(vD / 1.6) > 0.55) discard;
            gl_FragColor = vec4(vC.rgb * a, a);
          }`,
      });
      this.ribbons = new THREE.Mesh(new THREE.BufferGeometry(), mat);
      this.ribbons.renderOrder = 29;
      this.ribbons.frustumCulled = false;
      this.group.add(this.trackLines, this.ribbons);
    }
    const seg: number[] = [], col: number[] = [];
    const rp: number[] = [], rc: number[] = [], rd: number[] = [], idx: number[] = [];
    const white = new THREE.Color(0.85, 0.87, 0.9);
    for (const tr of tracks) {
      if (tr.presence <= 0.01 || T < tr.t_first) continue;
      const c = tr.moving ? ACCENT : white;
      const p = tr.pose;
      const [L, W, H] = tr.size;
      const k = tr.presence * this.amount * (p.ghost > 0.5 ? 0.5 : 1);
      // box: footprint, roof, uprights
      const cs = Math.cos(p.yaw), sn = Math.sin(p.yaw);
      const corner = (u: number, v: number, z: number) => [p.x + cs * u * L * 0.5 - sn * v * W * 0.5, p.y + sn * u * L * 0.5 + cs * v * W * 0.5, p.z + z];
      const q = [[1, 1], [1, -1], [-1, -1], [-1, 1]];
      for (let i = 0; i < 4; i++) {
        const a = q[i], b = q[(i + 1) % 4];
        for (const z of [0.03, H]) seg.push(...corner(a[0], a[1], z), ...corner(b[0], b[1], z));
        seg.push(...corner(a[0], a[1], 0.03), ...corner(a[0], a[1], H));
      }
      for (let i = 0; i < 12 * 2; i++) col.push(c.r * k, c.g * k, c.b * k);
      // the ribbon: its track over the last 6 s, fading with age, 0.45 m wide, on the road
      if (!tr.moving) continue;
      const t0 = Math.max(tr.t_first, T - 6);
      const n = Math.max(2, Math.ceil((T - t0) / 0.1));
      let dist = 0, prev: { x: number; y: number } | null = null;
      const base = rp.length / 3;
      for (let i = 0; i <= n; i++) {
        const t = t0 + ((T - t0) * i) / n;
        const s = i === n ? p : tr.at(t);
        if (prev) dist += Math.hypot(s.x - prev.x, s.y - prev.y);
        prev = s;
        const nx = -Math.sin(s.yaw) * 0.225, ny = Math.cos(s.yaw) * 0.225;
        const age = ease(1 - (T - t) / 6);
        const a = 0.55 * age * tr.presence * this.amount * (s.ghost > 0.5 ? -1 : 1);
        rp.push(s.x + nx, s.y + ny, s.z + 0.06, s.x - nx, s.y - ny, s.z + 0.06);
        rc.push(c.r, c.g, c.b, a, c.r, c.g, c.b, a);
        rd.push(dist, dist);
        if (i > 0) {
          const v = base + (i - 1) * 2;
          idx.push(v, v + 1, v + 2, v + 1, v + 3, v + 2);
        }
      }
    }
    const lg = new LineSegmentsGeometry();
    if (seg.length) {
      lg.setPositions(seg);
      lg.setColors(col);
    }
    this.trackLines.geometry.dispose();
    this.trackLines.geometry = lg;
    this.trackLines.visible = seg.length > 0;
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(rp, 3));
    g.setAttribute("rgba", new THREE.Float32BufferAttribute(rc, 4));
    g.setAttribute("dist", new THREE.Float32BufferAttribute(rd, 1));
    g.setIndex(idx);
    this.ribbons!.geometry.dispose();
    this.ribbons!.geometry = g;
    this.ribbons!.visible = idx.length > 0;
  }
}
