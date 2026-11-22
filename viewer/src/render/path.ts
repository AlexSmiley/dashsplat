// Planned-path ribbon ahead of the car (blue while self-driving, grey otherwise), upright chevrons at upcoming
// turns, and the whole-route line for the map view. All derived from the future part of drive.json.

import * as THREE from "three";
import { Drive, wrapAngle } from "../scene/drive";

export type Turn = { s: number; dir: number; x: number; y: number; hin: number; hout: number; angle: number };

/** turns along the route: heading change of more than ~50 degrees across 24 m of path */
export function findTurns(drive: Drive): Turn[] {
  const step = 1;
  const n = Math.floor(drive.length / step);
  const h = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const a = drive.atS(Math.max(0, i * step - 1));
    const b = drive.atS(Math.min(drive.length, i * step + 1));
    h[i] = Math.atan2(b.y - a.y, b.x - a.x);
  }
  const W = 12;
  const d = new Float64Array(n);
  for (let i = W; i < n - W; i++) d[i] = wrapAngle(h[i + W] - h[i - W]);
  const out: Turn[] = [];
  for (let i = W; i < n - W; i++) {
    const v = Math.abs(d[i]);
    if (v < 0.9) continue;
    let max = true;
    for (let k = Math.max(0, i - 30); k < Math.min(n, i + 30); k++) if (Math.abs(d[k]) > v) { max = false; break; }
    if (!max) continue;
    if (out.length && i * step - out[out.length - 1].s < 40) continue;
    const p = drive.atS(i * step);
    out.push({ s: i * step, dir: Math.sign(d[i]), x: p.x, y: p.y, hin: h[i - W], hout: h[i + W], angle: d[i] });
  }
  return out;
}

/** plausible posted limit (mph) for each metre of route: 85th-percentile speed nearby, rounded down to 5 */
export function speedLimits(drive: Drive): Uint8Array {
  const n = Math.ceil(drive.length) + 1;
  const vmax = new Float32Array(n);
  for (let i = 0; i < drive.n; i++) {
    const k = Math.min(n - 1, Math.round(drive.s[i]));
    vmax[k] = Math.max(vmax[k], drive.speed[i]);
  }
  const out = new Uint8Array(n);
  const R = 250;
  for (let k = 0; k < n; k++) {
    const vals: number[] = [];
    for (let j = Math.max(0, k - R); j < Math.min(n, k + R); j += 5) if (vmax[j] > 0) vals.push(vmax[j]);
    vals.sort((a, b) => a - b);
    const p = vals.length ? vals[Math.floor(vals.length * 0.85)] : 0;
    const mph = p * 2.23694;
    out[k] = Math.max(15, Math.floor((mph + 2) / 5) * 5);
  }
  return out;
}

const ribbonVert = /* glsl */ `
attribute vec2 aUV;
varying vec2 vUV;
varying float vDist;
void main() {
  vUV = aUV;
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  vDist = -mv.z;
  gl_Position = projectionMatrix * mv;
  // depth as if 0.7 m (+2.5 % of the distance) nearer: ground splats (whose whole footprint is drawn at their centre's depth) cannot
  // cut into the band, anything genuinely in front of it still covers it
  float L = length(mv.xyz);
  vec4 pulled = projectionMatrix * vec4(mv.xyz * max(0.2, 1.0 - (0.7 + 0.025 * L) / L), 1.0);
  gl_Position.z = pulled.z / pulled.w * gl_Position.w;
}`;
const ribbonFrag = /* glsl */ `
uniform vec3 color;
uniform float opacity;
uniform vec3 fogColor;
uniform vec2 fogRange;
varying vec2 vUV;
varying float vDist;
void main() {
  float a = opacity;
  // emerges from under the nose, fades out ahead
  a *= smoothstep(0.0, 0.06, vUV.x);
  a *= 1.0 - smoothstep(0.3, 1.0, vUV.x);
  // soft edges with a slightly brighter core, like Tesla's band
  float e = abs(vUV.y);
  a *= 1.0 - smoothstep(0.72, 1.0, e);
  vec3 c = color * (1.0 + 0.22 * (1.0 - smoothstep(0.0, 0.7, e))) ;
  // brighter and a touch lighter near the car
  c = mix(c, vec3(0.55, 0.75, 1.0), 0.35 * (1.0 - smoothstep(0.0, 0.35, vUV.x)));
  c = mix(c, fogColor, 0.55 * smoothstep(fogRange.x, fogRange.y, vDist));
  // drawn before the splats with depth writes, so anything in front of it (a car, a pole) covers it
  if (a < 0.1) discard;
  gl_FragColor = vec4(c, a);
}`;

const SEG = 120;

export class PathViz {
  readonly ribbon: THREE.Mesh;
  readonly chevrons = new THREE.Group();
  readonly route: THREE.Group = new THREE.Group();
  private pos: Float32Array;
  private mat: THREE.ShaderMaterial;
  private length = 30;
  private chevMat: THREE.MeshBasicMaterial;
  private routePast?: THREE.Mesh;
  private routeFuture?: THREE.Mesh;
  private routeMat: THREE.MeshBasicMaterial;
  private routeMatPast: THREE.MeshBasicMaterial;

  constructor(private readonly drive: Drive, readonly turns: Turn[]) {
    const g = new THREE.BufferGeometry();
    this.pos = new Float32Array((SEG + 1) * 2 * 3);
    const uv = new Float32Array((SEG + 1) * 2 * 2);
    const idx: number[] = [];
    for (let i = 0; i <= SEG; i++) {
      uv.set([i / SEG, -1, i / SEG, 1], i * 4);
      if (i < SEG) {
        const a = i * 2;
        idx.push(a, a + 1, a + 2, a + 1, a + 3, a + 2);
      }
    }
    g.setAttribute("position", new THREE.BufferAttribute(this.pos, 3).setUsage(THREE.DynamicDrawUsage));
    g.setAttribute("aUV", new THREE.BufferAttribute(uv, 2));
    g.setIndex(idx);
    this.mat = new THREE.ShaderMaterial({
      vertexShader: ribbonVert,
      fragmentShader: ribbonFrag,
      uniforms: {
        color: { value: new THREE.Color(0.17, 0.42, 1.0) },
        opacity: { value: 0.9 },
        fogColor: { value: new THREE.Color(0.78, 0.81, 0.83) },
        fogRange: { value: new THREE.Vector2(40, 200) },
      },
      transparent: true,
      depthWrite: true,
    });
    this.ribbon = new THREE.Mesh(g, this.mat);
    this.ribbon.frustumCulled = false;
    // after the ground apron, before the splats
    this.ribbon.renderOrder = -50;
    this.ribbon.name = "path-ribbon";

    // chevrons: flat ">" shapes standing upright, pointing +x in their own frame
    const shape = new THREE.Shape();
    shape.moveTo(-0.32, 0.55);
    shape.lineTo(0.05, 0.55);
    shape.lineTo(0.42, 0);
    shape.lineTo(0.05, -0.55);
    shape.lineTo(-0.32, -0.55);
    shape.lineTo(0.05, 0);
    shape.closePath();
    const cg = new THREE.ShapeGeometry(shape);
    // stand it up: shape x stays horizontal (pointing), shape y becomes world z
    cg.rotateX(Math.PI / 2);
    this.chevMat = new THREE.MeshBasicMaterial({ color: new THREE.Color(0.55, 0.78, 1.0), transparent: true, opacity: 0, depthWrite: false, side: THREE.DoubleSide, toneMapped: false });
    for (let i = 0; i < 4; i++) {
      const m = new THREE.Mesh(cg, this.chevMat);
      m.renderOrder = 11;
      this.chevrons.add(m);
    }
    this.chevrons.name = "chevrons";

    this.routeMat = new THREE.MeshBasicMaterial({ color: new THREE.Color(0.2, 0.45, 1.0), transparent: true, opacity: 0.95, depthWrite: false, depthTest: false, toneMapped: false });
    this.routeMatPast = new THREE.MeshBasicMaterial({ color: new THREE.Color(0.55, 0.58, 0.62), transparent: true, opacity: 0.85, depthWrite: false, depthTest: false, toneMapped: false });
    this.route.name = "route";
    this.route.visible = false;
  }

  /** builds a ground strip along the route between arc lengths a and b */
  private strip(a: number, b: number, width: number, mat: THREE.Material) {
    const n = Math.max(2, Math.ceil((b - a) / 3));
    const pos = new Float32Array((n + 1) * 6);
    const idx: number[] = [];
    for (let i = 0; i <= n; i++) {
      const s = a + ((b - a) * i) / n;
      const p = this.drive.atS(s);
      const q = this.drive.atS(Math.min(this.drive.length, s + 1));
      const r = this.drive.atS(Math.max(0, s - 1));
      const h = Math.atan2(q.y - r.y, q.x - r.x);
      const lx = -Math.sin(h) * width / 2, ly = Math.cos(h) * width / 2;
      pos.set([p.x + lx, p.y + ly, p.z + 0.3, p.x - lx, p.y - ly, p.z + 0.3], i * 6);
      if (i < n) idx.push(i * 2, i * 2 + 1, i * 2 + 2, i * 2 + 1, i * 2 + 3, i * 2 + 2);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    g.setIndex(idx);
    const m = new THREE.Mesh(g, mat);
    m.frustumCulled = false;
    m.renderOrder = 12;
    return m;
  }

  /** map view route line; rebuilt when the width (zoom) or the current position changes enough */
  private routeKey = "";
  updateRoute(sNow: number, width: number) {
    const key = Math.round(sNow / 5) + ":" + width.toFixed(1);
    if (key === this.routeKey) return;
    this.routeKey = key;
    for (const m of [this.routePast, this.routeFuture]) if (m) { this.route.remove(m); m.geometry.dispose(); }
    this.routePast = sNow > 1 ? this.strip(0, sNow, width * 0.8, this.routeMatPast) : undefined;
    this.routeFuture = sNow < this.drive.length - 1 ? this.strip(sNow, this.drive.length, width, this.routeMat) : undefined;
    if (this.routePast) this.route.add(this.routePast);
    if (this.routeFuture) this.route.add(this.routeFuture);
  }

  update(sNow: number, speed: number, selfDriving: boolean, dt: number, fogColor: THREE.Color, fog: [number, number]) {
    // the ribbon starts at the front bumper and reaches further at speed
    const targetLen = THREE.MathUtils.clamp(14 + speed * 3.2, 14, 75);
    this.length += (targetLen - this.length) * (1 - Math.exp(-dt / 0.6));
    const s0 = sNow + 3.1;
    const s1 = Math.min(this.drive.length, s0 + this.length);
    const w = 0.98;
    const visible = s1 - s0 > 2;
    this.ribbon.visible = visible;
    if (visible) {
      for (let i = 0; i <= SEG; i++) {
        const s = s0 + ((s1 - s0) * i) / SEG;
        const p = this.drive.atS(s);
        const q = this.drive.atS(Math.min(this.drive.length, s + 0.8));
        const r = this.drive.atS(Math.max(0, s - 0.8));
        const h = Math.atan2(q.y - r.y, q.x - r.x);
        const lx = -Math.sin(h) * w, ly = Math.cos(h) * w;
        // lie on the road: the drive's height plus its cross-slope (roll), a few centimetres up
        const tilt = Math.sin(this.drive.rollAtS(s)) * w;
        this.pos.set([p.x + lx, p.y + ly, p.z + 0.035 + tilt, p.x - lx, p.y - ly, p.z + 0.035 - tilt], i * 6);
      }
      this.ribbon.geometry.attributes.position.needsUpdate = true;
    }
    const c = this.mat.uniforms.color.value as THREE.Color;
    const target = selfDriving ? new THREE.Color(0.17, 0.42, 1.0) : new THREE.Color(0.7, 0.72, 0.76);
    c.lerp(target, 1 - Math.exp(-dt / 0.25));
    this.mat.uniforms.opacity.value += ((selfDriving ? 0.9 : 0.62) - this.mat.uniforms.opacity.value) * (1 - Math.exp(-dt / 0.25));
    (this.mat.uniforms.fogColor.value as THREE.Color).copy(fogColor);
    (this.mat.uniforms.fogRange.value as THREE.Vector2).set(fog[0], fog[1]);

    // chevrons at the next turn
    const next = this.turns.find((t) => t.s > sNow + 6);
    let op = 0;
    if (next) {
      const d = next.s - sNow;
      op = THREE.MathUtils.clamp((150 - d) / 30, 0, 1) * THREE.MathUtils.clamp((d - 6) / 8, 0, 1);
      if (op > 0) {
        const fx = Math.cos(next.hin), fy = Math.sin(next.hin);
        const ox = Math.cos(next.hout), oy = Math.sin(next.hout);
        // a row across the far side of the junction, facing the approaching car, pointing into the turn
        const cx = next.x + fx * 9, cy = next.y + fy * 9;
        this.chevrons.children.forEach((m, i) => {
          const k = i - 0.5;
          m.position.set(cx + ox * k * 1.25, cy + oy * k * 1.25, 2.1);
          m.rotation.set(0, 0, next.hout);
        });
      }
    }
    this.chevMat.opacity += (op * 0.92 - this.chevMat.opacity) * (1 - Math.exp(-dt / 0.2));
    this.chevrons.visible = this.chevMat.opacity > 0.01;
  }
}
