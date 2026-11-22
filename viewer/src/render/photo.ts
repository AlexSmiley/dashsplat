// Photographic layers (pipeline/ibr/FORMAT.md, "photo-layers" v1): photographs chosen by which real exposure was
// taken nearest the virtual camera, drawn as a world-aligned sky at infinity, one blended backdrop curtain per
// selected knot, upright canonical cut-outs and per-exposure movers. Any splat scene can borrow another scene's
// layers: everything here lives in the layers' own (photo) frame, tied to the viewer's scene by a per-time
// transform that maps the photo drive's car pose onto the scene drive's car pose.
//
// Differences from the reference renderer (layer_player.py), chosen to look the same at the default chase pose:
//  - ground depth for the backdrop rule is the analytic road plane (drive z - 0.08 m), not the opacity-weighted
//    depth of the ground splats;
//  - the backdrop writes depth where its alpha > 0.5 and the splats are depth-tested against it, instead of being
//    composited over a ground depth buffer; cut-outs and movers are drawn before the splats with a depth prepass
//    (alpha > 0.5), so splats in front of a card cover it (and splats behind its soft edge show through there).

import * as THREE from "three";
import { Drive } from "../scene/drive";
import { GRADE_GLSL, type SplatGlobals } from "./splatShader";

type BaseRec = { t: number; n: number; s: number; pose: number[]; pano: number };
type Pano = { id: number; t: number; origin: [number, number, number]; radius: number; sky: string; rgb: string; matte: string };
type Card = {
  id: number;
  kind: number;
  plane: [number, number, number];
  extent: [number, number, number, number];
  rect: [number, number, number, number];
  rgb: string;
  matte: string;
  center: [number, number, number];
  source: [string, number, number];
  birth?: number;
  death?: number;
  fade?: number;
};
type Shard = { first: number; last: number; file: string };
type Exposure = { t: number; n: number; layers: Card[] };
export type PhotoJson = {
  format: string;
  version: number;
  contentRevision?: number;
  interval: [number, number];
  drive: string;
  skyElevation: [number, number];
  backdropElevation: [number, number];
  bases: Record<string, BaseRec[]>;
  panoramas: Pano[];
  cutouts: Card[];
  movers: Record<string, Shard[]>;
};
export type PhotoSwitches = { sky: boolean; backdrop: boolean; cutouts: boolean; movers: boolean };

const K = 4; // knots sampled at once (the reference selects at most 3)
const h = (x: number) => {
  const p = Math.min(1, Math.max(0, x));
  return p * p * (3 - 2 * p);
};
const D2R = Math.PI / 180;

// ---------------------------------------------------------------- textures: fetch, decode off-thread, premultiply

type Kind = "sky" | "far" | "card" | "mover";
type Tex = {
  kind: Kind;
  state: "loading" | "decoded" | "ready" | "failed";
  rgb?: ImageBitmap;
  matte?: ImageBitmap;
  tex?: THREE.Texture;
  rt?: THREE.WebGLRenderTarget;
  w: number;
  h: number;
  last: number;
  /** sky only: mean colour of the band just above the horizon */
  horizon?: THREE.Color;
  /** card atlases: the matte (one byte per pixel), for the cut-out sanity check */
  alpha?: Uint8Array;
};
const LIMIT: Record<Kind, number> = { sky: 40, far: 14, card: 260, mover: 160 };

class TexCache {
  readonly map = new Map<string, Tex>();
  private queue: { key: string; rgb: string; matte: string | null; kind: Kind; pri: number }[] = [];
  private inflight = 0;
  private frame = 0;
  private readonly pm: { scene: THREE.Scene; cam: THREE.Camera; mat: THREE.ShaderMaterial };
  skyElev: [number, number] = [-9, 64];

  constructor(private readonly base: string) {
    const mat = new THREE.ShaderMaterial({
      uniforms: { rgb: { value: null }, matte: { value: null } },
      vertexShader: `varying vec2 vUv; void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }`,
      // row 0 of the target is the image's top row, as for the bitmaps: lookups use top-down v throughout
      fragmentShader: `uniform sampler2D rgb; uniform sampler2D matte; varying vec2 vUv;
        void main() { float a = texture2D(matte, vUv).r; gl_FragColor = vec4(texture2D(rgb, vUv).rgb * a, a); }`,
      blending: THREE.NoBlending,
      depthTest: false,
      depthWrite: false,
    });
    const scene = new THREE.Scene();
    const q = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), mat);
    q.frustumCulled = false;
    scene.add(q);
    this.pm = { scene, cam: new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1), mat };
  }

  get pending() {
    return this.queue.length + this.inflight + [...this.map.values()].filter((t) => t.state === "decoded").length;
  }

  /** a texture if it is ready; otherwise queue it (lower pri = sooner) */
  want(key: string, rgb: string, matte: string | null, kind: Kind, pri: number): Tex | null {
    const t = this.map.get(key);
    if (t) {
      t.last = this.frame;
      return t.state === "ready" ? t : null;
    }
    const q = this.queue.find((x) => x.key === key);
    if (q) q.pri = Math.min(q.pri, pri);
    else this.queue.push({ key, rgb, matte, kind, pri });
    return null;
  }
  get(key: string) {
    const t = this.map.get(key);
    if (t?.state === "ready") {
      t.last = this.frame;
      return t;
    }
    return null;
  }

  private async decode(url: string) {
    const r = await fetch(new URL(url, this.base).href);
    if (!r.ok) throw new Error(`${url}: ${r.status}`);
    // raw code values: no colour management, no premultiplication (FORMAT.md, texture conventions)
    return createImageBitmap(await r.blob(), { colorSpaceConversion: "none", premultiplyAlpha: "none" });
  }

  /** start fetches, upload decoded images (a budget per frame), evict what has not been used for a while */
  pump(renderer: THREE.WebGLRenderer) {
    this.frame++;
    this.queue.sort((a, b) => a.pri - b.pri);
    while (this.inflight < 8 && this.queue.length) {
      const j = this.queue.shift()!;
      const t: Tex = { kind: j.kind, state: "loading", w: 0, h: 0, last: this.frame };
      this.map.set(j.key, t);
      this.inflight++;
      Promise.all([this.decode(j.rgb), j.matte ? this.decode(j.matte) : Promise.resolve(undefined)])
        .then(([rgb, matte]) => {
          if (this.map.get(j.key) !== t) {
            rgb.close();
            matte?.close();
            return;
          }
          t.rgb = rgb;
          t.matte = matte;
          t.w = rgb.width;
          t.h = rgb.height;
          t.state = "decoded";
        })
        .catch((e) => {
          console.warn("photo layer image failed", e);
          t.state = "failed";
        })
        .finally(() => this.inflight--);
    }
    // uploads: one backdrop or several small images per frame
    let budget = 4;
    for (const t of this.map.values()) {
      if (budget <= 0) break;
      if (t.state !== "decoded") continue;
      budget -= t.kind === "far" ? 4 : 1;
      this.upload(renderer, t);
    }
    // eviction, per kind, least recently used first
    for (const kind of Object.keys(LIMIT) as Kind[]) {
      const ls = [...this.map.entries()].filter(([, t]) => t.kind === kind && t.state !== "loading");
      if (ls.length <= LIMIT[kind]) continue;
      ls.sort((a, b) => a[1].last - b[1].last);
      for (const [key, t] of ls.slice(0, ls.length - LIMIT[kind])) {
        if (t.last >= this.frame - 2) break;
        t.tex?.dispose();
        t.rt?.dispose();
        t.rgb?.close();
        t.matte?.close();
        this.map.delete(key);
      }
    }
  }

  private upload(renderer: THREE.WebGLRenderer, t: Tex) {
    const src = (b: ImageBitmap) => {
      const x = new THREE.Texture(b);
      x.flipY = false;
      x.colorSpace = THREE.NoColorSpace;
      x.generateMipmaps = false;
      x.minFilter = x.magFilter = THREE.NearestFilter;
      x.needsUpdate = true;
      return x;
    };
    if (!t.matte) {
      // sky: the bitmap is the texture
      const x = src(t.rgb!);
      x.minFilter = x.magFilter = THREE.LinearFilter;
      x.wrapS = THREE.RepeatWrapping;
      x.wrapT = THREE.ClampToEdgeWrapping;
      t.tex = x;
      t.horizon = horizonColour(t.rgb!, this.skyElev);
      t.state = "ready";
      return;
    }
    if (t.kind === "card") t.alpha = matteBytes(t.matte!);
    // premultiply before any filtering, as the format asks (colour under a transparent matte never bleeds)
    const mip = t.kind === "card" || t.kind === "mover";
    const rt = new THREE.WebGLRenderTarget(t.w, t.h, {
      depthBuffer: false,
      stencilBuffer: false,
      generateMipmaps: mip,
      minFilter: mip ? THREE.LinearMipmapLinearFilter : THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
    });
    rt.texture.colorSpace = THREE.NoColorSpace;
    rt.texture.wrapS = t.kind === "far" ? THREE.RepeatWrapping : THREE.ClampToEdgeWrapping;
    const a = src(t.rgb!), b = src(t.matte!);
    this.pm.mat.uniforms.rgb.value = a;
    this.pm.mat.uniforms.matte.value = b;
    const prev = renderer.getRenderTarget();
    renderer.setRenderTarget(rt);
    renderer.render(this.pm.scene, this.pm.cam);
    renderer.setRenderTarget(prev);
    a.dispose();
    b.dispose();
    t.rgb!.close();
    t.matte!.close();
    t.rgb = t.matte = undefined;
    t.rt = rt;
    t.tex = rt.texture;
    t.state = "ready";
  }

  dispose() {
    for (const t of this.map.values()) {
      t.tex?.dispose();
      t.rt?.dispose();
      t.rgb?.close();
      t.matte?.close();
    }
    this.map.clear();
    this.queue = [];
    this.pm.mat.dispose();
  }
}

function matteBytes(b: ImageBitmap) {
  const c = new OffscreenCanvas(b.width, b.height);
  const x = c.getContext("2d", { willReadFrequently: true })!;
  x.drawImage(b, 0, 0);
  const px = x.getImageData(0, 0, b.width, b.height).data;
  const a = new Uint8Array(b.width * b.height);
  for (let i = 0; i < a.length; i++) a[i] = px[i * 4];
  return a;
}

/** how soft a card's matte is: of the pixels over 0.04, the share that is under 0.5 and more than 2 px from any
 *  solid pixel (a blur, not the anti-aliased rim of a crisp shape) */
function coverage(alpha: Uint8Array, w: number, rect: [number, number, number, number]) {
  const [L, T, W, H] = rect;
  let any = 0, solid = 0, blur = 0;
  const at = (x: number, y: number) => (x < L || y < T || x >= L + W || y >= T + H ? 0 : alpha[y * w + x]);
  for (let y = T; y < T + H; y++) for (let x = L; x < L + W; x++) {
    const a = alpha[y * w + x];
    if (a <= 10) continue;
    any++;
    if (a > 127) { solid++; continue; }
    let rim = false;
    for (let dy = -2; dy <= 2 && !rim; dy++) for (let dx = -2; dx <= 2; dx++) if (at(x + dx, y + dy) > 127) { rim = true; break; }
    if (!rim) blur++;
  }
  return { any, solid, blur: any ? blur / any : 1 };
}
const soft = (cv: { any: number; solid: number; blur: number }) =>
  cv.solid < 24 || cv.blur > 0.5 ? `matte mostly soft (${Math.round(cv.blur * 100)} % blurred, ${cv.solid} solid px)` : null;

function horizonColour(b: ImageBitmap, elev: [number, number]) {
  const c = new OffscreenCanvas(128, 32);
  const x = c.getContext("2d", { willReadFrequently: true });
  const out = new THREE.Color(0.8, 0.82, 0.84);
  if (!x) return out;
  x.drawImage(b, 0, 0, 128, 32);
  // elevation 0.5 to 4 degrees
  const r0 = Math.floor(((elev[1] - 4) / (elev[1] - elev[0])) * 32), r1 = Math.ceil(((elev[1] - 0.5) / (elev[1] - elev[0])) * 32);
  const px = x.getImageData(0, r0, 128, Math.max(1, r1 - r0)).data;
  let r = 0, g = 0, bl = 0;
  for (let i = 0; i < px.length; i += 4) { r += px[i]; g += px[i + 1]; bl += px[i + 2]; }
  const n = px.length / 4;
  return out.setRGB(r / n / 255, g / n / 255, bl / n / 255);
}

// ---------------------------------------------------------------- shaders

const backdropVert = /* glsl */ `
varying vec2 vNdc;
void main() {
  vNdc = position.xy;
  gl_Position = vec4(position.xy, 0.0, 1.0);
}`;

const backdropFrag = /* glsl */ `
uniform sampler2D far0;
uniform sampler2D far1;
uniform sampler2D far2;
uniform sampler2D far3;
uniform vec4 w;
uniform vec4 org[${K}];
uniform vec3 camP;
uniform mat3 camR;
uniform vec2 tanXY;
uniform vec2 nf;
uniform vec2 elev;
uniform float groundZ;
uniform float amount;
uniform vec4 GR;
uniform vec4 G2;
varying vec2 vNdc;
${GRADE_GLSL}
const float PI = 3.141592653589793;
void curtain(sampler2D tex, float wk, vec4 o, vec3 r, inout vec4 C, inout float zs) {
  if (wk <= 0.0) return;
  vec2 dxy = camP.xy - o.xy;
  float q = dot(r.xy, r.xy);
  float b = dot(dxy, r.xy);
  float disc = b * b - q * (dot(dxy, dxy) - o.w * o.w);
  if (disc <= 0.0 || q < 1e-8) return;
  float z = (-b + sqrt(disc)) / q;
  if (z <= 0.3) return;
  vec3 P = camP + z * r - o.xyz;
  float e = degrees(atan(P.z, o.w));
  if (e < elev.x || e > elev.y) return;
  vec2 uv = vec2(fract(0.5 + atan(P.y, P.x) / (2.0 * PI)), (elev.y - e) / (elev.y - elev.x));
  vec4 s = texture2D(tex, uv);
  C += wk * s;
  zs += wk * s.a * z;
}
void main() {
  // camera ray with forward component 1 (axial depths), in the photo frame
  vec3 r = camR * vec3(vNdc.x * tanXY.x, vNdc.y * tanXY.y, -1.0);
  float zg = 1e9;
  if (r.z < -0.001) { float t = (groundZ - camP.z) / r.z; if (t > 0.0) zg = t; }
  vec4 C = vec4(0.0);
  float zs = 0.0;
  curtain(far0, w.x, org[0], r, C, zs);
  curtain(far1, w.y, org[1], r, C, zs);
  curtain(far2, w.z, org[2], r, C, zs);
  curtain(far3, w.w, org[3], r, C, zs);
  float a = C.a;
  float Z = zs / max(a, 1e-5);
  // drawn only where it is nearer than the ground (+12 cm)
  if (a <= 0.001 || Z >= zg + 0.12) discard;
  vec3 c = rsGradePhoto(clamp(C.rgb / a, 0.0, 1.0), GR, G2, 0.0);
  float k = amount;
  gl_FragColor = vec4(c * a * k, a * k);
  // only the near-opaque backdrop occludes (splats behind it are hidden); the rest blends without depth
  float ndc = ((nf.y + nf.x) * Z - 2.0 * nf.y * nf.x) / ((nf.y - nf.x) * Z);
  gl_FragDepth = a > 0.5 && k > 0.5 ? 0.5 * ndc + 0.5 : 1.0;
}`;

const cardVert = /* glsl */ `
uniform vec3 centre;
varying vec2 vUv;
varying vec3 vP;
void main() {
  vUv = uv;
  vP = position + centre;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}`;

const cardFrag = /* glsl */ `
uniform sampler2D map;
uniform vec4 clampUv;
uniform vec2 nrm;
uniform float alpha;
uniform float prepass;
uniform vec3 camP;
uniform vec3 camF;
uniform vec4 GR;
uniform vec4 G2;
varying vec2 vUv;
varying vec3 vP;
${GRADE_GLSL}
void main() {
  vec3 rv = vP - camP;
  float z = dot(rv, camF);
  // plane depth rules of the format: in front of the near plane, not seen edge-on
  if (z <= 0.3 || abs(dot(nrm, rv.xy)) <= 0.025 * z) discard;
  vec4 s = texture2D(map, clamp(vUv, clampUv.xy, clampUv.zw));
  float a = s.a * alpha;
  if (prepass > 0.5) {
    if (a <= 0.5) discard;
    gl_FragColor = vec4(0.0);
    return;
  }
  if (a <= 0.002) discard;
  vec3 c = rsGradePhoto(clamp(s.rgb / max(s.a, 1e-4), 0.0, 1.0), GR, G2, 0.0);
  gl_FragColor = vec4(c * a, a);
}`;

// ---------------------------------------------------------------- selection

type Lens = { name: string; x: Float64Array; y: Float64Array; t: Float64Array; s: Float64Array; f: THREE.Vector3[]; pano: Int32Array };

function lens(name: string, rs: BaseRec[]): Lens {
  const n = rs.length;
  const L: Lens = { name, x: new Float64Array(n), y: new Float64Array(n), t: new Float64Array(n), s: new Float64Array(n), f: [], pano: new Int32Array(n) };
  const q = new THREE.Quaternion();
  rs.forEach((r, i) => {
    L.x[i] = r.pose[0];
    L.y[i] = r.pose[1];
    L.t[i] = r.t;
    L.s[i] = r.s;
    L.pano[i] = r.pano;
    q.set(r.pose[3], r.pose[4], r.pose[5], r.pose[6]);
    // sensor +Z forward
    L.f.push(new THREE.Vector3(0, 0, 1).applyQuaternion(q));
  });
  return L;
}

/** per-camera choice and knot weights (FORMAT.md "Source selection"); eye E and look direction D in the photo frame */
function select(lenses: Lens[], T: number, S: number, E: THREE.Vector3, D: THREE.Vector3) {
  const ch: { J: number; lens: Lens; i: number; f: number; angle: number }[] = [];
  const a = new THREE.Vector3();
  for (const L of lenses) {
    let best = Infinity, bi = 0, bf = 0, bL = 0;
    for (let i = 0; i + 1 < L.x.length; i++) {
      const ex = L.x[i], ey = L.y[i];
      const dx = L.x[i + 1] - ex, dy = L.y[i + 1] - ey;
      const f = Math.min(1, Math.max(0, ((E.x - ex) * dx + (E.y - ey) * dy) / Math.max(dx * dx + dy * dy, 1e-6)));
      const Ld = Math.hypot(ex + f * dx - E.x, ey + f * dy - E.y);
      const c = Ld + 0.015 * Math.abs(L.t[i] - T) + (Math.abs(L.s[i] - S) < 55 ? 0 : 1e5);
      if (c < best) { best = c; bi = i; bf = f; bL = Ld; }
    }
    a.copy(L.f[bi]).multiplyScalar(1 - bf).addScaledVector(L.f[bi + 1], bf).normalize();
    const angle = Math.acos(Math.min(1, Math.max(-1, a.dot(D)))) / D2R;
    const J = bL + 0.32 * angle + 0.025 * Math.abs((1 - bf) * L.t[bi] + bf * L.t[bi + 1] - T);
    ch.push({ J, lens: L, i: bi, f: bf, angle });
  }
  ch.sort((p, q) => p.J - q.J);
  const Jmin = ch[0]?.J ?? 0;
  let den = 0;
  const lw = ch.map((c) => { const v = h(1 - (c.J - Jmin) / 4); den += v; return v; });
  const knots = new Map<number, number>();
  ch.forEach((c, k) => {
    const wl = lw[k] / Math.max(den, 1e-9);
    if (wl <= 0) return;
    for (const [idx, share] of [[c.i, 1 - c.f], [c.i + 1, c.f]] as const) {
      if (share > 1e-6) knots.set(c.lens.pano[idx], (knots.get(c.lens.pano[idx]) ?? 0) + wl * share);
    }
  });
  return { knots, lens: ch[0]?.lens.name ?? "", angle: ch[0]?.angle ?? 0, lensWeights: ch.map((c, k) => [c.lens.name, +(lw[k] / den).toFixed(3)]) };
}

// ---------------------------------------------------------------- the layers

type CardMesh = { pre: THREE.Mesh; col: THREE.Mesh; u: Record<string, THREE.IUniform> };

export class PhotoLayers {
  /** cut-outs and movers, in the photo frame (the group's matrix maps it into the scene) */
  readonly group = new THREE.Group();
  readonly backdrop: THREE.Mesh;
  readonly drive: Drive;
  readonly interval: [number, number];
  readonly elevSky: [number, number];
  private readonly bu: Record<string, THREE.IUniform>;
  private readonly tex: TexCache;
  private readonly lenses: Lens[];
  private readonly arc: Float64Array;
  private readonly cards = new Map<number, CardMesh>();
  private readonly pool: CardMesh[] = [];
  private readonly shards: { name: string; list: Shard[]; loaded: Map<number, Exposure[] | "loading"> }[];
  private readonly lastMover = new Map<number, { t: number; card: Card }>();
  /** static cut-out kinds drawn (null = all): 2 building/wall, 4 fence, 5 pole, 6 light, 7 sign, 8 vegetation */
  kinds: Set<number> | null = null;
  /** debug: only these card ids */
  ids: Set<number> | null = null;
  /** chosen automatically (an ibr-* scene of the same drive) */
  auto = false;
  /** sanity checks on static cut-outs (on when the layers are borrowed by a splat scene) */
  sanity = false;
  /** cards the sanity checks dropped, with the reason (for the layer engineer) */
  readonly dropped = new Map<number, string>();
  private readonly geomReason = new Map<number, string>();
  // per-frame shared uniforms of every card
  private readonly camP = new THREE.Vector3();
  private readonly camF = new THREE.Vector3();
  /** photo -> scene */
  readonly toScene = new THREE.Matrix4();
  private yaw = 0;
  /** the selected knots (pano index -> weight), the winning camera and stats, for the HUD and checks */
  info = { knots: {} as Record<number, number>, lens: "", angle: 0, cards: 0, movers: 0, amount: 0 };
  /** sky textures and weights for the dome (sky.ts), already substituted for missing knots */
  sky = { tex: [null, null, null, null] as (THREE.Texture | null)[], w: new THREE.Vector4(), horizon: new THREE.Color(), yaw: 0, ready: false };

  constructor(readonly base: string, readonly json: PhotoJson, drive: Drive, private readonly G: SplatGlobals) {
    this.drive = drive;
    this.interval = json.interval;
    this.elevSky = json.skyElevation;
    this.tex = new TexCache(base);
    this.tex.skyElev = json.skyElevation;
    this.lenses = Object.entries(json.bases).map(([k, v]) => lens(k, v));
    this.arc = new Float64Array(drive.n);
    for (let i = 1; i < drive.n; i++) this.arc[i] = this.arc[i - 1] + Math.hypot(drive.x[i] - drive.x[i - 1], drive.y[i] - drive.y[i - 1]);
    this.shards = Object.entries(json.movers).map(([name, list]) => ({ name, list, loaded: new Map() }));
    this.checkGeometry();
    this.group.matrixAutoUpdate = false;
    this.group.name = "photo-layers";
    this.bu = {
      far0: { value: null }, far1: { value: null }, far2: { value: null }, far3: { value: null },
      w: { value: new THREE.Vector4() },
      org: { value: [0, 1, 2, 3].map(() => new THREE.Vector4()) },
      camP: { value: new THREE.Vector3() },
      camR: { value: new THREE.Matrix3() },
      tanXY: { value: new THREE.Vector2(1, 1) },
      nf: { value: new THREE.Vector2(0.3, 5000) },
      elev: { value: new THREE.Vector2(json.backdropElevation[0], json.backdropElevation[1]) },
      groundZ: { value: 0 },
      amount: { value: 0 },
      GR: { value: G.grade.value },
      G2: { value: G.grade2.value },
    };
    this.backdrop = new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.ShaderMaterial({
        vertexShader: backdropVert,
        fragmentShader: backdropFrag,
        uniforms: this.bu,
        transparent: true,
        depthTest: true,
        depthWrite: true,
        blending: THREE.CustomBlending,
        blendSrc: THREE.OneFactor,
        blendDst: THREE.OneMinusSrcAlphaFactor,
        blendSrcAlpha: THREE.OneFactor,
        blendDstAlpha: THREE.OneMinusSrcAlphaFactor,
      }),
    );
    this.backdrop.frustumCulled = false;
    // after the sky and the apron, before the ribbon and the splats (which are depth-tested against it)
    this.backdrop.renderOrder = -90;
    this.backdrop.name = "photo-backdrop";
  }

  static async load(url: string, sceneDrive: Drive | null, sceneDriveUrl: string | null, G: SplatGlobals): Promise<PhotoLayers | null> {
    try {
      const r = await fetch(url);
      if (!r.ok) return null;
      const j = (await r.json()) as PhotoJson;
      if (j.format !== "photo-layers" || j.version !== 1) {
        console.warn("photo layers: unsupported format", j.format, j.version);
        return null;
      }
      const base = new URL(".", new URL(url, location.href)).href;
      const dUrl = new URL(j.drive, base).href;
      let drive = sceneDrive;
      if (!drive || !sceneDriveUrl || new URL(sceneDriveUrl, location.href).href !== dUrl) {
        const { loadDrive } = await import("../scene/drive");
        drive = await loadDrive(dUrl);
      }
      return new PhotoLayers(base, j, drive, G);
    } catch (e) {
      console.warn("photo layers failed to load", e);
      return null;
    }
  }

  get pending() {
    return this.tex.pending;
  }

  /** weight of the layers at T: 0 outside the published interval, easing in and out over 0.4 s at its ends */
  amountAt(T: number) {
    const [a, b] = this.interval;
    return h((T - a) / 0.4) * h((b - T) / 0.4);
  }

  /** the per-time transform from the photo frame to the scene: the photo drive's car pose onto the scene's,
   *  the difference smoothed over a second (identity when both use the same drive) */
  private frame(T: number, scene: Drive) {
    const P = this.drive;
    if (scene === P) {
      this.toScene.identity();
      this.yaw = 0;
      return;
    }
    let dx = 0, dy = 0, dz = 0, dyaw = 0, ws = 0;
    for (let k = -12; k <= 12; k++) {
      const t = Math.min(scene.duration, Math.max(0, T + k * 0.1));
      const w = Math.exp(-0.5 * ((k * 0.1) / 0.5) ** 2);
      const a = scene.state(t), b = P.state(t);
      dx += w * (a.x - b.x);
      dy += w * (a.y - b.y);
      dz += w * (a.z - b.z);
      dyaw += w * (a.yaw - b.yaw);
      ws += w;
    }
    const c = P.state(T);
    const yaw = dyaw / ws;
    this.yaw = yaw;
    // scene = R(yaw) (p - c) + c + d
    const m = new THREE.Matrix4().makeTranslation(c.x + dx / ws, c.y + dy / ws, c.z + dz / ws);
    m.multiply(new THREE.Matrix4().makeRotationZ(yaw));
    m.multiply(new THREE.Matrix4().makeTranslation(-c.x, -c.y, -c.z));
    this.toScene.copy(m);
  }

  /** per frame. amounts: layer weights 0..1 (interval, switches, night, view); cardFade: the orbit fade of cut-outs */
  update(renderer: THREE.WebGLRenderer, T: number, rate: number, cam: THREE.PerspectiveCamera, scene: Drive, on: { sky: number; backdrop: number; cutouts: number; movers: number }) {
    this.tex.pump(renderer);
    this.frame(T, scene);
    this.group.matrix.copy(this.toScene);
    this.group.matrixWorldNeedsUpdate = true;
    const inv = this.toScene.clone().invert();
    cam.updateMatrixWorld();
    this.camP.copy(cam.position).applyMatrix4(inv);
    const rot = new THREE.Matrix4().extractRotation(cam.matrixWorld).premultiply(new THREE.Matrix4().makeRotationZ(-this.yaw));
    this.camF.set(0, 0, -1).applyMatrix4(rot).normalize();
    const any = Math.max(on.sky, on.backdrop, on.cutouts, on.movers);
    this.info.amount = any;
    this.group.visible = on.cutouts > 0 || on.movers > 0;
    this.backdrop.visible = on.backdrop > 0;
    if (any <= 0) {
      this.sky.ready = false;
      for (const c of this.cards.values()) c.pre.visible = c.col.visible = false;
      for (const c of this.pool) c.pre.visible = c.col.visible = false;
      return;
    }
    // ---- source selection
    const S = interp(this.drive.t, this.arc, T);
    const sel = select(this.lenses, T, S, this.camP, this.camF);
    this.info.knots = Object.fromEntries([...sel.knots].map(([k, v]) => [k, +v.toFixed(3)]));
    this.info.lens = sel.lens;
    this.info.angle = +sel.angle.toFixed(1);
    const knots = [...sel.knots].sort((a, b) => b[1] - a[1]).slice(0, K);
    const pans = this.json.panoramas;
    // warm the neighbours of the selected knots (a hand-over is a knot or two away)
    const warm = new Set<number>();
    for (const [k] of knots) for (const d of [-2, -1, 1, 2, 3]) if (k + d >= 0 && k + d < pans.length) warm.add(k + d);
    for (const k of warm) {
      if (on.sky > 0) this.tex.want("s" + k, pans[k].sky, null, "sky", 2);
      if (on.backdrop > 0) this.tex.want("f" + k, pans[k].rgb, pans[k].matte, "far", 2 + Math.abs(k - knots[0][0]));
    }
    // a knot whose images are not in yet borrows the nearest one that is (never a hole in the sky)
    const pick = (k: number, kind: "s" | "f") => {
      for (const d of [0, -1, 1, -2, 2, -3, 3, -4, 4]) {
        const i = k + d;
        if (i < 0 || i >= pans.length) continue;
        const t = d === 0
          ? this.tex.want(kind + i, kind === "s" ? pans[i].sky : pans[i].rgb, kind === "s" ? null : pans[i].matte, kind === "s" ? "sky" : "far", 0)
          : this.tex.get(kind + i);
        if (t) return { i, t };
      }
      return null;
    };
    // sky
    const sw = new THREE.Vector4();
    this.sky.horizon.setRGB(0, 0, 0);
    let swSum = 0;
    const st: (THREE.Texture | null)[] = [null, null, null, null];
    if (on.sky > 0) {
      knots.forEach(([k, w], j) => {
        const p = pick(k, "s");
        if (!p) return;
        st[j] = p.t.tex!;
        sw.setComponent(j, w);
        swSum += w;
        this.sky.horizon.r += (p.t.horizon?.r ?? 0.8) * w;
        this.sky.horizon.g += (p.t.horizon?.g ?? 0.82) * w;
        this.sky.horizon.b += (p.t.horizon?.b ?? 0.84) * w;
      });
    }
    if (swSum > 0) {
      sw.multiplyScalar(1 / swSum);
      this.sky.horizon.multiplyScalar(1 / swSum);
    }
    this.sky.tex = st;
    this.sky.w.copy(sw);
    this.sky.yaw = this.yaw;
    this.sky.ready = swSum > 0;
    // backdrop
    const bu = this.bu;
    const bw = bu.w.value as THREE.Vector4;
    bw.set(0, 0, 0, 0);
    if (on.backdrop > 0) {
      knots.forEach(([k, w], j) => {
        const p = pick(k, "f");
        if (!p) return;
        bu["far" + j].value = p.t.tex;
        bw.setComponent(j, w);
        const o = pans[p.i].origin;
        (bu.org.value as THREE.Vector4[])[j].set(o[0], o[1], o[2], pans[p.i].radius);
      });
    }
    bu.amount.value = on.backdrop;
    (bu.camP.value as THREE.Vector3).copy(this.camP);
    (bu.camR.value as THREE.Matrix3).setFromMatrix4(rot);
    const ty = Math.tan((cam.fov * D2R) / 2);
    (bu.tanXY.value as THREE.Vector2).set(ty * cam.aspect, ty);
    (bu.nf.value as THREE.Vector2).set(cam.near, cam.far);
    bu.groundZ.value = this.drive.state(T).z - 0.08;
    // ---- cut-outs and movers
    this.updateCards(T, on.cutouts);
    this.updateMovers(T, rate, on.movers);
  }

  private cardMesh(c: Card, t: Tex, existing?: CardMesh): CardMesh {
    const [nx, ny, d] = c.plane;
    const [u0, z0, u1, z1] = c.extent;
    const bx = -ny, by = nx;
    const cx = c.center[0], cy = c.center[1], cz = c.center[2];
    const P = (u: number, z: number) => [nx * d + bx * u - cx, ny * d + by * u - cy, z - cz];
    const [L, Tp, W, H] = c.rect;
    const ul = (L + 0.5) / t.w, ur = (L + W - 0.5) / t.w, vt = (Tp + 0.5) / t.h, vb = (Tp + H - 0.5) / t.h;
    const pos = [...P(u0, z1), ...P(u1, z1), ...P(u0, z0), ...P(u1, z0)];
    const uv = [ul, vt, ur, vt, ul, vb, ur, vb];
    let m = existing;
    if (!m) {
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
      g.setAttribute("uv", new THREE.Float32BufferAttribute(uv, 2));
      g.setIndex([0, 2, 1, 2, 3, 1]);
      const u: Record<string, THREE.IUniform> = {
        map: { value: t.tex },
        clampUv: { value: new THREE.Vector4(ul, vt, ur, vb) },
        nrm: { value: new THREE.Vector2(nx, ny) },
        alpha: { value: 1 },
        prepass: { value: 0 },
        centre: { value: new THREE.Vector3(cx, cy, cz) },
        camP: { value: this.camP },
        camF: { value: this.camF },
        GR: { value: this.G.grade.value },
        G2: { value: this.G.grade2.value },
      };
      const col = new THREE.Mesh(g, new THREE.ShaderMaterial({
        vertexShader: cardVert, fragmentShader: cardFrag, uniforms: u, side: THREE.DoubleSide, transparent: true, depthWrite: false,
        blending: THREE.CustomBlending, blendSrc: THREE.OneFactor, blendDst: THREE.OneMinusSrcAlphaFactor, blendSrcAlpha: THREE.OneFactor, blendDstAlpha: THREE.OneMinusSrcAlphaFactor,
      }));
      const pu = { ...u, prepass: { value: 1 } };
      const pre = new THREE.Mesh(g, new THREE.ShaderMaterial({ vertexShader: cardVert, fragmentShader: cardFrag, uniforms: pu, side: THREE.DoubleSide, transparent: true, colorWrite: false, depthWrite: true }));
      pre.renderOrder = -47;
      col.renderOrder = -46;
      m = { pre, col, u };
      this.group.add(pre, col);
    } else {
      const g = m.col.geometry;
      (g.getAttribute("position") as THREE.BufferAttribute).copyArray(pos).needsUpdate = true;
      (g.getAttribute("uv") as THREE.BufferAttribute).copyArray(uv).needsUpdate = true;
      g.computeBoundingSphere();
      m.u.map.value = t.tex;
      (m.u.clampUv.value as THREE.Vector4).set(ul, vt, ur, vb);
      (m.u.nrm.value as THREE.Vector2).set(nx, ny);
      (m.u.centre.value as THREE.Vector3).set(cx, cy, cz);
      const pu = (m.pre.material as THREE.ShaderMaterial).uniforms;
      pu.map.value = t.tex;
    }
    m.pre.position.set(cx, cy, cz);
    m.col.position.set(cx, cy, cz);
    m.pre.updateMatrix();
    m.col.updateMatrix();
    return m;
  }

  /** geometry checks, once: a base on the driven corridor, a card too wide for its kind */
  private checkGeometry() {
    const D = this.drive;
    const [t0, t1] = this.interval;
    const xs: number[] = [], ys: number[] = [];
    for (let i = 0; i < D.n; i += 2) if (D.t[i] > t0 - 30 && D.t[i] < t1 + 30) { xs.push(D.x[i]); ys.push(D.y[i]); }
    const dist = (x: number, y: number) => {
      let best = 1e9;
      for (let i = 0; i + 1 < xs.length; i++) {
        const ex = xs[i], ey = ys[i], dx = xs[i + 1] - ex, dy = ys[i + 1] - ey;
        const l2 = dx * dx + dy * dy;
        const f = l2 > 0 ? Math.min(1, Math.max(0, ((x - ex) * dx + (y - ey) * dy) / l2)) : 0;
        best = Math.min(best, Math.hypot(ex + f * dx - x, ey + f * dy - y));
      }
      return best;
    };
    for (const c of this.json.cutouts) {
      const [nx, ny, d] = c.plane;
      const [u0, z0, u1, z1] = c.extent;
      const w = u1 - u0, hgt = z1 - z0;
      // on the drivable corridor: base within 3.6 m of the driven path (the car's lane and half the next), unless
      // it is overhead
      if (z0 < 4.2) {
        let md = 1e9;
        for (let k = 0; k <= 8; k++) {
          const u = u0 + (w * k) / 8;
          md = Math.min(md, dist(nx * d - ny * u, ny * d + nx * u));
        }
        if (md < 3.6) { this.geomReason.set(c.id, `base ${md.toFixed(1)} m from the driven path`); continue; }
      }
      const wide =
        (c.kind === 5 && w > 1.5 && w > 0.25 * hgt) || (c.kind === 6 && w > 2 && w > 0.5 * hgt) || (c.kind === 7 && w > 3.5) || (c.kind === 4 && w > 25);
      if (wide) this.geomReason.set(c.id, `${w.toFixed(1)} m wide, ${hgt.toFixed(1)} m tall for kind ${c.kind}`);
    }
  }

  /** null if the card passes, else the reason; matte checks need its atlas */
  private judge(c: Card, t: Tex): string | null {
    if (!this.sanity) return null;
    const g = this.geomReason.get(c.id);
    if (g) return g;
    if (t.alpha) {
      const s = soft(coverage(t.alpha, t.w, c.rect));
      if (s) return s;
    }
    return null;
  }

  /** run every static cut-out of these kinds through the checks (fetches their mattes); for the report */
  async audit(kinds: number[] | null) {
    const out: { id: number; kind: number; reason: string; birth?: number; death?: number }[] = [];
    const cache = new Map<string, { a: Uint8Array; w: number }>();
    for (const c of this.json.cutouts) {
      if (kinds && !kinds.includes(c.kind)) continue;
      let reason = this.geomReason.get(c.id) ?? null;
      if (!reason) {
        let m = cache.get(c.matte);
        if (!m) {
          const r = await fetch(new URL(c.matte, this.base).href);
          const b = await createImageBitmap(await r.blob(), { colorSpaceConversion: "none", premultiplyAlpha: "none" });
          m = { a: matteBytes(b), w: b.width };
          b.close();
          cache.set(c.matte, m);
        }
        reason = soft(coverage(m.a, m.w, c.rect));
      }
      if (reason) out.push({ id: c.id, kind: c.kind, reason, birth: c.birth && +c.birth.toFixed(1), death: c.death && +c.death.toFixed(1) });
    }
    return out;
  }

  private setAlpha(m: CardMesh, a: number) {
    m.u.alpha.value = a;
    (m.pre.material as THREE.ShaderMaterial).uniforms.alpha.value = a;
    m.pre.visible = m.col.visible = a > 0.002;
  }

  private visibleCard(c: Card) {
    const dx = c.center[0] - this.camP.x, dy = c.center[1] - this.camP.y, dz = c.center[2] - this.camP.z;
    if (Math.hypot(dx, dy, dz) > 115) return false;
    return dx * this.camF.x + dy * this.camF.y + dz * this.camF.z >= -(c.extent[2] - c.extent[0]);
  }

  private updateCards(T: number, amount: number) {
    let n = 0;
    const seen = new Set<number>();
    if (amount > 0) {
      for (const c of this.json.cutouts) {
        if (this.kinds && !this.kinds.has(c.kind)) continue;
        if (this.ids && !this.ids.has(c.id)) continue;
        const b = c.birth ?? -1e9, d = c.death ?? 1e9, f = c.fade ?? 0.45;
        // fetch the atlas a little before birth so the fade-in is never late
        if (T < b - 1.5 || T > d) continue;
        if (!this.visibleCard(c)) continue;
        const t = this.tex.want("c" + c.rgb, c.rgb, c.matte, "card", 1 + Math.max(0, b - T));
        const life = h((T - b) / f) * h((d - T) / f);
        if (!t || life <= 0) continue;
        const why = this.judge(c, t);
        if (why) {
          if (!this.dropped.has(c.id)) {
            this.dropped.set(c.id, why);
            console.info(`photo layers: dropped cut-out ${c.id} (kind ${c.kind}): ${why}`);
          }
          continue;
        }
        let m = this.cards.get(c.id);
        if (!m) {
          m = this.cardMesh(c, t);
          this.cards.set(c.id, m);
        } else if (m.u.map.value !== t.tex) this.cardMesh(c, t, m);
        this.setAlpha(m, life * amount);
        seen.add(c.id);
        n++;
      }
    }
    for (const [id, m] of this.cards) {
      if (seen.has(id)) continue;
      m.pre.visible = m.col.visible = false;
      // free meshes of cards that are long gone
      if (this.cards.size > 160) {
        this.group.remove(m.pre, m.col);
        m.col.geometry.dispose();
        (m.col.material as THREE.Material).dispose();
        (m.pre.material as THREE.Material).dispose();
        this.cards.delete(id);
      }
    }
    this.info.cards = n;
  }

  private exposures(T: number, rate: number) {
    const out: { lens: string; list: Exposure[] }[] = [];
    for (const L of this.shards) {
      let i = L.list.findIndex((s) => T <= s.last + 0.3);
      if (i < 0) i = L.list.length - 1;
      const ahead = rate > 1.5 ? 3 : 2;
      for (let k = Math.max(0, i - 1); k <= Math.min(L.list.length - 1, i + ahead); k++) {
        if (L.loaded.has(k)) continue;
        L.loaded.set(k, "loading");
        fetch(new URL(L.list[k].file, this.base).href)
          .then((r) => r.json())
          .then((j: Exposure[]) => L.loaded.set(k, j))
          .catch(() => L.loaded.delete(k));
      }
      for (const k of [...L.loaded.keys()]) if (k < i - 3 || k > i + 6) L.loaded.delete(k);
      const list: Exposure[] = [];
      for (const k of [...L.loaded.keys()].sort((a, b) => a - b)) {
        const v = L.loaded.get(k);
        if (v && v !== "loading") list.push(...v);
      }
      out.push({ lens: L.name, list });
    }
    return out;
  }

  private updateMovers(T: number, rate: number, amount: number) {
    let used = 0;
    if (amount > 0) {
      const cands: Card[] = [];
      for (const { list } of this.exposures(T, rate)) {
        if (!list.length) continue;
        // nearest exposure (the earlier on a tie), dropped if more than 0.24 s away
        let lo = 0, hi = list.length - 1;
        while (hi - lo > 1) {
          const m = (lo + hi) >> 1;
          if (list[m].t <= T) lo = m;
          else hi = m;
        }
        const e = Math.abs(list[lo].t - T) <= Math.abs(list[hi].t - T) ? list[lo] : list[hi];
        if (Math.abs(e.t - T) <= 0.24) cands.push(...e.layers);
        // fetch the atlases of the next few exposures
        for (const x of list) if (x.t > T && x.t < T + 0.35 + 0.3 * rate) for (const c of x.layers) this.tex.want("m" + c.rgb, c.rgb, c.matte, "mover", 3 + (x.t - T));
      }
      // coalesce overlapping cameras: larger images first; another camera's candidate within 2.8 m / 1.6 m goes
      cands.sort((a, b) => b.rect[2] * b.rect[3] - a.rect[2] * a.rect[3]);
      const kept: Card[] = [];
      for (const c of cands) {
        if (kept.some((k) => k.source[0] !== c.source[0] && Math.hypot(k.center[0] - c.center[0], k.center[1] - c.center[1]) < 2.8 && Math.abs(k.center[2] - c.center[2]) < 1.6)) continue;
        kept.push(c);
      }
      for (const c0 of kept) {
        let c = c0;
        let t = this.tex.want("m" + c.rgb, c.rgb, c.matte, "mover", 0);
        if (!t) {
          // its image is not in yet: hold the same mover's previous exposure for up to 0.3 s
          const prev = this.lastMover.get(c.id);
          if (prev && T - prev.t < 0.3) {
            c = prev.card;
            t = this.tex.get("m" + c.rgb);
          }
        }
        if (!t) continue;
        if (c === c0) this.lastMover.set(c.id, { t: T, card: c });
        if (!this.visibleCard(c)) continue;
        let m = this.pool[used];
        m = this.cardMesh(c, t, m);
        if (!this.pool[used]) {
          m.pre.renderOrder = -47;
          m.col.renderOrder = -45;
          this.pool.push(m);
        }
        this.setAlpha(m, amount);
        used++;
      }
    }
    for (let i = used; i < this.pool.length; i++) this.pool[i].pre.visible = this.pool[i].col.visible = false;
    this.info.movers = used;
  }

  /** the photographed sky's dominant knot (for rebuilding the car's environment when it changes) */
  get skyKey() {
    const k = Object.entries(this.info.knots).sort((a, b) => b[1] - a[1])[0];
    return this.sky.ready && k ? Math.round(Number(k[0]) / 12) : -1;
  }

  dispose() {
    this.tex.dispose();
    for (const m of [...this.cards.values(), ...this.pool]) {
      m.col.geometry.dispose();
      (m.col.material as THREE.Material).dispose();
      (m.pre.material as THREE.Material).dispose();
    }
    this.cards.clear();
    this.group.clear();
    (this.backdrop.material as THREE.Material).dispose();
    this.backdrop.geometry.dispose();
  }
}

function interp(xs: Float64Array, ys: Float64Array, x: number) {
  const n = xs.length;
  if (x <= xs[0]) return ys[0];
  if (x >= xs[n - 1]) return ys[n - 1];
  let lo = 0, hi = n - 1;
  while (hi - lo > 1) {
    const m = (lo + hi) >> 1;
    if (xs[m] <= x) lo = m;
    else hi = m;
  }
  const a = (x - xs[lo]) / (xs[hi] - xs[lo]);
  return ys[lo] + (ys[hi] - ys[lo]) * a;
}
