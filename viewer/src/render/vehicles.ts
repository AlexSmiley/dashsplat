// Clean vehicles: every tracked and parked vehicle of objects.json drawn as a smooth 3D model with the ego car's
// look (scene environment light, clear-coated paint, dark glass, turning wheels, brake and running lights, the
// overcast contact shadow), instead of its scanned splat model. One model per kind: saloons and coupes use the
// Highland Model 3; SUVs, off-roaders, minivans, pickups, buses and lorries are procedural bodies (a lofted
// rounded section, glasshouse, lamps and wheel arches as material regions) on the Model 3's wheels. Size comes
// from the track's measured box, clamped to what the kind can be (scanned boxes are often cut short); paint from
// the track's name (the producer names the colour) or, failing that, the scan's body colour, else mid grey.

import * as THREE from "three";
import { AO_HALF_X, AO_HALF_Y, contactShadowTexture, EGO } from "./contactShadow";

export type VehicleKind = "saloon" | "coupe" | "suv" | "offroad" | "van" | "pickup" | "bus" | "lorry";

/** plausible length, width, height per kind (m) */
const DIMS: Record<VehicleKind, [number, number][]> = {
  saloon: [[4.4, 5.0], [1.8, 1.95], [1.38, 1.5]],
  coupe: [[4.4, 4.9], [1.8, 1.95], [1.3, 1.42]],
  suv: [[4.4, 4.95], [1.82, 1.98], [1.6, 1.8]],
  offroad: [[4.6, 5.0], [1.86, 2.0], [1.8, 1.96]],
  van: [[4.9, 5.25], [1.9, 2.0], [1.72, 1.82]],
  pickup: [[5.3, 5.9], [1.92, 2.05], [1.82, 1.95]],
  bus: [[10, 13.5], [2.45, 2.6], [3.0, 3.4]],
  lorry: [[7, 12], [2.4, 2.55], [3.0, 3.9]],
};

export function vehicleKind(name: string, cls: string, size?: number[]): VehicleKind {
  const n = name.toLowerCase();
  const c = cls.toLowerCase();
  if (c === "bus" || /\bbus\b/.test(n)) return "bus";
  if (/pickup/.test(n)) return "pickup";
  if (c === "truck" || /lorry|semi|\btruck\b/.test(n)) return "lorry";
  if (/minivan|\bvan\b/.test(n)) return "van";
  if (/off-?roader|jeep/.test(n)) return "offroad";
  if (/suv|crossover/.test(n)) return "suv";
  if (/coupe/.test(n)) return "coupe";
  if (/sedan|saloon|tesla|hatch/.test(n)) return "saloon";
  const h = size?.[2] ?? 1.45;
  return h >= 1.85 ? "offroad" : h >= 1.6 ? "suv" : "saloon";
}

export function vehicleDims(kind: VehicleKind, size?: number[]): [number, number, number] {
  const d = DIMS[kind];
  const mid = (i: number) => (d[i][0] + d[i][1]) / 2;
  return [0, 1, 2].map((i) => THREE.MathUtils.clamp(size?.[i] ?? mid(i), d[i][0], d[i][1])) as [number, number, number];
}

export type Paint = { color: THREE.Color; metal: number; rough: number };
// (coloured paints keep most of their diffuse colour: seen side-on a car mostly reflects the dark roadside, so
// a strongly metallic red would read black)
const NAMED: [RegExp, string, number, number][] = [
  [/burgundy|maroon|wine/, "#5c1520", 0.3, 0.32],
  [/\bred\b/, "#9a161f", 0.25, 0.3],
  [/dark blue|navy/, "#1b2848", 0.3, 0.32],
  [/\bblue\b/, "#2a4c86", 0.3, 0.3],
  [/white|pearl/, "#e3e4e1", 0.05, 0.36],
  [/black/, "#111214", 0.35, 0.28],
  [/silver/, "#bcc0c4", 0.75, 0.28],
  [/grey|gray/, "#666b70", 0.45, 0.32],
  [/\bdark\b/, "#2a2d31", 0.35, 0.32],
  [/green/, "#2f4b36", 0.3, 0.33],
  [/beige|tan\b|gold|champagne/, "#b3a486", 0.35, 0.33],
  [/brown/, "#4c3929", 0.3, 0.34],
  [/yellow/, "#caa42b", 0.15, 0.35],
  [/orange/, "#c4591b", 0.2, 0.34],
];
/** the colour the producer named, if any */
export function namedPaint(name: string): Paint | null {
  const n = name.toLowerCase();
  for (const [re, hex, metal, rough] of NAMED) if (re.test(n)) return { color: new THREE.Color(hex), metal, rough };
  return null;
}
export const GREY_PAINT: Paint = { color: new THREE.Color("#7c8186"), metal: 0.45, rough: 0.32 };

// ---------------------------------------------------------------------------------------------------------
// materials: per instance (a fade uniform and lamp levels each), sharing programs

type VUniforms = { ground: { value: number }; fade: { value: number }; tail: { value: number }; head: { value: number } };

/** over a generated clip (which has no depth): the share of each pixel covered by splats in front of the vehicles
 *  (render/vehicles + main.ts renderGenerative), read softly; x of info = on, yz = 1 / texture size */
export const VEHICLE_OCCLUSION = { tex: { value: null as THREE.Texture | null }, info: { value: new THREE.Vector4(0, 0, 0, 0) } };
const OCC_GLSL = /* glsl */ `
uniform sampler2D vOccTex;
uniform vec4 vOccInfo;
float vOcclusion() {
  if (vOccInfo.x < 0.5) return 0.0;
  vec2 uv = gl_FragCoord.xy * vOccInfo.zw;
  vec2 px = vec2(vOccInfo.y, vOccInfo.y);
  float o = texture2D(vOccTex, uv).a * 0.4;
  o += texture2D(vOccTex, uv + vec2(1.5, 1.5) * px).a * 0.15;
  o += texture2D(vOccTex, uv + vec2(-1.5, 1.5) * px).a * 0.15;
  o += texture2D(vOccTex, uv + vec2(1.5, -1.5) * px).a * 0.15;
  o += texture2D(vOccTex, uv + vec2(-1.5, -1.5) * px).a * 0.15;
  return smoothstep(0.08, 0.85, o);
}`;

/** ground occlusion (as on the ego car), a dithered fade, and for procedural bodies per-vertex regions (aMat:
 *  glass, trim, red lamp, white lamp) that set colour, metalness, roughness, clear coat and lamp light */
function patch(m: THREE.Material, U: VUniforms, regions: boolean) {
  m.onBeforeCompile = (sh) => {
    sh.uniforms.vGround = U.ground;
    sh.uniforms.vFade = U.fade;
    sh.uniforms.vTail = U.tail;
    sh.uniforms.vHead = U.head;
    sh.uniforms.vOccTex = VEHICLE_OCCLUSION.tex;
    sh.uniforms.vOccInfo = VEHICLE_OCCLUSION.info;
    sh.vertexShader = sh.vertexShader
      .replace("#include <common>", "#include <common>\nuniform float vGround;\nvarying float vCarH;" + (regions ? "\nattribute vec4 aMat;\nvarying vec4 vMat;" : ""))
      .replace("#include <project_vertex>", "#include <project_vertex>\n  vCarH = (modelMatrix * vec4(transformed, 1.0)).z - vGround;" + (regions ? "\n  vMat = aMat;" : ""));
    let f = sh.fragmentShader
      .replace("#include <common>", "#include <common>\nuniform float vFade;\nuniform float vTail;\nuniform float vHead;\nvarying float vCarH;" + OCC_GLSL + (regions ? "\nvarying vec4 vMat;" : ""))
      .replace("#include <dithering_fragment>", "#include <dithering_fragment>\n  gl_FragColor.a *= 1.0 - vOcclusion();")
      .replace(
        "#include <clipping_planes_fragment>",
        "#include <clipping_planes_fragment>\n  if (vFade < 0.999 && fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715)))) > vFade) discard;",
      )
      .replace(
        "#include <aomap_fragment>",
        `{
  vec3 wN = normalize((vec4(normal, 0.0) * viewMatrix).xyz);
  float hOcc = mix(0.3, 1.0, smoothstep(0.03, 0.8, vCarH));
  float down = smoothstep(0.25, -0.75, wN.z) * (1.0 - smoothstep(0.25, 0.95, vCarH));
  float ambientOcclusion = hOcc * (1.0 - 0.6 * down);
  reflectedLight.indirectDiffuse *= ambientOcclusion;
  #if defined( USE_CLEARCOAT )
    clearcoatSpecularIndirect *= ambientOcclusion;
  #endif
  #if defined( USE_ENVMAP ) && defined( STANDARD )
    float dotNV = saturate( dot( geometryNormal, geometryViewDir ) );
    reflectedLight.indirectSpecular *= computeSpecularOcclusion( dotNV, ambientOcclusion, material.roughness );
  #endif
}`,
      );
    if (regions) {
      f = f
        .replace(
          "#include <color_fragment>",
          `#include <color_fragment>
  {
    vec3 c = diffuseColor.rgb;
    c = mix(c, vec3(0.012, 0.014, 0.017), vMat.x);
    c = mix(c, vec3(0.022, 0.023, 0.025), vMat.y);
    c = mix(c, vec3(0.22, 0.015, 0.012), vMat.z);
    c = mix(c, vec3(0.75, 0.76, 0.78), vMat.w);
    diffuseColor.rgb = c;
  }`,
        )
        .replace("#include <metalnessmap_fragment>", "#include <metalnessmap_fragment>\n  metalnessFactor *= 1.0 - clamp(vMat.x + vMat.y + vMat.z + vMat.w, 0.0, 1.0);")
        .replace("#include <roughnessmap_fragment>", "#include <roughnessmap_fragment>\n  roughnessFactor = mix(roughnessFactor, 0.04, vMat.x); roughnessFactor = mix(roughnessFactor, 0.5, vMat.y); roughnessFactor = mix(roughnessFactor, 0.12, vMat.z + vMat.w);")
        .replace("#include <lights_physical_fragment>", "#include <lights_physical_fragment>\n  material.clearcoat *= 1.0 - vMat.y;")
        .replace("#include <emissivemap_fragment>", "#include <emissivemap_fragment>\n  totalEmissiveRadiance += vec3(1.0, 0.07, 0.03) * vMat.z * vTail + vec3(1.0, 0.96, 0.9) * vMat.w * vHead;");
    }
    sh.fragmentShader = f;
  };
  m.customProgramCacheKey = () => (regions ? "veh-regions" : "veh");
  m.needsUpdate = true;
}

// ---------------------------------------------------------------------------------------------------------
// procedural bodies

type Profile = {
  /** top line: [u, height fraction] from the rear (u -0.5) to the front (u 0.5) */
  top: [number, number][];
  belt: [number, number][];
  /** u range of the glasshouse, and of the painted roof */
  glass: [number, number];
  roof: [number, number];
  clear: number;
  /** wheel radius, wheelbase as a fraction of the length, front axle shift (fraction) */
  r: number;
  wb: number;
  shift: number;
  /** section squareness, plan squareness, roof width */
  p: number;
  q: number;
  tumble: number;
  /** lamp bands as height fractions: rear [z0, z1], front [z0, z1] */
  tail: [number, number];
  headL: [number, number];
  /** front face glass (buses, lorry cabs): height fractions */
  frontGlass?: [number, number];
};

const PROFILES: Record<Exclude<VehicleKind, "saloon" | "coupe">, Profile> = {
  suv: { top: [[-0.5, 0.6], [-0.485, 0.84], [-0.44, 0.96], [-0.36, 1.0], [0.08, 1.0], [0.16, 0.96], [0.27, 0.71], [0.42, 0.67], [0.5, 0.58]], belt: [[-0.5, 0.64], [0.3, 0.6]], glass: [-0.455, 0.28], roof: [-0.38, 0.1], clear: 0.2, r: 0.37, wb: 0.61, shift: 0.02, p: 5, q: 7, tumble: 0.8, tail: [0.63, 0.74], headL: [0.56, 0.64] },
  offroad: { top: [[-0.5, 0.62], [-0.492, 0.92], [-0.47, 1.0], [0.1, 1.0], [0.16, 0.98], [0.25, 0.74], [0.45, 0.7], [0.5, 0.62]], belt: [[-0.5, 0.62], [0.3, 0.6]], glass: [-0.475, 0.255], roof: [-0.46, 0.11], clear: 0.25, r: 0.4, wb: 0.6, shift: 0.0, p: 8, q: 10, tumble: 0.86, tail: [0.6, 0.72], headL: [0.56, 0.64] },
  van: { top: [[-0.5, 0.56], [-0.488, 0.88], [-0.45, 0.99], [-0.38, 1.0], [0.02, 1.0], [0.1, 0.96], [0.32, 0.62], [0.42, 0.57], [0.5, 0.5]], belt: [[-0.5, 0.58], [0.3, 0.55]], glass: [-0.465, 0.33], roof: [-0.42, 0.04], clear: 0.16, r: 0.36, wb: 0.6, shift: 0.04, p: 6, q: 8, tumble: 0.84, tail: [0.6, 0.74], headL: [0.5, 0.58] },
  pickup: { top: [[-0.5, 0.56], [-0.497, 0.6], [-0.1, 0.6], [-0.085, 0.95], [-0.06, 1.0], [0.1, 1.0], [0.16, 0.97], [0.25, 0.73], [0.46, 0.7], [0.5, 0.62]], belt: [[-0.5, 0.6], [0.3, 0.6]], glass: [-0.08, 0.255], roof: [-0.06, 0.12], clear: 0.24, r: 0.39, wb: 0.62, shift: 0.0, p: 7, q: 10, tumble: 0.84, tail: [0.46, 0.58], headL: [0.56, 0.64] },
  bus: { top: [[-0.5, 0.25], [-0.497, 0.97], [-0.49, 1.0], [0.49, 1.0], [0.497, 0.97], [0.5, 0.25]], belt: [[-0.5, 0.42], [0.5, 0.42]], glass: [-0.47, 0.5], roof: [-0.49, 0.49], clear: 0.32, r: 0.5, wb: 0.56, shift: -0.04, p: 12, q: 26, tumble: 0.95, tail: [0.18, 0.3], headL: [0.14, 0.22], frontGlass: [0.32, 0.93] },
  lorry: { top: [[-0.5, 0.25], [-0.497, 0.98], [-0.49, 1.0], [0.36, 1.0], [0.38, 0.88], [0.49, 0.86], [0.5, 0.25]], belt: [[-0.5, 0.95], [0.37, 0.95], [0.38, 0.5], [0.5, 0.5]], glass: [0.38, 0.5], roof: [-0.49, 0.37], clear: 0.35, r: 0.52, wb: 0.6, shift: 0.05, p: 14, q: 28, tumble: 0.97, tail: [0.12, 0.2], headL: [0.14, 0.22], frontGlass: [0.52, 0.84] },
};

function lerpProfile(pts: [number, number][], u: number) {
  if (u <= pts[0][0]) return pts[0][1];
  for (let i = 1; i < pts.length; i++) {
    if (u <= pts[i][0]) {
      const [u0, h0] = pts[i - 1], [u1, h1] = pts[i];
      const t = (u - u0) / Math.max(1e-6, u1 - u0);
      // eased between the points: a smooth line, no creases
      const s = t * t * (3 - 2 * t);
      return h0 + (h1 - h0) * (0.35 * t + 0.65 * s);
    }
  }
  return pts[pts.length - 1][1];
}

/** a lofted, rounded body with region attributes; origin on the ground under the centre, x forward */
function buildBody(pr: Profile, L: number, W: number, H: number) {
  const NU = 96, NV = 56;
  const us: number[] = [];
  // denser towards the ends, where the shape turns
  for (let i = 0; i <= NU; i++) us.push(-0.5 * Math.cos((Math.PI * i) / NU));
  const pos: number[] = [];
  const mat: number[] = [];
  const wbx = pr.wb * L / 2, shift = pr.shift * L;
  const wheels = [shift + wbx, shift - wbx];
  const tyre = 0.24;
  for (const u of us) {
    const au = Math.abs(u) / 0.5;
    const plan = Math.pow(Math.max(0, 1 - Math.pow(au, pr.q)), 1 / pr.q);
    const closeH = Math.pow(Math.max(0, 1 - Math.pow(au, pr.q * 1.6)), 1 / (pr.q * 1.6));
    const x = u * L;
    const zt0 = lerpProfile(pr.top, u) * H;
    const endRise = 0.12 * H * Math.pow(au, 6);
    const zb0 = pr.clear + endRise;
    const zm = (zt0 + zb0) / 2;
    const zt = zm + (zt0 - zm) * closeH, zb = zm + (zb0 - zm) * closeH;
    const hz = Math.max(1e-4, (zt - zb) / 2);
    const belt = lerpProfile(pr.belt, u) * H;
    for (let j = 0; j < NV; j++) {
      const th = (j / NV) * Math.PI * 2;
      const c = Math.cos(th), s = Math.sin(th);
      const ny = Math.sign(c) * Math.pow(Math.abs(c), 2 / pr.p);
      const nz = Math.sign(s) * Math.pow(Math.abs(s), 2 / pr.p);
      const z = zm + nz * hz;
      let hw = (W / 2) * plan * (1 - (1 - pr.tumble) * THREE.MathUtils.smoothstep(z, belt, zt));
      // wheel arches: the side dents in round the wheels (dark), so the tyres show
      let arch = 0;
      for (const wx of wheels) {
        const d = Math.hypot(x - wx, z - pr.r);
        arch = Math.max(arch, 1 - THREE.MathUtils.smoothstep(d, pr.r + 0.05, pr.r + 0.11));
      }
      arch *= THREE.MathUtils.smoothstep(Math.abs(ny), 0.25, 0.6);
      hw -= arch * (tyre + 0.05);
      const y = ny * hw;
      pos.push(x, y, z);
      // regions
      const inGlassU = u > pr.glass[0] && u < pr.glass[1];
      const roofTop = nz > 0.8 && u > pr.roof[0] && u < pr.roof[1];
      let glass = inGlassU && z > belt + 0.015 && !roofTop ? 1 : 0;
      if (pr.frontGlass && u > 0.494 && z > pr.frontGlass[0] * H && z < pr.frontGlass[1] * H && Math.abs(ny) < 0.92) glass = 1;
      const trim = Math.max(arch, z < zb + 0.09 ? 1 : 0, au > 0.985 && z < zb + 0.22 ? 1 : 0);
      const tail = u < -0.49 && z > pr.tail[0] * H && z < pr.tail[1] * H && Math.abs(ny) > 0.45 ? 1 : 0;
      const head = u > 0.49 && z > pr.headL[0] * H && z < pr.headL[1] * H && Math.abs(ny) > 0.5 && !glass ? 1 : 0;
      mat.push(glass * (1 - trim), trim, tail, head);
    }
  }
  const idx: number[] = [];
  for (let i = 0; i < NU; i++) for (let j = 0; j < NV; j++) {
    const a = i * NV + j, b = i * NV + ((j + 1) % NV), c2 = (i + 1) * NV + j, d = (i + 1) * NV + ((j + 1) % NV);
    // outward winding (normals out)
    idx.push(a, b, c2, b, d, c2);
  }
  // caps at both ends (the sections close to a line there; fans keep it watertight)
  for (const [ring, flip] of [[0, true], [NU, false]] as [number, boolean][]) {
    let cx = 0, cy = 0, cz = 0;
    for (let j = 0; j < NV; j++) { cx += pos[(ring * NV + j) * 3]; cy += pos[(ring * NV + j) * 3 + 1]; cz += pos[(ring * NV + j) * 3 + 2]; }
    const ci = pos.length / 3;
    pos.push(cx / NV, cy / NV, cz / NV);
    mat.push(0, 0, 0, 0);
    for (let j = 0; j < NV; j++) {
      const a = ring * NV + j, b = ring * NV + ((j + 1) % NV);
      if (flip) idx.push(ci, b, a); else idx.push(ci, a, b);
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute("aMat", new THREE.Float32BufferAttribute(mat, 4));
  g.setIndex(idx);
  g.computeVertexNormals();
  return { geometry: g, wheels, r: pr.r };
}

// ---------------------------------------------------------------------------------------------------------

export type VehicleSpec = { kind: VehicleKind; dims: [number, number, number]; paint: Paint; parked: boolean };
export type VehicleState = { x: number; y: number; z: number; yaw: number; speed: number; accel: number; dist: number; yawRate: number; fade: number };

const TYRE_R = 0.37;

/** shared parts: the Model 3 (body and wheels) to clone */
export class VehicleKit {
  constructor(readonly model: THREE.Object3D) {}

  make(spec: VehicleSpec): Vehicle {
    return new Vehicle(this, spec);
  }
}

export class Vehicle {
  readonly root = new THREE.Group();
  private readonly U: VUniforms = { ground: { value: 0 }, fade: { value: 0 }, tail: { value: 0 }, head: { value: 0 } };
  private readonly wheels: { node: THREE.Object3D; front: boolean; r: number }[] = [];
  private readonly tailMats: THREE.MeshStandardMaterial[] = [];
  private readonly mats: THREE.Material[] = [];
  private readonly shadow: THREE.Mesh;
  private paint!: THREE.MeshPhysicalMaterial;
  private spin = 0;
  private steer = 0;
  private brake = 0;
  private lastDist = NaN;
  readonly wheelbase: number;

  constructor(kit: VehicleKit, readonly spec: VehicleSpec) {
    const [L, W, H] = spec.dims;
    this.root.name = "vehicle";
    this.root.userData.vehicle = this;
    const paint = (this.paint = new THREE.MeshPhysicalMaterial({ color: spec.paint.color, metalness: spec.paint.metal, roughness: spec.paint.rough, clearcoat: 1, clearcoatRoughness: 0.05 }));
    const glass = new THREE.MeshPhysicalMaterial({ color: 0x040506, metalness: 0, roughness: 0.02, ior: 2.0, clearcoat: 0 });
    if (spec.kind === "saloon" || spec.kind === "coupe") {
      // the Model 3, scaled uniformly by length (round wheels) and centred on the box
      const s = THREE.MathUtils.clamp(L / (2 * EGO.halfL), 0.92, 1.08);
      const inner = kit.model.clone(true);
      inner.scale.setScalar(s);
      inner.position.set(-EGO.centreX * s, 0, spec.kind === "coupe" ? -0.01 : 0);
      inner.traverse((o) => {
        if (o.name.startsWith("wheel_")) this.wheels.push({ node: o, front: o.name.includes("_f"), r: TYRE_R * s });
        const m = o as THREE.Mesh;
        if (!m.isMesh) return;
        m.frustumCulled = true;
        const conv = (mat0: THREE.Material) => {
          const mm = mat0 as THREE.MeshStandardMaterial;
          if (mm.name === "paint") return paint;
          if (mm.name === "glass") return glass;
          const c = mm.clone();
          if (c.name === "Tail_Lamp") this.tailMats.push(c as THREE.MeshStandardMaterial);
          if ((c as THREE.MeshStandardMaterial).transparent) (c as THREE.MeshStandardMaterial).transparent = false;
          return c;
        };
        m.material = Array.isArray(m.material) ? m.material.map(conv) : conv(m.material);
        for (const mm of Array.isArray(m.material) ? m.material : [m.material]) if (!this.mats.includes(mm)) this.mats.push(mm);
      });
      this.root.add(inner);
      this.wheelbase = 2.875 * s;
    } else {
      const pr = PROFILES[spec.kind];
      const b = buildBody(pr, L, W, H);
      const body = new THREE.Mesh(b.geometry, paint);
      body.frustumCulled = true;
      this.root.add(body);
      this.mats.push(paint);
      // the Model 3's wheels, scaled to this vehicle's
      const src = { l: kit.model.getObjectByName("wheel_rl")!, r: kit.model.getObjectByName("wheel_rr")! };
      const k = b.r / TYRE_R;
      for (const [wx, front] of [[b.wheels[0], true], [b.wheels[1], false]] as [number, boolean][]) for (const side of [1, -1]) {
        const w = (side > 0 ? src.l : src.r).clone(true);
        w.position.set(wx, side * (W / 2 - 0.13 * k), b.r);
        w.scale.multiplyScalar(k);
        w.traverse((o) => {
          const m = o as THREE.Mesh;
          if (!m.isMesh) return;
          const conv = (mat0: THREE.Material) => {
            const c = mat0.clone();
            this.mats.push(c);
            return c;
          };
          m.material = Array.isArray(m.material) ? m.material.map(conv) : conv(m.material);
        });
        this.root.add(w);
        this.wheels.push({ node: w, front, r: b.r });
      }
      this.wheelbase = b.wheels[0] - b.wheels[1];
    }
    for (const m of this.mats) patch(m, this.U, m === paint && spec.kind !== "saloon" && spec.kind !== "coupe");
    // the contact shadow, the ego car's stretched to this footprint
    this.shadow = new THREE.Mesh(
      new THREE.PlaneGeometry(2 * AO_HALF_X * (L / (2 * EGO.halfL)), 2 * AO_HALF_Y * (W / (2 * EGO.halfW))),
      new THREE.MeshBasicMaterial({ map: contactShadowTexture(), transparent: true, depthWrite: false, color: 0x000000, toneMapped: false, polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -4 }),
    );
    const sm = this.shadow.material as THREE.MeshBasicMaterial;
    sm.onBeforeCompile = (sh) => {
      sh.uniforms.vOccTex = VEHICLE_OCCLUSION.tex;
      sh.uniforms.vOccInfo = VEHICLE_OCCLUSION.info;
      sh.fragmentShader = sh.fragmentShader
        .replace("#include <common>", "#include <common>" + OCC_GLSL)
        .replace("#include <dithering_fragment>", "#include <dithering_fragment>\n  gl_FragColor.a *= 1.0 - vOcclusion();");
    };
    sm.customProgramCacheKey = () => "veh-shadow";
    this.shadow.position.z = 0.03;
    this.shadow.renderOrder = 20;
    this.root.add(this.shadow);
  }

  /** over a generated clip the vehicle is blended (its alpha carries the occlusion); otherwise opaque */
  setBlended(on: boolean) {
    if (this.blended === on) return;
    this.blended = on;
    for (const m of this.mats) m.transparent = on;
  }
  private blended = false;

  /** a colour found later (the scan's) */
  setPaint(p: Paint) {
    this.spec.paint = p;
    this.paint.color.copy(p.color);
    this.paint.metalness = p.metal;
    this.paint.roughness = p.rough;
  }

  update(st: VehicleState, night: number, dt: number) {
    this.root.position.set(st.x, st.y, st.z);
    this.root.rotation.set(0, 0, st.yaw);
    this.root.visible = st.fade > 0.003;
    this.U.ground.value = st.z;
    this.U.fade.value = st.fade;
    (this.shadow.material as THREE.MeshBasicMaterial).opacity = st.fade;
    // wheels roll with the distance travelled (capped against strobing) and steer with the yaw rate
    const ds = Number.isNaN(this.lastDist) ? 0 : st.dist - this.lastDist;
    this.lastDist = st.dist;
    const r = this.wheels[0]?.r ?? TYRE_R;
    if (Math.abs(ds) > 3) this.spin = st.dist / r;
    else this.spin += THREE.MathUtils.clamp(ds / r, -0.42, 0.42);
    const steerT = st.speed > 0.8 ? THREE.MathUtils.clamp(Math.atan((this.wheelbase * st.yawRate) / st.speed), -0.55, 0.55) : this.steer;
    this.steer += (steerT - this.steer) * (1 - Math.exp(-dt / 0.15));
    for (const w of this.wheels) w.node.rotation.set(0, this.spin, w.front ? this.steer : 0, "ZYX");
    // brake lamps: decelerating, or stopped in traffic (not when parked); running lights at dawn
    const braking = !this.spec.parked && (st.accel < -1.3 || st.speed < 0.3) ? 1 : 0;
    this.brake += (braking - this.brake) * (1 - Math.exp(-dt / 0.08));
    const running = this.spec.parked ? 0 : 0.45;
    this.U.tail.value = Math.max(running, night * 1.1) + this.brake * 2.6;
    this.U.head.value = this.spec.parked ? 0 : 0.5 + night * 2;
    for (const m of this.tailMats) m.emissiveIntensity = this.U.tail.value;
  }

  dispose() {
    for (const m of this.mats) m.dispose();
    // the Model 3's geometries are shared with the kit; only procedural bodies and the shadow own theirs
    this.root.traverse((o) => {
      const m = o as THREE.Mesh;
      if (m.isMesh && (m === this.shadow || m.geometry.getAttribute("aMat"))) m.geometry.dispose();
    });
    (this.shadow.material as THREE.Material).dispose();
  }
}
