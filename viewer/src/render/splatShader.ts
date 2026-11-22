// Per-splat time and look on the GPU: a Spark dyno world modifier that reads each splat's packed times from a
// per-tile RGBA32UI texture (one texel per splat, indexed by the splat index) and applies existence,
// cross-fades, the scene grade, Memory-mode age colouring, the night grade with headlights and brake glow,
// distance haze and the far-field size clamp. All inputs are uniforms shared by every tile (one program), so
// scrubbing time in either direction only changes a float; Spark regenerates its splat buffer on the GPU.

import * as THREE from "three";
import { dyno } from "@sparkjsdev/spark";
import { contactShadowTexture, MOVER_SHADOW_GLSL, VEHICLE_CULL_GLSL } from "./contactShadow";

export { MOVER_SHADOW_GLSL };

export const TIME_TEX_WIDTH = 2048;

/** generative: a pre-rendered video of the chase view where there is one (render/generative.ts), live splats elsewhere */
export type ViewMode = "live" | "stitched" | "memory" | "generative";

/** Uniforms shared by every tile. */
export class SplatGlobals {
  readonly T = dyno.dynoFloat(0);
  /** x mode (0 live, 1 stitched, 2 memory), y night 0..1, z memory blend 0..1, w fade seconds */
  readonly params = dyno.dynoVec4(new THREE.Vector4(0, 0, 0, 0.12));
  /** x haze start, y haze end (metres from the haze centre), z haze strength, w 0..1: haze as fading out (over a
   *  photographed backdrop) instead of into the haze colour */
  readonly fog = dyno.dynoVec4(new THREE.Vector4(65, 140, 1, 0));
  /** over photographic layers: x, y fade-out start and end for non-ground splats (trees, poles) */
  readonly fog2 = dyno.dynoVec4(new THREE.Vector4(65, 140, 0, 0));
  /** xy haze centre (the car, or the camera in free view) */
  readonly hazeCentre = dyno.dynoVec4(new THREE.Vector4(0, 0, 0, 0));
  /** xy streaming focus, z loaded radius, w edge fade width */
  readonly loadFocus = dyno.dynoVec4(new THREE.Vector4(0, 0, 240, 40));
  readonly fogColor = dyno.dynoVec3(new THREE.Vector3(0.78, 0.81, 0.83));
  /** car pose for the night lights: x, y, cos yaw, sin yaw */
  readonly car = dyno.dynoVec4(new THREE.Vector4(0, 0, 1, 0));
  /** x headlights 0..1, y tail / brake glow 0..1, z car height */
  readonly lights = dyno.dynoVec4(new THREE.Vector4(0, 0, 0, 0));
  /** xyz white-balance gains, w "Natural" grade amount 0..1 (0 = camera colours) */
  readonly grade = dyno.dynoVec4(new THREE.Vector4(1, 1, 1, 1));
  /** x exposure, y contrast (S-curve mix), z saturation, w black point */
  readonly grade2 = dyno.dynoVec4(new THREE.Vector4(1.0, 0.48, 1.2, 0.05));
  /** movers' contact shadows, darkening the splats (and the ground atlas) under them: three RGBA32F texels per
   *  mover (row k): (x, y, ground z, strength x presence), (cos yaw, sin yaw, half length, half width),
   *  (reach^2, ego height / height, -, -); x of moverInfo is the count. The profile is the ego car's
   *  (render/contactShadow.ts) */
  readonly moverData = new Float32Array(MAX_MOVER_SHADOWS * 3 * 4);
  readonly moverTexture = (() => {
    const t = new THREE.DataTexture(this.moverData, 3, MAX_MOVER_SHADOWS, THREE.RGBAFormat, THREE.FloatType);
    t.minFilter = t.magFilter = THREE.NearestFilter;
    t.generateMipmaps = false;
    t.needsUpdate = true;
    return t;
  })();
  readonly movers = dyno.dynoSampler2D(this.moverTexture, "moverTex");
  readonly contactAo = dyno.dynoSampler2D(contactShadowTexture(), "contactAo");
  /** ground atlas coverage around the focus (render/groundAtlas.ts): RGBA16F, r = opaque share of the atlas,
   *  g = mesh height; atlasInfo = origin x, y, size (m), height tolerance (0 = no atlas) */
  readonly atlasCoverageTexture = (() => {
    const t = new THREE.DataTexture(new Uint16Array(ATLAS_COV_N * ATLAS_COV_N * 4), ATLAS_COV_N, ATLAS_COV_N, THREE.RGBAFormat, THREE.HalfFloatType);
    t.minFilter = t.magFilter = THREE.LinearFilter;
    t.generateMipmaps = false;
    t.needsUpdate = true;
    return t;
  })();
  readonly atlasCoverage = dyno.dynoSampler2D(this.atlasCoverageTexture, "atlasCov");
  readonly atlasInfo = dyno.dynoVec4(new THREE.Vector4(0, 0, 1, 0));
  readonly moverInfo = dyno.dynoVec4(new THREE.Vector4(0, 0, 0, 0));
  /** director looks (?director=, render/director.ts): x 0 none, 1 points, 2 depth, 3 height; y amount 0..1;
   *  z a point's size (radians per metre of distance); w the ground height under the car */
  readonly director = dyno.dynoVec4(new THREE.Vector4(0, 0, 3.2e-4, 0));
  /** the camera's position (director looks) */
  readonly camPos = dyno.dynoVec4(new THREE.Vector4(0, 0, 0, 0));
}

export const MAX_MOVER_SHADOWS = 24;
/** cells of the ground atlas coverage map (0.32 m each: 32 per 10.24 m atlas tile, 164 m across) */
export const ATLAS_COV_N = 512;

/** GLSL: the night look at a surface point: dark blue grade, low-beam pools, brake glow (shared by the splats and
 *  the ground atlas). car = x, y, cos yaw, sin yaw; L = headlights, brake glow, car height */
export const NIGHT_GLSL = /* glsl */ `
vec3 rsNight(vec3 rgb, vec3 c, vec4 car, vec4 L, float night) {
  vec2 dd = c.xy - car.xy;
  float ex = dd.x * car.z + dd.y * car.w;
  float ey = -dd.x * car.w + dd.y * car.z;
  float h = max(0.0, c.z - L.z);
  // low beams from the lamps at 0.7 m: inverse-square falloff with distance, a wide soft lateral pattern and a
  // cut-off above the beam
  float hl = 0.0;
  float ax = ex - 3.65;
  if (ax > 0.3 && h < 3.0) {
    float w = 1.0 + ax * 0.42;
    float q = ey / w;
    float r2 = ax * ax + ey * ey + 0.49;
    hl = exp(-q * q * 2.2) * smoothstep(0.3, 4.0, ax) * min(1.7, 95.0 / r2) * (1.0 - smoothstep(1.4, 3.0, h)) * L.x;
  }
  float br = 0.0;
  float bx = -0.98 - ex;
  if (bx > 0.0 && bx < 10.0 && h < 1.6) {
    float q = ey / (1.0 + bx * 0.22);
    br = exp(-q * q * 2.0) * min(1.0, 2.2 / (bx * bx + 0.8)) * L.y;
  }
  float amb = exp(-(ex - 1.4) * (ex - 1.4) / 30.0 - ey * ey / 6.0) * 0.14 * L.x;
  vec3 lit = rgb * (hl + amb) * vec3(1.12, 1.0, 0.84);
  vec3 dark = rgb * vec3(0.17, 0.19, 0.27) + lit + rgb * br * vec3(1.5, 0.18, 0.1) + vec3(0.06, 0.005, 0.003) * br;
  return mix(rgb, dark, night);
}`;

/** the Natural grade on the CPU, for colours that must match graded splats (the ground apron) */
export function gradeColour(c: THREE.Color, G: SplatGlobals, photo?: { kFloor: number }) {
  const g = G.grade.value;
  const g2 = photo ? new THREE.Vector4(G.grade2.value.x * 0.98, 0, G.grade2.value.z, 0) : G.grade2.value;
  const L = [0.2126, 0.7152, 0.0722];
  const dot = (v: number[]) => v[0] * L[0] + v[1] * L[1] + v[2] * L[2];
  const ss = (a: number, b: number, x: number) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };
  const src = [c.r, c.g, c.b];
  const mx = Math.max(...src), mn = Math.min(...src);
  const sat = (mx - mn) / Math.max(mx, 1e-3);
  const k = Math.max(photo?.kFloor ?? 0, (1 - 0.5 * ss(0.1, 0.45, sat)) * (0.55 + 0.45 * ss(0.02, 0.2, dot(src))));
  const gain = [g.x, g.y, g.z].map((v) => 1 + (v - 1) * k);
  const gl = dot(gain);
  const x = src.map((v, i) => (v * gain[i]) / gl);
  const m0 = Math.max(Math.min(src[0], src[2]) - src[1], 0), m1 = Math.min(x[0], x[2]) - x[1];
  if (m1 > m0) x[1] += m1 - m0;
  const pv = Math.min(x[0] + 0.03, x[2]) - x[1];
  const pd = ss(0, 0.06, pv) * (1 - ss(0.4, 0.7, sat));
  x[1] += Math.max(Math.min(x[0], x[2]) - x[1], 0) * pd * 0.85;
  const lp = dot(x);
  for (let i = 0; i < 3; i++) x[i] += (lp - x[i]) * pd * 0.6;
  const l0 = Math.max(dot(x), 1e-4);
  let t = Math.min(1, Math.max(0, (l0 * g2.x - g2.w) / (1 - g2.w)));
  t = t + (t * t * (3 - 2 * t) - t) * g2.y;
  const y = x.map((v) => (v * t) / l0);
  const l = dot(y);
  const sl = 1 + (g2.z - 1) * ss(0.05, 0.3, l) * (1 - ss(0.3, 0.65, sat));
  const out = y.map((v) => Math.min(1, Math.max(0, l + (v - l) * sl)));
  const w = g.w;
  return new THREE.Color(c.r + (out[0] - c.r) * w, c.g + (out[1] - c.g) * w, c.b + (out[2] - c.b) * w);
}

/** the Natural grade, hue-safe (GLSL, shared by splats and the sky panorama); mirrored by gradeColour() */
export const GRADE_GLSL = /* glsl */ `
// kFloor: the least share of the white balance any colour gets (1 for a photographed sky, whose tint is the
// camera's cast rather than an object's colour)
vec3 rsGradeK(vec3 rgb, vec4 GR, vec4 G2, float kFloor) {
  const vec3 LUM = vec3(0.2126, 0.7152, 0.0722);
  // partial chromatic adaptation: near-neutral colours (asphalt, concrete, cloud) get most of the white
  // balance, saturated colours and deep shadow much less, so greens stay green and brick, wood and skin keep
  // their hue; the gains are normalised so the balance never changes brightness
  float mx = max(rgb.r, max(rgb.g, rgb.b));
  float mn = min(rgb.r, min(rgb.g, rgb.b));
  float sat = (mx - mn) / max(mx, 1e-3);
  float lum = dot(rgb, LUM);
  float k = max(kFloor, (1.0 - 0.5 * smoothstep(0.1, 0.45, sat)) * mix(0.55, 1.0, smoothstep(0.02, 0.2, lum)));
  vec3 gain = mix(vec3(1.0), GR.xyz, k);
  vec3 x = rgb * gain / dot(gain, LUM);
  // magenta guard: the balance may not push green below both red and blue unless it already was
  float m0 = max(min(rgb.r, rgb.b) - rgb.g, 0.0);
  float m1 = min(x.r, x.b) - x.g;
  if (m1 > m0) x.g += m1 - m0;
  // and little in a road scene is really violet, lavender or purple (blue above green with red level with or
  // above green): that is sky light bled into foliage and shade, so muted ones are drawn back toward grey;
  // sky blue (green above red) and strongly saturated paint and signs are spared
  float pv = min(x.r + 0.03, x.b) - x.g;
  float pd = smoothstep(0.0, 0.06, pv) * (1.0 - smoothstep(0.4, 0.7, sat));
  x.g += max(min(x.r, x.b) - x.g, 0.0) * pd * 0.85;
  float lp = dot(x, LUM);
  x = mix(x, vec3(lp), pd * 0.6);
  // contrast on luminance only, keeping channel ratios: a black point and S-curve on each channel would
  // saturate dark colours and turn shaded foliage purple
  float l0 = max(dot(x, LUM), 1e-4);
  float t = clamp((l0 * G2.x - G2.w) / (1.0 - G2.w), 0.0, 1.0);
  t = mix(t, t * t * (3.0 - 2.0 * t), G2.y);
  x *= t / l0;
  // a little richness, but not in shadow and not on colours that are already saturated
  float l = dot(x, LUM);
  float sl = mix(1.0, G2.z, smoothstep(0.05, 0.3, l) * (1.0 - smoothstep(0.3, 0.65, sat)));
  x = clamp(mix(vec3(l), x, sl), 0.0, 1.0);
  return mix(rgb, x, GR.w);
}
vec3 rsGrade(vec3 rgb, vec4 GR, vec4 G2) { return rsGradeK(rgb, GR, G2, 0.0); }
// photographs (photo layers): the white balance and saturation only. The black point and S-curve make up for
// the soft contrast of fused splats; a photograph already has the camera's, and the curve took a sky that is
// brighter than the road in the photographs down to a storm sky
// (exposure 0.98: the splat road comes out about 10 % under the photographs' road after its curve; this splits
// the difference so the sky's brightness and its ratio to the road both stay within a few percent)
vec3 rsGradePhoto(vec3 rgb, vec4 GR, vec4 G2, float kFloor) { return rsGradeK(rgb, GR, vec4(G2.x * 0.98, 0.0, G2.z, 0.0), kFloor); }`;

/** GLSL: the director looks' colour ramps. Depth: warm white near, the accent blue, deep navy far. Height above
 *  the road: slate, blue at kerb height, pale cyan at a car's roof, warm white in the crowns, amber above */
export const DIRECTOR_GLSL = /* glsl */ `
vec3 rsDepthRamp(float t) {
  vec3 c = mix(vec3(1.0, 0.95, 0.88), vec3(0.32, 0.56, 1.0), smoothstep(0.0, 0.35, t));
  return mix(c, vec3(0.05, 0.08, 0.2), smoothstep(0.35, 1.0, t));
}
vec3 rsHeightRamp(float h) {
  vec3 c = mix(vec3(0.16, 0.19, 0.26), vec3(0.25, 0.5, 1.0), smoothstep(0.08, 0.6, h));
  c = mix(c, vec3(0.62, 0.85, 1.0), smoothstep(0.8, 2.5, h));
  c = mix(c, vec3(1.0, 0.94, 0.82), smoothstep(3.0, 7.0, h));
  return mix(c, vec3(1.0, 0.62, 0.28), smoothstep(9.0, 16.0, h));
}`;

export type TileUniforms = {
  timeTex: ReturnType<typeof dyno.dynoUsampler2D>;
  /** x time base, y quantum (s), z load / presence fade 0..1, w 1 = transient tile */
  tile: dyno.DynoVec4<THREE.Vector4, string>;
  /** x 1 = predicted content (never observed), y ghost 0..1 (a mover whose pose is predicted), z 1 = not ground,
   *  w 1 = a mover's own splats (its contact shadow does not darken it) */
  obj: dyno.DynoVec4<THREE.Vector4, string>;
};

export function makeTimeTexture(times: Uint32Array, count: number) {
  const rows = Math.max(1, Math.ceil(count / TIME_TEX_WIDTH));
  const tex = new THREE.DataTexture(times, TIME_TEX_WIDTH, rows, THREE.RGBAIntegerFormat, THREE.UnsignedIntType);
  tex.internalFormat = "RGBA32UI";
  tex.minFilter = THREE.NearestFilter;
  tex.magFilter = THREE.NearestFilter;
  tex.generateMipmaps = false;
  tex.needsUpdate = true;
  return tex;
}

export function makeTileUniforms(tex: THREE.DataTexture, base: number, quantum: number, transient: boolean, predicted = false, nonGround = false): TileUniforms {
  return {
    timeTex: dyno.dynoUsampler2D(tex, "timeTex"),
    tile: dyno.dynoVec4(new THREE.Vector4(base, quantum, 0, transient ? 1 : 0), "tileParams") as dyno.DynoVec4<THREE.Vector4, string>,
    obj: dyno.dynoVec4(new THREE.Vector4(predicted ? 1 : 0, 0, nonGround ? 1 : 0, 0), "objParams") as dyno.DynoVec4<THREE.Vector4, string>,
  };
}

export function makeTimeModifier(G: SplatGlobals, U: TileUniforms) {
  return dyno.dynoBlock({ gsplat: dyno.Gsplat }, { gsplat: dyno.Gsplat }, ({ gsplat }) => {
    const d = new dyno.Dyno({
      inTypes: {
        gsplat: dyno.Gsplat,
        timeTex: "usampler2D",
        tile: "vec4",
        obj: "vec4",
        T: "float",
        params: "vec4",
        fog: "vec4",
        fog2: "vec4",
        hazeCentre: "vec4",
        loadFocus: "vec4",
        fogColor: "vec3",
        car: "vec4",
        lights: "vec4",
        grade: "vec4",
        grade2: "vec4",
        moverTex: "sampler2D",
        contactAo: "sampler2D",
        moverInfo: "vec4",
        atlasCov: "sampler2D",
        atlasInfo: "vec4",
        director: "vec4",
        camPos: "vec4",
      },
      outTypes: { gsplat: dyno.Gsplat },
      inputs: {
        gsplat,
        timeTex: U.timeTex,
        tile: U.tile,
        obj: U.obj,
        T: G.T,
        params: G.params,
        fog: G.fog,
        fog2: G.fog2,
        hazeCentre: G.hazeCentre,
        loadFocus: G.loadFocus,
        fogColor: G.fogColor,
        car: G.car,
        lights: G.lights,
        grade: G.grade,
        grade2: G.grade2,
        moverTex: G.movers,
        contactAo: G.contactAo,
        moverInfo: G.moverInfo,
        atlasCov: G.atlasCoverage,
        atlasInfo: G.atlasInfo,
        director: G.director,
        camPos: G.camPos,
      },
      globals: () => [
        dyno.defineGsplat,
        `float rsDecode(uint v, vec4 tile) { return v == 65535u ? 1e9 : tile.x + float(v) * tile.y; }`,
        GRADE_GLSL,
        MOVER_SHADOW_GLSL,
        VEHICLE_CULL_GLSL,
        NIGHT_GLSL,
        DIRECTOR_GLSL,
      ],
      statements: ({ inputs: i, outputs: o }) => {
        const g = o.gsplat;
        return dyno.unindentLines(`
          ${g} = ${i.gsplat};
          if (isGsplatActive(${g}.flags)) {
            int idx = ${g}.index;
            uvec4 tw = texelFetch(${i.timeTex}, ivec2(idx & 2047, idx >> 11), 0);
            vec4 tile = ${i.tile};
            vec4 P = ${i.params};
            float T = ${i.T};
            uint b16 = tw.x & 65535u;
            uint d16 = tw.x >> 16;
            float birth = rsDecode(b16, tile);
            float death = rsDecode(d16, tile);
            float alpha = tile.z;
            float sc = 1.0;
            bool alive = true;
            if (P.x > 0.5 && P.x < 1.5) {
              // stitched: the final state of the map, static splats only
              alive = d16 == 65535u && tile.w < 0.5;
            } else {
              alive = T >= birth && T < death;
              float fd = min(P.w, 0.3 * (death - birth));
              float fin = clamp((T - birth) / fd, 0.0, 1.0);
              float fout = clamp((death - T) / fd, 0.0, 1.0);
              alpha *= fin * fin * (3.0 - 2.0 * fin) * fout * fout * (3.0 - 2.0 * fout);
              sc = mix(0.85, 1.0, fin);
            }
            vec3 c = ${g}.center;
            float dH = length(c.xy - ${i.hazeCentre}.xy);
            vec4 F = ${i.fog};
            vec4 LF = ${i.loadFocus};
            // gone by the end of the haze and at the edge of what is loaded
            alpha *= 1.0 - smoothstep(F.y - 4.0, F.y + 10.0, dH);
            alpha *= 1.0 - smoothstep(LF.z - LF.w, LF.z, length(c.xy - LF.xy));
            // ground texture atlas: ground-layer splats give way where a resident atlas texel is opaque and the
            // splat lies within the tolerance of the mesh (bilinear coverage, so the hand-over is soft; kept
            // through transparent texels, while tiles stream and outside the atlas)
            vec4 GA = ${i.atlasInfo};
            if (GA.w > 0.0 && ${i.obj}.z < 0.5 && ${i.obj}.w < 0.5) {
              vec2 auv = (c.xy - GA.xy) / GA.z;
              if (auv.x > 0.0 && auv.y > 0.0 && auv.x < 1.0 && auv.y < 1.0) {
                vec4 cv = texture(${i.atlasCov}, auv);
                if (abs(c.z - cv.g) < GA.w) alpha *= 1.0 - smoothstep(0.55, 0.95, cv.r);
              }
            }
            // clean vehicles: the static splats' leftovers of a vehicle give way inside its box
            if (${i.moverInfo}.y > 0.5 && ${i.obj}.w < 0.5 && alpha > 0.002) alpha *= rsVehicleCull(${i.moverTex}, ${i.moverInfo}.x, c);
            if (!alive || alpha <= 0.002) {
              ${g}.flags = ${g}.flags & ~GSPLAT_FLAG_ACTIVE;
            } else {
              vec3 rgb = ${g}.rgba.rgb;
              // Natural grade: white balance on the road surface, black point, gentle S-curve, a little saturation
              vec4 GR = ${i.grade};
              if (GR.w > 0.0) rgb = rsGrade(rgb, GR, ${i.grade2});
              vec4 OB = ${i.obj};
              float lumP = dot(rgb, vec3(0.3, 0.59, 0.11));
              if (OB.x > 0.5) {
                // predicted (never observed): softer everywhere, its own violet in Memory mode
                alpha *= mix(0.6, 0.5, P.z);
                sc *= 1.12;
                rgb = mix(rgb, vec3(lumP * 0.55 + 0.22, lumP * 0.45 + 0.18, lumP * 0.6 + 0.42), 0.8 * P.z);
              } else if (P.x > 1.5 && P.z > 0.0) {
                float e0 = rsDecode(tw.y & 65535u, tile);
                float s1 = rsDecode(tw.y >> 16, tile);
                float e1 = rsDecode(tw.z & 65535u, tile);
                float s2 = rsDecode(tw.z >> 16, tile);
                float e2 = rsDecode(tw.w & 65535u, tile);
                float seen = min(T, e0);
                if (T >= s1) seen = max(seen, min(T, e1));
                if (T >= s2) seen = max(seen, min(T, e2));
                float age = max(0.0, T - seen);
                float k = 1.0 - exp(-age / 15.0);
                float lum = dot(rgb, vec3(0.3, 0.59, 0.11));
                vec3 cool = vec3(lum * 0.55 + 0.05, lum * 0.68 + 0.08, lum * 0.92 + 0.17);
                float m = smoothstep(0.05, 0.6, age) * (0.45 + 0.55 * k);
                vec3 mem = mix(rgb, cool, m) * (1.0 - 0.3 * k);
                rgb = mix(rgb, mem, P.z);
                // a mover carried forward while no camera sees it reads as a ghost
                float gh = OB.y * P.z;
                rgb = mix(rgb, cool, 0.7 * gh);
                alpha *= 1.0 - 0.6 * gh;
              }
              // movers' soft contact shadows on the splats under them (sorted with everything else, so a tree in
              // front of a mover hides its shadow too); where there are no ground splats, a decal on the apron
              // (objects.ts) carries it
              if (${i.moverInfo}.x > 0.5 && OB.w < 0.5) rgb *= rsMoverShadow(${i.moverTex}, ${i.contactAo}, ${i.moverInfo}.x, c);
              if (P.y > 0.0) rgb = rsNight(rgb, c, ${i.car}, ${i.lights}, P.y);
              // distance haze into the sky's horizon colour, or (over photographic layers) fading out to show them
              float hz = smoothstep(F.x, F.y, dH);
              float hk = F.z * hz * (2.0 - hz);
              rgb = mix(rgb, ${i.fogColor}, hk * (1.0 - F.w));
              alpha *= 1.0 - hk * F.w;
              // non-ground splats (trees, poles; anything 2.5 m above the road, as ground tiles carry poles too) hand
              // over to the photographed backdrop sooner: far ones are blobs where the backdrop has the photograph
              if ((${i.obj}.z > 0.5 || c.z - ${i.lights}.z > 2.5) && F.w > 0.0) {
                vec4 F2 = ${i.fog2};
                float ho = smoothstep(F2.x, F2.y, dH);
                alpha *= 1.0 - ho * (2.0 - ho) * F.w;
              }
              // far-field streaks: cap the elongation of distant splats
              vec3 s = ${g}.scales;
              float far = smoothstep(45.0, 100.0, dH);
              if (far > 0.0) {
                float mx = max(s.x, max(s.y, s.z));
                float mn = min(s.x, min(s.y, s.z));
                float md = s.x + s.y + s.z - mx - mn;
                s = min(s, vec3(md * mix(8.0, 2.2, far)));
              }
              float a = ${g}.rgba.a * alpha;
              s *= sc;
              vec4 DR = ${i.director};
              if (DR.x > 0.5 && DR.y > 0.0) {
                float dc = length(c - ${i.camPos}.xyz);
                if (DR.x < 1.5) {
                  // points: each splat's centre as an opaque dot a few pixels across, growing back into the splat
                  // (in log scale, so small and large splats arrive together)
                  vec3 ps = vec3(max(1e-4, dc * DR.z));
                  s = exp(mix(log(max(s, vec3(1e-5))), log(ps), DR.y));
                  a = mix(a, step(0.04, ${g}.rgba.a) * alpha, DR.y);
                } else {
                  float lum = dot(rgb, vec3(0.3, 0.59, 0.11));
                  vec3 ramp = DR.x < 2.5 ? rsDepthRamp(clamp((dc - 4.0) / 110.0, 0.0, 1.0)) : rsHeightRamp(c.z - DR.w);
                  rgb = mix(rgb, ramp * (0.55 + 0.9 * lum), DR.y);
                }
              }
              ${g}.rgba = vec4(rgb, a);
              ${g}.scales = s;
              if (${g}.rgba.a < 0.002) ${g}.flags = ${g}.flags & ~GSPLAT_FLAG_ACTIVE;
            }
          }
        `);
      },
    });
    return { gsplat: d.outputs.gsplat };
  });
}
