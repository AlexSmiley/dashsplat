// Contact shadows for the ego car and the movers: ambient occlusion of an overcast sky. The footage (07:15-07:21
// on 21 Nov 2026 in Jacksonville: the sun 1-5 degrees up, behind cloud) shows no cast shadows at all, so the only
// shadow is how much of the sky the car hides from the road: nearly all of it under the body, about half right
// at its sides, fading out a metre or two away, darkest at the tyres. Computed once by casting rays from the road
// against a box model of the car under a CIE overcast sky (brighter towards the zenith).
//
// One texture serves every vehicle: a mover's point is mapped into the ego car's frame by scaling it to the box
// inside the footprint and keeping metres (scaled by height) outside it, so a bus's shadow reaches further than a
// hatchback's but has the same soft profile (MOVER_SHADOW_GLSL).

import * as THREE from "three";

/** ego car: half length and width of the body (m), height (m), centre ahead of the rear axle (m) */
export const EGO = { halfL: 2.355, halfW: 0.925, height: 1.44, centreX: 1.365 };
/** extent of the texture around the car centre (m) */
export const AO_HALF_X = 4.3;
export const AO_HALF_Y = 2.9;
/** strongest darkening (fraction of the light removed under the body) */
const MAX_OCC = 0.9;

type Box = [number, number, number, number, number, number];

function carBoxes(): Box[] {
  const b: Box[] = [
    // lower body, shoulder, glasshouse (centre-relative x, y; z from the road)
    [-2.33, 2.33, -0.92, 0.92, 0.16, 0.8],
    [-2.25, 2.22, -0.9, 0.9, 0.8, 0.98],
    [-1.75, 1.15, -0.72, 0.72, 0.98, 1.44],
  ];
  for (const cx of [-EGO.centreX, 2.875 - EGO.centreX]) for (const s of [-1, 1]) {
    const y0 = s * 0.67, y1 = s * 0.91;
    b.push([cx - 0.3, cx + 0.3, Math.min(y0, y1), Math.max(y0, y1), 0, 0.62]);
    b.push([cx - 0.2, cx + 0.2, Math.min(y0, y1), Math.max(y0, y1), 0, 0.1]);
  }
  return b;
}

function hit(b: Box, px: number, py: number, pz: number, dx: number, dy: number, dz: number) {
  let t0 = 0, t1 = 1e9;
  const o = [px, py, pz], d = [dx, dy, dz];
  for (let a = 0; a < 3; a++) {
    const lo = b[a * 2], hi = b[a * 2 + 1];
    if (Math.abs(d[a]) < 1e-9) {
      if (o[a] < lo || o[a] > hi) return false;
      continue;
    }
    let ta = (lo - o[a]) / d[a], tb = (hi - o[a]) / d[a];
    if (ta > tb) [ta, tb] = [tb, ta];
    if (ta > t0) t0 = ta;
    if (tb < t1) t1 = tb;
    if (t0 > t1) return false;
  }
  return true;
}

function computeAo(nx: number, ny: number) {
  const boxes = carBoxes();
  // directions: a Fibonacci hemisphere (equal areas), weighted by cos(theta) and the CIE overcast sky (1 + 2 cos)/3
  const N = 112;
  const dirs: number[][] = [];
  let wsum = 0;
  for (let i = 0; i < N; i++) {
    const z = 1 - (i + 0.5) / N;
    const r = Math.sqrt(1 - z * z);
    const a = i * 2.399963229728653;
    const w = z * (1 + 2 * z) / 3;
    dirs.push([Math.cos(a) * r, Math.sin(a) * r, z, w]);
    wsum += w;
  }
  const occ = new Float32Array(nx * ny);
  for (let j = 0; j < ny; j++) {
    const y = -AO_HALF_Y + ((j + 0.5) / ny) * 2 * AO_HALF_Y;
    // symmetric left and right
    const jm = ny - 1 - j;
    if (jm < j) {
      for (let i = 0; i < nx; i++) occ[j * nx + i] = occ[jm * nx + i];
      continue;
    }
    for (let i = 0; i < nx; i++) {
      const x = -AO_HALF_X + ((i + 0.5) / nx) * 2 * AO_HALF_X;
      let blocked = 0;
      for (const d of dirs) {
        for (const b of boxes) if (hit(b, x, y, 0.005, d[0], d[1], d[2])) { blocked += d[3]; break; }
      }
      occ[j * nx + i] = blocked / wsum;
    }
  }
  return occ;
}

let tex: THREE.DataTexture | null = null;
/** RGBA8, RGB 0 and alpha = darkening, x along the car (west to east texels = rear to front), y to the left */
export function contactShadowTexture() {
  if (tex) return tex;
  const nx = 144, ny = 96;
  const occ = computeAo(nx, ny);
  const data = new Uint8Array(nx * ny * 4);
  for (let k = 0; k < nx * ny; k++) {
    // a little ambient bounce: fully hidden sky still leaves some light under the car
    const a = MAX_OCC * Math.pow(occ[k], 0.9);
    data[k * 4 + 3] = Math.round(255 * Math.min(1, a));
  }
  tex = new THREE.DataTexture(data, nx, ny, THREE.RGBAFormat, THREE.UnsignedByteType);
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.generateMipmaps = true;
  tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.colorSpace = THREE.NoColorSpace;
  tex.needsUpdate = true;
  return tex;
}

/** GLSL: with clean vehicles (moverInfo.y 1), what is left of a vehicle in the static splats (a car parked or
 *  queued long enough to be baked into the map) gives way inside its box, so the clean model stands alone;
 *  the road under it stays (only above 0.15 m) */
export const VEHICLE_CULL_GLSL = /* glsl */ `
float rsVehicleCull(sampler2D tex, float count, vec3 p) {
  float keep = 1.0;
  for (int k = 0; k < 24; k++) {
    if (float(k) >= count) break;
    vec4 a = texelFetch(tex, ivec2(0, k), 0);
    vec2 d = p.xy - a.xy;
    float h = p.z - a.z;
    // (a.w: the box's reach squared, so most splats leave after one read)
    if (dot(d, d) > a.w || h < 0.15) continue;
    vec4 b = texelFetch(tex, ivec2(1, k), 0);
    vec4 c = texelFetch(tex, ivec2(2, k), 0);
    if (h > c.z + 0.4) continue;
    vec2 l = abs(vec2(d.x * b.x + d.y * b.y, -d.x * b.y + d.y * b.x)) - b.zw;
    float out_ = max(l.x, l.y);
    keep *= smoothstep(0.1, 0.45, out_);
  }
  return keep;
}`;

/** GLSL: the movers' contact shadows at a point (1 = unshadowed). Per mover (row k of the mover texture):
 *  (x, y, ground z, strength), (cos yaw, sin yaw, half length, half width), (reach^2, ego height / height, -, -) */
export const MOVER_SHADOW_GLSL = /* glsl */ `
float rsMoverShadow(sampler2D tex, sampler2D ao, float count, vec3 p) {
  float sh = 1.0;
  for (int k = 0; k < 24; k++) {
    if (float(k) >= count) break;
    vec4 a = texelFetch(tex, ivec2(0, k), 0);
    vec2 d = p.xy - a.xy;
    vec4 c = texelFetch(tex, ivec2(2, k), 0);
    float dz = abs(p.z - a.z);
    if (dz > 0.6 || dot(d, d) > c.x) continue;
    vec4 b = texelFetch(tex, ivec2(1, k), 0);
    vec2 l = vec2(d.x * b.x + d.y * b.y, -d.x * b.y + d.y * b.x);
    // into the ego car's frame: scaled inside the footprint, metres (times the height ratio) outside it
    vec2 al = abs(l);
    vec2 egoHalf = vec2(${EGO.halfL.toFixed(3)}, ${EGO.halfW.toFixed(3)});
    vec2 e = mix(al / b.zw * egoHalf, egoHalf + (al - b.zw) * c.y, step(b.zw, al)) * sign(l);
    vec2 uv = vec2(e.x / ${(2 * AO_HALF_X).toFixed(3)} + 0.5, e.y / ${(2 * AO_HALF_Y).toFixed(3)} + 0.5);
    if (uv.x <= 0.0 || uv.x >= 1.0 || uv.y <= 0.0 || uv.y >= 1.0) continue;
    float occ = textureLod(ao, uv, 0.0).a * a.w;
    sh *= 1.0 - occ * (1.0 - smoothstep(0.15, 0.6, dz));
  }
  return sh;
}`;
