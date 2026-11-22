// Background dome: the scene's equirectangular sky panorama when map.json names one (centre column faces east,
// up is +z), otherwise a procedural dawn-overcast gradient. A panorama is measured when it loads; a dark or
// blotchy one is blended toward the procedural sky (see panoWeight), and the car's reflections never get
// darker than the procedural sky. An environment map (map.json "environment": environment.json beside its
// equirectangular map, pipeline/envmap) replaces both: shown whole and sharp, without the cloud deck, its
// horizon colour is the haze colour, and the car reflects it. The band at the horizon and everything below it
// is exactly the haze colour the splats fade into, so the end of the map has no visible edge. Dithered to
// avoid banding. Its shader also draws the car's environment (render/environment.ts).

import * as THREE from "three";
import { GRADE_GLSL } from "./splatShader";

const vert = /* glsl */ `
varying vec3 vDir;
void main() {
  vDir = position;
  vec4 p = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  gl_Position = vec4(p.xy, p.w * 0.99999, p.w);
}`;

const frag = /* glsl */ `
uniform sampler2D map;
uniform float useMap;
uniform float night;
uniform vec3 zenith;
uniform vec3 fogColor;
uniform float band;
uniform float dither;
uniform vec4 gradeA;
uniform vec4 gradeB;
uniform float panoBlend;
uniform float envFloor;
uniform float envMap;
uniform float mapBias;
// photographed sky (photo layers): up to four knots' skies blended by weight, in the photo frame (rotated by
// photoYaw = cos, sin of the frame's yaw back), graded like the splats
uniform sampler2D photo0;
uniform sampler2D photo1;
uniform sampler2D photo2;
uniform sampler2D photo3;
uniform vec4 photoW;
uniform float photoAmt;
uniform vec2 photoElev;
uniform vec2 photoYaw;
// the car's environment only (CarEnvironment): the latest captured frame, projected back onto the directions it
// shows (vanishing points through the capture camera); directions outside it take its nearest edge, the rear
// hemisphere is mirrored forward; and a linear output for image-based lighting
uniform sampler2D cap;
uniform float capAmt;
uniform mat4 capVP;
uniform vec2 capCarFwd;
uniform vec2 capCamFwd;
uniform float capHalfFov;
uniform float envLinear;
// an overcast cloud deck (the footage's sky is broken cloud; the scene's panorama is a smooth gradient): its
// amount, the camera's position on it (km, so it slides overhead as the car drives) plus a slow drift
uniform float clouds;
uniform vec2 cloudPos;
varying vec3 vDir;
${GRADE_GLSL}
const float PI = 3.141592653589793;
float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
float vnoise(vec2 p) {
  vec2 i = floor(p), f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), u.x), mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), u.x), u.y);
}
float fbm(vec2 p) {
  float a = 0.5, s = 0.0;
  mat2 r = mat2(0.8, -0.6, 0.6, 0.8);
  for (int k = 0; k < 5; k++) { s += a * vnoise(p); p = r * p * 2.03 + 17.1; a *= 0.5; }
  return s;
}
void main() {
  vec3 d = normalize(vDir);
  vec3 proc = mix(fogColor, zenith, pow(clamp(d.z, 0.0, 1.0), 0.75));
  vec3 col = proc;
  if (useMap > 0.5 && panoBlend > 0.0) {
    float u = 0.5 - atan(d.y, d.x) / (2.0 * PI);
    float v = 0.5 + asin(clamp(d.z, -1.0, 1.0)) / PI;
    // a soft, mip-blurred read, graded like the splats (same cameras, same cast)
    vec3 pano = rsGrade(texture2D(map, vec2(u, max(v, 0.505)), mapBias).rgb, gradeA, gradeB);
    col = mix(proc, pano, panoBlend);
  }
  // the car's environment: never darker than the procedural sky
  col = max(col, proc * envFloor);
  if (clouds * (1.0 - envMap) > 0.0 && d.z > 0.0) {
    // the deck 1.5 km up, seen through the direction: wide soft masses, darker grey undersides, lighter gaps,
    // thinning towards the horizon where it is seen edge-on through the haze
    vec2 p = cloudPos + d.xy / max(d.z, 0.035) * 1.5;
    float n = fbm(p * 1.7);
    float cov = smoothstep(0.32, 0.6, n);
    vec3 under = col * vec3(0.74, 0.79, 0.82);
    vec3 cc = mix(col * 1.04, under, smoothstep(0.45, 0.8, n));
    col = mix(col, cc, cov * clouds * smoothstep(0.015, 0.16, d.z));
  }
  // horizon band: exactly the haze colour up to a few degrees, then blending into the sky
  col = mix(fogColor, col, smoothstep(0.0, band, d.z));
  if (photoAmt > 0.0) {
    vec3 dp = vec3(photoYaw.x * d.x - photoYaw.y * d.y, photoYaw.y * d.x + photoYaw.x * d.y, d.z);
    float e = degrees(atan(dp.z, length(dp.xy)));
    vec2 uv = vec2(fract(0.5 + atan(dp.y, dp.x) / (2.0 * PI)), clamp((photoElev.y - e) / (photoElev.y - photoElev.x), 0.0, 1.0));
    vec3 ps = vec3(0.0);
    if (photoW.x > 0.0) ps += photoW.x * texture2D(photo0, uv).rgb;
    if (photoW.y > 0.0) ps += photoW.y * texture2D(photo1, uv).rgb;
    if (photoW.z > 0.0) ps += photoW.z * texture2D(photo2, uv).rgb;
    if (photoW.w > 0.0) ps += photoW.w * texture2D(photo3, uv).rgb;
    // the full white balance (its tint is the camera's cast) and about half the chroma: a pale overcast, not teal
    vec3 pg = rsGradePhoto(ps, gradeA, gradeB, gradeA.w);
    pg = mix(vec3(dot(pg, vec3(0.2126, 0.7152, 0.0722))), pg, mix(1.0, 0.55, gradeA.w));
    col = mix(col, pg, photoAmt);
  }
  if (capAmt > 0.0 && d.z > -0.08) {
    vec3 dd = d;
    float df = dot(dd.xy, capCarFwd);
    float back = step(df, 0.0);
    dd.xy -= 2.0 * min(df, 0.0) * capCarFwd;
    // clamp the azimuth to the capture's field of view (keeping the elevation): the nearest frame edge
    float h = length(dd.xy);
    float az = atan(capCamFwd.x * dd.y - capCamFwd.y * dd.x, dot(capCamFwd, dd.xy));
    float ca = clamp(az, -capHalfFov, capHalfFov);
    float excess = abs(az - ca);
    float a0 = atan(capCamFwd.y, capCamFwd.x) + ca;
    vec3 dc = vec3(cos(a0) * h, sin(a0) * h, dd.z);
    vec4 cp = capVP * vec4(dc, 0.0);
    if (cp.w > 0.05) {
      vec2 uv = cp.xy / cp.w * 0.5 + 0.5;
      float w = (1.0 - 0.5 * smoothstep(0.0, 0.9, excess)) * mix(1.0, 0.8, back);
      // above the frame: the sky dome
      w *= 1.0 - smoothstep(0.96, 1.04, uv.y);
      w *= smoothstep(-0.08, -0.02, d.z);
      vec3 cc = texture2D(cap, clamp(uv, vec2(0.003), vec2(0.997))).rgb;
      col = mix(col, cc, w * capAmt);
    }
  }
  vec3 nightCol = col * vec3(0.10, 0.12, 0.19) + vec3(0.012, 0.016, 0.028);
  col = mix(col, nightCol, night);
  col += (hash(gl_FragCoord.xy) - 0.5) * dither / 255.0;
  // display code values to linear radiance, so a mirror shows the scene as bright as it is on screen
  // (three quarters of the colour: a silver car under the teal morning sky reads silver, as in the footage)
  if (envLinear > 0.5) col = pow(max(mix(vec3(dot(col, vec3(0.2126, 0.7152, 0.0722))), col, 0.75), vec3(0.0)), vec3(2.2));
  gl_FragColor = vec4(col, 1.0);
}`;

export class Sky {
  readonly mesh: THREE.Mesh;
  readonly material: THREE.ShaderMaterial;
  /** horizon colour of the panorama, ungraded (display space) */
  horizon = new THREE.Color(0.8, 0.82, 0.84);
  /** the procedural sky's horizon */
  static readonly PROC_HORIZON = new THREE.Color(0.8, 0.82, 0.84);
  /** measured panorama: mean luminance of the upper sky, blotchiness (mean colour deviation across each row),
   *  and the weight it is shown with (0 = procedural sky only) */
  panoStats: { lum: number; blotch: number; weight: number } | null = null;
  /** the scene's environment map, when it has one */
  environment: { map: string; horizon: [number, number, number] } | null = null;

  constructor() {
    this.material = new THREE.ShaderMaterial({
      vertexShader: vert,
      fragmentShader: frag,
      uniforms: {
        map: { value: null },
        useMap: { value: 0 },
        night: { value: 0 },
        zenith: { value: new THREE.Color(0.55, 0.62, 0.7) },
        fogColor: { value: new THREE.Color(0.8, 0.82, 0.84) },
        band: { value: 0.16 },
        dither: { value: 1.5 },
        gradeA: { value: new THREE.Vector4(1, 1, 1, 0) },
        gradeB: { value: new THREE.Vector4(1, 0, 1, 0) },
        panoBlend: { value: 1 },
        envFloor: { value: 0 },
        envMap: { value: 0 },
        mapBias: { value: 2.5 },
        photo0: { value: null },
        photo1: { value: null },
        photo2: { value: null },
        photo3: { value: null },
        photoW: { value: new THREE.Vector4() },
        photoAmt: { value: 0 },
        photoElev: { value: new THREE.Vector2(-9, 64) },
        photoYaw: { value: new THREE.Vector2(1, 0) },
        cap: { value: null },
        capAmt: { value: 0 },
        capVP: { value: new THREE.Matrix4() },
        capCarFwd: { value: new THREE.Vector2(1, 0) },
        capCamFwd: { value: new THREE.Vector2(1, 0) },
        capHalfFov: { value: 0.6 },
        envLinear: { value: 0 },
        clouds: { value: 0 },
        cloudPos: { value: new THREE.Vector2() },
      },
      side: THREE.BackSide,
      depthWrite: false,
      depthTest: false,
    });
    this.mesh = new THREE.Mesh(new THREE.SphereGeometry(400, 48, 24), this.material);
    this.mesh.renderOrder = -1000;
    this.mesh.frustumCulled = false;
    this.mesh.name = "sky";
  }

  async loadPanorama(url: string | null, envUrl: string | null = null) {
    this.material.uniforms.useMap.value = 0;
    this.material.uniforms.panoBlend.value = 1;
    this.material.uniforms.envMap.value = 0;
    this.material.uniforms.mapBias.value = 2.5;
    this.panoStats = null;
    this.environment = null;
    this.horizon.copy(Sky.PROC_HORIZON);
    if (envUrl) {
      try {
        const env = (await (await fetch(envUrl)).json()) as { map: string; horizon: [number, number, number] };
        this.material.uniforms.map.value = await loadMap(new URL(env.map, envUrl).href);
        this.material.uniforms.useMap.value = 1;
        this.material.uniforms.envMap.value = 1;
        this.material.uniforms.mapBias.value = 0;
        this.horizon.setRGB(env.horizon[0] / 255, env.horizon[1] / 255, env.horizon[2] / 255);
        this.environment = env;
        return;
      } catch (e) {
        console.warn("environment map failed, using the panorama", e);
      }
    }
    if (!url) return;
    try {
      const tex = await loadMap(url);
      this.material.uniforms.map.value = tex;
      this.material.uniforms.useMap.value = 1;
      const img = tex.image as CanvasImageSource;
      const c = document.createElement("canvas");
      c.width = 256;
      c.height = 128;
      const ctx = c.getContext("2d", { willReadFrequently: true });
      if (ctx) {
        ctx.drawImage(img, 0, 0, 256, 128);
        const px = ctx.getImageData(0, 58, 256, 5).data;
        let r = 0, g = 0, b = 0;
        for (let i = 0; i < px.length; i += 4) { r += px[i]; g += px[i + 1]; b += px[i + 2]; }
        const n = px.length / 4;
        this.horizon.setRGB(r / n / 255, g / n / 255, b / n / 255);
        this.panoStats = measure(ctx.getImageData(0, 0, 256, 52).data, 256, 52);
        this.material.uniforms.panoBlend.value = this.panoStats.weight;
        console.info(`sky.jpg: luminance ${this.panoStats.lum.toFixed(2)}, blotch ${this.panoStats.blotch.toFixed(3)} -> shown at ${(this.panoStats.weight * 100).toFixed(0)} %`);
      }
    } catch (e) {
      console.warn("sky panorama failed, using procedural sky", e);
    }
  }

  get hasPanorama() {
    return this.material.uniforms.useMap.value > 0.5;
  }
  /** weight the panorama is shown with */
  get panoWeight(): number {
    return this.hasPanorama ? this.material.uniforms.panoBlend.value : 0;
  }
  /** day horizon (= haze colour): the procedural horizon, blended toward the graded panorama's by its weight */
  dayHorizon(grade: (c: THREE.Color) => THREE.Color): THREE.Color {
    const w = this.panoWeight;
    return w > 0 ? Sky.PROC_HORIZON.clone().lerp(grade(this.horizon), w) : Sky.PROC_HORIZON.clone();
  }
  setGrade(a: THREE.Vector4, b: THREE.Vector4) {
    (this.material.uniforms.gradeA.value as THREE.Vector4).copy(a);
    (this.material.uniforms.gradeB.value as THREE.Vector4).copy(b);
  }

  /** the photographed sky: textures and weights (from PhotoLayers.sky), its elevation range and amount */
  setPhoto(tex: (THREE.Texture | null)[], w: THREE.Vector4, amount: number, elev: [number, number], yaw: number) {
    const u = this.material.uniforms;
    for (let i = 0; i < 4; i++) u["photo" + i].value = tex[i];
    (u.photoW.value as THREE.Vector4).copy(w);
    u.photoAmt.value = amount;
    (u.photoElev.value as THREE.Vector2).set(elev[0], elev[1]);
    (u.photoYaw.value as THREE.Vector2).set(Math.cos(-yaw), Math.sin(-yaw));
  }

  setNight(v: number) {
    this.material.uniforms.night.value = v;
  }
  setFog(c: THREE.Color) {
    (this.material.uniforms.fogColor.value as THREE.Color).copy(c);
  }

  follow(camera: THREE.Camera) {
    this.mesh.position.copy(camera.position);
  }

}

async function loadMap(url: string) {
  const tex = await new THREE.TextureLoader().loadAsync(url);
  // splats are written in display space, so the sky is sampled and written raw as well
  tex.colorSpace = THREE.NoColorSpace;
  tex.wrapS = THREE.RepeatWrapping;
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  tex.generateMipmaps = true;
  return tex;
}

/** brightness and blotchiness of the upper sky (rows from the zenith down to just above the horizon) */
function measure(px: Uint8ClampedArray, w: number, h: number) {
  let sum = 0, dev = 0;
  for (let y = 0; y < h; y++) {
    let r = 0, g = 0, b = 0;
    for (let x = 0; x < w; x++) { const i = (y * w + x) * 4; r += px[i]; g += px[i + 1]; b += px[i + 2]; }
    r /= w * 255; g /= w * 255; b /= w * 255;
    sum += 0.2126 * r + 0.7152 * g + 0.0722 * b;
    for (let x = 0; x < w; x++) {
      const i = (y * w + x) * 4;
      dev += Math.hypot(px[i] / 255 - r, px[i + 1] / 255 - g, px[i + 2] / 255 - b);
    }
  }
  const lum = sum / h, blotch = dev / (w * h);
  const ss = (a: number, b: number, x: number) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };
  // a clean overcast or clear sky is bright and nearly uniform along each row; fused camera panoramas that are
  // dark or streaky are mostly replaced by the procedural sky
  const weight = ss(0.45, 0.68, lum) * (1 - ss(0.035, 0.09, blotch));
  return { lum, blotch, weight };
}
