// The ego car: a Model 3 Highland glTF (see README for source and licence), pre-baked by tools/prepare-car.ts
// into the ego frame (x forward, y left, z up, metres, origin at the rear-axle centre on the ground). Lit only by
// its environment (render/environment.ts: the scene's sky, the captured surroundings, the road), with physically
// based clear-coated paint, tinted glass, satin trim, rubber and alloy; the body darkens towards the road (no
// baked occlusion in the model). Under it the overcast-sky contact shadow (render/contactShadow.ts). Wheels steer
// and spin from telemetry, brake lights follow `brake` or a deceleration over 1.3 m/s^2 (the telemetry's brake
// signal is missing after the car park), running lights are on at dawn, indicators follow `blink_l` / `blink_r`.

import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { MeshoptDecoder } from "three/addons/libs/meshopt_decoder.module.js";
import type { DriveState } from "../scene/drive";
import { AO_HALF_X, AO_HALF_Y, contactShadowTexture, EGO } from "./contactShadow";

const STEER_RATIO = 10.3; // steering-wheel degrees per road-wheel degree
const TYRE_R = 0.345;

// base colour (sRGB), metalness (metallic flake: the base reflects the environment tinted), roughness of the
// base layer under a glossy clear coat
export const PAINTS: { id: string; name: string; color: string; metal: number; rough: number }[] = [
  { id: "quicksilver", name: "Quicksilver", color: "#c8ccd0", metal: 0.9, rough: 0.22 },
  { id: "white", name: "Pearl White", color: "#e6e6e2", metal: 0.08, rough: 0.4 },
  { id: "grey", name: "Stealth Grey", color: "#55595e", metal: 0.65, rough: 0.38 },
  { id: "black", name: "Diamond Black", color: "#141518", metal: 0.5, rough: 0.35 },
  { id: "blue", name: "Deep Blue", color: "#24427a", metal: 0.72, rough: 0.34 },
  { id: "red", name: "Ultra Red", color: "#a3121c", metal: 0.55, rough: 0.34 },
];

/** occlusion near the road for every car material (the model has no baked ambient occlusion): the sills, bumpers,
 *  wheel arches and underbody see the dark road and little sky. carGround = the road height under the car */
const carGround = { value: 0 };
const patched = new WeakSet<THREE.Material>();
function addGroundOcclusion(m: THREE.Material) {
  if (patched.has(m)) return;
  patched.add(m);
  const prev = m.onBeforeCompile;
  m.onBeforeCompile = (sh, r) => {
    prev?.call(m, sh, r);
    sh.uniforms.carGround = carGround;
    sh.vertexShader = sh.vertexShader
      .replace("#include <common>", "#include <common>\nuniform float carGround;\nvarying float vCarH;")
      .replace("#include <project_vertex>", "#include <project_vertex>\n  vCarH = (modelMatrix * vec4(transformed, 1.0)).z - carGround;");
    sh.fragmentShader = sh.fragmentShader
      .replace("#include <common>", "#include <common>\nvarying float vCarH;")
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
  #if defined( USE_SHEEN )
    sheenSpecularIndirect *= ambientOcclusion;
  #endif
  #if defined( USE_ENVMAP ) && defined( STANDARD )
    float dotNV = saturate( dot( geometryNormal, geometryViewDir ) );
    reflectedLight.indirectSpecular *= computeSpecularOcclusion( dotNV, ambientOcclusion, material.roughness );
  #endif
}`,
      );
  };
  m.customProgramCacheKey = () => "car-ao";
  m.needsUpdate = true;
}

function glowTexture() {
  const c = document.createElement("canvas");
  c.width = c.height = 64;
  const g = c.getContext("2d")!;
  const r = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  r.addColorStop(0, "rgba(255,255,255,1)");
  r.addColorStop(0.18, "rgba(255,255,255,0.55)");
  r.addColorStop(0.5, "rgba(255,255,255,0.12)");
  r.addColorStop(1, "rgba(255,255,255,0)");
  g.fillStyle = r;
  g.fillRect(0, 0, 64, 64);
  return new THREE.CanvasTexture(c);
}

/** the red light of the tail lamps on the road behind the car (additive): brightest at the bumper (u = 1),
 *  fading to nothing at every edge */
function spillTexture() {
  const W = 128, H = 96;
  const cv = document.createElement("canvas");
  cv.width = W;
  cv.height = H;
  const g = cv.getContext("2d")!;
  const img = g.createImageData(W, H);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const u = (x + 0.5) / W, v = (y + 0.5) / H;
    const d = Math.min(1, Math.hypot((1 - u) / 1.0, (v - 0.5) / 0.5));
    const a = Math.pow(1 - d, 2.2) * Math.min(1, (1 - u) * 12 + 0.25);
    const o = (y * W + x) * 4;
    img.data[o] = img.data[o + 1] = img.data[o + 2] = 255;
    img.data[o + 3] = Math.round(255 * a);
  }
  g.putImageData(img, 0, 0);
  return new THREE.CanvasTexture(cv);
}

export class Car {
  readonly root = new THREE.Group();
  private wheels: { node: THREE.Object3D; front: boolean }[] = [];
  private paint = new THREE.MeshPhysicalMaterial({ name: "paint", clearcoat: 1, clearcoatRoughness: 0.04, envMapIntensity: 1 });
  // tinted glass: nearly black, with the reflection of a 1.5 index surface (strong at grazing angles)
  // (the coated roof and windows reflect more than bare glass: index 2.0 for the reflection, about 11 % head-on)
  private glass = new THREE.MeshPhysicalMaterial({ name: "glass", color: 0x040506, metalness: 0, roughness: 0.02, ior: 2.0, transparent: true, opacity: 0.92, envMapIntensity: 1.5, specularIntensity: 1, depthWrite: true });
  private spill: THREE.Mesh;
  private spin = 0;
  private lastS = NaN;
  private lastSpeed = NaN;
  private decel = 0;
  /** 0..1 eased brake lamp */
  brake = 0;
  private glows: { sprite: THREE.Sprite; kind: "tail" | "head" }[] = [];
  private tail: THREE.MeshStandardMaterial[] = [];
  private head: THREE.MeshStandardMaterial[] = [];
  private indL: THREE.MeshStandardMaterial[] = [];
  private indR: THREE.MeshStandardMaterial[] = [];
  readonly shadow: THREE.Mesh;
  /** the loaded model (cloned for clean vehicles) */
  model: THREE.Object3D | null = null;
  ready: Promise<void>;
  steerSmoothed = 0;
  paintId = PAINTS[0].id;

  constructor(url: string) {
    this.root.name = "ego-car";
    // contact shadow: the overcast sky's occlusion on the road (PlaneGeometry lies in x-y, the ego ground plane)
    const shadow = new THREE.Mesh(
      new THREE.PlaneGeometry(2 * AO_HALF_X, 2 * AO_HALF_Y),
      new THREE.MeshBasicMaterial({ map: contactShadowTexture(), transparent: true, depthWrite: false, color: 0x000000, toneMapped: false, polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -4 }),
    );
    shadow.position.set(EGO.centreX, 0, 0.03);
    shadow.renderOrder = 20;
    this.shadow = shadow;
    this.root.add(shadow);
    // the tail lamps' red light on the road behind
    this.spill = new THREE.Mesh(
      new THREE.PlaneGeometry(3.2, 2.6),
      new THREE.MeshBasicMaterial({ map: spillTexture(), color: 0xff2410, transparent: true, opacity: 0, depthWrite: false, blending: THREE.AdditiveBlending, toneMapped: false, polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -4 }),
    );
    this.spill.position.set(-0.99 - 1.6, 0, 0.032);
    this.spill.renderOrder = 23;
    this.root.add(this.spill);
    for (const m of [this.paint, this.glass]) addGroundOcclusion(m);
    // night glows on the lamps (a soft additive halo, as the lamps bloom in the reference night frame)
    const glowTex = glowTexture();
    for (const [x, y, z, kind] of [[-0.99, 0.62, 0.94, "tail"], [-0.99, -0.62, 0.94, "tail"], [3.72, 0.64, 0.72, "head"], [3.72, -0.64, 0.72, "head"]] as const) {
      const mat = new THREE.SpriteMaterial({ map: glowTex, color: kind === "tail" ? 0xff2a14 : 0xfff1dc, transparent: true, opacity: 0, depthWrite: false, blending: THREE.AdditiveBlending, toneMapped: false });
      const sp = new THREE.Sprite(mat);
      sp.position.set(x, y, z);
      sp.scale.setScalar(kind === "tail" ? 0.9 : 1.3);
      sp.renderOrder = 25;
      this.root.add(sp);
      this.glows.push({ sprite: sp, kind });
    }
    this.setPaint(PAINTS[0].id);
    this.ready = this.load(url);
  }

  setPaint(id: string) {
    const p = PAINTS.find((x) => x.id === id) ?? PAINTS[0];
    this.paintId = p.id;
    this.paint.color.set(p.color);
    this.paint.metalness = p.metal;
    this.paint.roughness = p.rough;
    // light paints carry a fine metallic flake; dark ones read glossier
    this.paint.sheen = 0;
  }

  private async load(url: string) {
    const loader = new GLTFLoader();
    loader.setMeshoptDecoder(MeshoptDecoder);
    const gltf = await loader.loadAsync(url);
    const model = gltf.scene;
    this.model = model;
    this.root.add(model);
    model.traverse((o) => {
      if (o.name.startsWith("wheel_")) this.wheels.push({ node: o, front: o.name.includes("_f") });
      const m = o as THREE.Mesh;
      if (!m.isMesh) return;
      m.frustumCulled = false;
      const mats = Array.isArray(m.material) ? m.material : [m.material];
      mats.forEach((mat0, i) => {
        const mat = mat0 as THREE.MeshStandardMaterial;
        if (mat.name === "Paint_Color") {
          if (Array.isArray(m.material)) m.material[i] = this.paint;
          else m.material = this.paint;
          return;
        }
        if (!mat.isMeshStandardMaterial) return;
        // a dark interior behind the tinted glass
        if (/int|cockpit/i.test(mat.name)) {
          mat.color.setRGB(0.02, 0.022, 0.025);
          mat.map = null;
          mat.emissiveMap = null;
          mat.emissive?.setRGB(0, 0, 0);
          mat.roughness = 0.6;
          mat.needsUpdate = true;
        }
        // tinted glass (windows and the glass roof): dark, glossy, reflecting the sky
        if (mat.transparent) {
          if (Array.isArray(m.material)) m.material[i] = this.glass;
          else m.material = this.glass;
          m.renderOrder = 2;
          return;
        }
        const n = mat.name;
        addGroundOcclusion(mat);
        if (n === "Tire1Mtl") {
          // rubber: the sidewall texture kept, darkened
          mat.color.multiplyScalar(0.42);
          mat.roughness = 0.86;
          mat.metalness = 0;
        } else if (n === "Georimblurlfsub021Mtl") {
          // machined alloy faces of the wheels
          mat.color.setRGB(0.5, 0.52, 0.55);
          mat.metalness = 1;
          mat.roughness = 0.27;
        } else if (n === "Georimblurlfsub01Mtl") {
          // black trim, window surrounds, the wheels' dark faces: satin
          mat.color.setRGB(0.022, 0.024, 0.027);
          mat.metalness = 0;
          mat.roughness = 0.36;
        } else if (n === "Geohoodsub00031Mtl") {
          mat.metalness = 1;
          mat.roughness = 0.14;
        } else if (n === "Geodoorl2sub11Mtl" || n === "Geohoodsub00021Mtl") {
          // gloss black pillars and the lower grille
          mat.color.setRGB(0.012, 0.013, 0.015);
          mat.metalness = 0;
          mat.roughness = 0.18;
        }
        const lamp = (arr: THREE.MeshStandardMaterial[], color: number) => {
          mat.emissive = new THREE.Color(color);
          mat.emissiveIntensity = 0;
          mat.toneMapped = false;
          arr.push(mat);
        };
        if (mat.name === "Tail_Lamp" || mat.name === "Head_Lamp") {
          // lamp lenses: glossy
          mat.roughness = 0.1;
          mat.metalness = 0.2;
        }
        if (mat.name === "Tail_Lamp") lamp(this.tail, 0xff1a0d);
        else if (mat.name === "Head_Lamp") lamp(this.head, 0xfff3e0);
        else if (mat.name === "Indicator_L") lamp(this.indL, 0xff8a10);
        else if (mat.name === "Indicator_R") lamp(this.indR, 0xff8a10);
      });
    });
  }

  update(st: DriveState, night: number, dt: number) {
    this.root.position.set(st.x, st.y, st.z);
    this.root.rotation.set(st.roll, -st.pitch, st.yaw, "ZYX");
    carGround.value = st.z;
    // positive steering-wheel angle turns right on this car
    const target = THREE.MathUtils.degToRad(-st.steer / STEER_RATIO);
    this.steerSmoothed += (target - this.steerSmoothed) * (1 - Math.exp(-dt / 0.06));
    // the wheels roll with the distance driven, at most 24 degrees a frame: faster, a five-spoke wheel strobes
    // (turns backwards or stands still at 60 fps), so it is shown turning steadily forward instead
    const ds = Number.isNaN(this.lastS) ? 0 : st.s - this.lastS;
    this.lastS = st.s;
    if (Math.abs(ds) > 3) this.spin = st.s / TYRE_R;
    else this.spin += THREE.MathUtils.clamp(ds / TYRE_R, -0.42, 0.42);
    for (const w of this.wheels) w.node.rotation.set(0, this.spin, w.front ? this.steerSmoothed : 0, "ZYX");
    // braking: the telemetry's flag, a deceleration over 1.3 m/s^2, or holding at a standstill
    if (dt > 0 && !Number.isNaN(this.lastSpeed) && Math.abs(st.speed - this.lastSpeed) < 2) {
      const a = (st.speed - this.lastSpeed) / dt;
      this.decel += (-a - this.decel) * (1 - Math.exp(-dt / 0.3));
    } else if (dt > 0) this.decel = 0;
    if (dt > 0) this.lastSpeed = st.speed;
    const braking = st.brake || this.decel > 1.3 || st.speed < 0.3 ? 1 : 0;
    this.brake += (braking - this.brake) * (1 - Math.exp(-dt / 0.07));
    const brake = this.brake;
    // running lights on at dawn (the footage's cars have theirs on), brighter by night, the brake on top
    for (const m of this.tail) m.emissiveIntensity = Math.max(0.45, night * 1.1) + brake * 2.6;
    for (const g of this.glows) {
      const target = g.kind === "tail" ? night * (0.18 + 0.5 * brake) + (1 - night) * brake * 0.14 : night * 0.55;
      (g.sprite.material as THREE.SpriteMaterial).opacity += (target - (g.sprite.material as THREE.SpriteMaterial).opacity) * (1 - Math.exp(-dt / 0.06));
      g.sprite.visible = (g.sprite.material as THREE.SpriteMaterial).opacity > 0.005;
    }
    const sm = this.spill.material as THREE.MeshBasicMaterial;
    sm.opacity = 0.04 + 0.1 * night + brake * (0.2 + 0.25 * night);
    for (const m of this.head) m.emissiveIntensity = 0.5 + night * 2.0;
    const phase = st.t % 0.7 < 0.36 ? 1 : 0;
    for (const m of this.indL) m.emissiveIntensity = st.blinkL ? phase * 3 : 0;
    for (const m of this.indR) m.emissiveIntensity = st.blinkR ? phase * 3 : 0;
  }

  set visible(v: boolean) {
    this.root.visible = v;
  }
}
