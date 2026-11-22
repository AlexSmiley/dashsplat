// Camera modes:
//   chase  Tesla-style follow camera above and behind the car; drag orbits, pinch / wheel zooms, eases back
//   map    top-down, heading-up map view; drag pans, pinch / wheel zooms, recentres after a pause
//   cams   sits at one of the rig cameras looking along it (cycle with the 3 key or the camera button)
//   free   debug fly camera: drag to look, WASD / QE to move, shift for speed

import * as THREE from "three";
import type { DriveState } from "../scene/drive";
import type { CamGeom } from "../scene/rig";
import { CHASE, placeChase, type ChaseInputs } from "./chasePose";

export { CHASE };

export type CamMode = "chase" | "map" | "cams" | "free";

const D2R = Math.PI / 180;
const damp = (dt: number, tau: number) => 1 - Math.exp(-dt / tau);


export class CameraController {
  mode: CamMode = "chase";
  camIndex = 0;
  readonly camera: THREE.PerspectiveCamera;
  // chase orbit offsets (eased back to zero after interaction)
  private az = 0;
  private el = 0;
  private zoom = 1;
  // map
  private mapH = 220;
  private pan = new THREE.Vector2();
  // free fly
  private freePos = new THREE.Vector3();
  private freeYaw = 0;
  private freePitch = -0.2;
  private keys = new Set<string>();
  private lastInteract = -1e9;
  private dragging = false;
  private pointers = new Map<number, { x: number; y: number }>();
  private pinchDist = 0;
  onModeChange?: (m: CamMode) => void;
  /** set while a drag moved more than a few pixels (suppresses click-through) */
  moved = false;

  constructor(private readonly el0: HTMLElement) {
    this.camera = new THREE.PerspectiveCamera(CHASE.vfov, 1, 0.3, 5000);
    this.camera.up.set(0, 0, 1);
    el0.addEventListener("pointerdown", this.onDown);
    window.addEventListener("pointermove", this.onMove);
    window.addEventListener("pointerup", this.onUp);
    window.addEventListener("pointercancel", this.onUp);
    el0.addEventListener("wheel", this.onWheel, { passive: false });
    window.addEventListener("keydown", (e) => {
      if ((e.target as HTMLElement)?.tagName === "INPUT" || (e.target as HTMLElement)?.tagName === "SELECT") return;
      this.keys.add(e.key.toLowerCase());
    });
    window.addEventListener("keyup", (e) => this.keys.delete(e.key.toLowerCase()));
    window.addEventListener("blur", () => this.keys.clear());
  }

  setMode(m: CamMode) {
    if (m === "free" && this.mode !== "free") {
      this.freePos.copy(this.camera.position);
      const d = new THREE.Vector3();
      this.camera.getWorldDirection(d);
      this.freeYaw = Math.atan2(d.y, d.x);
      this.freePitch = Math.asin(THREE.MathUtils.clamp(d.z, -1, 1));
    }
    if (m === "map" && this.mode !== "map") {
      this.pan.set(0, 0);
      this.mapH = 220;
    }
    this.mode = m;
    this.az = this.el = 0;
    this.zoom = 1;
    this.onModeChange?.(m);
  }

  /** kept for callers that jump in time: the chase pose is a function of T, so there is nothing to reset */
  snap() {}

  /** chase orbit offsets: azimuth and elevation (radians) and zoom factor */
  get orbit() {
    return { az: this.az, el: this.el, zoom: this.zoom };
  }

  /** at the default chase pose (no orbit or zoom offset, or nearly back from one) */
  get atDefaultChase() {
    return this.mode === "chase" && !this.dragging && Math.abs(this.az) < 0.02 && Math.abs(this.el) < 0.02 && Math.abs(this.zoom - 1) < 0.02;
  }

  /** fixed orbit offset for screenshots and checks (degrees, zoom factor); it does not ease back */
  private hold = false;
  setOrbit(azDeg: number, elDeg: number, zoom: number) {
    this.az = azDeg * D2R;
    this.el = elDeg * D2R;
    this.zoom = zoom;
    this.hold = true;
  }

  private onDown = (e: PointerEvent) => {
    if (e.button !== 0 && e.pointerType === "mouse") return;
    this.hold = false;
    this.pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    this.el0.setPointerCapture?.(e.pointerId);
    this.dragging = true;
    this.moved = false;
    this.lastInteract = performance.now();
    if (this.pointers.size === 2) {
      const [a, b] = [...this.pointers.values()];
      this.pinchDist = Math.hypot(a.x - b.x, a.y - b.y);
    }
  };

  private onMove = (e: PointerEvent) => {
    const p = this.pointers.get(e.pointerId);
    if (!p) return;
    const dx = e.clientX - p.x, dy = e.clientY - p.y;
    p.x = e.clientX;
    p.y = e.clientY;
    if (Math.abs(dx) + Math.abs(dy) > 2) this.moved = true;
    this.lastInteract = performance.now();
    if (this.pointers.size >= 2) {
      const [a, b] = [...this.pointers.values()];
      const d = Math.hypot(a.x - b.x, a.y - b.y);
      if (this.pinchDist > 0) this.applyZoom(this.pinchDist / d);
      this.pinchDist = d;
      return;
    }
    const k = 1 / Math.max(320, Math.min(window.innerWidth, window.innerHeight));
    if (this.mode === "chase") {
      this.az -= dx * k * 3.2;
      this.el = THREE.MathUtils.clamp(this.el + dy * k * 2.2, -CHASE.elev + 3 * D2R, 72 * D2R - CHASE.elev);
    } else if (this.mode === "map") {
      const s = (this.mapH * 1.6) / Math.max(320, window.innerHeight);
      this.pan.x -= dx * s;
      this.pan.y += dy * s;
    } else if (this.mode === "free") {
      this.freeYaw -= dx * k * 3;
      this.freePitch = THREE.MathUtils.clamp(this.freePitch - dy * k * 3, -1.5, 1.5);
    }
  };

  private onUp = (e: PointerEvent) => {
    this.pointers.delete(e.pointerId);
    if (this.pointers.size < 2) this.pinchDist = 0;
    if (this.pointers.size === 0) this.dragging = false;
    this.lastInteract = performance.now();
  };

  private onWheel = (e: WheelEvent) => {
    e.preventDefault();
    this.lastInteract = performance.now();
    const f = Math.exp(e.deltaY * (e.deltaMode === 1 ? 0.05 : 0.0015));
    this.applyZoom(f);
  };

  private applyZoom(f: number) {
    if (this.mode === "chase") this.zoom = THREE.MathUtils.clamp(this.zoom * f, 0.42, 2.8);
    else if (this.mode === "map") this.mapH = THREE.MathUtils.clamp(this.mapH * f, 35, 1600);
    else if (this.mode === "free") this.freePos.addScaledVector(this.camera.getWorldDirection(new THREE.Vector3()), (1 - f) * 25);
  }

  /** ground point the streaming should centre on, and how far it should reach */
  focus(st: DriveState): { x: number; y: number; radius: number; mapFog: boolean } {
    if (this.mode === "map") {
      return { x: this.camTarget.x, y: this.camTarget.y, radius: THREE.MathUtils.clamp(this.mapH * 1.5, 160, 650), mapFog: true };
    }
    if (this.mode === "free") {
      const d = this.camera.getWorldDirection(new THREE.Vector3());
      return { x: this.camera.position.x + d.x * 50, y: this.camera.position.y + d.y * 50, radius: 210, mapFog: false };
    }
    // load what is inside the haze (it ends 140 m from the car, further in optimised scenes), biased ahead
    const c = Math.cos(st.yaw), s = Math.sin(st.yaw);
    return { x: st.x + c * this.reach.ahead, y: st.y + s * this.reach.ahead, radius: this.reach.radius, mapFog: false };
  }

  private camTarget = new THREE.Vector3();
  /** chase streaming reach: radius around a point this far ahead of the car (set per scene with the haze) */
  reach = { radius: 172, ahead: 25 };

  update(dt: number, st: DriveState, cams: CamGeom[], aspect: number, chase: ChaseInputs) {
    const now = performance.now();
    const cam = this.camera;
    cam.aspect = aspect;
    const idle = !this.dragging && !this.hold && now - this.lastInteract > 1400;

    if (this.mode === "chase") {
      if (idle) {
        const k = damp(dt, 0.5);
        this.az -= this.az * k;
        this.el -= this.el * k;
        this.zoom += (1 - this.zoom) * k;
      }
      this.camTarget.copy(placeChase(cam, chase, aspect, this.az, this.el, this.zoom));
    } else if (this.mode === "map") {
      if (idle && now - this.lastInteract > 5000) this.pan.multiplyScalar(1 - damp(dt, 0.6));
      cam.fov = Math.max(45, 2 * Math.atan(Math.tan(32 * D2R) / aspect) / D2R);
      const head = chase.heading;
      const c = Math.cos(head), s = Math.sin(head);
      // pan is in the heading-up screen frame: x right, y forward
      const tx = chase.x + c * this.pan.y + s * this.pan.x;
      const ty = chase.y + s * this.pan.y - c * this.pan.x;
      const pitch = THREE.MathUtils.lerp(58, 86, THREE.MathUtils.smoothstep(this.mapH, 60, 900)) * D2R;
      const R = this.mapH / Math.sin(pitch);
      cam.position.set(tx - c * Math.cos(pitch) * R, ty - s * Math.cos(pitch) * R, chase.z + this.mapH);
      this.camTarget.set(tx, ty, chase.z);
      cam.up.set(c, s, 0);
      cam.lookAt(this.camTarget);
      cam.up.set(0, 0, 1);
    } else if (this.mode === "cams") {
      const g = cams[this.camIndex % Math.max(1, cams.length)];
      if (g) {
        const c = Math.cos(st.yaw), s = Math.sin(st.yaw);
        const toW = (v: number[]) => new THREE.Vector3(v[0] * c - v[1] * s, v[0] * s + v[1] * c, v[2]);
        const p = toW(g.pos);
        cam.position.set(st.x + p.x, st.y + p.y, st.z + p.z);
        const f = toW(g.fwd), d = toW(g.down);
        cam.up.copy(d).negate();
        cam.lookAt(cam.position.clone().add(f));
        cam.up.set(0, 0, 1);
        // fit the whole image: horizontal and vertical extent, capped for the fisheye rear camera
        const hf = Math.min(g.hfovDeg, 128), vf = Math.min(g.vfovDeg, 100);
        cam.fov = Math.min(118, Math.max(vf, 2 * Math.atan(Math.tan((hf * D2R) / 2) / aspect) / D2R));
        this.camTarget.copy(cam.position);
      }
    } else {
      const speed = (this.keys.has("shift") ? 40 : 12) * dt;
      const f = new THREE.Vector3(Math.cos(this.freeYaw) * Math.cos(this.freePitch), Math.sin(this.freeYaw) * Math.cos(this.freePitch), Math.sin(this.freePitch));
      const r = new THREE.Vector3(Math.sin(this.freeYaw), -Math.cos(this.freeYaw), 0);
      if (this.keys.has("w")) this.freePos.addScaledVector(f, speed);
      if (this.keys.has("s")) this.freePos.addScaledVector(f, -speed);
      if (this.keys.has("d")) this.freePos.addScaledVector(r, speed);
      if (this.keys.has("a")) this.freePos.addScaledVector(r, -speed);
      if (this.keys.has("e")) this.freePos.z += speed;
      if (this.keys.has("q")) this.freePos.z -= speed;
      cam.fov = Math.max(60, 2 * Math.atan(Math.tan(40 * D2R) / aspect) / D2R);
      cam.position.copy(this.freePos);
      cam.lookAt(this.freePos.clone().add(f));
      this.camTarget.copy(this.freePos);
    }
    cam.near = this.mode === "map" ? Math.max(1, this.mapH * 0.05) : this.mode === "cams" ? 0.05 : 0.3;
    cam.updateProjectionMatrix();
    cam.updateMatrixWorld();
  }

  get interacting() {
    return this.dragging;
  }
}
