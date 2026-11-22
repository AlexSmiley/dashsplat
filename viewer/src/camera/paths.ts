// Camera paths for captured shots (tools/capture.ts, ?campath=<json>): each one a pure function of drive time, so
// a shot is repeatable frame for frame. They follow the chase pose's own smoothed inputs (car position, a
// Gaussian-smoothed heading), so nothing judders with pose noise.
//
//   { "kind": "chase" }                                   the default chase camera
//   { "kind": "orbit", "radius": 9, "height": 3, "deg0": 180, "degPerSec": 12, "t0": 199, "fov": 42 }
//        round the car (angle relative to its heading, 180 = behind), looking at its centre
//   { "kind": "crane", "t0": 116, "up": 3, "hold": 2, "down": 3, "height": 70 }
//        from the chase pose up to a top-down view above the car (heading up) and back, eased
//   { "kind": "spline", "frame": "car", "keys": [{ "t": 300, "pos": [-8, 3, 2], "look": [1.4, 0, 0.8], "fov": 40 }, ...] }
//        a smooth path (Catmull-Rom) through keyframes, in the car's frame (x forward, y left, z up, metres from
//        the rear axle on the ground) or the world's; "reach": 320 streams and hazes round the look point instead
//        of the car (a view detached from it)
//   { "kind": "map", "height": 260, "fov": 40, "northUp": false }
//        locked top-down view following the car

import * as THREE from "three";
import { CHASE, placeChase, type ChaseInputs } from "./chasePose";

export type PathKey = { t: number; pos: number[]; look: number[]; fov?: number };
export type CamPath =
  | { kind: "chase" }
  | { kind: "orbit"; radius: number; height: number; deg0: number; degPerSec: number; t0: number; fov?: number; lookZ?: number }
  | { kind: "crane"; t0: number; up: number; hold: number; down: number; height?: number; fov?: number }
  | { kind: "spline"; frame?: "car" | "world"; keys: PathKey[]; reach?: number }
  | { kind: "map"; height: number; fov?: number; northUp?: boolean };

const D2R = Math.PI / 180;
const ease = (x: number) => {
  const t = THREE.MathUtils.clamp(x, 0, 1);
  // smootherstep: zero velocity and acceleration at both ends
  return t * t * t * (t * (t * 6 - 15) + 10);
};

/** 0..1 -> 0..1 with a trapezoid velocity: ramps up over the first r, down over the last r */
function ramp(x: number, r: number) {
  const u = THREE.MathUtils.clamp(x, 0, 1);
  const v = 1 / (1 - r);
  if (u < r) return (v * u * u) / (2 * r);
  if (u > 1 - r) return 1 - (v * (1 - u) * (1 - u)) / (2 * r);
  return v * (r / 2 + (u - r));
}

/** car frame -> world, around the chase inputs' smoothed heading */
function toWorld(p: ChaseInputs, v: number[]) {
  const c = Math.cos(p.heading), s = Math.sin(p.heading);
  return new THREE.Vector3(p.x + c * v[0] - s * v[1], p.y + s * v[0] + c * v[1], p.z + v[2]);
}

function catmull(p0: THREE.Vector3, p1: THREE.Vector3, p2: THREE.Vector3, p3: THREE.Vector3, t: number) {
  const t2 = t * t, t3 = t2 * t;
  return new THREE.Vector3(
    0.5 * (2 * p1.x + (-p0.x + p2.x) * t + (2 * p0.x - 5 * p1.x + 4 * p2.x - p3.x) * t2 + (-p0.x + 3 * p1.x - 3 * p2.x + p3.x) * t3),
    0.5 * (2 * p1.y + (-p0.y + p2.y) * t + (2 * p0.y - 5 * p1.y + 4 * p2.y - p3.y) * t2 + (-p0.y + 3 * p1.y - 3 * p2.y + p3.y) * t3),
    0.5 * (2 * p1.z + (-p0.z + p2.z) * t + (2 * p0.z - 5 * p1.z + 4 * p2.z - p3.z) * t2 + (-p0.z + 3 * p1.z - 3 * p2.z + p3.z) * t3),
  );
}

/** the top-down pose above the car, heading up (or north up) */
function topDown(cam: THREE.PerspectiveCamera, p: ChaseInputs, height: number, northUp = false) {
  const c = Math.cos(p.heading), s = Math.sin(p.heading);
  cam.position.set(p.x + c * CHASE.pivotX, p.y + s * CHASE.pivotX, p.z + height);
  cam.up.set(northUp ? 0 : c, northUp ? 1 : s, 0);
  cam.lookAt(cam.position.x, cam.position.y, p.z);
  cam.up.set(0, 0, 1);
}

/** place the camera on the path at T; returns where streaming should centre and how far it should reach */
export function applyPath(path: CamPath, cam: THREE.PerspectiveCamera, T: number, p: ChaseInputs, aspect: number): { x: number; y: number; radius: number; mapFog: boolean } | null {
  cam.aspect = aspect;
  const near = { x: p.x, y: p.y, radius: 200, mapFog: false };
  switch (path.kind) {
    case "chase":
      return null;
    case "orbit": {
      const a = p.heading + (path.deg0 + path.degPerSec * (T - path.t0)) * D2R;
      const c = Math.cos(p.heading), s = Math.sin(p.heading);
      const pivot = new THREE.Vector3(p.x + c * CHASE.pivotX, p.y + s * CHASE.pivotX, p.z + (path.lookZ ?? 0.8));
      cam.position.set(pivot.x + Math.cos(a) * path.radius, pivot.y + Math.sin(a) * path.radius, p.z + path.height);
      cam.up.set(0, 0, 1);
      cam.lookAt(pivot);
      cam.fov = path.fov ?? 42;
      return { ...near, radius: Math.max(200, path.radius * 6) };
    }
    case "crane": {
      // 0 at the chase pose, 1 straight down from height above the car
      const t = T - path.t0;
      const u = t < path.up ? ease(t / path.up) : t < path.up + path.hold ? 1 : ease(1 - (t - path.up - path.hold) / path.down);
      placeChase(cam, p, aspect);
      if (u <= 0) return null;
      const q0 = cam.quaternion.clone(), p0 = cam.position.clone(), f0 = cam.fov;
      const H = path.height ?? 70;
      // a crane arc round the car: elevation from the chase angle to vertical, distance from the chase distance to H
      const c = Math.cos(p.heading), s = Math.sin(p.heading);
      const pivot = new THREE.Vector3(p.x + c * CHASE.pivotX, p.y + s * CHASE.pivotX, p.z + CHASE.pivotZ * (1 - u));
      const d0 = p0.distanceTo(pivot);
      const e = THREE.MathUtils.lerp(Math.asin(THREE.MathUtils.clamp((p0.z - pivot.z) / d0, -1, 1)), 89.6 * D2R, u);
      const D = THREE.MathUtils.lerp(d0, H, u * u * (3 - 2 * u));
      const back = p.heading + Math.PI;
      cam.position.set(pivot.x + Math.cos(back) * Math.cos(e) * D, pivot.y + Math.sin(back) * Math.cos(e) * D, pivot.z + Math.sin(e) * D);
      // orientation: from the chase view to looking down at the car, heading up
      cam.up.set(c, s, 0);
      cam.lookAt(pivot);
      const q1 = cam.quaternion.clone();
      cam.up.set(0, 0, 1);
      cam.quaternion.copy(q0).slerp(q1, u);
      cam.fov = THREE.MathUtils.lerp(f0, path.fov ?? 45, u);
      return { ...near, radius: Math.max(220, D * 2.2), mapFog: u > 0.6 };
    }
    case "spline": {
      const k = path.keys;
      if (!k.length) return null;
      const P = (key: PathKey) => (path.frame === "world" ? new THREE.Vector3(...(key.pos as [number, number, number])) : toWorld(p, key.pos));
      const Lk = (key: PathKey) => (path.frame === "world" ? new THREE.Vector3(...(key.look as [number, number, number])) : toWorld(p, key.look));
      // path time: starts and stops softly (velocity ramps over the first and last 15 %), constant in between
      const tA = k[0].t, tB = k[k.length - 1].t;
      const tau = tB > tA ? tA + (tB - tA) * ramp((T - tA) / (tB - tA), 0.15) : tA;
      let i = 0;
      while (i < k.length - 2 && tau >= k[i + 1].t) i++;
      const a = k[Math.max(0, i - 1)], b = k[i], c2 = k[Math.min(k.length - 1, i + 1)], d = k[Math.min(k.length - 1, i + 2)];
      const u = c2.t > b.t ? THREE.MathUtils.clamp((tau - b.t) / (c2.t - b.t), 0, 1) : 0;
      const pos = catmull(P(a), P(b), P(c2), P(d), u);
      const look = catmull(Lk(a), Lk(b), Lk(c2), Lk(d), u);
      cam.position.copy(pos);
      cam.up.set(0, 0, 1);
      cam.lookAt(look);
      cam.fov = THREE.MathUtils.lerp(b.fov ?? 42, c2.fov ?? b.fov ?? 42, u);
      // reach: stream and haze round the look point out to this many metres (a view detached from the car)
      return path.reach ? { x: look.x, y: look.y, radius: path.reach, mapFog: true } : { x: look.x, y: look.y, radius: 220, mapFog: false };
    }
    case "map": {
      topDown(cam, p, path.height, path.northUp);
      cam.fov = path.fov ?? 40;
      const view = path.height * Math.tan(((cam.fov * D2R) / 2)) * Math.max(1, aspect);
      return { x: p.x, y: p.y, radius: THREE.MathUtils.clamp(view * 1.6, 160, 900), mapFog: true };
    }
  }
}

export function parsePath(s: string | null): CamPath | null {
  if (!s) return null;
  try {
    const p = JSON.parse(s) as CamPath;
    return p && typeof p === "object" && "kind" in p ? p : null;
  } catch {
    return null;
  }
}
