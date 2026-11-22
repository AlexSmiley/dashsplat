// The default chase pose (Tesla's follow camera) as a pure function of drive time T and the view's aspect ratio.
// Nothing here depends on wall-clock time or frame history, so the live view at any playback rate, after any
// seek, and a video pre-rendered offline (Generative mode, tools/chase-track.ts) all agree frame for frame.

import * as THREE from "three";
import type { Drive, DriveState } from "../scene/drive";

const D2R = Math.PI / 180;

// measured on Tesla's Reality View frames (ref_15.5, ref_24, ref_31): horizon 28 % from the top, the car (as
// seen) 62 % down and centred, its body 7.2 % of the view's width at 12 m/s. Camera 16.9 deg above the car
// centre, pitched down 11.6 deg, about 5.5 m above the road and 15.5 m behind the car centre at rest
export const CHASE = { dist: 16.2, elev: 16.9 * D2R, tilt: 5.3 * D2R, pan: 0, pivotX: 1.45, pivotZ: 0.8, vfov: 50, minHfov: 64 };

/** what the chase pose follows at T: the car's position, a smoothed heading, height and speed */
export type ChaseInputs = { x: number; y: number; z: number; heading: number; speed: number };

const lerpAt = (drive: Drive, arr: ArrayLike<number>, t: number) => {
  const i = drive.index(t);
  const t0 = drive.t[i], t1 = drive.t[i + 1];
  const a = t1 > t0 ? Math.min(1, Math.max(0, (t - t0) / (t1 - t0))) : 0;
  return arr[i] + (arr[i + 1] - arr[i]) * a;
};

// heading: Gaussian (sigma 0.25 s) around T + 0.05 s, so pose noise never reaches the camera and it does not
// lag in turns; height: sigma 0.2 s; speed (for the pull-back): trailing exponential, 1.2 s
const H_LEAD = 0.05, H_SIGMA = 0.25, Z_SIGMA = 0.2, STEP = 0.05, TAPS = 12;
const W_H = Array.from({ length: 2 * TAPS + 1 }, (_, k) => Math.exp(-0.5 * (((k - TAPS) * STEP) / H_SIGMA) ** 2));
const W_Z = Array.from({ length: 2 * TAPS + 1 }, (_, k) => Math.exp(-0.5 * (((k - TAPS) * STEP) / Z_SIGMA) ** 2));

export function chaseInputs(drive: Drive, T: number, st?: DriveState): ChaseInputs {
  const s = st ?? drive.state(T);
  const end = drive.duration;
  let hw = 0, h = 0, zw = 0, z = 0;
  for (let k = 0; k <= 2 * TAPS; k++) {
    const dt = (k - TAPS) * STEP;
    h += W_H[k] * lerpAt(drive, drive.yaw, Math.min(end, Math.max(0, T + H_LEAD + dt)));
    hw += W_H[k];
    z += W_Z[k] * lerpAt(drive, drive.z, Math.min(end, Math.max(0, T + dt)));
    zw += W_Z[k];
  }
  let vw = 0, v = 0;
  for (let k = 0; k < 9; k++) {
    const w = Math.exp(-(k * 0.3) / 1.2);
    v += w * Math.max(0, lerpAt(drive, drive.speed, Math.max(0, T - k * 0.3)));
    vw += w;
  }
  return { x: s.x, y: s.y, z: z / zw, heading: h / hw, speed: v / vw };
}

/** vertical field of view of the chase camera for a view aspect (width / height) */
export function chaseFov(aspect: number) {
  // landscape: keep at least the reference's horizontal field of view; portrait: a taller view that keeps the
  // car readable instead of a fisheye-wide one
  const vfovMin = (2 * Math.atan(Math.tan((CHASE.minHfov * D2R) / 2) / aspect)) / D2R;
  return aspect >= 1 ? Math.max(CHASE.vfov, vfovMin) : Math.min(74, Math.max(CHASE.vfov, vfovMin));
}

/** place a camera at the chase pose (plus an orbit offset: azimuth and elevation in radians, zoom factor);
 *  returns the pivot it looks at */
export function placeChase(cam: THREE.PerspectiveCamera, p: ChaseInputs, aspect: number, az = 0, el = 0, zoom = 1) {
  cam.aspect = aspect;
  cam.fov = chaseFov(aspect);
  const c = Math.cos(p.heading), s = Math.sin(p.heading);
  const pivot = new THREE.Vector3(p.x + c * CHASE.pivotX, p.y + s * CHASE.pivotX, p.z + CHASE.pivotZ);
  const a = p.heading + Math.PI + az;
  const e = CHASE.elev + el;
  // a little pull-back with speed (+8 % at 20 m/s)
  const D = CHASE.dist * zoom * (aspect < 1 ? 0.78 : 1) * (1 + 0.004 * Math.min(25, p.speed));
  cam.up.set(0, 0, 1);
  cam.position.set(pivot.x + Math.cos(a) * Math.cos(e) * D, pivot.y + Math.sin(a) * Math.cos(e) * D, pivot.z + Math.sin(e) * D);
  cam.lookAt(pivot);
  // tilt up so the car sits in the lower part of the frame, as on the Tesla screen
  cam.rotateX(CHASE.tilt);
  cam.rotateY(-CHASE.pan);
  cam.near = 0.3;
  cam.updateProjectionMatrix();
  cam.updateMatrixWorld();
  return pivot;
}
