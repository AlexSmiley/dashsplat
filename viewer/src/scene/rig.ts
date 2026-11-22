// Camera rig in the calib/rig.json format (PLAN.md). Used by the viewer's "from the cameras" mode and by
// the synthetic scene generator's visibility model. Real scenes ship their own rig.json; these defaults are
// plausible HW4 Model 3 placements measured by eye, not calibration.

export type RigCamera = {
  model: "OPENCV" | "OPENCV_FISHEYE" | "PINHOLE";
  width: number;
  height: number;
  params: number[];
  /** 4x4 row-major, camera (OpenCV: x right, y down, z forward) to ego (x forward, y left, z up) */
  T_ego_cam: number[][];
  time_offset_s?: number;
  rs_readout_s?: number | null;
  mask?: string;
};

export type RigJson = { cameras: Record<string, RigCamera> };

export const CAMERA_ORDER = ["front", "left_pillar", "right_pillar", "left_repeater", "right_repeater", "back"] as const;

export const CAMERA_LABELS: Record<string, string> = {
  front: "Front",
  back: "Rear",
  left_pillar: "Left pillar",
  right_pillar: "Right pillar",
  left_repeater: "Left repeater",
  right_repeater: "Right repeater",
};

const D2R = Math.PI / 180;

/** T_ego_cam for a camera at ego position p looking along yaw (CCW from forward) and pitch (up positive). */
export function poseFromYawPitch(p: [number, number, number], yawDeg: number, pitchDeg: number): number[][] {
  const y = yawDeg * D2R;
  const th = pitchDeg * D2R;
  const zc = [Math.cos(th) * Math.cos(y), Math.cos(th) * Math.sin(y), Math.sin(th)];
  const xc = [Math.sin(y), -Math.cos(y), 0];
  const yc = [Math.sin(th) * Math.cos(y), Math.sin(th) * Math.sin(y), -Math.cos(th)];
  return [
    [xc[0], yc[0], zc[0], p[0]],
    [xc[1], yc[1], zc[1], p[1]],
    [xc[2], yc[2], zc[2], p[2]],
    [0, 0, 0, 1],
  ];
}

function fisheye(w: number, h: number, hfovDeg: number, vfovDeg: number): number[] {
  // equidistant: half width in pixels = f * half angle
  return [w / 2 / ((hfovDeg * D2R) / 2), h / 2 / ((vfovDeg * D2R) / 2), w / 2, h / 2, 0, 0, 0, 0];
}

/** Fallback rig shaped like the real calibration (scenes/<name>/rig.json wins when present). */
export function defaultRig(): RigJson {
  const cam = (w: number, h: number, hf: number, vf: number, p: [number, number, number], yaw: number, pitch: number): RigCamera => ({
    model: "OPENCV_FISHEYE",
    width: w,
    height: h,
    params: fisheye(w, h, hf, vf),
    T_ego_cam: poseFromYawPitch(p, yaw, pitch),
  });
  return {
    cameras: {
      front: cam(2896, 1876, 46, 29, [1.72, 0.04, 1.3], 0, 0),
      back: cam(1448, 938, 174, 113, [-0.98, 0.05, 0.84], 179, -33),
      left_pillar: cam(1448, 938, 83, 52, [0.94, 0.7, 1.26], 70, -4),
      right_pillar: cam(1448, 938, 83, 52, [0.94, -0.7, 1.26], -70, -4),
      left_repeater: cam(1448, 938, 83, 52, [2.39, 0.9, 0.73], 142, -5),
      right_repeater: cam(1448, 938, 83, 52, [2.39, -0.9, 0.73], -142, -5),
    },
  };
}

export type CamGeom = {
  name: string;
  /** half angle covered by the image width / height (radians, equidistant model for fisheye) */
  thH: number;
  thV: number;
  /** wide lenses need the angular test; narrow ones use a tan pyramid */
  wide: boolean;
  tanH: number;
  tanV: number;
  /** position in ego frame */
  pos: [number, number, number];
  /** unit axes in ego frame (OpenCV camera axes) */
  right: [number, number, number];
  down: [number, number, number];
  fwd: [number, number, number];
  hfovDeg: number;
  vfovDeg: number;
  aspect: number;
};

export function cameraGeometry(rig: RigJson): CamGeom[] {
  const out: CamGeom[] = [];
  const names = [...CAMERA_ORDER.filter((n) => rig.cameras[n]), ...Object.keys(rig.cameras).filter((n) => !(CAMERA_ORDER as readonly string[]).includes(n))];
  for (const name of names) {
    const c = rig.cameras[name];
    const M = c.T_ego_cam;
    const fx = c.params[0];
    const fy = c.params[1] ?? fx;
    const fish = c.model === "OPENCV_FISHEYE";
    const thH = fish ? c.width / 2 / fx : Math.atan(c.width / 2 / fx);
    const thV = fish ? c.height / 2 / fy : Math.atan(c.height / 2 / fy);
    out.push({
      name,
      thH,
      thV,
      wide: thH > 1.15 || thV > 1.15,
      tanH: Math.tan(Math.min(thH, 1.5)),
      tanV: Math.tan(Math.min(thV, 1.5)),
      pos: [M[0][3], M[1][3], M[2][3]],
      right: [M[0][0], M[1][0], M[2][0]],
      down: [M[0][1], M[1][1], M[2][1]],
      fwd: [M[0][2], M[1][2], M[2][2]],
      hfovDeg: (2 * thH) / D2R,
      vfovDeg: (2 * thV) / D2R,
      aspect: c.width / c.height,
    });
  }
  return out;
}

/** Is a point at camera coordinates (xc right, yc down, zc forward) inside the image? */
export function inImage(g: CamGeom, xc: number, yc: number, zc: number): boolean {
  if (!g.wide) {
    if (zc <= 0.05) return false;
    return Math.abs(xc) <= g.tanH * zc && Math.abs(yc) <= g.tanV * zc;
  }
  const rho = Math.hypot(xc, yc);
  if (rho < 1e-6) return zc > 0;
  const th = Math.atan2(rho, zc);
  return (th * Math.abs(xc)) / rho <= g.thH && (th * Math.abs(yc)) / rho <= g.thV;
}
