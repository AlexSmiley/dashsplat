// The default chase camera for every frame of a stretch of the drive, for rendering a Generative clip from
// exactly the pose the viewer uses (src/camera/chasePose.ts).
//   bun tools/chase-track.ts <scene dir> <t0> <t1> [fps=30] [width=1920] [height=1200] > track.json
// Each frame: t (drive time, s), pos (metres, the scene's frame, z up), quat (x, y, z, w: three.js camera
// convention, looking down its local -z with +y up), plus the vertical field of view for the image's aspect.

import { readFileSync } from "node:fs";
import path from "node:path";
import * as THREE from "three";
import { Drive, type DriveJson } from "../src/scene/drive";
import { chaseInputs, placeChase } from "../src/camera/chasePose";

const [dir, a0, a1, fpsArg = "30", wArg = "1920", hArg = "1200"] = process.argv.slice(2);
if (!dir || a0 === undefined || a1 === undefined) {
  console.error("usage: bun tools/chase-track.ts <scene dir> <t0> <t1> [fps] [width] [height]");
  process.exit(1);
}
const drive = new Drive(JSON.parse(readFileSync(path.join(dir, "drive.json"), "utf8")) as DriveJson);
const [t0, t1, fps, w, h] = [a0, a1, fpsArg, wArg, hArg].map(Number);
const cam = new THREE.PerspectiveCamera(50, w / h, 0.3, 5000);
const frames: { t: number; pos: number[]; quat: number[] }[] = [];
for (let i = 0; t0 + i / fps <= t1 + 1e-9; i++) {
  const t = t0 + i / fps;
  placeChase(cam, chaseInputs(drive, t), w / h);
  const r = (v: number, d: number) => +v.toFixed(d);
  frames.push({ t: r(t, 4), pos: cam.position.toArray().map((v) => r(v, 4)), quat: cam.quaternion.toArray().map((v) => r(v, 6)) });
}
console.log(JSON.stringify({ pose: "chase", fps, width: w, height: h, vfov: +cam.fov.toFixed(4), near: 0.3, frames }));
