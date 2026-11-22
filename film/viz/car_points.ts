#!/usr/bin/env bun
// Samples the viewer's car model (viewer/public/models/model3.glb, ego frame, metres) into surface points for the
// film's point-cloud shots. Output: little-endian float32 records x y z nx ny nz r g b (colour 0..1 from the
// material's base colour factor). "2024 Tesla Model 3" by RBLXSupercars (the credit embedded in the file; the project's notes also recorded "Tesla model 3 2024" by brandonleong28), CC BY 4.0 (see viewer/tools/prepare-car.ts).
// usage: bun viz/car_points.ts <out.bin> [points per m2]
import { NodeIO } from "@gltf-transform/core";
import { ALL_EXTENSIONS } from "@gltf-transform/extensions";
import { MeshoptDecoder } from "meshoptimizer";
import path from "node:path";

const out = process.argv[2];
const density = Number(process.argv[3] ?? 40000);
await MeshoptDecoder.ready;
const io = new NodeIO().registerExtensions(ALL_EXTENSIONS).registerDependencies({ "meshopt.decoder": MeshoptDecoder });
const doc = await io.read(path.resolve(import.meta.dir, "../../viewer/public/models/model3.glb"));
const recs: number[] = [];
let rnd = 12345;
const rand = () => ((rnd = (rnd * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);
doc.getRoot().getDefaultScene()!.traverse((node) => {
  const mesh = node.getMesh();
  if (!mesh) return;
  const M = node.getWorldMatrix();
  const tf = (x: number, y: number, z: number) => [
    M[0] * x + M[4] * y + M[8] * z + M[12],
    M[1] * x + M[5] * y + M[9] * z + M[13],
    M[2] * x + M[6] * y + M[10] * z + M[14],
  ];
  for (const p of mesh.listPrimitives()) {
    const pos = p.getAttribute("POSITION");
    if (!pos) continue;
    const mat = p.getMaterial();
    const c = mat ? mat.getBaseColorFactor() : [0.7, 0.7, 0.7, 1];
    const idx = p.getIndices();
    const n = idx ? idx.getCount() : pos.getCount();
    const get = (i: number) => { const v = pos.getElement(idx ? idx.getScalar(i) : i, [0, 0, 0]); return tf(v[0], v[1], v[2]); };
    for (let i = 0; i + 2 < n; i += 3) {
      const a = get(i), b = get(i + 1), d = get(i + 2);
      const u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]], v = [d[0] - a[0], d[1] - a[1], d[2] - a[2]];
      const cr = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
      const area2 = Math.hypot(cr[0], cr[1], cr[2]);
      if (area2 < 1e-12) continue;
      const nn = cr.map((x) => x / area2);
      let k = (area2 / 2) * density;
      while (k > 0) {
        if (k < 1 && rand() > k) break;
        k -= 1;
        let r1 = rand(), r2 = rand();
        if (r1 + r2 > 1) { r1 = 1 - r1; r2 = 1 - r2; }
        recs.push(a[0] + u[0] * r1 + v[0] * r2, a[1] + u[1] * r1 + v[1] * r2, a[2] + u[2] * r1 + v[2] * r2, nn[0], nn[1], nn[2], c[0], c[1], c[2]);
      }
    }
  }
});
await Bun.write(out, new Float32Array(recs));
console.log("points", recs.length / 9);
