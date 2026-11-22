#!/usr/bin/env bun
// Prepares public/models/model3.glb from a Sketchfab Model 3 Highland model. The file's embedded metadata credits "2024 Tesla Model 3" by RBLXSupercars; the notes recorded "Tesla model 3 2024" by brandonleong28
// (Sketchfab, CC BY 4.0): https://sketchfab.com/3d-models/tesla-model-3-2024-36c52f3f89f6439c90310f14e8ff33f2
//
// Output is in the scene contract's ego frame, in metres: x forward, y left, z up, origin at the rear-axle
// centre on the ground (wheelbase 2.875 m, length 4.72 m). The four wheels become nodes pivoted at their hubs
// (wheel_fl, wheel_fr, wheel_rl, wheel_rr) so the viewer can steer and spin them. Materials the viewer drives
// are renamed: Paint_Color, Tail_Lamp, Head_Lamp, Indicator_L, Indicator_R. Textures go to 1024 px WebP and
// geometry is meshopt-compressed.
//
// Usage: bun tools/prepare-car.ts <path to source/2024_tesla_model_3.glb from the Sketchfab download>

import { NodeIO, PropertyType, type Document, type Material, type Primitive, type Node } from "@gltf-transform/core";
import { ALL_EXTENSIONS, EXTMeshoptCompression } from "@gltf-transform/extensions";
import { dedup, meshopt, prune, quantize, reorder, textureCompress, weld } from "@gltf-transform/functions";
import { MeshoptEncoder } from "meshoptimizer";
import sharp from "sharp";
import path from "node:path";

const input = process.argv[2];
if (!input) throw new Error("usage: bun tools/prepare-car.ts <2024_tesla_model_3.glb>");
const OUT = path.resolve(import.meta.dir, "../public/models/model3.glb");

// measured on the source mesh (raw z-up units): tyre hubs and size
const REAR_X = -10.429, FRONT_X = 0.232, MID_Y = -9.9295;
const S = 2.875 / (FRONT_X - REAR_X); // metres per source unit
const HUB_Z = 1.28 * S, TYRE_R = 1.28 * S, HUB_Y = ((-6.938 - -12.921) / 2) * S;
const WHEELS = [
  { name: "wheel_rl", x: 0, y: HUB_Y },
  { name: "wheel_rr", x: 0, y: -HUB_Y },
  { name: "wheel_fl", x: 2.875, y: HUB_Y },
  { name: "wheel_fr", x: 2.875, y: -HUB_Y },
];

await MeshoptEncoder.ready;
const io = new NodeIO().registerExtensions(ALL_EXTENSIONS).registerDependencies({ "meshopt.encoder": MeshoptEncoder });
const doc: Document = await io.read(input);
const root = doc.getRoot();
const scene = root.listScenes()[0];
const buffer = root.listBuffers()[0];
const out = doc.createNode("tesla_model_3_highland");

// world (glTF y-up, after the source's -90 degree X root rotation) -> ego metres
const toEgo = (wx: number, wy: number, wz: number): [number, number, number] => [(wx - REAR_X) * S, (-wz - MID_Y) * S, wy * S];
const dirToEgo = (wx: number, wy: number, wz: number): [number, number, number] => [wx, -wz, wy];

const meshNodes: Node[] = [];
scene.traverse((n) => { if (n.getMesh()) meshNodes.push(n); });

// 1. bake every mesh into ego coordinates
const prims: Primitive[] = [];
for (const n of meshNodes) {
  const M = n.getWorldMatrix();
  const mesh = n.getMesh()!;
  for (const p of mesh.listPrimitives()) {
    const pos = p.getAttribute("POSITION")!;
    const a = pos.getArray()! as Float32Array;
    const o = new Float32Array(a.length);
    for (let i = 0; i < a.length; i += 3) {
      const x = a[i], y = a[i + 1], z = a[i + 2];
      const wx = M[0] * x + M[4] * y + M[8] * z + M[12];
      const wy = M[1] * x + M[5] * y + M[9] * z + M[13];
      const wz = M[2] * x + M[6] * y + M[10] * z + M[14];
      o.set(toEgo(wx, wy, wz), i);
    }
    p.setAttribute("POSITION", doc.createAccessor().setType("VEC3").setArray(o).setBuffer(buffer));
    const nor = p.getAttribute("NORMAL");
    if (nor) {
      const b = nor.getArray()! as Float32Array;
      const q = new Float32Array(b.length);
      for (let i = 0; i < b.length; i += 3) {
        const x = b[i], y = b[i + 1], z = b[i + 2];
        const wx = M[0] * x + M[4] * y + M[8] * z, wy = M[1] * x + M[5] * y + M[9] * z, wz = M[2] * x + M[6] * y + M[10] * z;
        const d = dirToEgo(wx, wy, wz);
        const l = Math.hypot(d[0], d[1], d[2]) || 1;
        q.set([d[0] / l, d[1] / l, d[2] / l], i);
      }
      p.setAttribute("NORMAL", doc.createAccessor().setType("VEC3").setArray(q).setBuffer(buffer));
    }
    prims.push(p);
  }
}

// 2. split triangles: wheels by cylinder, lamps by side / end
function triangles(p: Primitive) {
  const idx = p.getIndices();
  const n = p.getAttribute("POSITION")!.getCount();
  return idx ? Array.from(idx.getArray()!) : Array.from({ length: n }, (_, i) => i);
}
function subPrimitive(p: Primitive, tris: number[], offset: [number, number, number], mat: Material | null) {
  const used = [...new Set(tris)].sort((a, b) => a - b);
  const remap = new Map(used.map((v, i) => [v, i]));
  const np = doc.createPrimitive().setMaterial(mat);
  for (const sem of p.listSemantics()) {
    const src = p.getAttribute(sem)!;
    const k = src.getElementSize();
    const sa = src.getArray()! as Float32Array;
    const o = new Float32Array(used.length * k);
    used.forEach((v, i) => { for (let j = 0; j < k; j++) o[i * k + j] = sa[v * k + j]; });
    if (sem === "POSITION") for (let i = 0; i < used.length; i++) { o[i * 3] -= offset[0]; o[i * 3 + 1] -= offset[1]; o[i * 3 + 2] -= offset[2]; }
    np.setAttribute(sem, doc.createAccessor().setType(src.getType()).setArray(o).setBuffer(buffer));
  }
  np.setIndices(doc.createAccessor().setType("SCALAR").setArray(new Uint32Array(tris.map((v) => remap.get(v)!))).setBuffer(buffer));
  return np;
}
const renamed = new Map<string, Material>();
function namedMaterial(base: Material | null, name: string) {
  if (!base) return null;
  let m = renamed.get(name);
  if (!m) { m = base.clone().setName(name); renamed.set(name, m); }
  return m;
}
const bodyMesh = doc.createMesh("body");
const wheelMeshes = WHEELS.map((w) => doc.createMesh(w.name));
let wheelTris = 0;
for (const p of prims) {
  const pos = p.getAttribute("POSITION")!.getArray()! as Float32Array;
  const mat = p.getMaterial();
  const mname = mat?.getName() ?? "";
  const tris = triangles(p);
  const groups = new Map<string, number[]>();
  for (let t = 0; t < tris.length; t += 3) {
    const a = tris[t] * 3, b = tris[t + 1] * 3, c = tris[t + 2] * 3;
    const x = (pos[a] + pos[b] + pos[c]) / 3, y = (pos[a + 1] + pos[b + 1] + pos[c + 1]) / 3, z = (pos[a + 2] + pos[b + 2] + pos[c + 2]) / 3;
    let key = "body";
    for (let w = 0; w < WHEELS.length; w++) {
      const W = WHEELS[w];
      if (Math.sign(y) !== Math.sign(W.y) || Math.abs(y) < 0.52 || Math.abs(y) > 1.0) continue;
      if (Math.hypot(x - W.x, z - HUB_Z) < TYRE_R + 0.012) { key = "w" + w; break; }
    }
    if (key === "body") {
      if (mname === "Ln1Mtl") key = y > 0 ? "m:Indicator_L" : "m:Indicator_R";
      else if (mname === "Ln12Mtl" && x < 1.5) key = "m:Tail_Lamp";
      else if (mname === "Ln7Mtl" && x > 2.5) key = "m:Head_Lamp";
      else if (mname === "Geohoodsub00021Mtl") key = "m:Paint_Color";
    }
    let g = groups.get(key);
    if (!g) groups.set(key, (g = []));
    g.push(tris[t], tris[t + 1], tris[t + 2]);
  }
  for (const [key, g] of groups) {
    if (key.startsWith("w")) {
      const w = Number(key.slice(1));
      wheelTris += g.length / 3;
      wheelMeshes[w].addPrimitive(subPrimitive(p, g, [WHEELS[w].x, WHEELS[w].y, HUB_Z], mat));
    } else if (key.startsWith("m:")) bodyMesh.addPrimitive(subPrimitive(p, g, [0, 0, 0], namedMaterial(mat, key.slice(2))));
    else bodyMesh.addPrimitive(subPrimitive(p, g, [0, 0, 0], mat));
  }
}
console.log("wheel triangles", wheelTris, "scale", S.toFixed(5), "hub", HUB_Y.toFixed(3), HUB_Z.toFixed(3));

// 3. new scene graph: one body node and four wheel nodes, nothing else
for (const n of [...scene.listChildren()]) scene.removeChild(n);
out.addChild(doc.createNode("body").setMesh(bodyMesh));
WHEELS.forEach((w, i) => out.addChild(doc.createNode(w.name).setMesh(wheelMeshes[i]).setTranslation([w.x, w.y, HUB_Z])));
scene.addChild(out);
for (const n of meshNodes) n.dispose();

await doc.transform(
  prune(),
  // keep the renamed lamp and paint materials distinct even where they equal their source
  dedup({ propertyTypes: [PropertyType.ACCESSOR, PropertyType.MESH, PropertyType.TEXTURE] }),
  weld(),
  textureCompress({ encoder: sharp, targetFormat: "webp", resize: [1024, 1024] }),
  reorder({ encoder: MeshoptEncoder }),
  quantize(),
  meshopt({ encoder: MeshoptEncoder, level: "medium" }),
  prune(),
);
doc.createExtension(EXTMeshoptCompression).setRequired(true).setEncoderOptions({ method: EXTMeshoptCompression.EncoderMethod.QUANTIZE });
await io.write(OUT, doc);
console.log("wrote", OUT, (Bun.file(OUT).size / 1e6).toFixed(2), "MB");
