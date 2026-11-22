#!/usr/bin/env bun
// Converts a scene's contract PLY tiles into the viewer's compact .rsb cache (about 33 bytes per splat instead
// of 96), written inside the scene so it goes away with it:
//   scenes/<name>/rsb/<tile id>.rsb   and   scenes/<name>/rsb/index.json
// The viewer uses the cache when rsb/index.json exists and falls back to the PLY tiles otherwise. Tiles whose
// PLY is newer than its cache entry are rebuilt; run again after a scene is replaced.
//
// Usage: bun tools/build-tile-cache.ts <scene dir> [--force]

import { existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import path from "node:path";
import { encodeRsb, parsePly } from "../src/scene/tileCodec";
import { PathIndex, RoadColour, type WhiteBalance } from "../src/scene/whiteBalance";

const dir = process.argv[2];
if (!dir) throw new Error("usage: bun tools/build-tile-cache.ts <scene dir> [--force]");
const force = process.argv.includes("--force");
const map = JSON.parse(readFileSync(path.join(dir, "map.json"), "utf8")) as { tiles: { id: string; file: string; count: number }[] };
const out = path.join(dir, "rsb");
mkdirSync(out, { recursive: true });
const idxPath = path.join(out, "index.json");
const index: { v: number; tiles: Record<string, { file: string; bytes: number; count: number; src: string; srcBytes: number; srcMtime: number }>; grade?: WhiteBalance; objects?: boolean; generated?: boolean; photo?: boolean; ground?: boolean } = existsSync(idxPath) ? JSON.parse(readFileSync(idxPath, "utf8")) : { v: 1, tiles: {} };
let inBytes = 0, outBytes = 0, built = 0;
// road colour for the viewer's Natural grade, sampled near the driven path
const drive = JSON.parse(readFileSync(path.join(dir, "drive.json"), "utf8")) as { x: number[]; y: number[]; z?: number[] };
const road = force || !index.grade ? new RoadColour(new PathIndex(drive.x, drive.y, drive.z ?? drive.x.map(() => 0))) : null;
const SH_C0 = 0.28209479177387814;
const t0 = performance.now();
for (const t of map.tiles) {
  const src = path.join(dir, t.file);
  const st = statSync(src);
  const prev = index.tiles[t.id];
  const file = `${t.id}.rsb`;
  inBytes += st.size;
  if (!force && !road && prev && prev.count === t.count && prev.srcBytes === st.size && prev.srcMtime === st.mtimeMs && existsSync(path.join(out, file))) {
    outBytes += prev.bytes;
    continue;
  }
  const buf = readFileSync(src);
  const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer;
  if (road && road.count < 120000) {
    const ply = parsePly(ab);
    const [X, Y, Z, C0, C1, C2, OP] = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity"].map((n) => ply.get(n));
    const step = Math.max(1, Math.floor(ply.count / 20000));
    for (let i = 0; i < ply.count; i += step) road.add(0.5 + SH_C0 * C0(i), 0.5 + SH_C0 * C1(i), 0.5 + SH_C0 * C2(i), 1 / (1 + Math.exp(-OP(i))), X(i), Y(i), Z(i));
  }
  const rsb = encodeRsb(ab);
  writeFileSync(path.join(out, file), rsb);
  index.tiles[t.id] = { file, bytes: rsb.byteLength, count: t.count, src: t.file, srcBytes: st.size, srcMtime: st.mtimeMs };
  outBytes += rsb.byteLength;
  built++;
  if (built % 10 === 0) writeFileSync(idxPath, JSON.stringify(index));
}
const wb = road?.result();
if (wb) index.grade = wb;
index.objects = existsSync(path.join(dir, "objects.json"));
index.generated = existsSync(path.join(dir, "generated.json"));
index.photo = existsSync(path.join(dir, "ibr.json"));
index.ground = existsSync(path.join(dir, "ground", "ground.json"));
writeFileSync(idxPath, JSON.stringify(index));
console.log(`  grade: ${index.grade ? index.grade.gains.map((v) => v.toFixed(3)).join(" ") + " from " + index.grade.samples + " road samples" : "none"}, objects.json: ${index.objects}, generated.json: ${index.generated}, ibr.json: ${index.photo}, ground: ${index.ground}`);
console.log(`${dir}: ${map.tiles.length} tiles (${built} built) ${(inBytes / 1e6).toFixed(0)} MB PLY -> ${(outBytes / 1e6).toFixed(0)} MB rsb in ${((performance.now() - t0) / 1000).toFixed(1)} s`);
