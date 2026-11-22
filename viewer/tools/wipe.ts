// Registered wipes and split screens between captures of the same shot (same camera path, drive times, size and
// fps, so every pixel lines up; README "Capture"). Vertical edge, hard or soft, eased across the frame.
//
//   bun tools/wipe.ts --out ladder-wipe.mp4 --in a.mp4 --in b.mp4 [--in c.mp4 ...] --wipe 1.5:3 [--wipe 6:7.5]
//                     [--soft 0.04] [--line] [--rtl]
//   bun tools/wipe.ts --out split.mp4 --in a.mp4 --in b.mp4 --split 0.5 [--soft 0] [--line]
//   [--desc "..."]: add the result to the manifest beside it (film/captures/manifest.json) with its sources
//
// --wipe t0:t1 (seconds into the clips) hands input k over to input k+1: the edge sweeps left to right (--rtl:
// right to left) with a smoothstep ease, the new input on the side it has passed. --split x holds the edge at x
// (0..1 of the width; left the first input, right the second). --soft is the edge's width as a share of the
// frame's (0 = a hard cut), --line draws a thin light line on the edge while it is inside the frame.
// HEVC CRF 15 4:2:0 like the captures; the inputs must match in size and frame rate.

import { spawnSync } from "node:child_process";
import { existsSync, readFileSync, statSync, writeFileSync } from "node:fs";
import path from "node:path";

const args = process.argv.slice(2);
const all = (k: string) => args.flatMap((a, i) => (a === k ? [args[i + 1]] : []));
const one = (k: string) => all(k)[0];
const inputs = all("--in");
const out = one("--out");
const wipes = all("--wipe").map((w) => w.split(":").map(Number) as [number, number]);
const split = one("--split");
const soft = Number(one("--soft") ?? 0.03);
const line = args.includes("--line");
const rtl = args.includes("--rtl");
if (!out || inputs.length < 2 || (split === undefined && wipes.length !== inputs.length - 1)) {
  console.error("usage: bun tools/wipe.ts --out f.mp4 --in a.mp4 --in b.mp4 [...] (--wipe t0:t1 per hand-over | --split x) [--soft s] [--line] [--rtl]");
  process.exit(2);
}

/** the edge position (0..1 of the width) as an ffmpeg expression of T */
function pos(w: [number, number] | null) {
  if (!w) return String(Number(split));
  const [t0, t1] = w;
  const s = `clip((T-${t0})/${Math.max(1e-3, t1 - t0)},0,1)`;
  const e = `(${s}*${s}*(3-2*${s}))`;
  return rtl ? `(1-${e})` : e;
}

/** one blend: A the picture so far, B the next input; M = B's weight at this pixel */
function blend(w: [number, number] | null) {
  const P = `(${pos(w)}*W)`;
  // the side the edge has passed shows B (left of it for left-to-right and for a split's second input on the right)
  const side = w ? (rtl ? `(X-${P})` : `(${P}-X)`) : `(X-${P})`;
  const M = soft > 0 ? `clip(${side}/(${soft}*W)+0.5,0,1)` : `gte(${side},0)`;
  const base = `A*(1-${M})+B*${M}`;
  const p = pos(w);
  // the line: 0.18 % of the width, about 7 px at 4K, three quarters white, only while the edge is in frame
  const luma = line ? `if(lt(abs(X-${P}),0.0009*W)*gt(${p},0.001)*lt(${p},0.999),0.25*(${base})+0.75*235,${base})` : base;
  return `blend=c0_expr='${luma}':c1_expr='${base}':c2_expr='${base}'`;
}

const chain: string[] = [];
let last = "[0:v]";
const n = split !== undefined ? 1 : wipes.length;
for (let k = 0; k < n; k++) {
  const tag = `[w${k}]`;
  chain.push(`${last}[${k + 1}:v]${blend(split !== undefined ? null : wipes[k])}${tag}`);
  last = tag;
}
const ff = ["-v", "error", "-y", ...inputs.flatMap((f) => ["-i", f]), "-filter_complex", chain.join(";"), "-map", last, "-c:v", "libx265", "-preset", "medium", "-crf", "15", "-pix_fmt", "yuv420p", "-tag:v", "hvc1", "-x265-params", "log-level=error", "-movflags", "+faststart", "-shortest", out];
const r = spawnSync("ffmpeg", ff, { stdio: "inherit" });
// --desc: added to the manifest beside the output (its id the file name), with its sources
const desc = one("--desc");
const manifest = path.join(path.dirname(path.resolve(out)), "manifest.json");
if (r.status === 0 && desc && existsSync(manifest)) {
  const m = JSON.parse(readFileSync(manifest, "utf8"));
  const id = path.basename(out, ".mp4");
  const srcs = inputs.map((f) => path.basename(f, ".mp4"));
  const first = (m.shots as { id: string }[]).find((s) => s.id === srcs[0]) as Record<string, unknown> | undefined;
  m.shots = (m.shots as { id: string }[]).filter((s) => s.id !== id);
  m.shots.push({
    id,
    file: path.basename(out),
    composite: split !== undefined ? `split at ${split}` : `wipe${rtl ? " right to left" : ""} at ${wipes.map((x) => x.join("-")).join(", ")} s into the clip`,
    edge: { soft, line },
    sources: srcs,
    scene: first?.scene,
    viewer_commit: first?.viewer_commit,
    camera: first?.camera,
    t: first?.t,
    fps: first?.fps,
    size: first?.size,
    bytes: statSync(out).size,
    description: desc,
    captured: new Date().toISOString(),
  });
  m.updated = new Date().toISOString();
  writeFileSync(manifest, JSON.stringify(m, null, 1));
}
process.exit(r.status ?? 1);
