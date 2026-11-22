// Keeps the capture library small on this Mac (its disk is shared and nearly full). Every finished shot in
// film/captures/manifest.json is mirrored to the Studio (anomastudio:~/dev/reality-splat/film/captures/, same
// relative paths); only the shots named in film/captures/keep.txt (the ones the film uses or wipes are built
// from) and the newest few GB of the rest stay here. Each manifest entry records where its file is.
//
//   bun tools/offload.ts [--keep-gb 3] [--dry]      mirror, then remove what is not kept here
//   bun tools/offload.ts --fetch <id>[,<id>]        bring shots back from the Studio (add them to keep.txt too)
//
// keep.txt: one id per line (a trailing * matches a prefix), # comments.
//
// The author's tool: it copies with his ferry CLI to RS_STUDIO_HOST (default: his Mac Studio). Not needed to capture.

import { spawnSync } from "node:child_process";
import { existsSync, readFileSync, renameSync, rmSync, statSync, writeFileSync } from "node:fs";
import path from "node:path";

const HERE = path.dirname(new URL(import.meta.url).pathname);
const DIR = path.resolve(HERE, "..", "..", "film", "captures");
const HOST = process.env.RS_STUDIO_HOST ?? "anomastudio";
const REMOTE = "dev/reality-splat/film/captures";
const MANIFEST = path.join(DIR, "manifest.json");
const args = process.argv.slice(2);
const opt = (k: string) => {
  const i = args.indexOf(k);
  return i >= 0 ? args[i + 1] : undefined;
};
const keepBytes = Number(opt("--keep-gb") ?? 3) * 1e9;
const dry = args.includes("--dry");

type Entry = { id: string; file: string; bytes?: number; captured?: string; location?: string };

function readManifest(): { shots: Entry[]; [k: string]: unknown } {
  // a capture run may be writing it this instant: retry a torn read
  for (let i = 0; ; i++) {
    try {
      return JSON.parse(readFileSync(MANIFEST, "utf8"));
    } catch (e) {
      if (i > 20) throw e;
      Bun.sleepSync(250);
    }
  }
}
/** set fields on entries, re-reading the manifest right before the write (capture runs add to it meanwhile) */
function updateManifest(set: Map<string, Partial<Entry>>) {
  const m = readManifest();
  for (const s of m.shots) Object.assign(s, set.get(s.id) ?? {});
  const tmp = MANIFEST + ".tmp";
  writeFileSync(tmp, JSON.stringify(m, null, 1));
  renameSync(tmp, MANIFEST);
}

function ferry(argv: string[]) {
  const r = spawnSync("ferry", argv, { encoding: "utf8", maxBuffer: 1 << 26 });
  if (r.status !== 0) throw new Error(`ferry ${argv.join(" ")}: ${r.stderr || r.stdout}`);
  return r.stdout;
}
/** sizes of the files on the Studio */
function remoteSizes() {
  const out = spawnSync("ferry", ["sh", HOST, `mkdir -p ~/${REMOTE} && cd ~/${REMOTE} && for f in *.mp4; do [ -f "$f" ] && stat -f "%z %N" "$f"; done; true`], { encoding: "utf8", maxBuffer: 1 << 26 }).stdout;
  const m = new Map<string, number>();
  for (const l of out.split("\n")) {
    const k = l.indexOf(" ");
    if (k > 0) m.set(l.slice(k + 1), Number(l.slice(0, k)));
  }
  return m;
}

function keepList() {
  const f = path.join(DIR, "keep.txt");
  if (!existsSync(f)) return [] as string[];
  return readFileSync(f, "utf8").split("\n").map((l) => l.replace(/#.*/, "").trim()).filter(Boolean);
}
const kept = (id: string, list: string[]) => list.some((k) => (k.endsWith("*") ? id.startsWith(k.slice(0, -1)) : id === k));

const studio = (file: string) => `${HOST}:~/${REMOTE}/${file}`;

if (opt("--fetch")) {
  const ids = opt("--fetch")!.split(",");
  const m = readManifest();
  const set = new Map<string, Partial<Entry>>();
  for (const s of m.shots.filter((x) => ids.includes(x.id))) {
    ferry(["cp", `${HOST}:${REMOTE}/${s.file}`, DIR + "/"]);
    set.set(s.id, { location: `basebook+${studio(s.file)}` });
    console.log(`fetched ${s.file}`);
  }
  updateManifest(set);
  process.exit(0);
}

const m = readManifest();
const list = keepList();
// mirror every finished file that is here
const local = m.shots.filter((s) => existsSync(path.join(DIR, s.file)));
let remote = remoteSizes();
for (const s of local) {
  const size = statSync(path.join(DIR, s.file)).size;
  if (remote.get(s.file) === size) continue;
  if (dry) console.log(`would mirror ${s.file}`);
  else ferry(["cp", path.join(DIR, s.file), `${HOST}:${REMOTE}/`]);
}
if (!dry) {
  // the manifest, keep list and build patches go along, so the Studio copy describes itself
  for (const f of ["manifest.json", "keep.txt", "builds"]) if (existsSync(path.join(DIR, f))) ferry(["cp", path.join(DIR, f), `${HOST}:${REMOTE}/`]);
  remote = remoteSizes();
}
// what stays: the keep list, then the newest HUD-off shots up to the budget
const stay = new Set<string>();
let used = 0;
for (const s of local) if (kept(s.id, list)) {
  stay.add(s.id);
  used += statSync(path.join(DIR, s.file)).size;
}
const newest = local.filter((s) => !stay.has(s.id) && !s.id.endsWith("-hud")).sort((a, b) => String(b.captured).localeCompare(String(a.captured)));
let extra = 0;
for (const s of newest) {
  const size = statSync(path.join(DIR, s.file)).size;
  if (extra + size > keepBytes) break;
  stay.add(s.id);
  extra += size;
  used += size;
}
const set = new Map<string, Partial<Entry>>();
let freed = 0;
for (const s of m.shots) {
  const here = existsSync(path.join(DIR, s.file));
  const there = remote.has(s.file) && (!here || remote.get(s.file) === statSync(path.join(DIR, s.file)).size);
  if (here && there && !stay.has(s.id)) {
    if (dry) console.log(`would move ${s.file}`);
    else {
      freed += statSync(path.join(DIR, s.file)).size;
      rmSync(path.join(DIR, s.file));
      set.set(s.id, { location: studio(s.file) });
    }
    continue;
  }
  set.set(s.id, { location: here ? (there ? `basebook+${studio(s.file)}` : "basebook") : there ? studio(s.file) : "missing" });
}
if (!dry) {
  updateManifest(set);
  ferry(["cp", MANIFEST, `${HOST}:${REMOTE}/`]);
}
console.log(`${local.length} here before, ${stay.size} kept (${(used / 1e9).toFixed(2)} GB), ${(freed / 1e9).toFixed(2)} GB moved to the Studio`);
