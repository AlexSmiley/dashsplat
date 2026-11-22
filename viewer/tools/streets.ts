#!/usr/bin/env bun
// One-off: names the streets the drive turns onto and the place it ends, for the next-turn and trip cards.
// Reverse-geocodes a few points of scenes/dev/drive.json with OpenStreetMap Nominatim (1 request per second,
// data (c) OpenStreetMap contributors, ODbL) and writes public/streets.json. The viewer matches entries to any
// scene's drive by position, so scenes of the same drive share it; other drives fall back to generic labels.
//
// Usage: bun tools/streets.ts [path to drive.json]

import path from "node:path";
import { Drive, type DriveJson } from "../src/scene/drive";
import { findTurns } from "../src/render/path";

const src = process.argv[2] ?? path.resolve(import.meta.dir, "../../scenes/dev/drive.json");
const drive = new Drive((await Bun.file(src).json()) as DriveJson);
const OUT = path.resolve(import.meta.dir, "../public/streets.json");
type Entry = { kind: "turn" | "end" | "start"; s: number; lat: number; lon: number; road: string; city: string };
const entries: Entry[] = [];
const turns = findTurns(drive);
const points: { kind: Entry["kind"]; s: number }[] = [{ kind: "start", s: 0 }, ...turns.map((t) => ({ kind: "turn" as const, s: Math.min(drive.length, t.s + 35) })), { kind: "end", s: drive.length }];
for (const p of points) {
  const ll = drive.latLonAtS(p.s);
  if (!ll) throw new Error("drive.json has no lat/lon");
  const url = `https://nominatim.openstreetmap.org/reverse?format=jsonv2&zoom=17&lat=${ll[0]}&lon=${ll[1]}`;
  const res = await fetch(url, { headers: { "User-Agent": "reality-splat-viewer/0.1 (personal project)" } });
  const j = (await res.json()) as { address?: Record<string, string>; name?: string };
  const a = j.address ?? {};
  const road = a.road ?? a.pedestrian ?? a.service ?? j.name ?? "";
  const city = a.city ?? a.town ?? a.village ?? a.county ?? "";
  entries.push({ kind: p.kind, s: Math.round(p.s), lat: ll[0], lon: ll[1], road, city });
  console.log(p.kind, Math.round(p.s), road, "|", city);
  await new Promise((r) => setTimeout(r, 1100));
}
await Bun.write(OUT, JSON.stringify({ source: "OpenStreetMap Nominatim (ODbL)", entries }, null, 1));
console.log("wrote", OUT);
