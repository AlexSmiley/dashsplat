// Streams contract tiles around the car: static tiles by distance from a focus point, transient tiles by the
// time window t_min..t_max around playback time. Decoding happens in a small worker pool; recently unloaded
// tiles stay decoded in an LRU so scrubbing back and forth does not refetch.

import * as THREE from "three";
import { ExtSplats, SplatMesh } from "@sparkjsdev/spark";
import TileWorker from "./tileWorker?worker";
import type { GroundFilter, TileRequest, TileResult } from "./tileWorker";
import { makeTileUniforms, makeTimeModifier, makeTimeTexture, TIME_TEX_WIDTH, type SplatGlobals, type TileUniforms } from "../render/splatShader";

export type MapTile = {
  id: string;
  kind?: "static" | "transient";
  layer?: string;
  file: string;
  count: number;
  bbox: number[];
  t_min: number;
  t_max: number;
};
export type MapJson = { t_end?: number; tiles: MapTile[]; sky?: string | null; environment?: string | null };
/** optional compact cache written by tools/build-tile-cache.ts */
export type RsbIndex = { v: number; tiles: Record<string, { file: string; bytes: number; count?: number }> };

type Live = {
  tile: MapTile;
  mesh: SplatMesh;
  ext: ExtSplats;
  tex: THREE.DataTexture;
  U: TileUniforms;
  loadedAt: number;
  data: TileResult;
};

export type TileStats = {
  total: number;
  loaded: number;
  loading: number;
  queued: number;
  splatsLoaded: number;
  bytes: number;
  lastMs: number;
};

const transient = (t: MapTile) => t.kind === "transient";

export class TileManager {
  readonly group = new THREE.Group();
  private live = new Map<string, Live>();
  private inflight = new Set<string>();
  private queue: MapTile[] = [];
  private cache = new Map<string, TileResult>();
  private cacheBytes = 0;
  private workers: { w: Worker; busy: boolean }[] = [];
  private failed = new Set<string>();
  private lastPlan = 0;
  private byId = new Map<string, MapTile>();
  /** decoded tiles waiting to be installed, at most one per frame (texture uploads are the hitch) */
  private ready: { t: MapTile; r: TileResult }[] = [];
  stats: TileStats = { total: 0, loaded: 0, loading: 0, queued: 0, splatsLoaded: 0, bytes: 0, lastMs: 0 };
  onChange?: () => void;
  onInstall?: (data: TileResult) => void;
  /** phones: ground-layer splats under the opaque textured ground are left out at decode, so the budget goes to
   *  the scenery; the filter for a tile, "wait" while the atlas coverage there is still being read (at most 5 s),
   *  or null for none. With a filter, ground tiles load only within groundReach of the focus (farther they are
   *  hazed out, and what loads far out is scenery) */
  groundFilter?: (t: MapTile) => GroundFilter | "wait" | null;
  groundReach = 120;
  private waitingSince = new Map<string, number>();
  /** decoded (possibly filtered) splat counts, for the budget */
  private counts = new Map<string, number>();
  /** splats left out by the ground filter so far (stats) */
  filtered = 0;
  /** bumped by resetGround: ground tiles decoded under an older filter are thrown away */
  private groundGen = 0;
  private reqGen = new Map<string, number>();

  constructor(
    private readonly baseUrl: string,
    private readonly map: MapJson,
    private readonly G: SplatGlobals,
    private readonly opts: { workers: number; maxSplats: number; cacheBytes: number },
    private readonly rsb: RsbIndex | null = null,
  ) {
    this.group.name = "tiles";
    for (const t of map.tiles) this.byId.set(t.id, t);
    this.stats.total = map.tiles.length;
    for (let i = 0; i < opts.workers; i++) {
      const w = new TileWorker();
      const slot = { w, busy: false };
      w.onmessage = (ev: MessageEvent) => this.onResult(slot, ev.data);
      w.onerror = (e) => console.error("tile worker error", e.message);
      this.workers.push(slot);
    }
  }

  get meshes() {
    return [...this.live.values()].map((l) => l.mesh);
  }

  /** distance from (x, y) to a tile's footprint */
  private dist(t: MapTile, x: number, y: number) {
    const b = t.bbox;
    const dx = Math.max(b[0] - x, 0, x - b[3]);
    const dy = Math.max(b[1] - y, 0, y - b[4]);
    return Math.hypot(dx, dy);
  }

  /** seconds of drive time to prefetch transient tiles ahead of T (grows with playback speed) */
  lookahead = 3;
  /** director looks: a tile's presence (0..1) by drive time, times its fade-in after load */
  presence?: (t: MapTile) => number;
  private wanted(t: MapTile, x: number, y: number, T: number, radius: number, stitched: boolean) {
    if (transient(t)) {
      if (stitched) return false;
      return T >= t.t_min - this.lookahead && T <= t.t_max + 0.5 && this.dist(t, x, y) < radius;
    }
    if (!stitched && t.t_min > T + 4) return false; // nothing in it is born yet
    return this.dist(t, x, y) < radius;
  }

  /** decide what to load and unload; cheap enough to call every frame, plans at most 5 times a second */
  update(x: number, y: number, T: number, radius: number, stitched: boolean, now: number, force = false) {
    // per-tile fade-in after load
    for (const l of this.live.values()) {
      const f = Math.min(1, (now - l.loadedAt) / 450);
      l.U.tile.value.z = this.presence ? f * this.presence(l.tile) : f;
    }
    if (!force && now - this.lastPlan < 200) return;
    this.lastPlan = now;
    const unloadR = radius + 45;
    // unload
    for (const [id, l] of this.live) {
      const t = l.tile;
      const keep = transient(t)
        ? !stitched && T >= t.t_min - 6 && T <= t.t_max + 3 && this.dist(t, x, y) < unloadR
        : this.dist(t, x, y) < (this.groundFilter && t.layer?.startsWith("ground") ? this.groundReach + 60 : unloadR) && (stitched || t.t_min <= T + 8);
      if (!keep) this.unload(id);
    }
    // budget: drop the farthest static tiles first
    let total = 0;
    for (const l of this.live.values()) total += l.data.count;
    if (total > this.opts.maxSplats) {
      const far = [...this.live.values()].sort((a, b) => this.dist(b.tile, x, y) - this.dist(a.tile, x, y));
      for (const l of far) {
        if (total <= this.opts.maxSplats * 0.92) break;
        if (this.dist(l.tile, x, y) < 60) break;
        total -= l.data.count;
        this.unload(l.tile.id);
      }
    }
    // queue what is wanted, nearest first (transients near in time first)
    const want: { t: MapTile; p: number }[] = [];
    for (const t of this.map.tiles) {
      if (this.live.has(t.id) || this.inflight.has(t.id) || this.failed.has(t.id) || this.ready.some((q) => q.t === t)) continue;
      if (!this.wanted(t, x, y, T, radius, stitched)) continue;
      const d = this.dist(t, x, y);
      if (this.groundFilter && t.layer?.startsWith("ground") && d > this.groundReach) continue;
      want.push({ t, p: transient(t) ? d * 0.5 - 30 : d });
    }
    want.sort((a, b) => a.p - b.p);
    let budget = this.opts.maxSplats - total;
    this.queue = [];
    for (const w of want) {
      const n = this.counts.get(w.t.id) ?? w.t.count;
      if (n > budget && this.live.size > 0) continue;
      budget -= n;
      this.queue.push(w.t);
    }
    this.pump();
    this.updateStats();
  }

  private pump() {
    // decoded tiles still in the cache install immediately (at most a few per call to spread the uploads)
    let installed = 0;
    this.queue = this.queue.filter((t) => {
      const cached = this.cache.get(t.id);
      if (!cached || installed >= 4) return true;
      this.cache.delete(t.id);
      this.cacheBytes -= cachedBytes(cached);
      this.ready.push({ t, r: cached });
      installed++;
      return false;
    });
    const now = performance.now();
    for (const slot of this.workers) {
      if (slot.busy) continue;
      let filter: GroundFilter | null = null;
      const t = this.queue.find((q) => {
        if (this.cache.has(q.id)) return false;
        filter = null;
        if (!this.groundFilter || !q.layer?.startsWith("ground")) return true;
        const f = this.groundFilter(q);
        if (f !== "wait") {
          filter = f;
          return true;
        }
        const since = this.waitingSince.get(q.id) ?? now;
        this.waitingSince.set(q.id, since);
        return now - since > 5000;
      });
      if (!t) break;
      this.waitingSince.delete(t.id);
      this.reqGen.set(t.id, this.groundGen);
      this.queue.splice(this.queue.indexOf(t), 1);
      slot.busy = true;
      this.inflight.add(t.id);
      // use the compact cache only when it was built from this version of the tile
      const cached = this.rsb?.tiles[t.id];
      const file = cached && cached.count === t.count ? "rsb/" + cached.file : t.file;
      const req: TileRequest = { id: t.id, url: new URL(file, new URL(this.baseUrl, location.href)).href, texWidth: TIME_TEX_WIDTH, ...(filter ? { filter } : {}) };
      slot.w.postMessage(req);
    }
    this.updateStats();
  }

  private onResult(slot: { w: Worker; busy: boolean }, msg: { ok: boolean; result?: TileResult; id?: string; error?: string }) {
    slot.busy = false;
    if (!msg.ok || !msg.result) {
      console.warn("tile failed", msg.id, msg.error);
      if (msg.id) {
        this.inflight.delete(msg.id);
        this.failed.add(msg.id);
      }
      this.pump();
      return;
    }
    const r = msg.result;
    this.inflight.delete(r.id);
    this.stats.lastMs = r.ms;
    this.stats.bytes += r.bytes;
    const t = this.byId.get(r.id);
    const stale = !!t?.layer?.startsWith("ground") && this.reqGen.get(r.id) !== this.groundGen;
    if (t && !stale) {
      if (r.count < t.count) this.filtered += t.count - r.count;
      this.counts.set(r.id, r.count);
      this.ready.push({ t, r });
    }
    this.pump();
  }

  /** install one decoded tile and upload its textures now, so the upload cost is spread one tile per frame */
  tick(renderer: THREE.WebGLRenderer) {
    const next = this.ready.shift();
    if (!next) return;
    const l = this.install(next.t, next.r);
    if (l) {
      renderer.initTexture(l.tex);
      for (const tex of l.ext.textures) renderer.initTexture(tex);
    }
  }

  private install(t: MapTile, r: TileResult) {
    if (this.live.has(t.id)) return;
    const ext = new ExtSplats({ extArrays: [r.extA, r.extB], numSplats: r.count });
    const tex = makeTimeTexture(r.times, r.count);
    const U = makeTileUniforms(tex, r.base, r.quantum, transient(t), !!t.layer?.endsWith("-predicted"), !!t.layer && !t.layer.startsWith("ground"));
    const mesh = new SplatMesh({ extSplats: ext, worldModifier: makeTimeModifier(this.G, U), editable: false, raycastable: false });
    mesh.name = t.id;
    mesh.position.set(r.origin[0], r.origin[1], r.origin[2]);
    mesh.updateMatrixWorld();
    this.group.add(mesh);
    const l: Live = { tile: t, mesh, ext, tex, U, loadedAt: performance.now(), data: r };
    this.live.set(t.id, l);
    this.updateStats();
    this.onChange?.();
    this.onInstall?.(r);
    return l;
  }

  private unload(id: string) {
    const l = this.live.get(id);
    if (!l) return;
    this.group.remove(l.mesh);
    l.mesh.dispose();
    l.tex.dispose();
    this.live.delete(id);
    // keep the decoded arrays for a while
    this.cache.set(id, l.data);
    this.cacheBytes += cachedBytes(l.data);
    while (this.cacheBytes > this.opts.cacheBytes && this.cache.size) {
      const [k, v] = this.cache.entries().next().value as [string, TileResult];
      this.cache.delete(k);
      this.cacheBytes -= cachedBytes(v);
    }
    this.updateStats();
  }

  /** drop the ground-layer tiles (live and decoded) so they load again: the ground filter was switched */
  resetGround() {
    for (const [id, l] of [...this.live]) if (l.tile.layer?.startsWith("ground")) {
      this.unload(id);
    }
    for (const [id, v] of [...this.cache]) if (this.byId.get(id)?.layer?.startsWith("ground")) {
      this.cache.delete(id);
      this.cacheBytes -= cachedBytes(v);
    }
    for (const id of [...this.counts.keys()]) if (this.byId.get(id)?.layer?.startsWith("ground")) this.counts.delete(id);
    this.ready = this.ready.filter((q) => !q.t.layer?.startsWith("ground"));
    this.filtered = 0;
    this.groundGen++;
    this.lastPlan = 0;
  }

  /** mark all tile meshes dirty so Spark regenerates them (time or shading uniforms changed) */
  touch() {
    for (const l of this.live.values()) l.mesh.updateVersion();
  }

  private updateStats() {
    let s = 0;
    for (const l of this.live.values()) s += l.data.count;
    this.stats.loaded = this.live.size;
    this.stats.loading = this.inflight.size;
    this.stats.queued = this.queue.length + this.ready.length;
    this.stats.splatsLoaded = s;
  }

  dispose() {
    for (const id of [...this.live.keys()]) this.unload(id);
    this.cache.clear();
    for (const s of this.workers) s.w.terminate();
    this.workers = [];
  }
}

function cachedBytes(r: TileResult) {
  return r.extA.byteLength + r.extB.byteLength + r.times.byteLength;
}
