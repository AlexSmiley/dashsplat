// Frame-exact shot capture for the film (README: "Capture"). Renders shots offline, one frame at a time, in a
// headless Chrome with real WebGL (Metal), each frame settled before it is grabbed: tiles loaded and faded in,
// the splat sort finished, the ground atlas and movers in, the generated clip seeked to the exact frame, the car's
// environment re-captured from the settled frame. Playback speed is irrelevant; the same shot renders the same.
//
//   bun tools/capture.ts <shots.json> [--only id1,id2] [--out ../film/captures] [--force]
//
// shots.json: an array of shots:
//   { "id": "ladder-r6-v5-atlas-110", "build": "head" | "<commit>", "scene": "stitched-opt-v5",
//     "t0": 110, "t1": 122, "fps": 30, "size": [3840, 2160], "hud": ["off", "on"],
//     "params": { "mode": "live", "ground": "1" }, "campath": { "kind": "orbit", ... } | null,
//     "keys": { "maskAmt": [[296, 0], [298, 1]] }, "preroll": 1, "description": "..." }
// "head" draws with the working tree's viewer (viewer/dist, built first); a commit with that commit's viewer, built
// in a temporary git worktree. Scenes are proxied from a published host: RS_STUDIO, default http://127.0.0.1:8090
// (bun deploy/serve.ts; the author pointed it at his Mac Studio). Each HUD variant is its own
// file (<id>.mp4, <id>-hud.mp4): HEVC, CRF 15, 4:2:0, a keyframe a second (visually lossless). Every finished
// file is added to <out>/manifest.json at once, so cutting can start with the first shots.

import { spawn, spawnSync, type ChildProcess } from "node:child_process";
import { closeSync, existsSync, mkdirSync, openSync, readFileSync, renameSync, statSync, unlinkSync, writeFileSync } from "node:fs";
import { hostname } from "node:os";
import path from "node:path";

const HERE = path.dirname(new URL(import.meta.url).pathname);
const VIEWER = path.resolve(HERE, "..");
const REPO = path.resolve(VIEWER, "..");
const STUDIO = process.env.RS_STUDIO ?? "http://127.0.0.1:8090";
const OUT_DEFAULT = path.join(REPO, "film", "captures");
const WT = process.env.RS_WORKTREES ?? path.join(process.env.TMPDIR ?? "/tmp", "rs-capture-worktrees");
const MIN_FREE = Number(process.env.RS_CAPTURE_MIN_FREE_GB ?? 6) * 1e9;
const FMT = (process.env.RS_CAPTURE_FMT ?? "png") as "png" | "jpeg" | "webp";
const CHROME = process.env.RS_CHROME ?? "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";

type Shot = {
  id: string;
  build: string;
  scene: string;
  t0: number;
  t1: number;
  fps: number;
  size: [number, number];
  hud: ("off" | "on")[];
  params?: Record<string, string>;
  campath?: unknown;
  /** values keyed over drive time (eased between keys): "maskAmt" the clip's Predicted overlay, "dir" the
   *  director look's amount */
  keys?: Record<string, [number, number][]>;
  preroll?: number;
  /** drive seconds per output second (default 1): a timelapse */
  speed?: number;
  description: string;
  /** chase pose of the build ("today" or "authentic"), for the manifest */
  camera?: string;
};

// ---------------------------------------------------------------------------------------------- builds

function git(args: string[], cwd = REPO) {
  const r = spawnSync("git", args, { cwd, encoding: "utf8" });
  if (r.status !== 0) throw new Error(`git ${args.join(" ")}: ${r.stderr}`);
  return r.stdout.trim();
}

/** the dist folder that draws a build: the working tree's (built now) or a commit's (in a worktree); with
 *  RS_BUILDS (another machine, no repository) a prebuilt one, <RS_BUILDS>/<build>/dist and commit.txt, made with
 *  --export-builds here and copied over */
function ensureBuild(build: string): { dist: string; commit: string } {
  if (process.env.RS_BUILDS) {
    const dir = path.join(process.env.RS_BUILDS, build);
    return { dist: path.join(dir, "dist"), commit: readFileSync(path.join(dir, "commit.txt"), "utf8").trim() };
  }
  if (build === "head") {
    // the working tree, built into a folder of its own for this run (edits and rebuilds meanwhile change nothing)
    const head = git(["rev-parse", "--short", "HEAD"]);
    // the viewer's changes on top of HEAD, new files included (they decide the build as much as the diff does)
    const scope = ["viewer/src", "viewer/index.html", "viewer/public"];
    let diff = spawnSync("git", ["diff", "HEAD", "--", ...scope], { cwd: REPO, encoding: "utf8", maxBuffer: 1 << 28 }).stdout;
    for (const f of git(["ls-files", "--others", "--exclude-standard", "--", ...scope]).split("\n").filter(Boolean).sort())
      diff += `\n--- new file ${f}\n` + readFileSync(path.join(REPO, f), "utf8");
    const commit = diff ? `${head}+wt-${new Bun.CryptoHasher("sha1").update(diff).digest("hex").slice(0, 7)}` : head;
    // kept beside the captures, so a shot drawn by an uncommitted viewer can be traced exactly
    if (diff) {
      mkdirSync(path.join(OUT_DEFAULT, "builds"), { recursive: true });
      writeFileSync(path.join(OUT_DEFAULT, "builds", `${commit}.patch`), diff);
    }
    const dist = path.join(WT, "head-" + commit);
    if (!existsSync(path.join(dist, "index.html"))) {
      const r = spawnSync("bunx", ["vite", "build", "--logLevel", "error", "--outDir", dist, "--emptyOutDir"], { cwd: VIEWER, encoding: "utf8" });
      if (r.status !== 0) throw new Error("viewer build failed: " + r.stderr);
    }
    return { dist, commit };
  }
  const commit = git(["rev-parse", "--short", build]);
  const dir = path.join(WT, commit);
  const dist = path.join(dir, "viewer", "dist");
  if (existsSync(path.join(dist, "index.html"))) return { dist, commit };
  mkdirSync(WT, { recursive: true });
  if (!existsSync(dir)) git(["worktree", "add", "--detach", dir, commit]);
  const v = path.join(dir, "viewer");
  for (const [cmd, args] of [["bun", ["install", "--frozen-lockfile"]], ["bunx", ["vite", "build", "--logLevel", "error"]]] as [string, string[]][]) {
    const r = spawnSync(cmd, args, { cwd: v, encoding: "utf8" });
    if (r.status !== 0) {
      // older lockfiles: a plain install
      if (cmd === "bun") {
        const r2 = spawnSync("bun", ["install"], { cwd: v, encoding: "utf8" });
        if (r2.status !== 0) throw new Error(`${commit}: bun install failed: ${r2.stderr}`);
        continue;
      }
      throw new Error(`${commit}: ${cmd} ${args.join(" ")} failed: ${r.stderr}`);
    }
  }
  return { dist, commit };
}

// ---------------------------------------------------------------------------------------------- server

const TYPES: Record<string, string> = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".json": "application/json", ".glb": "model/gltf-binary", ".woff2": "font/woff2", ".png": "image/png", ".jpg": "image/jpeg", ".webp": "image/webp", ".svg": "image/svg+xml", ".mp4": "video/mp4" };

/** the build's files on 127.0.0.1, scenes proxied from the Studio */
function serve(dist: string) {
  const server = Bun.serve({
    port: 0,
    hostname: "127.0.0.1",
    idleTimeout: 120,
    async fetch(req) {
      const u = new URL(req.url);
      if (u.pathname.startsWith("/scenes")) {
        const h = new Headers();
        const range = req.headers.get("range");
        if (range) h.set("range", range);
        const r = await fetch(STUDIO + u.pathname + u.search, { headers: h });
        const out = new Headers();
        for (const k of ["content-type", "content-length", "content-range", "accept-ranges", "last-modified"]) {
          const v = r.headers.get(k);
          if (v) out.set(k, v);
        }
        out.set("cache-control", "no-cache");
        return new Response(r.body, { status: r.status, headers: out });
      }
      let p = path.join(dist, decodeURIComponent(u.pathname));
      if (!p.startsWith(dist) || !existsSync(p) || statSync(p).isDirectory()) p = path.join(dist, "index.html");
      return new Response(Bun.file(p), { headers: { "content-type": TYPES[path.extname(p)] ?? "application/octet-stream" } });
    },
  });
  return server;
}

// ---------------------------------------------------------------------------------------------- chrome / CDP

class CDP {
  private id = 0;
  private pending = new Map<number, { res: (v: any) => void; rej: (e: Error) => void }>();
  private listeners = new Map<string, ((params: any) => void)[]>();
  on(method: string, fn: (params: any) => void) {
    this.listeners.set(method, [...(this.listeners.get(method) ?? []), fn]);
  }
  constructor(private ws: WebSocket) {
    ws.onclose = (ev) => {
      for (const p of this.pending.values()) p.rej(new Error(`CDP closed (${ev.code} ${ev.reason})`));
      this.pending.clear();
    };
    ws.onmessage = (ev) => {
      const msg = JSON.parse(String(ev.data));
      if (msg.id && this.pending.has(msg.id)) {
        const p = this.pending.get(msg.id)!;
        this.pending.delete(msg.id);
        if (msg.error) p.rej(new Error(JSON.stringify(msg.error)));
        else p.res(msg.result);
      } else if (msg.method) for (const fn of this.listeners.get(msg.method) ?? []) fn(msg.params);
    };
  }
  static async connect(url: string) {
    const ws = new WebSocket(url);
    await new Promise<void>((res, rej) => {
      ws.onopen = () => res();
      ws.onerror = () => rej(new Error("CDP connect failed"));
    });
    return new CDP(ws);
  }
  send(method: string, params: object = {}, sessionId?: string): Promise<any> {
    const id = ++this.id;
    this.ws.send(JSON.stringify({ id, method, params, ...(sessionId ? { sessionId } : {}) }));
    return new Promise((res, rej) => this.pending.set(id, { res, rej }));
  }
  close() {
    this.ws.close();
  }
}

async function launchChrome(): Promise<{ proc: ChildProcess; cdp: CDP }> {
  const port = Number(process.env.RS_CDP_PORT) || 9300 + Math.floor(Math.random() * 500);
  const profile = path.join(process.env.TMPDIR ?? "/tmp", `rs-capture-chrome-${port}`);
  const proc = spawn(CHROME, [
    "--headless=new",
    `--remote-debugging-port=${port}`,
    "--remote-debugging-address=127.0.0.1",
    `--user-data-dir=${profile}`,
    "--use-angle=metal",
    "--enable-gpu",
    "--ignore-gpu-blocklist",
    "--enable-webgl",
    "--disable-background-timer-throttling",
    "--disable-renderer-backgrounding",
    "--disable-backgrounding-occluded-windows",
    "--autoplay-policy=no-user-gesture-required",
    "--mute-audio",
    "--hide-scrollbars",
    "--no-first-run",
    "--no-default-browser-check",
    "--window-size=1920,1080",
    "about:blank",
  ], { stdio: ["ignore", "ignore", "pipe"] });
  let ws = "";
  for (let i = 0; i < 100 && !ws; i++) {
    await Bun.sleep(150);
    try {
      ws = (await (await fetch(`http://127.0.0.1:${port}/json/version`)).json()).webSocketDebuggerUrl;
    } catch {}
  }
  if (!ws) throw new Error("chrome did not start");
  return { proc, cdp: await CDP.connect(ws) };
}

// in-page helper, generic over the viewer's builds: today's has window.__capture; older ones are driven through
// window.__rs (every build exposes the app there) with the render loop running and playback paused
const HELPER = `
window.__capHelper = {
  frame() { return new Promise((r) => { let done = false; requestAnimationFrame(() => { if (!done) { done = true; r(); } }); setTimeout(() => { if (!done) { done = true; r(); } }, 250); }); },
  // a splat sort counts once it has finished and started after the step's camera was in place (Spark sorts
  // continuously while it regenerates, so its own flags never go quiet)
  watchSort() {
    const sp = window.__rs && window.__rs.spark;
    if (!sp || sp.__capWrapped || !sp.driveSort) return;
    const orig = sp.driveSort.bind(sp);
    sp.driveSort = async function () {
      const before = sp.lastSortTime;
      await orig();
      if (sp.lastSortTime !== before) sp.__sortedAt = sp.lastSortTime;
    };
    sp.__capWrapped = true;
  },
  sortAfter: 0,
  sorted() {
    const sp = window.__rs && window.__rs.spark;
    return !sp || !sp.__capWrapped || (sp.__sortedAt || 0) > this.sortAfter;
  },
  state() {
    const app = window.__rs, cap = window.__capture;
    if (cap) {
      const s = cap.state();
      return { ...s, sorting: s.splats !== false && !this.sorted() };
    }
    const ts = app.tiles && app.tiles.stats;
    let unfaded = 0;
    if (app.tiles && app.tiles.live) for (const l of app.tiles.live.values()) if (l.U && l.U.tile && l.U.tile.value.z < 0.999) unfaded++;
    const sp = app.spark;
    const ready = app.tiles && app.tiles.ready ? app.tiles.ready.length : 0;
    return { T: app.T, tiles: ts ? ts.loading + ts.queued + ready : 0, unfaded, sorting: !this.sorted(),
      atlas: 0, movers: app.objects && app.objects.movers ? app.objects.movers.filter((m) => m.state === "loading").length : 0, kit: true, gen: 0, env: 0 };
  },
  busy(s) { return s.tiles > 0 || s.unfaded > 0 || s.sorting || s.atlas > 0 || s.movers > 0 || !s.kit || s.gen > 0 || s.env > 0; },
  async settle(minFrames, maxMs) {
    const t0 = performance.now();
    let ok = 0, n = 0, last;
    while (performance.now() - t0 < maxMs) {
      await this.frame();
      n++;
      last = this.state();
      if (this.busy(last)) ok = 0; else ok++;
      if (ok >= minFrames && n >= minFrames + 1) return { ok: true, frames: n, ms: Math.round(performance.now() - t0) };
    }
    return { ok: false, frames: n, ms: Math.round(performance.now() - t0), state: last };
  },
  async step(T, dt, keys) {
    const app = window.__rs, cap = window.__capture;
    this.watchSort();
    if (cap) {
      for (const k in keys || {}) cap.set ? cap.set(k, keys[k]) : k === "maskAmt" && cap.maskAmt(keys[k]);
      cap.step(T, dt);
    } else {
      if (app.playing) app.setPlaying(false);
      if (app.hud && app.hud.setPlaying) app.hud.setPlaying(true);
      if (!app.__capFrame) {
        // the frame's dt is the simulated step, once per step (0 while it settles), so the damped follow cameras
        // of the early builds lag behind the car as they do at 1x playback
        const orig = app.frame;
        app.frame = function (d, now) { const v = app.__vdt || 0; app.__vdt = 0; return orig.call(this, v, now); };
        app.__capFrame = true;
      }
      app.__vdt = dt;
      app.seek(T);
    }
    // the camera is in place two frames on; a sort must start after that
    await this.frame();
    await this.frame();
    this.sortAfter = performance.now();
    let r = await this.settle(3, 45000);
    if (cap) {
      // the car's surroundings from the settled frame, then the frame drawn with them
      cap.requestEnv();
      await this.frame();
      await this.frame();
      this.sortAfter = performance.now();
      const r2 = await this.settle(3, 10000);
      r = { ...r, ok: r.ok && r2.ok, frames: r.frames + r2.frames, ms: r.ms + r2.ms };
    }
    // the HUD updates on a 60 ms clock
    await new Promise((res) => setTimeout(res, 70));
    await this.frame();
    return r;
  },
  hud(on) {
    const cap = window.__capture, app = window.__rs;
    if (cap) cap.hud(on); else if (app.hud && app.hud.root) app.hud.root.style.visibility = on ? "visible" : "hidden";
  },
};`;

// ---------------------------------------------------------------------------------------------- encoding

/** RS_ENCODER: "x265" (default; CRF 15, preset medium), "vt" (the Mac's hardware HEVC, constant quality RS_VT_Q,
 *  default 80) or "png" (the frames themselves, into <file>.png/) */
const ENC = process.env.RS_ENCODER ?? "x265";
function encoder(file: string, fps: number) {
  const codec =
    ENC === "vt"
      ? ["-c:v", "hevc_videotoolbox", "-q:v", process.env.RS_VT_Q ?? "80", "-g", String(fps), "-pix_fmt", "yuv420p", "-tag:v", "hvc1", "-movflags", "+faststart", file]
      : ENC === "png"
        ? ["-c:v", "copy", "-f", "image2", (mkdirSync(file + ".png", { recursive: true }), path.join(file + ".png", "%05d.png"))]
        : ["-c:v", "libx265", "-preset", "medium", "-crf", "15", "-pix_fmt", "yuv420p", "-tag:v", "hvc1", "-x265-params", `keyint=${fps}:min-keyint=${fps}:log-level=error`, "-movflags", "+faststart", file];
  const ff = spawn("ffmpeg", ["-v", "error", "-y", "-f", "image2pipe", "-framerate", String(fps), "-c:v", FMT === "jpeg" ? "mjpeg" : FMT, "-i", "-", ...codec], { stdio: ["pipe", "inherit", "inherit"] });
  const done = new Promise<number>((res) => ff.on("close", (c) => res(c ?? 1)));
  return {
    async write(buf: Buffer) {
      if (!ff.stdin!.write(buf)) await new Promise((r) => ff.stdin!.once("drain", r));
    },
    async end() {
      ff.stdin!.end();
      return done;
    },
  };
}

function keyAt(keys: [number, number][] | undefined, T: number): number | null {
  if (!keys?.length) return null;
  if (T <= keys[0][0]) return keys[0][1];
  for (let i = 1; i < keys.length; i++) if (T <= keys[i][0]) {
    const [a, va] = keys[i - 1], [b, vb] = keys[i];
    const u = (T - a) / Math.max(1e-6, b - a);
    return va + (vb - va) * u * u * (3 - 2 * u);
  }
  return keys[keys.length - 1][1];
}

// ---------------------------------------------------------------------------------------------- manifest

function addToManifest(out: string, entry: Record<string, unknown>) {
  const lock = path.join(out, "manifest.lock");
  for (let i = 0; ; i++) {
    try {
      closeSync(openSync(lock, "wx"));
      break;
    } catch {
      // a lock older than 30 s is a crashed writer's
      try {
        if (Date.now() - statSync(lock).mtimeMs > 30000) unlinkSync(lock);
      } catch {}
      if (i > 400) throw new Error("manifest lock");
      Bun.sleepSync(50);
    }
  }
  try {
    writeManifest(out, entry);
  } finally {
    unlinkSync(lock);
  }
}
function writeManifest(out: string, entry: Record<string, unknown>) {
  const file = path.join(out, "manifest.json");
  const m = existsSync(file) ? JSON.parse(readFileSync(file, "utf8")) : { about: "Shot library for the film, captured frame by frame from the viewer (viewer/tools/capture.ts; viewer/README.md, Capture).", shots: [] };
  m.shots = (m.shots as { id: string }[]).filter((s) => s.id !== entry.id);
  m.shots.push(entry);
  m.updated = new Date().toISOString();
  writeFileSync(file + ".tmp", JSON.stringify(m, null, 1));
  renameSync(file + ".tmp", file);
}

// ---------------------------------------------------------------------------------------------- main

/** requests in flight (not media streams), and when the last one started or ended */
const net = { inflight: new Map<string, string>(), changed: 0 };
function watchNetwork(cdp: CDP) {
  cdp.on("Network.requestWillBeSent", (p) => {
    // what the page itself fetches (worker scripts never report finishing to the page, media streams stay open)
    if (!["Fetch", "XHR", "Image"].includes(p.type) || p.request?.url?.startsWith("data:")) return;
    net.inflight.set(p.requestId, `${p.type} ${p.request?.url}`);
    net.changed = performance.now();
  });
  const done = (p: { requestId: string }) => {
    if (net.inflight.delete(p.requestId)) net.changed = performance.now();
  };
  cdp.on("Network.loadingFinished", done);
  cdp.on("Network.loadingFailed", done);
}

async function captureShot(cdp: CDP, session: string, shot: Shot, base: string, commit: string, out: string) {
  const [W, H] = shot.size;
  const scale = W >= 3840 ? 2 : W >= 2560 ? 2 : 1;
  await cdp.send("Emulation.setDeviceMetricsOverride", { width: Math.round(W / scale), height: Math.round(H / scale), deviceScaleFactor: scale, mobile: false }, session);
  const pre = shot.preroll ?? 1;
  const q = new URLSearchParams({ scene: shot.scene, t: String(shot.t0 - pre), paused: "1", dpr: String(scale), capture: "1", ...(shot.params ?? {}) });
  if (shot.campath) q.set("campath", JSON.stringify(shot.campath));
  await cdp.send("Page.navigate", { url: `${base}/?${q}` }, session);
  // the scene loaded: drive and tiles (or a scene without tiles)
  for (let i = 0; i < 600; i++) {
    await Bun.sleep(200);
    const r = await cdp.send("Runtime.evaluate", { expression: "!!(window.__rs && window.__rs.drive && (window.__rs.tiles || window.__rs.sceneList))", returnByValue: true }, session).catch(() => null);
    if (r?.result?.value) break;
    if (i === 599) throw new Error(`${shot.id}: scene did not load`);
  }
  await cdp.send("Runtime.evaluate", { expression: HELPER }, session);
  const today = (await cdp.send("Runtime.evaluate", { expression: "!!window.__capture", returnByValue: true }, session)).result.value as boolean;
  if (process.env.RS_CAPTURE_DEBUG) console.log("  helper in", JSON.stringify((await cdp.send("Runtime.evaluate", { expression: "JSON.stringify({gl: (() => { const c = document.querySelector('canvas'); const g = c && c.getContext('webgl2'); const d = g && g.getExtension('WEBGL_debug_renderer_info'); return d ? g.getParameter(d.UNMASKED_RENDERER_WEBGL) : 'n/a'; })(), state: window.__capHelper.state() })", returnByValue: true }, session)).result.value));
  const files = shot.hud.map((h) => ({ hud: h, file: path.join(out, `${shot.id}${h === "on" ? "-hud" : ""}.mp4`) }));
  const enc = files.map((f) => encoder(f.file + ".part.mp4", shot.fps));
  const speed = shot.speed ?? 1;
  const n = Math.round(((shot.t1 - shot.t0) * shot.fps) / speed);
  const nPre = Math.round((pre * shot.fps) / speed);
  let worst = 0, unsettled = 0;
  const t0 = performance.now();
  for (let i = -nPre; i < n; i++) {
    const T = shot.t0 + (i * speed) / shot.fps;
    const kv: Record<string, number> = {};
    for (const [k, v] of Object.entries(shot.keys ?? {})) kv[k] = keyAt(v, T)!;
    const r = await cdp.send("Runtime.evaluate", { expression: `window.__capHelper.step(${T}, ${1 / shot.fps}, ${JSON.stringify(kv)})`, awaitPromise: true, returnByValue: true }, session);
    const st = r.result?.value;
    if (process.env.RS_CAPTURE_DEBUG && net.inflight.size) console.log("  inflight", [...net.inflight.values()].slice(0, 4).join(" | "));
    if (!today) {
      // the early builds: also until nothing has been in flight for 250 ms, then settled again
      for (let k = 0; k < 200 && (net.inflight.size > 0 || performance.now() - net.changed < 250); k++) {
        await Bun.sleep(50);
        if (net.inflight.size === 0 && performance.now() - net.changed >= 250) await cdp.send("Runtime.evaluate", { expression: "window.__capHelper.settle(3, 20000)", awaitPromise: true, returnByValue: true }, session);
      }
    }
    if (process.env.RS_CAPTURE_DEBUG) console.log("  step", i, T.toFixed(3), JSON.stringify(st ?? r.exceptionDetails ?? r).slice(0, 400));
    if (i < 0) continue;
    if (st && !st.ok) unsettled++;
    worst = Math.max(worst, st?.ms ?? 0);
    for (let k = 0; k < files.length; k++) {
      await cdp.send("Runtime.evaluate", { expression: `window.__capHelper.hud(${files[k].hud === "on"})` }, session);
      const ts = performance.now();
      const shotR = await cdp.send("Page.captureScreenshot", { format: FMT, ...(FMT === "jpeg" ? { quality: 100 } : {}), optimizeForSpeed: true, captureBeyondViewport: false }, session);
      const ts2 = performance.now();
      await enc[k].write(Buffer.from(shotR.data, "base64"));
      if (process.env.RS_CAPTURE_DEBUG) console.log("  shot", k, shotR.data.length, Math.round(ts2 - ts), "ms grab", Math.round(performance.now() - ts2), "ms pipe");
    }
    if (i % 30 === 0 || process.env.RS_CAPTURE_DEBUG) process.stdout.write(`  ${shot.id} frame ${i}/${n} T ${T.toFixed(2)} (${((performance.now() - t0) / 1000).toFixed(0)} s)\n`);
  }
  for (let k = 0; k < files.length; k++) {
    const code = await enc[k].end();
    if (code !== 0) throw new Error(`${shot.id}: ffmpeg exited ${code}`);
    if (ENC !== "png") renameSync(files[k].file + ".part.mp4", files[k].file);
    addToManifest(out, {
      id: `${shot.id}${files[k].hud === "on" ? "-hud" : ""}`,
      file: path.basename(files[k].file),
      scene: shot.scene,
      viewer_commit: commit,
      camera: shot.campath ?? `chase (${shot.camera ?? "the build's default"})`,
      hud: files[k].hud === "on",
      flags: shot.params ?? {},
      keys: shot.keys ?? undefined,
      t: [shot.t0, shot.t1],
      fps: shot.fps,
      speed: speed !== 1 ? speed : undefined,
      size: shot.size,
      frames: n,
      bytes: ENC === "png" ? 0 : statSync(files[k].file).size,
      encoder: ENC === "vt" ? `hevc_videotoolbox q ${process.env.RS_VT_Q ?? "80"}` : ENC === "png" ? "png frames" : "libx265 crf 15",
      // where the file is: RS_LOCATION (e.g. "anomastudio:~/dev/reality-splat/film/captures/") + its name
      location: process.env.RS_LOCATION ? process.env.RS_LOCATION + path.basename(files[k].file) : undefined,
      rendered_on: process.env.RS_MACHINE ?? hostname(),
      description: shot.description,
      unsettled_frames: unsettled,
      captured: new Date().toISOString(),
    });
  }
  return { secs: (performance.now() - t0) / 1000, worst, unsettled };
}

async function main() {
  // (a pending CDP reply alone does not keep Bun's event loop alive)
  const keepAlive = setInterval(() => {}, 1000);
  const args = process.argv.slice(2);
  const specFile = args.find((a) => !a.startsWith("--"));
  if (!specFile) throw new Error("usage: bun tools/capture.ts <shots.json> [--only id,...] [--out dir] [--force]");
  const opt = (k: string) => {
    const i = args.indexOf(k);
    return i >= 0 ? args[i + 1] : undefined;
  };
  const out = path.resolve(opt("--out") ?? path.join(REPO, "film", "captures"));
  mkdirSync(out, { recursive: true });
  const only = opt("--only")?.split(",");
  // --export-builds <dir>: build every build the shots use into <dir>/<build>/ (dist + commit.txt) and stop
  const exp = opt("--export-builds");
  if (exp) {
    for (const b of new Set((JSON.parse(readFileSync(specFile, "utf8")) as Shot[]).map((s) => s.build))) {
      const { dist, commit } = ensureBuild(b);
      mkdirSync(path.join(exp, b), { recursive: true });
      spawnSync("rsync", ["-a", "--delete", dist + "/", path.join(exp, b, "dist") + "/"]);
      writeFileSync(path.join(exp, b, "commit.txt"), commit + "\n");
      console.log(`${b}: ${commit}`);
    }
    process.exit(0);
  }
  const force = args.includes("--force");
  let shots = JSON.parse(readFileSync(specFile, "utf8")) as Shot[];
  if (only) shots = shots.filter((s) => only.includes(s.id));
  if (!force) shots = shots.filter((s) => !s.hud.every((h) => existsSync(path.join(out, `${s.id}${h === "on" ? "-hud" : ""}.mp4`))));
  const { proc, cdp } = await launchChrome();
  // stopped from outside: take the browser with it
  for (const sig of ["SIGTERM", "SIGINT", "SIGHUP"] as const) process.on(sig, () => {
    proc.kill();
    process.exit(143);
  });
  try {
    // the window's own tab (a second, background target gets no animation frames)
    const targets = (await cdp.send("Target.getTargets")).targetInfos as { targetId: string; type: string }[];
    const targetId = targets.find((t) => t.type === "page")?.targetId ?? (await cdp.send("Target.createTarget", { url: "about:blank" })).targetId;
    const { sessionId } = await cdp.send("Target.attachToTarget", { targetId, flatten: true });
    await cdp.send("Page.enable", {}, sessionId);
    await cdp.send("Runtime.enable", {}, sessionId);
    watchNetwork(cdp);
    await cdp.send("Network.enable", {}, sessionId);
    await cdp.send("Page.bringToFront", {}, sessionId);
    await cdp.send("Emulation.setFocusEmulationEnabled", { enabled: true }, sessionId);
    // every build first, so the working tree can change while the shots render
    const servers = new Map<string, { base: string; commit: string }>();
    for (const b of new Set(shots.map((s) => s.build))) {
      const { dist, commit } = ensureBuild(b);
      servers.set(b, { base: `http://127.0.0.1:${serve(dist).port}`, commit });
    }
    const failed: string[] = [];
    for (const shot of shots) {
      // the disk on this Mac is nearly full: stop well before it is
      const free = Number(spawnSync("df", ["-k", out], { encoding: "utf8" }).stdout.trim().split("\n").pop()!.split(/\s+/)[3]) * 1024;
      if (free < MIN_FREE) throw new Error(`only ${(free / 1e9).toFixed(1)} GB free under ${out}; stopping before ${shot.id}`);
      const sv = servers.get(shot.build)!;
      console.log(`${shot.id}: ${shot.scene} ${shot.t0}-${shot.t1} s @${shot.fps} ${shot.size.join("x")} (${sv.commit})`);
      let r;
      try {
        r = await captureShot(cdp, sessionId, shot, sv.base, sv.commit, out);
      } catch (e) {
        console.error(`  ${shot.id} FAILED: ${e instanceof Error ? e.message : e}`);
        failed.push(shot.id);
        continue;
      }
      console.log(`  done in ${r.secs.toFixed(0)} s (slowest frame ${r.worst} ms${r.unsettled ? `, ${r.unsettled} frames not fully settled` : ""})`);
    }
    if (failed.length) console.error(`failed: ${failed.join(", ")}`);
  } finally {
    clearInterval(keepAlive);
    cdp.close();
    proc.kill();
  }
  process.exit(0);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
