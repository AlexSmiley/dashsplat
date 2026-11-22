// Generative playback: a pre-rendered (generated) video of the default chase view is shown in place of the
// splat render while the camera is at that pose and T is inside a clip; the car model, its shadow and the path
// ribbon are drawn over it. The hand-over is a 0.25 s cross-fade: it finishes exactly at a clip's last frame
// (so the video is still playing while it fades), starts at its first, and follows the camera when the user
// orbits away from the chase pose and back (the video's opacity falls with the angle, so it never slides). The
// video is kept on the timeline at any playback rate and when scrubbing, measured on the frames actually
// presented (requestVideoFrameCallback). A clip may pack a per-pixel provenance mask under the picture, shown
// by the "Predicted" overlay. Format: README, "Generative mode".

import * as THREE from "three";
import { GRADE_GLSL } from "./splatShader";

export type GeneratedClip = {
  /** mp4 (H.264), relative to the scene folder */
  file: string;
  /** drive time of the first and last frame (s) */
  t0: number;
  t1: number;
  fps?: number;
  /** camera the clip was rendered from; only "chase" (the default chase pose) is shown */
  pose?: string;
  /** vertical field of view the clip was rendered with (degrees, default 50 = the chase camera's) */
  vfov?: number;
  /** "free": the ego car's region is left for the viewer's car model (default); "generated": the video has it */
  ego?: "free" | "generated";
  /** "below": every frame is twice the picture's height, the provenance mask under the picture (grey codes:
   *  255 observed, 170 repaired, 85 filled, 0 none) */
  provenance?: "below";
  /** "overlay": the clip has no vehicles; the viewer draws every vehicle over it (clean models) */
  vehicles?: "overlay";
};
export type GeneratedJson = { clips: GeneratedClip[] };
export type Overlay = "auto" | "car" | "path" | "none";

type Clip = GeneratedClip & { url?: string; loading?: boolean };

/** cross-fade (s of wall time); the slower one where a clip ends and the splat view takes over */
const FADE = 0.25;
const FADE_OUT = 1.0;

const vert = /* glsl */ `
uniform vec2 rect;
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = vec4(position.xy * rect, 0.0, 1.0);
}`;
const frag = /* glsl */ `
uniform sampler2D map;
uniform float opacity;
uniform float feather;
/** x 1 = mask packed below, y overlay amount, z half a texel (v) */
uniform vec3 packing;
/** the same Natural grade as the splats (one grade for both modes) */
uniform vec4 gradeA;
uniform vec4 gradeB;
varying vec2 vUv;
${GRADE_GLSL}
void main() {
  vec2 uv = vUv;
  // a packed clip: the picture is the top half (v 0.5..1 once the video texture is flipped), the mask below;
  // half a texel kept clear of the seam so neither bleeds into the other
  vec2 pu = packing.x > 0.5 ? vec2(uv.x, clamp(0.5 + 0.5 * uv.y, 0.5 + packing.z, 1.0)) : uv;
  // display-referred video, graded exactly as the splats are
  vec3 c = texture2D(map, pu).rgb;
  if (gradeA.w > 0.0) c = rsGrade(c, gradeA, gradeB);
  if (packing.x > 0.5 && packing.y > 0.0) {
    float m = texture2D(map, vec2(uv.x, clamp(0.5 * uv.y, 0.0, 0.5 - packing.z))).r;
    float filled = smoothstep(0.17, 0.25, m) * (1.0 - smoothstep(0.42, 0.58, m));
    float repaired = smoothstep(0.42, 0.58, m) * (1.0 - smoothstep(0.75, 0.9, m));
    // a quiet violet wash: filled (never observed) clearly, repaired lightly
    vec3 tint = vec3(0.56, 0.5, 1.0);
    float k = packing.y * (0.42 * filled + 0.18 * repaired);
    c = mix(c, mix(c * 0.7, tint, 0.55), k);
  }
  float a = opacity;
  if (feather > 0.0) a *= smoothstep(0.0, feather, vUv.x) * smoothstep(0.0, feather, 1.0 - vUv.x);
  gl_FragColor = vec4(c, a);
}`;

type Player = { video: HTMLVideoElement; texture: THREE.VideoTexture; clip: Clip | null; presented: { media: number; at: number } };

export class Generative {
  private readonly players: [Player, Player];
  private active = 0;
  private readonly mat: THREE.ShaderMaterial;
  private readonly scene = new THREE.Scene();
  private readonly cam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  private readonly clips: Clip[];
  /** 0 = splats, 1 = video (eased in time), and the camera's alignment with the clip's pose (0..1) */
  fade = 0;
  align = 1;
  /** last sync: seconds into the clip the presented frame shows and should show */
  sync = { clip: -1, want: 0 };
  /** the "Predicted" overlay (provenance mask), eased */
  predicted = false;
  private maskAmt = 0;
  /** the fade under way is the clip's end handing over to the splats (the slow one) */
  private ending = false;
  /** capture: the fade as a function of drive time (as it plays at 1x) instead of eased in wall time, and the
   *  provenance overlay's amount set from outside */
  timeFade = false;
  maskOverride: number | null = null;

  constructor(private readonly base: string, json: GeneratedJson) {
    this.clips = (json.clips ?? []).filter((c) => c.file && c.t1 > c.t0 && (c.pose ?? "chase") === "chase").sort((a, b) => a.t0 - b.t0);
    const mk = (): Player => {
      const v = document.createElement("video");
      v.muted = true;
      v.playsInline = true;
      v.preload = "auto";
      v.crossOrigin = "anonymous";
      const texture = new THREE.VideoTexture(v);
      texture.colorSpace = THREE.NoColorSpace;
      texture.minFilter = texture.magFilter = THREE.LinearFilter;
      texture.generateMipmaps = false;
      const p: Player = { video: v, texture, clip: null, presented: { media: -1, at: 0 } };
      const onFrame = (_: number, meta: { mediaTime: number }) => {
        p.presented = { media: meta.mediaTime, at: performance.now() };
        v.requestVideoFrameCallback(onFrame);
      };
      if ("requestVideoFrameCallback" in v) v.requestVideoFrameCallback(onFrame);
      return p;
    };
    this.players = [mk(), mk()];
    this.mat = new THREE.ShaderMaterial({
      vertexShader: vert,
      fragmentShader: frag,
      uniforms: { map: { value: this.players[0].texture }, opacity: { value: 0 }, feather: { value: 0 }, rect: { value: new THREE.Vector2(1, 1) }, packing: { value: new THREE.Vector3() }, gradeA: { value: new THREE.Vector4(1, 1, 1, 0) }, gradeB: { value: new THREE.Vector4(1, 0, 1, 0) } },
      transparent: true,
      depthTest: false,
      depthWrite: false,
    });
    const quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this.mat);
    quad.frustumCulled = false;
    this.scene.add(quad);
  }

  /** the video element of the clip on screen */
  get video() {
    return this.players[this.active].video;
  }
  private get current() {
    return this.players[this.active].clip;
  }

  get count() {
    return this.clips.length;
  }
  get ranges(): [number, number][] {
    return this.clips.map((c) => [c.t0, c.t1]);
  }
  /** the clip on screen (or fading), if any */
  get clip(): GeneratedClip | null {
    return this.fade > 0 ? this.current : null;
  }
  /** whether a clip covers drive time T */
  covering(T: number) {
    return !!this.clipAt(T);
  }
  /** whether the clip on screen wants the vehicles drawn over it */
  get vehiclesOverlay() {
    return this.current?.vehicles === "overlay";
  }
  /** whether the clip on screen carries a provenance mask */
  get hasMask() {
    return this.current?.provenance === "below";
  }

  /** the clip at T: half-open [t0, t1), so at a join the next clip already counts; the last frame of the last */
  private clipAt(T: number) {
    return this.clips.find((c) => T >= c.t0 && T < c.t1) ?? this.clips.find((c) => T >= c.t0 && T <= c.t1) ?? null;
  }
  /** the clip that starts where this one ends (within a frame), for a seamless join */
  private nextOf(c: Clip) {
    return this.clips.find((n) => n !== c && Math.abs(n.t0 - c.t1) <= 1.5 / (c.fps ?? 30)) ?? null;
  }

  /** clips are fetched whole into memory (a blob seeks instantly and needs no byte ranges from the server) when
   *  T comes within 20 s of them, and released when it is far away */
  private prefetch(T: number) {
    for (const c of this.clips) {
      const near = T > c.t0 - 20 && T < c.t1 + 10;
      if (near && !c.url && !c.loading) {
        c.loading = true;
        fetch(new URL(c.file, new URL(this.base, location.href)).href)
          .then((r) => (r.ok ? r.arrayBuffer() : Promise.reject(new Error(`${c.file}: ${r.status}`))))
          .then((b) => (c.url = URL.createObjectURL(new Blob([b], { type: "video/mp4" }))))
          .catch((e) => console.warn("generated clip failed", e))
          .finally(() => (c.loading = false));
      } else if (!near && c.url && !this.players.some((p) => p.clip === c) && (T < c.t0 - 60 || T > c.t1 + 60)) {
        URL.revokeObjectURL(c.url);
        c.url = undefined;
      }
    }
  }

  private load(p: Player, c: Clip) {
    p.clip = c;
    p.video.src = c.url!;
    p.presented = { media: -1, at: 0 };
    p.video.currentTime = 0.5 / (c.fps ?? 30);
  }

  /** per frame: keeps the video on the timeline and eases the cross-fade; eligible = Generative mode in the
   *  chase view by day; align = how closely the camera sits at the clip's pose (1 there, 0 from about 2 degrees
   *  of orbit); returns the opacity the clip is drawn with */
  update(dt: number, T: number, playing: boolean, rate: number, eligible: boolean, align: number, camFov: number): number {
    this.prefetch(T);
    const c = this.clipAt(T);
    let P = this.players[this.active];
    const Q = this.players[1 - this.active];
    if (c?.url && P.clip !== c) {
      if (Q.clip === c && Q.video.readyState >= 2) {
        // a join: the next clip is cued in the other player; switch to it in the same frame, no fade
        P.video.pause();
        this.active = 1 - this.active;
        P = Q;
      } else if (this.fade === 0) this.load(P, c);
    }
    // cue the clip that follows on without a gap, a few seconds ahead
    const next = P.clip ? this.nextOf(P.clip) : null;
    const other = this.players[1 - this.active];
    if (next?.url && other.clip !== next && P.clip && T > P.clip.t1 - 4) this.load(other, next);
    // and set it rolling a moment before the join, so it is already moving on its first frame when it takes over
    if (next && other.clip === next && P.clip && playing && rate === 1 && T > P.clip.t1 - 0.1 && other.video.paused && other.video.readyState >= 2) {
      other.video.playbackRate = 1;
      other.video.play().catch(() => {});
    }
    const v = P.video;
    let want = false;
    this.ending = false;
    if (c && c === P.clip && v.readyState >= 1) {
      const fps = c.fps ?? 30;
      const target = Math.min(Math.max(0, T - c.t0), Math.max(0, (v.duration || c.t1 - c.t0) - 0.5 / fps));
      const shown = this.shownTime(P, playing);
      this.keep(P, target, shown, playing, rate, fps);
      this.sync = { clip: shown, want: target };
      // where a clip ends and the splats take over: a one-second fade that ends on the clip's last frame (none
      // when another clip follows on)
      const end = Math.min(c.t1, v.duration ? c.t0 + v.duration : c.t1);
      const tail = next ? Infinity : end - FADE_OUT * Math.max(1, rate);
      this.ending = T >= tail;
      want = eligible && align > 0.02 && v.readyState >= 2 && Math.abs((c.vfov ?? 50) - camFov) < 0.5 && Math.abs(shown - target) < 0.25 + 0.1 * rate && !this.ending;
    } else if (c && other.clip === c) {
      // the join's first frame, before the cued clip is ready to take over: hold (no dip)
      want = eligible && align > 0.02;
    } else if (!v.paused) v.pause();
    this.fade = Math.min(1, Math.max(0, this.fade + (want ? dt / FADE : -dt / (this.ending ? FADE_OUT : FADE))));
    if (this.timeFade) {
      // in from the first frame of a chain of clips, out over the last FADE_OUT s of it, as at 1x playback
      const cc = c && (c === P.clip || other.clip === c) ? c : null;
      if (cc && eligible && align > 0.02) {
        const fps = cc.fps ?? 30;
        let first = cc;
        for (let prev = this.clips.find((q) => q !== first && Math.abs(q.t1 - first.t0) <= 1.5 / fps); prev; prev = this.clips.find((q) => q !== first && Math.abs(q.t1 - first.t0) <= 1.5 / fps)) first = prev;
        let last = cc;
        for (let nx = this.nextOf(last); nx; nx = this.nextOf(last)) last = nx;
        const fin = THREE.MathUtils.clamp((T - first.t0) / FADE, 0, 1);
        const fout = THREE.MathUtils.clamp((last.t1 - T) / FADE_OUT, 0, 1);
        this.fade = Math.min(fin, fout);
      } else this.fade = 0;
    }
    this.align = align;
    this.maskAmt += ((this.predicted && this.hasMask ? 1 : 0) - this.maskAmt) * Math.min(1, dt / 0.15);
    if (this.maskOverride !== null) this.maskAmt = this.hasMask ? this.maskOverride : 0;
    if (this.fade === 0 && !c && P.clip) {
      v.pause();
      P.clip = null;
    }
    this.mat.uniforms.map.value = P.texture;
    return this.opacity;
  }

  /** what the clip adds on screen: the eased fade times the camera's alignment */
  get opacity() {
    const f = this.fade * this.fade * (3 - 2 * this.fade);
    return f * this.align;
  }

  /** seconds into the clip of the frame on screen: the last presented frame advanced by the time since, while
   *  playing (currentTime runs ahead of what is shown); currentTime when paused or before the first callback */
  private shownTime(P: Player, playing: boolean) {
    const v = P.video;
    const p = P.presented;
    if (p.media < 0 || v.seeking || !playing || v.paused) return v.currentTime;
    return p.media + ((performance.now() - p.at) / 1000) * v.playbackRate;
  }

  private keep(P: Player, target: number, shown: number, playing: boolean, rate: number, fps: number) {
    const v = P.video;
    if (!playing) {
      if (!v.paused) v.pause();
      // paused or scrubbing: the exact frame (its middle, so rounding never shows the neighbour)
      const want = (Math.floor(target * fps) + 0.5) / fps;
      if (!v.seeking && Math.abs(v.currentTime - want) > 0.25 / fps) v.currentTime = want;
      return;
    }
    const d = shown - target;
    // a jump (scrub, skip, rate change): seek; small drift: nudge the playback rate
    if (Math.abs(d) > 0.25 + 0.05 * rate) {
      if (!v.seeking) v.currentTime = target;
      return;
    }
    // within a frame: the plain rate; beyond, nudged up to 15 % to close the gap within about a second
    const r = Math.abs(d) < 0.5 / fps ? rate : Math.min(16, Math.max(0.0625, rate * (1 - Math.min(0.15, Math.max(-0.15, d * 0.8)))));
    if (Math.abs(v.playbackRate - r) > 0.002) v.playbackRate = r;
    if (v.paused && !v.seeking) v.play().catch(() => {});
  }

  setGrade(a: THREE.Vector4, b: THREE.Vector4) {
    (this.mat.uniforms.gradeA.value as THREE.Vector4).copy(a);
    (this.mat.uniforms.gradeB.value as THREE.Vector4).copy(b);
  }

  /** whether the car model should be drawn over the clip */
  carOver(overlay: Overlay) {
    return overlay === "car" || (overlay === "auto" && (this.current?.ego ?? "free") !== "generated");
  }

  /** where the clip lies on screen: the chase camera's vertical field of view, centred; width = its aspect */
  private rect(viewAspect: number, camFov: number) {
    const v = this.video;
    const va = v.videoWidth && v.videoHeight ? v.videoWidth / (v.videoHeight / (this.hasMask ? 2 : 1)) : 1.6;
    const D2R = Math.PI / 180;
    const h = Math.tan(((this.current?.vfov ?? 50) * D2R) / 2) / Math.tan((camFov * D2R) / 2);
    return { w: (h * va) / viewAspect, h };
  }
  /** the clip covers the whole view (so the splats need not be drawn under it) */
  covers(viewAspect: number, camFov: number) {
    const r = this.rect(viewAspect, camFov);
    return r.w >= 0.999 && r.h >= 0.999;
  }

  draw(renderer: THREE.WebGLRenderer, viewAspect: number, camFov: number) {
    if (this.fade <= 0 || !this.current) return;
    const { w, h } = this.rect(viewAspect, camFov);
    const u = this.mat.uniforms;
    (u.rect.value as THREE.Vector2).set(w, h);
    // a narrower clip than the view blends into the splats at its sides
    u.feather.value = w < 0.999 ? 0.04 : 0;
    u.opacity.value = this.opacity;
    (u.packing.value as THREE.Vector3).set(this.hasMask ? 1 : 0, this.maskAmt, this.video.videoHeight ? 1 / this.video.videoHeight : 0.001);
    renderer.render(this.scene, this.cam);
  }

  dispose() {
    for (const p of this.players) {
      p.video.pause();
      p.video.removeAttribute("src");
      p.video.load();
      p.texture.dispose();
    }
    for (const c of this.clips) if (c.url) URL.revokeObjectURL(c.url);
    this.mat.dispose();
  }
}
