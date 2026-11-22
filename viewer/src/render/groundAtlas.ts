// Ground texture atlas (pipeline/groundtex/FORMAT.md, "reality-ground" v1): the road surface as a sharp texture
// (1-2 cm per texel) on a height-grid mesh, 10.24 m tiles streamed around the focus. Drawn before the splats: an
// opaque pass writes depth where the atlas is opaque (alpha >= 0.98), a blended pass adds its soft edges. Splats
// standing on it (kerbs, grass relief, objects, movers) depth-test against it; ground-layer splats give way where it
// is opaque (a coverage map the splat shader reads), so the two never z-fight, and come back through transparent
// texels, while tiles stream and outside the atlas.
//
// Textures: decoded off-thread with alpha premultiplied, so the mip chain filters premultiplied colour (no dark or
// light halos at alpha edges); trilinear with the device's maximum anisotropy. Tiles near the camera use the full
// texture, farther ones a quarter-size decode (the GPU would pick those mips anyway).

import * as THREE from "three";
import type { GroundFilter } from "../scene/tileWorker";
import { ATLAS_COV_N, GRADE_GLSL, MOVER_SHADOW_GLSL, NIGHT_GLSL, type SplatGlobals } from "./splatShader";

type TileJson = {
  id: string;
  bounds: [number, number, number, number];
  pixels: number;
  gutter: number;
  texelSize: number;
  texture: string;
  bytes?: number;
  /** lossless grey PNG, same raster as the texture: 0 unsupported, 127 predicted, 255 observed */
  provenance?: string;
  grid: { rows: number; cols: number; z: number[] };
};
export type GroundJson = {
  format: string;
  version: number;
  colour?: string;
  alpha?: string;
  hideGroundSplats?: { layer: string; opaqueAlpha: number; heightTolerance: number };
  provenance?: { kind?: string; values?: { predicted?: number } };
  tiles: TileJson[];
};

/** metres from the camera within which a tile uses its full-resolution texture */
const NEAR = 45;
/** one texture upload (with its mip chain) every this many frames: with the outer ring about 170 tiles are
 *  resident and a recentre brings in a few dozen at once */
const UP_EVERY = 2;
/** coverage cells per tile edge (0.32 m each) */
const CELLS = 32;
const CELL = 0.32;

type Res = {
  t: TileJson;
  cx: number;
  cy: number;
  blob?: Blob;
  fetching?: boolean;
  /** decoded level in flight / uploaded: 0 full, 2 quarter */
  decoding?: number;
  level?: number;
  bitmap?: ImageBitmap;
  bitmapLevel?: number;
  tex?: THREE.Texture;
  /** the "Predicted" overlay: predicted share times coverage, at a quarter of the texture's size, filtered */
  prov?: THREE.DataTexture;
  provLoading?: boolean;
  opaque?: THREE.Mesh;
  soft?: THREE.Mesh;
  /** coverage cells (CELLS x CELLS): opaque share, mesh height; and as RGBA half floats, rows ready to copy */
  cov?: Float32Array;
  covHalf?: Uint16Array;
  last: number;
};

const ZERO = (() => {
  const t = new THREE.DataTexture(new Uint8Array(4), 1, 1);
  t.needsUpdate = true;
  return t;
})();

const vert = /* glsl */ `
varying vec2 vUv;
varying vec3 vW;
void main() {
  vUv = uv;
  vec4 w = modelMatrix * vec4(position, 1.0);
  vW = w.xyz;
  gl_Position = projectionMatrix * viewMatrix * w;
}`;

const frag = /* glsl */ `
uniform sampler2D map;
uniform sampler2D prov;
uniform float provAmt;
uniform float pass;
uniform float opacity;
uniform vec4 GR;
uniform vec4 G2;
uniform vec4 P;
uniform vec4 F;
uniform vec4 HC;
uniform vec3 fogColor;
uniform vec4 car;
uniform vec4 lights;
uniform sampler2D moverTex;
uniform sampler2D contactAo;
uniform vec4 moverInfo;
varying vec2 vUv;
varying vec3 vW;
${GRADE_GLSL}
${MOVER_SHADOW_GLSL}
${NIGHT_GLSL}
void main() {
  vec4 s = texture2D(map, vUv);
  float a = s.a;
  // opaque pass: the confirmed interior, with depth; soft pass: partial coverage, blended, without depth
  if (pass < 0.5) { if (a < 0.98) discard; }
  else { if (a >= 0.98 || a < 0.004) discard; }
  vec3 rgb = s.rgb / max(a, 1e-4);
  // "Predicted": the predicted share of the coverage here (weights filtered times coverage, divided back)
  float pk = 0.0;
  if (provAmt > 0.0) pk = provAmt * clamp(texture2D(prov, vUv).r / max(a, 0.02), 0.0, 1.0);
  // the splats' colour path, once: Natural grade, movers' shadows, night, haze
  if (GR.w > 0.0) rgb = rsGrade(rgb, GR, G2);
  if (moverInfo.x > 0.5) rgb *= rsMoverShadow(moverTex, contactAo, moverInfo.x, vW);
  if (P.y > 0.0) rgb = rsNight(rgb, vW, car, lights, P.y);
  if (pk > 0.0) rgb = mix(rgb, mix(rgb * 0.7, vec3(0.56, 0.5, 1.0), 0.55), 0.5 * pk);
  float dH = length(vW.xy - HC.xy);
  float hz = smoothstep(F.x, F.y, dH);
  float hk = F.z * hz * (2.0 - hz);
  rgb = mix(rgb, fogColor, hk * (1.0 - F.w));
  float k = opacity * (1.0 - hk * F.w) * (1.0 - smoothstep(F.y - 4.0, F.y + 10.0, dH));
  if (pass < 0.5) {
    // the opaque pass may not blend (it writes depth): fade it towards the haze instead when it thins out
    rgb = mix(fogColor, rgb, k);
    gl_FragColor = vec4(rgb, 1.0);
  } else {
    a *= k;
    gl_FragColor = vec4(rgb * a, a);
  }
}`;

export class GroundAtlas {
  readonly group = new THREE.Group();
  private readonly res = new Map<string, Res>();
  private readonly mat: { opaque: THREE.ShaderMaterial; soft: THREE.ShaderMaterial };
  private readonly uniforms: Record<string, THREE.IUniform>;
  private inflight = 0;
  private frame = 0;
  private covOrigin = new THREE.Vector2(NaN, NaN);
  private covDirty = true;
  /** the coverage map's origin moved: rewrite it whole */
  private covRecentred = true;
  /** tiles written into the coverage map, at their cell offset */
  private covPlaced = new Map<Res, [number, number]>();
  private covWritten = 0;
  private covTile = new THREE.DataTexture(new Uint16Array(CELLS * CELLS * 4), CELLS, CELLS, THREE.RGBAFormat, THREE.HalfFloatType);
  readonly tolerance: number;
  /** coverage cells of every atlas tile read so far (kept when the tile itself is dropped; for the ground
   *  filter), least recently used first */
  private covCache = new Map<string, Float32Array>();
  /** tiles whose coverage the ground filter is waiting for */
  private covWant = new Set<Res>();
  /** loaded / uploaded / pending, for the stats */
  stats = { tiles: 0, full: 0, pending: 0 };
  /** still to land for what is wanted now: decoded textures not yet uploaded, the coverage map not yet rewritten,
   *  provenance masks loading (capture: a frame is final only at 0) */
  unsettled = 0;
  /** director looks: a tile's presence (0..1) by drive time, as its opacity (the opaque pass fades towards the
   *  haze colour, so over a background of that colour it reads as a fade) */
  presence?: (bounds: [number, number, number, number]) => number;
  /** capture: the "Predicted" overlay's amount set from outside */
  provOverride: number | null = null;

  constructor(private readonly base: string, readonly json: GroundJson, private readonly G: SplatGlobals, private readonly anisotropy: number) {
    this.group.name = "ground-atlas";
    this.tolerance = json.hideGroundSplats?.heightTolerance ?? 0.2;
    for (const t of json.tiles) {
      const [x0, y0, x1, y1] = t.bounds;
      this.res.set(t.id, { t, cx: (x0 + x1) / 2, cy: (y0 + y1) / 2, last: 0 });
    }
    this.uniforms = {
      map: { value: null },
      prov: { value: ZERO },
      provAmt: { value: 0 },
      pass: { value: 0 },
      opacity: { value: 1 },
      GR: { value: G.grade.value },
      G2: { value: G.grade2.value },
      P: { value: G.params.value },
      F: { value: G.fog.value },
      HC: { value: G.hazeCentre.value },
      fogColor: { value: G.fogColor.value },
      car: { value: G.car.value },
      lights: { value: G.lights.value },
      moverTex: { value: G.moverTexture },
      contactAo: { value: G.contactAo.value },
      moverInfo: { value: G.moverInfo.value },
    };
    const common = { vertexShader: vert, fragmentShader: frag, transparent: true };
    this.mat = {
      opaque: new THREE.ShaderMaterial({ ...common, uniforms: { ...this.uniforms, pass: { value: 0 } }, depthWrite: true, blending: THREE.NoBlending }),
      soft: new THREE.ShaderMaterial({
        ...common,
        uniforms: { ...this.uniforms, pass: { value: 1 } },
        depthWrite: false,
        blending: THREE.CustomBlending,
        blendSrc: THREE.OneFactor,
        blendDst: THREE.OneMinusSrcAlphaFactor,
        blendSrcAlpha: THREE.OneFactor,
        blendDstAlpha: THREE.OneMinusSrcAlphaFactor,
      }),
    };
    this.covTile.minFilter = this.covTile.magFilter = THREE.NearestFilter;
    this.covTile.generateMipmaps = false;
  }

  static async load(base: string, G: SplatGlobals, anisotropy: number) {
    try {
      const r = await fetch(base + "ground/ground.json");
      if (!r.ok) return null;
      const j = (await r.json()) as GroundJson;
      if (j.format !== "reality-ground" || j.version !== 1 || !j.tiles?.length) return null;
      return new GroundAtlas(base + "ground/", j, G, anisotropy);
    } catch {
      return null;
    }
  }


  private geometry(t: TileJson) {
    const { rows, cols, z } = t.grid;
    const [x0, y0, x1, y1] = t.bounds;
    const pos = new Float32Array(rows * cols * 3);
    const uv = new Float32Array(rows * cols * 2);
    const total = t.pixels + 2 * t.gutter;
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
      const s = c / (cols - 1), q = r / (rows - 1);
      const i = r * cols + c;
      pos[i * 3] = s * (x1 - x0);
      pos[i * 3 + 1] = q * (y1 - y0);
      pos[i * 3 + 2] = z[i];
      // row 0 of the raster faces south; the bitmap is uploaded without a flip
      uv[i * 2] = (t.gutter + t.pixels * s) / total;
      uv[i * 2 + 1] = (t.gutter + t.pixels * q) / total;
    }
    const idx: number[] = [];
    for (let r = 0; r + 1 < rows; r++) for (let c = 0; c + 1 < cols; c++) {
      const sw = r * cols + c, se = sw + 1, nw = sw + cols, ne = nw + 1;
      // split SW-NE, upward winding
      idx.push(sw, se, ne, sw, ne, nw);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    g.setAttribute("uv", new THREE.BufferAttribute(uv, 2));
    g.setIndex(idx);
    g.computeBoundingSphere();
    return g;
  }

  private adoptCoverage(r: Res, c: Float32Array) {
    r.cov = c;
    const hf = new Uint16Array(CELLS * CELLS * 4);
    for (let q = 0; q < CELLS * CELLS; q++) {
      hf[q * 4] = THREE.DataUtils.toHalfFloat(c[q * 2]);
      hf[q * 4 + 1] = THREE.DataUtils.toHalfFloat(c[q * 2 + 1]);
    }
    r.covHalf = hf;
  }

  /** the ground filter for a splat tile's footprint (render cells of 0.32 m): cells whose atlas texels are all
   *  opaque, with the mesh height; null with the coverage of an overlapping atlas tile still to be read (it is
   *  requested) */
  filterGrid(x0: number, y0: number, x1: number, y1: number): GroundFilter | null {
    const T = 10.24;
    const tx0 = Math.floor(x0 / T), ty0 = Math.floor(y0 / T), tx1 = Math.floor(x1 / T), ty1 = Math.floor(y1 / T);
    const nx = (tx1 - tx0 + 1) * CELLS, ny = (ty1 - ty0 + 1) * CELLS;
    const op = new Uint8Array(nx * ny), z = new Float32Array(nx * ny);
    let missing = false;
    for (let ty = ty0; ty <= ty1; ty++) for (let tx = tx0; tx <= tx1; tx++) {
      const r = this.res.get(`${tx}_${ty}`);
      if (!r) continue;
      const c = this.covCache.get(r.t.id);
      if (!c) {
        missing = true;
        this.covWant.add(r);
        continue;
      }
      const ox = (tx - tx0) * CELLS, oy = (ty - ty0) * CELLS;
      for (let y = 0; y < CELLS; y++) for (let x = 0; x < CELLS; x++) {
        const k = (y * CELLS + x) * 2, o = (oy + y) * nx + ox + x;
        op[o] = c[k] >= 0.999 ? 1 : 0;
        z[o] = c[k + 1];
      }
    }
    if (missing) return null;
    return { x0: tx0 * T, y0: ty0 * T, cell: CELL, nx, ny, op, z, tol: this.tolerance };
  }

  /** read a tile's coverage without drawing it (for the ground filter) */
  private requestCoverage(r: Res) {
    if (r.fetching || r.decoding !== undefined || this.inflight >= 6) return false;
    const go = async () => {
      this.inflight++;
      r.fetching = true;
      try {
        const resp = await fetch(new URL(r.t.texture, new URL(this.base, location.href)).href);
        if (!resp.ok) throw new Error(`${r.t.texture}: ${resp.status}`);
        const blob = await resp.blob();
        const total = r.t.pixels + 2 * r.t.gutter;
        const size = Math.max(16, Math.round(total / 4));
        const bm = await createImageBitmap(blob, { premultiplyAlpha: "premultiply", colorSpaceConversion: "none", resizeWidth: size, resizeHeight: size, resizeQuality: "high" });
        this.coverage(r, bm);
        bm.close();
        // not drawn: keep only the cells
        if (!r.opaque) r.cov = r.covHalf = undefined;
      } catch (e) {
        console.warn("ground atlas coverage failed", e);
        this.covCache.set(r.t.id, new Float32Array(CELLS * CELLS * 2));
      } finally {
        r.fetching = false;
        this.inflight--;
      }
    };
    void go();
    return true;
  }

  /** height of the atlas mesh at a point, if a tile there is resident and drawn */
  surfaceAt(x: number, y: number): number | null {
    if (!this.group.visible) return null;
    const id = `${Math.floor(x / 10.24)}_${Math.floor(y / 10.24)}`;
    const r = this.res.get(id);
    if (!r?.opaque) return null;
    return this.heightAt(r.t, x, y);
  }

  private heightAt(t: TileJson, x: number, y: number) {
    const { rows, cols, z } = t.grid;
    const [x0, y0, x1, y1] = t.bounds;
    const fx = THREE.MathUtils.clamp(((x - x0) / (x1 - x0)) * (cols - 1), 0, cols - 1.0001);
    const fy = THREE.MathUtils.clamp(((y - y0) / (y1 - y0)) * (rows - 1), 0, rows - 1.0001);
    const c = Math.floor(fx), r = Math.floor(fy), a = fx - c, b = fy - r;
    const Z = (rr: number, cc: number) => z[rr * cols + cc];
    return (Z(r, c) * (1 - a) + Z(r, c + 1) * a) * (1 - b) + (Z(r + 1, c) * (1 - a) + Z(r + 1, c + 1) * a) * b;
  }

  /** coverage of a tile from a small decode: per 0.32 m cell, the share of opaque texels and the mesh height */
  private coverage(r: Res, bm: ImageBitmap) {
    const t = r.t;
    const w = bm.width, h = bm.height;
    const cv = new OffscreenCanvas(w, h);
    const ctx = cv.getContext("2d", { willReadFrequently: true })!;
    ctx.drawImage(bm, 0, 0);
    const px = ctx.getImageData(0, 0, w, h).data;
    const total = t.pixels + 2 * t.gutter;
    const g = (t.gutter / total) * w, inner = (t.pixels / total) * w;
    const out = new Float32Array(CELLS * CELLS * 2);
    const [x0, y0, x1, y1] = t.bounds;
    for (let cy = 0; cy < CELLS; cy++) for (let cx = 0; cx < CELLS; cx++) {
      const ia = Math.floor(g + (cx * inner) / CELLS), ib = Math.max(ia + 1, Math.floor(g + ((cx + 1) * inner) / CELLS));
      const ja = Math.floor(g + (cy * inner) / CELLS), jb = Math.max(ja + 1, Math.floor(g + ((cy + 1) * inner) / CELLS));
      let n = 0, op = 0;
      for (let j = ja; j < jb; j++) for (let i = ia; i < ib; i++) {
        n++;
        if (px[(j * w + i) * 4 + 3] >= 0.98 * 255) op++;
      }
      // canvas rows run top-down as decoded, and decoded row 0 faces south
      const k = (cy * CELLS + cx) * 2;
      out[k] = n ? op / n : 0;
      out[k + 1] = this.heightAt(t, x0 + ((cx + 0.5) / CELLS) * (x1 - x0), y0 + ((cy + 0.5) / CELLS) * (y1 - y0));
    }
    r.cov = out;
    this.covCache.delete(t.id);
    this.covCache.set(t.id, out);
    while (this.covCache.size > 1600) this.covCache.delete(this.covCache.keys().next().value as string);
    const hf = new Uint16Array(CELLS * CELLS * 4);
    for (let q = 0; q < CELLS * CELLS; q++) {
      hf[q * 4] = THREE.DataUtils.toHalfFloat(out[q * 2]);
      hf[q * 4 + 1] = THREE.DataUtils.toHalfFloat(out[q * 2 + 1]);
    }
    r.covHalf = hf;
  }

  private request(r: Res, level: number) {
    if (r.decoding !== undefined || r.fetching) return;
    if (this.inflight >= 6) return;
    const go = async () => {
      this.inflight++;
      try {
        if (!r.blob) {
          r.fetching = true;
          const resp = await fetch(new URL(r.t.texture, new URL(this.base, location.href)).href);
          if (!resp.ok) throw new Error(`${r.t.texture}: ${resp.status}`);
          r.blob = await resp.blob();
          r.fetching = false;
        }
        r.decoding = level;
        const total = r.t.pixels + 2 * r.t.gutter;
        const size = level === 0 ? total : Math.max(16, Math.round(total / 4));
        const bm = await createImageBitmap(r.blob, {
          premultiplyAlpha: "premultiply",
          colorSpaceConversion: "none",
          ...(size !== total ? { resizeWidth: size, resizeHeight: size, resizeQuality: "high" as const } : {}),
        });
        if (!r.cov) {
          // coverage from the cells read before, else from a quarter-size decode (off the main thread; reading a
          // full 1040 x 1040 bitmap back would stall the frame)
          const c = this.covCache.get(r.t.id);
          if (c) this.adoptCoverage(r, c);
          else if (size === total) {
            const q = Math.max(16, Math.round(total / 4));
            const small = await createImageBitmap(r.blob, { premultiplyAlpha: "premultiply", colorSpaceConversion: "none", resizeWidth: q, resizeHeight: q, resizeQuality: "high" });
            this.coverage(r, small);
            small.close();
          } else this.coverage(r, bm);
        }
        r.bitmap?.close();
        r.bitmap = bm;
        r.bitmapLevel = level;
      } catch (e) {
        console.warn("ground atlas tile failed", e);
        r.fetching = false;
      } finally {
        r.decoding = undefined;
        this.inflight--;
      }
    };
    void go();
  }

  private upload(r: Res) {
    const bm = r.bitmap!;
    const tex = new THREE.Texture(bm);
    tex.flipY = false;
    tex.premultiplyAlpha = false;
    tex.colorSpace = THREE.NoColorSpace;
    tex.generateMipmaps = true;
    tex.minFilter = THREE.LinearMipmapLinearFilter;
    tex.magFilter = THREE.LinearFilter;
    tex.anisotropy = this.anisotropy;
    tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping;
    tex.needsUpdate = true;
    r.tex?.dispose();
    r.tex = tex;
    r.level = r.bitmapLevel;
    r.bitmap = undefined;
    if (!r.opaque) {
      const g = this.geometry(r.t);
      const mk = (m: THREE.ShaderMaterial, order: number) => {
        const mat = m.clone();
        // share the per-frame uniforms, own the textures
        mat.uniforms = { ...m.uniforms, map: { value: tex }, prov: { value: r.prov ?? ZERO }, opacity: { value: 1 } };
        const mesh = new THREE.Mesh(g, mat);
        mesh.position.set(r.t.bounds[0], r.t.bounds[1], 0);
        mesh.renderOrder = order;
        mesh.name = "ground-tile-" + r.t.id;
        this.group.add(mesh);
        return mesh;
      };
      // after the apron (-100), before the backdrop (-90), the ribbon (-50) and the splats
      r.opaque = mk(this.mat.opaque, -98);
      r.soft = mk(this.mat.soft, -97);
    } else {
      (r.opaque.material as THREE.ShaderMaterial).uniforms.map.value = tex;
      (r.soft!.material as THREE.ShaderMaterial).uniforms.map.value = tex;
    }
  }

  /** the "Predicted" overlay for the drawn tiles (when the manifest declares provenance masks) */
  predicted = false;
  private provAmt = 0;
  get hasProvenance() {
    return !!this.json.provenance;
  }

  /** a tile's provenance as overlay weights: the mask's codes read without filtering (nearest, quarter size),
   *  turned into the predicted weight times the texture's coverage, then filtered (never the codes themselves) */
  private requestProvenance(r: Res) {
    if (r.provLoading || r.prov || !r.t.provenance) return;
    r.provLoading = true;
    const total = r.t.pixels + 2 * r.t.gutter;
    const q = Math.max(16, Math.round(total / 4));
    const base = new URL(this.base, location.href);
    const code = this.json.provenance?.values?.predicted ?? 127;
    void (async () => {
      try {
        const [pb, tb] = await Promise.all([
          fetch(new URL(r.t.provenance!, base).href).then((x) => (x.ok ? x.blob() : Promise.reject(new Error(String(x.status))))),
          r.blob ? Promise.resolve(r.blob) : fetch(new URL(r.t.texture, base).href).then((x) => x.blob()),
        ]);
        const opts = { premultiplyAlpha: "none" as const, colorSpaceConversion: "none" as const, resizeWidth: q, resizeHeight: q };
        const [pm, tm] = await Promise.all([createImageBitmap(pb, { ...opts, resizeQuality: "pixelated" }), createImageBitmap(tb, { ...opts, resizeQuality: "high" })]);
        const read = (bm: ImageBitmap) => {
          const cv = new OffscreenCanvas(q, q);
          const ctx = cv.getContext("2d", { willReadFrequently: true })!;
          ctx.drawImage(bm, 0, 0);
          bm.close();
          return ctx.getImageData(0, 0, q, q).data;
        };
        const p = read(pm), t = read(tm);
        const w = new Uint8Array(q * q * 4);
        for (let k = 0; k < q * q; k++) w[k * 4] = Math.abs(p[k * 4] - code) < 40 ? t[k * 4 + 3] : 0;
        const tex = new THREE.DataTexture(w, q, q);
        tex.generateMipmaps = true;
        tex.minFilter = THREE.LinearMipmapLinearFilter;
        tex.magFilter = THREE.LinearFilter;
        tex.colorSpace = THREE.NoColorSpace;
        tex.needsUpdate = true;
        if (!r.opaque) {
          tex.dispose();
          return;
        }
        r.prov = tex;
        for (const m of [r.opaque, r.soft]) if (m) (m.material as THREE.ShaderMaterial).uniforms.prov.value = tex;
      } catch (e) {
        console.warn("ground provenance failed", e);
      } finally {
        r.provLoading = false;
      }
    })();
  }

  private drop(r: Res) {
    if (r.opaque) {
      this.group.remove(r.opaque, r.soft!);
      r.opaque.geometry.dispose();
      (r.opaque.material as THREE.Material).dispose();
      (r.soft!.material as THREE.Material).dispose();
    }
    r.tex?.dispose();
    r.prov?.dispose();
    r.prov = undefined;
    r.bitmap?.close();
    r.opaque = r.soft = undefined;
    r.tex = r.bitmap = undefined;
    r.level = r.bitmapLevel = undefined;
    r.cov = undefined;
    r.covHalf = undefined;
    // out of the map: the compressed file goes too (a long drive would otherwise keep the whole atlas)
    if (!r.fetching && r.decoding === undefined) r.blob = undefined;
    this.covDirty = true;
  }

  /** per frame: stream tiles inside the coverage map around the focus, upload at most one texture, keep the
   *  coverage map the splat shader reads in step with what is resident */
  update(renderer: THREE.WebGLRenderer, focusX: number, focusY: number, cam: THREE.Vector3, show: boolean, now: number) {
    const enabled = true;
    this.frame++;
    const half = (ATLAS_COV_N * CELL) / 2;
    const T = 10.24;
    // recentre the coverage map (tile-aligned) when the focus has moved on
    const ox = Math.floor((focusX - half) / T) * T, oy = Math.floor((focusY - half) / T) * T;
    if (Math.abs(ox - this.covOrigin.x) > 30 || Math.abs(oy - this.covOrigin.y) > 30 || Number.isNaN(this.covOrigin.x)) {
      this.covOrigin.set(ox, oy);
      this.covDirty = true;
      this.covRecentred = true;
    }
    const size = ATLAS_COV_N * CELL;
    const inMap = (r: Res) => r.t.bounds[0] >= this.covOrigin.x - 1e-6 && r.t.bounds[1] >= this.covOrigin.y - 1e-6 && r.t.bounds[2] <= this.covOrigin.x + size + 1e-6 && r.t.bounds[3] <= this.covOrigin.y + size + 1e-6;
    let pending = 0, tiles = 0, full = 0, provRequests = 0;
    this.provAmt += ((this.predicted ? 1 : 0) - this.provAmt) * 0.15;
    if (Math.abs(this.provAmt - (this.predicted ? 1 : 0)) < 0.01) this.provAmt = this.predicted ? 1 : 0;
    if (this.provOverride !== null) this.provAmt = this.provOverride;
    this.uniforms.provAmt.value = this.provAmt;
    let uploaded = this.frame % UP_EVERY !== 0;
    // nearest first
    const want: [Res, number][] = [];
    for (const r of this.res.values()) {
      const keep = enabled && inMap(r);
      if (!keep) {
        if (r.opaque || r.bitmap || r.cov) this.drop(r);
        continue;
      }
      const d = Math.hypot(r.cx - cam.x, r.cy - cam.y);
      want.push([r, d]);
    }
    want.sort((a, b) => a[1] - b[1]);
    for (const [r, d] of want) {
      const level = d < NEAR ? 0 : 2;
      if (r.bitmap && !uploaded) {
        // a tile drawn for the first time adds to the coverage map; a sharper texture for it does not change it
        if (!r.opaque) this.covDirty = true;
        this.upload(r);
        uploaded = true;
      }
      if (r.level !== level && r.bitmapLevel !== level && !(r.level === 0 && level === 2)) {
        // a full texture is only fetched near the camera; a far tile never goes back down unless it is dropped
        // (and is not waiting for anything)
        this.request(r, level);
        pending++;
      }
      if (r.opaque) {
        const op = this.presence ? this.presence(r.t.bounds) : 1;
        for (const mm of [r.opaque, r.soft!]) {
          (mm.material as THREE.ShaderMaterial).uniforms.opacity.value = op;
          mm.visible = op > 0.002;
        }
        tiles++;
        if (r.level === 0) full++;
        r.last = this.frame;
        if (this.predicted && !r.prov && !r.provLoading && provRequests < 2) {
          this.requestProvenance(r);
          provRequests++;
        }
      }
    }
    // coverage the ground filter waits for, two requests a frame
    let n = 0;
    for (const r of this.covWant) {
      if (this.covCache.has(r.t.id)) {
        this.covWant.delete(r);
        continue;
      }
      if (n >= 2) break;
      if (this.requestCoverage(r)) {
        this.covWant.delete(r);
        n++;
      }
    }
    this.stats = { tiles, full, pending };
    let unsettled = this.covDirty ? 1 : 0;
    for (const [r] of want) if (r.bitmap || (this.predicted && r.t.provenance && r.opaque && !r.prov)) unsettled++;
    this.unsettled = unsettled;
    // the coverage map follows what is resident, at most four times a second (a 2 MB upload)
    if (this.covDirty && now - this.covWritten > 250) {
      this.covWritten = now;
      this.writeCoverage(renderer);
    }
    // hidden (Memory mode, or switched to the splat ground): the ground splats come back everywhere
    this.group.visible = show;
    this.G.atlasInfo.value.set(this.covOrigin.x, this.covOrigin.y, size, show ? this.tolerance : 0);
  }

  /** the coverage map the splat shader reads: tiles that came or went since the last write (all of them after
   *  a recentre) copied in or cleared, row by row */
  private writeCoverage(renderer: THREE.WebGLRenderer) {
    void renderer;
    this.covDirty = false;
    const tex = this.G.atlasCoverageTexture;
    const data = tex.image.data as Uint16Array;
    const N = ATLAS_COV_N;
    const zero = new Uint16Array(CELLS * 4);
    if (this.covRecentred) {
      this.covRecentred = false;
      data.fill(0);
      this.covPlaced.clear();
    }
    for (const [r, [ci, cj]] of this.covPlaced) {
      if (r.opaque && r.covHalf) continue;
      for (let y = 0; y < CELLS; y++) data.set(zero, ((cj + y) * N + ci) * 4);
      this.covPlaced.delete(r);
    }
    for (const r of this.res.values()) {
      if (!r.covHalf || !r.opaque || this.covPlaced.has(r)) continue;
      const [x0, y0] = r.t.bounds;
      const ci = Math.round((x0 - this.covOrigin.x) / CELL), cj = Math.round((y0 - this.covOrigin.y) / CELL);
      if (ci < 0 || cj < 0 || ci + CELLS > N || cj + CELLS > N) continue;
      for (let y = 0; y < CELLS; y++) data.set(r.covHalf.subarray(y * CELLS * 4, (y + 1) * CELLS * 4), ((cj + y) * N + ci) * 4);
      this.covPlaced.set(r, [ci, cj]);
    }
    tex.needsUpdate = true;
  }

  dispose() {
    for (const r of this.res.values()) this.drop(r);
    this.mat.opaque.dispose();
    this.mat.soft.dispose();
    this.covTile.dispose();
    this.G.atlasInfo.value.w = 0;
    this.group.clear();
  }
}
