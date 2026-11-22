// Tile decoding shared by the browser worker and the cache builder (no DOM).
//
// Input: contract PLY tiles, or the compact ".rsb" cache written by tools/build-tile-cache.ts.
// Output: Spark ExtSplats arrays (float32 centres relative to the tile origin) plus one RGBA32UI texel of
// packed times per splat:
//   t_birth, t_death, e0, s1, e1, s2, e2 -> uint16 steps of `quantum` seconds after `base` (65535 = never)
//   word0 = birth | death << 16, word1 = e0 | s1 << 16, word2 = e1 | s2 << 16, word3 = e2
//
// RSB layout (little endian): "RSB1", u32 header bytes, JSON header (padded to 4), then per splat arrays:
//   pos u16x3 (quantised in the header bbox), scale u8x3 (ln scale in [lnMin, lnMax]), rot u32 (Spark's
//   octahedral axis-angle), rgba u8x4 (DC colour, opacity), times u32x4 (as above). 33 bytes per splat.

export type Decoded = {
  count: number;
  origin: [number, number, number];
  extA: Uint32Array;
  extB: Uint32Array;
  times: Uint32Array;
  base: number;
  quantum: number;
  hasTimes: boolean;
};

const SH_C0 = 0.28209479177387814;
const NEVER = 1e8;
export const SPLAT_TEX_WIDTH = 2048;

const f32 = new Float32Array(1);
const u32 = new Uint32Array(f32.buffer);
// float -> IEEE half bits (Spark's toHalfJS)
export function toHalf(f: number) {
  f32[0] = f;
  const bits = u32[0];
  const sign = (bits >> 31) & 1, exp = (bits >> 23) & 255, frac = bits & 8388607;
  const halfSign = sign << 15;
  if (exp === 255) return frac !== 0 ? halfSign | 32767 : halfSign | 31744;
  const newExp = exp - 127 + 15;
  if (newExp >= 31) return halfSign | 31744;
  if (newExp <= 0) return newExp < -10 ? halfSign : halfSign | ((frac | 8388608) >> (1 - newExp + 13));
  return halfSign | (newExp << 10) | (frac >> 13);
}
// Spark's encodeQuatOctXy1010R12 (MIT, sparkjsdev/spark)
export function encodeQuat(qx: number, qy: number, qz: number, qw: number) {
  const qlen = Math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw) || 1;
  const s = qw < 0 ? -1 : 1;
  const qnx = (s * qx) / qlen, qny = (s * qy) / qlen, qnz = (s * qz) / qlen, qnw = Math.min(1, (s * qw) / qlen);
  const theta = 2 * Math.acos(qnw);
  const n = Math.sqrt(qnx * qnx + qny * qny + qnz * qnz);
  const ax = n < 1e-6 ? 1 : qnx / n, ay = n < 1e-6 ? 0 : qny / n, az = n < 1e-6 ? 0 : qnz / n;
  const sum = Math.abs(ax) + Math.abs(ay) + Math.abs(az);
  let p_x = ax / sum, p_y = ay / sum;
  if (az < 0) {
    const t = p_x;
    p_x = (1 - Math.abs(p_y)) * (p_x >= 0 ? 1 : -1);
    p_y = (1 - Math.abs(t)) * (p_y >= 0 ? 1 : -1);
  }
  const qu = Math.round((p_x * 0.5 + 0.5) * 1023), qv = Math.round((p_y * 0.5 + 0.5) * 1023);
  return ((Math.round(theta * (4095 / Math.PI)) << 20) | (qv << 10) | qu) >>> 0;
}

// ------------------------------------------------------------------ PLY
const SIZES: Record<string, number> = { char: 1, uchar: 1, int8: 1, uint8: 1, short: 2, ushort: 2, int16: 2, uint16: 2, int: 4, uint: 4, int32: 4, uint32: 4, float: 4, float32: 4, double: 8, float64: 8 };
type Reader = (i: number) => number;
export type Ply = { count: number; get: (name: string) => Reader; has: (name: string) => boolean };

export function parsePly(buf: ArrayBuffer): Ply {
  const bytes = new Uint8Array(buf);
  const pat = [101, 110, 100, 95, 104, 101, 97, 100, 101, 114]; // end_header
  let headEnd = -1;
  for (let i = 0; i < Math.min(bytes.length, 65536) && headEnd < 0; i++) {
    let ok = true;
    for (let j = 0; j < pat.length; j++) if (bytes[i + j] !== pat[j]) { ok = false; break; }
    if (ok) {
      let k = i + pat.length;
      while (bytes[k] !== 10) k++;
      headEnd = k + 1;
    }
  }
  if (headEnd < 0) throw new Error("PLY: no end_header");
  const header = new TextDecoder().decode(bytes.subarray(0, headEnd));
  if (!header.includes("binary_little_endian")) throw new Error("PLY: only binary_little_endian is supported");
  let count = 0, inVertex = false, stride = 0;
  const props: { name: string; type: string; offset: number }[] = [];
  for (const line of header.split("\n")) {
    const w = line.trim().split(/\s+/);
    if (w[0] === "element") {
      inVertex = w[1] === "vertex";
      if (inVertex) count = Number(w[2]);
    } else if (w[0] === "property" && inVertex) {
      if (w[1] === "list") throw new Error("PLY: list properties in vertex element");
      props.push({ name: w[2], type: w[1], offset: stride });
      stride += SIZES[w[1]] ?? 4;
    }
  }
  const P = new Map(props.map((p) => [p.name, p]));
  const allFloat = props.every((p) => p.type === "float" || p.type === "float32");
  const fl = allFloat ? (headEnd % 4 === 0 ? new Float32Array(buf, headEnd, (count * stride) / 4) : new Float32Array(buf.slice(headEnd, headEnd + count * stride))) : null;
  const dv = new DataView(buf, headEnd);
  const sf = stride / 4;
  const get = (name: string): Reader => {
    const p = P.get(name);
    if (!p) return () => NaN;
    if (fl) { const o = p.offset / 4; return (i) => fl[i * sf + o]; }
    const o = p.offset;
    switch (p.type) {
      case "double": case "float64": return (i) => dv.getFloat64(i * stride + o, true);
      case "uchar": case "uint8": return (i) => dv.getUint8(i * stride + o);
      case "ushort": case "uint16": return (i) => dv.getUint16(i * stride + o, true);
      case "int": case "int32": return (i) => dv.getInt32(i * stride + o, true);
      case "uint": case "uint32": return (i) => dv.getUint32(i * stride + o, true);
      default: return (i) => dv.getFloat32(i * stride + o, true);
    }
  };
  for (const n of ["x", "y", "z", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3", "opacity", "f_dc_0", "f_dc_1", "f_dc_2"]) if (!P.has(n)) throw new Error("PLY: missing property " + n);
  return { count, get, has: (n) => P.has(n) };
}

/** packs the seven contract times of every splat into one uint32x4 texel each */
export function packTimes(ply: Ply, texWidth: number) {
  const { count } = ply;
  const rows = Math.max(1, Math.ceil(count / texWidth));
  const times = new Uint32Array(texWidth * rows * 4);
  const TR = ["t_birth", "t_death", "e0", "s1", "e1", "s2", "e2"].map((n) => ply.get(n));
  const hasTimes = ply.has("t_birth");
  let base = Infinity, top = -Infinity;
  if (hasTimes) for (let i = 0; i < count; i++) {
    const b = TR[0](i);
    if (b < base) base = b;
    for (let k = 0; k < 7; k++) { const t = TR[k](i); if (t < NEVER && t > top) top = t; }
  }
  if (!Number.isFinite(base)) base = 0;
  base = Math.floor(base * 100) / 100;
  const quantum = Math.max(0.01, (top - base) / 65534);
  const enc = (t: number) => (!(t < NEVER) ? 65535 : Math.max(0, Math.min(65534, Math.round((t - base) / quantum))));
  for (let i = 0; i < count; i++) {
    const o = i * 4;
    if (!hasTimes) {
      times[o] = (65535 << 16) >>> 0;
      times[o + 1] = (65534 | (65535 << 16)) >>> 0;
      times[o + 2] = 0xffffffff;
      times[o + 3] = 65535;
      continue;
    }
    const b0 = TR[0](i);
    const birth = enc(Number.isNaN(b0) ? base : b0);
    let death = enc(TR[1](i));
    if (death !== 65535 && death <= birth) death = birth + 1;
    const e0v = TR[2](i);
    times[o] = (birth | (death << 16)) >>> 0;
    times[o + 1] = (enc(Number.isNaN(e0v) ? b0 : e0v) | (enc(TR[3](i)) << 16)) >>> 0;
    times[o + 2] = (enc(TR[4](i)) | (enc(TR[5](i)) << 16)) >>> 0;
    times[o + 3] = enc(TR[6](i));
  }
  return { times, base, quantum, hasTimes };
}

function bboxOf(ply: Ply) {
  const X = ply.get("x"), Y = ply.get("y"), Z = ply.get("z");
  const b = [Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity];
  for (let i = 0; i < ply.count; i++) {
    const x = X(i), y = Y(i), z = Z(i);
    if (x < b[0]) b[0] = x; if (y < b[1]) b[1] = y; if (z < b[2]) b[2] = z;
    if (x > b[3]) b[3] = x; if (y > b[4]) b[4] = y; if (z > b[5]) b[5] = z;
  }
  if (!ply.count) return [0, 0, 0, 0, 0, 0];
  return b;
}

function allocExt(count: number) {
  const maxSplats = Math.max(SPLAT_TEX_WIDTH, Math.ceil(count / SPLAT_TEX_WIDTH) * SPLAT_TEX_WIDTH);
  return { extA: new Uint32Array(maxSplats * 4), extB: new Uint32Array(maxSplats * 4) };
}

export function decodePly(buf: ArrayBuffer, texWidth: number): Decoded {
  const ply = parsePly(buf);
  const { count } = ply;
  const b = bboxOf(ply);
  const origin: [number, number, number] = [(b[0] + b[3]) / 2, (b[1] + b[4]) / 2, (b[2] + b[5]) / 2];
  const { extA, extB } = allocExt(count);
  const extAf = new Float32Array(extA.buffer);
  const X = ply.get("x"), Y = ply.get("y"), Z = ply.get("z");
  const S0 = ply.get("scale_0"), S1 = ply.get("scale_1"), S2 = ply.get("scale_2");
  const R0 = ply.get("rot_0"), R1 = ply.get("rot_1"), R2 = ply.get("rot_2"), R3 = ply.get("rot_3");
  const OP = ply.get("opacity"), C0 = ply.get("f_dc_0"), C1 = ply.get("f_dc_1"), C2 = ply.get("f_dc_2");
  for (let i = 0; i < count; i++) {
    const i4 = i * 4;
    extAf[i4] = X(i) - origin[0];
    extAf[i4 + 1] = Y(i) - origin[1];
    extAf[i4 + 2] = Z(i) - origin[2];
    extA[i4 + 3] = toHalf(1 / (1 + Math.exp(-OP(i))));
    extB[i4] = (toHalf(0.5 + SH_C0 * C0(i)) | (toHalf(0.5 + SH_C0 * C1(i)) << 16)) >>> 0;
    extB[i4 + 1] = (toHalf(0.5 + SH_C0 * C2(i)) | (toHalf(S0(i)) << 16)) >>> 0;
    extB[i4 + 2] = (toHalf(S1(i)) | (toHalf(S2(i)) << 16)) >>> 0;
    extB[i4 + 3] = encodeQuat(R1(i), R2(i), R3(i), R0(i)); // PLY rot is w x y z
  }
  const t = packTimes(ply, texWidth);
  return { count, origin, extA, extB, ...t };
}

// ------------------------------------------------------------------ RSB
const align4 = (n: number) => (n + 3) & ~3;

export function encodeRsb(buf: ArrayBuffer, texWidth = SPLAT_TEX_WIDTH): Uint8Array {
  const ply = parsePly(buf);
  const n = ply.count;
  const bb = bboxOf(ply);
  const S = [ply.get("scale_0"), ply.get("scale_1"), ply.get("scale_2")];
  let lnMin = Infinity, lnMax = -Infinity;
  for (let i = 0; i < n; i++) for (const s of S) { const v = s(i); if (v < lnMin) lnMin = v; if (v > lnMax) lnMax = v; }
  if (!Number.isFinite(lnMin)) { lnMin = -10; lnMax = 0; }
  if (lnMax - lnMin < 1e-3) lnMax = lnMin + 1e-3;
  const t = packTimes(ply, texWidth);
  const header = { v: 1, count: n, bbox: bb, lnMin, lnMax, base: t.base, quantum: t.quantum, hasTimes: t.hasTimes };
  const hj = new TextEncoder().encode(JSON.stringify(header));
  const hlen = align4(hj.length);
  const oPos = 8 + hlen;
  const oScale = align4(oPos + n * 6);
  const oRot = align4(oScale + n * 3);
  const oRgba = oRot + n * 4;
  const oTimes = oRgba + n * 4;
  const total = oTimes + n * 16;
  const out = new Uint8Array(total);
  out.set([82, 83, 66, 49], 0); // RSB1
  new DataView(out.buffer).setUint32(4, hlen, true);
  out.set(hj, 8);
  for (let k = hj.length; k < hlen; k++) out[8 + k] = 32;
  const pos = new Uint16Array(out.buffer, oPos, n * 3);
  const sc = new Uint8Array(out.buffer, oScale, n * 3);
  const rot = new Uint32Array(out.buffer, oRot, n);
  const rgba = new Uint8Array(out.buffer, oRgba, n * 4);
  const tm = new Uint32Array(out.buffer, oTimes, n * 4);
  const X = [ply.get("x"), ply.get("y"), ply.get("z")];
  const R0 = ply.get("rot_0"), R1 = ply.get("rot_1"), R2 = ply.get("rot_2"), R3 = ply.get("rot_3");
  const OP = ply.get("opacity"), C = [ply.get("f_dc_0"), ply.get("f_dc_1"), ply.get("f_dc_2")];
  const q = (v: number) => Math.max(0, Math.min(255, Math.round(v * 255)));
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < 3; k++) {
      const span = bb[3 + k] - bb[k];
      pos[i * 3 + k] = span > 0 ? Math.round(((X[k](i) - bb[k]) / span) * 65535) : 0;
      sc[i * 3 + k] = Math.round(((S[k](i) - lnMin) / (lnMax - lnMin)) * 255);
      rgba[i * 4 + k] = q(0.5 + SH_C0 * C[k](i));
    }
    rgba[i * 4 + 3] = q(1 / (1 + Math.exp(-OP(i))));
    rot[i] = encodeQuat(R1(i), R2(i), R3(i), R0(i));
  }
  tm.set(t.times.subarray(0, n * 4));
  return out;
}

export function isRsb(buf: ArrayBuffer) {
  const b = new Uint8Array(buf, 0, 4);
  return b[0] === 82 && b[1] === 83 && b[2] === 66 && b[3] === 49;
}

export function decodeRsb(buf: ArrayBuffer, texWidth: number): Decoded {
  const dv = new DataView(buf);
  const hlen = dv.getUint32(4, true);
  const h = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 8, hlen))) as { count: number; bbox: number[]; lnMin: number; lnMax: number; base: number; quantum: number; hasTimes: boolean };
  const n = h.count;
  const oPos = 8 + hlen;
  const oScale = align4(oPos + n * 6);
  const oRot = align4(oScale + n * 3);
  const oRgba = oRot + n * 4;
  const oTimes = oRgba + n * 4;
  const pos = new Uint16Array(buf, oPos, n * 3);
  const sc = new Uint8Array(buf, oScale, n * 3);
  const rot = new Uint32Array(buf, oRot, n);
  const rgba = new Uint8Array(buf, oRgba, n * 4);
  const tm = new Uint32Array(buf, oTimes, n * 4);
  const bb = h.bbox;
  const origin: [number, number, number] = [(bb[0] + bb[3]) / 2, (bb[1] + bb[4]) / 2, (bb[2] + bb[5]) / 2];
  const step = [0, 1, 2].map((k) => (bb[3 + k] - bb[k]) / 65535);
  const off = [0, 1, 2].map((k) => bb[k] - origin[k]);
  const lnStep = (h.lnMax - h.lnMin) / 255;
  // half-float lookup tables for the 8-bit channels
  const halfLn = new Uint16Array(256), halfU8 = new Uint16Array(256);
  for (let v = 0; v < 256; v++) { halfLn[v] = toHalf(h.lnMin + v * lnStep); halfU8[v] = toHalf(v / 255); }
  const { extA, extB } = allocExt(n);
  const extAf = new Float32Array(extA.buffer);
  for (let i = 0; i < n; i++) {
    const i4 = i * 4, i3 = i * 3;
    extAf[i4] = off[0] + pos[i3] * step[0];
    extAf[i4 + 1] = off[1] + pos[i3 + 1] * step[1];
    extAf[i4 + 2] = off[2] + pos[i3 + 2] * step[2];
    extA[i4 + 3] = halfU8[rgba[i4 + 3]];
    extB[i4] = (halfU8[rgba[i4]] | (halfU8[rgba[i4 + 1]] << 16)) >>> 0;
    extB[i4 + 1] = (halfU8[rgba[i4 + 2]] | (halfLn[sc[i3]] << 16)) >>> 0;
    extB[i4 + 2] = (halfLn[sc[i3 + 1]] | (halfLn[sc[i3 + 2]] << 16)) >>> 0;
    extB[i4 + 3] = rot[i];
  }
  const rows = Math.max(1, Math.ceil(n / texWidth));
  const times = new Uint32Array(texWidth * rows * 4);
  times.set(tm);
  return { count: n, origin, extA, extB, times, base: h.base, quantum: h.quantum, hasTimes: h.hasTimes };
}
