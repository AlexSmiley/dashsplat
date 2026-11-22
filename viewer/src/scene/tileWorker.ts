/// <reference lib="webworker" />
// Fetches a tile (contract PLY or the compact .rsb cache) and decodes it off the main thread into Spark
// ExtSplats arrays plus the packed per-splat time texels; see tileCodec.ts for the formats.

import { decodePly, decodeRsb, isRsb, type Decoded } from "./tileCodec";

/** ground-layer splats to leave out (phones: where the textured ground is opaque they are hidden anyway): cells of
 *  `cell` metres from (x0, y0), nx by ny; op 1 = the atlas is opaque there, z = its mesh height */
export type GroundFilter = { x0: number; y0: number; cell: number; nx: number; ny: number; op: Uint8Array; z: Float32Array; tol: number };
export type TileRequest = { id: string; url: string; texWidth: number; filter?: GroundFilter };
export type TileResult = Decoded & { id: string; bytes: number; ms: number };

self.onmessage = async (ev: MessageEvent<TileRequest>) => {
  const { id, url, texWidth, filter } = ev.data;
  const t0 = performance.now();
  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    const buf = await res.arrayBuffer();
    const r = isRsb(buf) ? decodeRsb(buf, texWidth) : decodePly(buf, texWidth);
    if (filter) r.count = dropUnderAtlas(r, filter);
    const out: TileResult = { id, ...r, bytes: buf.byteLength, ms: performance.now() - t0 };
    (self as unknown as Worker).postMessage({ ok: true, result: out }, [r.extA.buffer, r.extB.buffer, r.times.buffer]);
  } catch (e) {
    (self as unknown as Worker).postMessage({ ok: false, id, error: String(e) });
  }
};

/** compact the splats in place, leaving out those on an opaque atlas cell within the height tolerance */
function dropUnderAtlas(r: Decoded, f: GroundFilter) {
  const fa = new Float32Array(r.extA.buffer, r.extA.byteOffset, r.extA.length);
  const [ox, oy, oz] = r.origin;
  let j = 0;
  for (let i = 0; i < r.count; i++) {
    const i4 = i * 4;
    const cx = Math.floor((fa[i4] + ox - f.x0) / f.cell), cy = Math.floor((fa[i4 + 1] + oy - f.y0) / f.cell);
    if (cx >= 0 && cy >= 0 && cx < f.nx && cy < f.ny) {
      const k = cy * f.nx + cx;
      if (f.op[k] && Math.abs(fa[i4 + 2] + oz - f.z[k]) < f.tol) continue;
    }
    if (j !== i) {
      const j4 = j * 4;
      r.extA.copyWithin(j4, i4, i4 + 4);
      r.extB.copyWithin(j4, i4, i4 + 4);
      r.times.copyWithin(j4, i4, i4 + 4);
    }
    j++;
  }
  return j;
}
