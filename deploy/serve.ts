// Static host for the built viewer plus the scene folder, with byte ranges for the large tile files.
// Usage: bun serve.ts <dist dir> <scenes dir> [port] [host]   (host defaults to 127.0.0.1; pass 0.0.0.0 to serve a LAN)
import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import path from "node:path";

const [dist, scenes, port = "8090", hostname = "127.0.0.1"] = process.argv.slice(2);
const DIST = path.resolve(dist), SCENES = path.resolve(scenes);

function listScenes() {
  return readdirSync(SCENES, { withFileTypes: true })
    .filter((d) => (d.isDirectory() || d.isSymbolicLink()) && existsSync(path.join(SCENES, d.name, "drive.json")))
    .map((d) => {
      let tiles = 0, splats = 0;
      try {
        const m = JSON.parse(readFileSync(path.join(SCENES, d.name, "map.json"), "utf8"));
        tiles = m.tiles?.length ?? 0;
        splats = (m.tiles ?? []).reduce((a: number, t: { count?: number }) => a + (t.count ?? 0), 0);
      } catch {}
      return { name: d.name, url: `/scenes/${d.name}/`, hasMap: tiles > 0, tiles, splats };
    });
}

function file(root: string, rel: string, req: Request, cache: string) {
  const p = path.resolve(root, "." + rel);
  if (!p.startsWith(root + path.sep) || !existsSync(p) || !statSync(p).isFile()) return null;
  const f = Bun.file(p), size = f.size;
  const headers: Record<string, string> = { "Cache-Control": cache, "Accept-Ranges": "bytes", "Last-Modified": new Date(f.lastModified).toUTCString() };
  const range = /^bytes=(\d*)-(\d*)$/.exec(req.headers.get("range") ?? "");
  if (range) {
    const start = range[1] ? Number(range[1]) : Math.max(0, size - Number(range[2]));
    const end = range[1] && range[2] ? Math.min(Number(range[2]), size - 1) : size - 1;
    if (start > end || start >= size) return new Response(null, { status: 416, headers: { "Content-Range": `bytes */${size}` } });
    headers["Content-Range"] = `bytes ${start}-${end}/${size}`;
    return new Response(f.slice(start, end + 1), { status: 206, headers });
  }
  if (req.headers.get("if-modified-since") === headers["Last-Modified"]) return new Response(null, { status: 304, headers });
  return new Response(f, { headers });
}

Bun.serve({
  port: Number(port),
  hostname,
  idleTimeout: 120,
  fetch(req) {
    const url = decodeURIComponent(new URL(req.url).pathname);
    if (url === "/scenes/index.json" || url === "/scenes/" || url === "/scenes")
      return Response.json({ scenes: listScenes() }, { headers: { "Cache-Control": "no-cache" } });
    if (url.startsWith("/scenes/")) return file(SCENES, url.slice(7), req, "no-cache") ?? new Response("not found", { status: 404 });
    return (
      file(DIST, url === "/" ? "/index.html" : url, req, url.startsWith("/assets/") ? "public, max-age=31536000, immutable" : "no-cache") ??
      file(DIST, "/index.html", req, "no-cache") ??
      new Response("not built", { status: 404 })
    );
  },
});
console.log(`serving ${DIST} and ${SCENES} on http://${hostname}:${port}`);
