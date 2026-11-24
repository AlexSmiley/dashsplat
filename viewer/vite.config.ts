import { defineConfig, type Plugin } from "vite";
import { createReadStream, existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import type { IncomingMessage, ServerResponse } from "node:http";

const HERE = path.dirname(fileURLToPath(import.meta.url));
// the repo-root scenes/ folder is served at /scenes (dev and preview); /scenes/index.json lists the scenes
const SCENES = path.resolve(HERE, "../scenes");

const TYPES: Record<string, string> = {
  ".json": "application/json",
  ".ply": "application/octet-stream",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".webp": "image/webp",
};

function listScenes() {
  if (!existsSync(SCENES)) return [];
  return readdirSync(SCENES, { withFileTypes: true })
    .filter((d) => d.isDirectory() && existsSync(path.join(SCENES, d.name, "drive.json")))
    .map((d) => {
      const mapPath = path.join(SCENES, d.name, "map.json");
      let tiles = 0;
      let splats = 0;
      if (existsSync(mapPath)) {
        try {
          const m = JSON.parse(readFileSync(mapPath, "utf8"));
          tiles = m.tiles?.length ?? 0;
          splats = (m.tiles ?? []).reduce((a: number, t: { count?: number }) => a + (t.count ?? 0), 0);
        } catch {
          /* unreadable map.json: listed without tiles */
        }
      }
      return { name: d.name, url: `/scenes/${d.name}/`, hasMap: tiles > 0, tiles, splats };
    });
}

function handler(req: IncomingMessage, res: ServerResponse, next: () => void) {
  const url = decodeURIComponent((req.url ?? "/").split("?")[0]);
  if (url === "/index.json" || url === "/") {
    if (!existsSync(SCENES)) return next(); // use bundled demo index from viewer/public
    res.setHeader("Content-Type", "application/json");
    res.setHeader("Cache-Control", "no-cache");
    res.end(JSON.stringify({ scenes: listScenes() }));
    return;
  }
  const file = path.resolve(SCENES, "." + url);
  if (!file.startsWith(SCENES + path.sep) || !existsSync(file) || !statSync(file).isFile()) {
    return next(); // Vite serves viewer/public/scenes/demo when local scenes are absent
  }
  const st = statSync(file);
  const mtime = st.mtime.toUTCString();
  res.setHeader("Last-Modified", mtime);
  res.setHeader("Cache-Control", "no-cache");
  if (req.headers["if-modified-since"] === mtime) {
    res.statusCode = 304;
    res.end();
    return;
  }
  res.setHeader("Content-Type", TYPES[path.extname(file).toLowerCase()] ?? "application/octet-stream");
  res.setHeader("Content-Length", String(st.size));
  createReadStream(file).pipe(res);
  void next;
}

function scenes(): Plugin {
  return {
    name: "dashsplat-scenes",
    configureServer(server) {
      server.middlewares.use("/scenes", handler);
    },
    configurePreviewServer(server) {
      server.middlewares.use("/scenes", handler);
    },
  };
}

export default defineConfig({
  plugins: [scenes()],
  server: { host: "127.0.0.1", port: 5188, strictPort: false },
  preview: { host: "127.0.0.1", port: 5189 },
  worker: { format: "es" },
  build: { target: "es2022", chunkSizeWarningLimit: 4096 },
});
