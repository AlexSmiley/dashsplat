#!/bin/sh
# Builds the compact .rsb tile cache for every scene on the Mac Studio's stable host (anomastudio), in place:
# ~/dev/reality-splat/scene/<name>/rsb/. The cache tool and its codec are pushed to
# ~/dev/reality-splat/viewer-cache-tool/ first (the Studio has bun). Unchanged tiles are skipped, so run it
# whenever a scene is added or replaced there.
# The author's helper (his ferry CLI and Studio paths). On your machine use: bun run cache <scene dir>
set -e
cd "$(dirname "$0")/.."
ferry sh anomastudio 'mkdir -p ~/dev/reality-splat/viewer-cache-tool/tools ~/dev/reality-splat/viewer-cache-tool/src/scene'
ferry cp tools/build-tile-cache.ts anomastudio:dev/reality-splat/viewer-cache-tool/tools/
ferry cp src/scene/tileCodec.ts anomastudio:dev/reality-splat/viewer-cache-tool/src/scene/
ferry cp src/scene/whiteBalance.ts anomastudio:dev/reality-splat/viewer-cache-tool/src/scene/
ferry sh anomastudio 'cd ~/dev/reality-splat/viewer-cache-tool && for d in ~/dev/reality-splat/scene/*/; do [ -f "$d/map.json" ] && bun tools/build-tile-cache.ts "$d"; done; true'
