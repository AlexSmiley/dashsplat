#!/usr/bin/env bash
# Install only the runner, its path plumbing and the existing chase-camera code.
# The author's helper for a checkout on one Mac and the installation on another, copied with his ferry tool.
# Not needed when you run pipeline/run/drive.py from this checkout: build the chase tool there instead with
#   bun build viewer/tools/chase-track.ts --target=bun --minify --outdir=tools/run-chase
set -euo pipefail
cd "$(dirname "$0")/../.."
dst=${RS_REMOTE:-anomastudio:dev/reality-splat}
out=$(mktemp -d); trap 'rm -rf "$out"' EXIT
bun build viewer/tools/chase-track.ts --target=bun --minify --outdir="$out"
ferry cp "$out/chase-track.js" "$dst/tools/run-chase/"
ferry cp pipeline/run/*.py pipeline/run/rig-baseline.json pipeline/run/README.md pipeline/run/install.sh "$dst/pipeline/run/"
for f in pipeline/ibr/{morph_sources,teacher}.py pipeline/livemap/audit.py pipeline/movers/removal.py pipeline/stitch/{v2data,v5world,v5fit,v5release}.py; do
  ferry cp "$f" "$dst/$(dirname "$f")/"
done
