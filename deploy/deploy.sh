#!/bin/sh
# Build the viewer into deploy/dist. Run from anywhere; the dev server (viewer/: bun run dev) stays separate.
#
# On your machine: serve the build and the published scenes with
#   bun deploy/serve.ts deploy/dist scene 8090 127.0.0.1
# (scene/ is where pipeline/run publishes; the runner's default RS_VIEWER_URL is http://127.0.0.1:8090).
#
# To copy the build to another machine set RS_DEPLOY_TARGET (and optionally RS_DEPLOY_URL to print its link).
# The copy uses the author's ferry tool. The author's setup was
#   RS_DEPLOY_TARGET=anomastudio:dev/reality-splat/web RS_DEPLOY_URL=http://<studio LAN address>:8090/
# with deploy/dev.vogel.reality-splat-viewer.plist keeping serve.ts running there.
set -e
cd "$(dirname "$0")/.."
(cd viewer && bunx vite build --outDir ../deploy/dist --emptyOutDir)
if [ -z "${RS_DEPLOY_TARGET:-}" ]; then
  echo "built deploy/dist; serve it with: bun deploy/serve.ts deploy/dist scene 8090 127.0.0.1"
  exit 0
fi
ferry cp ./deploy/dist/ "$RS_DEPLOY_TARGET/dist"
ferry cp ./deploy/serve.ts "$RS_DEPLOY_TARGET/"
echo "published: ${RS_DEPLOY_URL:-$RS_DEPLOY_TARGET}"
