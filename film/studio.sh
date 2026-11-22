#!/bin/sh
# Ship the film project to the Studio and run commands there (rendering happens on anomastudio, not basebook).
#   sh film/studio.sh push            copy source, sound, conform tables and the small data (not media)
#   sh film/studio.sh push-media      copy film/work/media (no capture links) and film/work/audio
#   sh film/studio.sh build           cargo build --release there (low priority)
#   sh film/studio.sh run '<cmd>'     run a command in ~/dev/reality-splat/film/render there, e.g. 'journey strip Depth -n 8 -o ../work/review/s.png'
#   sh film/studio.sh fetch <path>    bring a file back from ~/dev/reality-splat/film/<path> into film/<path>
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
R=dev/reality-splat/film
ENV='export PATH=$HOME/.cargo/bin:/opt/homebrew/bin:$PATH; cd ~/dev/reality-splat/film'
case "$1" in
  push)
    tmp=$(mktemp -d)
    tar -C "$HERE" --exclude render/target --exclude 'render/media/*.ttf' -czf "$tmp/film-src.tgz" render sound viz conform.json conform.py deliver.sh STORYBOARD.md README.md notes work/data
    ferry cp "$tmp/film-src.tgz" "anomastudio:$R/" >/dev/null
    ferry sh anomastudio "$ENV && tar -xzf film-src.tgz && rm film-src.tgz && mkdir -p render/media work/media work/audio work/review out"
    for f in "$HERE"/render/media/*.ttf; do ferry cp "$f" "anomastudio:$R/render/media/" >/dev/null; done
    rm -rf "$tmp" ;;
  push-media)
    for f in "$HERE"/work/media/*; do case "$(basename "$f")" in cap-*) ;; *) [ -L "$f" ] || ferry cp "$f" "anomastudio:$R/work/media/" >/dev/null ;; esac; done
    ferry cp "$HERE/work/audio" "anomastudio:$R/work/" >/dev/null ;;
  build)
    ferry sh anomastudio "$ENV/render && nice -n 15 cargo build --release 2>&1 | grep -E '^(error|warning: unused)' -A 8 | head -40; ls -la target/release/journey" ;;
  run)
    ferry sh anomastudio "$ENV/render && nice -n 15 ./target/release/$2" ;;
  fetch)
    mkdir -p "$(dirname "$HERE/$2")"
    ferry cp "anomastudio:$R/$2" "$(dirname "$HERE/$2")/" >/dev/null ;;
esac
