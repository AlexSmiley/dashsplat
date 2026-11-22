#!/bin/sh
# Render every deliverable into film/out/ on the machine it runs on (the Studio: one low-priority job at a time).
#   out/reality-splat-journey.mp4, out/reality-splat-journey-silent.mp4, out/stems/{mix,ambience,effects}.wav,
#   out/chapters/NN-name.mp4, out/contact-sheet.png
# usage: sh film/deliver.sh [--skip-master] [--skip-chapters] [--only-chapters "4 9"]
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-$HERE/out}
B="nice -n 15 $HERE/render/target/release/journey"
UV="nice -n 15 uv run --quiet --with numpy --with scipy --with pyloudnorm python"
need_space() {  # stop rather than fill the disk
  free=$(df -g "$HOME" | awk 'NR==2 {print $4}')
  if [ "$free" -lt "$1" ]; then echo "only ${free} GB free, need $1 GB; stopping" >&2; exit 3; fi
}
mkdir -p "$OUT/stems" "$OUT/chapters"
cd "$HERE/render"
case " $* " in *" --skip-master "*) ;; *)
  need_space 8
  $B render -o "$OUT/raw.mp4"
  $B audio render -o "$OUT/stems/mix-raw.wav"
  STEM=amb $B audio render -o "$OUT/stems/amb-raw.wav"
  STEM=sfx $B audio render -o "$OUT/stems/sfx-raw.wav"
  $UV "$HERE/sound/master.py" "$OUT/stems"
  nice -n 15 ffmpeg -v error -y -i "$OUT/raw.mp4" -i "$OUT/stems/mix.wav" -map 0:v -map 1:a -c:v copy -c:a aac -b:a 320k -ar 48000 -movflags +faststart "$OUT/reality-splat-journey.mp4"
  nice -n 15 ffmpeg -v error -y -i "$OUT/raw.mp4" -map 0:v -c:v copy -movflags +faststart "$OUT/reality-splat-journey-silent.mp4"
  rm -f "$OUT/raw.mp4" "$OUT/stems/"*-raw.wav
  for s in mix ambience effects; do  # 24-bit PCM: plenty, and a third smaller than float
    ffmpeg -v error -y -i "$OUT/stems/$s.wav" -c:a pcm_s24le "$OUT/stems/$s.tmp.wav" && mv "$OUT/stems/$s.tmp.wav" "$OUT/stems/$s.wav"
  done
  $B strip all -n 60 --columns 6 --width 640 -o "$OUT/contact-sheet.png"
;; esac
case " $* " in *" --skip-chapters "*) ;; *)
  ONLY=$(echo " $* " | sed -n 's/.*--only-chapters \([0-9 ]*\).*/\1/p')
  c=0
  for name in cold-open input geometry first-ground first-world optimiser experiments road-atlas vehicles image-model finished-view one-command close; do
    if [ -n "$ONLY" ] && ! echo " $ONLY " | grep -q " $c "; then c=$((c + 1)); continue; fi
    need_space 4
    f=$(printf "%02d-%s" "$c" "$name")
    CHAPTER=$c $B render -o "$OUT/chapters/$f-raw.mp4"
    CHAPTER=$c $B audio render -o "$OUT/chapters/$f-mix-raw.wav"
    CHAPTER=$c STEM=amb $B audio render -o "$OUT/chapters/$f-amb-raw.wav"
    CHAPTER=$c STEM=sfx $B audio render -o "$OUT/chapters/$f-sfx-raw.wav"
    GAIN=$(cat "$OUT/stems/gain.txt") $UV "$HERE/sound/master.py" "$OUT/chapters" "$f-"
    nice -n 15 ffmpeg -v error -y -i "$OUT/chapters/$f-raw.mp4" -i "$OUT/chapters/$f-mix.wav" -map 0:v -map 1:a -c:v copy -c:a aac -b:a 320k -movflags +faststart "$OUT/chapters/$f.mp4"
    rm -f "$OUT/chapters/$f-"*.wav "$OUT/chapters/$f-raw.mp4"
    c=$((c + 1))
  done
;; esac
ls -la "$OUT" "$OUT/stems" "$OUT/chapters"
