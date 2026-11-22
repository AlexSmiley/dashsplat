#!/usr/bin/env bash
set -eo pipefail
export VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1
root=work/genvideo/round2
for name in "$@"; do
  dst="$root/eval/$name"; mkdir -p "$dst"
  anchors=pipeline/genvideo/round2/anchors.json
  if [[ $name == vace ]]; then
    .venv/bin/python -c 'import json,sys;print(json.dumps([[s,i,round(x/2),round(y/2)] for s,i,x,y in json.load(open(sys.argv[1]))]))' "$anchors" > "$dst/anchors-half.json";anchors="$dst/anchors-half.json"
  fi
  .venv/bin/python -m pipeline.genvideo.evaluate "$root/$name.mp4" --out "$dst" --raw "$root/raw" --anchors "$anchors" >"$dst/score.log"
  .venv/bin/python -m pipeline.genvideo.photos "$dst/generated.mp4" "$root" >>"$dst/score.log"
done
