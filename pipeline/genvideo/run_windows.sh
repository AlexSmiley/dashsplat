#!/usr/bin/env bash
# Run from $RS with the prepared, exact-track inputs. Resume only an explicit pressure stop.
set -eo pipefail
export PYTORCH_ENABLE_MPS_FALLBACK=1
export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
GV=work/genvideo
quiet() {
.venv/bin/python - <<'PY'
import re, subprocess, time
last=None; consecutive=0; deadline=time.monotonic()+3600
while consecutive<90 and time.monotonic()<deadline:
    level=int(subprocess.check_output(['sysctl','-n','kern.memorystatus_vm_pressure_level']))
    pct=int(re.search(r'([0-9]+)%',subprocess.check_output(['memory_pressure','-Q'],text=True)).group(1))
    swaps=int(re.search(r'Swapouts: *([0-9]+)',subprocess.check_output(['vm_stat'],text=True)).group(1))
    consecutive=consecutive+10 if level==1 and pct>=35 and last==swaps else 0
    last=swaps
    if consecutive<90:time.sleep(10)
if consecutive<90:raise SystemExit('No quiet 90-second interval within an hour')
PY
}
for win in 000-81 069-81; do
  prev=(); if [[ $win == 069-81 ]]; then prev=(--previous "$GV/track-000-81.mp4"); fi
  out="$GV/track-$win.mp4"; if [[ -f "$GV/track-$win.json" && -s "$out" ]]; then continue; fi
  for attempt in 1 2 3; do
    quiet; log="$GV/run-$win-$attempt.log"
    if envs/genvideo/bin/python -u -m pipeline.genvideo.mps --window "$win" --size 512 320 --steps 20 --cfg 5 --tiled --checkpoint-steps "${prev[@]}" --output "$out" >"$log" 2>&1; then break; fi
    grep -q 'Stopping: shared Mac under sustained paging pressure' "$log" || { tail -20 "$log"; exit 1; }
    echo "Stopped on pressure; window $win, attempt $attempt"; if [[ $attempt == 3 ]]; then exit 1; fi
  done
done
.venv/bin/python -m pipeline.genvideo.evaluate "$GV/track-000-81.mp4" "$GV/track-069-81.mp4"
.venv/bin/python -m pipeline.genvideo.photos "$GV/generated.mp4"
