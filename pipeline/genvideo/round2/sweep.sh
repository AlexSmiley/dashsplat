#!/usr/bin/env bash
set -eo pipefail
export PYTORCH_ENABLE_MPS_FALLBACK=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1
p=work/genvideo/round2
for setting in pure medium tight; do
 case "$setting" in pure) sigma=1; scale=1;; medium) sigma=.5; scale=1.25;; tight) sigma=.2; scale=1.25;; esac
 steps=20; [[ $setting != tight ]] || steps=8
 more=(); [[ $setting != tight ]] || more=(--codec-out "$p/codec.mp4")
 if [[ -s "$p/$setting.mp4" && -f "$p/$setting.json" ]]; then continue; fi
 for attempt in 1 2 3; do
  .venv/bin/python - <<'PY'
import re,subprocess,time
last=None; quiet=0; deadline=time.monotonic()+3600
while quiet<12 and time.monotonic()<deadline:
 level=int(subprocess.check_output(['sysctl','-n','kern.memorystatus_vm_pressure_level']));out=subprocess.check_output(['vm_stat'],text=True);swaps=int(re.search('Swapouts: *([0-9]+)',out).group(1));head=int(re.search('([0-9]+)%',subprocess.check_output(['memory_pressure','-Q'],text=True)).group(1));quiet=quiet+1 if level==1 and last==swaps and head>=48 else 0;last=swaps
 if quiet<12:time.sleep(10)
assert quiet>=12,'No quiet window on Studio'
PY
  log="$p/$setting-$(date +%s)-$attempt.log"
  if envs/genvideo/bin/python -u -m pipeline.genvideo.mps --inputs "$p/inputs" --window 000-81 --size 832 480 --refs 6 --sigma "$sigma" --cond-scale "$scale" --cfg 1 --steps "$steps" --seed 42 --tiled --checkpoint-steps --cache-controls --query-chunk 192 "${more[@]}" --output "$p/$setting.mp4" >"$log" 2>&1; then break; fi
  grep -q 'Stopping: shared Mac' "$log" || { tail -18 "$log"; exit 1; }; [[ $attempt != 3 ]] || exit 1
 done
done
