#!/bin/bash
set -euo pipefail
source /root/rs-ngc-paths
cd "$RS"; export RS_INSTALL="$PWD" HF_HOME="$PWD/third_party/hf" TORCH_HOME="$PWD/third_party/torch" PYTHONUNBUFFERED=1 DIFIX_RENDER_ONLY_DECODE=1 PIP_NO_CACHE_DIR=1 MAX_JOBS=3
mkdir -p bench/results
python - <<'PY'
import json,os,subprocess
from pathlib import Path
p=Path('/sys/fs/cgroup/cpu.max')
if p.exists(): s=p.read_text().strip()
else:
    d=Path('/sys/fs/cgroup/cpu');s=' '.join((d/f'cpu.cfs_{k}_us').read_text().strip() for k in ('quota','period')) if (d/'cpu.cfs_quota_us').is_file() else 'unknown'
print(json.dumps(dict(receipt='hardware',affinity=len(os.sched_getaffinity(0)),cpu_max=s,model=next(x.split(':',1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name')),gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],text=True).strip())),flush=True)
if s.split()[0].isdigit() and int(s.split()[0])/int(s.split()[1])<6:raise SystemExit('host offers fewer than six CPU cores')
PY
for attempt in 1 2 3 4; do
  if [[ ! -f bench/results/setup.ok ]]; then
    if bash pipeline/cuda/setup-drive.sh >bench/results/setup.log 2>&1; then touch bench/results/setup.ok; else tail -45 bench/results/setup.log >&2; fi
  fi
  if [[ -f bench/results/setup.ok ]] && .venv/bin/python pipeline/cuda/drive_receipts.py; then exit 0; fi
  [[ $attempt -lt 4 ]] || exit 1
  echo '{"receipt":"retry_needed","attempt":'$attempt'}'; deadline=$((SECONDS+600)); while [[ ! -f /root/rs/repair-ready && $SECONDS -lt $deadline ]]; do sleep 2; done
  [[ -f /root/rs/repair-ready ]] || exit 1; rm /root/rs/repair-ready
done
