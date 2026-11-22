#!/usr/bin/env bash
set -euo pipefail
export RS="${RS:-$HOME/dev/reality-splat}" OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
cd "$RS"; py="$RS/.venv/bin/python"; mkdir -p work/road
for stem in ego ego_causal; do
  if [[ ! -f poses/${stem}_v1.npz ]]; then
    "$py" -c 'import numpy as n,sys; assert "road_pitch" not in n.load(sys.argv[1])' "poses/$stem.npz"
    cp "poses/$stem.npz" "poses/${stem}_v1.npz"
  fi
done
cmake -S pipeline/poses -B work/road/build -DCMAKE_BUILD_TYPE=Release
cmake --build work/road/build -j2
"$py" -um pipeline.road.pairs
"$py" -um pipeline.road.mosaic planar offline-v1 causal-v1
"$py" - <<'PY'
from pipeline.road.display import SEG,segment
for seg in SEG:segment('planar',*seg)
PY
for stage in fit poses evaluate diagnose deliver; do "$py" -um "pipeline.road.$stage"; done
"$py" -um pipeline.road.mosaic offline causal
"$py" -um pipeline.road.display
"$py" -um pipeline.road.report
