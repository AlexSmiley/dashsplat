#!/usr/bin/env bash
set -euo pipefail
export RS="${RS:-$HOME/dev/reality-splat}" OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
cd "$RS"
PY="$RS/.venv/bin/python"
cmake -S pipeline/poses -B work/poses/build -DCMAKE_BUILD_TYPE=Release
cmake --build work/poses/build -j2
for stage in tracks wide causal offline validate timing roadcheck roadscore report; do
  "$PY" -um "pipeline.poses.$stage"
done
