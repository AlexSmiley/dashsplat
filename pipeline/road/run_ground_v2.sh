#!/usr/bin/env bash
set -euo pipefail
export RS="${RS:-$HOME/dev/reality-splat}" GROUND_VERSION=2 GROUND_THREADS="${GROUND_THREADS:-14}"
export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
cd "$RS"; py=.venv/bin/python; work=maps/ground-v2; mkdir -p "$work"
c++ -std=c++17 -O3 -dynamiclib pipeline/road/ground_native.cpp -o "$work/ground.dylib"
"$py" -um pipeline.photo.live_v2
"$py" -um pipeline.road.scene live stitched
"$py" -um pipeline.road.check_scene
"$py" -um pipeline.road.proof_v2
