#!/usr/bin/env bash
set -euo pipefail
export RS="${RS:-$HOME/dev/reality-splat}" OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export GROUND_THREADS="${GROUND_THREADS:-14}"
cd "$RS"; py="$RS/.venv/bin/python"; work="$RS/maps/ground-work"; mkdir -p "$work"
c++ -std=c++17 -O3 -dynamiclib pipeline/road/ground_native.cpp -o "$work/ground.dylib"
c++ -std=c++17 -O3 -dynamiclib pipeline/road/discs.cpp -o "$work/discs.dylib"
"$py" -um pipeline.photo.sample
"$py" -c 'from pipeline.photo.sample import collect; collect(True)'
"$py" -um pipeline.photo.fit
"$py" -um pipeline.road.field
"$py" -um pipeline.road.scene live stitched
"$py" -um pipeline.road.check_scene
"$py" -um pipeline.road.render
