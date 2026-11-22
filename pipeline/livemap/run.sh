#!/usr/bin/env bash
set -euo pipefail
export RS="${RS:-$HOME/dev/reality-splat}" OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=2
cd "$RS"; p=.venv/bin/python; w=maps/object-v3; mkdir -p "$w"
compile() { for n in fusion raster; do c++ -O3 -std=c++17 -dynamiclib "pipeline/livemap/$n.cpp" -o "$w/$n.new.dylib"; mv "$w/$n.new.dylib" "$w/$n.dylib"; done; }
compile_now() { mkdir -p maps/live-v4; for n in now raster; do c++ -O3 -std=c++17 -dynamiclib "pipeline/livemap/$n.cpp" -o "maps/live-v4/$n.new.dylib"; mv "maps/live-v4/$n.new.dylib" "maps/live-v4/$n.dylib"; done; }
case "${1:?inputs, audit, front, sides, back, final or proof}" in
 inputs) "$p" -um pipeline.livemap.inputs ;;
 audit) for m in stitched live; do "$p" -um pipeline.livemap.audit "$m"; done; "$p" -um pipeline.livemap.causal ;;
 front|sides|back|final) compile; for m in stitched live; do "$p" -um pipeline.livemap.build "$m" "$1"; if [[ "$1" != final ]]; then "$p" -um pipeline.livemap.render "$m" "$1"; fi; done ;;
 proof) LIVEMAP_CUTOFF=101 "$p" -um pipeline.livemap.causal; LIVEMAP_CUTOFF=101 "$p" -um pipeline.livemap.build live final; "$p" -um pipeline.livemap.proof; "$p" -um pipeline.livemap.report ;;
 v4-inputs) "$p" -um pipeline.livemap.v4_inputs slice ;;
 v4-final) compile_now; "$p" -um pipeline.livemap.v4_build slice; "$p" -um pipeline.livemap.v4_verify slice ;;
 v4-proof) "$p" -um pipeline.livemap.v4_proof; "$p" -um pipeline.livemap.v4_temporal; "$p" -um pipeline.livemap.v4_report ;;
 v4-whole) export V4_DEST="$RS/scene/live-v4-whole"; compile_now; "$p" -um pipeline.livemap.v4_inputs whole; "$p" -um pipeline.livemap.v4_build whole; "$p" -um pipeline.livemap.v4_verify whole ;;
 *) exit 2 ;;
esac
