#!/usr/bin/env bash
# From a prepared, licensed Linux/NVIDIA env. Does not provision infrastructure.
set -euo pipefail
bundle=$(cd "$(dirname "$0")" && pwd); cd "${COSMOS_REPO:?Set COSMOS_REPO to the prepared checkout}"; [[ $(git rev-parse HEAD) == 2ff49d0717af02057ae79bc75c00fbff9da1b4e7 ]]; which nvidia-smi >/dev/null
variant=${1:-multi}; [[ $variant == edge || $variant == multi ]]; for f in input edges disparity opaque; do test "$(ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames -of csv=p=0 "$bundle/$f.mp4")" = 81; done
.venv/bin/python examples/inference.py -i "$bundle/$variant.json" -o "$bundle/output-$variant"
