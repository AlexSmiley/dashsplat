#!/usr/bin/env bash
# Guarded round-2 measurements. Export only text logs and statistics.
set -eu -o pipefail
cd "$RS"; mkdir -p envs bench/results third_party work/depth/cache work/depth/report
export PYTHONPATH="$RS" HF_HOME="$RS/third_party/hf" TORCHINDUCTOR_COMPILE_THREADS=2 TORCHINDUCTOR_CACHE_DIR="$RS/third_party/inductor" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
r=bench/results
snap() { "$1" - "$2" "$3" <<'PY'
import sys
from pathlib import Path
from huggingface_hub import snapshot_download
p=Path(snapshot_download(sys.argv[1],revision=sys.argv[2],ignore_patterns=['*.bin'])); root=p.parent.parent/'refs';root.mkdir(exist_ok=True);(root/'main').write_text(p.name)
PY
}
clone() { git init -q "$3" && git -C "$3" fetch -q --depth=1 "$1" "$2" && git -C "$3" checkout -q --detach FETCH_HEAD; }
fix_setup() {
 clone https://github.com/nv-tlabs/Difix3D c76edc595586e16732c91ddee82f3a6d83a8a9cc third_party/Difix3D && python -m venv --system-site-packages envs/genfill &&
 envs/genfill/bin/pip install --no-cache-dir -c pipeline/cuda/constraints-genfill.txt 'numpy==1.26.4' diffusers transformers peft huggingface-hub accelerate safetensors einops &&
 snap envs/genfill/bin/python nvidia/difix_ref d4830559772a5795c9d136302c5b197d6418d3fb && snap envs/genfill/bin/python nvidia/difix 2b0c6fb5797c26b01154dfdeb19d36e5e2eaf388
}
depth_setup() {
 clone https://github.com/ByteDance-Seed/Depth-Anything-3 3d835ec1a5802d64a8b8b15f817a1ab54809bfe4 third_party/Depth-Anything-3 && python -m venv --system-site-packages envs/depth &&
 envs/depth/bin/pip install --no-cache-dir -c pipeline/cuda/constraints-depth.txt 'transformers==4.57.6' 'huggingface-hub<1' 'numpy<2' 'opencv-python-headless==4.10.0.84' scipy einops timm hydra-core orjson natsort python-box plyfile trimesh tensorboard e3nn evo pycolmap addict jaxtyping imageio &&
 envs/depth/bin/pip install --no-deps -e third_party/Depth-Anything-3 &&
 snap envs/depth/bin/python depth-anything/DA3-LARGE-1.1 0e109ae307c5982f319a67cf6f9f99ccdc0ec97c && snap envs/depth/bin/python facebook/mask2former-swin-large-cityscapes-panoptic 544d76fe93971ee046dacae19b6d4f6ecb5d9088
}
run() { label=$1; shift; echo "$label START"; if "$@" >"$r/$label.log" 2>&1; then echo "$label OK"; else code=$?; echo "$label exit $code" >> "$r/errors.txt"; tail -8 "$r/$label.log"; fi; }
if fix_setup >"$r/install-fix.log" 2>&1; then
 for w in 1024 1472; do for ref in 1 0; do run "eager-$w-$ref" timeout -k 10 220 envs/genfill/bin/python pipeline/cuda/profile_difix.py --width "$w" --ref "$ref"; done; done
else tail -15 "$r/install-fix.log"; fi
if depth_setup >"$r/install-depth.log" 2>&1; then
 for task in mask depth; do run "$task" timeout -k 10 340 envs/depth/bin/python pipeline/cuda/profile_depth.py "$task"; done
else tail -15 "$r/install-depth.log"; fi
for ref in 1 0; do for w in 1024 1472; do
 if grep -q '"case": "cudnn_benchmark"' "$r/difix-$w-$ref.jsonl"; then run "compile-$w-$ref" timeout -k 8 300 envs/genfill/bin/python pipeline/cuda/profile_difix.py --width "$w" --ref "$ref" --compile-only; fi
done; done
