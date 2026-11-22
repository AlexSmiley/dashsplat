#!/bin/bash
# One-time setup of the genfill environment on the Studio. Idempotent.
set -euo pipefail
RS=${RS:-$HOME/dev/reality-splat}
cd "$RS"
mkdir -p third_party envs
[ -d third_party/Difix3D ] || git clone --depth 1 https://github.com/nv-tlabs/Difix3D third_party/Difix3D
[ -d third_party/Fixer ] || git clone --depth 1 https://github.com/nv-tlabs/Fixer third_party/Fixer
[ -x envs/genfill/bin/python ] || uv venv --python 3.12 envs/genfill
uv pip install --python envs/genfill/bin/python torch torchvision "numpy<2" \
  "diffusers==0.25.1" "transformers==4.38.0" "peft==0.9.0" "huggingface-hub==0.25.1" \
  accelerate safetensors einops lpips opencv-python-headless scipy pillow
export HF_HOME="$RS/third_party/hf"
envs/genfill/bin/huggingface-cli download nvidia/difix_ref
envs/genfill/bin/huggingface-cli download nvidia/difix
echo SETUP-DONE
