#!/usr/bin/env bash
# Clean nvcr.io/nvidia/pytorch:25.03-py3 development image, root, x86_64.
set -euo pipefail
cd "$(dirname "$0")/../.."; export RS_INSTALL="$PWD" HF_HOME="$PWD/third_party/hf" TORCH_HOME="$PWD/third_party/torch" DEBIAN_FRONTEND=noninteractive
unset PIP_CONSTRAINT PIP_BUILD_CONSTRAINT HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
apt-get update -qq; apt-get install -y --no-install-recommends libceres-dev libeigen3-dev cmake g++ ninja-build ffmpeg git ca-certificates unzip curl procps python3-venv
mkdir -p tools/run-chase envs third_party work/road/build work/ibr/m2 maps/object-v3 scene
clone() { if ! test -d "$3/.git"; then git init -q "$3"; git -C "$3" fetch -q --depth 1 "$1" "$2"; git -C "$3" checkout -q --detach FETCH_HEAD; fi; test "$(git -C "$3" rev-parse HEAD)" = "$2"; }
venv() { test -x "$1/bin/python" || python -m venv --system-site-packages "$1" || { python -m pip install virtualenv; python -m virtualenv --system-site-packages "$1"; }; }
venv .venv; venv envs/genfill; ln -sfn ../.venv envs/stitch
clone https://github.com/ByteDance-Seed/Depth-Anything-3 3d835ec1a5802d64a8b8b15f817a1ab54809bfe4 third_party/Depth-Anything-3
clone https://github.com/tchauffi/metalsplat 84e8c3a433f71c107e9387d3f72a765436c11f6e third_party/metalsplat
clone https://github.com/nv-tlabs/Difix3D c76edc595586e16732c91ddee82f3a6d83a8a9cc third_party/Difix3D
# Retain NGC's matching CUDA torch/torchvision rather than pulling a second CUDA runtime.
python - <<'PY' >envs/ngc.txt
from importlib.metadata import PackageNotFoundError, version
for name in ('torch', 'torchvision'):
    print(name+'==='+version(name))
# NGC may bundle Triton without a standalone distribution named triton.
for name in ('triton', 'pytorch-triton'):
    try: print(name+'==='+version(name))
    except PackageNotFoundError: pass
PY
grep -v '^torchvision==' pipeline/cuda/constraints-depth.txt >envs/depth-constraints.txt
.venv/bin/pip install --no-cache-dir -c envs/ngc.txt -c envs/depth-constraints.txt 'numpy==1.26.4' 'opencv-python-headless==4.10.0.84' 'transformers==4.57.6' 'huggingface-hub<1' scipy pillow av einops timm hydra-core orjson natsort python-box plyfile trimesh e3nn evo pycolmap addict jaxtyping imageio sentencepiece protobuf matplotlib safetensors ninja
.venv/bin/pip install --no-deps -e third_party/Depth-Anything-3
 envs/genfill/bin/pip install --no-cache-dir -c envs/ngc.txt -c pipeline/cuda/constraints-genfill.txt 'numpy==1.26.4' 'opencv-python-headless==4.10.0.84' scipy av pillow diffusers transformers peft huggingface-hub accelerate safetensors einops
.venv/bin/python pipeline/cuda/drive_models.py
cmake -S pipeline/calib -B work/calib/build -DCMAKE_BUILD_TYPE=Release; cmake --build work/calib/build -j4
cmake -S pipeline/poses -B work/poses/build -DCMAKE_BUILD_TYPE=Release; cmake --build work/poses/build -j4; ln -sf ../../poses/build/trajectory_ba work/road/build/trajectory_ba
c++ -O3 -std=c++17 -shared -fPIC pipeline/ibr/m2_surface.cpp -o work/ibr/m2/m2_surface.dylib
c++ -O3 -std=c++17 -shared -fPIC pipeline/livemap/raster.cpp -o maps/object-v3/raster.dylib
if ! command -v bun || [[ $(bun --version) != 1.4.4 ]]; then curl -fsSL --retry 2 https://github.com/oven-sh/bun/releases/download/bun-v1.4.4/bun-linux-x64.zip -o /tmp/rs-bun.zip; unzip -op /tmp/rs-bun.zip bun-linux-x64/bun >/usr/local/bin/bun;chmod 755 /usr/local/bin/bun; fi
(cd viewer && bun install --frozen-lockfile && bun --bun x vite build); ln -sfn viewer viewer-cache-tool; bun build viewer/tools/chase-track.ts --target=bun --minify --outdir=tools/run-chase
.venv/bin/python -c 'import av,cv2,torch,scipy,transformers; assert torch.cuda.is_available(); print(torch.__version__,torch.cuda.get_device_name())'
