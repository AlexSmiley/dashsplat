#!/usr/bin/env bash
# Component worker; run in the versioned Runpod image listed in PORT.md.
set -eu -o pipefail
cd "$RS"
mkdir -p bench/results third_party envs work/depth/{cache,report} maps/object-v3
export PYTHONPATH="$PWD" HF_HOME="$PWD/third_party/hf" MAX_JOBS=4 TORCH_HOME="$PWD/third_party/torch"
python -c 'import torch; print(torch.__version__,torch.version.cuda,torch.cuda.get_device_name()); assert torch.cuda.is_available()'
c++ -O3 -std=c++17 -fPIC -shared pipeline/livemap/raster.cpp -o maps/object-v3/raster.so
python -c "import ctypes; assert ctypes.CDLL('maps/object-v3/raster.so').raster"
clone() { git init -q "$3"; git -C "$3" fetch -q --depth 1 "$1" "$2"; git -C "$3" checkout -q --detach FETCH_HEAD; }
setup_difix() {
 clone https://github.com/nv-tlabs/Difix3D c76edc595586e16732c91ddee82f3a6d83a8a9cc third_party/Difix3D &&
 python -m venv --system-site-packages envs/genfill &&
 envs/genfill/bin/pip install -c pipeline/cuda/constraints-genfill.txt --disable-pip-version-check --no-cache-dir 'numpy==1.26.4' 'diffusers==0.25.1' 'transformers==4.38.0' 'peft==0.9.0' 'huggingface-hub==0.25.1' 'accelerate==0.29.3' safetensors einops &&
 envs/genfill/bin/python - <<'PY'
from huggingface_hub import snapshot_download
from pathlib import Path
p=Path(snapshot_download('nvidia/difix_ref',revision='d4830559772a5795c9d136302c5b197d6418d3fb'));r=p.parent.parent/'refs';r.mkdir(exist_ok=True);(r/'main').write_text(p.name)
PY
}
setup_depth() {
 mkdir -p "$TORCH_HOME/hub" &&
 clone https://github.com/facebookresearch/dinov2 7764ea0f912e53c92e82eb78a2a1631e92725fc8 "$TORCH_HOME/hub/facebookresearch_dinov2_main" &&
 clone https://github.com/ByteDance-Seed/Depth-Anything-3 3d835ec1a5802d64a8b8b15f817a1ab54809bfe4 third_party/Depth-Anything-3 &&
 clone https://github.com/facebookresearch/map-anything 3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9 third_party/map-anything &&
 python -m venv --system-site-packages envs/depth &&
 envs/depth/bin/pip install -c pipeline/cuda/constraints-depth.txt --no-cache-dir 'transformers==4.57.6' 'huggingface-hub<1' 'numpy<2' 'opencv-python-headless==4.10.0.84' scipy einops timm hydra-core orjson natsort python-box plyfile trimesh tensorboard 'uniception==0.1.7' e3nn evo pycolmap addict jaxtyping imageio &&
 envs/depth/bin/pip install --no-deps -e third_party/Depth-Anything-3 -e third_party/map-anything &&
 envs/depth/bin/python - <<'PY'
from huggingface_hub import snapshot_download
from pathlib import Path
for repo,rev in [('depth-anything/DA3-LARGE-1.1','0e109ae307c5982f319a67cf6f9f99ccdc0ec97c'),('facebook/map-anything','a1d87e9086706fb9974f3be5a3e3a0ca5401c5aa'),('facebook/mask2former-swin-large-cityscapes-panoptic','544d76fe93971ee046dacae19b6d4f6ecb5d9088')]:
 p=Path(snapshot_download(repo,revision=rev,ignore_patterns=['*.bin']));r=p.parent.parent/'refs';r.mkdir(exist_ok=True);(r/'main').write_text(p.name)
PY
}
benchargs=
if [[ -e bench/small-only ]]; then benchargs=--small-only; fi
if setup_difix >bench/results/setup-difix.log 2>&1; then
 for dtype in float16 float32; do envs/genfill/bin/python pipeline/cuda/bench_difix.py --dtype "$dtype" $benchargs >"bench/results/$dtype.log" 2>&1 || tail -15 "bench/results/$dtype.log"; done
 if grep -q '"fps"' bench/results/difix.jsonl; then timeout 300 envs/genfill/bin/python pipeline/cuda/bench_difix.py --dtype float16 --compile $benchargs >bench/results/compile.log 2>&1 || tail -8 bench/results/compile.log; fi
else tail -20 bench/results/setup-difix.log; fi
if [[ -e bench/skip-depth ]]; then :
elif setup_depth >bench/results/setup-depth.log 2>&1; then
 envs/depth/bin/python pipeline/depth/bench_e2e.py --t0 233 --k 12 --configs production >bench/results/depth.log 2>&1 && cp work/depth/report/e2e.json bench/results/ || tail -20 bench/results/depth.log
else tail -20 bench/results/setup-depth.log; fi
python -m venv --system-site-packages envs/splat && (
 export PATH="$PWD/envs/splat/bin:/usr/local/cuda/bin:$PATH" CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=2
 pip install -c pipeline/cuda/constraints-splat.txt --no-cache-dir 'numpy==1.26.4' pillow 'plyfile==1.1.2' ninja rich jaxtyping typing_extensions packaging && pip install --no-deps gsplat==1.5.3 && python pipeline/cuda/bench_splat.py
) >bench/results/splat.log 2>&1 || tail -30 bench/results/splat.log
python - <<'PY'
import json
from pathlib import Path
p=Path('bench/results'); rows=[json.loads(line) for line in (p/'difix.jsonl').read_text().splitlines()]
sizes=[[1024,576]] if Path('bench/small-only').exists() else [[1472,960],[1024,576]]
for size in sizes:
 for dtype in ('float16','float32'):
  if not any(r['size']==size and r['dtype']==dtype and r['batch']==1 and not r['compile'] and r.get('fps',0)>0 for r in rows): raise SystemExit(f'missing eager result: {size} {dtype}')
assert json.loads((p/'splat.json').read_text())['iter_s']>0
if not Path('bench/skip-depth').exists(): assert json.loads((p/'e2e.json').read_text())['keyframes']==12
PY
