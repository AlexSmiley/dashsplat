#!/usr/bin/env bash
set -eu -o pipefail
source /root/rs-ngc-paths; cd "$RS"; mkdir -p bench/results envs third_party /work/models
export PYTHONPATH="$RS" HF_HOME="$RS/third_party/hf" PIP_NO_CACHE_DIR=1 HF_HUB_DISABLE_TELEMETRY=1 TORCHINDUCTOR_COMPILE_THREADS=2 PYTHONUNBUFFERED=1 MAX_JOBS=3
unset PIP_CONSTRAINT PIP_BUILD_CONSTRAINT
r=bench/results
# Dedicated SSH stdout for receipts: the safety guard may delete the disk before archive collection.
if [[ ${RS_CHILD_INSTALL:-0} != 1 ]]; then exec 3>&1; fi
export RS_RECEIPT_FD=3
clone() { git init -q "$3" && git -C "$3" fetch -q --depth 1 "$1" "$2" && git -C "$3" checkout -q --detach FETCH_HEAD; }
venv() { python -m venv --system-site-packages "$1" || { python -m pip install virtualenv && python -m virtualenv --system-site-packages "$1"; }; }
python - <<'PY' >bench/results/torch-constraints.txt
from importlib.metadata import version
for k in ('torch','torchvision','triton'):
 try: print(k+'==='+version(k))
 except Exception: pass
PY
setup_legacy() {
 clone https://github.com/nv-tlabs/Difix3D c76edc595586e16732c91ddee82f3a6d83a8a9cc third_party/Difix3D && venv envs/legacy &&
 envs/legacy/bin/pip install -c "$r/torch-constraints.txt" -c pipeline/cuda/constraints-genfill.txt 'numpy==1.26.4' diffusers transformers peft huggingface-hub accelerate safetensors einops &&
 envs/legacy/bin/python - <<'PY'
from huggingface_hub import snapshot_download
from pathlib import Path
p=Path(snapshot_download('nvidia/difix_ref',revision='d4830559772a5795c9d136302c5b197d6418d3fb',ignore_patterns=['*.bin'])); refs=p.parent.parent/'refs';refs.mkdir(exist_ok=True);(refs/'main').write_text(p.name)
PY
}
fixer_setup() {
 clone https://github.com/nv-tlabs/Fixer b39dfcaf4eeec90dc943b057ff368c16252c6c6e third_party/Fixer && venv envs/fixer &&
 envs/fixer/bin/pip install -c "$r/torch-constraints.txt" 'numpy==1.26.4' 'transformers==4.51.3' 'diffusers==0.33.1' 'huggingface-hub==0.32.4' 'accelerate==1.6.0' 'peft==0.14.0' 'qwen-vl-utils==0.0.11' decord 'einops>=0.8.1' einops-exts ninja 'omegaconf==2.3.0' fvcore boto3 loguru natsort imageio imageio-ffmpeg hydra-core mediapy ftfy sentencepiece attrs webdataset pynvml better-profanity ipdb pycocotools termcolor &&
 envs/fixer/bin/pip install -c "$r/torch-constraints.txt" --no-deps --no-build-isolation 'megatron-core==0.10.0' 'cosmos-predict2==1.0.9' &&
 envs/fixer/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download('nvidia/Fixer',revision='ca20a25b55baa15dfc557db4c9b99168da8c2e14',local_dir='/work/models',allow_patterns=['base/*','pretrained/*','README.md'])
PY
}
run() { label=$1; shift; echo "$label start" >&2; if "$@" >"$r/$label.log" 2>&1; then echo "$label OK" >&2; else code=$?; echo "$label: $code" >> "$r/errors.txt"; tail -30 "$r/$label.log" >&2; return "$code"; fi; }
if [[ ${1:-} == install ]]; then
  # All setup, probe and any directed dependency repair share this one timeout.
  begin=$SECONDS
  if fixer_setup && envs/fixer/bin/python pipeline/cuda/round3_fixer.py --width 1024 --stage probe; then exit 0; fi
  printf '{"receipt":"setup_failed","elapsed_s":%d}
' "$((SECONDS-begin))" >&3
  while (( SECONDS-begin < 1780 )); do
    if [[ -f /root/rs/stop-install ]]; then exit 1; fi
    if [[ -f /root/rs/repair-ready ]]; then
      rm -f /root/rs/repair-ready
      if bash /root/rs/repair.sh && envs/fixer/bin/python pipeline/cuda/round3_fixer.py --width 1024 --stage probe; then exit 0; fi
      printf '{"receipt":"setup_failed","elapsed_s":%d}
' "$((SECONDS-begin))" >&3
    fi
    sleep 2
  done
  exit 1
fi
run legacy-setup setup_legacy
run difix envs/legacy/bin/python pipeline/cuda/round3_difix.py
# Keep the dependency log off the pod as it is produced, including a failing import or first real inference.
export RS_CHILD_INSTALL=1
if ! timeout -k 8 1800 bash pipeline/cuda/setup.sh install 3>&3 > >(tee "$r/install.log" >&2) 2>&1; then echo 'Fixer installation/probe failed' >&2; exit 1; fi
printf '{"receipt":"fixer_install_ok"}
' >&3
if ! command -v ffmpeg >/dev/null; then mkdir -p envs/fixer/local-bin; ln -sf "$(envs/fixer/bin/python -c 'import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())')" envs/fixer/local-bin/ffmpeg; export PATH="$RS/envs/fixer/local-bin:$PATH"; fi
for w in 1024 1472; do
  if ! run "speed-$w" timeout -k 8 240 envs/fixer/bin/python pipeline/cuda/round3_fixer.py --width "$w" --stage speed; then
    [[ $w == 1472 ]] || exit 1
    run speed-upscaled envs/fixer/bin/python pipeline/cuda/round3_fixer.py --width 1472 --stage speed --upscale
    touch "$r/upscale"
  fi
done
for stage in three sequence; do
  for w in 1024 1472; do
    extra=(); if [[ $w == 1472 && -f "$r/upscale" ]]; then extra=(--upscale); fi
    run "$stage-$w" timeout -k 8 340 envs/fixer/bin/python pipeline/cuda/round3_fixer.py --width "$w" --stage "$stage" "${extra[@]}"
  done
done
for w in 1024 1472; do
  extra=(); if [[ $w == 1472 && -f "$r/upscale" ]]; then extra=(--upscale); fi
  run "compile-$w" timeout -k 8 360 envs/fixer/bin/python pipeline/cuda/round3_fixer.py --width "$w" --compile "${extra[@]}" || :
done
run difix-compiled timeout -k 8 340 envs/legacy/bin/python pipeline/cuda/round3_difix.py --compiled || :
