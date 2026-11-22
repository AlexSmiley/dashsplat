#!/bin/bash
set -euo pipefail
source /root/rs-ngc-paths
cd "$RS"; mkdir -p bench/results
export PYTHONPATH="$RS" STITCH_START=224 STITCH_END=248 STITCH_V2_WORK="$RS/v2" STITCH_V5_WORK="$RS/v5" STITCH_V5_META="$RS/metadata" STITCH_TEACHER="$RS/teacher/manifest.json" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MAX_JOBS=2
unset PIP_CONSTRAINT PIP_BUILD_CONSTRAINT
# The pinned MetalSplat tree supplies the device-neutral spec of projection and tile ordering.
python -c 'import cv2,scipy,numpy' || python -m pip install --no-cache-dir opencv-python-headless scipy
command -v ninja || python -m pip install --no-cache-dir ninja
python -u - <<'PY'
import hashlib,json,subprocess,sys,threading,time
from pathlib import Path
from pipeline.stitch.v5fit import fit,W5,torch
from pipeline.stitch.fit_raster import _native
assert torch.cuda.is_available()
def receipt(name, **values):
    print(json.dumps({'receipt':name,**values}),flush=True)
peak=[0];done=threading.Event()
def sample():
    while not done.is_set():
        out=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
        peak[0]=max(peak[0],int(out.strip()));done.wait(.25)
# Record the one-time CUDA JIT separately so the paired fit clocks start at fit().
tic=time.perf_counter();_native(0);receipt('build',seconds=time.perf_counter()-tic,gpu=torch.cuda.get_device_name(),torch=torch.__version__)
thread=threading.Thread(target=sample,daemon=True);thread.start();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();tic=time.perf_counter()
fit(Path('native'),1600);torch.cuda.synchronize();seconds=time.perf_counter()-tic
done.set();thread.join();scene=W5/'fit';meta=json.loads((scene/'map.json').read_text());stats=json.loads((W5/'fit.json').read_text());receipt('fit',seconds=seconds,steps_per_s=1600/seconds,peak_device_mib=peak[0],peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,fit=stats,counts={l:sum(t['count'] for t in meta['tiles'] if t['layer']==l) for l in ('ground','objects')})
# Emit the exact fitted tile data immediately. The immutable ground is already on the Studio.
receipt('scene',map=meta);ground={}
for tile in meta['tiles']:
    fn=scene/tile['file'];data=fn.read_bytes();sha=hashlib.sha256(data).hexdigest()
    if tile['layer']=='ground':ground[tile['file']]=sha;continue
    print(json.dumps({'media':fn.name,'bytes':len(data),'sha256':sha}),flush=True);sys.stdout.buffer.write(data);sys.stdout.buffer.flush()
receipt('ground_hashes',tiles=ground)
PY
