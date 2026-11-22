"""Verify the streamed final release on the Studio, register it and compare its published chase frames."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

import av
import cv2 as cv
import numpy as np

from transactions import content_signature

NAME = "hw4-cuda-224-254"
REFERENCE = "hw4-raw-224-254"
HOST = os.environ.get("RS_VIEWER_URL", "http://127.0.0.1:8090").rstrip("/")


def read(p):
    return json.loads(p.read_text())


def rows(scene):
    tiles = read(scene / "map.json")["tiles"]
    tracks = read(scene / "objects.json")["objects"]
    summary = {layer: dict(tiles=len(group), splats=sum(t["count"] for t in group))
               for layer in ("ground", "objects")
               for group in [[t for t in tiles if t["layer"] == layer]]}
    classes = [o['class'] for o in tracks]
    summary.update(atlas=len(read(scene / "ground/ground.json")["tiles"]),
                   tracks=len(tracks), classes={c:classes.count(c) for c in sorted(set(classes))})
    for o in tracks:
        k = o['track']; t = np.asarray(k['t']); pos = np.stack([k[a] for a in 'xyz'], axis=1)
        assert len(t) and np.isfinite(pos).all() and np.all(np.diff(t) >= 0)
        assert .1 < min(o['size']) <= max(o['size']) < 25
    return summary


def frame(scene, t):
    e = next(x for x in read(scene/'generated.json')['clips'] if x['t0'] <= t <= x['t1'])
    i = round((t-e['t0'])*e['fps']); assert abs(e['t0']+i/e['fps']-t) < .001
    with av.open(scene/e['file']) as f:
        for j, fr in enumerate(f.decode(video=0)):
            if j == i:
                rgb = fr.to_ndarray(format='bgr24')
                if e.get('provenance') == 'below':
                    assert rgb.shape[:2] == (1800, 1440)
                    rgb = rgb[:900]
                return rgb
    raise ValueError(f'missing published frame at {t}')


def get(url):
    with urllib.request.urlopen(HOST+url, timeout=30) as response: return response.read()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('streamed', type=Path); ap.add_argument('sha256'); ap.add_argument('review',type=Path)
    a=ap.parse_args(); src=a.streamed.resolve(strict=True); home=Path.cwd()
    assert content_signature(src)==a.sha256, 'streamed scene differs from worker receipt'
    entries=sorted(read(src/'generated.json')['clips'],key=lambda e:e['t0']); end=224.
    for e in entries:
        assert abs(end-e['t0']) < .0001 and (src/e['file']).is_file();end=e['t1']
    assert end==254. and read(src/'rsb/index.json')['generated']
    dst=home/'scene'/NAME; ref=home/'scene'/REFERENCE; a.review.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(dst) and dst.resolve() != src:raise ValueError('unrelated Studio destination')
    subprocess.run(['bun',home/'viewer-cache-tool/tools/build-tile-cache.ts',src],check=True)
    if not dst.exists(): dst.symlink_to(src)
    try:
        listing=json.loads(get('/scenes/index.json'))['scenes'];assert any(x['name']==NAME and x['hasMap'] for x in listing)
        assert b'<html' in get('/?scene='+NAME).lower(); mp=read(src/'map.json'); ix=read(src/'rsb/index.json');g=read(src/'ground/ground.json')
        for f in ['map.json','drive.json','rig.json','objects.json','generated.json','rsb/index.json','ground/ground.json', 'rsb/'+ix['tiles'][mp['tiles'][0]['id']]['file'], 'ground/'+g['tiles'][0]['texture'], *[e['file'] for e in entries]]:
            assert hashlib.sha256(get('/scenes/'+NAME+'/'+f)).digest() == hashlib.sha256((src/f).read_bytes()).digest(), f
    except BaseException:
        if dst.is_symlink() and dst.resolve()==src:dst.unlink()
        raise
    comp = dict(reference=rows(ref),cuda=rows(src),chase=[])
    for t in (233.5,235.0,236.5):
        left,right=frame(ref,t),frame(src,t);err=np.abs(left.astype('i2')-right)
        images=[]
        for tag,im in zip(('Studio v4','CUDA v5'),(left,right)):
            shot=cv.resize(im,(720,450),interpolation=cv.INTER_AREA);shot=cv.copyMakeBorder(shot,32,0,0,0,cv.BORDER_CONSTANT,value=(25,25,25));cv.putText(shot,f'{tag} | {t:.3f} s',(8,22),cv.FONT_HERSHEY_SIMPLEX,.6,(235,235,235),1,cv.LINE_AA);images.append(shot)
        assert cv.imwrite(str(a.review/f'chase-{t:.1f}.jpg'),np.hstack(images),[cv.IMWRITE_JPEG_QUALITY,94]);comp['chase'].append(dict(t=t,mae=float(err.mean()),p99=float(np.percentile(err,99))))
    (a.review/'comparison.json').write_text(json.dumps(comp,indent=2));print(json.dumps(comp),flush=True)


if __name__=='__main__': main()
