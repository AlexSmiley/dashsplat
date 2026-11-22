#!/usr/bin/env python3
"""Cut every source the film uses into film/work/media/ (the folder the renderer loads at runtime).

Reads film/conform.json: a list of entries
  {"name": "out.mp4", "src": "<path from repo root> | studio:<path from $RS>", "ss": 0, "t": 5,
   "crop": [x, y, w, h], "scale": [w, h], "speed": 1.0, "fps": 30}
Videos become 30 fps H.264 (short GOP so the renderer can seek); images are cropped and written as PNG or JPEG.
A studio: source is copied from the Studio once into film/work/studio/. Entries whose output is newer than the
spec and the source are skipped, so re-running is cheap.
usage: python3 film/conform.py [name ...]
"""
import json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT = os.path.join(HERE, 'work', 'media')
CACHE = os.path.join(HERE, 'work', 'studio')
os.makedirs(OUT, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)
spec_path = os.path.join(HERE, 'conform.json')
spec = json.load(open(spec_path))
only = set(sys.argv[1:])


def source(src):
    if src.startswith('studio:'):
        rel = src[len('studio:'):]
        local = os.path.join(CACHE, rel.replace('/', '__'))
        if not os.path.exists(local):
            subprocess.run(['ferry', 'cp', f'anomastudio:dev/reality-splat/{rel}', local], check=True)
        return local
    return os.path.join(REPO, src)


for e in spec:
    name = e['name']
    if only and name not in only:
        continue
    src = source(e['src'])
    if not os.path.exists(src):
        print('MISSING source for', name, '->', e['src'], flush=True)
        continue
    out = os.path.join(OUT, name)
    if os.path.exists(out) and os.path.getmtime(out) > max(os.path.getmtime(src), os.path.getmtime(spec_path)) and not only:
        continue
    vf = []
    if 'crop' in e:
        x, y, w, h = e['crop']
        vf.append(f'crop={w}:{h}:{x}:{y}')
    if 'scale' in e:
        vf.append(f"scale={e['scale'][0]}:{e['scale'][1]}:flags={'neighbor' if e.get('nearest') else 'lanczos'}")
    if e.get('copy'):
        cmd = ['cp', src, out]
    elif name.endswith(('.png', '.jpg')):
        cmd = ['ffmpeg', '-v', 'error', '-y', '-i', src] + (['-vf', ','.join(vf)] if vf else []) + ['-frames:v', '1', '-q:v', '2', out]
    else:
        speed = e.get('speed', 1.0)
        if speed != 1.0:
            vf.append(f'setpts=PTS/{speed}')
        vf.append(f"fps={e.get('fps', 30)}")
        vf.append('format=yuv420p')
        cmd = ['ffmpeg', '-v', 'error', '-y', '-ss', str(e.get('ss', 0)), '-i', src]
        if 't' in e:
            cmd += ['-t', str(e['t'] / speed if speed != 1.0 else e['t'])]
        if speed != 1.0:
            cmd = ['ffmpeg', '-v', 'error', '-y', '-ss', str(e.get('ss', 0)), '-t', str(e['t']), '-i', src]
        cmd += ['-an', '-vf', ','.join(vf), '-c:v', 'libx264', '-preset', 'medium', '-crf', '12', '-g', '15', '-bf', '0', out]
    print(name, '<-', e['src'], flush=True)
    if subprocess.run(cmd).returncode != 0:
        print('FAILED', name, flush=True)
