#!/usr/bin/env python3
"""Bring the camera department's shot library into the film.

Every finished HUD-off shot present in film/captures/ becomes film/work/media/cap-<id>.mp4: a symlink when it is
already 30 fps (so the camera department's offload can still free the disk: a shot moved to the Studio simply
disappears and the film falls back to its placeholder), otherwise a 30 fps H.264 conform (60 fps hero shots play
at real speed in the 30 fps film). Shots the film needs locally are listed in film/captures/keep.txt.
DERIVED cuts a stretch of a shot into its own clip (the renderer plays clips from their first frame), named
cap-<id>@<name>.mp4. A summary of the library goes to film/work/data/captures.json.
usage: python3 film/viz/ingest_captures.py
"""
import json, os, subprocess

FILM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAP = os.path.join(FILM, 'captures')
MEDIA = os.path.join(FILM, 'work', 'media')
# (shot id, start in seconds from the clip's start, length, name)
DERIVED = [
    ('hero-gen-palms-junction-299-329-60fps', 15.0, 7.0, 'open'),     # 314-321 s, the cold open
    ('hero-gen-palms-junction-299-329-60fps', 7.0, 15.0, 'final2'),   # 306-321 s, beside the six cameras (cams306)
    ('hero-gen-100-122-60fps', 5.0, 11.5, 'final1'),                  # 105-116.5 s
    ('hero-gen-100-122-60fps', 13.0, 9.0, 'close'),                   # 113-122 s
    ('hero-gen-redlight-190-206', 6.5, 7.0, 'final3'),                # 196.5-203.5 s, stopped at the light
]


def conform(src, dst, ss=None, t=None):
    cmd = ['nice', '-n', '15', 'ffmpeg', '-v', 'error', '-y']
    if ss is not None:
        cmd += ['-ss', str(ss)]
    cmd += ['-i', src]
    if t is not None:
        cmd += ['-t', str(t)]
    cmd += ['-an', '-vf', 'fps=30,format=yuv420p', '-c:v', 'libx264', '-preset', 'medium', '-crf', '14', '-g', '15', '-bf', '0', dst + '.tmp.mp4']
    subprocess.run(cmd, check=True)
    os.replace(dst + '.tmp.mp4', dst)


m = json.load(open(os.path.join(CAP, 'manifest.json')))
out = []
for name in os.listdir(MEDIA):  # drop links whose shot was offloaded or is a HUD variant
    p = os.path.join(MEDIA, name)
    if name.startswith('cap-') and (name.endswith('-hud.mp4') or (os.path.islink(p) and not os.path.exists(p))):
        os.remove(p)
for s in m.get('shots', []):
    src = os.path.join(CAP, s['file'])
    if s['id'].endswith('-hud') or not os.path.exists(src):
        continue
    dst = os.path.join(MEDIA, 'cap-' + s['id'] + '.mp4')
    if s.get('fps', 30) == 30:
        if not os.path.islink(dst):
            if os.path.exists(dst):
                os.remove(dst)
            os.symlink(os.path.relpath(src, MEDIA), dst)
            print('linked', s['id'])
    elif not os.path.exists(dst) or os.path.getmtime(dst) < os.path.getmtime(src):
        conform(src, dst)
        print('conformed', s['id'], s.get('fps'), '-> 30 fps')
    for (sid, ss, t, name) in DERIVED:
        if sid == s['id']:
            d = os.path.join(MEDIA, f"cap-{sid}@{name}.mp4")
            if not os.path.exists(d) or os.path.getmtime(d) < os.path.getmtime(src):
                conform(src, d, ss, t)
                print('cut', sid, name)
    out.append({k: s.get(k) for k in ('id', 'scene', 'viewer_commit', 'camera', 'hud', 't', 'fps', 'size', 'frames', 'description')})
json.dump(out, open(os.path.join(FILM, 'work', 'data', 'captures.json'), 'w'), indent=1)
print(len(out), 'shots in the library')
