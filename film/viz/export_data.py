#!/usr/bin/env python3
"""Small JSON tables the renderer compiles in (film/render/data/): the drive path and telemetry, the commit list,
the tracked vehicles and the ground-atlas image placement. All values are copied from project data.
usage: uv run --with numpy python film/viz/export_data.py
"""
import json, os, subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FILM = os.path.dirname(HERE)
REPO = os.path.dirname(FILM)
D = os.path.join(FILM, 'work', 'data')
OUT = os.path.join(FILM, 'render', 'data')
os.makedirs(OUT, exist_ok=True)

tl = np.load(os.path.join(D, 'timeline.npz'))
ego = np.load(os.path.join(D, 'ego.npz'))
t = ego['t']; p = ego['T_world_ego'][:, :3, 3]
yaw = np.arctan2(ego['T_world_ego'][:, 1, 0], ego['T_world_ego'][:, 0, 0])
step = 36 // 6  # about 6 Hz
idx = np.arange(0, len(t), step)
spd = np.interp(t[idx], tl['t'], tl['speed'])
steer = np.interp(t[idx], tl['t'], tl['steer'])
dist = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1))])
drive = dict(t=np.round(t[idx], 3).tolist(), x=np.round(p[idx, 0], 2).tolist(), y=np.round(p[idx, 1], 2).tolist(),
             yaw=np.round(yaw[idx], 4).tolist(), speed=np.round(spd, 2).tolist(), steer=np.round(steer, 1).tolist(),
             dist=np.round(dist[idx], 1).tolist(), duration=round(float(t[-1]), 3), length_m=round(float(dist[-1]), 1),
             speed_max=round(float(tl['speed'].max()), 2))
json.dump(drive, open(os.path.join(OUT, 'drive.json'), 'w'), separators=(',', ':'))

log = subprocess.run(['git', '-C', REPO, 'log', '--reverse', '--format=%h|%ad|%s', '--date=format:%Y-%m-%d %H:%M'],
                     capture_output=True, text=True, check=True).stdout.strip().splitlines()
commits = []
t0 = None
for line in log:
    h, d, s = line.split('|', 2)
    day, hm = d.split(' ')
    hh, mm = map(int, hm.split(':'))
    minutes = hh * 60 + mm + (24 * 60 if day.endswith('22') else 0)
    t0 = minutes if t0 is None else t0
    commits.append(dict(hash=h, time=hm, minute=minutes - t0, subject=s))
# the film covers the first day and night: 15:45 to 04:15 (commit 3b7a42b)
end = next(i for i, c in enumerate(commits) if c['hash'] == '3b7a42b')
commits = commits[: end + 1]
json.dump(commits, open(os.path.join(OUT, 'commits.json'), 'w'), indent=0)

obj = json.load(open(os.path.join(D, 'objects.json')))
movers = []
for o in obj['objects']:
    tr = o['track']
    movers.append(dict(id=o['id'], cls=o['class'], size=o['size'], t=[round(v, 2) for v in tr['t'][::2]],
                       x=[round(v, 2) for v in tr['x'][::2]], y=[round(v, 2) for v in tr['y'][::2]],
                       yaw=[round(v, 3) for v in tr['yaw'][::2]], state=tr['state'][::2]))
json.dump(movers, open(os.path.join(OUT, 'movers.json'), 'w'), separators=(',', ':'))
g = json.load(open(os.path.join(FILM, 'work', 'ground-v5-0.2.png.json')))
json.dump(g, open(os.path.join(OUT, 'ground.json'), 'w'))
print('drive', len(idx), 'samples', drive['length_m'], 'm', drive['duration'], 's; commits', len(commits), 'movers', len(movers))
