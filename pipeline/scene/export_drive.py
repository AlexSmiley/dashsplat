#!/usr/bin/env python3
"""Write scene/<name>/drive.json for the viewer from the master timeline (and refined poses when present)."""
import argparse, json, os
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('root'); parser.add_argument('name'); parser.add_argument('--poses', default=None, help='Trajectory NPZ, relative to root or absolute')
args = parser.parse_args(); root, name = args.root, args.name
tl = np.load(os.path.join(root, 'data/drive/timeline.npz'))
meta = json.load(open(os.path.join(root, 'data/drive/meta.json')))
t = tl['t']; n = len(t)
x, y, yaw = tl['x'], tl['y'], tl['yaw']
z = np.zeros(n); pitch = np.zeros(n); roll = np.zeros(n)
ego = os.path.join(root, args.poses or 'poses/ego.npz')
if args.poses is not None and not os.path.isfile(ego): raise FileNotFoundError(ego)
if os.path.exists(ego):
    p = np.load(ego); T = p['T_world_ego']; tp = p['t']
    R = T[:, :3, :3]
    yaw_p = np.unwrap(np.arctan2(R[:, 1, 0], R[:, 0, 0]))
    pitch_p = -np.arcsin(np.clip(R[:, 2, 0], -1, 1))
    roll_p = np.arctan2(R[:, 2, 1], R[:, 2, 2])
    x, y, z = (np.interp(t, tp, T[:, i, 3]) for i in range(3))
    yaw, pitch, roll = (np.interp(t, tp, a) for a in (yaw_p, pitch_p, roll_p))
r = lambda a, d: [round(float(v), d) for v in a]
out = {'fps': meta['fps'], 'duration': float(t[-1]), 'origin': {'lat': meta['origin_lat'], 'lon': meta['origin_lon']},
       't': r(t, 4), 'x': r(x, 3), 'y': r(y, 3), 'z': r(z, 3), 'yaw': r(yaw, 5), 'pitch': r(pitch, 5), 'roll': r(roll, 5),
       'speed': r(tl['speed'], 3), 'steer': r(tl['steer'], 1), 'ap': [int(v) for v in tl['ap']],
       'brake': [int(v) for v in tl['brake']], 'blink_l': [int(v) for v in tl['blink_l']],
       'blink_r': [int(v) for v in tl['blink_r']], 'gear': [int(v) for v in tl['gear']],
       'lat': r(tl['lat'], 7), 'lon': r(tl['lon'], 7)}
d = os.path.join(root, 'scene', name); os.makedirs(d, exist_ok=True)
json.dump(out, open(os.path.join(d, 'drive.json'), 'w'), separators=(',', ':'))
print('wrote', os.path.join(d, 'drive.json'), n, 'samples')
