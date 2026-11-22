#!/usr/bin/env python3
"""Unproject the project's per-keyframe depth maps into coloured world points for the film.

usage: pack_points.py <out.npz> <hero t> <t0> <t1> <every k-th keyframe> <grid stride>   (run from $RS)
Hero: all six cameras at the keyframe nearest <hero t>, full depth grid (724x469), with pixel coordinates kept
so the film can lift the photograph into depth along its own rays. Others: every k-th keyframe in [t0, t1],
every <stride>-th grid sample. World = ENU metres minus the ego position at the hero time (stored as 'origin').
Uses pipeline.common.rig for the calibrated camera models and the refined trajectory; nothing is re-estimated.
"""
import json, sys
import cv2, numpy as np
sys.path.insert(0, '.')
from pipeline.common import rig as R

out, hero_t, t0, t1, every, stride = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
cams = R.load_rig('calib/rig.json')
traj = R.Trajectory('poses/ego.npz')
kf = json.load(open('depth/keyframes.json'))['frames']
origin = traj.T_world_ego(np.array(hero_t))[:3, 3].copy()

def points(cam, rec, s):
    c = cams[cam]
    n, t = rec['n'], rec['t']
    z = np.load(f'depth/{cam}/{n:06d}.npy').astype(np.float32)
    h, w = z.shape
    img = cv2.imread(f'data/drive/frames/{cam}/{n:06d}.jpg')
    img = cv2.cvtColor(cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    dyn = cv2.imread(f'masks/dyn/{cam}/{n:06d}.png', cv2.IMREAD_UNCHANGED)
    ii, jj = np.mgrid[0:h:s, 0:w:s]
    zz = z[ii, jj]
    ok = zz > 0
    ii, jj, zz = ii[ok], jj[ok], zz[ok]
    uv = np.stack(((jj + 0.5) * c.width / w, (ii + 0.5) * c.height / h), -1)
    ray = c.unproject(uv)
    good = np.isfinite(ray).all(-1) & (ray[:, 2] > 1e-3)
    ii, jj, zz, ray = ii[good], jj[good], zz[good], ray[good]
    rng = zz / ray[:, 2]
    Xc = ray * rng[:, None]
    T = R.T_world_cam(c, t, trajectory=traj)
    Xw = Xc @ T[:3, :3].T + T[:3, 3] - origin
    flag = (dyn[ii, jj] > 0).astype(np.uint8) if dyn is not None else np.zeros(len(ii), np.uint8)
    return dict(xyz=Xw.astype(np.float32), rgb=img[ii, jj], dyn=flag, ij=np.stack((ii, jj), -1).astype(np.int16),
                rng=rng.astype(np.float32), T=T, t=t, n=n)

res = {'origin': origin, 'hero_t': hero_t}
for cam in cams:
    recs = kf[cam]
    hero = min(recs, key=lambda r: abs(r['t'] - hero_t))
    p = points(cam, hero, 1)
    for k, v in p.items():
        res[f'hero/{cam}/{k}'] = np.asarray(v)
    sel = [r for r in recs if t0 <= r['t'] <= t1 and r['kind'] == 'distance'][::every]
    acc = [points(cam, r, stride) for r in sel]
    res[f'acc/{cam}/xyz'] = np.concatenate([a['xyz'] for a in acc])
    res[f'acc/{cam}/rgb'] = np.concatenate([a['rgb'] for a in acc])
    res[f'acc/{cam}/dyn'] = np.concatenate([a['dyn'] for a in acc])
    res[f'acc/{cam}/t'] = np.concatenate([np.full(len(a['xyz']), a['t'], np.float32) for a in acc])
    print(cam, 'hero n', hero['n'], 't', round(hero['t'], 3), 'pts', len(p['xyz']), 'acc frames', len(sel), 'pts', len(res[f'acc/{cam}/xyz']), flush=True)
ego = traj.T_world_ego(np.linspace(t0, t1, 200))
res['ego_t'] = np.linspace(t0, t1, 200)
res['ego_T'] = ego
res['T_ego_cam'] = np.stack([cams[c].T_ego_cam for c in cams])
res['cams'] = np.array(list(cams))
np.savez(out, **res)
print('wrote', out)
