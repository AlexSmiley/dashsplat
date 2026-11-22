#!/usr/bin/env python3
"""DEPTH shot: the front photograph at t=111 s lifts into depth along its own rays, the other five cameras join,
then every third keyframe from 104-120 s adds its points and the cloud thickens towards splats.

Data (all real, from $RS): six frames and depth maps at the hero keyframe (work/hero111), the calibrated rig and
camera poses (pipeline/common/rig.py, points-111.npz from pack_points.py), accumulated keyframe points.
usage: uv run --with torch --with numpy --with opencv-python-headless viz/depthfly.py <out.mp4> [--test t1,t2,...]
"""
import sys, os, math
import cv2, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
from pipeline.common import rig as RIG
from pointcloud import DEV, Splatter, Writer, ease, ease_t, smooth_keys, look_at, pose_to_keys

W, H, FPS = 3840, 2160, 30
WORK = os.path.join(os.path.dirname(HERE), 'work')
HERO = os.path.join(WORK, 'hero111')
CAMS = ['front', 'left_pillar', 'right_pillar', 'left_repeater', 'right_repeater', 'back']
DUR = 13.5

npz = np.load(os.path.join(WORK, 'points-111.npz'))
D0 = 6.0
origin = npz['origin']
rig = RIG.load_rig(os.path.join(HERO, 'calib', 'rig.json'))


def hero_points(cam, up=2):
    c = rig[cam]
    z = np.load(os.path.join(HERO, cam + '.npy')).astype(np.float32)
    h, w = z.shape
    W2, H2 = w * up, h * up
    m = (z > 0).astype(np.float32)
    zs = cv2.resize(z * m, (W2, H2), interpolation=cv2.INTER_LINEAR)
    ms = cv2.resize(m, (W2, H2), interpolation=cv2.INTER_LINEAR)
    zz = np.where(ms > 0.999, zs / np.maximum(ms, 1e-6), 0)
    # drop samples on depth discontinuities (they smear along their rays when seen from the side)
    big = cv2.dilate(zz, np.ones((5, 5), np.uint8))
    small = cv2.erode(np.where(zz > 0, zz, 1e4).astype(np.float32), np.ones((5, 5), np.uint8))
    zz = np.where((big - small) / np.maximum(zz, 1e-3) > 0.3, 0, zz)
    img = cv2.cvtColor(cv2.resize(cv2.imread(os.path.join(HERO, cam + '.jpg')), (W2, H2), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    inside = c.mask_at(np.stack(((np.arange(W2)[None, :] + 0.5) * c.width / W2 + 0 * np.arange(H2)[:, None],
                                  (np.arange(H2)[:, None] + 0.5) * c.height / H2 + 0 * np.arange(W2)[None, :]), -1))
    # depth-less pixels (sky, depth edges) stay on the photograph's plane and fade out; only the front camera keeps them
    keep_all = (zz > 0) | (inside if cam == 'front' else False)
    ii, jj = np.nonzero(keep_all)
    jit = np.random.default_rng(len(ii)).uniform(-0.45, 0.45, (len(ii), 2))
    uv = np.stack(((jj + 0.5 + jit[:, 0]) * c.width / W2, (ii + 0.5 + jit[:, 1]) * c.height / H2), -1)
    ray = c.unproject(uv)
    ok = np.isfinite(ray).all(-1) & (ray[:, 2] > 1e-3)
    ii, jj, ray = ii[ok], jj[ok], ray[ok]
    valid = zz[ii, jj] > 0
    rng = np.where(valid, zz[ii, jj], D0 * ray[:, 2]) / ray[:, 2]
    T = npz[f'hero/{cam}/T'].copy()
    T[:3, 3] -= origin
    rw = ray @ T[:3, :3].T
    fgrid = c.params[0] * W2 / c.width
    return dict(o=np.broadcast_to(T[:3, 3], rw.shape).astype(np.float32), d=rw.astype(np.float32), rng=rng.astype(np.float32),
                valid=valid.astype(np.float32),
                rgb=img[ii, jj], col=(jj / W2).astype(np.float32), foot=(rng / fgrid).astype(np.float32), T=T)


hero = {c: hero_points(c) for c in CAMS}
for c in CAMS:
    print(c, len(hero[c]['rng']), flush=True)

# accumulated keyframes (static content only), subsampled
ego_t, ego_T = npz['ego_t'], npz['ego_T'].copy()
ego_T[:, :3, 3] -= origin
acc_P, acc_C, acc_t, acc_f = [], [], [], []
rs = np.random.default_rng(3)
for c in ['front', 'back']:
    xyz, rgb, dyn, tt = npz[f'acc/{c}/xyz'], npz[f'acc/{c}/rgb'], npz[f'acc/{c}/dyn'], npz[f'acc/{c}/t']
    keep = (dyn == 0) & (rs.random(len(xyz)) < (0.6 if c == 'front' else 0.35)) & (xyz[:, 2] < 14)
    xyz, rgb, tt = xyz[keep], rgb[keep], tt[keep]
    ep = np.stack([np.interp(tt, ego_t, ego_T[:, k, 3]) for k in range(3)], -1)
    rng = np.linalg.norm(xyz - ep, axis=1)
    near = rng < 32
    xyz, rgb, tt, rng = xyz[near], rgb[near], tt[near], rng[near]
    step = 0.45 * ((8 / rig[c].params[0]) if c == 'front' else (4 / rig[c].params[0]))
    acc_P.append(xyz); acc_C.append(rgb); acc_t.append(tt); acc_f.append(rng * step)
acc_P, acc_C, acc_t, acc_f = (np.concatenate(a) for a in (acc_P, acc_C, acc_t, acc_f))
print('acc points', len(acc_P), flush=True)

def T(a):
    a = np.ascontiguousarray(a)
    return torch.as_tensor(a.astype(np.float32) if a.dtype == np.float64 else a, device=DEV)
HO = T(np.concatenate([hero[c]['o'] for c in CAMS]))
HD = T(np.concatenate([hero[c]['d'] for c in CAMS]))
HR = T(np.concatenate([hero[c]['rng'] for c in CAMS]))
HC = T(np.concatenate([hero[c]['rgb'] for c in CAMS])).float()
HF = T(np.concatenate([hero[c]['foot'] for c in CAMS]))
HCOL = T(np.concatenate([hero[c]['col'] for c in CAMS]))
HVAL = T(np.concatenate([hero[c]['valid'] for c in CAMS]))
HCAM = T(np.concatenate([np.full(len(hero[c]['rng']), i, np.int64) for i, c in enumerate(CAMS)]))
AP, AC, AT, AF = T(acc_P), T(acc_C).float(), T(acc_t), T(acc_f)
NH = len(HR)

# depth colour: near warm white -> accent blue -> deep navy (log range 3..120 m)
def depth_colour(r):
    x = ((torch.log(r.clamp(3, 120)) - math.log(3)) / (math.log(120) - math.log(3)))[:, None]
    a = torch.tensor([242., 237., 228.], device=DEV); b = torch.tensor([59., 123., 255.], device=DEV); c = torch.tensor([14., 26., 51.], device=DEV)
    return torch.where(x < 0.45, a + (b - a) * (x / 0.45), b + (c - b) * ((x - 0.45) / 0.55))

HDC = depth_colour(HR)
HC = HC * 1.12 + 6
AC = AC * 1.12 + 6

# car frame at the hero time
i0 = int(np.argmin(abs(ego_t - npz['hero_t'])))
cpos, fwd, left = ego_T[i0, :3, 3], ego_T[i0, :3, 0], ego_T[i0, :3, 1]
up = np.array([0, 0, 1.0])
# the car itself (the viewer's model, point-sampled) at its pose at the hero time
car = np.fromfile(os.path.join(WORK, 'car-points.bin'), dtype=np.float32).reshape(-1, 9)
Lc = np.array([0.35, 0.45, 0.82]); Lc /= np.linalg.norm(Lc)
car_rgb = (np.clip(np.abs(car[:, 3:6] @ Lc) * 0.55 + 0.3, 0, 1)[:, None] * np.array([214, 216, 220])).astype(np.float32)
car_w = car[:, :3] @ ego_T[i0, :3, :3].T + ego_T[i0, :3, 3]
CARP, CARC = T(car_w.astype(np.float32)), T(car_rgb)
CARF = torch.full((len(CARP),), 0.007, device=DEV)
k0 = pose_to_keys(hero['front']['T'], 20)
fov0 = math.degrees(2 * math.atan(H / 2 / (W / 2896 * rig['front'].params[0])))
car = lambda a, l, z: cpos + fwd * a + left * l + up * z
KEYS = [
    (0.0, dict(**k0, fov=fov0)),
    (2.3, dict(**k0, fov=fov0)),
    (6.4, dict(eye=car(-7, 6.5, 4.5), target=car(18, -2, 1.5), up=up, fov=44.0)),
    (9.6, dict(eye=car(-12, -8, 12.0), target=car(12, 0, 0.0), up=up, fov=50.0)),
    (DUR, dict(eye=car(-17, -13, 19.0), target=car(22, 2, 0.0), up=up, fov=50.0)),
]

SS = 2
S = Splatter(int(W * SS), int(H * SS), rmax=12)
S.set_random(NH + len(AP) + len(CARP))


def frame(t):
    k = smooth_keys(KEYS, t)
    Rcw, eye = look_at(k['eye'], k['target'], k['up'])
    # depth wipe on the front photograph (2.3 -> 3.6 s), colours return as points land
    wipe = ease((t - 0.9) / 1.4)
    is_front = HCAM == 0
    # lift: front first, then the other cameras; near points land first
    rn = ((torch.log(HR.clamp(2, 150)) - math.log(2)) / (math.log(150) - math.log(2)))
    start = torch.where(is_front, torch.full_like(HR, 2.4), torch.full_like(HR, 4.2)) + 0.9 * rn
    s = ease_t((t - start) / 2.6)
    dist = D0 + (HR - D0) * s
    P = HO + HD * dist[:, None]
    show_depth = is_front & (HCOL < wipe)
    back_to_photo = ease_t((t - start - 1.6) / 1.2)
    dcol = torch.where((HVAL > 0)[:, None], HDC, torch.tensor([11., 12., 14.], device=DEV).expand_as(HDC))
    C = torch.where(show_depth[:, None], dcol * (1 - back_to_photo[:, None]) + HC * back_to_photo[:, None], HC)
    oth = torch.where(is_front, torch.ones_like(HR), ease_t((t - 3.9 - 0.6 * rn) / 0.8))
    oth = torch.where(HVAL > 0, oth, 1 - ease_t(torch.full_like(HR, (t - 2.5) / 0.9)))
    # footprint: solid photo at rest, points while flying, thick discs at the end
    sc_fly = 1.0 - 0.55 * ease((t - 2.4) / 2.0)
    sc_end = 1.0 + 0.5 * ease((t - 11.5) / 2.0)
    scale = sc_fly * sc_end
    # accumulated keyframes appear in drive order
    at = 8.6 + (AT - 104.0) / 16.0 * 4.2
    aa = ease_t((t - at) / 0.35)
    car_a = torch.full((len(CARP),), ease((t - 3.0) / 1.5), device=DEV)
    P2 = torch.cat([P, AP, CARP]); C2 = torch.cat([C, AC, CARC]); F2 = torch.cat([HF, AF, CARF]); A2 = torch.cat([oth, aa, car_a])
    img = S.render(P2, C2, F2, Rcw, eye, k['fov'], scale=scale, alpha=A2)
    img = torch.nn.functional.avg_pool2d(img.permute(2, 0, 1)[None], SS)[0].permute(1, 2, 0)
    return img


if __name__ == '__main__':
    out = sys.argv[1]
    if '--test' in sys.argv:
        ts = [float(x) for x in sys.argv[sys.argv.index('--test') + 1].split(',')]
        for t in ts:
            img = frame(t).clamp(0, 255).to(torch.uint8).cpu().numpy()
            cv2.imwrite(out.replace('.mp4', f'-{t:05.2f}.jpg'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
            print('frame', t, flush=True)
    else:
        w = Writer(out, W, H, FPS)
        n = int(round(DUR * FPS))
        for i in range(n):
            w.write(frame(i / FPS))
            if i % 30 == 0:
                print(i, '/', n, flush=True)
        w.close()
        print('wrote', out)
