#!/usr/bin/env python3
"""UNFOLD shot: the car (the viewer's model, point-sampled) and its six cameras. Each camera's real frame at
t = 111 s unfolds from the lens into its true direction: every pixel is placed along its own calibrated ray
(pipeline/common/rig.py, calib/rig.json), so the fisheye back camera becomes a wide cap and the narrow front
camera a small window. Labels give each camera's name and resolution.
usage: uv run --with torch --with numpy --with opencv-python-headless --with pillow viz/unfold.py <out.mp4> [--test t,...]
"""
import sys, os, math
import cv2, numpy as np, torch
from PIL import Image, ImageDraw, ImageFont
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO); sys.path.insert(0, HERE)
from pipeline.common import rig as RIG
from pointcloud import DEV, Splatter, Writer, ease, smooth_keys, look_at

W, H, FPS, DUR, SS = 3840, 2160, 30, 9.0, 2
WORK = os.path.join(os.path.dirname(HERE), 'work')
HERO = os.path.join(WORK, 'hero111')
FONT = os.path.join(os.path.dirname(HERE), 'render', 'media', 'BerkeleyMono-Regular.ttf')
rig = RIG.load_rig(os.path.join(HERO, 'calib', 'rig.json'))
ORDER = ['front', 'left_pillar', 'right_pillar', 'left_repeater', 'right_repeater', 'back']
NAMES = {'front': 'FRONT', 'left_pillar': 'LEFT PILLAR', 'right_pillar': 'RIGHT PILLAR', 'left_repeater': 'LEFT REPEATER',
         'right_repeater': 'RIGHT REPEATER', 'back': 'BACK'}
D = 4.2
CLEAN = os.environ.get('CLEAN', '') not in ('', '0')  # no camera names: source footage without lettering

cams = {}
for i, c in enumerate(ORDER):
    cam = rig[c]
    gw, gh = 724, 469
    img = cv2.cvtColor(cv2.resize(cv2.imread(os.path.join(HERO, c + '.jpg')), (gw, gh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    jj, ii = np.meshgrid(np.arange(gw), np.arange(gh))
    uv = np.stack(((jj + 0.5) * cam.width / gw, (ii + 0.5) * cam.height / gh), -1).reshape(-1, 2)
    ok = cam.mask_at(uv)
    ray = cam.unproject(uv)
    ok &= np.isfinite(ray).all(-1)
    T = cam.T_ego_cam
    rw = ray[ok] @ T[:3, :3].T
    o = T[:3, 3]
    col = img.reshape(-1, 3)[ok]
    # corner rays for the frustum lines (mask-valid extremes of the image)
    corners = []
    for (u, v) in [(0.04, 0.04), (0.96, 0.04), (0.96, 0.96), (0.04, 0.96)]:
        r = cam.unproject(np.array([[u * cam.width, v * cam.height]]))[0]
        if not np.isfinite(r).all():
            r = cam.unproject(np.array([[(0.5 + (u - 0.5) * 0.8) * cam.width, (0.5 + (v - 0.5) * 0.8) * cam.height]]))[0]
        corners.append(T[:3, :3] @ r)
    centre = T[:3, :3] @ cam.unproject(np.array([[cam.width / 2, cam.height / 2]]))[0]
    foot = (D * math.pi / cam.params[0]) * (cam.width / gw) * 0.62
    cams[c] = dict(o=o, d=rw.astype(np.float32), rgb=col, corners=np.array(corners), centre=centre, foot=foot,
                   label=f"{NAMES[c]}  {cam.width}×{cam.height}")
    print(c, ok.sum(), flush=True)

car = np.fromfile(os.path.join(WORK, 'car-points.bin'), dtype=np.float32).reshape(-1, 9)
L = np.array([0.35, 0.45, 0.82]); L /= np.linalg.norm(L)
shade = np.abs(car[:, 3:6] @ L) * 0.55 + 0.3
car_rgb = (np.clip(shade, 0, 1)[:, None] * np.array([214, 216, 220])).astype(np.float32)

tt = lambda a: torch.as_tensor(np.ascontiguousarray(a.astype(np.float32) if a.dtype == np.float64 else a), device=DEV)
CP, CC = tt(car[:, :3]), tt(car_rgb)
CF = torch.full((len(CP),), 0.009, device=DEV)
# faint polar grid on the ground (rings every 2 m, radial lines every 30 degrees) for scale
gr = []
for r in (2.0, 4.0, 6.0, 8.0):
    a = np.linspace(0, 2 * np.pi, int(r * 900), endpoint=False)
    gr.append(np.stack([1.4 + r * np.cos(a), r * np.sin(a), np.zeros_like(a)], -1))
for a in np.radians(np.arange(0, 360, 30)):
    rr = np.linspace(1.0, 8.0, 5000)
    gr.append(np.stack([1.4 + rr * np.cos(a), rr * np.sin(a), np.zeros_like(rr)], -1))
gr = np.concatenate(gr).astype(np.float32)
GP = tt(gr); GC = torch.full((len(GP), 3), 46.0, device=DEV); GF = torch.full((len(GP),), 0.012, device=DEV)
CP, CC, CF = torch.cat([CP, GP]), torch.cat([CC, GC]), torch.cat([CF, GF])
O = tt(np.concatenate([np.broadcast_to(cams[c]['o'], cams[c]['d'].shape) for c in ORDER]).astype(np.float32))
DIRS = tt(np.concatenate([cams[c]['d'] for c in ORDER]))
COL = tt(np.concatenate([cams[c]['rgb'] for c in ORDER])).float()
FT = tt(np.concatenate([np.full(len(cams[c]['d']), cams[c]['foot'], np.float32) for c in ORDER]))
CID = tt(np.concatenate([np.full(len(cams[c]['d']), i, np.int64) for i, c in enumerate(ORDER)]))
START = [0.6, 1.25, 1.45, 2.05, 2.25, 2.9]
up = np.array([0, 0, 1.0])
KEYS = [
    (0.0, dict(eye=np.array([-4.4, 2.6, 1.9]), target=np.array([1.5, 0, 0.75]), up=up, fov=40.0)),
    (4.2, dict(eye=np.array([-10.5, 7.5, 7.0]), target=np.array([1.4, 0, 0.4]), up=up, fov=50.0)),
    (DUR, dict(eye=np.array([-3.2, 0.6, 17.5]), target=np.array([1.4, 0, 0.0]), up=np.array([1.0, 0, 0]), fov=54.0)),
]
S = Splatter(W * SS, H * SS, rmax=10)
S.set_random(len(CP) + len(O))
font = ImageFont.truetype(FONT, 40)


def project(Rcw, eye, fov, P):
    f = 0.5 * H / math.tan(math.radians(fov) / 2)
    X = (np.asarray(P) - eye) @ Rcw.T
    return np.stack([f * X[..., 0] / X[..., 2] + W / 2, f * X[..., 1] / X[..., 2] + H / 2], -1), X[..., 2]


def frame(t):
    k = smooth_keys(KEYS, t)
    Rcw, eye = look_at(k['eye'], k['target'], k['up'])
    st = torch.tensor(START, device=DEV)[CID]
    s = ((t - st) / 1.5).clamp(0, 1)
    s = torch.where(s < 0.5, 4 * s ** 3, 1 - (-2 * s + 2) ** 3 / 2)
    P = O + DIRS * (0.05 + (D - 0.05) * s)[:, None]
    alpha = (s * 4).clamp(0, 1)
    allP = torch.cat([CP, P]); allC = torch.cat([CC, COL]); allF = torch.cat([CF, FT * (0.15 + 0.85 * s)])
    allA = torch.cat([torch.ones(len(CP), device=DEV), alpha])
    img = S.render(allP, allC, allF, Rcw, eye, k['fov'], scale=1.0, alpha=allA)
    img = torch.nn.functional.avg_pool2d(img.permute(2, 0, 1)[None], SS)[0].permute(1, 2, 0)
    out = Image.fromarray(img.clamp(0, 255).to(torch.uint8).cpu().numpy())
    ov = Image.new('RGBA', (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(ov)
    for i, c in enumerate(ORDER):
        si = ease((t - START[i]) / 1.5)
        if si <= 0:
            continue
        cm = cams[c]
        dist = 0.05 + (D - 0.05) * si
        pts, z = project(Rcw, eye, k['fov'], np.vstack([cm['o'], cm['o'] + cm['corners'] * dist]))
        if (z <= 0.2).any():
            continue
        a = int(120 * min(1, si * 3))
        for j in range(4):
            d.line([tuple(pts[0]), tuple(pts[1 + j])], fill=(235, 232, 225, a // 2), width=2)
        poly = [tuple(p) for p in pts[1:]] + [tuple(pts[1])]
        d.line(poly, fill=(235, 232, 225, a), width=2)
        la = ease((t - START[i] - 0.9) / 0.6)
        if la > 0 and not CLEAN:
            lp, lz = project(Rcw, eye, k['fov'], cm['o'] + cm['centre'] * (D + 0.9))
            if lz > 0.2:
                txt = cm['label']
                tw = d.textlength(txt, font=font)
                d.text((lp[0] - tw / 2, lp[1] - 20), txt, font=font, fill=(235, 232, 225, int(220 * la)))
    out = Image.alpha_composite(out.convert('RGBA'), ov).convert('RGB')
    return np.asarray(out)


if __name__ == '__main__':
    out = sys.argv[1]
    if '--test' in sys.argv:
        for t in [float(x) for x in sys.argv[sys.argv.index('--test') + 1].split(',')]:
            cv2.imwrite(out.replace('.mp4', f'-{t:05.2f}.jpg'), cv2.cvtColor(frame(t), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    else:
        w = Writer(out, W, H, FPS)
        for i in range(int(round(DUR * FPS))):
            w.write(frame(i / FPS))
        w.close()
        print('wrote', out)
