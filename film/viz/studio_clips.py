#!/usr/bin/env python3
"""Film source clips cut on the Studio straight from the project's data (run from $RS with the project venv).

  cams    <t0> <dur> <tag>            six camera clips, 1448x938, 30 fps, nearest frame on the drive clock
  masks   <t0> <dur> <tag>            front camera at 3840x2160 with the project's sky/road masks and the
                                      moving-object instances (class, moving flag and speed from masks/dyn/*.json)
  genfill <clip dir> <n0> <n1> <tag>  the image model's per-frame input render, published repaired output,
                                      reference photograph and provenance codes as four aligned clips

Everything is written to film/work/clips/. Nothing is re-estimated; this only reads and draws project data.
"""
import bisect, json, os, subprocess, sys
import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = 'film/work/clips'
os.makedirs(OUT, exist_ok=True)
CAMS = ['front', 'left_pillar', 'right_pillar', 'left_repeater', 'right_repeater', 'back']
ACCENT = (255, 123, 59)      # BGR of #3B7BFF
WARM = (214, 230, 237)
GREY = (118, 111, 107)


def writer(path, w, h, fps=30, crf=13):
    return subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{w}x{h}', '-r', str(fps),
                             '-i', '-', '-c:v', 'libx264', '-preset', 'medium', '-crf', str(crf), '-g', '15', '-bf', '0',
                             '-pix_fmt', 'yuv420p', path], stdin=subprocess.PIPE)


def table(cam):
    t = json.load(open(f'data/drive/cams/{cam}.json'))
    return [r['t'] for r in t], [r['n'] for r in t]


def nearest(tab, T):
    ts, ns = tab
    i = min(max(bisect.bisect_left(ts, T), 1), len(ts) - 1)
    return ns[i] if abs(ts[i] - T) < abs(ts[i - 1] - T) else ns[i - 1]


def cams(t0, dur, tag):
    for cam in CAMS:
        tab = table(cam)
        w = writer(f'{OUT}/{tag}-{cam}.mp4', 1448, 938)
        for k in range(round(dur * 30)):
            n = nearest(tab, t0 + k / 30)
            im = cv2.imread(f'data/drive/frames/{cam}/{n:06d}.jpg')
            if im.shape[1] != 1448:
                im = cv2.resize(im, (1448, 938), interpolation=cv2.INTER_AREA)
            w.stdin.write(im.tobytes())
        w.stdin.close(); w.wait()
        print('wrote', cam, flush=True)


def masks(t0, dur, tag):
    tab = table('front')
    kf = json.load(open('depth/keyframes.json'))['frames']['front']
    kts = [r['t'] for r in kf]
    W, H, SH = 3840, 2160, 2488
    yoff = 120
    w = writer(f'{OUT}/{tag}.mp4', W, H)
    mono = ImageFont.truetype(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'BerkeleyMono-Regular.ttf'), 30)
    for k in range(round(dur * 30)):
        T = t0 + k / 30
        n = nearest(tab, T)
        im = cv2.resize(cv2.imread(f'data/drive/frames/front/{n:06d}.jpg'), (W, SH), interpolation=cv2.INTER_CUBIC)
        j = min(max(bisect.bisect_left(kts, T), 1), len(kts) - 1)
        j = j if abs(kts[j] - T) < abs(kts[j - 1] - T) else j - 1
        kn = kf[j]['n']
        sky = cv2.imread(f'masks/sky/front/{kn:06d}.png', cv2.IMREAD_GRAYSCALE)
        dyn = cv2.imread(f'masks/dyn/front/{kn:06d}.png', cv2.IMREAD_UNCHANGED)
        meta = json.load(open(f'masks/dyn/front/{kn:06d}.json')) if os.path.exists(f'masks/dyn/front/{kn:06d}.json') else []
        sx, sy = W / sky.shape[1], SH / sky.shape[0]
        reveal = min(1.0, k / (0.9 * 30))  # scan from left to right over the first 0.9 s
        xr = int(reveal * W)
        over = im.copy()
        skym = cv2.resize(sky, (W, SH), interpolation=cv2.INTER_LINEAR).astype(np.float32)[..., None] / 255
        tint = np.array([70, 34, 14], np.float32)
        over = (over * (1 - 0.55 * skym) + tint * 0.55 * skym).astype(np.uint8)
        edges = cv2.Canny((skym[..., 0] * 255).astype(np.uint8), 60, 160)
        over[cv2.dilate(edges, np.ones((3, 3), np.uint8)) > 0] = (255, 190, 150)
        dynr = cv2.resize(dyn.astype(np.float32), (W, SH), interpolation=cv2.INTER_NEAREST).astype(np.int32)
        labels = []
        placed = []
        for m in meta:
            if m.get('score', 0) < 0.5:
                continue
            sel = dynr == m['id']
            if not sel.any():
                continue
            moving = m.get('moving')
            col = np.array(ACCENT if moving else GREY, np.float32)
            over[sel] = (over[sel] * 0.55 + col * 0.45).astype(np.uint8)
            x0, y0, x1, y1 = m['bbox']
            x0, x1 = int(x0 * sx), int(x1 * sx); y0, y1 = int(y0 * sy), int(y1 * sy)
            L = max(10, min(40, (x1 - x0) // 4))
            c = tuple(int(v) for v in (ACCENT if moving else (200, 200, 200)))
            for (ax, ay, bx, by) in [(x0, y0, x0 + L, y0), (x0, y0, x0, y0 + L), (x1, y0, x1 - L, y0), (x1, y0, x1, y0 + L),
                                     (x0, y1, x0 + L, y1), (x0, y1, x0, y1 - L), (x1, y1, x1 - L, y1), (x1, y1, x1, y1 - L)]:
                cv2.line(over, (ax, ay), (bx, by), c, 3, cv2.LINE_AA)
            if (x1 - x0) > 60 and moving and m.get('speed_mps') is not None:
                txt = f"{m['class']} {m['speed_mps']:.1f} m/s"
                box = (x0, y0 - 50, x0 + 22 * len(txt), y0)
                if all(box[2] < b[0] or box[0] > b[2] or box[3] < b[1] or box[1] > b[3] for b in placed):
                    placed.append(box)
                    labels.append((x0, y0 - 14, txt, c))
        if labels and os.environ.get('CLEAN', '') in ('', '0'):  # CLEAN=1: no class and speed lettering
            pil = Image.fromarray(cv2.cvtColor(over, cv2.COLOR_BGR2RGB))
            d = ImageDraw.Draw(pil)
            for (x, y, txt, c) in labels:
                txt = txt.upper()
                d.text((x, y - 34), txt, font=mono, fill=(c[2], c[1], c[0]), stroke_width=4, stroke_fill=(10, 10, 12))
            over = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
        out = im.copy()
        out[:, :xr] = over[:, :xr]
        if 0 < xr < W:
            cv2.line(out, (xr, 0), (xr, SH), (255, 200, 160), 4, cv2.LINE_AA)
        w.stdin.write(np.ascontiguousarray(out[yoff:yoff + H]).tobytes())
    w.stdin.close(); w.wait()
    print('wrote masks', flush=True)


def genfill(clip, n0, n1, tag):
    meta = json.load(open(f'{clip}/frames.json'))
    x, y, cw, ch = meta['crop']
    fr = {f['n']: f for f in meta['frames']}
    outs = {k: writer(f'{OUT}/{tag}-{k}.mp4', cw, ch) for k in ('render', 'result', 'prov', 'change')}
    # change map: mean absolute RGB difference between the image model's output and its input render,
    # 0 -> background, 40+ levels -> white, through the accent blue
    lut = np.zeros((256, 3), np.float32)
    c_lo, c_mid, c_hi = np.array([14, 12, 11.]), np.array(ACCENT, float), np.array([235, 235, 240.])
    for v in range(256):
        u = min(v / 40.0, 1.0)
        lut[v] = c_lo + (c_mid - c_lo) * (u / 0.6) if u < 0.6 else c_mid + (c_hi - c_mid) * ((u - 0.6) / 0.4)
    outs['ref'] = writer(f'{OUT}/{tag}-ref.mp4', 1448, 938)
    refs = []
    pal = np.zeros((256, 3), np.uint8)
    pal[0] = ACCENT; pal[1] = WARM; pal[2] = GREY; pal[3] = (20, 18, 16)
    for n in range(n0, n1):
        r = cv2.imread(f'{clip}/render/{n:04d}.png')
        b = cv2.imread(f'{clip}/b/{n:04d}.png')
        p = cv2.imread(f'{clip}/b-prov/{n:04d}.png', cv2.IMREAD_GRAYSCALE)
        if r is None or b is None or p is None:
            print('missing', n); continue
        r = r[y:y + ch, x:x + cw] if r.shape[0] != ch else r
        b = b[y:y + ch, x:x + cw] if b.shape[0] != ch else b
        p = p[y:y + ch, x:x + cw] if p.shape[0] != ch else p
        outs['render'].stdin.write(np.ascontiguousarray(r).tobytes())
        outs['result'].stdin.write(np.ascontiguousarray(b).tobytes())
        outs['prov'].stdin.write(np.ascontiguousarray(pal[np.clip(p, 0, 3)]).tobytes())
        diff = np.abs(b.astype(np.int16) - r.astype(np.int16)).mean(2).clip(0, 255).astype(np.uint8)
        outs['change'].stdin.write(np.ascontiguousarray(lut[diff].astype(np.uint8)).tobytes())
        ref = fr[n]['ref']
        im = cv2.imread(f"data/drive/frames/{ref['c']}/{ref['n']:06d}.jpg")
        outs['ref'].stdin.write(cv2.resize(im, (1448, 938), interpolation=cv2.INTER_AREA).tobytes())
        refs.append(dict(n=n, t=fr[n]['t'], ref_cam=ref['c'], ref_n=ref['n'], ref_t=ref['t']))
    for wtr in outs.values():
        wtr.stdin.close(); wtr.wait()
    json.dump(refs, open(f'{OUT}/{tag}-refs.json', 'w'))
    print('wrote genfill', tag, len(refs), flush=True)


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'cams':
        cams(float(sys.argv[2]), float(sys.argv[3]), sys.argv[4])
    elif cmd == 'masks':
        masks(float(sys.argv[2]), float(sys.argv[3]), sys.argv[4])
    elif cmd == 'genfill':
        genfill(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5])
