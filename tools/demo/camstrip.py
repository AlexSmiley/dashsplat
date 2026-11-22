#!/usr/bin/env python3
"""Render the six dashcam views as one stacked strip video, time-aligned to the drive clock.

usage: camstrip.py <RS root> <t0> <seconds> <out.mp4> [fps]
Layout (456 px wide): front, [left pillar | right pillar], [left repeater | right repeater], back.
"""
import bisect, json, os, subprocess, sys
import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont

root, t0, dur, out = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
fps = float(sys.argv[5]) if len(sys.argv) > 5 else 30.0
W, GAP = 456, 4
BIG = (W, round(W * 938 / 1448)); SMALL = ((W - GAP) // 2, round((W - GAP) // 2 * 938 / 1448))
rows = [[('front', 'FRONT', BIG)],
        [('left_pillar', 'LEFT PILLAR', SMALL), ('right_pillar', 'RIGHT PILLAR', SMALL)],
        [('left_repeater', 'LEFT REPEATER', SMALL), ('right_repeater', 'RIGHT REPEATER', SMALL)],
        [('back', 'BACK', BIG)]]
H = sum(r[0][2][1] for r in rows) + GAP * (len(rows) - 1)
H += H % 2
tables = {}
for r in rows:
    for cam, _, _ in r:
        t = json.load(open(os.path.join(root, 'data/drive/cams', cam + '.json')))
        tables[cam] = ([x['t'] for x in t], [x['n'] for x in t])
font = ImageFont.truetype('/System/Library/Fonts/SFNSMono.ttf', 11)

def frame(cam, T, size):
    ts, ns = tables[cam]
    i = min(max(bisect.bisect_left(ts, T), 1), len(ts) - 1)
    i = i if abs(ts[i] - T) < abs(ts[i - 1] - T) else i - 1
    img = cv2.imread(os.path.join(root, 'data/drive/frames', cam, f'{ns[i]:06d}.jpg'))
    return cv2.cvtColor(cv2.resize(img, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)

ff = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(fps),
                       '-i', '-', '-c:v', 'libx264', '-crf', '10', '-preset', 'medium', '-pix_fmt', 'yuv444p', out], stdin=subprocess.PIPE)
for k in range(round(dur * fps)):
    T = t0 + k / fps
    canvas = Image.new('RGB', (W, H), (0, 0, 0)); d = ImageDraw.Draw(canvas); y = 0
    for r in rows:
        x = 0
        for cam, label, size in r:
            canvas.paste(Image.fromarray(frame(cam, T, size)), (x, y))
            tw = d.textlength(label, font=font)
            d.rectangle([x + 6, y + size[1] - 22, x + 14 + tw, y + size[1] - 6], fill=(0, 0, 0))
            d.text((x + 10, y + size[1] - 21), label, font=font, fill=(235, 235, 235))
            x += size[0] + GAP
        y += r[0][2][1] + GAP
    ff.stdin.write(np.asarray(canvas).tobytes())
ff.stdin.close(); ff.wait()
print('wrote', out, W, H)
