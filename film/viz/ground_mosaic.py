#!/usr/bin/env python3
"""Compose the published ground atlas (scene/<name>/ground) into one top-down image for the film.

usage: ground_mosaic.py <scene dir> <metres per pixel> <out.png> [x0 y0 x1 y1]
Writes <out.png> (RGBA, alpha = coverage) and <out>.json with the world bounds of the image.
North is up, east is right. Runs on the Studio with the project venv (numpy, OpenCV).
"""
import json, os, sys
import cv2, numpy as np

scene, mpp, out = sys.argv[1], float(sys.argv[2]), sys.argv[3]
g = json.load(open(os.path.join(scene, 'ground', 'ground.json')))
tiles = g['tiles']
x0 = min(t['bounds'][0] for t in tiles); y0 = min(t['bounds'][1] for t in tiles)
x1 = max(t['bounds'][2] for t in tiles); y1 = max(t['bounds'][3] for t in tiles)
if len(sys.argv) > 4:
    x0, y0, x1, y1 = (float(v) for v in sys.argv[4:8])
    tiles = [t for t in tiles if t['bounds'][2] > x0 and t['bounds'][0] < x1 and t['bounds'][3] > y0 and t['bounds'][1] < y1]
W, H = int(np.ceil((x1 - x0) / mpp)), int(np.ceil((y1 - y0) / mpp))
canvas = np.zeros((H, W, 4), np.uint8)
for t in tiles:
    im = cv2.imread(os.path.join(scene, 'ground', t['texture']), cv2.IMREAD_UNCHANGED)
    if im is None:
        continue
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGRA)
    if im.shape[2] == 3:
        im = np.dstack([im, np.full(im.shape[:2], 255, np.uint8)])
    gut = t.get('gutter', 0)
    if gut and im.shape[0] == t['pixels'] + 2 * gut:
        im = im[gut:-gut, gut:-gut]
    im = im[::-1]  # raster row 0 faces south
    bx0, by0, bx1, by1 = t['bounds']
    c0 = int(round((bx0 - x0) / mpp)); c1 = int(round((bx1 - x0) / mpp))
    r0 = int(round((y1 - by1) / mpp)); r1 = int(round((y1 - by0) / mpp))
    if c1 <= c0 or r1 <= r0:
        continue
    small = cv2.resize(im, (c1 - c0, r1 - r0), interpolation=cv2.INTER_AREA if (c1 - c0) < im.shape[1] else cv2.INTER_LINEAR)
    # clip to the canvas
    sc0, sr0 = max(0, -c0), max(0, -r0)
    cc0, cr0, cc1, cr1 = max(c0, 0), max(r0, 0), min(c1, W), min(r1, H)
    if cc1 <= cc0 or cr1 <= cr0:
        continue
    small = small[sr0:sr0 + (cr1 - cr0), sc0:sc0 + (cc1 - cc0)]
    a = small[..., 3:4].astype(np.float32) / 255.0
    dst = canvas[cr0:cr1, cc0:cc1].astype(np.float32)
    dst[..., :3] = small[..., :3] * a + dst[..., :3] * (1 - a)
    dst[..., 3:4] = np.maximum(dst[..., 3:4], small[..., 3:4])
    canvas[cr0:cr1, cc0:cc1] = dst.astype(np.uint8)
cv2.imwrite(out, canvas)
json.dump({'bounds': [x0, y0, x1, y1], 'mpp': mpp, 'size': [W, H], 'tiles': len(tiles)}, open(out + '.json', 'w'))
print(out, W, H, x0, y0, x1, y1)
