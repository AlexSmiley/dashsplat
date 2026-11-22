#!/usr/bin/env python3
"""Contact sheet of front-camera frames at given drive times (for choosing hero moments).
usage: peek.py <out.jpg> <cam> t1 t2 ...   (run from $RS)"""
import bisect, json, sys
import cv2, numpy as np
out, cam, ts = sys.argv[1], sys.argv[2], [float(x) for x in sys.argv[3:]]
tab = json.load(open(f'data/drive/cams/{cam}.json'))
tt = [r['t'] for r in tab]
tiles = []
for t in ts:
    i = min(bisect.bisect_left(tt, t), len(tt) - 1)
    im = cv2.imread(f"data/drive/frames/{cam}/{tab[i]['n']:06d}.jpg")
    im = cv2.resize(im, (720, int(720 * im.shape[0] / im.shape[1])), interpolation=cv2.INTER_AREA)
    cv2.putText(im, f"t={t:.1f} n={tab[i]['n']}", (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    tiles.append(im)
while len(tiles) % 3:
    tiles.append(np.zeros_like(tiles[0]))
rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]
cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
