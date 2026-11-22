import sys, json, glob, cv2, numpy as np, subprocess
sys.path.insert(0, "pipeline/depth")
from rscommon import *
tiles = []
for cam in ["front", "left_pillar", "right_repeater", "back"]:
    best, bn = -1, None
    files = sorted(glob.glob(f"{RS}/masks/dyn/{cam}/*.json"))
    for f in files:
        if f.endswith(("tracks.json", "state.json")): continue
        d = json.load(open(f))
        s = sum(1 for x in d if x["area"] > 300)
        if s > best: best, bn = s, int(f.split("/")[-1][:6])
    out = f"work/depth/tmp/mx_{cam}.jpg"
    subprocess.run([sys.executable, "pipeline/masks/viz_masks.py", cam, str(bn), out], check=True)
    im = cv2.imread(out)
    cv2.putText(im, f"{cam} n={bn}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    tiles.append(im)
g = np.concatenate([np.concatenate(tiles[:2], 1), np.concatenate(tiles[2:], 1)], 0)
cv2.imwrite("work/depth/report/masks_examples.jpg", g, [cv2.IMWRITE_JPEG_QUALITY, 80])
print(g.shape)
