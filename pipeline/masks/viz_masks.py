import sys, json, cv2, numpy as np
sys.path.insert(0, "pipeline/depth")
from rscommon import *
cam, n = sys.argv[1], int(sys.argv[2])
im = load_image(cam, n, (724, 469))
sky = cv2.imread(f"{RS}/masks/sky/{cam}/{n:06d}.png", 0) > 0
road = cv2.imread(f"{RS}/masks/road/{cam}/{n:06d}.png", 0) > 0
ids = cv2.imread(f"{RS}/masks/dyn/{cam}/{n:06d}.png", cv2.IMREAD_UNCHANGED)
info = json.load(open(f"{RS}/masks/dyn/{cam}/{n:06d}.json"))
v = im.copy()
v[sky] = (0.5 * v[sky] + [0, 0, 128]).astype(np.uint8)
v[road] = (0.6 * v[road] + [0, 80, 0]).astype(np.uint8)
for d in info:
    m = ids == d["id"]
    col = np.array(np.random.RandomState(d["id"]).randint(80, 255, 3))
    v[m] = (0.4 * v[m] + 0.6 * col).astype(np.uint8)
    x0, y0 = d["bbox"][:2]
    cv2.putText(v, f'{d["id"]} {d["class"]}', (x0, max(12, y0 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
cv2.imwrite(sys.argv[3], v[..., ::-1])
