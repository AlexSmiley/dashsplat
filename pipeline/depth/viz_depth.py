import sys, json, cv2, numpy as np
sys.path.insert(0, "pipeline/depth")
from rscommon import *
rows = []
for cam, n in [(a.split(":")[0], int(a.split(":")[1])) for a in sys.argv[2:]]:
    z = np.load(f"{RS}/depth/{cam}/{n:06d}.npy").astype(np.float32)
    im = load_image(cam, n, (z.shape[1], z.shape[0]))
    rows.append(np.concatenate([im, colorize_depth(z, 1, 60)], 1))
cv2.imwrite(sys.argv[1], np.concatenate(rows, 0)[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 80])
