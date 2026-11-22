"""Geometry checks used by the bake-off and by the production depth tool.

All functions work on pinhole (rectified) depth maps: z-depth in metres on a grid with intrinsics K.
"""
import cv2
import numpy as np


def backproject(depth, K, mask=None):
    """(N,3) camera points and (N,2) integer pixel (row, col) for valid pixels."""
    h, w = depth.shape
    ii, jj = np.nonzero((depth > 0) if mask is None else (mask & (depth > 0)))
    z = depth[ii, jj].astype(np.float64)
    x = (jj + 0.5 - K[0, 2]) / K[0, 0] * z
    y = (ii + 0.5 - K[1, 2]) / K[1, 1] * z
    return np.stack([x, y, z], -1), np.stack([ii, jj], -1)


def fit_road_plane(depth, K, road_mask, up_cam, h_expected, zmax=30.0, iters=300, thresh=0.04, rng=None):
    """RANSAC plane on road pixels. up_cam: expected up direction in camera coords (unit).

    Returns dict(height=distance camera->plane, normal, inliers, n) or None. A depth map with the
    right metric scale gives height == the camera's mounting height above the road."""
    rng = rng or np.random.default_rng(0)
    m = road_mask & (depth > 0.5) & (depth < zmax)
    P, _ = backproject(depth, K, m)
    if len(P) < 200:
        return None
    if len(P) > 20000:
        P = P[rng.choice(len(P), 20000, replace=False)]
    best, best_cnt = None, 0
    for _ in range(iters):
        s = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn
        if n @ up_cam < 0:
            n = -n
        if n @ up_cam < np.cos(np.deg2rad(12)):
            continue
        d = -n @ s[0]
        # relative threshold: scale-free inlier test (residual relative to the plane distance)
        r = np.abs(P @ n + d) / max(abs(d), 1e-3)
        cnt = int((r < thresh).sum())
        if cnt > best_cnt:
            best, best_cnt = (n, d), cnt
    if best is None:
        return None
    n, d = best
    inl = np.abs(P @ n + d) / max(abs(d), 1e-3) < thresh
    Q = P[inl]
    c = Q.mean(0)
    u, s, vt = np.linalg.svd(Q - c, full_matrices=False)
    n = vt[2]
    if n @ up_cam < 0:
        n = -n
    d = -n @ c
    return dict(height=float(abs(d)), normal=n, inliers=int(inl.sum()), n=len(P),
                scale=float(h_expected / abs(d)))


def match_features(img_a, img_b, mask_a=None, mask_b=None, nfeat=6000, ratio=0.8):
    """SIFT matches between two RGB images. Returns (pts_a, pts_b) in COLMAP pixel coords."""
    sift = cv2.SIFT_create(nfeatures=nfeat)
    ga = cv2.cvtColor(img_a, cv2.COLOR_RGB2GRAY)
    gb = cv2.cvtColor(img_b, cv2.COLOR_RGB2GRAY)
    ka, da = sift.detectAndCompute(ga, None if mask_a is None else mask_a.astype(np.uint8) * 255)
    kb, db = sift.detectAndCompute(gb, None if mask_b is None else mask_b.astype(np.uint8) * 255)
    if da is None or db is None or len(ka) < 20 or len(kb) < 20:
        return np.zeros((0, 2)), np.zeros((0, 2))
    bf = cv2.BFMatcher(cv2.NORM_L2)
    mm = bf.knnMatch(da, db, k=2)
    good = [m for m, n in (x for x in mm if len(x) == 2) if m.distance < ratio * n.distance]
    pa = np.array([ka[m.queryIdx].pt for m in good]) + 0.5
    pb = np.array([kb[m.trainIdx].pt for m in good]) + 0.5
    return pa, pb


def relative_pose(pa, pb, K, t_metric, T_ba_prior=None, px=0.75):
    """Two-view relative pose b<-a from matches, translation scaled to the metric baseline |t_metric|.

    Returns (R, t, inlier mask, angle between vision and prior translation direction in degrees)."""
    if len(pa) < 30:
        return None
    try:
        E, inl = cv2.findEssentialMat(pa - 0.5, pb - 0.5, K, cv2.RANSAC, 0.9999, px)
        if E is None or E.shape != (3, 3):
            return None
        _, R, t, inl2 = cv2.recoverPose(E, pa - 0.5, pb - 0.5, K, mask=inl.copy())
    except cv2.error:
        return None
    t = t[:, 0] / np.linalg.norm(t)
    ang = None
    if T_ba_prior is not None:
        tp = T_ba_prior[:3, 3]
        if np.linalg.norm(tp) > 1e-6:
            ang = float(np.degrees(np.arccos(np.clip(t @ tp / np.linalg.norm(tp), -1, 1))))
    return R, t * t_metric, inl2[:, 0] > 0, ang


def triangulate(pa, pb, K, R, t, min_parallax_deg=1.0, max_err_px=1.0):
    """Triangulate matches given pose b<-a. Returns (z_in_b, pixel_b, ok mask)."""
    P0 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P1 = K @ np.hstack([R, t[:, None]])
    X = cv2.triangulatePoints(P0, P1, (pa - 0.5).T, (pb - 0.5).T)
    X = (X[:3] / X[3]).T
    Xb = X @ R.T + t
    ra = X / np.linalg.norm(X, axis=1, keepdims=True)
    cb = -R.T @ t
    rb = (X - cb) / np.linalg.norm(X - cb, axis=1, keepdims=True)
    par = np.degrees(np.arccos(np.clip((ra * rb).sum(1), -1, 1)))

    def proj(Y):
        return (Y[:, :2] / Y[:, 2:3]) * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]

    err = np.maximum(np.linalg.norm(proj(X) - pa, axis=1), np.linalg.norm(proj(Xb) - pb, axis=1))
    ok = (X[:, 2] > 0.5) & (Xb[:, 2] > 0.5) & (par > min_parallax_deg) & (err < max_err_px)
    return Xb[:, 2], pb, ok


def sample(depth, pts, K_pts, K_depth):
    """Nearest-pixel sample of depth at points given in another pinhole grid of the same view."""
    u = (pts[:, 0] - K_pts[0, 2]) / K_pts[0, 0] * K_depth[0, 0] + K_depth[0, 2]
    v = (pts[:, 1] - K_pts[1, 2]) / K_pts[1, 1] * K_depth[1, 1] + K_depth[1, 2]
    j = np.floor(u).astype(int)
    i = np.floor(v).astype(int)
    h, w = depth.shape
    ok = (i >= 0) & (i < h) & (j >= 0) & (j < w)
    out = np.zeros(len(pts))
    out[ok] = depth[i[ok], j[ok]]
    return out


def warp_consistency(d_a, d_b, K, R, t, valid_a, valid_b, rel_tol=0.05):
    """Move frame a's depth into frame b with pose b<-a and compare with b's depth.

    Returns dict(absrel median, absrel mean, frac within rel_tol, n)."""
    P, ij = backproject(d_a, K, valid_a)
    if len(P) == 0:
        return None
    Q = P @ R.T + t
    z = Q[:, 2]
    keep = z > 0.5
    Q, z = Q[keep], z[keep]
    u = Q[:, 0] / z * K[0, 0] + K[0, 2]
    v = Q[:, 1] / z * K[1, 1] + K[1, 2]
    j = np.floor(u).astype(int)
    i = np.floor(v).astype(int)
    h, w = d_b.shape
    ok = (i >= 0) & (i < h) & (j >= 0) & (j < w)
    i, j, z = i[ok], j[ok], z[ok]
    db = d_b[i, j]
    good = valid_b[i, j] & (db > 0)
    # visibility: the warped point must not be hidden behind b's surface by much (occlusions)
    z, db = z[good], db[good]
    if len(z) < 100:
        return None
    rel = np.abs(z - db) / db
    return dict(absrel_med=float(np.median(rel)), absrel_mean=float(np.mean(np.minimum(rel, 1.0))),
                inlier=float((rel < rel_tol).mean()), n=int(len(z)),
                scale_med=float(np.median(z / db)))
