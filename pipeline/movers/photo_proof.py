"""Raised physical camera proof: real front RGB/depth, source removed, exported PLY drawn.
No generated image, screen-space vehicle paste or per-proof pose/colour correction.
"""

from pathlib import Path
import pickle, json, numpy as np, cv2
from scipy.spatial.transform import Rotation
from scipy.ndimage import distance_transform_edt
from pipeline.common.rig import load_rig, Trajectory, T_world_cam
from .model import interp
from .demo import R
from .photographic import calibration
from .proof import text

OUT = R / "report"
OUT.mkdir(exist_ok=True)
records = []


def remap(im, u, mode=cv2.INTER_LINEAR):
    return cv2.remap(
        im,
        np.nan_to_num(u, nan=-1, posinf=-1, neginf=-1).astype("f"),
        None,
        mode,
        borderMode=cv2.BORDER_REPLICATE,
    )


def donor(clean, m, dep, cam, C, tr, n):
    # Pull real static texels exposed just before/after the vehicle crossed.
    # At the mask only, use nearby static depth with the donor's own mask veto.
    yy, xx = np.where(
        cv2.resize(m, (cam.width, cam.height), interpolation=cv2.INTER_NEAREST) > 0
    )
    src = np.c_[xx + 0.5, yy + 0.5]
    ray = cam.unproject(src)
    ray /= np.maximum(0.05, ray[:, 2:])
    z = dep[yy // 4, xx // 4]
    world = (ray * z[:, None]) @ C[:3, :3].T + C[:3, 3]
    seq = {f["n"]: f for f in json.load(open("data/drive/cams/front.json"))}
    best = np.full(len(xx), np.inf)
    for dd in [-28, 27, -54, 50]:
        k = n + dd
        if k not in seq:
            continue
        D = T_world_cam(cam, seq[k]["t"], trajectory=tr)
        uv, ok = cam.project((world - D[:3, 3]) @ D[:3, :3])
        ix = np.clip(uv[:, 0].astype(int), 0, cam.width - 1)
        iy = np.clip(uv[:, 1].astype(int), 0, cam.height - 1)
        root = Path("masks/dyn/front")
        p = root / f"{k:06}.png"
        if not p.is_file():
            continue
        mm = cv2.dilate(
            (cv2.imread(str(p), -1) > 0).astype("uint8"), np.ones((5, 5), np.uint8)
        )
        ok &= mm[iy // 4, ix // 4] == 0
        score = abs(dd) + 0.015 * np.linalg.norm(uv - src, axis=1)
        sel = ok & (score < best)
        if not sel.any():
            continue
        donor = cv2.imread(f"data/drive/frames/front/{k:06}.jpg")
        col = calibration(False).sample(
            "front", k, uv[sel], donor[iy[sel], ix[sel], ::-1]
        )
        clean[yy[sel], xx[sel]] = col[:, ::-1]
        best[sel] = score[sel]
    return float(np.mean(np.isfinite(best)))


def chase(t):
    if not 93 <= t <= 140:
        return None
    a, b = (93, 110) if t <= 110 else (110, 125) if t <= 125 else (125, 140)
    if (a, b) not in tracks:
        tracks[a, b] = json.load(open(f"work/ibr/task-m/chase-{a}-{b}.json"))["frames"]
    k = int(np.clip(round((t - a) * 30), 0, len(tracks[a, b]) - 1))
    q = tracks[a, b][k]
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(q["quat"]).as_matrix() @ np.diag([1, -1, -1])
    T[:3, 3] = q["pos"]
    return q["t"], T


tracks = {}


def pinhole(w, T):
    v = (w - T[:3, 3]) @ T[:3, :3]
    return v[..., :2] / np.maximum(0.01, v[..., 2:]) * focal + [720, 450]


focal = 450 / np.tan(np.radians(25.0))


def proof(id, bank, objs):
    o = next(o for o in objs if o["id"] == id)
    sp, rows, *_ = pickle.load(open(R / f"fit-{id}.pickle", "rb"))
    rs = [
        r
        for r in rows
        if r["c"] == "front"
        and r["key"]
        and not r.get("held")
        and r["bb"][0] > 3
        and r["bb"][2] < 720
        and Path(f"depth/front/{r['n']:06}.npy").exists()
    ]
    from .shape import silhouette
    from .model import projection

    pp = silhouette(o["class"], o["size"])
    cam = load_rig()["front"]
    tr = Trajectory("poses/ego.npz")

    def quality(r):
        uv, ok, *_ = projection(cam, tr, r["t"], o["track"], pp, pp * 0)
        a = np.r_[uv.min(0), uv.max(0)]
        from .packets import iou

        shot = chase(r["rt"])
        loc = interp(o["track"], r["rt"])[:3] + [0, 0, o["size"][2] * 0.5]
        pt = pinhole(loc, shot[1]) if shot else [720, 450]
        in_frame = 0.015 if not (65 < pt[0] < 1375 and 65 < pt[1] < 835) else 1.0
        return (
            in_frame
            * np.sqrt(np.prod(np.subtract(r["bb"][2:], r["bb"][:2])))
            * iou(a, r["bb"]) ** 2
            * (0.2 if min(a[0], 724 - a[2]) < 3 else 1.0)
        )

    q = max(rs, key=quality)
    n = q["n"]
    t = q["rt"]
    tf = q["t"]
    cam = load_rig()["front"]
    tr = Trajectory("poses/ego.npz")
    C = T_world_cam(cam, tf, trajectory=tr)
    ahead = tr.T_world_ego(t)[:3, 0]
    shift = np.array([0, 0, 1.15]) - ahead * 1.50
    delta = shift @ C[:3, :3]
    center = interp(o["track"], t)
    ca, sa = np.cos(center[3]), np.sin(center[3])
    rot = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
    size = np.array(o["size"])

    def project(w):
        return cam.project((w - (C[:3, 3] + shift)) @ C[:3, :3])[0]

    im = cv2.imread(f"data/drive/frames/front/{n:06}.jpg")
    im = calibration(False).apply("front", n, im[:, :, ::-1])[:, :, ::-1].copy()
    m = cv2.imread(f"masks/dyn/front/{n:06}.png", -1) == q["id"]
    m = cv2.dilate(m.astype("uint8") * 255, np.ones((5, 5), np.uint8))
    extra = Path(f"maps/movers-v1/remove/front/{n:06}.png")
    if extra.exists():
        m |= cv2.resize(
            cv2.imread(str(extra), 0), (724, 469), interpolation=cv2.INTER_NEAREST
        )
    clean = cv2.inpaint(
        im,
        cv2.resize(m, (cam.width, cam.height), interpolation=cv2.INTER_NEAREST),
        4,
        cv2.INPAINT_TELEA,
    )
    dep = np.load(f"depth/front/{n:06}.npy").astype("f")
    valid = (dep > 0.5) & (dep < 250) & (m == 0)
    ix = distance_transform_edt(~valid, return_distances=False, return_indices=True)
    dep = dep[tuple(ix)]
    donor_fraction = donor(clean, m, dep, cam, C, tr, n)
    uv0 = project(center[:3] + [0, 0, size[2] * 0.5])
    bw = (q["bb"][2] - q["bb"][0]) * 4
    cw = int(np.clip(2.65 * bw, 576, 1240))
    h = 320
    w = 640
    ch = cw * h / w
    origin = np.clip(
        uv0 - [cw * 0.5, ch * 0.50], [0, 0], [cam.width - cw, cam.height - ch]
    )
    sy = cw / w
    y, x = np.mgrid[:h, :w]
    native = origin + (np.stack([x, y], -1) + 0.5) * sy
    rays = cam.unproject(native)
    rays = rays / np.maximum(0.03, rays[:, :, 2:])
    uv = native.copy()
    for _ in range(4):
        z = remap(dep, uv / 4 - 0.5)
        uv = cam.project(rays * np.maximum(1, z - delta[2])[:, :, None] + delta)[0]
    original = remap(im, native - 0.5)
    background = remap(clean, uv - 0.5)
    shot = chase(t)
    plate = "raised front RGB-D, alternate native donors + inpaint"
    view_C = C[:3, :3]
    O = C[:3, 3] + shift
    if shot:
        t, T = shot
        view_C = T[:3, :3]
        O = T[:3, 3]
        shift = O - C[:3, 3]
        center = interp(o["track"], t)
        ca, sa = np.cos(center[3]), np.sin(center[3])
        rot = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])

        def project(w):
            return pinhole(w, T)

        mid = project(center[:3] + [0, 0, size[2] * 0.5])
        xy = project(pp @ rot.T + center[:3])
        cw = int(np.clip(2.8 * np.ptp(xy[:, 0]), 150, 760))
        sy = cw / w
        origin = np.clip(mid - [cw / 2, cw / 4], [0, 0], [1440 - cw, 900 - cw / 2])
        native = origin + (np.stack([x, y], -1) + 0.5) * sy
        movie = str(bank / "generated/morph-photo-85-140.mp4")
        cap = cv2.VideoCapture(movie)
        cap.set(cv2.CAP_PROP_POS_FRAMES, round((t - 85) * 30))
        ok, im_chase = cap.read()
        cap.release()
        assert ok and im_chase.shape[:2] == (900, 1440)
        background = remap(im_chase, native - 0.5)
        rays = np.dstack([(native - [720, 450]) / focal, np.ones((h, w))])
        plate = "published morph-photo-85-140.mp4 with matching chase pose"
        donor_fraction = None
    rgb = background.copy()
    amount = 0.0
    # Contact kernel evaluated on the real road plane in object coordinates.
    wr = rays @ view_C.T
    kernel = o["contact_shadow"]
    depth = (center[2] + kernel["offset_xyz"][2] - O[2]) / np.minimum(
        -0.001, wr[:, :, 2]
    )
    world = O + wr * depth[:, :, None]
    rel = (world - center[:3]) @ rot
    sig = np.array(kernel["sigma_xy_m"])
    shadow = kernel["peak_opacity"] * np.exp(
        -0.5 * np.sum((rel[:, :, :2] / sig) ** 2, axis=-1)
    )
    shadow[(wr[:, :, 2] >= -0.001) | (depth < 0)] = 0
    rgb = np.uint8(rgb * (1 - shadow[:, :, None]))
    amount = shadow.max()
    occluder = (
        remap((valid & (m == 0)).astype("uint8"), uv / 4 - 0.5, cv2.INTER_NEAREST) > 0
    )
    staticz = remap(dep, uv / 4 - 0.5) - delta[2]
    rows = []
    for suff in [".pred", ""]:
        a = np.frombuffer(
            (bank / "objects" / f"{id}{suff}.ply")
            .read_bytes()
            .split(b"end_header" + bytes([10]), 1)[1],
            "<f4",
        ).reshape(-1, 24)
        rows.append(a[(a[:, 17] <= t) & (a[:, 18] > t)])
    a = np.concatenate(rows)
    xyz = a[:, :3] @ rot.T + center[:3]
    normal = a[:, 3:6] @ rot.T
    loc = (xyz - O) @ view_C
    z = loc[:, 2]
    face = np.sum((O - xyz) * normal, axis=-1)
    v = (project(xyz) - origin) / sy - 0.5
    scale = np.exp(a[:, 10:13])
    quat = Rotation.from_quat(a[:, 13:17][:, [1, 2, 3, 0]]).as_matrix()
    world_axes = rot @ (quat * scale[:, None, :])
    jac = np.stack(
        [
            (project(xyz + 0.02 * np.eye(3)[k]) - project(xyz)) / (0.02 * sy)
            for k in range(3)
        ],
        axis=2,
    )
    cov = jac @ world_axes
    ev, vec = np.linalg.eigh(cov @ cov.transpose(0, 2, 1))
    ax = np.maximum(0.48, 1.85 * np.sqrt(np.maximum(0, ev)))
    az = np.degrees(np.arctan2(vec[:, 1, 1], vec[:, 0, 1]))
    cols = np.clip((0.5 + 0.28209479 * a[:, 6:9]) * 255, 0, 255)[:, ::-1].astype(int)
    drawn = 0
    for i in np.argsort(-z):
        if (
            z[i] < 0.5
            or face[i] < 0
            or not np.isfinite(v[i]).all()
            or not (-3 < v[i, 0] < w + 3 and -3 < v[i, 1] < h + 3)
        ):
            continue
        ui = np.clip(np.rint(v[i]).astype(int), [0, 0], [w - 1, h - 1])
        xx, yy = ui
        if not shot and occluder[yy, xx] and staticz[yy, xx] < 0.76 * z[i]:
            continue
        cv2.ellipse(
            rgb,
            tuple(np.int32(np.round(v[i] * 16))),
            tuple(np.int32(np.round(ax[i, ::-1] * 16))),
            az[i],
            0,
            360,
            tuple(cols[i].tolist()),
            -1,
            cv2.LINE_AA,
            4,
        )
        drawn += 1
    # Context frame and compositor at identical crop magnification; only the middle/right panels share pose.
    cols = [original, background, rgb]
    labels = [
        "front photograph",
        "published chase photograph" if shot else "raised photo / movers removed",
        "raised photo + exported vehicle",
    ]
    bar = np.full((34, 3 * w, 3), 24, np.uint8)
    for j, name in enumerate(labels):
        text(bar, name, (w * j + 12, 23), scale=0.52)
    sheet = np.vstack([bar, np.hstack(cols)])
    filename = sp["name"] + "-raised.jpg"
    cv2.imwrite(str(OUT / filename), sheet, [cv2.IMWRITE_JPEG_QUALITY, 87])
    rec = dict(
        name=sp["name"],
        file=filename,
        n=n,
        t=t,
        pose_offset_world=shift.tolist(),
        background_crop_width_px=cw,
        scene=str(bank),
        foreground_depth_test=not bool(shot),
        texels_drawn=drawn,
        real_donor_fraction=donor_fraction,
        shadow_peak=float(amount),
        plate=plate,
        model="exported measured and inferred PLY records at exposure",
    )
    assert drawn > 35 and (OUT / filename).stat().st_size < 500000
    return rec


def run():
    bank = Path("scene/stitched-opt-v3-movers-photo-v1")
    oo = json.load(open(bank / "objects.json"))["objects"]
    items = [proof(id, bank, oo) for id in [1, 2, 3]]
    (OUT / "raised-proofs.json").write_text(json.dumps(items, indent=2))
    print(items)


if __name__ == "__main__":
    run()
