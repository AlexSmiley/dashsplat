"""Front-only chase pseudo-views. No inpainting, display tiles or synthesized colours."""

from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor
import argparse, csv, multiprocessing as mp, os
from scipy.spatial.transform import Rotation, Slerp
from .morph_sources import (
    RS,
    TR,
    PHOTO,
    dt,
    ds,
    seq,
    track,
    choose,
    rig,
    coord,
    proj,
    resize,
    remap,
    blur,
    smooth,
    J,
    savej,
    cv,
    np,
    OW,
    OH,
    L,
    H,
    XY,
    F,
)
from .morph_m2 import View, raster
from .morph_ghost import edges, refmask
from pipeline.stitch.v2data import schedule, physical_range, masks

D = RS / os.getenv("TEACHER_OUT", "work/ibr/teacher-85-140")
START = float(os.getenv("TEACHER_START", 85))
END = float(os.getenv("TEACHER_END", 140))
C = rig["front"]
K = [float(F), float(F), 720.0, 450.0]
SW, SH = 362, 234
TYPES = ["chase", "left2", "right2", "up1.5", "down1.5", "yawLeft8", "yawRight8"]
MIN_OFFSET = 0.025
PARS = [cv.IMWRITE_PNG_COMPRESSION, 1]
ALL = coord(SW, SH)
RAYS = C.unproject((ALL + 0.5) * [C.width / SW, C.height / SH]).astype("f")
FORM = np.ones((3, 3), "B")


def stations():
    fr = []
    for a, b in ([(START, END)] if "TEACHER_START" in os.environ else [(85, 93), (93, 110), (110, 125), (125, 140)]):
        fr += track(a, b)[int(bool(fr)) :]
    times = np.array([t for t, _ in fr])
    T = np.array([p for _, p in fr])
    arc = choose(fr)
    s0, s1 = np.interp([START, END], dt, ds)
    st = np.r_[np.arange(s0, s1, 1.0), s1]
    ts = np.interp(st, ds, dt)
    ts[[0, -1]] = [START, END]
    rs = Slerp(times, Rotation.from_matrix(T[:, :3, :3]))(ts).as_matrix()
    xyz = np.column_stack([np.interp(ts, times, T[:, j, 3]) for j in range(3)])
    delay = np.interp(ts, times, arc)
    rows = []
    for j, (t, d, r, p, s) in enumerate(zip(ts, st, rs, xyz, delay)):
        for kind in TYPES if j % 5 == 0 else TYPES[:1]:
            A = np.eye(4)
            A[:3, :3] = r
            A[:3, 3] = p
            right = r[:, 0].copy()
            right[2] = 0
            right /= np.linalg.norm(right)
            if kind in TYPES[1:3]:
                A[:3, 3] += right * (2 if kind == "right2" else -2)
            elif kind in TYPES[3:5]:
                A[2, 3] += 1.5 if kind == "up1.5" else -1.5
            elif kind in TYPES[5:]:
                A[:3, :3] = (
                    Rotation.from_rotvec(
                        [0, 0, np.deg2rad(8 if kind == "yawLeft8" else -8)]
                    ).as_matrix()
                    @ r
                )
            rows.append(
                dict(
                    id=f"{j:04d}-{kind}",
                    station=j,
                    kind=kind,
                    t=float(t),
                    travel_m=float(d - s0),
                    source_arc_m=float(s),
                    K=K,
                    T_world_cam=A.tolist(),
                )
            )
    return rows


@lru_cache(None)
def inputs():
    rows = schedule()["front"]
    held = {e["n"] for e in rows if e["hold"]}
    good = [e for e in rows if e["depth"] and not e["hold"]]
    pre = J(RS / "depth/keyframes.json")["frames"]["front"]
    idx = J(RS / "depth/front/index.json")
    for e in pre:
        if not float(os.getenv("IBR_SOURCE_START", 79.5)) < e["t"] < good[0]["t"]:
            continue
        path = RS / "depth/front" / f"{e['n']:06}.npy"
        ent = idx.get(str(e["n"]), idx.get(f"{e['n']:06}", {}))
        if (
            path.exists()
            and not set(ent.get("window", [])) & held
            and (RS / "masks/sky/front" / f"{e['n']:06}.png").exists()
        ):
            good.append(dict(**e, depth=str(path)))
    good.sort(key=lambda e: e["t"])
    ts, ns, poses, arc = seq("front")
    ok = np.array(
        [
            i
            for i, (t, n) in enumerate(zip(ts, ns))
            if t >= good[0]["t"]
            and n not in held
            and (RS / "maps/movers-v1/remove/front" / f"{n:06}.png").is_file()
        ]
    )
    return good, np.array([e["t"] for e in good]), ok


def select(e):
    ts, ns, S, arc = seq("front")
    keys, kt, ids = inputs()
    t = e["t"]
    ids = ids[(ts[ids] <= t) & (ts[ids] >= t - 16)]
    assert len(ids) > 1
    s = float(np.clip(e["source_arc_m"], arc[ids[0]], arc[ids[-1]]))
    k = np.clip(np.searchsorted(arc[ids], s, side="right"), 1, len(ids) - 1)
    a, b = ids[k - 1 : k + 1]
    mix = float(np.clip((s - arc[a]) / max(1e-7, arc[b] - arc[a]), 0, 1))
    base, other, blend = (a, b, mix) if mix <= 0.5 else (b, a, 1 - mix)
    u = int(np.clip(np.searchsorted(kt, ts[base]), 1, len(keys) - 1))
    return base, other, blend, u - 1, u


@lru_cache(12)
def photo(i):
    ts, ns, _, _ = seq("front")
    n = int(ns[i])
    im = cv.imread(str(RS / "data/drive/frames/front" / f"{n:06}.jpg"))
    mask = cv.imread(str(RS / "maps/movers-v1/remove/front" / f"{n:06}.png"), 0)
    assert im.shape[:2] == (C.height, C.width) and mask is not None
    im = PHOTO.apply("front", n, im[:, :, ::-1])[:, :, ::-1].copy()
    return im, cv.dilate(
        cv.resize((mask > 0).astype("f"), (SW, SH), interpolation=cv.INTER_AREA), FORM
    )


@lru_cache(32)
def rowposes(t):
    return (
        TR.T_world_ego(C.row_time(t, (np.arange(SH) + 0.5) * C.height / SH))
        @ C.T_ego_cam
    ).astype("f")


@lru_cache(24)
def key(j):
    e = inputs()[0][j]
    ts = e["t"]
    rr, wt = physical_range("front", np.load(e["depth"]).astype("f"))
    rr = resize(rr, SW, SH)
    wt = resize(wt, SW, SH)
    sky, dyn, road = [
        resize((a > 0).astype("f"), SW, SH) for a in masks("front", e["n"])
    ]
    P = rowposes(ts)
    dir = np.einsum("hij,hwj->hwi", P[:, :3, :3], RAYS)
    iv = 1 / np.maximum(rr, 1)
    span = cv.dilate(iv, FORM) - cv.erode(iv, FORM)
    valid = (
        (wt > 0.05)
        & (sky < 0.25)
        & (dyn < 0.04)
        & (rr > 2)
        & (rr < 205)
        & (span < 0.018)
    )
    xyz = P[:, :3, 3][:, None] + dir * rr[..., None]
    attrs = np.dstack([sky, dyn, road, wt])
    return xyz, valid, attrs


def surface(V, j):
    P, ok, _ = key(j)
    xyz = (P - V.T[:3, 3]) @ V.T[:3, :3]
    iz = 1 / np.maximum(0.1, xyz[..., 2])
    p = xyz[..., :2] * iz[..., None] * (F / 5) + [L / 2 - 0.5, H / 2 - 0.5]
    v = np.ascontiguousarray(np.dstack([p, iz, ok & (xyz[..., 2] > 0.5)]), "f")
    z = np.full((H, L), -1, "f")
    raster()(v.ctypes.data, SW, SH, L, H, z.ctypes.data)
    return z


def project(V, t, z):
    rows = rowposes(t)
    S = rows[SH // 2]
    dr = V.wr + (V.T[:3, 3] - S[:3, 3]) * z[..., None]
    p = proj("front", dr @ S[:3, :3], SW, SH)
    for _ in range(2):
        y = p[..., 1].clip(0, SH - 1)
        k = y.astype("i")
        a = (y - k)[..., None]
        l = np.minimum(k + 1, SH - 1)

        def r(ix):
            return np.einsum(
                "hwi,hwij->hwj",
                V.wr + (V.T[:3, 3] - rows[ix, :3, 3]) * z[..., None],
                rows[ix, :3, :3],
            )

        v = r(k)
        v += a * (r(l) - v)
        p = proj("front", v, SW, SH)
    rim = np.minimum.reduce(
        [p[..., 0], SW - 1 - p[..., 0], p[..., 1], SH - 1 - p[..., 1]]
    )
    return p, smooth((rim - 1) / 12) * (v[..., 2] > 0.06)


def geometry(V, u, v, base):
    keys, kt, _ = inputs()
    ts = seq("front")[0]
    d0, d1 = surface(V, u), surface(V, v)
    zs = [d0, d1]
    finite = (d0 > 0) | (d1 > 0)
    f = float(np.clip((ts[base] - kt[u]) / max(1e-6, kt[v] - kt[u]), 0, 1))
    sel = np.where(
        d1 > 0, np.where(d0 > 0, (1 - f) * d0 + f * d1, d1), np.maximum(d0, 0)
    )
    road = np.ones_like(sel)
    sky = road.copy()
    dyn = np.zeros_like(sel)
    support = np.ones_like(sel)
    for j, z in zip([u, v], zs):
        _, _, A = key(j)
        p, ok = project(V, kt[j], V.gr)
        a = remap(A, p)
        road = np.minimum(road, a[..., 2] * ok * (a[..., 1] < 0.02))
        p, ok = project(V, kt[j], np.zeros_like(z))
        a = remap(A, p)
        sky = np.minimum(sky, a[..., 0] * ok * (a[..., 1] < 0.02))
        p, ok = project(V, kt[j], sel)
        a = remap(A, p)
        dyn = np.maximum(dyn, a[..., 1])
        support = np.minimum(support, ok * (a[..., 3] > 0.05))
    gr = (V.gr > 0.004) & (road > 0.86) & ~((d0 > V.gr + 0.008) | (d1 > V.gr + 0.008))
    sk = (sky > 0.965) & ~finite & ~gr
    z = np.where(gr, V.gr, sel).astype("f")
    z[sk] = 0
    V0 = seq("front")[2][base]
    baseDist = np.linalg.norm(V.T[:3, 3] - V0[:3, 3])
    err = F * baseDist * abs(d1 - d0)
    gradient = cv.dilate(z, FORM) - cv.erode(
        z, FORM
    )  # disagreement and depth cliff measured in implied target-image displacement
    relief = (
        0.81
        * np.exp(-((err / 4.2) ** 2))
        * np.exp(-((gradient * baseDist * F / 14) ** 2))
        * support
        * (d0 > 0)
        * (d1 > 0)
    )
    relief *= np.where(z > 1 / 25, 0.62, 1.0)
    relief *= np.exp(-0.09 * max(0, baseDist - 5))
    wt = np.where(gr, 0.97, relief)
    wt[~finite & ~gr] = 0
    wt[dyn > 0.035] = 0
    return z, wt.astype("f"), gr, sk


def strength(p):
    # Singular values of native-source texels per output pixel. Reject inverse folds,
    # area expansion, anisotropic streaks and undersampling at grazing incidence.
    a, b = (
        np.gradient(p.astype("f") / 5, axis=1),
        np.gradient(p.astype("f") / 5, axis=0),
    )
    det = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    s = (a * a + b * b).sum(2)
    large = np.sqrt(
        np.maximum(0, (s + np.sqrt(np.maximum(0, s * s - 4 * det * det))) * 0.5)
    )
    small = det / np.maximum(0.001, large)
    return (
        smooth((small - 0.28) / 0.75)
        * smooth((16 - large) / 6)
        * smooth((14 - large / np.maximum(0.04, small)) / 6)
    )


def camera_image(V, i, z):
    ts = seq("front")[0]
    p, rim = project(V, ts[i], z)
    im, m = photo(i)
    uv = ((p + 0.5) * [C.width / SW, C.height / SH] - 0.5).astype("f")
    q = C.mask_at((uv + 0.5).reshape(-1, 2)).reshape(H, L)
    ok = cv.erode((q & (rim > 0) & (remap(m, p) < 0.012)).astype("B"), FORM).astype(
        bool
    )
    return im, uv, rim * ok, ok


def restricted(im, reference, conf):
    """Same Canny/normal-pair/tangent test as morph_ghost, both endpoints inside confidence."""
    ed, nx, ny = edges(im)
    valid = cv.erode(conf.astype("B"), np.ones((5, 5), "B")).astype(bool)
    ey, ex = np.nonzero((ed > 0) & valid)
    take = (ex + 3 * ey) % 4 == 0
    ex = ex[take]
    ey = ey[take]
    rf = refmask(reference) > 0
    bad = np.zeros(len(ex), bool)
    xi = nx[ey, ex]
    yi = ny[ey, ex]
    ma = rf[ey, ex]
    # A single owner flags exactly when one endpoint, but not both, has a photo edge.
    for dist in range(3, 41):
        for sign in [-1, 1]:
            u = (ex + np.rint(sign * dist * xi).astype("i")).clip(0, OW - 1)
            v = (ey + np.rint(sign * dist * yi).astype("i")).clip(0, OH - 1)
            bad |= (
                valid[v, u]
                & (ed[v, u] > 0)
                & (xi * nx[v, u] + yi * ny[v, u] > 0.945)
                & (ma != rf[v, u])
            )
    support = np.zeros(len(ex), bool)
    mark = np.zeros_like(ed)
    mark[ey[bad], ex[bad]] = 1
    mark = cv.dilate(mark, FORM)
    for ds_ in [-6, -4, 4, 6]:
        u = (ex + np.rint(-yi * ds_).astype("i")).clip(0, OW - 1)
        v = (ey + np.rint(xi * ds_).astype("i")).clip(0, OH - 1)
        support |= mark[v, u] > 0
    return [len(ex), int((bad & support).sum())]


def hi(a):
    return resize(a, OW, OH)


def render(e, positive=False):
    t = e["t"]
    T = np.array(e["T_world_cam"], "f")
    V = View(t, T, 50)
    base, neighbor, f, u, v = select(e)
    ts, ns, S, arc = seq("front")
    z, wt, ground, sky = geometry(V, u, v, base)
    im, uv, rim, ok = camera_image(V, base, z)
    im2, uv2, rim2, ok2 = camera_image(V, neighbor, z)
    wt *= strength(uv) * rim
    sky &= ok
    solo = remap(im, hi(uv))
    ta = remap(im, uv)
    tb = remap(im2, uv2)
    g = lambda im: cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(g(ta), g(tb), None, 0.5, 2, 13, 3, 5, 1.1, 0)
    fl = blur(flow, 1.0)
    mag = np.linalg.norm(fl, axis=2)
    align = (~ground) & (wt > 0.1) & ok2 & (mag < 0.15)
    adjust = np.where(align[..., None], fl, 0)
    p2 = remap(uv2, XY + adjust)
    nbr = remap(im2, hi(p2))
    diff = np.max(abs(solo.astype("i2") - nbr.astype("i2")), axis=2).astype("f")
    dis = cv.boxFilter(diff, -1, (9, 9))
    trusted = (hi(ok2.astype("f")) > 0.999) & (hi(wt) > 0.07)
    mix = f * trusted * smooth((8 - dis) / 5) * smooth((10 - diff) / 6)
    out = np.uint8(solo + (nbr.astype("f") - solo) * mix[..., None] + 0.5)
    # Independent neighbouring observation also vetoes uncertain geometry (including
    # photo-only silhouettes crossing measured road). It can never extend the footprint.
    weight = hi(wt) * smooth((30 - dis) / 19) * (hi(ok.astype("f")) > 0.999)
    weight[(hi((wt > 0).astype("f")) < 0.999)] = 0
    # Semantic sky is coarse. Exclude narrow dark wires/poles from negative-opacity supervision.
    skyhi = hi(sky.astype("f")) > 0.999
    luma = g(solo)
    close = cv.morphologyEx(luma, cv.MORPH_CLOSE, np.ones((11, 11), "B"))
    delta = (close.astype("i2") - luma) > 9
    skyhi &= ~cv.dilate(delta.astype("B"), np.ones((5, 5), "B")).astype(bool)
    weight[skyhi] = 0
    w = np.uint8(np.clip(weight, 0, 1) * 255 + 0.5)
    w[w < 4] = 0
    conf = w >= 128
    depth = np.uint16(
        np.where(w > 0, np.minimum(65535, np.rint(100 / np.maximum(0.001, hi(z)))), 0)
    )
    out[(w == 0) & ~skyhi] = 0
    keys = inputs()[0]
    e = dict(
        **e,
        base_camera="front",
        base_frame=int(ns[base]),
        base_t=float(ts[base]),
        base_effective_t=float(C.row_time(ts[base], C.height / 2)),
        base_distance_m=float(np.linalg.norm(T[:3, 3] - S[base, :3, 3])),
        base_horizontal_m=float(np.linalg.norm(T[:2, 3] - S[base, :2, 3])),
        neighbor_frame=int(ns[neighbor]),
        neighbor_t=float(ts[neighbor]),
        neighbor_max_blend=f,
        depth_keys=[int(keys[j]["n"]) for j in [u, v]],
        width=OW,
        height=OH,
        depth_unit_m=0.01,
        confident_fraction=float(conf.mean()),
        weighted_fraction=float((w > 0).mean()),
        sky_fraction=float(skyhi.mean()),
        mean_weight=float(w.mean() / 255),
        aligned_blend_fraction=float(((mix > 0.01) & (w > 0)).mean()),
        ground_confident_fraction=float(
            (conf & (hi(ground.astype("f")) > 0.99)).mean()
        ),
    )
    if e["kind"] != "chase" and conf.mean() < MIN_OFFSET or not w.any():
        return dict(**e, dropped=True)
    for key, a in [
        ("rgb", out),
        ("weight", w),
        ("sky", skyhi.astype("B") * 255),
        ("depth", depth),
    ]:
        path = D / key / (e["id"] + ".png")
        assert cv.imwrite(str(path), a, PARS)
        e[key] = str(path)
    # Audit actual decoded output against independently warped, *unmixed* base.
    dec = cv.imread(e["rgb"])
    e["audit"] = restricted(dec, solo, conf)
    if positive:
        control = solo.copy()
        control[(w == 0) & ~skyhi] = 0
        e["solo_audit"] = restricted(control, solo, conf)
        e["shift9_audit"] = restricted(
            cv.addWeighted(control, 0.5, np.roll(control, 9, axis=1), 0.5, 0),
            solo,
            conf,
        )
    return e


def batch(args):
    es, check = args
    rows = []
    for e in es:
        rows.append(render(e, e["id"] in check))
    return rows


def run(args):
    for d in ["rgb", "weight", "sky", "depth", "review"]:
        (D / d).mkdir(exist_ok=True, parents=True)
    all_ = stations()
    central = [e for e in all_ if e["kind"] == "chase"]
    test = [
        min(central, key=lambda e: abs(e["t"] - t)) for t in np.linspace(START, END, 20)
    ]
    check = {e["id"] for e in test}
    if args.pilot:
        all_ = [
            min(central, key=lambda e: abs(e["t"] - t))
            for t in [85, 93, 105, 110, 112, 117.5, 125, 140]
        ]
    groups = [(all_[i : i + 12], check) for i in range(0, len(all_), 12)]
    rows = []
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn")) as pool:
        for i, g in enumerate(pool.map(batch, groups)):
            rows.extend(g)
            if i % 10 == 0:
                print("views", len(rows), "/", len(all_), flush=True)
    if args.pilot:
        savej(D / "pilot.json", rows)
        print(
            [
                (
                    e["id"],
                    round(e["t"], 2),
                    round(e["confident_fraction"], 3),
                    e.get("audit"),
                )
                for e in rows
            ]
        )
        return
    accepted = [e for e in rows if not e.get("dropped")]
    savej(D / "manifest.json", accepted)
    savej(D / "rejected.json", [e for e in rows if e.get("dropped")])
    savej(D / "review/ids.json", [e["id"] for e in test])
    cols = [
        "id",
        "t",
        "travel_m",
        "kind",
        "base_frame",
        "base_t",
        "base_distance_m",
        "confident_fraction",
        "weighted_fraction",
        "sky_fraction",
        "ground_confident_fraction",
        "aligned_blend_fraction",
        "edges",
        "twins",
        "dropped",
    ]
    with (D / "review/views.csv").open("w") as f:
        w = csv.DictWriter(f, cols, extrasaction="ignore")
        w.writeheader()
        for e in rows:
            a, b = e.get("audit", [0, 0])
            w.writerow(dict(**e, edges=a, twins=b))
    print("manifest", len(accepted), "rejected", len(rows) - len(accepted), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--workers", type=int, default=7)
    run(p.parse_args())
