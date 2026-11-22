"""Decode the viewer export, sample the actual height triangles; independent camera and splat comparisons."""

import argparse, ctypes as ct
from scipy.spatial.transform import Rotation
from .atlas import *
from pipeline.road.render import Scene as DiscScene
from pipeline.road.geometry import pixels_to_ground

RPT = Path(os.getenv("GT_REPORT", "pipeline/groundtex/report"))
RPT.mkdir(parents=True, exist_ok=True)
SPL = Path(os.getenv("GT_SPLATS", "scene/stitched-opt-v4"))
bg = np.array([28, 31, 33], "B")


class Atlas:
    def __init__(self):
        self.meta = read(OUT / "ground.json")
        self.idx = {tuple(map(int, t["id"].split("_"))): t for t in self.meta["tiles"]}

    @lru_cache(120)
    def data(self, k, l):
        if l == 0:
            arr = np.asarray(
                Image.open(OUT / self.idx[k]["texture"]).convert("RGBA")
            ).copy()
            arr[:, :, :3] = np.uint8(
                arr[:, :, :3].astype("f") * (arr[:, :, 3:4] / 255) + 0.5
            )
            return arr
        return cv.resize(
            self.data(k, l - 1), None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA
        )

    @staticmethod
    def groups(xy):
        keys = np.floor(xy.reshape(-1, 2) / M).astype("i")
        encoded = (keys[:, 0] + 32768) * 65536 + (keys[:, 1] + 32768)
        ix = np.argsort(encoded, kind="stable")
        b = np.r_[0, 1 + np.flatnonzero(np.diff(encoded[ix])), len(ix)]
        return [(tuple(keys[ix[l]]), ix[l:r]) for l, r in zip(b[:-1], b[1:]) if r > l]

    def z(self, xy, default):
        sh = xy.shape[:-1]
        flat = xy.reshape(-1, 2)
        out = np.full(len(flat), default, "f")
        for k, ix in self.groups(flat):
            if k not in self.idx:
                continue
            z = np.array(self.idx[k]["grid"]["z"]).reshape(9, 9)
            uv = (flat[ix] - np.array(k) * M) / M * 8
            ij = np.floor(uv).astype(int).clip(0, 7)
            x, y = ij.T
            u, v = (uv - ij).T
            z00 = z[y, x]
            z11 = z[y + 1, x + 1]
            out[ix] = np.where(
                u >= v,
                (1 - u) * z00 + (u - v) * z[y, x + 1] + v * z11,
                (1 - v) * z00 + (v - u) * z[y + 1, x] + u * z11,
            )
        return out.reshape(sh)

    def lookup(self, xy, foot=None):
        flat = xy.reshape(-1, 2)
        out = np.zeros((len(flat), 4), "B")
        foot = None if foot is None else foot.ravel()
        for k, ix in self.groups(flat):
            t = self.idx.get(k)
            if not t:
                continue
            uv = (flat[ix] - np.array(k) * M) / t["texelSize"] + t["gutter"]
            levels = (
                np.zeros(len(ix), "i")
                if foot is None
                else np.clip(
                    np.rint(np.log2(np.maximum(foot[ix] / t["texelSize"], 1))), 0, 4
                ).astype("i")
            )
            for l in np.unique(levels):
                pos = levels == l
                j = ix[pos]
                st = (uv[pos] / (2**l) - 0.5).astype("f")
                tex = self.data(k, l)
                for start in range(0, len(j), 30000):
                    sl = slice(start, start + 30000)
                    out[j[sl]] = cv.remap(tex, st[sl, None, :], None, cv.INTER_LINEAR)[
                        :, 0
                    ]
        return out.reshape(xy.shape[:-1] + (4,))

    def rgba(self, xy, foot=None):
        rgba = self.lookup(xy, foot)
        return rgba[:, :, :3].astype("f"), rgba[:, :, 3:4].astype("f") / 255


class Splats(DiscScene):
    def __init__(self):
        super().__init__("stitched", 2)
        self.root = SPL
        self.index = read(SPL / "map.json")
        self.index["tiles"] = [e for e in self.index["tiles"] if e["layer"] == "ground"]
        self.drive = read(SPL / "drive.json")
        self.rast = ct.CDLL("work/stitch/report/v2/wide.dylib").raster
        P = ct.c_void_p
        I = ct.c_int
        self.rast.argtypes = [P, I, ct.c_float, I, P, P, I, I, I, P]

    def chase(self, t, T, K, wh):
        a, _ = self.local(t, T[:3, 3], 150)
        q = np.ascontiguousarray(np.column_stack((a, np.zeros(len(a), "f"))), "f")
        T = np.repeat(T[None], 33, 0).astype("d")
        k = np.r_[K, np.zeros(4)].astype("d")
        w, h = wh
        out = np.zeros((h, w, 6), "f")
        out[:, :, :3] = bg
        self.rast(
            q.ctypes.data,
            len(q),
            t,
            2,
            T.ctypes.data,
            k.ctypes.data,
            0,
            w,
            h,
            out.ctypes.data,
        )
        return out[:, :, :3].clip(0, 255).astype("B")


def label(rgb, s):
    im = cv.copyMakeBorder(
        np.ascontiguousarray(rgb), 28, 0, 0, 0, cv.BORDER_CONSTANT, value=(26, 26, 26)
    )
    cv.putText(
        im, s, (7, 19), cv.FONT_HERSHEY_SIMPLEX, 0.49, (228, 228, 228), 1, cv.LINE_AA
    )
    return im


def save(name, rgb, quality=93):
    cv.imwrite(
        str(RPT / name),
        np.ascontiguousarray(rgb[:, :, ::-1]),
        [cv.IMWRITE_JPEG_QUALITY, quality],
    )


def chase(at, sp, t):
    d = read(Path("pipeline/groundtex/report") / f"pose-{t}.json")
    p = d["frames"][0]
    pos = np.array(p["pos"])
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(p["quat"]).as_matrix() @ np.diag([1, -1, -1])
    T[:3, 3] = pos
    w, h = d["width"], d["height"]
    f = h / 2 / np.tan(np.deg2rad(d["vfov"] / 2))
    K = [f, f, w / 2, h / 2]
    xx, yy = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    ray = (
        np.stack([(xx - w / 2) / f, (yy - h / 2) / f, np.ones_like(xx)], -1)
        @ T[:3, :3].T
    )
    z = float(at.z(pos[:2][None], pos[2] - 5.5)[0])
    safe = np.minimum(-0.02, ray[:, :, 2])
    dep = np.clip((z - pos[2]) / safe, 0, 160)
    xy = pos[:2] + ray[:, :, :2] * dep[:, :, None]
    for _ in range(3):
        Z = at.z(xy, z)
        dep = np.clip((Z - pos[2]) / safe, 0, 160)
        xy = pos[:2] + ray[:, :, :2] * dep[:, :, None]
    dx = np.gradient(xy, axis=1)
    dy = np.gradient(xy, axis=0)
    foot = np.maximum(np.linalg.norm(dx, axis=2), np.linalg.norm(dy, axis=2) / 4)
    old = sp.chase(t, T, K, (w, h))
    new = old.astype("f")
    rgb = np.zeros_like(new)
    alpha = np.zeros((h, w, 1), "f")
    for k in [-0.375, -0.125, 0.125, 0.375]:
        r, a = at.rgba(xy + dy * k, foot)
        rgb += r * 0.25
        alpha += a * 0.25
    ok = (ray[:, :, 2] < -0.02) & (dep < 150)
    new[ok] = (rgb + (1 - alpha) * old)[ok]
    new = new.clip(0, 255).astype("B")
    save(
        f"chase-{t}.jpg",
        np.hstack(
            [
                label(old, f"{SPL.name} ground splats | " + str(t) + " s"),
                label(new, f"Atlas over {SPL.name} fallback | same pose"),
            ]
        ),
    )
    return float(alpha[ok].mean())


def pose(t):
    T = inputs()[0].T_world_ego(t)
    fw = T[:2, 0].copy()
    fw /= np.linalg.norm(fw)
    right = np.array([fw[1], -fw[0]])
    return T[:2, 3], fw, right


def top_xy(t, along, rightward, w, h, cell):
    p, f, r = pose(t)
    centre = p + f * along + r * rightward
    X, Y = np.meshgrid(
        (np.arange(w) + 0.5 - w / 2) * cell, (np.arange(h) + 0.5 - h / 2) * cell
    )
    return (
        centre + X[:, :, None] * r - Y[:, :, None] * f,
        centre,
        np.arctan2(f[1], f[0]),
    )


def groundphoto(r, xy):
    # Interpolate source UV on the 4-texel lattice, keeping the proof's rotated sampling grid.
    cam = inputs()[1][r["cam"]]
    tr = inputs()[0]
    step = 4
    h, w = xy.shape[:2]
    xx, yy = np.meshgrid(np.arange(0, w + step, step), np.arange(0, h + step, step))
    co = (
        xy[0, 0]
        + xx[:, :, None] * (xy[0, 1] - xy[0, 0])
        + yy[:, :, None] * (xy[1, 0] - xy[0, 0])
    )
    co = corrected_xy(r, co)
    uv, _, _ = ground_to_pixels(cam, tr, r["t"], co, 30)
    xx, yy = np.meshgrid(np.arange(w, dtype="f") / step, np.arange(h, dtype="f") / step)
    uv = cv.remap(uv.astype("f"), xx, yy, cv.INTER_LINEAR)
    im = photo(r, uv, False)[:, :, ::-1]
    valid = cam.mask_at(uv)
    return np.where(valid[:, :, None], im, bg), valid


def best(obs, xy):
    p = xy[::20, ::20]
    centre = p.mean((0, 1))
    bestq = -1
    row = None
    for r in obs:
        if np.linalg.norm(centre - pose(r["t"])[0]) > (
            37 if r["cam"] == "front" else 23
        ):
            continue
        cam = inputs()[1][r["cam"]]
        uv, mask, _ = ground_to_pixels(cam, inputs()[0], r["t"], corrected_xy(r, p), 37)
        frac = mask.mean()
        dx = np.gradient(uv, axis=1) / 0.20
        dy = np.gradient(uv, axis=0) / 0.20
        a = (dx * dx).sum(2)
        b = (dy * dy).sum(2)
        c = (dx * dy).sum(2)
        smin = np.sqrt(np.maximum(0, (a + b - np.hypot(a - b, 2 * c)) * 0.5))
        score = frac**9 * (np.median(smin[mask]) if mask.any() else 0)
        if score > bestq:
            bestq = score
            row = r
    return row


def plate(at, sp, obs, t, along, side, w, h, tag):
    xy, cen, yaw = top_xy(t, along, side, w, h, 0.01)
    old = sp.draw(
        t, "top", centre=np.r_[cen, 0], span=w * 0.01, size=(w, h), fixed_yaw=yaw
    )[0]
    r, a = at.rgba(xy)
    im = (r + (1 - a) * old).clip(0, 255).astype("B")
    row = best(obs, xy)
    pic, valid = groundphoto(row, xy)
    save(
        f"detail-{tag}.jpg",
        np.hstack(
            [
                label(old, f"{SPL.name} splats"),
                label(im, "Atlas | 1 cm samples"),
                label(pic, f"One photo: {row['cam']} {row['n']}"),
            ]
        ),
    )
    return dict(
        time=t,
        along_m=along,
        right_m=side,
        source=row["n"],
        cam=row["cam"],
        alpha=float(a.mean()),
        photo_valid=float(valid.mean()),
    )


def discover(at):
    rows = []
    for t in [93, 96, 100, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128]:
        xy, _, _ = top_xy(t, 5, 0, 480, 570, 0.055)
        r, a = at.rgba(xy, 0.055 * np.ones(xy.shape[:2]))
        b = (r + (1 - a) * bg).astype("B")
        cv.circle(b, (240, 376), 4, (255, 100, 20), -1)
        rows.append(label(b, str(t) + " s / 5.5 cm per px"))
    for i in range(0, len(rows), 4):
        save(f"overview-{i // 4}.jpg", np.hstack(rows[i : i + 4]))


def eval_hold(at, obs, start=85.0, end=140.0):
    # Every 29th native front frame is unavailable to all atlas passes. Adjacent held
    # exposure is the physical comparison floor, no test-time image registration.
    tr, rig, _ = inputs()
    cam = rig["front"]
    tbl = table("front")
    src = {r["n"] for r in obs if r["cam"] == "front"}
    by_n = {v["n"]: i for i, v in enumerate(tbl)}
    held = [e for e in tbl if e["n"] % 29 == 7 and start + 0.8 <= e["t"] <= end - 0.8]
    scores = []
    for e in held[::2]:
        n = e["n"]
        candidates = [
            v
            for v in tbl[max(0, by_n[n] - 5) : by_n[n] + 6]
            if v["n"] != n and v["n"] not in src
        ]
        if not candidates:
            continue
        alt = min(candidates, key=lambda v: abs(v["t"] - e["t"]))
        if abs(alt["t"] - e["t"]) > 0.15:
            continue
        row = lambda v: dict(cam="front", n=v["n"], t=v["t"])
        x, y = np.meshgrid(
            np.arange(8, cam.width, 12) + 0.5,
            np.arange(cam.height // 2, cam.height - 5, 8) + 0.5,
        )
        uv = np.stack([x, y], 2).astype("f")
        zw, ok = pixels_to_ground(cam, tr, e["t"], uv, 18)
        p, f, r = pose(e["t"])
        along = (zw[:, :, :2] - p) @ f
        ok &= (along >= 5) & (along <= 15)
        xy = np.nan_to_num(zw[:, :, :2])
        out, a = at.rgba(xy)
        ref = photo(row(e), uv, False)[:, :, ::-1]
        v2, ok2, _ = ground_to_pixels(cam, tr, alt["t"], xy, 20)
        check = photo(row(alt), v2.astype("f"), False)[:, :, ::-1]
        v = ok & ok2 & (a[:, :, 0] > 0.98)
        mask = Path(f"masks/road/front/{n:06d}.png")
        if mask.exists():
            v &= sam(cv.imread(str(mask), 0), uv, cam, True) > 0
        for folder in ["masks/dyn", "maps/movers-v1/remove"]:
            m = Path(folder) / "front" / f"{n:06d}.png"
            if m.exists():
                v &= sam(np.uint8(cv.imread(str(m), -1) > 0), uv, cam, True) == 0
        if v.sum() < 300:
            continue
        v &= np.max(abs(check.astype("f") - ref), axis=2) < 55
        scores.append(
            dict(
                n=n,
                t=e["t"],
                floor_n=alt["n"],
                samples=int(v.sum()),
                mae=float(abs(out[v] - ref[v]).mean()),
                floor=float(abs(check[v].astype("f") - ref[v]).mean()),
            )
        )
    return scores


def measurements(at):
    rows = []
    tr = inputs()[0]
    dist = np.r_[0, np.cumsum(np.linalg.norm(np.diff(tr.p[:, :2], axis=0), axis=1))]
    windows = at.meta["intervals"].copy()
    if any(lo <= 85 and hi >= 330 for lo, hi in windows):
        windows += [[85, 140], [190, 215], [286, 330], [0, 85], [330, 361.42]]
    x, y = np.meshgrid(np.arange(-2, 10.01, 0.2), np.arange(-1.6, 1.601, 0.2))
    for lo, hi in windows:
        start, end = np.interp([lo, hi], tr.t, dist)
        rat = []
        tt = np.arange(lo, hi + 0.001, 0.25)
        stations = np.interp(np.arange(start, end, 1), dist, tr.t)
        for t in np.r_[tt, stations]:
            pos, f, r = pose(t)
            xy = pos + x[:, :, None] * f + y[:, :, None] * r
            rat.append(float((at.lookup(xy)[:, :, 3] >= 250).mean()))
        rows.append(
            dict(
                interval=[lo, hi],
                metres=round(end - start, 2),
                opaque_core=float(np.mean(rat[: len(tt)])),
                opaque_core_distance_weighted=float(np.mean(rat[len(tt) :])),
                arc_stations=len(stations),
            )
        )
    b = sum(t["bytes"] for t in at.idx.values())
    return dict(
        tiles=len(at.idx),
        bytes=b,
        manifest_bytes=(OUT / "ground.json").stat().st_size,
        intervals=rows,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["discovery", "chase", "detail", "metrics", "hold"])
    p.add_argument("--patches", default="pipeline/groundtex/patches.json")
    p.add_argument("--interval", nargs=2, type=float)
    args = p.parse_args()
    at = Atlas()
    obs = read(W / "observations.json")
    if args.mode == "discovery":
        discover(at)
    if args.mode == "chase":
        sp = Splats()
        print({t: chase(at, sp, t) for t in [100, 112, 118, 128]}, flush=True)
    if args.mode == "detail":
        sp = Splats()
        dump(
            RPT / "patches.json", [plate(at, sp, obs, **e) for e in read(args.patches)]
        )
    if args.mode == "metrics":
        dump(RPT / "coverage.json", measurements(at))
    if args.mode == "hold":
        if args.interval:
            lo, hi = args.interval
            dump(RPT / f"heldout-{lo:g}-{hi:g}.json", eval_hold(at, obs, lo, hi))
        else:
            dump(RPT / "heldout.json", eval_hold(at, obs))


if __name__ == "__main__":
    main()
