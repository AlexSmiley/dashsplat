"""Honest checks for the generative fixer.

real   PSNR / SSIM / LPIPS of render and fixed views against the held-out photo, on static
       non-sky pixels, split into rendered surface (alpha >= 0.5) and holes. Paint (lane
       markings, road text) agreement with the photo.
chase  no ground truth: multi-view consistency (warp each fixed view into its 2 m neighbour and
       its 30-degree orbit neighbour through the rendered depth; the render's own warp error is the
       baseline), paint kept / removed / invented relative to the render, change statistics.
sheet  labelled before/after images for the report (each under 500 KB).
"""

import argparse
import json
import os
from pathlib import Path

import cv2 as cv
import numpy as np

from pipeline.common.rig import Trajectory

RS = Path(os.environ.get("RS", Path.home() / "dev/reality-splat")).resolve()
traj = Trajectory(RS / "poses/ego_causal.npz")  # live-v4 follows the causal poses


def J(p):
    return json.loads(Path(p).read_text())


def rgb(p):
    im = cv.imread(str(p), cv.IMREAD_COLOR)
    assert im is not None, p
    return np.ascontiguousarray(im[:, :, ::-1])


def render_aux(set_dir, vid):
    z = np.load(set_dir / "render" / f"{vid}.npz")
    return z["alpha"].astype("f"), z["z"].astype("f")


def psnr(a, b, m):
    if m.sum() < 100:
        return None
    d = (a.astype("f") - b.astype("f"))[m]
    return float(-10 * np.log10(max((d * d).mean() / 255**2, 1e-10)))


def ssim_map(a, b):
    a, b = a.astype("f8") / 255, b.astype("f8") / 255
    g = lambda x: cv.GaussianBlur(x, (11, 11), 1.5)
    ma, mb = g(a), g(b)
    va, vb, cab = g(a * a) - ma * ma, g(b * b) - mb * mb, g(a * b) - ma * mb
    c1, c2 = 0.01**2, 0.03**2
    s = ((2 * ma * mb + c1) * (2 * cab + c2)) / ((ma * ma + mb * mb + c1) * (va + vb + c2))
    return s.mean(-1)


_lp = None


def lpips_map(a, b):
    global _lp
    import torch
    import lpips

    if _lp is None:
        _lp = lpips.LPIPS(net="alex", spatial=True, verbose=False).eval()
    t = lambda x: torch.from_numpy(x.astype("f") / 127.5 - 1).permute(2, 0, 1)[None]
    with torch.no_grad():
        return _lp(t(a), t(b))[0, 0].numpy()


def masked(x, m):
    return float(x[m].mean()) if m.sum() >= 100 else None


def world_points(T, K, z):
    h, w = z.shape
    y, x = np.mgrid[:h, :w]
    ray = np.stack([(x + 0.5 - K[0, 2]) / K[0, 0], (y + 0.5 - K[1, 2]) / K[1, 1], np.ones((h, w))], -1)
    return (ray * z[..., None]) @ T[:3, :3].T + T[:3, 3]


def ground_mask(v, z, max_range=45.0):
    """Rendered pixels on the road surface: within 0.25 m of the car's ground height, and
    within max_range metres of the car."""
    T, K = np.array(v["T_world_cam"]), np.array(v["K"])
    X = world_points(T, K, z)
    E = traj.T_world_ego(v["t"])
    near = np.linalg.norm(X[..., :2] - E[:2, 3], axis=-1) < max_range
    return (z > 0.3) & near & (np.abs(X[..., 2] - E[2, 3]) < 0.25)


def paint(im, ground):
    """Bright thin road paint: white top-hat of luminance on the ground."""
    L = cv.cvtColor(im, cv.COLOR_RGB2LAB)[:, :, 0].astype("f")
    top = L - cv.morphologyEx(L, cv.MORPH_OPEN, np.ones((15, 15), "B"))
    return (top > 22) & ground


def paint_agree(ref, test, ground, tol=3):
    """recall: reference paint present in test within tol px; precision: test paint present in ref."""
    pr, pt = paint(ref, ground), paint(test, ground)
    k = np.ones((2 * tol + 1, 2 * tol + 1), "B")
    dr = cv.dilate(pr.astype("B"), k) > 0
    dt = cv.dilate(pt.astype("B"), k) > 0
    rec = float((pr & dt).sum() / max(pr.sum(), 1))
    prec = float((pt & dr).sum() / max(pt.sum(), 1))
    return dict(ref_px=int(pr.sum()), test_px=int(pt.sum()), recall=rec, precision=prec)


def real(set_dir, fixes, out):
    set_dir = Path(set_dir)
    views = J(set_dir / "views.json")["views"]
    rows = []
    for v in views:
        vid = v["id"]
        gt = rgb(set_dir / v["gt"])
        mk = cv.imread(str(set_dir / "gt" / f"{vid}-mask.png"), cv.IMREAD_UNCHANGED)
        alpha, z = render_aux(set_dir, vid)
        static = mk == 1
        cov, hole, sky = static & (alpha >= 0.5), static & (alpha < 0.5), mk == 2
        ground = ground_mask(v, z) & static
        row = dict(id=vid, c=v["c"], t=v["t"], coverage=float(cov.sum() / max(static.sum(), 1)),
                   ref_score=v["refs"][0]["score"] if v["refs"] else None)
        imgs = {"render": rgb(set_dir / "render" / f"{vid}.png")}
        for name, d in fixes.items():
            p = Path(d) / "fixed" / f"{vid}.png"
            if p.exists():
                imgs[name] = rgb(p)
        for name, im in imgs.items():
            s = ssim_map(im, gt)
            lp = lpips_map(im, gt)
            row[name] = dict(
                psnr=psnr(im, gt, static), psnr_cov=psnr(im, gt, cov), psnr_hole=psnr(im, gt, hole),
                psnr_sky=psnr(im, gt, sky), ssim=masked(s, static), ssim_cov=masked(s, cov),
                lpips=masked(lp, static), lpips_cov=masked(lp, cov), lpips_hole=masked(lp, hole),
                paint=paint_agree(gt, im, ground),
            )
        rows.append(row)
        print(vid, {k: (round(row[k]["psnr_cov"] or 0, 2), round(row[k]["lpips_cov"] or 0, 3)) for k in imgs}, flush=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(rows, indent=1))
    summarize_real(rows)


def summarize_real(rows):
    names = [k for k in rows[0] if isinstance(rows[0][k], dict)]
    keys = ["psnr", "psnr_cov", "psnr_hole", "ssim", "ssim_cov", "lpips", "lpips_cov", "lpips_hole", "psnr_sky"]
    for n in names:
        vals = {k: np.nanmean([r[n][k] if r[n][k] is not None else np.nan for r in rows if n in r]) for k in keys}
        pr = [r[n]["paint"] for r in rows if n in r and r[n]["paint"]["ref_px"] > 200]
        vals["paint_recall"] = np.mean([p["recall"] for p in pr])
        vals["paint_precision"] = np.mean([p["precision"] for p in pr])
        print(n, {k: round(float(x), 4) for k, x in vals.items()})


def warp(src, src_v, src_z, dst_v, dst_z):
    """Sample src (image) at the pixels of dst through dst's rendered depth; occlusion-tested."""
    Ts, Ks = np.array(src_v["T_world_cam"]), np.array(src_v["K"])
    X = world_points(np.array(dst_v["T_world_cam"]), np.array(dst_v["K"]), dst_z)
    v = (X - Ts[:3, 3]) @ Ts[:3, :3]
    zz = v[..., 2]
    u = (Ks[0, 0] * v[..., 0] / np.maximum(zz, 1e-3) + Ks[0, 2] - 0.5).astype("f")
    q = (Ks[1, 1] * v[..., 1] / np.maximum(zz, 1e-3) + Ks[1, 2] - 0.5).astype("f")
    h, w = src_z.shape
    inside = (dst_z > 0.3) & (zz > 0.3) & (u >= 0) & (u <= w - 1) & (q >= 0) & (q <= h - 1)
    zs = cv.remap(src_z, u, q, cv.INTER_NEAREST)
    vis = inside & (np.abs(zs - zz) < 0.05 * zz + 0.1)
    out = cv.remap(src, u, q, cv.INTER_LINEAR)
    return out, vis


def chase(set_dir, fixes, out, max_pairs=10**9):
    set_dir = Path(set_dir)
    views = {v["id"]: v for v in J(set_dir / "views.json")["views"]}
    pairs = []
    for vid, v in views.items():
        k = v["station"]
        nb = f"chase-{k + 1:04d}-{v['chase']}"
        if nb in views:
            pairs.append((vid, nb, "travel 2 m"))
        if v["chase"] == "mid":
            for side in ("orb-30", "orb+30"):
                nb = f"chase-{k:04d}-{side}"
                if nb in views:
                    pairs.append((vid, nb, "orbit 30 deg"))
    pairs = pairs[:max_pairs]
    cache = {}

    def load(name, vid):
        key = (name, vid)
        if key not in cache:
            p = set_dir / "render" / f"{vid}.png" if name == "render" else Path(fixes[name]) / "fixed" / f"{vid}.png"
            cache[key] = rgb(p) if p.exists() else None
        return cache[key]

    names = ["render"] + list(fixes)
    rows = []
    for a, b, kind in pairs:
        va, vb = views[a], views[b]
        _, za = render_aux(set_dir, a)
        ab, zb = render_aux(set_dir, b)
        row = dict(a=a, b=b, kind=kind, chase=va["chase"])
        for n in names:
            ia, ib = load(n, a), load(n, b)
            if ia is None or ib is None:
                continue
            wa, vis = warp(ia, va, za, vb, zb)
            vis &= ab >= 0.5
            d = np.abs(wa.astype("f") - ib.astype("f")).mean(-1)
            row[n] = dict(mae=masked(d, vis), psnr=psnr(wa, ib, vis), px=int(vis.sum()))
        rows.append(row)
    # paint and change statistics per view, relative to the render
    stats = []
    for vid, v in views.items():
        r = load("render", vid)
        alpha, z = render_aux(set_dir, vid)
        g = ground_mask(v, z)
        s = dict(id=vid, chase=v["chase"], coverage=v["coverage"],
                 ref_score=v["refs"][0]["score"] if v["refs"] else None)
        for n in fixes:
            f = load(n, vid)
            if f is None:
                continue
            ch = cv.imread(str(Path(fixes[n]) / "change" / f"{vid}.png"), cv.IMREAD_UNCHANGED).astype("f") / 255
            s[n] = dict(paint=paint_agree(r, f, g), change_cov=masked(ch, alpha >= 0.5))
        stats.append(s)
        cache.clear()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(dict(pairs=rows, views=stats), indent=1))
    summarize_chase(rows, stats, names)


def summarize_chase(rows, stats, names):
    for kind in sorted({r["kind"] for r in rows}):
        for n in names:
            rr = [r[n] for r in rows if r["kind"] == kind and n in r and r[n]["mae"] is not None]
            if rr:
                print(kind, n, "pairs", len(rr), "MAE", round(float(np.mean([x["mae"] for x in rr])), 2),
                      "PSNR", round(float(np.mean([x["psnr"] for x in rr])), 2))
    for n in names[1:]:
        ss = [s[n] for s in stats if n in s]
        pp = [s["paint"] for s in ss if s["paint"]["ref_px"] > 200]
        print(n, "views", len(ss), "paint recall", round(float(np.mean([p["recall"] for p in pp])), 3),
              "precision", round(float(np.mean([p["precision"] for p in pp])), 3),
              "change", round(float(np.mean([s["change_cov"] for s in ss if s["change_cov"] is not None])), 3))


def label(im, s):
    out = cv.copyMakeBorder(im, 26, 0, 0, 0, cv.BORDER_CONSTANT, value=(23, 26, 29))
    cv.putText(out, s, (8, 18), cv.FONT_HERSHEY_SIMPLEX, 0.52, (225, 231, 234), 1, cv.LINE_AA)
    return out


def save_small(p, im):
    p.parent.mkdir(parents=True, exist_ok=True)
    for q in (88, 82, 76, 70, 62, 55):
        ok, buf = cv.imencode(".jpg", im[:, :, ::-1], [cv.IMWRITE_JPEG_QUALITY, q])
        if len(buf) < 490_000:
            p.write_bytes(buf.tobytes())
            return
    raise ValueError(p)


def sheet(set_dir, fix, vid, out, title, width=720):
    set_dir = Path(set_dir)
    v = {x["id"]: x for x in J(set_dir / "views.json")["views"]}[vid]
    scale = width / v["size"][0]
    rs = lambda im: cv.resize(im, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)
    r = rs(rgb(set_dir / "render" / f"{vid}.png"))
    f = rs(rgb(Path(fix) / "fixed" / f"{vid}.png"))
    top = np.hstack([label(r, f"render | {title}"), label(f, "Difix-ref repaired")])
    panels = []
    if v.get("gt"):
        panels.append(label(rs(rgb(set_dir / v["gt"])), "held-out photo (truth)"))
    if v["refs"]:
        rf = v["refs"][0]
        panels.append(label(rs(rgb(set_dir / rf["path"])), f"reference: {rf['c']} {rf['t']:.1f} s"))
    if len(panels) < 2:
        ch = cv.imread(str(Path(fix) / "change" / f"{vid}.png"), cv.IMREAD_UNCHANGED)
        cl = cv.imread(str(Path(fix) / "class" / f"{vid}.png"), cv.IMREAD_UNCHANGED)
        vis = np.zeros(cl.shape + (3,), "B")
        vis[cl == 1] = (230, 170, 60)
        vis[cl == 2] = (220, 60, 70)
        base = rgb(Path(fix) / "fixed" / f"{vid}.png")
        over = (0.55 * base + 0.45 * vis).astype("B")
        over[cl == 0] = (0.5 * base[cl == 0]).astype("B")
        panels.append(label(rs(over), "changed: amber repaired, red fixer-only"))
    im = np.vstack([top, np.hstack(panels[:2])])
    save_small(Path(out), im)


def pairsheet(set_dir, fix, a, b, out, width=720):
    """Two neighbouring repaired views, and how far each disagrees with its neighbour once warped
    through the rendered depth (repaired versus the render's own disagreement)."""
    set_dir = Path(set_dir)
    views = {x["id"]: x for x in J(set_dir / "views.json")["views"]}
    va, vb = views[a], views[b]
    _, za = render_aux(set_dir, a)
    ab, zb = render_aux(set_dir, b)
    scale = width / va["size"][0]
    rs = lambda im: cv.resize(im, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)
    cells, heat = [], []
    for name, d in (("render", set_dir / "render"), ("repaired", Path(fix) / "fixed")):
        ia, ib = rgb(d / f"{a}.png"), rgb(d / f"{b}.png")
        wa, vis = warp(ia, va, za, vb, zb)
        vis &= ab >= 0.5
        e = np.abs(wa.astype("f") - ib.astype("f")).mean(-1)
        hm = cv.applyColorMap(np.clip(e * 4, 0, 255).astype("B"), cv.COLORMAP_INFERNO)[:, :, ::-1]
        hm[~vis] = (40, 40, 40)
        heat.append(label(rs(hm), f"{name}: |warped {a[6:10]} - {b[6:10]}|, MAE {e[vis].mean():.1f} levels"))
        if name == "repaired":
            cells = [label(rs(ia), f"repaired {a}"), label(rs(ib), f"repaired {b}")]
    save_small(Path(out), np.vstack([np.hstack(cells), np.hstack(heat)]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["real", "chase", "sheet", "pairsheet"])
    ap.add_argument("set", type=Path)
    ap.add_argument("--fix", action="append", default=[], help="name=folder (repeatable)")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--id")
    ap.add_argument("--id2")
    ap.add_argument("--title", default="")
    ap.add_argument("--max-pairs", type=int, default=10**9)
    a = ap.parse_args()
    fixes = dict(x.split("=", 1) for x in a.fix)
    fixes = {k: RS / v for k, v in fixes.items()}
    if a.mode == "real":
        real(RS / a.set, fixes, RS / a.out)
    elif a.mode == "chase":
        chase(RS / a.set, fixes, RS / a.out, a.max_pairs)
    elif a.mode == "pairsheet":
        pairsheet(RS / a.set, next(iter(fixes.values())), a.id, a.id2, a.out)
    else:
        sheet(RS / a.set, next(iter(fixes.values())), a.id, a.out, a.title)
