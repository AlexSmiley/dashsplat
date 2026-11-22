"""Observed sky environment map for one stretch: the six cameras' sky pixels at infinity, world frame.

Equirectangular in the viewer's convention (viewer/src/render/sky.ts): u = 0.5 - atan2(y, x) / 2pi,
v = 0.5 + asin(z) / pi, image row (1 - v) H; east in the centre column, up +z. Rotation only.

Per camera, every `step`-th frame in [ref - window / 2, ref + window / 2]:
  - the frame, photometry-corrected (photo/stitched-ground-v2.npz: exposure and shading per frame);
  - its sky mask (masks/sky, 724x469) eroded by `margin` px, minus movers (masks/dyn, dilated), the
    rig's body mask, and thin dark lines inside the sky (black-hat: wires, mast arms the mask missed);
  - every map pixel looks the frame up through the camera's pose at its own row (rolling shutter).
Per camera: per-pixel median over time; kept where at least `min_n` samples and 60 % of them agree
with it within `tol` levels (a wire or crown that passes through is outvoted); the agreeing samples
are averaged. Then residual per-camera gains from the overlaps (least squares on log ratios); a camera
that overlaps none (the front) is matched row by row to its neighbours across the gaps, the gain
interpolated across its width (edge_ramp); a feathered blend across cameras; and thin dark strokes
left in the result (crown fringes, wire remnants) are taken out of the observed sky.

Writes <out>/observed.png (RGB, black where unseen), observed-mask.png, coverage.png (16-bit
accepted samples per pixel), coverage.jpg, cams/<cam>.jpg, build.json.
"""

import argparse
import json
import os
import time
from pathlib import Path

import cv2 as cv
import numpy as np

from pipeline.common.rig import load_rig, T_world_cam
from pipeline.photo.model import Photometry

RS = Path(os.environ.get("RS", Path.home() / "dev/reality-splat")).resolve()
CAMS = ["front", "back", "left_pillar", "right_pillar", "left_repeater", "right_repeater"]
W, H = 4096, 2048
EL_MIN = -6.0
cv.setNumThreads(2)


def map_dirs(rows):
    """World unit directions of map pixels in the given rows (viewer's equirectangular mapping)."""
    u = (np.arange(W) + 0.5) / W
    v = 1 - (np.asarray(rows) + 0.5) / H
    az = (0.5 - u)[None, :] * 2 * np.pi
    el = (v - 0.5)[:, None] * np.pi
    return np.stack(np.broadcast_arrays(np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)), -1)


def rows_above(el_deg):
    return int(np.ceil((0.5 - np.radians(el_deg) / np.pi) * H))


def frames_of(cam, t0, t1):
    """Frames in [t0, t1] that have a sky mask."""
    tab = json.loads((RS / "data/drive/cams" / f"{cam}.json").read_text())
    return [(r["n"], r["t"]) for r in tab
            if t0 <= r["t"] <= t1 and (RS / "masks/sky" / cam / f"{r['n']:06d}.png").exists()]


def good_mask(C, cam, n, img, margin):
    """Usable sky at mask resolution: eroded sky, no movers, no body, no thin dark lines."""
    sky = cv.imread(str(RS / "masks/sky" / cam / f"{n:06d}.png"), cv.IMREAD_UNCHANGED) != 0
    dyn = cv.imread(str(RS / "masks/dyn" / cam / f"{n:06d}.png"), cv.IMREAD_UNCHANGED)
    mh, mw = sky.shape
    good = cv.erode(sky.astype("B"), np.ones((2 * margin + 1, 2 * margin + 1), "B")) > 0
    if dyn is not None:
        good &= cv.dilate((dyn != 0).astype("B"), np.ones((7, 7), "B")) == 0
    body = cv.resize(C.mask.astype("B"), (mw, mh), interpolation=cv.INTER_AREA) == 1
    good &= cv.erode(body.astype("B"), np.ones((5, 5), "B")) > 0
    # wires and thin arms inside the sky: dark, narrow, against a bright background
    L = cv.cvtColor(img, cv.COLOR_RGB2GRAY).astype("f")
    bh = cv.morphologyEx(L, cv.MORPH_BLACKHAT, cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 11)))
    wire = (bh > 6.0).astype("B")
    wire = cv.dilate(wire, np.ones((5, 5), "B"))
    wire = cv.resize(wire, (mw, mh), interpolation=cv.INTER_AREA) > 0
    return good & ~wire, wire & cv.dilate(sky.astype("B"), np.ones((5, 5), "B")).astype(bool)


def project_rs(C, d, R_top, R_bot):
    """Camera pixel of world directions d (N,3) with the pose interpolated to the row it lands on."""
    R_mid = 0.5 * (R_top + R_bot)
    uv, ok = C.project(d @ R_mid)
    s = np.clip(np.nan_to_num(uv[:, 1]) / C.height, 0, 1)[:, None]
    dc = d @ R_top + s * (d @ (R_bot - R_top))
    return C.project(dc)


def grid_remap(img, x, y, interp):
    """cv.remap for long 1-D lists of sample points (remap needs both sides under 32767)."""
    n = len(x)
    cols = 4096
    rows = (n + cols - 1) // cols
    pad = rows * cols - n
    mx = np.pad(x.astype("f"), (0, pad), constant_values=-10).reshape(rows, cols)
    my = np.pad(y.astype("f"), (0, pad), constant_values=-10).reshape(rows, cols)
    out = cv.remap(img, mx, my, interp, borderMode=cv.BORDER_CONSTANT)
    return out.reshape(rows * cols, -1)[:n]


def camera_layer(cam, C, ph, dirs, frames, margin, min_n, tol):
    """Robust per-pixel sky colour of one camera over the window, on the map pixels it can see."""
    flat = dirs.reshape(-1, 3)
    n_mid, t_mid = frames[len(frames) // 2]
    T = T_world_cam(C, t_mid)
    _, foot = C.project(flat @ T[:3, :3])
    foot = cv.dilate(foot.reshape(dirs.shape[:2]).astype("B"), np.ones((61, 61), "B")).reshape(-1) > 0
    idx = np.flatnonzero(foot)
    d = flat[idx]
    vals = np.zeros((len(frames), len(idx), 3), "B")
    ok = np.zeros((len(frames), len(idx)), bool)
    wires = 0
    for k, (n, t) in enumerate(frames):
        im = cv.imread(str(RS / "data/drive/frames" / cam / f"{n:06d}.jpg"),
                       cv.IMREAD_REDUCED_COLOR_2 if cam == "front" else cv.IMREAD_COLOR)
        rgb = np.ascontiguousarray(ph.apply(cam, n, im[:, :, ::-1].astype("f")).clip(0, 255).astype("B"))
        good, wire = good_mask(C, cam, n, rgb, margin)
        wires += int(wire.sum())
        R_top = T_world_cam(C, t, 0.0)[:3, :3]
        R_bot = T_world_cam(C, t, float(C.height))[:3, :3]
        uv, geom = project_rs(C, d, R_top, R_bot)
        uv = np.nan_to_num(uv, nan=-100.0)
        si = rgb.shape[1] / C.width
        sm = good.shape[1] / C.width
        mi = np.clip((uv[:, 0] * sm).astype(int), 0, good.shape[1] - 1)
        mj = np.clip((uv[:, 1] * sm).astype(int), 0, good.shape[0] - 1)
        inside = (uv[:, 0] >= 3) & (uv[:, 0] <= C.width - 4) & (uv[:, 1] >= 3) & (uv[:, 1] <= C.height - 4)
        ok[k] = geom & good[mj, mi] & inside  # no bilinear taps off the image (a dark rim at the frame edge)
        vals[k] = grid_remap(rgb, uv[:, 0] * si - 0.5, uv[:, 1] * si - 0.5, cv.INTER_LINEAR)
    # robust over time, in chunks
    col = np.zeros((len(idx), 3), "f")
    cnt = np.zeros(len(idx), "i4")
    for a in range(0, len(idx), 200_000):
        v = vals[:, a:a + 200_000].astype("f")
        m = ok[:, a:a + 200_000]
        v[~m] = np.nan
        with np.errstate(all="ignore"):
            med = np.nanmedian(v, 0)
            agree = m & (np.nanmax(np.abs(v - med[None]), -1) < tol)
            na = agree.sum(0)
            nv = m.sum(0)
            mean = np.where(agree[..., None], v, 0).sum(0) / np.maximum(na, 1)[:, None]
        keep = (na >= min_n) & (na >= 0.6 * nv)
        col[a:a + 200_000] = np.where(keep[:, None], mean, 0)
        cnt[a:a + 200_000] = np.where(keep, na, 0)
    layer = np.zeros(dirs.shape[:2] + (3,), "f")
    count = np.zeros(dirs.shape[:2], "i4")
    layer.reshape(-1, 3)[idx] = col
    count.reshape(-1)[idx] = cnt
    return layer, count, wires


def near(has, px):
    """Pixels within px columns of a region (wrapped across 360 deg)."""
    k = np.ones((1, 2 * px + 1), "B")
    wr = np.hstack([has[:, -px:], has, has[:, :px]]).astype("B")
    return cv.dilate(wr, k)[:, px:-px] > 0


def gains(layers, counts, adjacent=80):
    """Per-camera, per-channel log gain (sum zero) from pairwise overlaps, or where two cameras do not
    overlap but nearly touch (the front's narrow view), from row medians on either side of the gap."""
    names = list(layers)
    eqs, rhs, pairs = [], [], {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ha, hb = counts[a] > 0, counts[b] > 0
            both = ha & hb
            row = np.zeros(len(names))
            row[i], row[names.index(b)] = 1, -1
            if both.sum() >= 2000:
                r = np.median(np.log(layers[a][both] + 1) - np.log(layers[b][both] + 1), 0)
                pairs[f"{a}/{b}"] = dict(overlap=int(both.sum()), log_ratio=np.round(r, 4).tolist())
                eqs.append(row)
                rhs.append(r)
                continue
            sa, sb = ha & near(hb, adjacent), hb & near(ha, adjacent)
            rows = [y for y in range(ha.shape[0]) if sa[y].sum() >= 10 and sb[y].sum() >= 10]
            if len(rows) < 20:
                continue
            la, lb = np.log(layers[a] + 1), np.log(layers[b] + 1)
            r = np.median([np.median(la[y][sa[y]], 0) - np.median(lb[y][sb[y]], 0) for y in rows], 0)
            pairs[f"{a}/{b}"] = dict(adjacent_rows=len(rows), log_ratio=np.round(r, 4).tolist())
            eqs.append(row * 0.5)
            rhs.append(r * 0.5)
    eqs.append(np.ones(len(names)))
    rhs.append(np.zeros(3))
    g, *_ = np.linalg.lstsq(np.array(eqs), np.array(rhs), rcond=None)
    return {n: g[i] for i, n in enumerate(names)}, pairs


def edge_ramp(L, has, other, other_has, px=80, edge=40, sig=24.0):
    """A per-row log gain at a camera's left and right edges against its neighbours just across the gaps,
    smoothed over rows and interpolated across the camera's width. Returns the layer and the gains' range."""
    n = has.shape[0]
    gl, gr = np.full((n, 3), np.nan), np.full((n, 3), np.nan)
    xl, xr = np.full(n, -1), np.full(n, -1)
    lg = lambda a: np.log(a + 1.0)
    for y in range(n):
        xs = np.flatnonzero(has[y])
        if len(xs) < 2 * edge:
            continue
        a, b = xs.min(), xs.max()
        xl[y], xr[y] = a, b
        for side, (o0, o1, s0, s1) in ((gl, (max(0, a - px), a, a, a + edge)), (gr, (b + 1, b + 1 + px, b + 1 - edge, b + 1))):
            om, sm = other_has[y, o0:o1], has[y, s0:s1]
            if om.sum() >= 10 and sm.sum() >= 10:
                side[y] = np.median(lg(other[y, o0:o1][om]), 0) - np.median(lg(L[y, s0:s1][sm]), 0)
    rows = np.arange(n)
    for gg in (gl, gr):
        ok = ~np.isnan(gg[:, 0])
        if not ok.any():
            gg[:] = 0
            continue
        for c in range(3):
            gg[:, c] = np.interp(rows, rows[ok], gg[ok, c])
        gg[:] = cv.GaussianBlur(gg.astype("f")[:, None, :], (0, 0), sigmaX=0.1, sigmaY=sig)[:, 0, :]
    x = np.arange(L.shape[1])[None, :]
    t = np.clip((x - xl[:, None]) / np.maximum(xr - xl, 1)[:, None], 0, 1)[..., None]
    gain = gl[:, None, :] + t * (gr - gl)[:, None, :]
    out = np.where(has[..., None], L * np.exp(gain), L)
    used = (xl >= 0)
    return out, dict(left=np.round(gl[used].mean(0), 4).tolist(), right=np.round(gr[used].mean(0), 4).tolist())


def build(t0, t1, ref, window, step, margin, min_n, tol, out):
    out = Path(out)
    (out / "cams").mkdir(parents=True, exist_ok=True)
    rig = load_rig(RS / "calib/rig.json")
    ph = Photometry("stitched-ground-v2", RS)
    nrows = rows_above(EL_MIN)
    dirs = map_dirs(np.arange(nrows)).astype("f8")
    w0, w1 = ref - window / 2, ref + window / 2
    layers, counts, info = {}, {}, {}
    tic = time.monotonic()
    for cam in CAMS:
        fr = frames_of(cam, w0, w1)[::step]
        L, N, wires = camera_layer(cam, rig[cam], ph, dirs, fr, margin, min_n, tol)
        layers[cam], counts[cam] = L, N
        info[cam] = dict(frames=len(fr), first=fr[0][0], last=fr[-1][0], pixels=int((N > 0).sum()),
                         wire_pixels_masked=wires)
        print(cam, info[cam], round(time.monotonic() - tic, 1), "s", flush=True)
        small = cv.resize(np.clip(L, 0, 255).astype("B"), (1024, nrows // 4), interpolation=cv.INTER_AREA)
        cv.imwrite(str(out / "cams" / f"{cam}.jpg"), small[:, :, ::-1], [cv.IMWRITE_JPEG_QUALITY, 88])
    g, pairs = gains(layers, counts)
    wgt = {}
    for cam in CAMS:
        layers[cam] = layers[cam] * np.exp(-g[cam])[None, None]
        has = (counts[cam] > 0).astype("B")
        # feather: distance into the camera's accepted region (map pixels), wrapped across 360 deg
        wrapped = np.hstack([has[:, -64:], has, has[:, :64]])
        dist = cv.distanceTransform(wrapped, cv.DIST_L2, 5)[:, 64:-64]
        wgt[cam] = np.clip(dist / 48.0, 0, 1) ** 2 * has
    blend = lambda cams: (sum(wgt[c][..., None] * layers[c] for c in cams), sum(wgt[c] for c in cams))
    ramps = {}
    for cam in CAMS:
        if any(cam in k and "overlap" in v for k, v in pairs.items()):
            continue
        # no overlap with any camera (the front): match it to its neighbours across the gaps, row by row
        a, w = blend([c for c in CAMS if c != cam])
        other = a / np.maximum(w, 1e-4)[..., None]
        layers[cam], ramps[cam] = edge_ramp(layers[cam], counts[cam] > 0, other, w > 1e-4)
    acc, wsum = blend(CAMS)
    total = sum(counts[c] for c in CAMS)
    seen = wsum > 1e-4
    obs = np.zeros((H, W, 3), "B")
    obs[:nrows][seen] = np.clip(acc[seen] / wsum[seen][:, None] + 0.5, 0, 255).astype("B")
    # thin dark strokes left in the sky (crown fringes, wire remnants, specks): not observed sky, filled later
    L = cv.cvtColor(obs[:nrows], cv.COLOR_RGB2GRAY).astype("f")
    # (16 levels: crown fringes and wires are far darker than the sky; the clouds' own dark filaments are not)
    bh = cv.morphologyEx(L, cv.MORPH_BLACKHAT, cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 11))) > 16
    strokes = cv.dilate((bh & seen).astype("B"), np.ones((5, 5), "B")) > 0
    seen &= ~strokes
    obs[:nrows][~seen] = 0
    mask = np.zeros((H, W), "B")
    mask[:nrows][seen] = 255
    cov = np.zeros((H, W), "u2")
    cov[:nrows] = np.minimum(total, 65535)
    cv.imwrite(str(out / "observed.png"), obs[:, :, ::-1])
    cv.imwrite(str(out / "observed-mask.png"), mask)
    cv.imwrite(str(out / "coverage.png"), cov)
    vis = cv.applyColorMap(np.clip(cov[:nrows] / max(1, np.percentile(cov[:nrows][seen], 99)) * 255, 0, 255)
                           .astype("B"), cv.COLORMAP_VIRIDIS)
    vis[~seen] = 0
    cv.imwrite(str(out / "coverage.jpg"), cv.resize(vis, (1024, nrows // 4), interpolation=cv.INTER_AREA))
    # shares by elevation band (solid-angle weighted)
    el = np.degrees((0.5 - (np.arange(nrows) + 0.5) / H) * np.pi)
    sa = np.cos(np.radians(el))[:, None] * np.ones((1, W))
    bands = {}
    for lo, hi in ((0, 10), (10, 30), (30, 60), (60, 90), (0, 90)):
        r = (el >= lo) & (el < hi)
        bands[f"{lo}-{hi}"] = round(float((sa[r] * seen[r]).sum() / sa[r].sum()), 4)
    doc = dict(t0=t0, t1=t1, ref=ref, window=[w0, w1], step=step, margin_px_mask=margin, min_n=min_n, tol=tol,
               size=[W, H], el_min=EL_MIN, photometry="stitched-ground-v2", cams=info,
               gains={k: np.round(v, 4).tolist() for k, v in g.items()}, overlaps=pairs, edge_ramps=ramps,
               strokes_removed_px=int(strokes.sum()),
               observed_share_upper_hemisphere=bands, seconds=round(time.monotonic() - tic, 1))
    (out / "build.json").write_text(json.dumps(doc, indent=1))
    print(json.dumps(dict(observed=bands, gains=doc["gains"], seconds=doc["seconds"])), flush=True)


def compare(a, b):
    """How much the observed sky changes between two builds: mean |difference| where both saw sky, and
    the cloud shift between them (phase correlation over the band 4-22 deg up, in degrees)."""
    A, B = (cv.imread(str(RS / p / "observed.png"))[:, :, ::-1].astype("f") for p in (a, b))
    mA, mB = (cv.imread(str(RS / p / "observed-mask.png"), 0) > 0 for p in (a, b))
    both = mA & mB
    r0, r1 = rows_above(22.0), rows_above(4.0)
    ga, gb = (cv.cvtColor(X, cv.COLOR_RGB2GRAY)[r0:r1] for X in (A, B))
    # cloud texture only (high-pass), under a soft window inside the common area, so the masks' own
    # edges do not pull the peak to zero
    m = cv.GaussianBlur(cv.erode(both[r0:r1].astype("f"), np.ones((31, 31), "B")), (0, 0), 8)
    hp = lambda g: (g - cv.GaussianBlur(g, (0, 0), 24)) * m
    ga, gb = hp(ga).astype("f"), hp(gb).astype("f")
    (dx, dy), resp = cv.phaseCorrelate(ga, gb)
    res = dict(common_pixels=int(both.sum()), mean_abs=round(float(np.abs(A - B)[both].mean()), 2),
               p90_abs=round(float(np.percentile(np.abs(A - B).max(-1)[both], 90)), 2),
               shift_deg=[round(dx * 360 / W, 2), round(dy * 180 / H, 2)], response=round(float(resp), 3))
    print(a, b, json.dumps(res), flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--t0", type=float, default=300.0)
    ap.add_argument("--t1", type=float, default=316.0)
    ap.add_argument("--ref", type=float, default=308.0)
    ap.add_argument("--window", type=float, default=3.0)
    ap.add_argument("--step", type=int, default=3)
    ap.add_argument("--margin", type=int, default=3)
    ap.add_argument("--min-n", type=int, default=4)
    ap.add_argument("--tol", type=float, default=10.0)
    ap.add_argument("--out", default="work/envmap/300-316")
    ap.add_argument("--compare", nargs=2, default=None, help="two build folders")
    a = ap.parse_args()
    if a.compare:
        compare(*a.compare)
        raise SystemExit
    build(a.t0, a.t1, a.ref, a.window, a.step, a.margin, a.min_n, a.tol, RS / a.out)
