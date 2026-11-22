"""Projected evidence, held-mask residuals, overhead tracks and exported PLY turntables."""

import numpy as np, cv2, json, pickle
from collections import defaultdict
from pathlib import Path
from .inventory import W
from .model import projection
from .shape import silhouette
from .observations import read
from .publish import NAMES
from .tracking import playback
from pipeline.common.rig import load_rig, Trajectory
from scipy.spatial.transform import Rotation

OUT = W / "proof"
OUT.mkdir(exist_ok=True)


def text(im, v, p=(7, 16), color=(225, 225, 225), scale=0.43):
    cv2.putText(im, v, p, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def save(name, img):
    cv2.imwrite(str(OUT / (name + ".jpg")), img, [cv2.IMWRITE_JPEG_QUALITY, 83])
    assert (OUT / (name + ".jpg")).stat().st_size < 500000


def orb(a, t, size):
    a = a[(a[:, 17] <= t) & (a[:, 18] > t)]
    p = a[:, :3] - [0, 0, size[2] * 0.47]
    n = a[:, 3:6]
    col = np.clip((0.5 + a[:, 6:9] * 0.28209479) * 255, 0, 255)[:, ::-1]
    sig = np.exp(a[:, 10:13])
    R = Rotation.from_quat(a[:, 13:17][:, [1, 2, 3, 0]]).as_matrix()
    cells = []
    for j in range(8):
        ang = j * np.pi / 4
        el = 0.29
        look = np.array(
            [np.cos(ang) * np.cos(el), np.sin(ang) * np.cos(el), np.sin(el)]
        )
        right = np.array([-np.sin(ang), np.cos(ang), 0])
        up = np.cross(look, right)
        axes = np.stack([right, -up])
        scale = 153 / (size[0] + 0.22 * size[1])
        u = p @ axes.T * scale + [88, 55]
        image = np.full((114, 176, 3), [36, 32, 30], np.uint8)
        Z = p @ look
        cov = axes[None] @ (R * sig[:, None, :])
        val, vec = np.linalg.eigh(cov @ cov.transpose(0, 2, 1))
        axis = np.maximum(1, np.sqrt(np.maximum(0, val)) * 1.75 * scale)
        az = np.degrees(np.arctan2(vec[:, 1, 1], vec[:, 0, 1]))
        light = 0.87 + 0.13 * np.maximum(0, n @ look)
        rgb = (col * light[:, None]).astype(int)
        for i in np.argsort(Z):
            if n[i] @ look < -0.05:
                continue
            cv2.ellipse(
                image,
                tuple(np.rint(u[i] * 16).astype(int)),
                tuple(np.maximum(13, np.rint(axis[i, ::-1] * 16)).astype(int)),
                az[i],
                0,
                360,
                tuple(rgb[i].tolist()),
                -1,
                cv2.LINE_AA,
                4,
            )
        text(image, f"{j * 45} deg", (5, 108), scale=0.34)
        cells.append(image)
    return np.hstack(cells)


def plate(s, rows, track, size, mode):
    cs = load_rig()
    tr = Trajectory("poses/ego" + ("_causal" if mode == "live" else "") + ".npz")
    root = W / "extra" if s.get("input") == "extra" else Path(".")
    outline = silhouette(s["class"], size)
    box = np.array(
        [
            [x, y, z]
            for z in [0, size[2]]
            for x, y in [
                (-size[0] / 2, -size[1] / 2),
                (size[0] / 2, -size[1] / 2),
                (size[0] / 2, size[1] / 2),
                (-size[0] / 2, size[1] / 2),
            ]
        ]
    )
    allp = np.vstack([box, outline])
    metrics = defaultdict(list)
    cams = []
    for c in s["fragments"]:
        rr = [
            r
            for r in rows
            if r["c"] == c and r["key"] and track["t"][0] <= r["rt"] <= track["t"][-1]
        ]
        if not rr:
            unseen = [r for r in rows if r["c"] == c and r["key"]]
            if unseen:
                strip = []
                for q in (0.16, 0.49, 0.83):
                    r = unseen[min(len(unseen) - 1, int(q * len(unseen)))]
                    im = read(c, r["n"])
                    b = np.rint(r["bb"]).astype(int)
                    cv2.rectangle(im, tuple(b[:2]), tuple(b[2:]), (242, 125, 215), 2)
                    im = cv2.resize(im, (320, 180))
                    cap = np.full((23, 320, 3), 26, np.uint8)
                    text(cap, f"{c} {r['t']:.2f}s / no released track")
                    strip.append(np.vstack([im, cap]))
                cams.append(np.hstack(strip))
            continue
        show = [rr[min(len(rr) - 1, int(q * len(rr)))] for q in [0.16, 0.49, 0.83]]
        strip = []
        for r in rr:
            uv, ok, _, z = projection(cs[c], tr, r["t"], track, allp, allp * 0)
            bo = uv[:8]
            u = uv[8:]
            b = np.clip(np.r_[u.min(0), u.max(0)], [0, 0, 0, 0], [724, 469, 724, 469])
            m = np.array(r["bb"])
            inter = np.prod(
                np.maximum(0, np.minimum(m[2:], b[2:]) - np.maximum(m[:2], b[:2]))
            )
            iou = inter / max(
                1, np.prod(m[2:] - m[:2]) + np.prod(b[2:] - b[:2]) - inter
            )
            dpx = np.linalg.norm((b[2:] + b[:2] - m[:2] - m[2:]) * 0.5)
            modekey = "held" if r.get("held") else "fit"
            metrics[modekey].append(
                [
                    iou,
                    dpx,
                    abs(b[3] - m[3]),
                    dpx * np.median(z) / (cs[c].params[0] * 724 / cs[c].width),
                ]
            )
            if r["depth"]:
                metrics["depth_z_relative"].append(
                    abs(np.median(z) - r["depth"]) / r["depth"]
                )
            if r not in show:
                continue
            im = read(c, r["n"])
            uv = np.rint(bo * 16).astype(np.int32)
            mask = (
                cv2.imread(str(root / "masks/dyn" / c / f"{r['n']:06}.png"), -1)
                == r["id"]
            ).astype("uint8")
            conts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[
                0
            ]
            cv2.drawContours(im, conts, -1, (242, 125, 215), 1, cv2.LINE_AA)
            for a in range(4):
                for aa, bb, col in [
                    (a, (a + 1) % 4, (62, 217, 239)),
                    (a, a + 4, (165, 217, 157)),
                    (a + 4, (a + 1) % 4 + 4, (165, 217, 157)),
                ]:
                    cv2.line(im, uv[aa], uv[bb], col, 1, cv2.LINE_AA, 4)
            center = (m[:2] + m[2:]) / 2
            width = np.clip(max((m[2] - m[0]) * 1.5, (m[3] - m[1]) * 2.8), 94, 720)
            height = width * 0.57
            x = int(np.clip(center[0] - width / 2, 0, 724 - width))
            y = int(np.clip(center[1] - height / 2, 0, 469 - height))
            crop = cv2.resize(im[y : y + int(height), x : x + int(width)], (320, 180))
            cap = np.full((23, 320, 3), 26, np.uint8)
            text(cap, "%s  %.3f s  %s   IoU %.2f" % (c, r["t"], modekey, iou))
            strip.append(np.vstack([crop, cap]))
        while len(strip) < 3:
            strip.append(strip[-1] if strip else np.zeros((203, 320, 3), np.uint8))
        cams.append(np.hstack(strip))
    if cams:
        save(f"{mode}-{s['name']}-projection", np.vstack(cams))
    v = np.diff(np.c_[track["x"], track["y"]], axis=0) * 36
    speed = np.linalg.norm(v, axis=1)
    turn = np.degrees(np.abs(np.diff(np.unwrap(track["yaw"]))))
    result = {
        k: dict(
            n=len(v),
            p50=np.round(np.median(v, axis=0), 3).tolist(),
            p90=np.round(np.percentile(v, 90, axis=0), 3).tolist(),
        )
        for k, v in metrics.items()
    }
    held_total = sum(bool(r.get("held")) for r in rows)
    held = metrics.get("held", [])
    result.update(
        held_masks_total=held_total,
        held_no_track=held_total - len(held),
        held_projected_iou_below_half=sum(float(v[0]) < 0.5 for v in held),
        poses=len(track["t"]),
        time=np.round(track["t"][[0, -1]], 3).tolist(),
        predicted_s=round(track["state"].sum() / 36, 2),
        speed_p50=round(np.median(speed), 2),
        speed_p95=round(np.percentile(speed, 95), 2),
        yaw_step_p95_deg=round(np.percentile(turn, 95), 2),
        max_step_m=round(speed.max() / 36, 2),
        camera_ids=s["fragments"],
    )
    xyz = np.c_[track["x"], track["y"]]
    ego = tr.T_world_ego(track["t"])[:, :2, 3]
    pts = np.vstack([xyz, ego])
    lo, hi = pts.min(0), pts.max(0)
    center = (lo + hi) / 2
    scl = min(550 / max(hi[0] - lo[0], 2), 270 / max(hi[1] - lo[1], 2))
    uv = lambda a: np.rint((a - center) * [scl, -scl] + [320, 178]).astype(int)
    im = np.full((345, 640, 3), 25, np.uint8)
    cv2.polylines(im, [uv(ego)], False, (130, 130, 130), 2, cv2.LINE_AA)
    for i in range(1, len(xyz)):
        cv2.line(
            im,
            tuple(uv(xyz[i - 1])),
            tuple(uv(xyz[i])),
            (216, 119, 215) if track["state"][i] else (141, 220, 128),
            3,
            cv2.LINE_AA,
        )
    text(im, s["name"] + "  /  " + mode + "  /  north up")
    text(im, "ego grey   observed green   motion prediction violet", (8, 335))
    cv2.line(im, (515, 330), (int(515 + min(100, 10 * scl)), 330), (234, 234, 234), 2)
    text(im, f"{min(100, 10 * scl) / scl:.0f} m", (515, 320))
    save(f"{mode}-{s['name']}-track", im)
    scene = Path("scene", NAMES[mode] + "-movers-v1", "objects")
    a = np.concatenate(
        [
            np.frombuffer(
                (scene / f"{s['id']}{k}.ply").read_bytes().split(b"end_header\n", 1)[1],
                "<f4",
            ).reshape(-1, 24)
            for k in ["", ".pred"]
        ]
    )
    tmin, tmax = a[:, 17].min(), a[:, 17].max()
    tm = [tmin + 0.04, (tmin + tmax) * 0.5, tmax + 0.26]
    sheet = []
    for t in tm:
        cap = np.full((23, 1408, 3), 24, np.uint8)
        text(cap, f"{s['name']}   {mode}   {t:.3f} s   |   observed + inferred surface")
        sheet.append(np.vstack([cap, orb(a, t, size)]))
    save(f"{mode}-{s['name']}-model", np.vstack(sheet))
    return result


def run():
    results = {}
    for mode in NAMES:
        results[mode] = {}
        specs = json.loads(
            (
                (W / "subjects.json")
                if (W / "subjects.json").exists()
                else Path(__file__).with_name("subjects.json")
            ).read_text()
        )
        for sp in specs:
            if sp.get("review_only"):
                continue
            path = W / f"{mode}-{sp['id']}.pickle"
            if not path.exists():
                continue
            s, rows, t, size, stats, scan = pickle.load(open(path, "rb"))
            results[mode][s["name"]] = plate(s, rows, playback(t, mode), size, mode)
            print(mode, s["name"], results[mode][s["name"]].get("held"), flush=True)
    json.dump(results, open(OUT / "measurements.json", "w"), indent=2)


def prefix():
    """Compare a physically truncated observation stream with the published run."""
    from tempfile import TemporaryDirectory
    from .tracking import Factors, make_track
    from .model import bake, INF

    out = []
    for id, cut in [(1, 103.7), (2, 131.3)]:
        sp, rows, full, size, stats, scan = pickle.load(
            open(W / f"live-{id}.pickle", "rb")
        )
        rows = [r for r in rows if r["release"] <= cut]
        f = Factors([r for r in rows if not r.get("held")], sp["class"], "live")
        track, sc, _, knots = make_track(f, "live")
        take = track["t"] <= cut
        fulltake = full["t"] <= cut
        err = max(np.max(np.abs(track[k][take] - full[k][fulltake])) for k in track)
        assert err < 1e-8
        knots = {k: v[knots["t"] <= cut] for k, v in knots.items()}
        ct = {k: v[take] for k, v in track.items()}
        measured = []
        with TemporaryDirectory(dir=W) as tmp:
            bake((sp, rows, ct, size, stats, knots), "live", Path(tmp))
            for suffix in [".ply", ".pred.ply"]:

                def visible(path):
                    a = np.frombuffer(
                        path.read_bytes().split(b"end_header\n", 1)[1], "<f4"
                    ).reshape(-1, 24)
                    a = a[a[:, 17] <= cut].copy()
                    a[:, 18] = np.where(a[:, 18] > cut, INF, a[:, 18])
                    a[:, 19] = np.minimum(a[:, 19], cut)
                    for j in [20, 22]:
                        absent = a[:, j] > cut
                        a[:, j + 1] = np.minimum(a[:, j + 1], cut)
                        a[absent, j : j + 2] = INF
                    return a

                file = f"{id}{suffix}"
                aa = visible(Path(tmp) / file)
                bb = visible(Path("scene/live-v5-movers-v1/objects") / file)
                assert np.array_equal(aa, bb), (id, suffix, aa.shape, bb.shape)
                measured.append(len(aa))
        out.append(
            dict(
                id=id,
                cutoff_s=cut,
                track_samples=int(take.sum()),
                max_track_difference=float(err),
                identical_released_rows=measured,
            )
        )
        print(out[-1], flush=True)
    (OUT / "prefix.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    import sys

    prefix() if "--prefix" in sys.argv else run()
