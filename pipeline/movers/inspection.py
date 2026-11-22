"""Compare v3 bus-wheel contacts with two native cameras, plus junction human screen."""

from pathlib import Path
from collections import defaultdict
import pickle, json, cv2, numpy as np
from .demo import R
from .model import projection, interp
from .observations import read
from .shape import silhouette
from pipeline.common.rig import load_rig, Trajectory, frame_table
from .inventory import W
from .proof import text

OUT = R / "report"
OUT.mkdir(exist_ok=True)


def bus():
    scene = Path("scene/stitched-opt-v3-movers-v1")
    o = next(
        s
        for s in json.load(open(scene / "objects.json"))["objects"]
        if s["class"] == "bus"
    )
    s, rows, *_ = pickle.load(open(W / "stitched-2.pickle", "rb"))
    tr = o["track"]
    size = np.array(o["size"])
    p = silhouette("bus", size)
    w = np.array(
        [[a * size[0], b * size[1], 0.0] for a in [-0.315, 0.31] for b in [-0.49, 0.49]]
    )
    kin = Trajectory("poses/ego.npz")
    cs = load_rig()
    strips = []
    data = {}
    for c in ("left_pillar", "left_repeater"):
        cam = cs[c]
        rr = [r for r in rows if r["c"] == c and r["key"]]
        tiles = []
        bottom = []
        held = []
        picked = []
        roadz = []
        ego = []
        for r in rr:
            if r["bb"][0] <= 1 or r["bb"][2] >= 722 or r["bb"][2] - r["bb"][0] < 55:
                continue
            pos = interp(tr, r["rt"])
            E = kin.T_world_ego(r["rt"])
            rel = (pos[:3] - E[:3, 3]) @ E[:3, :3]
            ego.append([r["t"], *rel])
            uv, ok, _, depth = projection(
                cam, kin, r["t"], tr, np.vstack([w, p]), np.zeros((len(w) + len(p), 3))
            )
            valid = ok[:4]
            mask = cv2.imread(f"masks/dyn/{c}/{r['n']:06}.png", -1) == r["id"]
            im = read(c, r["n"])
            outline = np.rint(uv[4:]).astype(np.int32)
            cv2.polylines(
                im, [cv2.convexHull(outline)], True, (78, 204, 230), 1, cv2.LINE_AA
            )
            cv2.drawContours(
                im,
                cv2.findContours(
                    mask.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )[0],
                -1,
                (232, 100, 210),
                1,
                cv2.LINE_AA,
            )
            side = 0 if np.mean(depth[[0, 2]]) < np.mean(depth[[1, 3]]) else 1
            roadpath = Path(f"masks/road/{c}/{r['n']:06}.png")
            dpath = Path(f"depth/{c}/{r['n']:06}.npy")
            road = cv2.imread(str(roadpath), -1) > 0 if roadpath.exists() else None
            dep = np.load(dpath).astype(float) if dpath.exists() else None
            for j in [side, side + 2]:
                x, y = np.rint(uv[j]).astype(int)
                if not valid[j] or not 5 < x < 718 or not 5 < y < 459:
                    continue
                ys = np.where(mask[:, x - 2 : x + 3].any(1))[0]
                if len(ys):
                    residual = float(uv[j, 1] - ys[-1] - 0.5)
                    bottom.append(residual)
                    if r.get("held"):
                        held.append(residual)
                    cv2.circle(im, (x, y), 3, (40, 240, 250), -1)
                    cv2.line(im, (x - 4, ys[-1]), (x + 4, ys[-1]), (234, 104, 208), 1)
                if road is None or dep is None:
                    continue
                yy, xx = np.mgrid[
                    max(0, y - 3) : min(469, y + 14), max(0, x - 7) : min(724, x + 8)
                ]
                keep = road[yy, xx] & ~mask[yy, xx] & (dep[yy, xx] > 0)
                xx = xx[keep]
                yy = yy[keep]
                if len(xx) < 3:
                    continue
                pix = np.c_[xx + 0.5, yy + 0.5] * [cam.width / 724, cam.height / 469]
                rays = cam.unproject(pix)
                cw = kin.T_world_ego(cam.row_time(r["t"], pix[:, 1])) @ cam.T_ego_cam
                pw = (
                    np.einsum(
                        "nij,nj->ni",
                        cw[:, :3, :3],
                        rays * (dep[yy, xx] / rays[:, 2])[:, None],
                    )
                    + cw[:, :3, 3]
                )
                cs0 = np.cos(pos[3])
                sn = np.sin(pos[3])
                wx = pos[:2] + [
                    cs0 * w[j, 0] - sn * w[j, 1],
                    sn * w[j, 0] + cs0 * w[j, 1],
                ]
                near = np.linalg.norm(pw[:, :2] - wx, axis=1) < 1.4
                if near.sum() >= 3:
                    roadz.append(float(pos[2] - np.median(pw[near, 2])))
            b = np.array(r["bb"])
            ctr = (b[:2] + b[2:]) / 2
            width = min(720, max(105, int((b[2] - b[0]) * 1.32)))
            height = int(width * 0.55)
            x = int(np.clip(ctr[0] - width / 2, 0, 724 - width))
            y = int(np.clip(ctr[1] - height / 2, 0, 469 - height))
            crop = cv2.resize(im[y : y + height, x : x + width], (370, 206))
            cap = np.full((25, 370, 3), 24, "uint8")
            text(cap, f"{c}   {r['t']:.3f}s   left {rel[1]:.1f}m")
            tiles.append(np.vstack([crop, cap]))
            picked.append(r["t"])
        ix = np.linspace(0, len(tiles) - 1, 3).astype(int)
        strips.append(np.hstack([tiles[i] for i in ix]))
        data[c] = dict(
            shown=[picked[i] for i in ix],
            wheel_count=len(bottom),
            held_wheels=len(held),
            held_abs_p50_px=float(np.median(abs(np.array(held)))) if held else None,
            wheel_minus_mask_bottom_px_p50=float(np.median(bottom)),
            abs_p90_px=float(np.percentile(abs(np.array(bottom)), 90)),
            depth_road_neighbour_count=len(roadz),
            road_clearance_p50_m=float(np.median(roadz)) if roadz else None,
            road_clearance_abs_p90_m=float(np.percentile(abs(np.array(roadz)), 90))
            if roadz
            else None,
            ego_left_range_m=[min(v[2] for v in ego), max(v[2] for v in ego)],
        )
    cv2.imwrite(
        str(OUT / "bus-v3-road.jpg"), np.vstack(strips), [cv2.IMWRITE_JPEG_QUALITY, 85]
    )
    json.dump(data, open(OUT / "bus-road.json", "w"), indent=2)
    print(data, flush=True)


def people():
    cs = load_rig()
    rows = defaultdict(list)
    counts = {}
    for c, cam in cs.items():
        tab = frame_table(cam)
        frames = {}
        for root in [W / "extra", Path(".")]:
            for p in (root / "masks/dyn" / c).glob("[0-9]*.json"):
                n = int(p.stem)
                t = tab[n]["t"]
                if 190 <= t <= 215:
                    frames[n] = p
        counts[c] = len(frames)
        for n, p in sorted(frames.items()):
            for z in json.load(open(p)):
                if z["class"] in ("rider", "person", "motorcycle", "bicycle"):
                    rows[(c, str(p.parent.parent.parent.parent), z["id"])].append(
                        (n, tab[n]["t"], z)
                    )
    cells = []
    info = []
    for (c, root, uid), rr in rows.items():
        r = max(rr, key=lambda r: r[2]["area"])
        im = read(c, r[0])
        l, t, h, b = r[2]["bbox"]
        cv2.rectangle(im, (l, t), (h, b), (136, 224, 158), 1)
        width = min(720, max(80, int((h - l) * 2.2), int((b - t) * 2.8)))
        hh = width // 2
        x = int(np.clip((l + h - width) / 2, 0, 724 - width))
        y = int(np.clip((t + b - hh) / 2, 0, 469 - hh))
        im = cv2.resize(im[y : y + hh, x : x + width], (220, 110))
        cap = np.full((22, 220, 3), 25, "uint8")
        text(cap, f"{c} {uid} {r[2]['class']} {r[1]:.1f}", scale=0.36)
        cells.append(np.vstack([im, cap]))
        info.append(
            dict(
                camera=c,
                source=root,
                id=uid,
                cls=r[2]["class"],
                n=len(rr),
                time=[rr[0][1], rr[-1][1]],
                area=r[2]["area"],
            )
        )
    for i in range(0, len(cells), 16):
        chunk = cells[i : i + 16]
        chunk += [np.zeros_like(cells[0])] * (-len(chunk) % 4)
        cv2.imwrite(
            str(OUT / f"people-{i // 16}.jpg"),
            np.vstack([np.hstack(chunk[j : j + 4]) for j in range(0, len(chunk), 4)]),
        )
    json.dump(
        dict(frames=counts, candidates=info), open(OUT / "people.json", "w"), indent=2
    )
    print("junction frames", counts, "candidate fragments", len(info))


def hindsight_panel():
    # Use the actual v3 track where available (including its surface-contact
    # correction) and the upgraded late-window tracks on the live static base.
    from .packets import iou

    ego = Trajectory("poses/ego.npz")
    cs = load_rig()
    models = json.load(open(OUT / "shape-texture.json"))
    counts = np.zeros(3, int)
    detail = {}
    held = []
    for scene in ("stitched-opt-v3-movers-v1", "live-v5-traffic-hindsight"):
        src = Path("scene") / scene
        for o in json.load(open(src / "objects.json"))["objects"]:
            sp, rr, *_ = pickle.load(open(W / f"stitched-{o['id']}.pickle", "rb"))
            tr = o["track"]
            p = silhouette(
                o["class"], o["size"], models[sp["name"]].get("cabin_fraction", 0)
            )
            v = np.zeros(3, int)
            dist = []
            for r in rr:
                if not r.get("held"):
                    continue
                held.append(dict(label=sp["name"], cam=r["c"], frame=r["n"], t=r["rt"]))
                v[0] += 1
                if not o["t_first"] <= r["rt"] <= o["t_last"]:
                    v[1] += 1
                    continue
                uv, ok, *_ = projection(cs[r["c"]], ego, r["t"], tr, p, p * 0)
                bb = np.clip(
                    np.r_[uv.min(0), uv.max(0)], [0, 0, 0, 0], [724, 469, 724, 469]
                )
                obs = np.array(r["bb"])
                q = iou(bb, obs)
                v[2] += q < 0.5
                dist.append(np.linalg.norm((bb[:2] + bb[2:] - obs[:2] - obs[2:]) * 0.5))
            step = np.max(
                np.linalg.norm(
                    np.diff(np.c_[tr["x"], tr["y"], tr["z"]], axis=0), axis=1
                )
            )
            detail[sp["name"]] = dict(
                counts=v.tolist(),
                max_step_m=round(float(step), 4),
                centre_median_px=float(np.median(dist)) if dist else None,
            )
            counts += v
    assert counts[0] == 136
    json.dump(
        dict(
            total=counts.tolist(),
            objects=detail,
            flags_over_1_5=[k for k, o in detail.items() if o["max_step_m"] > 1.5],
        ),
        open(OUT / "hindsight-evaluation.json", "w"),
        indent=2,
    )
    json.dump(held, open(OUT / "held-136.json", "w"), indent=1)
    print("hindsight", counts.tolist())


def audit():
    from .model import FIELDS
    from .demo import targets

    out = {}
    scenes = [
        n
        for n in ("live-v5-movers-v2", "live-v5-movers-v3", "live-v5-traffic-hindsight")
        if (Path("scene") / n / "objects.json").exists()
    ] + [s + "-movers-v1" for s in targets()]
    for name in scenes:
        src = Path("scene") / name
        m = json.load(open(src / "map.json"))
        idx = json.load(open(src / "rsb/index.json"))
        assert idx.get("objects") is True
        assert all(
            v["id"] in idx["tiles"]
            and (src / "rsb" / idx["tiles"][v["id"]]["file"]).exists()
            for v in m["tiles"]
        )
        oo = json.load(open(src / "objects.json"))["objects"]
        gen = 0
        steps = {}
        for o in oo:
            tr = o["track"]
            assert (
                len(set(map(len, tr.values()))) == 1
                and o["t_last"] >= o["t_first"]
                and np.allclose(np.diff(tr["t"]), 1 / 36, atol=2e-6)
            )
            steps[o["id"]] = round(
                float(
                    np.max(
                        np.linalg.norm(
                            np.diff(np.c_[tr["x"], tr["y"], tr["z"]], axis=0), axis=1
                        )
                    )
                ),
                4,
            )
            assert set(tr["state"]) <= {0, 1}
            for key in ["file", "predicted_file"]:
                head, blob = (src / o[key]).read_bytes().split(b"end_header\n", 1)
                assert [
                    v.decode().split()[-1]
                    for v in head.splitlines()
                    if v.startswith(b"property float")
                ] == FIELDS
                a = np.frombuffer(blob, "<f4").reshape(-1, 24)
                assert (
                    len(a) == int(head.split(b"element vertex ")[1].splitlines()[0])
                    and np.isfinite(a).all()
                    and (a[:, 18] > a[:, 17]).all()
                )
                gen += len(a)
        out[name] = dict(
            tiles=len(m["tiles"]),
            sky=m.get("sky"),
            objects=len(oo),
            plys=len(oo) * 2,
            generations=gen,
            step_m=steps,
            flags_over_1_5=[u for u, v in steps.items() if v > 1.5],
        )
    json.dump(out, open(OUT / "published-audit.json", "w"), indent=2)
    print({k: v["objects"] for k, v in out.items()})


def prefix():
    # On two actual truncated streams, compare assignments, motion-confirmation
    # and the viewer's already released positions and colour generations.
    from .online import associate as old_associate
    from .handoff import associate as component_associate, robust_class
    from .semantics import release as semantic_release
    from .online_scene import scene_name
    from .filter import refine
    from .motion import release
    from .model import bake
    import tempfile

    events = pickle.load(open(R / "packets.pickle", "rb"))
    full = pickle.load(open(R / "identities.pickle", "rb"))
    tracks = {s[0]["id"]: s for s in pickle.load(open(R / "online.pickle", "rb"))}
    flow = pickle.load(open(R / "motion.pickle", "rb"))
    component = R.name == "component"
    sem = pickle.load(open(R / "semantics.pickle", "rb"))
    dst = Path("scene") / scene_name() / "objects"
    checks = {}
    eligible = {
        o["id"]: o for o in json.load(open(dst.parent / "objects.json"))["objects"]
    }

    def keys(rr):
        return [(r["c"], r["n"], r["id"]) for r in rr]

    def released(path, t):
        a = np.frombuffer(
            path.read_bytes().split(b"end_header\n", 1)[1], "<f4"
        ).reshape(-1, 24)
        a = a[a[:, 17] <= t].copy()
        a[:, 18:] = np.where(a[:, 18:] <= t, a[:, 18:], 1e9)
        return a

    for cut in [103.7, 131.3, 319.05] if component else [103.7, 131.3]:
        prefix_events = [ev for ev in events if ev[0] <= cut]
        hs = (
            component_associate(prefix_events)
            if component
            else [
                dict(id=v.uid, rows=v.rows, confirm=v.confirm, cls=v.birth_cls)
                for v in old_associate(prefix_events)
            ]
        )
        old = {h["id"]: h for h in full if h["rows"][0]["release"] <= cut}
        assert len(old) == len(hs)
        n = sum(len(h["rows"]) for h in hs)
        models = []
        for h in hs:
            uid = h["id"]
            orig = old[uid]
            assert keys(h["rows"]) == keys(
                [r for r in orig["rows"] if r["release"] <= cut]
            )
            assert h["confirm"] == (
                orig["confirm"]
                if orig["confirm"] is not None and orig["confirm"] <= cut
                else None
            )
            if (
                uid not in eligible
                or not eligible[uid]["t_first"] <= cut <= eligible[uid]["t_last"]
            ):
                continue
            flow_at = release([v for v in flow[uid] if v["t"] <= cut])
            if component:
                verified = semantic_release(
                    [v for v in sem[uid] if v[0] <= cut], h["cls"]
                )
                assert verified is not None
                t = max(verified, h["confirm"])
                h = dict(
                    h,
                    cls=robust_class([r for r in h["rows"] if r["release"] <= t]),
                    confirm=t,
                )
            check = refine(h, component, flow_at)
            assert check is not None
            tracked = tracks[uid]
            m = tracked[2]["t"] <= cut
            assert all(
                np.array_equal(check[2][k][: m.sum()], tracked[2][k][m])
                for k in tracked[2]
            )
            if flow_at is not None:
                assert flow_at == release(flow[uid])
            with tempfile.TemporaryDirectory(dir=R) as temp:
                temp = Path(temp)
                bake(check, "live", temp, True)
                counts = []
                for suf in [".ply", ".pred.ply"]:
                    fn = str(uid) + suf
                    a = released(temp / fn, cut)
                    b = released(dst / fn, cut)
                    assert np.array_equal(a, b), (uid, suf, a.shape, b.shape)
                    counts.append(len(a))
            models.append(
                dict(id=uid, knots=int(m.sum()), measured=counts[0], inferred=counts[1])
            )
        checks[str(cut)] = dict(
            discoveries=len(hs), assigned_observations=n, active_models=models
        )
    json.dump(checks, open(OUT / "causality.json", "w"), indent=2)
    print(checks)


def control_panel():
    # Evaluation only. Neither candidate discovery nor publication reads these
    # negative controls or the formerly reviewed blue-SUV mask identities.
    from .online_scene import scene_name

    events = pickle.load(open(R / "identities.pickle", "rb"))
    base = {h["id"]: h for h in pickle.load(open(W / "round2/identities.pickle", "rb"))}
    excluded = pickle.load(open(W / "live-18.pickle", "rb"))[1]
    eligible = {
        o["id"]: o
        for o in json.load(open(Path("scene") / scene_name() / "objects.json"))[
            "objects"
        ]
    }
    testsets = {
        "blue-suv-ambiguous": excluded,
        "storefront-sign": base[153]["rows"],
        "fuel-price-sign": base[1550]["rows"],
    }
    report = {}
    for name, rows in testsets.items():
        keys = {
            (r["c"], r["n"], r["id"]) for r in rows if r["key"] and not r.get("held")
        }
        matches = []
        for h in events:
            o = eligible.get(h["id"])
            if not o:
                continue
            hits = [
                r
                for r in h["rows"]
                if (r["c"], r["n"], r["id"]) in keys
                and o["t_first"] <= r["rt"] <= o["t_last"]
            ]
            if hits:
                matches.append(
                    dict(
                        id=h["id"],
                        exposures=len(hits),
                        t_first=hits[0]["t"],
                        t_last=hits[-1]["t"],
                    )
                )
        report[name] = dict(training_keys=len(keys), published_matches=matches)
    json.dump(report, open(OUT / "stationary-controls.json", "w"), indent=2)
    print(report)


def discoveries():
    from .online_scene import scene_name

    hs = pickle.load(open(R / "identities.pickle", "rb"))
    present = {
        o["id"]: o["t_first"]
        for o in json.load(open(Path("scene") / scene_name() / "objects.json"))[
            "objects"
        ]
    }
    cells = []
    ids = []
    for h in hs:
        rel = present.get(h["id"])
        if rel is None:
            continue
        rr = [r for r in h["rows"] if abs(r["release"] - rel) < 0.39]
        r = (
            max(rr, key=lambda r: r["area"])
            if rr
            else min(h["rows"], key=lambda r: abs(r["release"] - rel))
        )
        im = read(r["c"], r["n"])
        l, t, rx, b = r["bb"]
        cv2.rectangle(im, (int(l), int(t)), (int(rx), int(b)), (90, 230, 175), 1)
        w = int(np.clip(max(72, (rx - l) * 1.4, (b - t) * 1.9), 72, 720))
        hh = int(w * 0.60)
        x = int(np.clip((l + rx - w) / 2, 0, 724 - w))
        y = int(np.clip((t + b - hh) / 2, 0, 469 - hh))
        im = cv2.resize(im[y : y + hh, x : x + w], (195, 117))
        cap = np.full((21, 195, 3), 22, "uint8")
        text(cap, f"#{h['id']} {r['c']} {rel:.1f}", scale=0.35)
        cells.append(np.vstack([im, cap]))
        ids.append(h["id"])
    for ix in range(0, len(cells), 24):
        block = cells[ix : ix + 24]
        block += [np.zeros_like(cells[0])] * (-len(block) % 4)
        plate = np.vstack(
            [np.hstack(block[i : i + 4]) for i in range(0, len(block), 4)]
        )
        cv2.imwrite(
            str(OUT / f"published-crops-{ix // 24}.jpg"),
            plate,
            [cv2.IMWRITE_JPEG_QUALITY, 79],
        )
        print(ix, ids[ix : ix + 24])


def coverage():
    cs = load_rig()
    out = {}
    for c, cam in cs.items():
        ts = frame_table(cam)
        out[c] = {}
        for a, b in [(85, 140), (190, 215), (286, 330)]:
            row = {}
            for root, key, pat in [
                (Path("."), "depth", "*.npy"),
                (W / "extra", "depth", "*.npy"),
                (Path("."), "masks/dyn", "[0-9]*.json"),
                (W / "extra", "masks/dyn", "[0-9]*.json"),
            ]:
                row[str(root / key)] = sum(
                    a <= ts[int(p.stem)]["t"] <= b for p in (root / key / c).glob(pat)
                )
            out[c][f"{a}-{b}"] = row
    json.dump(out, open(OUT / "input-coverage.json", "w"), indent=2)


if __name__ == "__main__":
    import sys

    actions = {
        "audit": audit,
        "controls": control_panel,
        "discoveries": discoveries,
        "coverage": coverage,
        "prefix": prefix,
        "panel": hindsight_panel,
        "bus": bus,
        "people": people,
    }
    for name in sys.argv[1:] or ("bus", "people"):
        actions[name]()
