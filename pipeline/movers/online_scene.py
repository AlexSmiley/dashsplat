"""Export discovered IDs and evaluate the frozen 136-mask panel separately."""

import argparse, json, pickle, os
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from .demo import R
from .inventory import W
from .model import bake, projection
from .publish import source_tiles
from .online import outline
from .motion import release
from .semantics import release as semantic_release
from .packets import iou
from pipeline.common.rig import load_rig, Trajectory

OUT = R / "report"
OUT.mkdir(exist_ok=True)


def scene_name():
    if R.name.startswith("photo-"):
        return "live-v5-movers-v4" + ("-held" if "held" in R.name else "")
    return "live-v5-movers-v3" if R.name == "component" else "live-v5-movers-v2"


def exact_tags(objs, data):
    by_id = {d[0]["id"]: d[1] for d in data}
    return [
        dict(frame=r["n"], cam=r["c"], id=r["id"])
        for o in objs
        for r in by_id[o["id"]]
        if r["root"] == "." and o["t_first"] <= r["rt"] <= o["t_last"]
    ]


def publish():
    res = pickle.load(open(R / "online.pickle", "rb"))
    out = Path("scene") / scene_name()
    dst = out / "objects"
    dst.mkdir(exist_ok=True, parents=True)
    objs = []
    models = {}
    skips = []
    flow = pickle.load(open(R / "motion.pickle", "rb"))
    semantic = pickle.load(open(R / "semantics.pickle", "rb"))
    semantic_rejected = []
    motion_rejected = []
    footprint = (
        pickle.load(open(R / "footprint.pickle", "rb"))
        if R.name.startswith("photo-")
        else {}
    )
    for ix, data in enumerate(res):
        s, rr, tr, sz, stats, scan = data
        uid = s["id"]
        cls = s["class"]
        verified = semantic_release(semantic.get(uid, []), cls)
        if verified is None:
            semantic_rejected.append(uid)
            continue
        flow_time = release(flow[uid])
        geometry_time = stats.get("geometry_release")
        ready = min(
            [x for x in [flow_time, geometry_time] if x is not None], default=None
        )
        if ready is None or ready >= tr["t"][-1]:
            motion_rejected.append(uid)
            continue
        try:
            if footprint:
                from .photographic import bake_epochs

                m = bake_epochs(data, dst)
            else:
                m = bake(data, "live", dst, True)
        except ValueError as e:
            if not str(e).startswith("no photographs"):
                raise
            skips.append(uid)
            continue
        # Some far/misaligned episodes have no usable surface until after birth.
        # Evaluate render availability from its actual first released photo.
        t0 = max(tr["t"][0], m["first_photo"], ready, verified)
        end = tr["t"][-1]
        if end <= t0:
            skips.append(uid)
            continue
        models[uid] = {
            **m,
            **stats,
            "photo_motion_release": flow_time,
            "semantic_release": verified,
        }
        objs.append(
            dict(
                id=uid,
                **{"class": cls},
                size=sz.tolist(),
                t_first=t0,
                t_last=end,
                file=f"objects/{uid}.ply",
                predicted_file=f"objects/{uid}.pred.ply",
                track={k: np.round(v, 6).tolist() for k, v in tr.items()},
            )
        )
        if footprint:
            from .photographic import metadata

            objs[-1].update(metadata(data, footprint[uid], t0, end))
        if ix % 15 == 0:
            print("baked", ix + 1, "of", len(res), flush=True)
    if R.name == "component" or bool(footprint):
        from .retire import retire

        arbitration = retire(objs, res)
        (OUT / "handover-retirements.json").write_text(
            json.dumps(arbitration, indent=2)
        )
    # Exact matched mask exposures only; other static and transient rows survive.
    links, removed = source_tiles(
        "live", Path("scene/live-v5"), out, exact_tags(objs, res)
    )
    temp = out / "objects.next"
    temp.write_text(json.dumps(dict(objects=objs), separators=(",", ":")))
    os.replace(temp, out / "objects.json")
    json.dump(
        dict(
            models=models,
            skipped=skips,
            not_motion_verified=motion_rejected,
            not_category_verified=semantic_rejected,
            links=links,
            removed=removed,
        ),
        open(OUT / "online-bakes.json", "w"),
        indent=1,
    )
    print(
        "published",
        len(objs),
        "photo misses",
        len(skips),
        "removed",
        removed,
        flush=True,
    )


def eval_scene(rendered=True, native=False, output=None):
    data = pickle.load(open(R / "online.pickle", "rb"))
    eligible = {}
    ends = {}
    if rendered:
        eligible = {
            o["id"]: o["t_first"]
            for o in json.load(open(Path("scene") / scene_name() / "objects.json"))[
                "objects"
            ]
        }
        ends = {
            o["id"]: o["t_last"]
            for o in json.load(open(Path("scene") / scene_name() / "objects.json"))[
                "objects"
            ]
        }
        data = [d for d in data if d[0]["id"] in eligible]
    obs = defaultdict(list)
    cs = load_rig()
    ego = Trajectory("poses/ego_causal.npz")
    truth = []
    for p in W.glob("stitched-*.pickle"):
        truth.append(pickle.load(open(p, "rb"))[:2])
    if R.name.endswith("native") or native:
        from .shutters import correct

        truth = [(sp, [correct(r) for r in rr]) for sp, rr in truth]
    truth.sort(key=lambda d: d[0]["id"])
    score = np.zeros((len(truth), len(data)), int)
    for j, d in enumerate(data):
        for r in d[1]:
            obs[(r["c"], r["n"])].append((j, np.asarray(r["bb"])))
    for i, (sp, rows) in enumerate(truth):
        for r in rows:
            if not r["key"] or r.get("held"):
                continue
            m = np.array(r["bb"])
            pairs = [(iou(m, b), j) for j, b in obs[r["c"], r["n"]]]
            if pairs and max(pairs)[0] > 0.67:
                score[i, max(pairs)[1]] += 1
    one = dict(zip(*linear_sum_assignment(-score)))
    owner = np.argmax(score, axis=0)
    report = {}
    steps = {
        d[0]["id"]: round(
            float(
                np.max(
                    np.linalg.norm(
                        np.diff(np.c_[d[2]["x"], d[2]["y"], d[2]["z"]], axis=0), axis=1
                    )
                )
            ),
            4,
        )
        for d in data
        if len(d[2]["t"]) > 1
    }

    def overlap(d, r):
        s, rr, track, sz, *_ = d
        t = r["rt"]
        if t < max(track["t"][0], eligible.get(s["id"], -1e9)) or t > min(
            track["t"][-1], ends.get(s["id"], 1e9)
        ):
            return None
        epochs = d[4].get("class_epochs", [])
        p = outline(next((c for at, c in reversed(epochs) if at <= t), s["class"]))
        uv, ok, *_ = projection(cs[r["c"]], ego, r["t"], track, p, p * 0)
        b = np.clip(np.r_[uv.min(0), uv.max(0)], [0, 0, 0, 0], [724, 469, 724, 469])
        bb = np.asarray(r["bb"])
        return iou(b, bb), float(
            np.linalg.norm((b[:2] + b[2:] - bb[:2] - bb[2:]) * 0.5)
        )

    total = np.zeros((2, 3), int)
    for i, (s, rr) in enumerate(truth):
        j = one.get(i)
        primary = [j] if j is not None and score[i, j] > 0 else []
        sub = [k for k in range(len(data)) if owner[k] == i and score[i, k] >= 2]
        counts = []
        centres = []
        med_iou = []
        for ids in [primary, sub]:
            vals = []
            errors = []
            absent = 0
            for r in rr:
                if not r.get("held"):
                    continue
                v = [q for k in ids if (q := overlap(data[k], r)) is not None]
                if not v:
                    absent += 1
                else:
                    q, e = max(v)
                    vals.append(q)
                    errors.append(e)
            centres.append(float(np.median(errors)) if errors else None)
            med_iou.append(float(np.median(vals)) if vals else None)
            counts.append(
                [
                    absent + len(vals),
                    absent,
                    int(np.count_nonzero(np.array(vals) < 0.5)),
                ]
            )
        total += counts
        report[s["name"]] = dict(
            strict=counts[0],
            centre_median_px=centres[0],
            median_iou=med_iou[0],
            allow_fragment=counts[1],
            primary=[data[k][0]["id"] for k in primary],
            associated_fragments=[data[k][0]["id"] for k in sub],
            max_step_m=steps.get(data[primary[0]][0]["id"]) if primary else None,
        )
    file = OUT / ("online-evaluation.json" if rendered else "filter-evaluation.json")
    result = dict(
        objects=len(data),
        total_strict=total[0].tolist(),
        total_allow_fragment=total[1].tolist(),
        objects_benchmark=report,
        max_frame_step_m_by_discovered_id=steps,
        flags_over_1_5=[uid for uid, v in steps.items() if v > 1.5],
    )
    file = Path(output) if output else file
    file.write_text(json.dumps(result, indent=2))
    print(file, total.tolist(), "step max", max(steps.values()), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["publish", "score", "raw"])
    a = p.parse_args().action
    if a == "publish":
        publish()
    else:
        eval_scene(a == "score")
