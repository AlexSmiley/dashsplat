"""Rebake the reviewed demo with the same photometric reference as its plate."""

import pickle, json, numpy as np
from .demo import R, W, attach, targets
from .model import bake
from .photographic import metadata


def run():
    src = W / "round2/hindsight"
    manifest = json.load(open(src / "objects.json"))
    profiles = json.load(open(W / "round2/report/shape-texture.json"))
    bank = R / "hindsight"
    dst = bank / "objects"
    dst.mkdir(parents=True, exist_ok=True)
    out = {}
    retained = []
    for o in manifest["objects"]:
        s, rr, tr, _, stats, scan = pickle.load(
            open(W / f"stitched-{o['id']}.pickle", "rb")
        )
        sz = np.array(o["size"])
        tr = {k: np.asarray(v) for k, v in o["track"].items()}
        from .shutters import correct, clock
        from .tracking import Factors, make_track
        from .upgrade import smooth_corrections

        rr = sorted(
            [correct(r) for r in rr if r["n"] in clock(r["c"])],
            key=lambda r: r["release"],
        )
        f = Factors([r for r in rr if not r.get("held")], s["class"], "stitched")
        result = make_track(f, "stitched")
        if result is None:
            continue
        tr = smooth_corrections(result[0])
        scan = tr
        stats = result[2]
        o["track"] = {k: np.round(v, 6).tolist() for k, v in tr.items()}
        o["t_first"] = float(tr["t"][0])
        o["t_last"] = float(tr["t"][-1])
        retained.append(o)
        data = s, rr, tr, sz, stats, scan
        pickle.dump(data, open(R / f"fit-{s['id']}.pickle", "wb"))
        profile = profiles[s["name"]]
        out[s["name"]] = bake(
            data,
            "stitched",
            dst,
            True,
            profile.get("cabin_fraction", 0.0),
            photographic=True,
        )
        m = metadata(data, [], o["t_first"], o["t_last"])
        m.pop("motion_confidence")
        m["photometry"]["source"] = "stitched-ground-v2"
        o.update(m)
        print(s["name"], flush=True)
    manifest["objects"] = retained
    (bank / "objects.json").write_text(json.dumps(manifest, separators=(",", ":")))
    (R / "report").mkdir(exist_ok=True)
    (R / "report/hindsight-bakes.json").write_text(json.dumps(out, indent=2))
    for parent in targets():
        attach(parent, bank, parent + "-movers-photo-v1", assets=bank)
    attach(
        "live-v5",
        bank,
        "live-v5-traffic-photo-hindsight",
        [(190, 215), (286, 330)],
        bank,
    )


if __name__ == "__main__":
    run()
