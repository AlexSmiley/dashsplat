import argparse, json, pickle
from pathlib import Path
from .observations import observations
from .inventory import W
from .tracking import Factors, make_track, released_rows


def run(ids=None):
    specs = json.loads(
        (
            (W / "subjects.json")
            if (W / "subjects.json").exists()
            else Path(__file__).with_name("subjects.json")
        ).read_text()
    )
    for s in specs:
        if ids and s["id"] not in ids:
            continue
        rows = observations(s)
        if not rows:
            continue
        # Order by complete readout, so prefix fits never consume unreleased rows.
        rows.sort(key=lambda r: r["release"])
        for mode in ["live", "stitched"]:
            rr = released_rows(rows) if mode == "live" else rows
            f = Factors([r for r in rr if not r.get("held")], s["class"], mode)
            result = make_track(f, mode)
            path = W / f"{mode}-{s['id']}.pickle"
            if result:
                track, sc, stats = result[:3]
                scan = result[3] if mode == "live" else track
                pickle.dump((s, rr, track, f.size * sc, stats, scan), open(path, "wb"))
                print(
                    mode,
                    s["name"],
                    len(rows),
                    [round(track["t"][i], 2) for i in (0, -1)],
                    sc,
                    stats,
                    flush=True,
                )
            else:
                path.unlink(missing_ok=True)
                print(mode, s["name"], "motion rejected", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("ids", nargs="*", type=int)
    run(p.parse_args().ids)
