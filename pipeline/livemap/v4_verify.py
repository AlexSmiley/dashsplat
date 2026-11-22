"""Verify the actual scene, instance row references, near/far births and capture cadence."""

from .v4_render import *


def verify(scope="slice"):
    root = Path(os.environ.get("V4_DEST", RS / "scene/live-v4"))
    doc = json.loads((root / "map.json").read_text())
    gr = RS / "scene/road-live-v2"
    gm = json.loads((gr / "map.json").read_text())
    assert [i for i in doc["tiles"] if i["layer"] == "ground"] == gm["tiles"]
    assert all(os.path.samefile(gr / i["file"], root / i["file"]) for i in gm["tiles"])
    assert cv.imread(str(root / doc["sky"])).shape == (512, 1024, 3)
    sc = json.loads((V4 / (scope + "-summary.json")).read_text())
    p = json.loads((V4 / (scope + "-plan.json")).read_text())
    side = json.loads(
        (
            V4 / ("instances.json" if scope == "slice" else "whole-instances.json")
        ).read_text()
    )
    rng = {}
    null = 0
    tagrows = {}
    undrawn = 0
    for e in side["frames"]:
        for inst in e["instances"]:
            runs = inst.get("tile_runs", [])
            assert sum(i["count"] for i in runs) == inst["seeds"]
            null += inst["centroid_world"] is None
            undrawn += not runs
            for r in runs:
                tagrows.setdefault(r["tile"], []).append(
                    (r["first"], r["count"], e["released"])
                )
    tiles = [i for i in doc["tiles"] if i["layer"] == "objects"]
    count = 0
    dead = 0
    last = 0
    maxlive = 0
    born = []
    end = []
    for tile in tiles:
        a = reader(root / tile["file"])
        assert len(a) == tile["count"] and a.shape[1] == 24 and np.isfinite(a).all()
        assert (
            np.all(a[:, 17] < a[:, 18])
            and np.all(a[:, 17] <= a[:, 19])
            and np.all(np.diff(a[:, 19:], axis=1) >= 0)
        )
        assert np.all(
            np.where(a[:, 19:] < 9e8, a[:, 19:], a[:, 17, None]) <= a[:, 18, None]
        )
        assert np.allclose(np.linalg.norm(a[:, 13:17], axis=1), 1, atol=1e-4)
        count += len(a)
        dead += int((a[:, 18] < sc["last_release"]).sum())
        born.append(
            np.searchsorted(
                np.sort(a[:, 17]), np.arange(0, gm["t_end"] + 0.1, 0.2), "right"
            )
        )
        end.append(
            np.searchsorted(
                np.sort(a[:, 18]), np.arange(0, gm["t_end"] + 0.1, 0.2), "right"
            )
        )
        if tile["kind"] == "transient":
            fin = 0
            for lo, n, t in sorted(tagrows.pop(tile["id"])):
                assert lo == fin and np.all(abs(a[lo : lo + n, 17] - t) < 0.00005)
                fin += n
            assert fin == len(a)
    assert not tagrows and count == sc["records"]
    maxlive = int((np.sum(born, 0) - np.sum(end, 0)).max())
    raw = np.memmap(V4 / (scope + ".bin"), "f", "r", shape=(count, 22))
    tags = np.memmap(V4 / (scope + ".tags"), "i", "r", shape=(count, 3))
    timeline = np.load(RS / "data/drive/timeline.npz")
    for c, es in p.items():
        ci = NAMES.index(c)
        m = tags[:, 0] == ci
        n = tags[m, 1]
        centres = np.array([rows("live", c, e["t"])[0][H // 2, :3, 3] for e in es])
        r = np.linalg.norm(raw[m, :3] - centres[n], axis=1)
        ts = np.array([e["t"] for e in es])
        release = np.array([ready("live", c, e) for e in es])
        dt = np.diff(ts)
        speed = abs(
            np.interp((ts[:-1] + ts[1:]) * 0.5, timeline["t"], timeline["speed"])
        )
        rng[c] = dict(
            frames=len(es),
            rate=(len(es) - 1) / np.ptp(ts),
            moving_hz=1 / np.mean(dt[speed >= 0.2]) if np.any(speed >= 0.2) else None,
            stopped_hz=1 / np.mean(dt[speed < 0.2]) if np.any(speed < 0.2) else None,
            p95_gap=float(np.percentile(dt, 95)),
            max_gap=float(dt.max()),
            release_lag_max=float((release - ts).max()),
            release_reorders=int((np.diff(release) < 0).sum()),
            far_100=int((r >= 100).sum()),
            range_max=float(r.max()),
            range_p95=float(np.percentile(r, 95)),
        )
    whole = plan("whole")
    ready_inputs = {}
    for c, es in whole.items():
        later = [e for e in es if e["t"] > 140]
        ready_inputs[c] = dict(
            required=len(es),
            after_140=len(later),
            depth_after_140=sum(
                (RS / "depth" / c / f"{e['n']:06d}.npy").exists() for e in later
            ),
            sky_after_140=sum(
                (RS / "masks/sky" / c / f"{e['n']:06d}.png").exists() for e in later
            ),
        )
    info = dict(
        records_verified=count,
        ground_links=len(gm["tiles"]),
        object_tiles=len(tiles),
        transient_tiles=sum(t["kind"] == "transient" for t in tiles),
        retired_before_last=dead,
        peak_extant_5hz=maxlive,
        instances_with_no_valid_depth=null,
        instances_without_exported_seed=undrawn,
        range_and_rate=rng,
        whole_input_snapshot=ready_inputs,
    )
    (V4 / (scope + "-verified.json")).write_text(json.dumps(info, indent=2))
    print(info, flush=True)


if __name__ == "__main__":
    verify(sys.argv[1] if len(sys.argv) > 1 else "slice")
