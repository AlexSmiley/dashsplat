"""Fixed-rate live inputs. Isolate pose-conditioned depth; never modify the shared producer tree."""

from .geometry import *
from .inputs import os, Path, subprocess, sys

V4 = RS / "maps/live-v4"
DV = V4 / "inputs"
V4.mkdir(parents=True, exist_ok=True)


def plan(scope="slice"):
    shared = json.loads((RS / "depth/keyframes.json").read_text())
    timeline = np.load(RS / "data/drive/timeline.npz")
    old = frames()
    p = {}
    lo, hi = (84, 140) if scope == "slice" else (0, timeline["t"][-1])
    ticks = []
    t = lo
    while t <= hi:
        ticks.append(t)
        t += (
            0.5
            if abs(np.interp(t, timeline["t"], timeline["speed"])) < 0.2
            else 1 / 4.5
        )
    for c in CAMS:
        f = frame_times(c)
        ids = np.array(sorted(f))
        ts = np.array([f[n] for n in ids])
        odo = np.interp(
            ts,
            timeline["t"],
            np.r_[
                0, np.cumsum(np.hypot(np.diff(timeline["x"]), np.diff(timeline["y"])))
            ],
        )
        existing = set(e["n"] for e in shared["frames"][c])
        hold = {e["n"] for e in old[c] if e["hold"]}
        pri = {e["n"] for e in old[c] if e["train"]}
        p[c] = []
        last = -1
        for tick in ticks:
            opts = []
            for j in range(
                np.searchsorted(ts, tick - 0.103), np.searchsorted(ts, tick + 0.105)
            ):
                n = int(ids[j])
                if n == last or n in hold or (scope == "slice" and n not in existing):
                    continue
                w = [n]
                i = j
                o = odo[i]
                for _ in range(2 if c == "back" else 4):
                    i = np.searchsorted(odo[:i], o - 1, side="right") - 1
                    if i < 0:
                        break
                    w.append(int(ids[i]))
                    o = odo[i]
                if hold.intersection(w):
                    continue
                opts.append(
                    (
                        abs(ts[j] - tick)
                        - (0.043 if n in pri else 0)
                        - (0.035 if n in existing else 0),
                        j,
                        w,
                    )
                )
            if not opts:
                continue
            _, j, w = min(opts)
            last = int(ids[j])
            p[c].append(
                dict(
                    n=last,
                    t=float(ts[j]),
                    t_key=float(tick),
                    window=w,
                    ordinal=len(p[c]),
                    odo=float(odo[j]),
                )
            )
    (V4 / (scope + "-plan.json")).write_text(json.dumps(p))
    return p


def prepare(scope="slice"):
    p = plan(scope)
    root = DV if scope == "slice" else V4 / "whole-inputs"
    (root / "depth").mkdir(parents=True, exist_ok=True)
    shared = json.loads((RS / "depth/keyframes.json").read_text())
    oldbase = W / "live"
    for n in (
        ["calib", "data", "poses", "third_party", "masks"]
        if scope == "slice"
        else ["calib", "data", "poses", "third_party"]
    ):
        if not (root / n).exists():
            (root / n).symlink_to(RS / n)
    for c, entries in p.items():
        cache = root / "depth" / c
        cache.mkdir(exist_ok=True)
        ip = cache / "index.json"
        idx = json.loads(ip.read_text()) if ip.exists() else {}
        sources = [oldbase]
        if scope == "whole":
            sources.append(DV)
        if (
            Path(
                shared.get("depth", {})
                .get("cameras", {})
                .get(c, {})
                .get("trajectory", shared.get("depth", {}).get("trajectory", ""))
            ).name
            == "ego_causal.npz"
        ):
            sources.append(RS)
        for source in sources:
            ixp = source / "depth" / c / "index.json"
            si = json.loads(ixp.read_text()) if ixp.exists() else {}
            for e in entries:
                k = str(e["n"])
                fn = f"{e['n']:06d}.npy"
                v = si.get(k, {})
                src = source / "depth" / c / fn
                dst = cache / fn
                if (
                    not dst.exists()
                    and v.get("window") == e["window"]
                    and v.get("scale_applied") == 1
                    and src.exists()
                    and src.stat().st_size >= H * B * 2
                ):
                    os.link(src, dst)
                    idx[k] = v
        ip.write_text(json.dumps(idx))
        print(
            scope,
            c,
            len(entries),
            "cached",
            sum(str(e["n"]) in idx for e in entries),
            flush=True,
        )
    (root / "depth/keyframes.json").write_text(
        json.dumps(dict(frames=p, selections=shared["selections"]))
    )
    return root, p


def complete(root, c, es):
    ip = root / "depth" / c / "index.json"
    idx = json.loads(ip.read_text())
    for e in es:
        k = str(e["n"])
        fn = f"{e['n']:06d}"
        assert (
            k in idx
            and idx[k]["window"] == e["window"]
            and (root / "depth" / c / (fn + ".npy")).stat().st_size >= H * B * 2
        ), (c, k)
        for m in ["sky", "dyn", "road"]:
            assert (root / "masks" / m / c / (fn + ".png")).is_file(), (m, c, k)


def produce(scope):
    root, p = prepare(scope)
    if scope == "whole":
        # Never let a mask producer write through a link into the other engineer's workspace.
        assert not (root / "masks").is_symlink()
        (root / "masks").mkdir(exist_ok=True)
        subprocess.run(
            [sys.executable, str(RS / "pipeline/masks/run_masks.py"), "--cams", "all"],
            env={**os.environ, "RS": str(root)},
            check=True,
        )
    for c in CAMS:
        assert all(
            (root / "masks/sky" / c / f"{e['n']:06d}.png").exists() for e in p[c]
        ), (
            f"{c}: missing masks for requested cadence, generate masks in isolated workspace first"
        )
        subprocess.run(
            [
                sys.executable,
                str(RS / "pipeline/depth/run_depth.py"),
                "--cams",
                c,
                "--trajectory",
                str(RS / "poses/ego_causal.npz"),
                "--scale-factor",
                "1",
            ],
            env={**os.environ, "RS": str(root)},
            check=True,
        )
        complete(root, c, p[c])
        (root / ("complete-" + c)).write_text(str(len(p[c])))


if __name__ == "__main__":
    produce(sys.argv[1] if len(sys.argv) > 1 else "slice")
