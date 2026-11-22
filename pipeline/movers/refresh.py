"""Bounded Studio watcher. Rebuild production only on new complete source pairs."""

from contextlib import closing
import argparse, hashlib, os, select, subprocess, sys, time
from pathlib import Path
from .demo import R, W
from .shutters import clock
from pipeline.common.rig import load_rig


def sources():
    return [
        (rt, c, rt / "masks/dyn" / c)
        for rt in (Path("."), W / "extra")
        for c in load_rig()
        if (rt / "masks/dyn" / c).is_dir()
    ]


def fingerprint(dirs):
    h = hashlib.sha256()
    for root, c, p in dirs:
        ts = clock(c)
        for q in sorted(p.glob("[0-9]*.json")):
            n = int(q.stem)
            mask = q.with_suffix(".png")
            if (
                n not in ts
                or (root != Path(".") and not 190 <= ts[n] <= 215)
                or not mask.is_file()
            ):
                continue
            h.update(f"{q}:{q.stat().st_mtime_ns}:{mask.stat().st_mtime_ns};".encode())
    return h.hexdigest()


def follow(minutes):
    if R.name != "photo-full-native":
        raise ValueError("use production photo-full-native")
    fp = R / "removal-sources.sha256"
    dirs = sources()
    end = time.monotonic() + minutes * 60
    fds = []
    with closing(select.kqueue()) as kq:
        for _, _, p in dirs:
            fd = os.open(p, os.O_RDONLY)
            fds.append(fd)
            kq.control(
                [
                    select.kevent(
                        fd,
                        filter=select.KQ_FILTER_VNODE,
                        flags=select.KQ_EV_ADD | select.KQ_EV_CLEAR,
                        fflags=select.KQ_NOTE_WRITE,
                    )
                ],
                0,
                0,
            )
        try:
            last = fp.read_text().strip() if fp.exists() else fingerprint(dirs)
            fp.write_text(last)
            print(
                "watching",
                len(fds),
                "source directories for",
                minutes,
                "minutes",
                flush=True,
            )
            while time.monotonic() < end:
                current = fingerprint(dirs)
                if current != last:
                    # Re-association can renumber IDs. Refresh the matching scene
                    # and its cache from the same production packet snapshot.
                    for cmd in [
                        "packets --all",
                        "handoff",
                        "footprint",
                        "motion",
                        "semantics",
                        "filter",
                        "landmarks",
                        "removal",
                        "online_scene publish",
                        "skin",
                    ]:
                        mod, *args = cmd.split()
                        subprocess.run(
                            [sys.executable, "-m", "pipeline.movers." + mod, *args],
                            check=True,
                        )
                    subprocess.run(
                        [
                            "bun",
                            "viewer-cache-tool/tools/build-tile-cache.ts",
                            str(Path("scene/live-v5-movers-v4").resolve()),
                        ],
                        check=True,
                    )
                    fp.write_text(current)
                    last = current
                    continue
                if not kq.control([], len(fds), max(0.0, end - time.monotonic())):
                    break
                # Coalesce a burst of newly exported camera PNG/JSON pairs.
                while time.monotonic() < end and kq.control(
                    [], len(fds), min(3.0, max(0.0, end - time.monotonic()))
                ):
                    pass
        finally:
            for fd in fds:
                os.close(fd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes", type=int, required=True)
    follow(parser.parse_args().minutes)
