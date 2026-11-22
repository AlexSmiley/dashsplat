"""Advance one immutable window or video chunk at a time under the parent drive lease."""

import os
import signal
import shlex
import subprocess
import time
from pathlib import Path
from contextlib import nullcontext
from transactions import (
    atomic_json,
    begin,
    initialize,
    identity,
    lock,
    content_signature,
    promote,
)
from drive import (
    FULL,
    PREVIEW,
    HERE,
    HOME,
    heavy,
    stamp,
    validate,
    read,
    digest,
    wait_worker,
    sources,
    exclusions,
)
from clips import schedule


def link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        dst.symlink_to(src.resolve(strict=True))


def support(inv, index):
    a, b = inv["windows"][index]
    spans = [[a, b]]
    # A short initial crawl cannot calibrate six depth bands in isolation.
    # After a stop the car resumes in the same place: borrow moving observations
    # on that side of the stop. No parked frame is selected or counted as coverage.
    if b - a < 10 and index + 1 < len(inv["windows"]):
        c, d = inv["windows"][index + 1]
        if c > b:
            spans.append([c, min(d, c + 18)])
    return spans


def advance(base, root, env, m, manifest, lease, args, select):
    root = root.resolve()
    inv = read(root / "inventory.json")
    wins = inv["windows"]
    clips = schedule(wins, select)
    py = HOME / ".venv/bin/python"
    build = (
        base
        / "progress"
        / digest(
            [m["stages"]["publish"]["key"], m["stages"]["publish"]["content_sha256"]]
        )[:16]
    )
    build.mkdir(parents=True, exist_ok=True)
    pub = base / "progress-public"
    pub.mkdir(exist_ok=True)
    link(root / "inventory.json", pub / "inventory.json") if not (
        pub / "inventory.json"
    ).is_symlink() else None
    records = m.setdefault("units", {})
    windows = []
    videos = []
    previous = None
    own = identity()
    plan = [dict(start=a, end=b) for a, b in wins]
    total = sum(b - a for a, b in inv["moving"])
    m["clip_selection"] = select
    m["plan"] = dict(windows=plan, clips=clips, moving_seconds=total)
    atomic_json(manifest, m)

    def progress():
        intervals = [
            [max(a, wins[i - 1][1]) if i else a, b]
            for i, (a, b) in enumerate(wins[: len(windows)])
        ]
        m["progress"] = dict(
            moving_seconds=total,
            preview_seconds=total,
            full_seconds=sum(b - a for a, b in intervals),
            full_intervals=intervals,
            clip_seconds=sum(x["entry"]["t1"] - x["entry"]["t0"] for x in videos),
            clips=len(videos),
            full_windows=len(windows),
            windows=len(wins),
        )
        atomic_json(manifest, m)

    def version(k):
        deps = [HERE / "progressive.py"] + (
            [HERE / "parity.py"]
            if k == "optimize"
            else [HERE / "clips.py", HERE / "growing.py"]
            if k in ("clip", "attach", "publish-full")
            else []
        )
        return digest([sources(k), [content_signature(x) for x in deps]])

    def task(label, k, inp, cwd, cmd, files, gen=None, envadd=None):
        # Once completed, pin the measured producer version for that unit. New
        # versions affect future units. --from explicitly invalidates prior work.
        key = digest([str(build), label, inp])
        old = records.get(label, {})
        forced = (
            args.start in PREVIEW
            or args.start in FULL
            and FULL.index(k) >= FULL.index(args.start)
        )
        if (
            old.get("status") in ("done", "prepared")
            and old.get("key") == key
            and not forced
        ):
            place = Path(old["generation"])
            stored = old["hashes"]
            if all(
                (place / f).exists() and content_signature(place / f) == h
                for f, h in stored.items()
            ):
                if old["status"] == "prepared":
                    if gen:
                        promote(gen, place)
                    old.update(status="done", recovered=stamp())
                    atomic_json(manifest, m)
                print(label, "cached", flush=True)
                return place, key
        if args.start in FULL and FULL.index(k) < FULL.index(args.start):
            raise ValueError(f"{label}: prerequisite missing or changed")
        if args.only in FULL and FULL.index(k) < FULL.index(args.only):
            raise ValueError(f"{label}: prerequisite missing or changed")
        log = base / "logs" / f"{time.time_ns()}-{label}.log"
        r = dict(
            status="waiting",
            key=key,
            parameters=inp,
            queued=stamp(),
            owner=own,
            command=list(map(str, cmd)),
            log=str(log),
            code_version=version(k),
        )
        records[label] = r
        atomic_json(manifest, m)
        machine = (
            lock(
                HOME / "work/locks/reconstruction.lock",
                True,
                own,
                drive=args.name,
                stage=label,
            )
            if k != "publish-full"
            else nullcontext()
        )
        start = None
        try:
            with machine as gpu:
                while gpu and (busy := heavy()):
                    r["blockers"] = busy
                    atomic_json(manifest, m)
                    time.sleep(10)
                start = time.monotonic()
                place = begin(gen, k, exclusions(gen, k)) if gen else cwd
                place.mkdir(exist_ok=True, parents=True)
                e = dict(
                    env,
                    RS=str(place),
                    RUN_ROOT=str(place),
                    PYTHONPATH=os.pathsep.join([str(place), str(root)]),
                )
                e.update(envadd or {})
                r.update(
                    status="running",
                    start=stamp(),
                    generation=str(place),
                    environment=envadd or {},
                )
                r.pop("blockers", None)
                atomic_json(manifest, m)
                with log.open("w") as f:
                    print("$", shlex.join(list(map(str, cmd))), file=f, flush=True)
                    child = subprocess.Popen(
                        cmd,
                        cwd=place,
                        env=e,
                        start_new_session=True,
                        pass_fds=tuple(x.fileno() for x in (lease, gpu) if x),
                        stdout=f,
                        stderr=subprocess.STDOUT,
                    )
                    try:
                        rc = wait_worker(child, bool(gpu))
                    except BaseException:
                        os.killpg(child.pid, signal.SIGCONT)
                        os.killpg(child.pid, signal.SIGTERM)
                        child.wait()
                        raise
                    if rc:
                        raise RuntimeError(f"{label}: exit {rc}; {log}")
                if k == "clip":
                    n = read(place / "frames.json")["frames"][-1]["n"]
                    files = (
                        list(files)
                        + ["frames.json"]
                        + [
                            f"{sub}/{n:04d}.{ext}"
                            for sub, ext in [
                                ("render", "npz"),
                                ("render", "png"),
                                ("repaired", "png"),
                                ("repaired-prov", "png"),
                            ]
                        ]
                    )
                hashes = (
                    validate(place, k)[1]
                    if gen
                    else {str(f): content_signature(place / f) for f in files}
                )
                r["commands_run"] = [
                    line[2:]
                    for line in log.read_text(errors="replace").splitlines()
                    if line.startswith("$ ")
                ]
                r.update(
                    status="prepared",
                    hashes=hashes,
                    finish=stamp(),
                    duration_s=round(time.monotonic() - start, 2),
                )
                atomic_json(manifest, m)
                if gen:
                    promote(gen, place)
                r["status"] = "done"
                atomic_json(manifest, m)
                return place, key
        except BaseException as e:
            r.update(
                status="failed",
                detail=str(e),
                finish=stamp(),
                duration_s=round(time.monotonic() - start, 2) if start else 0,
            )
            atomic_json(manifest, m)
            raise

    final_release = None

    def release(label, scene):
        nonlocal final_release
        key = [str(scene), [v["hash"] for v in videos]]
        wd = build / "updates" / digest(key)[:16]
        wd.mkdir(parents=True, exist_ok=True)
        request = dict(scene=str(scene), clips=videos, root=str(root), public=str(pub))
        atomic_json(wd / "request.json", request)
        place, _ = task(
            label,
            "publish-full",
            key,
            wd,
            [py, HERE / "growing.py", "publish"],
            ["publication.json"],
        )
        receipt = read(place / "publication.json")
        final_release = receipt
        m["viewer"] = receipt["url"]
        atomic_json(manifest, m)
        return receipt

    if not m.get("progress"):
        progress()
    for i, (a, b) in enumerate(wins):
        workspace = build / f"w{i:03}"
        workspace.mkdir(exist_ok=True)
        gen = workspace / "active"
        initialize(gen)
        initial = workspace / ".generations/active-initial"
        spans = support(inv, i)
        wid = [spans[0][0], spans[-1][1]]
        if not (initial / "inventory.json").exists():
            for f in [
                "data",
                "calib",
                "poses",
                "photo",
                "pipeline",
                "third_party",
                "envs",
                "work/poses",
                "work/road",
                "work/run/atlas",
                "maps/ground-v2",
                "scene/road-live-v2",
                "scene/road-stitched-v2",
            ]:
                link(root / f, initial / f)
            atomic_json(
                initial / "inventory.json",
                dict(inv, windows=[wid], moving=spans, window=wid, coverage=[a, b]),
            )
        up = str(build)
        last = None
        for k in FULL[:5]:
            if args.only in FULL and FULL.index(k) > FULL.index(args.only):
                break
            last, up = task(
                f"w{i:03}-{k}",
                k,
                [up, a, b],
                workspace,
                [py, HERE / "stages.py", k],
                None,
                gen,
                {"RUN_PROGRESSIVE": "1"},
            )
            up = digest([up, records[f"w{i:03}-{k}"]["hashes"]])
        if args.only in FULL[:4]:
            continue
        windows.append(
            dict(path=str(last / "full/scene/optimized"), window=[a, b], receipt=up)
        )
        bucket = workspace / "combined" / digest(windows)[:16]
        bucket.mkdir(parents=True, exist_ok=True)
        atomic_json(bucket / "request.json", dict(windows=windows, root=str(root)))
        combined, _ = task(
            f"w{i:03}-merge",
            "attach",
            windows,
            bucket,
            [py, HERE / "growing.py", "merge"],
            ["scene"],
        )
        scene = combined / "scene"
        release(f"w{i:03}-publish", scene)
        progress()
        if args.only == "attach":
            continue
        for j, s in enumerate(clips):
            if s["window"] != i:
                continue
            a0, b0 = s["start"], s["end"]
            carry = (
                previous
                if previous and abs(previous["entry"]["t1"] - a0) < 0.00011
                else None
            )
            inp = [up, s, carry["hash"] if carry else None]
            cd = workspace / "clips" / digest(inp)[:16]
            cd.mkdir(parents=True, exist_ok=True)
            # Private per-input partial frame cache: reusable only with identical
            # producer and source hashes. A new producer gets new scratch.
            lab = f"w{i:03}-clip-{round(a0 * 10000)}"
            prior = records.get(lab, {})
            producer = (
                prior.get("code_version")
                if prior.get("status") == "done"
                else version("clip")
            )
            out = cd / producer[:16]
            out.mkdir(exist_ok=True)
            cmd = [
                py,
                HERE / "growing.py",
                "clip",
                str(scene),
                str(a0),
                str(b0),
                str(out),
                carry["scratch"] if carry else "-",
            ]
            files = ["clip.json", "bundle"]
            e = dict(RS=str(last / "full"), PYTHONPATH=str(last / "full"))
            w, _ = task(lab, "clip", inp, out, cmd, files, envadd=e)
            r = read(w / "clip.json")
            v = dict(
                entry=r["entry"], scratch=str(w), hash=records[lab]["hashes"]["bundle"]
            )
            videos.append(v)
            previous = v
            release(lab + "-publish", scene)
            progress()
    if final_release and (HOME / "scene" / args.name).resolve() != Path(
        final_release["directory"]
    ):
        # A changed --clip selection can end at an already verified release.
        # Validate it again and restore that target rather than serving stale videos.
        dst = HOME / "scene" / args.name
        rdir = Path(final_release["directory"])
        if (
            rdir.parent != base / "releases"
            or content_signature(rdir) != final_release["content_sha256"]
        ):
            raise ValueError("stale final release")
        if not dst.is_symlink() or dst.resolve().parent != rdir.parent:
            raise ValueError("unrelated public destination")
        tmp = dst.with_name("." + args.name + "-next")
        tmp.unlink(missing_ok=True)
        tmp.symlink_to(rdir)
        tmp.replace(dst)
        atomic_json(pub / "work/run/published-full.json", final_release)
    progress()
    print("coverage", m["progress"], flush=True)
