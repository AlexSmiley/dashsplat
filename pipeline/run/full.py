"""Fresh-drive full tier: calibrated depth, per-window v2 geometry + joint v4 fit."""

import json
import math
import os
import shutil
import sys
from pathlib import Path

import cv2 as cv
import numpy as np
from scipy.spatial import cKDTree

from stages import (
    HOME,
    PY,
    bake_movers,
    binlink,
    command,
    keyframes,
    masks,
    mod,
    publish,
    read,
    tracking,
    write,
)


def link(src, dst):
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        dst.symlink_to(src.resolve(), target_is_directory=src.is_dir())


def chase(scene, a, b, p):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tool = HOME / "tools/run-chase/chase-track.js"
    if not tool.is_file():
        raise FileNotFoundError(
            f"{tool}: deploy the bundled viewer chase tool with pipeline/run/install.sh"
        )
    with p.open("w") as f:
        command("bun", tool, scene, a, b, 30, 1440, 900, stdout=f)
    frames = read(p)["frames"]
    if len(frames) < 2:
        raise ValueError("too short for a chase camera track")
    return frames


def chunks():
    return [
        Path(f"work/run/chunk-{i:04}")
        for i, _ in enumerate(read("inventory.json")["windows"])
    ]


def params(dst, a, b):
    p = dst.resolve()
    ep = HOME / "envs/stitch/lib/python3.12/site-packages"
    return dict(
        os.environ,
        STITCH_START=str(a),
        STITCH_END=str(b),
        STITCH_V2_WORK=str(p / "v2"),
        STITCH_V4_WORK=str(p / "v4"),
        STITCH_WORK=str(p / "stitch"),
        STITCH_TEACHER=str(p / "teacher/manifest.json"),
        STITCH_MASKS=str(Path("work/run/static-masks").resolve()),
        STITCH_GROUND_SELECTION=str(p / "ground-selection.json"),
        STITCH_DEPTH_MEASUREMENTS=str(
            Path("work/run/depth-measurements.json").resolve()
        ),
        TEACHER_OUT=str(p / "teacher"),
        IBR_HOST=str(Path("scene/road-stitched-v2").resolve()),
        IBR_SOURCE_START=str(max(0, a - 16)),
        IBR_SOURCE_END=str(b + 1),
        PYTHONPATH=os.pathsep.join([str(Path.cwd()), str(ep)]),
    )


def prepare_audit():
    root = Path("maps/object-v3")
    keys = read("depth/keyframes.json")["frames"]
    plan = {}
    for c, rows in keys.items():
        index = read(f"depth/{c}/index.json")
        held = {r["n"] for r in rows[7::8]}
        plan[c] = []
        for i, e in enumerate(rows):
            n = e["n"]
            win = index[str(n)]["window"]
            plan[c].append(
                dict(
                    e,
                    ordinal=i,
                    hold=n in held,
                    window=win,
                    train=n not in held and not held.intersection(win),
                )
            )
    write(root / "plan.json", plan)
    link(Path("depth"), root / "stitched/depth")
    a, b = read("inventory.json")["window"]
    mod(
        "livemap.audit",
        "stitched",
        env=dict(os.environ, AUDIT_START=str(a), AUDIT_END=str(b)),
    )
    doc = read(root / "stitched/audit.json")
    rejected = [c for c in keys if not doc[c]["bands_admitted"]]
    write(
        "work/run/unadmitted-depth.json",
        dict(
            lenses=rejected,
            action="excluded from standing geometry; keep physical imagery",
        ),
    )
    if rejected and (
        not os.getenv("RUN_PROGRESSIVE") or "front" in rejected or len(rejected) > 2
    ):
        raise ValueError(
            "insufficient drive-local metric depth evidence for one or more cameras"
        )
    write(
        "work/run/depth-measurements.json",
        {
            "depth": {
                "stitched": {
                    c: dict(outer_rim_frac=v["rim"], rim_start_deg=v["rim_start"])
                    for c, v in doc.items()
                }
            }
        },
    )
    # Cap extraction to audited ranges/angles before giving V2 the depth. Production outputs remain immutable.
    for kind in ["sky", "road"]:
        link(Path("masks") / kind, Path("work/run/static-masks") / kind)
    from pipeline.livemap.geometry import lattice

    for c, rows in keys.items():
        ray, _ = lattice(c)
        angle = np.degrees(np.arccos(np.clip(ray[:, :, 2], -1, 1)))
        pol = read(root / "stitched" / f"{c}-policy.json")
        for e in rows:
            n = e["n"]
            fn = f"{n:06d}.png"
            raw = cv.imread(f"masks/dyn/{c}/{fn}", -1)
            remove = cv.imread(f"maps/movers-v1/remove/{c}/{fn}", 0)
            if raw is None or remove is None:
                raise ValueError(f"missing static/removal evidence at {c} {n}")
            mask = (raw > 0) | (
                cv.resize(
                    remove, (raw.shape[1], raw.shape[0]), interpolation=cv.INTER_AREA
                )
                > 0
            )
            # Pixels with unaudited depth must not seed static geometry. Preserve sky/road labels for teacher support.
            v = pol[str(n)]
            z = np.load(f"depth/{c}/{n:06d}.npy").astype(float)
            rr = z / np.maximum(ray[:, :, 2], 0.04)
            rr /= np.maximum(0.55, v["a"] + v["b"] * rr)
            good = np.zeros_like(mask)
            for lo, hi in v["bands_admitted"]:
                good |= (rr >= lo) & (rr < hi)
            ro = cv.imread(f"masks/road/{c}/{fn}", 0)
            sk = cv.imread(f"masks/sky/{c}/{fn}", 0)
            mask |= (~good | (angle > v["max_angle"])) & (ro == 0) & (sk == 0)
            if c in rejected:
                mask[:] = True
            dst = Path("work/run/static-masks/dyn") / c / fn
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not cv.imwrite(str(dst), mask.astype("uint16")):
                raise IOError(dst)


def teacher():
    from pipeline.common.rig import Trajectory
    from pipeline.livemap.v4_sky import sky

    prepare_audit()
    tr = Trajectory()
    src = Path("scene/road-stitched-v2")
    gm = read(src / "map.json")
    windows = read("inventory.json")["windows"]
    reports = []
    p = Path("scene/live-v4/sky.jpg")
    p.parent.mkdir(parents=True, exist_ok=True)
    stat = sky(read("depth/keyframes.json")["frames"], Path.cwd(), p)
    if not stat["frames"]:
        raise ValueError("no photographic sky observations")
    binlink(HOME / "work/ibr/m2/m2_surface.dylib", "work/ibr/m2/m2_surface.dylib")
    for dst, (a, b) in zip(chunks(), windows):
        dst.mkdir(parents=True, exist_ok=True)
        env = params(dst, a, b)
        route = tr.T_world_ego(np.linspace(a, b, max(2, math.ceil((b - a) * 4))))[
            :, :2, 3
        ]
        selected = []
        for tile in gm["tiles"]:
            lo, hi = np.array(tile["bbox"]).reshape(2, 3)
            diff = np.maximum(0, np.maximum(lo[:2] - route, route - hi[:2]))
            if np.linalg.norm(diff, axis=1).min() < 85:
                selected.append(tile)
        write(dst / "ground-selection.json", {"tiles": selected})
        command(
            PY, "-c", "from pipeline.stitch.v2data import schedule; schedule()", env=env
        )
        ff = read(dst / "v2/exposures.json")["front"]
        valid = [r for r in ff if r["depth"]]
        if len(valid) < 2:
            raise ValueError(
                f"chunk {a}-{b}: fewer than two independent front depth exposures"
            )
        # Whole-second boundaries: the chase tool emits inclusive 30 Hz endpoints for SLERP.
        move = read("inventory.json")["moving"]
        ca, cb = max(
            ((max(a, x), min(b, y)) for x, y in move if x < b and y > a),
            key=lambda q: q[1] - q[0],
        )
        begin = float(
            math.ceil(
                max(ca, min(e["t"] for e in valid if ca <= e["t_key"] <= cb) + 0.65)
            )
        )
        end = float(math.floor(cb))
        if end <= begin:
            raise ValueError(f"chunk {a}-{b}: insufficient supported chase duration")
        chase(src, begin, end, f"work/ibr/task-m/chase-{begin}-{end}.json")
        env.update(TEACHER_START=str(begin), TEACHER_END=str(end))
        mod("ibr.teacher", "--workers", 3, env=env)
        es = read(dst / "teacher/manifest.json")
        if not any(e["station"] % 10 and e["weighted_fraction"] > 0 for e in es):
            raise ValueError(f"no usable training teacher observations: {dst}")
        reports.append(
            dict(window=[a, b], path=str(dst / "teacher/manifest.json"), count=len(es))
        )
    write("work/run/teacher.json", reports)


def merge():
    from collections import defaultdict

    from pipeline.common.rig import Trajectory
    from pipeline.road.scene import binary_header, reader

    dst = Path("scene/optimized")
    (dst / "tiles").mkdir(parents=True, exist_ok=True)
    route = Trajectory()
    banks = defaultdict(list)
    trees = []
    inv = read("inventory.json")
    windows = inv["windows"]
    for i, (p, (a, b)) in enumerate(zip(chunks(), windows)):
        src = p / "fit"
        tt = np.linspace(a, b, max(2, math.ceil((b - a) * 5)))
        trees.append(cKDTree(route.T_world_ego(tt)[:, :2, 3]))
        for tile in read(src / "map.json")["tiles"]:
            banks[tile["id"]].append((i, src, tile))
    tiles = []  # Stable tie-break: lowest chunk index. Owner is computed on the same XY in all competing tiles.
    for id, entries in banks.items():
        parts = []
        for i, src, e in entries:
            data = reader(src / e["file"])
            owners = [j for j, _, _ in entries]
            if len(entries) > 1:
                dd = np.stack([trees[j].query(data[:, :2])[0] for j in owners])
                data = data[np.argmin(dd, axis=0) == owners.index(i)]
            if len(data):
                parts.append(data)
        if not parts:
            continue
        data = np.concatenate(parts)
        e = entries[0][2]
        p = dst / e["file"]
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as f:
            f.write(binary_header(len(data)))
            data.astype("<f4", copy=False).tofile(f)
        tiles.append(
            dict(
                e,
                count=len(data),
                bbox=np.r_[data[:, :3].min(0), data[:, :3].max(0)].tolist(),
                t_min=float(data[:, 17].min()),
            )
        )
    src = Path("scene/road-stitched-v2")
    for f in ("drive.json", "rig.json"):
        shutil.copy2(src / f, dst / f)
    shutil.copy2("scene/live-v4/sky.jpg", dst / "sky.jpg")
    write(
        dst / "map.json",
        dict(read(src / "map.json"), tiles=tiles, sky="sky.jpg", windows=windows),
    )
    if not any(e["layer"] == "objects" for e in tiles):
        raise ValueError("optimized merge produced no static object splats")


def run(stage):
    base = Path.cwd()
    root = base / "full"
    root.mkdir(exist_ok=True)
    for n in [
        "data",
        "calib",
        "poses",
        "photo",
        "pipeline",
        "third_party",
        "envs",
        "inventory.json",
        "work/poses",
        "work/road",
        "maps/ground-v2",
        "scene/road-live-v2",
        "scene/road-stitched-v2",
    ]:
        link(base / n, root / n)
    os.chdir(root)
    os.environ.update(RS=str(root), PYTHONPATH=str(root))
    sys.path.insert(0, str(root))
    if stage == "production":
        keyframes(True)
        masks()
        command(
            PY,
            "pipeline/depth/run_depth.py",
            "--cams",
            "all",
            "--model",
            "da3-large-1.1",
        )
        command(PY, "pipeline/depth/verify_outputs.py")
    elif stage == "removal":
        tracking()
        mod("movers.landmarks")
        # Nearest physical shutters can straddle a front-clock boundary. Include
        # all lens keyframes actually selected for each moving interval.
        keys = [
            e for rows in read("depth/keyframes.json")["frames"].values() for e in rows
        ]
        intervals = []
        for a, b in read("inventory.json")["moving"]:
            lens_times = [e["t"] for e in keys if a <= e["t_key"] <= b]
            if lens_times:
                intervals.append(
                    [
                        min(a, min(lens_times)),
                        math.nextafter(max(b, max(lens_times)), math.inf),
                    ]
                )
        mod(
            "movers.removal", env=dict(os.environ, MOVERS_WINDOWS=json.dumps(intervals))
        )
    elif stage == "teacher":
        teacher()
    elif stage == "optimize":
        for dst, (a, b) in zip(chunks(), read("inventory.json")["windows"]):
            env = params(dst, a, b)
            if os.environ.get("RUN_PROGRESSIVE"):
                from parity import fit

                fit(dst, a, b, env)
                continue
            mod("stitch.v2targets", env=env)
            mod("stitch.v2seed", env=env)
            mod("stitch.v4fit", dst / "v2/dense-init", dst / "fit", 1600, env=env)
        merge()
    elif stage == "attach":
        bake_movers()
        out = Path("scene/optimized")
        shutil.copy2("work/run/objects.json", out / "objects.json")
        shutil.copytree("work/run/objects", out / "objects", dirs_exist_ok=True)
        shutil.copytree(base / "work/run/atlas", out / "ground", dirs_exist_ok=True)
    elif stage == "publish-full":
        publish(
            Path("scene/optimized"),
            Path("scene/optimized"),
            base / "work/run/atlas",
            "full",
        )
    else:
        raise ValueError(stage)
