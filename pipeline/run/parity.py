"""Unattended v5 extension path, using only this window's poses, masks and depth."""

from pathlib import Path
import shutil
from stages import mod, read, write


def fit(chunk, a, b, env):
    work = chunk.resolve() / "v5"
    meta = chunk.resolve() / "metadata"
    meta.mkdir()
    for f in ("drive.json", "rig.json"):
        shutil.copy2(Path("scene/road-stitched-v2") / f, meta / f)
    shutil.copy2("scene/live-v4/sky.jpg", meta / "sky.jpg")
    env = dict(
        env,
        STITCH_V5_WORK=str(work),
        STITCH_V5_META=str(meta),
        STITCH_V5_RANGE_MASKS=str(Path("work/run/static-masks/dyn").resolve()),
    )
    mod("stitch.v5masks", env=env)
    mod("stitch.v5world", "seed", env=env)
    mod("stitch.v5world", work / "seed", "native", env=env)
    mod("stitch.v5fit", work / "native", 1600, env=env)
    mod("stitch.v5structures", env=env)
    mod("stitch.v5release", work / "fit", env=env)
    src = work / "final"
    if any(q.get("verified_texels", 0) for q in read(work / "planes.json")):
        mod("stitch.v5wallguard", src, work / "guarded", env=env)
        src = work / "guarded"
    mod("stitch.v5visibility", src, chunk / "fit", env=env)
    write(
        chunk / "recipe.json",
        dict(
            recipe="v5-native-auto",
            capacity=read(work / "native.json"),
            fit=read(work / "fit.json"),
            installed=read(work / "installed.json"),
            visibility=read(work / "visibility.json"),
        ),
    )
