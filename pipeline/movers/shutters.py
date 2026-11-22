from functools import lru_cache
from pipeline.common.rig import frame_table
from .online import camera


@lru_cache(None)
def clock(c):
    return {r["n"]: r["t"] for r in frame_table(camera(c))}


def correct(r):
    c = camera(r["c"])
    t = clock(r["c"])[r["n"]]
    return dict(
        r,
        t=t,
        rt=float(c.row_time(t, r["foot"][1] * c.height / 469)),
        release=max(t, t + c.time_offset_s + abs(c.rs_readout_s) * 0.5),
    )
