"""Persist train-only rectified RGB, real-pixel labels, optical z and confidence."""

from .v2data import *
from concurrent.futures import ThreadPoolExecutor
import time


def prep():
    tic = time.monotonic()
    dst = V / "targets"
    dst.mkdir(exist_ok=True)
    es = [
        e
        for rows in schedule().values()
        for e in rows
        if not e["hold"] and (e["depth"] or e["i"] % 3 == 1)
    ]

    def one(e):
        c = e["c"]
        out = []
        for yaw in designs(c):
            p = dst / c / f"{e['n']:06d}_{yaw}.npz"
            p.parent.mkdir(exist_ok=True)
            im, lab, z, wt = target(c, e["n"], yaw, e["depth"])
            np.savez(
                p,
                rgb=im,
                lab=lab,
                z=z.astype("float16") if z is not None else np.zeros(0, "float16"),
                wt=np.uint8(np.clip(wt * 255, 0, 255))
                if wt is not None
                else np.zeros(0, "B"),
            )
            out.append(
                dict(
                    c=c,
                    n=e["n"],
                    t=e["t"],
                    yaw=yaw,
                    odo=e["odo"],
                    file=str(p),
                    depth=bool(e["depth"]),
                )
            )
        return out

    with ThreadPoolExecutor(3) as pool:
        views = [v for group in pool.map(one, es) for v in group]
        savej(V / "train.json", views)
        savej(
            V / "targets-stats.json",
            dict(seconds=time.monotonic() - tic, physical=len(es), views=len(views)),
        )
        print(len(views), time.monotonic() - tic)


if __name__ == "__main__":
    prep()
