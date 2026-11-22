"""Prepare newly published base exposures in order. Final global tone/ownership waits for the whole drive."""

from .refine import *
import select, shutil


def main(workers):
    W.mkdir(parents=True, exist_ok=True)
    CS.mkdir(exist_ok=True)
    FS.mkdir(exist_ok=True)
    path = BASE / "observations.json"
    d = os.open(BASE, os.O_RDONLY)
    queue = select.kqueue()
    queue.control(
        [
            select.kevent(
                d,
                filter=select.KQ_FILTER_VNODE,
                flags=select.KQ_EV_ADD | select.KQ_EV_CLEAR,
                fflags=select.KQ_NOTE_WRITE,
            )
        ],
        0,
        0,
    )
    done = set()
    stage = Path("work/groundtex-candidate/state")
    while True:
        obs = read(path)
        rows = [r for r in obs if proj_key(r) not in done and not 85 <= r["t"] <= 140]
        print("new exposure fields", len(rows), flush=True)
        with ThreadPoolExecutor(workers) as ex:
            for j in range(0, len(rows), workers * 3):
                batch = rows[j : j + workers * 3]
                list(ex.map(match, batch))
                done.update(proj_key(r) for r in batch)
                if j % (workers * 36) == 0:
                    print(j, len(rows), "prepared total", len(done), flush=True)
        # Phase five's observations are committed before its manifest. Wait until both are present.
        meta = read(Path("scene/stitched-opt-v4/ground/ground.json"))
        full = any(hi >= 361.4 for lo, hi in meta["intervals"])
        if full and len(obs) == len(read(path)):
            break
        queue.control(None, 1, 20)
    # Reuse the measured first-slice warp, but remap the low-resolution source
    # from the native exposure afresh for the global illumination solve.
    first = [r for r in obs if 85 <= r["t"] <= 140]
    prior = {proj_key(r): r for r in read(stage / "observations.json")}

    def reuse(r):
        k = proj_key(r)
        r.pop("gain", None)
        src = stage / "fields" / (k + ".npz")
        dst = FS / (k + ".npz")
        if not src.exists():
            return match(r)
        shutil.copyfile(src, dst)
        r["field"] = str(dst)
        r["local_matches"] = prior[k].get("local_matches", 0)
        box, rgb, q, sem = coarse(r)
        q[guide_box(box)[:, :, 3] < 0.012] = 0
        np.savez_compressed(
            CS / (k + ".npz"), box=box, rgb=rgb, q=q.astype("f2"), sem=sem
        )

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(reuse, first))
    print("global solve", flush=True)
    run(0, 361.42, workers)


if __name__ == "__main__":
    import sys

    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
