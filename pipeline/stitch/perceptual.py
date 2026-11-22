"""LPIPS on a fixed, wholly valid crop of each native held-out photo, selected without renders."""

from .evaluate import *
import torch, lpips


def run():
    net = lpips.LPIPS(net="squeeze", verbose=False).to("mps").eval()
    es = held()
    out = []
    batch = []
    keys = []
    tic = time.monotonic()

    def flush():
        if not batch:
            return
        imgs = (
            torch.from_numpy(np.array(batch)).to("mps").permute(0, 1, 4, 2, 3).float()
            / 127.5
            - 1
        )
        with torch.no_grad():
            d = [net(imgs[:, 0], imgs[:, i]).flatten().cpu().numpy() for i in (1, 2)]
        out.extend(
            dict(**e, before=float(d[0][j]), after=float(d[1][j]))
            for j, e in enumerate(keys)
        )
        batch.clear()
        keys.clear()

    for e in es:
        im, valid = reference(e)
        sz = 96
        inte = cv.integral(valid.astype("B"))
        s = inte[sz:, sz:] - inte[:-sz, sz:] - inte[sz:, :-sz] + inte[:-sz, :-sz]
        yy, xx = np.where(s == sz**2)
        if not len(yy):
            continue
        # Pick geometrically central eligible square. Never select from rendered quality.
        j = np.argmin(
            (xx + sz / 2 - DIMS[0] / 2) ** 2 + (yy + sz / 2 - DIMS[1] / 2) ** 2
        )
        y, x = yy[j], xx[j]
        images = [im]
        for st in ["before", "after"]:
            images.append(
                readjpg(REPORT / st / "camera" / e["c"] / ("%06d.jpg" % e["n"]))
            )
        batch.append([a[y : y + sz, x : x + sz] for a in images])
        keys.append(e)
        if len(keys) == 16:
            flush()
    flush()
    savej(REPORT / "lpips-rows.json", out)
    summary = {
        c: {
            k: dict(
                n=len(
                    q := [e for e in out if e["c"] == c and (k == "all" or e["clean"])]
                ),
                before=float(np.mean([e["before"] for e in q])),
                after=float(np.mean([e["after"] for e in q])),
            )
            for k in ["all", "clean"]
        }
        for c in CAMS
    }
    savej(REPORT / "lpips.json", summary)
    print(summary, "seconds", time.monotonic() - tic)


if __name__ == "__main__":
    run()
