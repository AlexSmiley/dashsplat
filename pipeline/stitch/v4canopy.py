"""Recover off-centre object texture from real cameras. Replace a coarse surfel only if both children resolve different, repeat-observed colours."""

from .v4teacher import *
from .v2posterior import observations


def children(a):
    R = Rotation.from_quat(a[:, [14, 15, 16, 13]]).as_matrix()
    s = np.exp(a[:, 10:12])
    ax = (s[:, 1] > s[:, 0]).astype(int)
    vec = R[np.arange(len(a)), :, ax] * s[np.arange(len(a)), ax, None]
    B = np.repeat(a, 2, axis=0)
    B[::2, :3] -= 0.61 * vec
    B[1::2, :3] += 0.61 * vec
    B[np.arange(len(B)), 10 + np.repeat(ax, 2)] += np.log(0.73)
    alpha = np.clip(1 / (1 + np.exp(-a[:, 9])), 0.001, 0.998)
    alpha = 1 - (1 - alpha) ** (1 / 1.46)
    B[:, 9] = np.repeat(np.log(alpha / (1 - alpha)), 2)
    return B


def evidence(src):
    meta = J(src / "map.json")
    A = np.concatenate(
        [reader(src / t["file"]) for t in meta["tiles"] if t["layer"] == "objects"]
    )
    vol = Volume()
    g, h, _ = vol.lookup(A[:, :3])
    sz = np.exp(A[:, 10:12]).max(1)
    idx = np.flatnonzero(
        (A[:, 2] > g + 0.72)
        & (A[:, 2] > h + 0.35)
        & (sz > 0.115)
        & (sz < 1.6)
        & (A[:, 9] > -0.85)
    )
    B = children(A[idx])
    st, di, n = observations(np.ascontiguousarray(B, "f"))
    srf = st[:, 2]
    wt = st[:, 6]
    mean = st[:, 7:10] / np.maximum(0.001, wt[:, None])
    best = st[:, 11:14]
    scatter = np.sqrt(
        np.maximum(0, st[:, 15] / np.maximum(0.001, wt) - mean.mean(1) ** 2)
    )
    angle = np.unpackbits(di.view("B").reshape(-1, 4), axis=1).sum(1)
    ok = (
        (srf > 1.4)
        & (srf > st[:, 0] * 1.8)
        & (wt > 1.1)
        & (scatter < 21)
        & (angle >= 2)
        & (st[:, 14] < np.repeat(sz[idx], 2) * 0.8)
    )
    ref = 0.42 * mean + 0.58 * np.clip(best, mean - 24, mean + 24)
    physical = mean.reshape(-1, 2, 3)
    detail = abs(physical[:, 0] - physical[:, 1]).max(1)
    keep = ok.reshape(-1, 2).all(1) & (detail > 6) & (detail < 68)
    np.savez(
        OUT / "canopy.npz",
        indices=idx[keep],
        rgb=ref.reshape(-1, 2, 3)[keep],
        original=A[idx[keep], :3],
    )
    print(
        dict(
            proposed=len(idx),
            resolved=int(keep.sum()),
            images=n,
            median_colour_separation=float(np.median(detail[keep])),
        ),
        flush=True,
    )


def install(src, dst):
    E = np.load(OUT / "canopy.npz")
    idx = E["indices"]
    ref = E["rgb"]
    rows = []
    off = 0
    for t in J(src / "map.json")["tiles"]:
        a = reader(src / t["file"])
        if t["layer"] == "ground":
            rows.append(a)
            continue
        inside = (idx >= off) & (idx < off + len(a))
        i = idx[inside] - off
        off += len(a)
        assert np.allclose(a[i, :3], E["original"][inside], atol=0.0001)
        b = children(a[i]) if len(i) else np.empty((0, 24), "f")
        c = (b[:, 6:9] * C0 + 0.5) * 255
        r = ref[inside].reshape(
            -1, 3
        )  # Preserve the learned mean and extract the real intra-surface contrast.
        m = np.repeat(ref[inside].mean(1), 2, axis=0)
        b[:, 6:9] = (
            np.clip(c + 0.30 * (m - c) + 0.95 * (r - m), 0, 255) / 255 - 0.5
        ) / C0
        rows.append(np.concatenate([np.delete(a, i, axis=0), b]))
    # A sky/depth warp loses thin subpixel rims. Give the physically triangulated columns precedence over that mask.
    shaft = cKDTree(np.load(RS / "work/stitch-v3/uprights.npy")[:, :3])
    restored = 0
    reinforced = 0
    baseline = {t["id"]: t for t in J(SRC / "map.json")["tiles"]}
    for n, t in enumerate(J(src / "map.json")["tiles"]):
        if t["layer"] != "objects":
            continue
        ref = reader(SRC / baseline[t["id"]]["file"])
        ref = ref[shaft.query(ref[:, :3], distance_upper_bound=0.0002)[0] < 0.0002]
        if not len(ref):
            continue
        b = rows[n]
        dist, idx = cKDTree(b[:, :3]).query(ref[:, :3], distance_upper_bound=0.035)
        same = dist < 0.035
        if same.any():
            j = np.flatnonzero(same)
            same[j] &= np.max(np.abs(b[idx[j], 10:13] - ref[j, 10:13]), axis=1) < 0.30
        j = idx[same]
        reinforced += int(np.sum(b[j, 9] < ref[same, 9]))
        b[j, 9] = np.maximum(b[j, 9], ref[same, 9])
        rows[n] = np.concatenate([b, ref[~same]])
        restored += int((~same).sum())
    print(
        dict(restored_shaft_elements=restored, reinforced_shaft_elements=reinforced),
        flush=True,
    )
    write_scene(src, dst, rows)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        evidence(Path(sys.argv[1]))
    else:
        install(Path(sys.argv[1]), Path(sys.argv[2]))
