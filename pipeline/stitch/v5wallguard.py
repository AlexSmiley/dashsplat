"""Reject whole generic facade hypotheses crossing a drove-through surface or unsupported by physical ranges."""

from .v5world import *


def guard(src, dst):
    A = np.load(W5 / "walls.npy")
    planes = J(W5 / "planes.json")
    vol = Volume()
    sizes = np.array([q["verified_texels"] for q in planes])
    assert sizes.sum() == len(A)
    beg = np.r_[0, sizes.cumsum()]
    valid = []
    samples = []
    gi = []
    flags = []
    for i, (lo, hi) in enumerate(zip(beg[:-1], beg[1:])):
        B = A[lo:hi]
        P = B[:: max(1, len(B) // 700), :3]
        lat = rt.query(P[:, :2])[0]
        g, h, owned = vol.lookup(P)
        road_conflict = ((lat < 8.5) | (owned.astype(bool) & (lat < 32))) & (
            P[:, 2] < g + 11
        )
        flags.append(bool(road_conflict.sum() > max(2, len(P) * 0.035)))
        samples.append(B[:: max(1, len(B) // 700)])
        gi.extend([i] * len(P))
    S = np.concatenate(samples)
    s = np.zeros((len(S), 12), "f")
    s[:, 5] = 1e8
    s[:, 6] = -1e8
    lib = W5 / "wallguard.dylib"
    subprocess.run(
        [
            "c++",
            "-O3",
            "-std=c++17",
            *(["-shared", "-fPIC"] if sys.platform.startswith("linux") else ["-dynamiclib"]),
            Path(__file__).with_name("v5visibility.cpp"),
            "-o",
            lib,
        ],
        check=True,
    )
    fn = ct.CDLL(str(lib)).verify_world
    P = ct.c_void_p
    fn.argtypes = [P, ct.c_int, P, P, P, P, P, ct.c_float]
    for c in CAMS:
        C = rig[c]
        K = np.ascontiguousarray(
            C.params * np.r_[[724 / C.width, 469 / C.height] * 2, [1] * 4], "d"
        )
        for e in frames(c, 4.5):
            T = poses(c, e["t"])
            rr = depth(e)
            lab = labels(c, e["n"])
            fn(
                S.ctypes.data,
                len(S),
                s.ctypes.data,
                T.ctypes.data,
                K.ctypes.data,
                lab.ctypes.data,
                rr.ctypes.data,
                e["odo"],
            )
    gi = np.array(gi)
    records = []
    keep = []
    for i, (lo, hi) in enumerate(zip(beg[:-1], beg[1:])):
        q = s[gi == i]
        yes = q[:, 3]
        no = q[:, 0] + q[:, 1] + q[:, 2]
        vote = (yes > 1) & (yes > 1.7 * no) & (q[:, 6] - q[:, 5] > 2)
        fraction = float(vote.mean()) if len(q) else 0
        allow = not flags[i] and fraction >= 0.52
        records.append(
            dict(
                id=i,
                road_intersection=flags[i],
                range_supported_fraction=fraction,
                accepted=allow,
            )
        )
        if allow:
            keep.append(A[lo:hi])
    allowed = np.concatenate(keep) if keep else np.empty((0, 24), "f")
    tree = cKDTree(allowed[:, :3])
    parts = []
    removed = 0
    for t in J(src / "map.json")["tiles"]:
        if t["layer"] != "objects":
            continue
        a = reader(src / t["file"])
        if t["id"].startswith("facade_"):
            ok = tree.query(a[:, :3], distance_upper_bound=0.001)[0] < 0.001
            removed += int((~ok).sum())
            a = a[ok]
        parts.append((t, a))
    assemble(dst, src, parts)
    savej(
        W5 / "wallguard.json",
        dict(
            kept_planes=sum(r["accepted"] for r in records),
            rejected_planes=sum(not r["accepted"] for r in records),
            removed=removed,
            checks=records,
        ),
    )
    print("facade guard", sum(r["accepted"] for r in records), removed, flush=True)


if __name__ == "__main__":
    guard(Path(sys.argv[1]), Path(sys.argv[2]))
