"""Prune unsupported traffic depth sheets using independent row-timed physical visibility before merging."""

from .v5world import *


def clean(src, dst):
    rec = [t for t in J(src / "map.json")["tiles"] if t["layer"] == "objects"]
    A = [reader(src / t["file"]) for t in rec]
    S = [np.zeros((len(a), 12), "f") for a in A]
    for s in S:
        s[:, 5] = 1e8
        s[:, 6] = -1e8
    ctr = np.array([a[:, :3].mean(0) for a in A])
    rad = np.array([np.linalg.norm(a[:, :3] - q, axis=1).max() for a, q in zip(A, ctr)])
    lib = W5 / "check.dylib"
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
    with ThreadPoolExecutor(6) as ex:
        for c in CAMS:
            C = rig[c]
            K = np.ascontiguousarray(
                C.params * np.r_[[724 / C.width, 469 / C.height] * 2, [1] * 4], "d"
            )
            num = 0
            for e in frames(c, 1.65 if c == "front" else 2.35):
                T = poses(c, e["t"])
                dp = depth(e)
                m = labels(c, e["n"])
                delta = ctr - T[4, :3, 3]
                ids = np.flatnonzero(
                    (np.linalg.norm(delta, axis=1) < 145 + rad)
                    & (delta @ T[4, :3, 2] > -rad)
                )

                def one(i):
                    fn(
                        A[i].ctypes.data,
                        len(A[i]),
                        S[i].ctypes.data,
                        T.ctypes.data,
                        K.ctypes.data,
                        m.ctypes.data,
                        dp.ctypes.data,
                        e["odo"],
                    )

                list(ex.map(one, ids))
                num += 1
            print("verify", c, num, flush=True)
    parts = []
    cnt = np.zeros(4, int)
    vol = Volume()
    for t, a, s in zip(rec, A, S):
        sup = s[:, 3]
        bad = s[:, 0] * 2.2 + s[:, 1] * 1.6 + s[:, 2] * 1.35
        f = sup / (0.35 + sup + bad)
        span = s[:, 6] - s[:, 5]
        g, h, own = vol.lookup(a[:, :3])
        lat = rt.query(a[:, :2])[0]
        park = (s[:, 8] > 0.35 * sup) & (sup > 1.4)
        verified = t["id"].startswith(("columns_", "facade_"))
        keep = (sup > 1.10) & (span > 1.45) & (f > 0.48)
        keep |= park & (f > 0.40) & (span > 1.2)
        keep |= verified & (bad < 2 + 2 * sup)
        keep &= ~((s[:, 1] > 1.5 * sup + 1) & (own != 0) & (a[:, 2] < g + 4))
        keep &= ~(
            (lat < 5)
            & (a[:, 2] > g + 0.34)
            & (a[:, 2] < g + 4.6)
            & (s[:, 1] > 0.4 * sup)
            & ~park
        )
        keep &= ~((s[:, 0] > sup + 2) & (s[:, 9] < 0.35 * sup))
        alpha = 1 / (1 + np.exp(-a[:, 9]))
        alpha = np.clip(alpha * np.clip(0.36 + 0.79 * f, 0.42, 1), 0.01, 0.98)
        a = a.copy()
        a[:, 9] = np.log(alpha / (1 - alpha))
        parts.append((t, a[keep]))
        cnt += [len(a), keep.sum(), (keep & park).sum(), (keep & verified).sum()]
    assemble(dst, src, parts)
    savej(
        W5 / "visibility.json",
        dict(
            input=int(cnt[0]),
            retained=int(cnt[1]),
            vehicle_class_supported=int(cnt[2]),
            verified_structure_survivors=int(cnt[3]),
        ),
    )
    print(cnt, flush=True)


if __name__ == "__main__":
    clean(Path(sys.argv[1]), Path(sys.argv[2]))
