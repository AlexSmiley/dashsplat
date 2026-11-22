"""Native-shutter, native-resolution moving-object mattes for delayed photographs.

Production input only. Hindsight on a SHORT continuous camera-local token is
appropriate for removing past pixels, but is never sent back to live discovery.
"""

from collections import defaultdict, Counter
from functools import lru_cache
from pathlib import Path
import json, os, pickle, bisect, cv2, numpy as np
from PIL import Image
from pipeline.common.rig import load_rig, frame_table
from .demo import R, W
from .observations import read
from .semantics import release as category
from .motion import release as motion

ROOT = Path(os.environ.get("RS", ".")) / "maps/movers-v1"
GRID = (469, 724)
cv2.setNumThreads(1)


def policy():
    if not R.name.startswith("photo-full"):
        raise ValueError("removal requires non-held production input")
    votes = pickle.load(open(R / "footprint.pickle", "rb"))
    hs = pickle.load(open(R / "identities.pickle", "rb"))
    sem = pickle.load(open(R / "semantics.pickle", "rb"))
    flow = pickle.load(open(R / "motion.pickle", "rb"))
    keep = defaultdict(set)
    counts = Counter()
    records = []
    extra = defaultdict(list)
    if (R / "landmarks.pickle").is_file():
        for e in pickle.load(open(R / "landmarks.pickle", "rb")):
            extra[e["key"]].append(e)
    support = defaultdict(list)
    for key, events in extra.items():
        support[key] = [
            e for e in events if e["state"] == "moving" and e["confidence"] >= 0.75
        ]
    active = {}
    track = {}
    source = R / "online.pickle"
    if source.exists():
        for sp, rr, tr, sz, st, scan in pickle.load(open(source, "rb")):
            active[sp["id"]] = (tr["t"][0], tr["t"][-1])
            track[sp["id"]] = (sp, tr, st)
    for h in hs:
        uid = h["id"]
        scores = sem.get(uid, [])
        cat = category(scores, h["cls"] or "car")
        flow_at = motion(flow.get(uid, []))
        sign = cat is None and any(
            dict(items).get("sign", 0) > 0.25 for t, c, items in scores
        )
        group = defaultdict(list)
        for v in votes[uid]:
            group[v["c"], v["root"], v["id"]].append(v)
        witnessed = []
        projectable = {}
        if uid in track:
            from .online import box
            from .model import interp
            from .packets import iou

            sp, tr, st = track[uid]
            for r in h["rows"]:
                pose = interp(tr, r["rt"])
                cls = next(
                    (
                        cl
                        for at, cl in reversed(st["class_epochs"])
                        if at <= r["release"]
                    ),
                    sp["class"],
                )
                ok = (
                    active[uid][0] <= r["rt"] <= active[uid][1]
                    and iou(box(r, pose[:2], pose[3], cls), r["bb"]) >= 0.38
                )
                if ok:
                    projectable[r["c"], r["n"], r["id"]] = True
                    for e in support[r["root"], r["c"], r["id"]]:
                        if e["lo"] <= r["release"] <= e["hi"]:
                            witnessed.append((r["rt"], r["c"], r["xy"], e["at"]))
        for (c, root, id), rv in group.items():
            good = [
                v for v in rv if v["state"] != "unknown" and v["confidence"] >= 0.64
            ]
            stamps = [v["t"] for v in good]
            for v in rv:
                state = v["state"]
                reason = "footprint"
                ctx = v["context"]
                t = v["t"]
                conf = v["confidence"]
                at = v["release"]
                if state == "unknown" and good:
                    j = bisect.bisect_left(stamps, t)
                    near = min(
                        good[max(0, j - 1) : j + 1], key=lambda x: abs(x["t"] - t)
                    )
                    if abs(near["t"] - t) < 1.3:
                        state = near["state"]
                        conf = near["confidence"]
                        reason = "same-token-neighbour"
                        at = max(at, near["release"])
                remove = state == "moving" or (
                    state == "unknown"
                    and (
                        ctx == "lane-ahead"
                        or (
                            ctx != "roadside"
                            and flow_at is not None
                            and t >= flow_at - 0.35
                            and cat is not None
                        )
                    )
                )
                if (
                    remove
                    and state == "unknown"
                    and ctx != "lane-ahead"
                    and flow_at is not None
                    and cat is not None
                ):
                    at = max(at, flow_at, cat)
                e = [
                    z for z in extra[root, c, id] if z["lo"] <= v["release"] <= z["hi"]
                ]
                if e:
                    z = max(
                        e,
                        key=lambda e: (
                            len(e["cameras"]),
                            e["source"] == "joint-null",
                            e["confidence"],
                        ),
                    )
                    state = z["state"]
                    conf = z["confidence"]
                    reason = z["source"]
                    at = max(at, z["at"])
                    remove = state == "moving" or (
                        state == "unknown" and ctx == "lane-ahead"
                    )
                if (
                    state == "unknown"
                    and reason != "joint-null"
                    and uid in active
                    and active[uid][0] <= t <= active[uid][1]
                    and ctx != "roadside"
                ):
                    remove = True
                    reason = "photo-track"
                    at = max(at, active[uid][0])
                if state == "unknown" and not e and projectable.get((c, v["n"], id)):
                    nearby = [
                        q for q in witnessed if q[1] != c and abs(q[0] - t) < 1.15
                    ]
                    if nearby:
                        near = min(nearby, key=lambda q: abs(q[0] - t))
                        state = "moving"
                        conf = 0.70
                        remove = True
                        reason = "corroborated-track-handover"
                        at = max(at, near[3])
                if sign:
                    remove = False
                    reason = "sign-veto"
                    at = max(at, max(s[0] for s in scores))
                if remove:
                    keep[c, v["n"], root].add(id)
                counts[state, remove] += 1
                records.append(
                    [
                        c,
                        v["n"],
                        id,
                        uid,
                        state,
                        conf,
                        ctx,
                        int(remove),
                        reason,
                        round(float(at), 6),
                        root,
                    ]
                )
    # Preserve the same unknown policy for an isolated released detection. It
    # has no world identity, not an implicit parked classification.
    from .footprint import context

    assigned = {(v[0], v[1], v[2]) for v in records}
    for t, c, n, rows in pickle.load(open(R / "packets.pickle", "rb")):
        for r in rows:
            if (c, n, r["id"]) in assigned:
                continue
            ctx = context(r)
            remove = ctx == "lane-ahead"
            if remove:
                keep[c, n, r["root"]].add(r["id"])
            records.append(
                [
                    c,
                    n,
                    r["id"],
                    None,
                    "unknown",
                    0.5,
                    ctx,
                    int(remove),
                    "unbound-context",
                    round(r["release"], 6),
                    r["root"],
                ]
            )
    (ROOT / "remove").mkdir(parents=True, exist_ok=True)
    dump(
        ROOT / "decisions.json",
        dict(
            fields=[
                "cam",
                "frame",
                "mask_id",
                "track_id",
                "state",
                "confidence",
                "context",
                "remove",
                "evidence",
                "available_at_s",
                "mask_namespace",
            ],
            rows=records,
        ),
    )
    return keep


def dump(p, v):
    q = p.with_suffix(".next")
    q.write_text(json.dumps(v, separators=(",", ":"), allow_nan=False))
    os.replace(q, p)


@lru_cache(16)
def gray(c, n):
    return read(c, n, True)


@lru_cache(16)
def matte(c, n, root, ids):
    if not ids:
        return np.zeros(GRID, np.uint8)
    m = cv2.imread(str(Path(root) / "masks/dyn" / c / f"{n:06}.png"), -1)
    b = np.isin(m, ids).astype("uint8") * 255
    road = Path(root) / "masks/road" / c / f"{n:06}.png"
    gr = cv2.imread(str(road), 0) if road.exists() else None
    if gr is not None:
        g = gray(c, n)
        ref = cv2.blur(g, (19, 15)).astype(float)
        adj = cv2.dilate(b, np.ones((9, 9), np.uint8))
        top = cv2.dilate(b, np.ones((7, 1), np.uint8), anchor=(0, 6))
        shadow = (top > 0) & (adj > 0) & (gr > 0) & (g < 0.74 * ref) & (g < 100)
        b[shadow] = 255
    return b


def warp(c, n, src, mask):
    if not np.any(mask):
        return mask
    ga = gray(c, src)
    gb = gray(c, n)
    out = mask * 0
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        a = (labels == i).astype("uint8") * 255
        pts = cv2.goodFeaturesToTrack(ga, 48, 0.008, 3, mask=a)
        if pts is not None and len(pts) >= 3:
            q, ok, _ = cv2.calcOpticalFlowPyrLK(
                ga, gb, pts, None, winSize=(21, 21), maxLevel=3
            )
            back, valid, _ = cv2.calcOpticalFlowPyrLK(gb, ga, q, None)
            sel = (
                (ok[:, 0] > 0)
                & (valid[:, 0] > 0)
                & (np.linalg.norm(pts[:, 0] - back[:, 0], axis=1) < 1.8)
            )
            if sel.sum() >= 3:
                A, st = cv2.estimateAffinePartial2D(
                    pts[sel], q[sel], method=cv2.RANSAC, ransacReprojThreshold=1.9
                )
                if A is not None and 0.56 < np.linalg.det(A[:, :2]) < 1.72:
                    out |= cv2.warpAffine(a, A, (724, 469), flags=cv2.INTER_NEAREST)
                    continue
        # Translation from a local appearance match, rather than assuming no motion.
        l = max(0, x - 24)
        t = max(0, y - 20)
        r = min(724, x + w + 24)
        b = min(469, y + h + 20)
        template = ga[y : y + h, x : x + w]
        corr = cv2.matchTemplate(gb[t:b, l:r], template, cv2.TM_CCOEFF_NORMED)
        _, v, _, z = cv2.minMaxLoc(corr)
        dx, dy = (l + z[0] - x, t + z[1] - y) if v > 0.48 else (0, 0)
        out |= cv2.warpAffine(
            a, np.float32([[1, 0, dx], [0, 1, dy]]), (724, 469), flags=cv2.INTER_NEAREST
        )
    return out


def write(c, n, cam, b):
    dst = ROOT / "remove" / c
    dst.mkdir(parents=True, exist_ok=True)
    p = dst / f"{n:06}.png"
    q = p.with_suffix(".next")
    q.unlink(missing_ok=True)
    if not b.any():
        blank = ROOT / f"empty-{cam.width}x{cam.height}.png"
        if not blank.exists():
            Image.new("1", (cam.width, cam.height)).save(blank)
        os.link(blank, q)
    else:
        # Two grid pixels = 8 native px front / 4 px surround.
        b = cv2.dilate(b, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        im = (
            Image.fromarray(b)
            .convert("1")
            .resize((cam.width, cam.height), Image.Resampling.NEAREST)
        )
        im.save(q, format="PNG", compress_level=1)
    os.replace(q, p)


def reconcile(manifest):
    # Preserve earlier generated files outside the consumer namespace when a
    # shutter number vanishes or an updated segmentation no longer supports it.
    moved = []
    for c, cam in load_rig().items():
        ts = {f["n"]: f["t"] for f in frame_table(cam)}
        bad = set()
        for window in manifest["windows"].values():
            for lo, hi in window[c]["unsupported_runs"]:
                bad.update(range(lo, hi + 1))
        for src in (ROOT / "remove" / c).glob("[0-9]*.png"):
            n = int(src.stem)
            if n in ts and n not in bad:
                continue
            dst = R / "obsolete-removal" / c / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.replace(src, dst)
            moved.append((c, n))
    return sorted(moved)


def run(only=None):
    keep = policy()
    rig = load_rig()
    anchor_bank = json.load(open(R / "mask-sources.json"))
    manifest = {
        "format": "binary PNG; white=remove, black=keep, native resolution; indices refer to cams/<cam>.json",
        "windows": {},
        "unsupported": "missing PNG means no segmentation support; never assume empty",
    }
    if only and (ROOT / "remove-index.json").exists():
        manifest = json.load(open(ROOT / "remove-index.json"))
    for start, end in json.loads(os.environ["MOVERS_WINDOWS"]) if "MOVERS_WINDOWS" in os.environ else [
        (105, 140),
        (286, 330),
        (190, 215),
        (0, 105),
        (140, 190),
        (215, 286),
        (330, 86400),
    ]:
        if only and (start, end) not in only:
            continue
        result = {}
        for c, cam in rig.items():
            seq = frame_table(cam)
            ts = {f["n"]: f["t"] for f in seq}
            roots = {int(n): root for n, root in anchor_bank[c].items() if int(n) in ts}
            nn = np.array(sorted(roots))
            cnt = Counter()
            gaps = []
            for n in [n for n, t in ts.items() if start <= t < end]:
                n = int(n)
                j = np.searchsorted(nn, n)
                anchors = [
                    int(k)
                    for k in nn[max(0, j - 1) : j + 1]
                    if abs(ts[k] - ts[n])
                    <= (0.80 if start in (105, 190, 286) else 0.34)
                ]
                key = n in roots
                if not anchors:
                    cnt["no-source"] += 1
                    gaps.append(n)
                    continue
                if key:
                    anchors = [n]
                mask = np.zeros(GRID, np.uint8)
                for k in anchors:
                    root = roots[k]
                    a = matte(c, k, root, tuple(sorted(keep.get((c, k, root), ()))))
                    if k != n and a.any() and abs(ts[k] - ts[n]) > 0.17:
                        middle = sorted(
                            (u for u in ts if min(k, n) < u < max(k, n)), reverse=k > n
                        )[3::4]
                        prev = k
                        for u in [*middle, n]:
                            a = warp(c, u, prev, a)
                            prev = u
                        mask |= a
                    else:
                        mask |= a if k == n else warp(c, n, k, a)
                write(c, n, cam, mask)
                cnt["key" if key else "propagated"] += 1
                cnt["nonempty"] += int(mask.any())
            result[c] = dict(
                counts=dict(cnt), unsupported_runs=[[g[0], g[-1]] for g in runs(gaps)]
            )
            gray.cache_clear()
            matte.cache_clear()
        manifest["windows"][f"{start}-{end}"] = result
        print(start, end, {k: v["counts"] for k, v in result.items()}, flush=True)

    import hashlib

    manifest["input_mask_sources_sha256"] = hashlib.sha256(
        (R / "mask-sources.json").read_bytes()
    ).hexdigest()
    manifest["decision_ledger_sha256"] = hashlib.sha256(
        (ROOT / "decisions.json").read_bytes()
    ).hexdigest()
    dump(ROOT / "sources.json", anchor_bank)
    reconcile(manifest)
    dump(ROOT / "remove-index.json", manifest)


def runs(v):
    out = []
    for n in v:
        if not out or out[-1][-1] != n - 1:
            out.append([])
        out[-1].append(n)
    return out


if __name__ == "__main__":
    run()
