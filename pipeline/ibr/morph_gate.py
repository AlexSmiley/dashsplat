"""Offline release gate. Every encoded frame is audited, and a failed candidate never edits the scene manifest."""

from .morph_ghost import *
from collections import defaultdict
import csv


class Original:
    def __call__(self, t, bg, allq, uses, ga, gr, flows):
        cards = [(bg, "floor")]
        weights = []
        total = np.zeros((H, L), "f")
        for q in allq:
            w = q["weight"]
            total += w
            for k, f in [("p", 1 - q["fg"]), ("fp", q["fg"])]:
                cards.append((q, k))
                weights.append(w * f)
        return cv.resize(
            np.argmax(np.stack([np.maximum(0, 1 - total)] + weights), axis=0).astype(
                "B"
            ),
            (OW, OH),
            interpolation=cv.INTER_NEAREST,
        ), cards


def src_id(card):
    q, k = card
    return ("floor", -1, k) if k == "floor" else (q["c"], q["n"], k)


def add_anchor(ev, lab, ids):
    x, y, w, h = ev["worstBox"]
    xx, yy, _, _ = ev.pop("points")
    ok = (xx >= x) & (xx < x + w) & (yy >= y) & (yy < y + h)
    ev.pop("mark")
    j = int(lab[yy[ok][0], xx[ok][0]]) if ok.any() else int(lab[y + h // 2, x + w // 2])
    ev["reference"] = list(ids[j])
    return ev


def block(args):
    a, b, lo, hi, candidate = args
    ts = track(a, b)
    arc, plan = clip_clock(a, b)
    p = OUT / f"{a}-{b}"
    p.mkdir(exist_ok=True)
    base = cv.VideoCapture(str(W / "85-140/chase.mp4"))
    base.set(cv.CAP_PROP_POS_FRAMES, round((ts[lo][0] - 85) * 30))
    owner = Owner() if candidate else Original()
    entries = []
    scratch = []
    caps = []
    if candidate:
        file = p / f"part-{lo:04}.mp4"
        enc = encode(file, OW, OH)
        cf = p / f"control-{lo:04}.mp4"
        ce = encode(cf, OW, OH)
    for i in range(max(0, lo - 8) if candidate else lo, hi):
        t, T = ts[i]
        v = render(t, T, arc[i], plan=plan, resolver=owner)
        if i < lo:
            continue
        if candidate:
            im, meta, lab, patches, ol, oc = v
            ol = cv.resize(ol, (OW, OH), interpolation=cv.INTER_NEAREST)
            enc.stdin.write(im.tobytes())
            f = meta["uses"][0][2]
            q = oc[1 + 2 * int(f > 0.5)]
            solo = full(*q, (0, 0, OW, OH))
            ce.stdin.write(solo.tobytes())
            refs = references(lab, patches)
            ids = {j: ident for j, ident in owner.ids_for_report.items()}
            scratch.append(
                (
                    pack(lab),
                    [(j, r, pack(m)) for j, (r, m) in refs.items()],
                    ids,
                    pack(refmask(solo)),
                )
            )
            meta.pop("uses", None)
        else:
            ol, oc = v
            meta = None
        ok, fm = base.read()
        assert ok
        oldrefs = references(ol, cards=oc)
        old = add_anchor(audit(fm, ol, oldrefs, True), ol, [src_id(q) for q in oc])
        entries.append(dict(t=t, frame=i, published=old, stageMeta=meta))
    base.release()
    if candidate:
        for e in (enc, ce):
            e.stdin.close()
            assert e.wait() == 0
        cap = cv.VideoCapture(str(file))
        solo = cv.VideoCapture(str(cf))
        assert int(cap.get(cv.CAP_PROP_FRAME_COUNT)) == hi - lo
        for en, (lb, packed, ids, m) in zip(entries, scratch):
            ok, im = cap.read()
            assert ok
            lab = unpack(lb)
            refs = {j: (r, unpack(rf)) for j, r, rf in packed}
            en["candidate"] = add_anchor(audit(im, lab, refs, True), lab, ids)
            ok, rgb = solo.read()
            assert ok
            control = audit(
                rgb, np.ones((OH, OW), "B"), {1: ((0, 0, OW, OH), unpack(m))}
            )
            en["encodedSolo"] = control
        cap.release()
        solo.release()
    savej(p / f"audit-{lo:04}.json", entries)
    print("audited", a, lo, hi, flush=True)
    return entries


def run(a, b, candidate):
    ts = track(a, b)
    N = len(ts)
    cuts = list(range(0, N, 45))
    p = OUT / f"{a}-{b}"
    p.mkdir(exist_ok=True)
    args = [(a, b, i, min(i + 45, N), candidate) for i in cuts]
    with ProcessPoolExecutor(
        7 if candidate else 5, mp_context=mp.get_context("spawn")
    ) as pool:
        rows = [r for ls in pool.map(block, args) for r in ls]
    assert len(rows) == N
    savej(p / "frames.json", rows)
    if candidate:
        ls = p / "concat.txt"
        ls.write_text("".join(f"file 'part-{i:04}.mp4'\n" for i in cuts))
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(ls),
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                str(p / "candidate.mp4"),
            ],
            check=True,
        )
        assert int(Jprobe(p / "candidate.mp4")["nb_frames"]) == N
    per = []
    keys = ["published"] + (["candidate", "encodedSolo"] if candidate else [])
    for sec in range(a, b + 1):
        rr = [r for r in rows if int(r["t"]) == sec]
        j = dict(second=sec, frames=len(rr))
        for key in keys:
            best = max(rr, key=lambda r: r[key]["structureFraction"])
            vals = [r[key] for r in rr]
            j[key + "TwinPct"] = (
                100
                * sum(r["twins"] for r in vals)
                / max(1, sum(r["edges"] for r in vals))
            )
            j[key + "StructurePct"] = (
                100
                * sum(r["structureTwins"] for r in vals)
                / max(1, sum(r["structureEdges"] for r in vals))
            )
            j[key + "WorstT"] = best["t"]
            j[key + "WorstStructurePct"] = 100 * best[key]["structureFraction"]
        per.append(j)
    with (p / "per-second.csv").open("w") as f:
        csvw = csv.DictWriter(f, fieldnames=list(per[0]))
        csvw.writeheader()
        csvw.writerows(per)
    report = dict(
        span=[a, b],
        frames=N,
        sourceSHA256=J(W / "85-140/receipt.json")["sha256"],
        perSecond=per,
        gate="withhold" if candidate else "published control",
        criterion="no excess twin pairs above the matched encoded single-warp control on any frame; additionally inspect seams, structures and unmeasured regions in motion",
        stats={
            k: dict(
                allTwinPct=100
                * sum(r[k]["twins"] for r in rows)
                / max(1, sum(r[k]["edges"] for r in rows)),
                structureTwinPct=100
                * sum(r[k]["structureTwins"] for r in rows)
                / max(1, sum(r[k]["structureEdges"] for r in rows)),
                worstT=max(rows, key=lambda r: r[k]["structureFraction"])["t"],
                worstStructurePct=100 * max(r[k]["structureFraction"] for r in rows),
            )
            for k in keys
        },
    )
    if candidate:
        report["framesAboveSolo"] = sum(
            r["candidate"]["structureFraction"] > r["encodedSolo"]["structureFraction"]
            for r in rows
        )
        report["stageSHA256"] = hashlib.sha256(
            (p / "candidate.mp4").read_bytes()
        ).hexdigest()
    if candidate:
        above = sum(
            r["candidate"]["fraction"] > r["encodedSolo"]["fraction"] for r in rows
        )
        gate = dict(
            encodedSHA256=report["stageSHA256"],
            frames=N,
            framesAboveSolo=above,
            structureFramesAboveSolo=report["framesAboveSolo"],
        )
        g = W / "release-gates"
        g.mkdir(exist_ok=True)
        savej(g / (gate["encodedSHA256"] + ".json"), gate)
        report["gate"] = (
            "withhold"
            if (above or report["framesAboveSolo"])
            else "numeric pass; inspect structures in motion"
        )
    savej(p / "result.json", report)
    print("summary", a, b, report["stats"], flush=True)
    gallery(a, b, rows, keys)


def gallery(a, b, rows, keys):
    p = OUT / f"{a}-{b}"
    ts = track(a, b)
    arc, plan = clip_clock(a, b)
    pics = []
    for key in [k for k in keys if k != "encodedSolo"]:
        ranked = sorted(rows, key=lambda r: r[key]["structureFraction"], reverse=True)
        seen = set()
        best = []
        for row in ranked:
            sec = int(row["t"])
            if sec in seen:
                continue
            best.append(row)
            seen.add(sec)
            if len(best) == 4:
                break
        v = cv.VideoCapture(
            str(p / "candidate.mp4" if key == "candidate" else W / "85-140/chase.mp4")
        )
        gallery_rows = []
        for r in best:
            i = r["frame"]
            t, T = ts[i]
            _, cards = render(t, T, arc[i], plan=plan, resolver=Original())
            ev = r[key]
            rc = tuple(ev["worstBox"])
            x, y, w, h = rc
            card = next(q for q in cards if list(src_id(q)) == ev["reference"])
            native = full(*card, rc)
            v.set(
                cv.CAP_PROP_POS_FRAMES,
                round((t - (a if key == "candidate" else 85)) * 30),
            )
            ok, im = v.read()
            assert ok
            ctx = cv.resize(im, (272, 170))
            cv.rectangle(
                ctx,
                (x * 272 // OW, y * 170 // OH),
                ((x + w) * 272 // OW, (y + h) * 170 // OH),
                (20, 170, 255),
                1,
            )
            crop = im[y : y + h, x : x + w]
            stripe = np.zeros((24, 2 * w + 272, 3), "B")
            title = f"{key}  {t:.2f}s  {ev['structureFraction'] * 100:.2f}%  | output crop | single warped {ev['reference'][0]} {ev['reference'][1]}  [{x},{y}]"
            cv.putText(stripe, title, (4, 17), 0, 0.44, (242, 242, 242), 1, cv.LINE_AA)
            gallery_rows.append(np.vstack([np.hstack([ctx, crop, native]), stripe]))
        v.release()
        cv.imwrite(str(p / f"worst-{key}.jpg"), np.vstack(gallery_rows))
        pics += gallery_rows
    cv.imwrite(str(p / "worst-frames.jpg"), np.vstack(pics))


if __name__ == "__main__":
    run(int(sys.argv[1]), int(sys.argv[2]), "candidate" in sys.argv[3:])
