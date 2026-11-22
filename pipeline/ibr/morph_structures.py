"""Manually identified structural ROIs on the actual published drive; counts use the full-frame audit's same edge sample."""

from .morph_gate import *


def probes():
    p = OUT
    cap = cv.VideoCapture(str(W / "85-140/chase.mp4"))
    stage = cv.VideoCapture(str(p / "110-125/candidate.mp4"))
    rows = []
    data = []
    for t, regions in [
        (
            112,
            [
                ("office roof / facade", (585, 177, 335, 127)),
                ("utility pole + wires", (822, 7, 118, 245)),
                ("streetlamp / skyline", (480, 24, 175, 230)),
            ],
        ),
        (
            133,
            [
                ("signal mast + arms", (306, 149, 545, 110)),
                ("roadside sign", (1030, 250, 102, 154)),
                ("skyline behind signals", (391, 184, 422, 110)),
            ],
        ),
    ]:
        a, b = (110, 125) if t == 112 else (125, 140)
        ts = track(a, b)
        arc, plan = clip_clock(a, b)
        i = int((t - a) * 30)
        T = ts[i][1]
        ol, oc = render(t, T, arc[i], plan=plan, resolver=Original())
        cap.set(cv.CAP_PROP_POS_FRAMES, int((t - 85) * 30))
        ok, im = cap.read()
        assert ok
        refs = references(ol, cards=oc)
        ev = audit(im, ol, refs, True)
        ed = edges(im)[0] > 0
        Y, X = np.indices(ed.shape)
        sample = ((X + 3 * Y) % 4) == 0
        cand = None
        if t == 112:
            # Recreate the identical warm-start and segment-local ownership used by the encoded candidate.
            o = Owner()
            lo = (i // 45) * 45
            for idx in range(max(0, lo - 8), i + 1):
                ci, me, cl, pa, _, _ = render(
                    ts[idx][0], ts[idx][1], arc[idx], plan=plan, resolver=o
                )
            stage.set(cv.CAP_PROP_POS_FRAMES, i)
            ok, cand = stage.read()
            assert ok
            ce = audit(cand, cl, references(cl, pa), True)
            ced = edges(cand)[0] > 0
        for name, rc in regions:
            x, y, w, h = rc
            sl = np.s_[y : y + h, x : x + w]
            hits = ev["mark"][sl] != 0
            den = int((sample[sl] & ed[sl]).sum())
            old = int(hits.sum())
            label = int(
                np.bincount(ol[sl][hits] if hits.any() else ol[sl].ravel()).argmax()
            )
            ref = full(*oc[label], rc)
            panels = [im[sl], ref]
            cs = None
            if cand is not None:
                cs = (
                    int((ce["mark"][sl] != 0).sum()),
                    int((sample[sl] & ced[sl]).sum()),
                )
                panels.append(cand[sl])
            else:
                panels.append(np.zeros_like(ref))
            dh = 145
            dw = 380
            scaled = []
            for panel in panels:
                k = min(dw / w, dh / h)
                pw, ph = int(w * k), int(h * k)
                can = np.zeros((dh, dw, 3), "B")
                can[
                    (dh - ph) // 2 : (dh - ph) // 2 + ph,
                    (dw - pw) // 2 : (dw - pw) // 2 + pw,
                ] = cv.resize(panel, (pw, ph))
                scaled.append(can)
            title = (
                f"{name}  {t}s | published {old}/{den} | source {src_id(oc[label])} | staged "
                + (f"{cs[0]}/{cs[1]}" if cs else "not rendered")
            )
            strip = np.zeros((24, 3 * dw, 3), "B")
            cv.putText(strip, title, (6, 16), 0, 0.48, (245, 245, 245), 1, cv.LINE_AA)
            rows.append(np.vstack([np.hstack(scaled), strip]))
            data.append(
                dict(
                    t=t,
                    feature=name,
                    box=rc,
                    publishedHits=old,
                    publishedEdges=den,
                    stagedHits=cs[0] if cs else None,
                    stagedEdges=cs[1] if cs else None,
                    reference=src_id(oc[label]),
                )
            )
    cap.release()
    stage.release()
    cv.imwrite(str(p / "structural-crops.jpg"), np.vstack(rows))
    savej(p / "structural-probes.json", data)
    print(data, flush=True)


if __name__ == "__main__":
    probes()
