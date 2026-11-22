"""Refine automatically assigned observations using only the released prefix."""

import pickle
import numpy as np
from .tracking import Factors
from .inventory import SIZE
from .online import camera, outline, cap
from .demo import R


class PacketFactors(Factors):
    def __init__(self, rows, cls):
        self.rows = rows
        self.t = np.array([r["rt"] for r in rows])
        self.release = np.array([r["release"] for r in rows])
        self.n = len(rows)
        self.bb = np.array([r["bb"] for r in rows])
        self.R = np.array([r["C"][:3, :3] for r in rows])
        self.O = np.array([r["C"][:3, 3] for r in rows])
        self.plane = np.array([r["plane"] for r in rows])
        self.V = np.array([r["velocity"] for r in rows])
        self.omega = np.array([r["omega"] for r in rows])
        self.hz = np.array([r["foot"][1] for r in rows])
        self.factor = np.array([camera(r["c"]).rs_readout_s / 469 for r in rows])
        self.size = np.array(SIZE[cls])
        self.outline = outline(cls)
        self.first = np.ones(self.n, bool)
        self.g = np.array([np.r_[r["xy"] - r["dir"] * 0.75, 0] for r in rows])
        self.p = np.array(
            [
                camera(r["c"]).params
                * np.r_[
                    724 / camera(r["c"]).width,
                    469 / camera(r["c"]).height,
                    724 / camera(r["c"]).width,
                    469 / camera(r["c"]).height,
                    1,
                    1,
                    1,
                    1,
                ]
                for r in rows
            ]
        )


def refine(h, component=False, flow_at=None, decisions=None):
    rows = h["rows"]
    cls = h["cls"]
    f = PacketFactors(rows, cls)
    scl = np.ones(3)
    out = []
    snap = []
    reject = 0
    prior = None
    last_i = -1
    anchor = 0.0
    last_seen = -999
    begin = h["confirm"]
    plane = f.plane[0]
    hstep = 1 / 36
    speed_limit = 9 if cls == "person" else 39
    epochs = [(float(begin), cls)]
    geometry_time = None
    last_try = -1e9
    last_good_fit = -1
    positive = 0
    for j in range(
        int(np.ceil(begin * 36)), int(np.floor((f.release[-1] + 0.84) * 36)) + 1
    ):
        t = j / 36
        k = np.searchsorted(f.release, t, side="right") - 1
        if (
            k >= 2
            and k > last_i
            and (not component or t - last_try >= 0.08)
            and (prior is None or t - anchor >= 0.095)
        ):
            ii = np.flatnonzero((f.release <= t) & (f.t >= t - 0.92))
            ii = (
                ii[np.unique(np.linspace(0, len(ii) - 1, min(16, len(ii))).astype(int))]
                if len(ii)
                else ii
            )
            if len(ii) >= 3 and np.ptp(f.t[ii]) >= 0.075:
                if decisions is not None and cls == "car":
                    recent = [
                        r
                        for r in rows[max(0, k - 12) : k + 1]
                        if t - r["release"] < 0.55
                    ]
                    bus = [r for r in recent if r["cls"] == "bus" and r["score"] > 0.82]
                    if len(bus) >= 3 and len(bus) >= 0.42 * len(recent):
                        cls = "bus"
                        f = PacketFactors(rows, cls)
                        epochs.append((t, cls))
                last_try = t
                seed = (
                    None
                    if prior is None
                    else np.r_[prior[:2] + (t - anchor) * prior[2:], prior[2:]]
                )
                cur, _, err = f.fit(ii, t, seed, seed, scl)
                if seed is not None and err > 1.2:
                    alt, _, cost = f.fit(ii, t, None, seed, scl)
                    if cost < err:
                        cur, err = alt, cost
                # No retrospective motion decision: compare stationary and translating
                # silhouettes on the prefix at birth (three or more sightings).
                if prior is None:
                    _, _, stat = f.fit(ii, t, cur, sz=scl, stationary=True)
                    high = (
                        np.mean([r["score"] for r in rows[max(0, k - 4) : k + 1]])
                        > 0.987
                    )
                    if component:
                        # Real pixel motion can resolve short, poorly conditioned box fits.
                        # Without it, require two nonstationary fits on distinct new images
                        # and a credible world speed and class-sized silhouette.
                        speed = np.linalg.norm(cur[2:])
                        observed_move = flow_at is not None and flow_at <= t
                        nonlinear = (
                            stat > max(0.82, err * 1.53)
                            and err < 1.65
                            and 3.1 < speed < 32.5
                            and np.ptp(f.t[ii]) >= 0.18
                        )
                        if nonlinear and k != last_good_fit:
                            positive += 1
                            last_good_fit = k
                        elif not nonlinear:
                            positive = 0
                        if decisions is not None:
                            decision = decisions[k]
                            lane = decision["context"] == "lane-ahead"
                            park = decision["state"] == "parked"
                            known = decision["state"] == "moving"
                            provisional = lane or (
                                decision["context"] != "roadside"
                                and err < 1.8
                                and stat > max(0.55, err * 1.32)
                                and 2.4 < speed < 35
                                and np.ptp(f.t[ii]) >= 0.14
                            )
                            if park and not observed_move:
                                continue
                            if known or provisional:
                                geometry_time = t
                        if positive >= 2:
                            geometry_time = t
                        if (
                            decisions is not None
                            and geometry_time is None
                            and err < 2.85
                            and (
                                cls in ("bus", "truck")
                                or (
                                    rows[k]["score"] > 0.94
                                    and stat > max(0.62, err * 1.32)
                                    and np.ptp(f.t[ii]) >= 0.17
                                )
                            )
                        ):
                            geometry_time = t
                        if not observed_move and geometry_time is None:
                            continue
                        if err > 2.8:
                            continue
                    elif np.linalg.norm(cur[2:]) < (2.1 if high else 3.2) or stat < max(
                        0.50 if high else 0.83, err * (1.24 if high else 1.55)
                    ):
                        continue
                    cur[2:] = cap(cur[2:], speed_limit)
                    xy = cur[:2].copy()
                    dv = cur[2:].copy()
                    z = f.height(k, xy)
                    yaw = np.arctan2(dv[1], dv[0])
                    prior = cur
                    anchor = t
                    plane = f.plane[k]
                    last_seen = t
                else:
                    d = cur[:2] - seed[:2]
                    o = rows[k]
                    u = o["dir"]
                    ok = (
                        abs(d @ u) < 2.5 + 0.115 * o["range"] + 12 * (t - anchor)
                        and abs(d @ [-u[1], u[0]])
                        < 1.5 + 0.021 * o["range"] + 4 * (t - anchor)
                        and err < 3.1
                    )
                    if component and not ok and t - last_seen > 0.36 and len(ii) >= 4:
                        trial, _, trial_err = f.fit(ii, t, None, None, scl)
                        dd = trial[:2] - seed[:2]
                        if trial_err < 1.45 and np.linalg.norm(dd) < 2.5 + 0.22 * o[
                            "range"
                        ] + 8 * (t - last_seen):
                            cur = trial
                            d = dd
                            ok = True
                    if not ok and decisions is not None and t - last_seen > 0.22:
                        # A clipped outgoing silhouette can bias range by tens of metres.
                        # The incoming camera constrains a fresh metric target. Never
                        # teleport the presentation; it still follows the bounded controller.
                        jj = ii[
                            [
                                rows[l]["c"] == o["c"] and rows[l]["id"] == o["id"]
                                for l in ii
                            ]
                        ]
                        if len(jj) >= 3 and np.ptp(f.t[jj]) > 0.09:
                            trial, _, fit = f.fit(jj, t, None, None, scl)
                            dd = trial[:2] - seed[:2]
                            if (
                                fit < 2.1
                                and np.linalg.norm(dd) < 12 + 0.52 * o["range"]
                                and np.linalg.norm(trial[2:]) < 40
                            ):
                                cur = trial
                                d = dd
                                ok = True
                    if ok:
                        cur[2:] = cap(
                            seed[2:]
                            + 0.65 * cap(cur[2:] - seed[2:], 6 + 12 * (t - anchor)),
                            speed_limit,
                        )
                        cur[:2] = seed[:2] + 0.83 * d
                        prior = cur
                        anchor = t
                        last_seen = t
                        plane = f.plane[k]
                    else:
                        reject += 1
                last_i = k
                snap.append(
                    (
                        t,
                        *prior[:2] + (t - anchor) * prior[2:],
                        *prior[2:],
                        plane[:2] @ (prior[:2] + (t - anchor) * prior[2:]) + plane[2],
                    )
                )
        if prior is None:
            continue
        if (
            decisions is not None
            and decisions[k]["state"] == "parked"
            and decisions[k]["confidence"] >= 0.84
            and decisions[k]["camera_agreement"] > 0
        ):
            if (
                t - last_seen > 0.28
                or sum(z["state"] == "parked" for z in decisions[max(0, k - 6) : k + 1])
                >= 5
            ):
                break
        if t - last_seen > 1.25:
            if decisions is None or t - f.release[k] > 0.35:
                break
            # A newly released local component is a measurement, not a licence
            # to wait indefinitely. Require several coherent pixels for reset.
            jj = np.flatnonzero(
                (f.release <= t)
                & (f.t >= t - 0.85)
                & np.array(
                    [r["c"] == rows[k]["c"] and r["id"] == rows[k]["id"] for r in rows]
                )
            )
            if len(jj) >= 3:
                trial, _, err = f.fit(jj[-10:], t, None, None, scl)
                if err < 2.4:
                    prior = trial
                    prior[2:] = cap(prior[2:], speed_limit)
                    anchor = t
                    last_seen = t
                    plane = f.plane[k]
            if t - last_seen > 1.55:
                break
        wanted = prior[:2] + (t - anchor) * prior[2:]
        cmd = cap(prior[2:] + (wanted - xy) * 6.5, speed_limit)
        dv += cap(cmd - dv, 2.8)
        xy = xy + dv * hstep
        z += np.clip(plane[:2] @ xy + plane[2] - z, -0.13, 0.13)
        heading = np.arctan2(prior[3], prior[2])
        yaw += np.clip(
            np.arctan2(np.sin(heading - yaw), np.cos(heading - yaw)), -0.095, 0.095
        )
        out.append(
            (
                t + hstep,
                *xy,
                z,
                yaw,
                int(
                    t - f.release[k] > (1 / 36 if component else 0.12)
                    or t - last_seen > 0.32
                ),
            )
        )
    if not out:
        return None
    tr = dict(zip(("t", "x", "y", "z", "yaw", "state"), np.array(out).T))
    tr["state"] = tr["state"].astype(int)
    scan = dict(zip(("t", "x", "y", "vx", "vy", "z"), np.array(snap).T))
    s = dict(id=h["id"], name=f"online-{h['id']}", **{"class": cls})
    return (
        s,
        rows,
        tr,
        f.size,
        dict(
            discovery_confirm=begin,
            fit_rejections=reject,
            geometry_release=geometry_time,
            class_epochs=epochs,
        ),
        scan,
    )


def run():
    hs = pickle.load(open(R / "identities.pickle", "rb"))
    photo = R.name.startswith("photo-")
    component = photo or R.name == "component"
    footprints = pickle.load(open(R / "footprint.pickle", "rb")) if photo else {}
    if component:
        from .motion import release
        from .semantics import release as semrelease
        from .handoff import robust_class

        sem = pickle.load(open(R / "semantics.pickle", "rb"))
        flow = pickle.load(open(R / "motion.pickle", "rb"))
    res = []
    for ix, h in enumerate(hs):
        if ix % 250 == 0:
            print(ix, "of", len(hs), "released", len(res), flush=True)
        if h["confirm"] is None:
            continue
        flow_at = None
        if component:
            verified = semrelease(sem.get(h["id"], []), h["cls"])
            flow_at = release(flow.get(h["id"], []))
            if verified is None:
                continue
            t = max(verified, h["confirm"])
            h = dict(
                h,
                confirm=t,
                cls=robust_class([r for r in h["rows"] if r["release"] <= t]),
            )
        s = refine(h, component, flow_at, footprints.get(h["id"]))
        if s is not None:
            if photo:
                from .image_lock import steer

                s = steer(s)
            res.append(s)
    pickle.dump(res, open(R / "online.pickle", "wb"))
    print(
        "released",
        len(res),
        "tracks;",
        sum(s[4]["fit_rejections"] for s in res),
        "rejected updates",
        flush=True,
    )


if __name__ == "__main__":
    run()
