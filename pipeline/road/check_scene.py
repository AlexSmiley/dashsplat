"""Read every exported PLY, replay existence, and probe the actual driven corridor."""

import os, json
import numpy as np
from pipeline.common.rig import Trajectory
from .scene import RS, INF, reader
import csv


def run(version=1):
    out = RS / ("maps/ground-v2/debug" if version == 2 else "maps/ground-debug")
    out.mkdir(exist_ok=True, parents=True)
    res = {}
    report = []
    for name in ["live", "stitched"]:
        root = RS / "scene" / f"road-{name}-v{version}"
        idx = json.loads((root / "map.json").read_text())
        drive = json.loads((root / "drive.json").read_text())
        tr = Trajectory(
            RS / "poses" / ("ego_causal.npz" if name == "live" else "ego.npz")
        )
        assert (
            len(drive["t"]) == len(tr.t) == 12927
            and np.max(abs(np.array(drive["x"]) - tr.poses[:, 0, 3])) < 0.00051
            and np.max(abs(np.array(drive["z"]) - tr.poses[:, 2, 3])) < 0.00051
        )
        assert (root / "rig.json").read_bytes() == (RS / "calib/rig.json").read_bytes()
        grids = {}
        deaths = {}
        breaks = {}
        direction_checks = []
        tot = 0
        runct = np.zeros(3, int)
        allocated = 0
        files = set()
        time0 = np.inf
        time1 = 0.0
        after = 0
        for tile in idx["tiles"]:
            assert (
                tile["kind"] == "static"
                and tile["layer"] == "ground"
                and tile["file"] not in files
            )
            files.add(tile["file"])
            a = reader(root / tile["file"])
            assert len(a) == tile["count"] and np.isfinite(a).all()
            assert (
                np.max(abs(np.r_[a[:, :3].min(0), a[:, :3].max(0)] - tile["bbox"]))
                < 0.0001
            )
            assert np.max(abs((a[:, 13:17] ** 2).sum(1) - 1)) < 1e-5
            assert np.max(abs((a[:, 3:6] ** 2).sum(1) - 1)) < 1e-5
            assert np.max(abs(np.exp(a[:, 12]) - 0.008)) < 1e-6
            rgb = 0.5 + a[:, 6:9] * 0.28209479
            assert rgb.min() > -1e-5 and rgb.max() < 1.00001
            birth, death, e0, s1, e1, s2, e2 = a[:, 17:24].T
            assert np.all(
                (birth >= 0) & (birth < death) & (birth <= idx["t_end"] + 0.0001)
            )
            assert np.all(
                (birth <= e0)
                & (e0 <= death)
                & (e0 <= s1)
                & (s1 <= e1)
                & (e1 <= s2)
                & (s2 <= e2)
            )
            assert np.all((s1 == INF) == (e1 == INF)) and np.all(
                (s2 == INF) == (e2 == INF)
            )
            assert np.all((e1 == INF) | (e1 <= death)) and np.all(
                (e2 == INF) | (e2 <= death)
            )
            assert abs(tile["t_min"] - birth.min()) < 1e-6
            tx, ty = map(int, tile["id"].split("_"))
            x = np.floor((a[:, 0].astype("d") - tx * 50) * 220 / 50).astype(int)
            y = np.floor((a[:, 1].astype("d") - ty * 50) * 220 / 50).astype(int)
            assert x.min() >= 0 and y.min() >= 0 and x.max() < 220 and y.max() < 220
            grid = np.full(220 * 220, INF, "f")
            np.minimum.at(grid, y * 220 + x, birth)
            grids[tx, ty] = grid.reshape(220, 220)
            end = np.zeros(220 * 220, "f")
            np.maximum.at(end, y * 220 + x, death)
            deaths[tx, ty] = end.reshape(220, 220)
            # Cells may be rejected after close-range contradictions and later
            # re-admitted. Locate exact gaps in the union of exported lifetimes.
            if version == 2 and name == "live":
                cells = y * 220 + x
                order = np.lexsort((birth, cells))
                cells = cells[order]
                offset = 1024.0 * cells
                bb = birth[order]
                dd = (
                    np.maximum.accumulate(
                        offset + np.minimum(death[order], idx["t_end"] + 1)
                    )
                    - offset
                )
                h = (
                    np.flatnonzero(
                        (cells[1:] == cells[:-1]) & (bb[1:] > dd[:-1] + 1e-5)
                    )
                    + 1
                )
                if len(h):
                    breaks[(tx, ty)] = np.column_stack((cells[h], dd[h - 1], bb[h]))
            if version == 2 and name == "live":
                # Beyond 32 m only the front lens contributes new ground. Check
                # the actual exported major axis against bearing at release.
                q = a[::64]
                t0 = q[:, 17]
                dv = q[:, :2] - np.column_stack(
                    [np.interp(t0, drive["t"], drive[c]) for c in ("x", "y")]
                )
                yaw = np.interp(t0, drive["t"], drive["yaw"])
                rng = np.linalg.norm(dv, axis=1)
                jj = (
                    (rng > 32)
                    & (np.sum(dv * np.stack((np.cos(yaw), np.sin(yaw)), 1), 1) > 30)
                    & (q[:, 10] - q[:, 11] > np.log(2))
                )
                q = q[jj]
                w, x, y, z = q[:, 13:17].T
                axis = np.stack((1 - 2 * (y * y + z * z), 2 * (w * z + x * y)), 1)
                direction_checks.extend(
                    np.rad2deg(
                        np.arccos(
                            np.clip(
                                abs((axis * dv[jj]).sum(1))
                                / (np.linalg.norm(axis, axis=1) * rng[jj]),
                                0,
                                1,
                            )
                        )
                    )
                )
            tot += len(a)
            allocated += np.sum(grid != INF)
            runct += np.bincount((s1 < INF).astype(int) + (s2 < INF), minlength=3)
            time0 = min(time0, float(birth.min()))
            time1 = max(time1, float(birth.max()))
            after += np.sum(death < INF)
            if name == "stitched":
                assert np.all(birth == 0) and np.all(death == INF)
        assert {
            p.relative_to(root).as_posix() for p in (root / "tiles").glob("*.ply")
        } == files
        assert (
            tot
            < (
                {"live": 30000000, "stitched": 22000000 if version == 2 else 15000000}[
                    name
                ]
            )
        )  # sample replay at measured times, then terminal coverage of the same centres
        xy = np.stack(
            np.meshgrid(np.arange(-2, 10.01, 0.25), np.arange(-7.75, 8, 0.25)), -1
        ).reshape(-1, 2)
        band = [
            abs(xy[:, 1]) <= 1.6,
            (abs(xy[:, 1]) > 1.6) & (abs(xy[:, 1]) <= 4.5),
            abs(xy[:, 1]) >= 6,
        ]
        rows = []
        for t in np.arange(0.5, idx["t_end"], 0.5):
            E = tr.T_world_ego(t)
            yaw = np.arctan2(E[1, 0], E[0, 0])
            R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            P = xy @ R.T + E[:2, 3]
            jk = np.floor(P / 50).astype(int)
            ij = np.floor((P - jk * 50) * 220 / 50).astype(int)
            bt = np.full(len(ij), INF)
            dead = np.zeros(len(ij))
            final = np.zeros(len(ij), bool)
            mask = jk[:, 0] * 100000 + jk[:, 1]
            for m in np.unique(mask):
                u = mask == m
                g = grids.get(tuple(jk[np.flatnonzero(u)[0]]))
                if g is not None:
                    bt[u] = g[ij[u, 1], ij[u, 0]]
                    dead[u] = deaths[tuple(jk[np.flatnonzero(u)[0]])][
                        ij[u, 1], ij[u, 0]
                    ]
                    final[u] = (bt[u] <= idx["t_end"]) & (idx["t_end"] < dead[u])
                    group = breaks.get(tuple(jk[np.flatnonzero(u)[0]]))
                    if group is not None:
                        missing = group[(group[:, 1] <= t) & (t < group[:, 2]), 0]
                        local = np.flatnonzero(u)
                        bt[local[np.isin(ij[u, 1] * 220 + ij[u, 0], missing)]] = INF
            rows.append(
                [
                    t,
                    *[float(np.mean((bt[u] <= t) & (t < dead[u]))) for u in band],
                    *[float(np.mean(final[u])) for u in band],
                ]
            )
        a = np.array(rows)
        report.extend([[name, *v] for v in rows])
        stats = {}
        for label, lo, hi in [
            ("whole", 0, 362),
            ("initial stop", 5.67, 81.83),
            ("car-park exit", 84, 105),
            ("first arterial", 110, 130),
            ("light stop", 199, 202.65),
            ("fast straight", 293, 310),
            ("arterial turn", 316, 326),
        ]:
            sub = a[(a[:, 0] >= lo) & (a[:, 0] <= hi), 1:]
            stats[label] = np.mean(sub, axis=0).tolist()
        weak = a[:, 1] < 0.90
        starts = np.flatnonzero(weak & ~np.r_[False, weak[:-1]])
        ends = np.flatnonzero(weak & ~np.r_[weak[1:], False])
        weakest = sorted(
            [
                (float(a[i, 0]), float(a[j, 0]), float(a[i : j + 1, 1].min()))
                for i, j in zip(starts, ends)
            ],
            key=lambda z: z[0] - z[1],
        )[:7]
        res[name] = dict(
            validated=tot,
            run_histogram=runct.tolist(),
            birth_range=[time0, time1],
            retired=int(after),
            coverage=stats,
            weakest=weakest,
            accepted_area_m2=allocated * (50 / 220) ** 2,
        )
        if version == 2:
            if direction_checks:
                res[name]["far_axis_error_deg"] = dict(
                    samples=len(direction_checks),
                    median_p90=np.percentile(direction_checks, [50, 90]).tolist(),
                )
            tl = np.load(RS / "data/drive/timeline.npz")
            moving = np.arange(0.0, idx["t_end"], 0.25)
            moving = moving[np.interp(moving, tl["t"], tl["speed"]) > 0.2]
            reach = []
            at = [20, 40, 60, 75]
            side = np.linspace(-1.5, 1.5, 13)
            lateral = []
            llong, lside = np.meshgrid(np.arange(0, 11, 2), [-19.5, -18, 18, 19.5])
            for now in moving:
                E = tr.T_world_ego(now)
                fwd = E[:2, 0]
                fwd = fwd / np.linalg.norm(fwd)
                left = np.array([-fwd[1], fwd[0]])
                P = (
                    E[:2, 3]
                    + np.array(at)[:, None, None] * fwd
                    + side[None, :, None] * left
                )
                jk = np.floor(P / 50).astype(int)
                ij = np.floor((P - jk * 50) * 220 / 50).astype(int)
                yes = np.zeros((4, len(side)), bool)
                for rr in range(4):
                    for cc in range(len(side)):
                        grid = grids.get(tuple(jk[rr, cc]))
                        xx, yy = ij[rr, cc]
                        yes[rr, cc] = (
                            grid is not None
                            and grid[yy, xx] <= now < deaths[tuple(jk[rr, cc])][yy, xx]
                        )
                        group = breaks.get(tuple(jk[rr, cc]))
                        if group is not None and yes[rr, cc]:
                            yes[rr, cc] = not np.any(
                                (group[:, 0] == yy * 220 + xx)
                                & (group[:, 1] <= now)
                                & (group[:, 2] > now)
                            )
                reach.append(yes)
                P = E[:2, 3] + llong.reshape(-1, 1) * fwd + lside.reshape(-1, 1) * left
                jj = np.floor(P / 50).astype(int)
                ii = np.floor((P - 50 * jj) * 220 / 50).astype(int)
                at_side = []
                for tile, (xx, yy) in zip(map(tuple, jj), ii):
                    grid = grids.get(tile)
                    found = (
                        grid is not None and grid[yy, xx] <= now < deaths[tile][yy, xx]
                    )
                    group = breaks.get(tile)
                    if found and group is not None:
                        found = not np.any(
                            (group[:, 0] == yy * 220 + xx)
                            & (group[:, 1] <= now)
                            & (now < group[:, 2])
                        )
                    at_side.append(found)
                lateral.append(at_side)
            hit = np.stack(reach)
            res[name]["reach"] = dict(
                samples=len(hit),
                supported_lateral_18_to_19_5m=float(np.mean(lateral)),
                distances=at,
                centre=hit[:, :, 6].mean(0).tolist(),
                fast_straight_centre=hit[(moving >= 293) & (moving <= 310), :, 6]
                .mean(0)
                .tolist(),
                lane_fraction=hit.mean((0, 2)).tolist(),
                half_lane=(hit.mean(2) >= 0.5).mean(0).tolist(),
            )
            print(name, "reach", res[name]["reach"], flush=True)
        print(name, tot, stats["whole"], weakest[:2], flush=True)
    with open(out / "ground-coverage.csv", "w") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "variant",
                "t",
                "core_now",
                "adjacent_now",
                "verge_now",
                "core_final",
                "adjacent_final",
                "verge_final",
            ]
        )
        w.writerows(report)
    (out / "verified.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    run(int(os.environ.get("GROUND_VERSION", "1")))
