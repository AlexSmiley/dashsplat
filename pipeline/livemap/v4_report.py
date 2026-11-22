"""Write the reproducible slice handoff from verified PLYs and rendered evidence."""

from .v4_verify import *


def report():
    root = RS / "pipeline/livemap"
    s, v, p, m = [
        json.loads((V4 / ("slice-" + f + ".json")).read_text())
        for f in ["summary", "verified", "proof", "traffic"]
    ]
    out = dict(build=s, scene=v, photographs_and_video=p, traffic=m)
    (root / "measurements-v4.json").write_text(json.dumps(out, indent=2) + "\n")
    movie = p["movie"]
    r = v["range_and_rate"]
    later = sum(q["after_140"] for q in v["whole_input_snapshot"].values())
    md = [
        "# Live world v4: 85–140 s",
        "",
        "[10-second chase, 30 fps](report/chase-v4-94-104.mp4) · [held-out storefront](report/v4/camera-left_repeater.jpg) · [exit orbit](report/v4/orbit-exit.jpg) · [arterial orbit](report/v4/orbit-arterial.jpg)",
        "",
        "## Published scene",
        "",
        f"Studio: `$RS/scene/live-v4/map.json`. {v['ground_links']:,} hard-linked original ground tiles; {v['object_tiles']:,} non-ground tiles ({v['transient_tiles']:,} transient). {s['static']:,} immutable static generations and {s['movers']:,} instance-tagged transient generations; **{s['records']:,}** records total, including retirements ({s['bytes'] / 1e6:.1f} MB of object PLYs). 5 Hz sampling found a peak of {v['peak_extant_5hz']:,} extant records. Fusion including preparation ran at {s['fps']:.1f} camera frames/s on Studio; depth, export and rendering are excluded.",
        "",
        f"Distance extrapolation: **{s['whole_distance'] / 1e6:.2f}M**. Time extrapolation: **{s['whole_time'] / 1e6:.2f}M**. Both include retired records, against a 60M limit. These are slice estimates, not a whole-drive measurement. The six-lens whole schedule needs {later:,} source exposures after 140 s; the input inventory below is independent of these projections.",
        "",
        "## Held-out real cameras",
        "",
        "Exactly the v3 set: ten scored physical exposures per camera, unseen by object depth windows and the original ground schedule. The additional storefront row is unscored because ground v2 saw it. Denominator: calibrated non-sky, non-road pixels **including movable instances** (a mover overrides a road label). Results are pooled by pixel within each lens. Geometry requires accumulated alpha >0.50; objects require non-ground alpha contribution >0.45 after depth compositing. The same slice panorama is behind both maps and contributes zero to both tests. PSNR uses the unmodified photographs, including misses; opacity coverage is not correct placement.",
        "",
        "| Camera / comparison | v3 geom. | v4 geom. | Lowest v4 frame | v3 objects | v4 objects | Non-ground PSNR v3 → v4 | v4 objects over sky |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c in CAMS:
        a = p["cameras"][c]["v3"]
        b = p["cameras"][c]["v4"]
        md.append(
            f"| [{c}](report/v4/camera-{c}.jpg) | {a['non_ground_geometry']:.1%} | {b['non_ground_geometry']:.1%} | {b['min_frame_geometry']:.1%} | {a['non_ground_objects']:.1%} | {b['non_ground_objects']:.1%} | {a['photo_psnr']:.2f} → {b['photo_psnr']:.2f} dB | {b['sky_silhouette']:.1%} |"
        )
    passed = sum(
        i["v4"]["geometry"] > 0.90 * i["v4"]["pixels"]
        for c in CAMS
        for i in p["cameras"][c]["frames"]
    )
    md += [
        "",
        f"**{passed}/60 individual held-out images** exceed 90% geometry coverage. Scores on the stricter object-contribution test are shown separately. See [the measurements](measurements-v4.json) for every frame and the mover-only subset.",
        "",
        "## World-space evidence and temporal behaviour",
        "",
        "Both chase and orbit use the stored scene from a camera 6 m back and 3 m up. Each slow orbit freezes scene time and turns through 360° in 12 s; the chase video advances both the car and scene clock for 300 consecutive frames. No source photo, synthetic car, or unprojected backdrop is inserted. Studio videos: `$RS/work/livemap/orbit-v4-<name>.mp4` and `$RS/work/livemap/chase-v4-94-104.mp4`.",
        "",
        "| Moment | Chase v3 / v4 | v4 orbit | Traffic detail |",
        "|---|---|---|---|",
    ]
    for name, val in p["orbits"].items():
        md.append(
            f"| {name} · {val['t']:.2f} s | [chase](report/v4/chase-{name}.jpg) | [film](report/v4/orbit-v4-{name}.mp4) · [sheet](report/v4/orbit-{name}.jpg) | "
            + (
                f"[crop sequence](report/v4/traffic-handover-{name}.jpg)"
                if name in m
                else "—"
            )
            + " |"
        )
    md += [
        "",
        "**What holds together:** the exit storefront, red fascia, planting island and opposite tree line stay recognisable through the turn. The forward arterial has a horizon again. Neither the sign lettering nor the signal gantry is reliably legible.",
        "",
        "**What still moves:** during the 94–104 s chase, lane markings mostly stay put while curb vegetation and the farther trees change shape and slide relative to them. The quarter-second fade softens the static refresh, but around high-contrast trunks and façade columns it produces two soft edges and local flicker. Turning the 135 s orbit backwards shows retained roadsides; some verge and canopy sheets float above the asphalt, and several poles lean or stop short of it. This is a near-car reconstruction, not stable arbitrary-view geometry.",
        "",
        "**Traffic:** the bus is recognisable across four updates, but its red roof and rear edge trail during handover. The white crossing SUV is almost entirely missing at its photographed location; pale fragments appear behind it. In this sequence its depth centroids move 2–5 m per update, and capture-to-release plus the fade delays the displayed position. The sidecar supplies measurements for a rigid-track model; these transients do not already implement one.",
        "",
        "**Coverage tradeoff:** the producer already invalidates depth edges. The final pass erodes between two measured, disagreeing ranges rather than treating an absent neighbour as zero metres and eroding again. This recovered thin and distant fragments, but side-camera splats now cover up to 35% of held-out sky. The reserved photographs were used to diagnose this sampling choice; their pixels were never fused, but these are not blind model-selection results.",
        "",
        "During the chase, consecutive frames were warped by the known change in camera pose and compared only where both depths agree. This measures image change on geometrically matched pixels, not position error against a photograph.",
        "",
        "| Region | Mean RGB change /255 | Fraction changing by >25 | Mean of framewise p90 |",
        "|---|---:|---:|---:|",
    ]
    for region, (mean, frac, p90) in movie["warp"].items():
        md.append(f"| {region} | {mean:.2f} | {frac:.2%} | {p90:.2f} |")
    ev = s["events"]
    md += [
        "",
        f"Static agreement extended {ev['confirmations']:,} visibility observations; {ev['finer']:,} finer handovers, {ev['see_through']:,} free-space/sky retirements and {ev['sheet_replaced']:,} displaced front sheets were recorded. The quarter-second overlap is real; it does not guarantee constant opacity or perfect registration. Rear memory coarsening retired {ev['memory_decimated']:,} old records. The playback tape still includes them.",
        "",
        "For the traffic sequences, the green overlay is visible transient alpha >0.35 inside the tracked mask; red is visible transient outside **all** instance masks dilated by two pixels, within the crop. These are illustrative training-time photographs evaluated at capture time, not additional holdouts. The displayed scene can only use packets already released. A mask classification or timing error can also look like a trail.",
        "",
    ]
    for name, rows0 in m.items():
        md.append(
            f"- **{name}:** {len(rows0)} photographic instants; mean transient spill outside all nearby masks {np.mean([a['fraction_outside_any_dilated_instance'] for a in rows0]):.1%}. Mean target-mask coverage by transients {np.mean([a['tagged_fraction'] for a in rows0]):.1%}."
        )
    md += [
        "",
        "## Inputs, range and sky",
        "",
        "Requested 4.5 Hz moving, 2 Hz stopped, with a one-second warm-up from 84 s. Actual intervals vary because inputs must be complete and held-out frames cannot occur in depth windows. The scene clock uses calibrated release time, not an unreleased physical exposure.",
        "",
        "| Camera | Packets | Moving Hz | Stopped Hz | p95 gap (s) | Max release lag (s) | Records born ≥100 m | Maximum range (m) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c, q in r.items():
        stopped = "—" if q["stopped_hz"] is None else f"{q['stopped_hz']:.2f}"
        md.append(
            f"| {c} | {q['frames']} | {q['moving_hz']:.2f} | {stopped} | {q['p95_gap']:.3f} | {q['release_lag_max']:.3f} | {q['far_100']:,} | {q['range_max']:.1f} |"
        )
    md += [
        "",
        f"Sky colour is measured in {s['panorama']['frames']} training exposures. Direct bins occupy {s['panorama']['upper_hemisphere_measured']:.1%} of the upper-hemisphere texels before wrapped filtering and smooth extension. The sky is a single non-causal slice-wide texture by design. No ground or object position is inferred from it.",
        "",
        f"All {v['records_verified']:,} PLY records passed header/extent, finite geometry, quaternion and lifetime checks. Every transient row has a source camera, ordinal and mask ID, with its PLY tile range and per-frame robust world centroid in `$RS/maps/live-v4/instances.json`. {v['instances_with_no_valid_depth']} segmented observations had no valid depth centroid and are explicitly null. IDs are camera-local; these are not rigid tracks.",
        "",
        f"Whole-drive inventory after 140 s at verification: {sum(q['depth_after_140'] for q in v['whole_input_snapshot'].values()):,}/{later:,} scheduled shared depths and {sum(q['sky_after_140'] for q in v['whole_input_snapshot'].values()):,}/{later:,} shared sky masks. See [reproduction](README.md) and [the updated live contract](DESIGN.md).",
        "",
    ]
    (root / "REPORT-v4.md").write_text("\n".join(md))


if __name__ == "__main__":
    report()
