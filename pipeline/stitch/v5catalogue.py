"""Check upright tracks against the actual post-pruning tile payloads; preserve all unmatched mask sightings."""

from .v5world import *


def catalogue(src, dst):
    D = W5
    slice0 = f"{START:g}-{END:g}"
    m = J(src / "map.json")
    tiles = m["tiles"]
    A = np.concatenate(
        [reader(src / t["file"]) for t in tiles if t["id"].startswith("columns_")]
    )
    tree = cKDTree(A[:, :2])
    inv = J(D / "catalogue.json")
    obs = J(D / "instances.json")
    rr = inv["records"]
    for r in rr:
        r["release_status"] = r["status"]
        if r["status"] not in [
            "continuous shaft",
            "partial",
            "panel; mount unverified",
            "panel + continuous shaft",
        ]:
            continue
        shaft = r["class"] == "pole/shaft"
        ii = np.array(tree.query_ball_point(r["xy"], 0.018 if shaft else 1.65), int)
        b = A[ii]
        b = b[abs(b[:, 19] - r["time"]) < 0.002]
        b = b[b[:, 9] > -0.1]
        r["release_splats"] = len(b)
        if not len(b):
            r["release_status"] = "not in release"
            continue
        if r["status"] == "continuous shaft":
            hz = np.linspace(r["floor"] + 0.055, r["top"], 67)
            delta = abs(hz[:, None] - b[None, :, 2])
            occupied = (delta < np.exp(b[None, :, 12]) * 1.5).any(1)
            r["height_coverage"] = float(occupied.mean())
            if not occupied.all():
                r["release_status"] = "partial shaft"
    for r in rr:
        if r["release_status"] == "panel + continuous shaft" and not any(
            x["id"] in r.get("mounts", []) and x["release_status"] == "continuous shaft"
            for x in rr
        ):
            r["release_status"] = "panel; mount unverified"
    rec = []
    for r in rr:
        samples = [obs[i] for i in r["component_ids"]]
        rec.append(
            dict(
                id=r["id"],
                kind=r["class"],
                status=r["release_status"],
                xy=[round(float(v), 3) for v in r["xy"]],
                path_m=round(r["path_m"], 2),
                ground_z=round(r["floor"], 2),
                top_z=round(r["top"], 2),
                views=r["views"],
                height_bins=f"{r['supported_height_bins']}/17",
                mounts=r.get("mounts", []),
                same_as=r.get("same_as"),
                first=[samples[0]["c"], samples[0]["n"], samples[0]["bb"]],
                last=[samples[-1]["c"], samples[-1]["n"], samples[-1]["bb"]],
            )
        )
    missing = [
        dict(c=obs[i]["c"], n=obs[i]["n"], bb=obs[i]["bb"], kind=obs[i]["k"])
        for i in inv["unassigned_component_ids"]
    ]
    savej(
        dst / ("uprights-" + slice0 + ".json"),
        dict(
            status_meaning="Observed shaft span is measured; lamp head and signal arm are not implied. Unverified lines are hypotheses, not counts of physical objects.",
            tracks=rec,
            unmatched_image_components=missing,
            arms=inv.get("arms", []),
            ridge_only=inv.get("other_ridge_tracks", []),
        ),
    )


if __name__ == "__main__":
    catalogue(Path(sys.argv[1]), Path(sys.argv[2]))
