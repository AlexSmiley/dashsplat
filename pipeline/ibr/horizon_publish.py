"""Extend only the panorama stream. Stage and validate every new path before atomically replacing the manifest."""

from .horizon_surface import *
import hashlib, shutil


def manifest(group=3):
    original = W / "prior-ibr-r3.json"
    j = load(original if original.exists() else SCENE / "ibr.json")
    assert (
        j["format"] == "photo-layers"
        and j["version"] == 1
        and j["contentRevision"] == 3
    ), "concurrent or unexpected manifest revision"
    j["contentRevision"] = 4
    intervals = [
        [[85.0, 140.0]],
        [[85.0, 140.0], [286.0, 330.0]],
        [[85.0, 140.0], [190.0, 215.0], [286.0, 330.0]],
        [[0.0, END]],
    ][group]
    j["interval"] = [intervals[0][0], intervals[-1][1]]
    j["layerIntervals"] = {
        "sky": intervals,
        "backdrop": intervals,
        "ground": [[93, 140]],
        "cutouts": [[93, 140]],
        "movers": [[93, 140]],
    }
    j["skyElevation"] = SKYEL
    j["backdropElevation"] = BACKEL
    j["backdropRange"] = {
        "startM": 45.0,
        "splatFadeM": [34.0, 49.0],
        "reference": "horizontal-eye-range",
        "bands": 1,
    }
    if (W / "handover.json").is_file():
        j["backdropRange"].update(load(W / "handover.json"))
    j["chase"] = "chase-r4.json"
    j["panoramas"] = []
    j["bases"] = {c: [] for c in C3}
    active = [i for i in anchors() if priority(rows()[i]["front"]["t"]) <= group]
    for k, i in enumerate(active):
        s = rows()[i]
        u = i
        j["panoramas"].append(
            dict(
                id=k,
                t=s["front"]["t"],
                origin=origins()[u].tolist(),
                radius=float(radii()[u]),
                rangeStartM=45.0,
                sky="horizon-r4/sky.jpg",
                rgb=f"horizon-r4/surface-{u:04}.jpg",
                matte=f"horizon-r4/surface-{u:04}.png",
            )
        )
        for c in C3:
            ls = j["bases"][c]
            e = s[c]
            if ls and ls[-1]["n"] == e["n"]:
                continue
            ls.append(dict(e, pano=k, prev=len(ls) - 1 if ls else None, next=None))
            if len(ls) > 1:
                ls[-2]["next"] = len(ls) - 1
    return j


def stage(group=3):
    j = manifest(group)
    p = W / "candidate"
    p.mkdir(exist_ok=True)
    for name in [
        "drive.json",
        "rig.json",
        "map.json",
        "movers",
        "objects",
        "rsb",
        "tiles",
    ]:
        link = p / name
        if not link.exists():  # Reader constrains asset roots: hardlink the small metadata and mover shard files used by its constructor.
            src = SCENE / name
            if src.is_file():
                os.link(src, link)
            elif name == "movers":
                link.mkdir()
                [os.link(x, link / x.name) for x in src.glob("*.json")]
    for f in [p / "horizon-r4", p / "chase-r4.json"]:
        if f.is_symlink():
            f.unlink()
    dst = p / "horizon-r4"
    dst.mkdir(exist_ok=True)
    sources = [(OUT / "sky.jpg", dst / "sky.jpg")] + [
        (SURF / f"{u:04}.{ext}", dst / f"surface-{u:04}.{ext}")
        for u in anchors()
        for ext in ["jpg", "png"]
    ]
    for x, target in sources:
        if x.exists() and not target.exists():
            os.link(x, target)
    if not (p / j["chase"]).exists():
        os.link(W / "live-chase.json", p / j["chase"])
    dump(p / "ibr.json", j)
    return p


def publish(group=3):
    assert group == 3, (
        "publish the four verified offline coverage stages as one atomic revision"
    )
    oldBytes = (SCENE / "ibr.json").read_bytes()
    p = stage(group)
    j = load(p / "ibr.json")
    ps = j["panoramas"]
    if group == 3:
        assert ps[0]["t"] <= 0.04 and ps[-1]["t"] >= END - 0.04
    assert len(load(p / j["chase"])["frames"]) == int(END * 30) + 1
    assert cv.imread(str(p / ps[0]["sky"])).shape[:2] == (SH, SW)
    info = []
    for q in ps:
        im = cv.imread(str(p / q["rgb"]))
        ma = cv.imread(str(p / q["matte"]), 0)
        assert im.shape[:2] == ma.shape == (BH, BW)
        info.append(float(np.mean(ma.max(0) > 127)))
    for c, ls in j["bases"].items():
        assert np.all(np.diff([v["t"] for v in ls]) > 0) and all(
            ps[e["pano"]]["id"] == e["pano"] for e in ls
        )
    old = load(SCENE / "ibr.json")
    assert old["contentRevision"] == 3
    assert (
        j["cutouts"] == old["cutouts"]
        and j["movers"] == old["movers"]
        and j["ground"] == old["ground"]
    )
    saved = W / "prior-ibr-r3.json"
    if not saved.is_file():
        shutil.copy2(SCENE / "ibr.json", saved)
    dst = SCENE / "horizon-r4"
    dst.mkdir(exist_ok=True)
    for f in {name for pg in ps for name in [pg["rgb"], pg["matte"], pg["sky"]]} | {
        j["chase"]
    }:
        if not (SCENE / f).is_file():
            os.link(p / f, SCENE / f)
    assert oldBytes == (SCENE / "ibr.json").read_bytes(), "concurrent manifest change"
    # Hash each referenced image, chase path, and preserved ground/near asset before the public pointer changes.
    for f in {
        j["chase"],
        *{pg[field] for pg in ps for field in ["sky", "rgb", "matte"]},
    }:
        assert (
            os.path.samefile(p / f, SCENE / f)
            or hashlib.sha256((p / f).read_bytes()).digest()
            == hashlib.sha256((SCENE / f).read_bytes()).digest()
        ), f
    paths = {j["drive"], j["rig"], j["ground"]["map"]}
    cards = list(j["cutouts"])
    for shards in j["movers"].values():
        for sh in shards:
            paths.add(sh["file"])
            cards.extend(card for e in load(SCENE / sh["file"]) for card in e["layers"])
    paths.update(card[key] for card in cards for key in ["rgb", "matte"])
    paths.update(tile["file"] for tile in load(SCENE / j["ground"]["map"])["tiles"])
    assert all((SCENE / f).is_file() for f in paths)
    dump(SCENE / "ibr.json", j)
    diag = load(W / "stations.json")
    q = [r for d in diag for r in d["rotation"] if r["accepted"]]
    near = [r for d in diag for r in d["rotation"] if r["n"] >= 9]
    metrics = dict(
        knots=len(ps),
        inputKnots=len(rows()),
        interval=j["interval"],
        contentRevision=j["contentRevision"],
        layerIntervals=j["layerIntervals"],
        minCoverage=float(min(info)),
        medianCoverage=float(np.median(info)),
        zeroCoverageKnots=int(np.sum(np.array(info) == 0)),
        stations=len(diag),
        medianSeams=int(np.median([v["seams"] for v in diag])),
        maxUnobservedCols=max(v["unseenCols"] for v in diag),
        rotationFits=len(q),
        eligibleFits=len(near),
        medianPrePx=float(np.median([r["pre"] for r in q])) if q else None,
        medianPostPx=float(np.median([r["post"] for r in q])) if q else None,
        p90PostPx=float(np.percentile([r["post"] for r in q], 90)) if q else None,
        photoBytes=sum(x.stat().st_size for x in (SCENE / "horizon-r4").iterdir()),
        sceneSHA256=hashlib.sha256((SCENE / "ibr.json").read_bytes()).hexdigest(),
    )
    dump(W / f"release-{group}.json", metrics)
    if group == 3:
        dump(W / "release.json", metrics)
    print(metrics, flush=True)


if __name__ == "__main__":
    publish(int(sys.argv[2])) if len(sys.argv) > 2 and sys.argv[
        1
    ] == "publish" else print(stage())
