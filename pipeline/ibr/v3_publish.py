"""Validate and publish the self-contained scene. Refuse to replace an existing scene."""

from .v3_pano import *
from .v3_ground import RoadMask
from .v3_objects import blocked
from .layer_player import Player
import hashlib, shutil


def center(a):
    n = np.array(a["plane"][:2])
    u, z, U, Z = a["extent"]
    a["center"] = [
        *(n * a["plane"][2] + np.array([-n[1], n[0]]) * (u + U) * 0.5),
        (z + Z) * 0.5,
    ]
    return a


def verify(root):
    p = Player(root)
    assets = set()
    dims = {}
    mask = RoadMask(W / "roadmask.npz")
    assert len(p.poses) == 1411 and p.doc["interval"] == [93, 140]
    for pg in p.pan:
        for k, shape in [("sky", (SH, SW)), ("rgb", (BH, BW)), ("matte", (BH, BW))]:
            actual = cv.imread(str(p.asset(pg[k])), 0).shape
            assert actual == (
                (448, 4096) if p.doc["contentRevision"] >= 3 and k != "sky" else shape
            )
            assets.add(pg[k])
    for c, ls in p.cams.items():
        assert all(ls[i]["t"] < ls[i + 1]["t"] for i in range(len(ls) - 1))
        assert ls[0]["t"] < 93 and ls[-1]["t"] >= HI - 1 / 60
    for ls in [p.cards] + [
        [a for f in es for a in f["layers"]] for es in p.dynamic.values()
    ]:
        for a in ls:
            assert abs(np.linalg.norm(a["plane"][:2]) - 1) < 1e-5
            paths = (a["rgb"], a["matte"])
            if paths not in dims:
                m = cv.imread(str(p.asset(paths[1])), 0)
                im = cv.imread(str(p.asset(paths[0])))
                assert m.shape == im.shape[:2]
                dims[paths] = m.shape
                assets.update(paths)
            h, w = dims[paths]
            x, y, X, Y = a["rect"]
            assert 0 <= x < x + X <= w and 0 <= y < y + Y <= h
    assert len({a["id"] for a in p.cards}) == len(p.cards)
    assert all(not blocked(a, mask) for a in p.cards)
    for e in p.tiles:
        assert len(p.tile(e["file"])) == e["count"]
    return dict(
        textures=len(assets),
        canonical=len(p.cards),
        moverExposures=sum(map(len, p.dynamic.values())),
        groundTiles=len(p.tiles),
    )


def publish():
    assert not SCENE.exists(), f"{SCENE} exists; never overwrite it"
    sets = getsets()
    bases = {}
    for c in C3:
        ls = []
        for k, s in enumerate(sets):
            if c not in s:
                continue
            e = s[c]
            if ls and e["n"] == ls[-1]["n"]:
                continue
            R, O = rowmat(e, np.array([0.5]))
            ls.append(
                dict(
                    t=e["t"],
                    s=e["s"],
                    n=e["n"],
                    pose=pack.pose(np.vstack((np.c_[R[0], O[0]], [0, 0, 0, 1]))),
                    rowPoses=e["rowPoses"],
                    pano=k,
                )
            )
        for i, e in enumerate(ls):
            e["prev"] = i - 1 if i else None
            e["next"] = i + 1 if i + 1 < len(ls) else None
        bases[c] = ls
    movers = {}
    dyn = json.load(open(W / "dynamic.json"))
    for c, es in dyn.items():
        buckets = {}
        for f in es:
            for a in f["layers"]:
                center(a)
            b = int(np.floor((f["t"] - LO) / 2))
            buckets.setdefault(b, []).append(f)
        movers[c] = []
        for bi, fs in buckets.items():
            name = f"movers/{c}-{bi:03}.json"
            dump(BUILD / name, fs)
            movers[c].append(dict(first=fs[0]["t"], last=fs[-1]["t"], file=name))
    ps = json.load(open(W / "panos.json"))
    cards = [center(a) for a in json.load(open(W / "cards.json"))]
    doc = dict(
        format="photo-layers",
        version=1,
        contentRevision=1,
        interval=[LO, HI],
        units="m",
        world="drive-z-up",
        drive="drive.json",
        rig="rig.json",
        chase="chase.json",
        ground=dict(map="map.json", sourceScene="road-stitched-v2", bakedMask=True),
        skyElevation=SKY_EL,
        backdropElevation=BACK_EL,
        bases=bases,
        panoramas=ps,
        cutouts=cards,
        movers=movers,
    )
    dump(BUILD / "ibr.json", doc)
    # Standard viewer fallback is a 2:1 equirectangular sky, in the pre-existing map's azimuth convention.
    im = cv.imread(str(BUILD / ps[len(ps) // 2]["sky"]))
    y, x = np.mgrid[:512, :1024]
    az = (0.5 - (x + 0.5) / 1024) * 2 * np.pi
    el = 90 - 180 * (y + 0.5) / 512
    uv = np.stack(
        [
            ((az / (2 * np.pi) + 0.5) * SW - 0.5) % SW,
            np.clip((SKY_EL[1] - el) / (SKY_EL[1] - SKY_EL[0]) * SH - 0.5, 0, SH - 1),
        ],
        2,
    ).astype("f")
    cv.imwrite(
        str(BUILD / "sky.jpg"),
        cv.remap(im, uv, None, cv.INTER_LINEAR, borderMode=cv.BORDER_WRAP),
    )
    stats = verify(BUILD)
    os.rename(BUILD, SCENE)
    print("published", SCENE, stats, flush=True)


if __name__ == "__main__":
    publish()
