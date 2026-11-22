"""Install a verified candidate by immutable asset name, then rename its manifest."""

from pathlib import Path
import json, hashlib, os, argparse


def run(src, report, dest, drive_reference=None):
    if drive_reference is not None:
        # Exact equality also verifies timing, orientation, vertical datum and origin.
        if (dest.parent / "drive.json").read_bytes() != drive_reference.read_bytes():
            raise ValueError(
                "Drive coordinate frame differs; re-registration required before publication"
            )
    raw = (src / "ground.json").read_bytes()
    meta = json.loads(raw)
    receipt = json.loads(report.read_text())
    assert receipt["tiles"] == len(meta["tiles"]) and receipt["manifest_bytes"] == len(
        raw
    )
    size = 0
    mask_bytes = 0
    (dest / "tiles").mkdir(exist_ok=True, parents=True)
    for t in meta["tiles"]:
        p = src / t["texture"]
        d = dest / t["texture"]
        b = p.read_bytes()
        assert len(b) == t["bytes"] and hashlib.sha256(b).hexdigest()[:12] in p.stem
        size += len(b)
        if d.exists():
            assert d.read_bytes() == b
        else:
            os.link(p, d)
        if "provenance" in t:
            mask = src / t["provenance"]
            data = mask.read_bytes()
            assert (
                len(data) == t["provenanceBytes"]
                and hashlib.sha256(data).hexdigest()[:12] in mask.stem
            )
            installed = dest / t["provenance"]
            installed.parent.mkdir(exist_ok=True)
            mask_bytes += len(data)
            if installed.exists():
                assert installed.read_bytes() == data
            else:
                os.link(mask, installed)
    assert size == receipt["textures_bytes"] and mask_bytes == receipt.get(
        "provenance_bytes", 0
    )
    if os.path.exists(dest / "ground.json") and os.path.samefile(
        src / "ground.json", dest / "ground.json"
    ):
        print("Already published", len(meta["tiles"]))
        return
    tmp = dest / "ground.next"
    if tmp.exists():
        raise FileExistsError(tmp)
    os.link(src / "ground.json", tmp)
    tmp.replace(dest / "ground.json")
    assert (dest / "ground.json").read_bytes() == raw
    print("Published", len(meta["tiles"]), size + mask_bytes + len(raw), "bytes.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("source", type=Path)
    p.add_argument("receipt", type=Path)
    p.add_argument("--dest", type=Path, default=Path("scene/stitched-opt-v4/ground"))
    p.add_argument("--drive-reference", type=Path)
    a = p.parse_args()
    run(a.source, a.receipt, a.dest, a.drive_reference)
