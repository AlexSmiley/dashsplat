"""Studio/APFS stage generations. Only the validated pointer is visible to readers."""

import contextlib
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def boot_id():
    if sys.platform.startswith("linux"):
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    text = subprocess.check_output(
        ["/usr/sbin/sysctl", "-n", "kern.boottime"], text=True
    )
    parts = re.search(r"sec\s*=\s*(\d+),\s*usec\s*=\s*(\d+)", text)
    return ".".join(parts.groups())


def identity():
    return dict(
        pid=os.getpid(),
        boot=boot_id(),
        started=subprocess.check_output(
            ["/bin/ps", "-p", str(os.getpid()), "-o", "lstart="], text=True
        ).strip(),
    )


def syncdir(p):
    fd = os.open(p, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=p.parent, prefix="." + p.name + "-")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=1, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, p)
        syncdir(p.parent)
    finally:
        Path(name).unlink(missing_ok=True)


@contextlib.contextmanager
def lock(p, blocking, who, **labels):
    # Never unlink the inode: a concurrent waiter may already hold its descriptor.
    # The kernel is authoritative even across reboot or PID reuse. The receipt
    # is diagnostic, not permission to break a live kernel lock.
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        f.seek(0)
        old = f.read()
        try:
            prior = json.loads(old) if old else {"state": "unowned"}
        except (ValueError, UnicodeError):
            prior = {"state": "incomplete"}
        owner = dict(who, **labels, state="held", acquired_unix=time.time())
        if prior.get("state") == "held":
            owner["recovered"] = {
                k: prior.get(k) for k in ("pid", "boot", "started", "drive", "stage")
            }
            print("reclaimed stale lock", p, owner["recovered"], flush=True)

        def store():
            f.seek(0)
            f.truncate()
            f.write(json.dumps(owner) + "\n")
            f.flush()
            os.fsync(f.fileno())

        store()
        try:
            yield f
        finally:
            owner.update(state="released", released_unix=time.time())
            store()


def promote(root, version):
    version = version.resolve(strict=True)
    parent = root.parent.resolve()
    if version.parent != parent / ".generations" or not version.name.startswith(
        root.name + "-"
    ):
        raise ValueError("unexpected stage generation: " + str(version))
    temp = parent / ("." + root.name + "-next")
    temp.unlink(missing_ok=True)
    temp.symlink_to(version, target_is_directory=True)
    os.replace(temp, root)
    syncdir(parent)


def initialize(root):
    versions = root.parent / ".generations"
    versions.mkdir(exist_ok=True)
    first = versions / (root.name + "-initial")
    if root.is_symlink():
        if not root.is_dir():
            raise ValueError(f"broken workspace pointer: {root}")
        return
    if root.exists():
        # Idempotent migration of the old real directory. A reboot between these
        # two renames leaves the entire prior workspace at its known bootstrap path.
        if first.exists():
            raise ValueError(f"conflicting legacy workspace: {first}")
        os.rename(root, first)
        syncdir(versions)
        syncdir(root.parent)
    else:
        first.mkdir(exist_ok=True)
    promote(root, first)


def clone(src, dst):
    flags = ["--reflink=auto", "-a"] if sys.platform.startswith("linux") else ["-cRp"]
    subprocess.run(["/bin/cp", *flags, str(src), str(dst)], check=True)


def begin(root, k, exclusions):
    source = root.resolve(strict=True)
    versions = source.parent
    dst = Path(tempfile.mkdtemp(prefix=root.name + "-" + k + "-", dir=versions))
    excluded = set(exclusions) | {".retired"}

    def copy(p, out, rel):
        if rel in excluded:
            return
        if p.is_symlink():
            out.symlink_to(os.readlink(p))
            return
        if any(x.startswith(rel + "/") for x in excluded) and p.is_dir():
            out.mkdir()
            for q in p.iterdir():
                copy(q, out / q.name, rel + "/" + q.name)
        else:
            # -c clones data blocks on APFS; -p preserves source times used by
            # upstream cache logic. No hard links to mutable producer outputs.
            clone(p, out)

    for p in source.iterdir():
        if p.name in excluded:
            continue
        # Raw pixels and inventory are immutable between stages. Clone calib: the
        # camera API resolves rig.json to find the sibling poses directory.
        if p.name in ("data", "inventory.json"):
            (dst / p.name).symlink_to(p.resolve(strict=True))
            continue
        copy(p, dst / p.name, p.name)
    return dst


def content_signature(p):
    # Include nested directory links (e.g. derived mask roots). Stream bytes;
    # timestamps alone cannot detect a preallocated or silently damaged tile.
    h = hashlib.sha256()

    def add(q, relative):
        h.update(os.fsencode(relative) + b"\0")
        if q.is_dir():
            h.update(b"D")
            for a in sorted(q.iterdir(), key=lambda f: f.name):
                add(a, relative + "/" + a.name)
        else:
            with q.open("rb") as f:
                h.update(b"F" + hashlib.file_digest(f, "sha256").digest())

    add(p, ".")
    return h.hexdigest()
