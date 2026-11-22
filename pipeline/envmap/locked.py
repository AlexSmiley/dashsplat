"""Run a command under the Studio's reconstruction lock (pipeline/run/transactions.py): waits while the
whole-drive run holds it for a heavy stage, and holds that run off while the command runs.

    python -m pipeline.envmap.locked <stage> -- <command ...>
"""

import subprocess
import sys
import time

from pipeline.envmap.fill import RS
from pipeline.run.transactions import identity, lock

if __name__ == "__main__":
    stage, cmd = sys.argv[1], sys.argv[sys.argv.index("--") + 1:]
    with lock(RS / "work/locks/reconstruction.lock", True, identity(), drive="envmap-300-316", stage=stage):
        print("lock acquired", stage, time.ctime(), flush=True)
        sys.exit(subprocess.call(cmd))
