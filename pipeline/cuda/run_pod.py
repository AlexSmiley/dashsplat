import hashlib, json, math, os, sys, subprocess as s, time, uuid, pathlib

D = pathlib.Path(os.environ["RS_JOB_DIR"]).resolve()
exe = os.environ.get("RUNPODCTL", "runpodctl")
drive_job = os.environ.get("RS_PROFILE") == "5"
fit_job = drive_job or os.environ.get("RS_PROFILE") == "4"
prefix = "rs-drive5-" if drive_job else "rs-fit4-" if fit_job else "rs-cuda-"
wall_limit = 18000 if drive_job else 4200
disk_rate = float(os.environ.get("RS_CONTAINER_GB", "60")) * .10 / 720
start = 0.0 if fit_job else float(os.environ["RS_START_BALANCE"])
fit_rate_cap = float(os.environ.get("RS_MAX_GPU_RATE", "1.1")) + max(0.01, disk_rate)
fit_prior = (
    sum(json.loads(f.read_text())["meter_usd"] for f in D.glob(prefix + "*.meter.json"))
    if fit_job
    else 0.0
)
sshkey = pathlib.Path(
    os.environ.get("RS_SSH_KEY", "~/.ssh/id_ed25519_personal")
).expanduser()
# SSH command prefix for the machine that holds the private input and receives releases (the author's Studio).
studio_ssh = os.environ.get("RS_STUDIO_SSH", "pssh anomastudio-ts").split()


def cli(*a, timeout=35):
    return json.loads(s.check_output([exe, *a], timeout=timeout))


def mine(name):
    return [p["id"] for p in cli("pod", "list") if p["name"] == name]


def delete(name):
    for id in mine(name):
        s.run([exe, "pod", "delete", id], check=True, stdout=s.DEVNULL, timeout=35)
    if mine(name):
        raise RuntimeError("pod still listed")


def guard_reason(name, owner):
    end = time.monotonic() + wall_limit
    unknown = None
    while True:
        now = time.monotonic()
        due = min(end, unknown + 180 if unknown is not None else end)
        if now >= due:
            return "wall clock" if due == end else "balance unknown for 180s"
        try:
            os.kill(owner, 0)
        except ProcessLookupError:
            return "parent exited"
        if (D / (name + ".complete")).exists():
            return "complete"
        try:
            account = cli("user", timeout=min(35, due - now))
            balance = float(account["clientBalance"])
            burn = float(account["currentSpendPerHr"]) if fit_job else 0.0
            if not math.isfinite(balance) or not math.isfinite(burn) or burn < 0:
                raise ValueError("invalid balance")
        except (OSError, ValueError, KeyError, TypeError, s.SubprocessError) as e:
            if unknown is None:
                unknown = now
            print("balance unknown:", type(e).__name__, flush=True)
        else:
            unknown = None
            if fit_job:
                # Leave room for the 180 s unknown-balance grace, deletion and delayed billing.
                reserve = (burn + fit_rate_cap) * 300 / 3600
                if balance <= 25.43 + max(1.0, reserve):
                    return "account floor"
                if fit_prior + fit_rate_cap * (wall_limit - end + now + 180) / 3600 >= 15:
                    return "task runtime cap"
            elif start - balance >= 34:
                return "known spend threshold"
        time.sleep(
            min(
                15,
                max(
                    0,
                    min(end, unknown + 180 if unknown is not None else end)
                    - time.monotonic(),
                ),
            )
        )


def export_release(row, job, ssh):
    import re
    directory = row['directory']; tag = row['stage']; dest = '~/.cache/rs-task5/' + job + '/' + tag
    if not re.fullmatch('/root/rs/work/drives/hw4-cuda-224-254/releases/(preview|progress)-[0-9]+', directory) or not re.fullmatch('[a-z0-9-]+', tag):
        raise ValueError('invalid release path')
    files = '.' if 'clip-' not in tag else 'generated generated.json'
    with s.Popen(ssh + [f'tar -cf - -C {directory} {files}'], stdout=s.PIPE) as source:
        s.run(studio_ssh + [f'umask 077; mkdir -p {dest} && tar xf - -C {dest}'], stdin=source.stdout, check=True, timeout=1200)
        source.stdout.close()
        if source.wait(timeout=15): raise ValueError('incomplete release stream')


def save_stream(stream, job, ssh=None):
    # Binary media immediately follow their JSON header on the same private SSH stream.
    media = D / "review"
    media.mkdir(exist_ok=True, mode=0o700)
    with (D / (job + ".receipts.jsonl")).open("ab", buffering=0) as receipts:
        for line in iter(stream.readline, b""):
            if not line.startswith(b"{"):
                print(line.decode(errors="replace").rstrip(), flush=True)
                continue
            try:
                row = json.loads(line)
            except ValueError:
                print(line.decode(errors="replace").rstrip(), flush=True)
                continue
            if drive_job and row.get('receipt') == 'drive_release':
                export_release(row, job, ssh)
            if drive_job and row.get('receipt')=='retry_needed':
                (D / 'failure.json').write_text(json.dumps(row))
                import socket
                path=os.environ.get('RS_EVENT_SOCKET')
                if path:
                    try:
                        with socket.socket(socket.AF_UNIX,socket.SOCK_DGRAM) as sock: sock.sendto(b'retry_needed',path)
                    except OSError: pass
            if "media" in row:
                fn = row["media"]
                assert pathlib.Path(fn).name == fn and pathlib.Path(fn).suffix in (
                    ".jpg",
                    ".mp4",
                    *((".ply",) if fit_job else ()),
                )
                left = row["bytes"]
                assert 0 < left <= 128 * 1024 * 1024
                tmp = media / (fn + ".part")
                digest = hashlib.sha256()
                with tmp.open("wb") as f:
                    while left:
                        block = stream.read(min(left, 65536))
                        if not block:
                            raise EOFError("partial media: " + fn)
                        f.write(block)
                        digest.update(block)
                        left -= len(block)
                    f.flush()
                    os.fsync(f.fileno())
                if digest.hexdigest() != row["sha256"]:
                    raise ValueError("media checksum: " + fn)
                tmp.replace(media / fn)
            if "receipt" in row or "media" in row:
                receipts.write(line)
                os.fsync(receipts.fileno())
                print(
                    "saved",
                    row.get("receipt", row.get("media")),
                    row.get("size", ""),
                    row.get("case", row.get("stage", "")),
                    flush=True,
                )
            else:
                print(line.decode(errors="replace").rstrip(), flush=True)
            if row.get("receipt") == "setup_failed" and os.environ.get(
                "RS_EVENT_SOCKET"
            ):
                import socket

                try:
                    with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sock:
                        sock.sendto(b"setup_failed", os.environ["RS_EVENT_SOCKET"])
                except OSError:
                    pass


if len(sys.argv) > 1 and sys.argv[1] == "guard":
    name, owner = sys.argv[2:]
    reason = guard_reason(name, int(owner))
    print(reason, flush=True)
    if reason == "complete":
        sys.exit()
    for i in range(6):
        try:
            delete(name)
            break
        except Exception:
            time.sleep(10)
    sys.exit()
name = prefix + uuid.uuid4().hex[:16]
assert not mine(name)
u = cli("user")
if fit_job:
    if any(
        not f.with_suffix(".meter.json").exists() for f in D.glob(prefix + "*.owned")
    ):
        raise RuntimeError("previous fit allocation has no closing meter")
    window_hours = (wall_limit + 300) / 3600
    max_cost = fit_rate_cap * window_hours
    if fit_prior + max_cost >= 15 or float(
        u["clientBalance"]
    ) <= 26.43 + max_cost + window_hours * float(u["currentSpendPerHr"]):
        raise RuntimeError("insufficient budget for bounded fit")
else:
    assert (
        start - u["clientBalance"] < 33
        and u["currentSpendPerHr"] <= 0.60
        and not cli("pod", "list")
        and not cli("nv", "list")
        and not cli("serverless", "list")
    )
successor = fit_job or os.environ.get("RS_PROFILE") == "3"
profile = successor or os.environ.get("RS_PROFILE") == "2"
bundle = D / ("profile.tar" if profile else "job.tar.gz")
if not bundle.is_file():
    raise FileNotFoundError(bundle)
log = open(D / (name + ".guard.log"), "w")
s.Popen(
    [sys.executable, __file__, "guard", name, str(os.getpid())],
    stdout=log,
    stderr=log,
    stdin=s.DEVNULL,
    start_new_session=True,
)
# Provider-provisioned in-pod identity only. External controller also deletes on image-pull failure, timeout or parent exit.
watch = """stopself() { runpodctl pod delete "$RUNPOD_POD_ID" || runpodctl remove pod "$RUNPOD_POD_ID" || runpodctl stop pod "$RUNPOD_POD_ID"; }; (sleep 3900; stopself) & (sleep 300; while pgrep -f '^bash pipeline/cuda/setup.sh' >/dev/null; do sleep 30; done; sleep 75; stopself) & exec /start.sh"""
if profile:
    watch = watch.replace("sleep 300; while", "sleep 660; while")
args = json.dumps({"entrypoint": ["/bin/bash", "-c"], "cmd": [watch]})
pub = pathlib.Path(str(sshkey) + ".pub").read_text().strip()
image = "runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404@sha256:60baa36d3fb6b98fd4f4ece6b96776c83c01a8b7c540e54460ab4d496816141f"
if successor:
    image = "nvcr.io/nvidia/pytorch:25.03-py3@sha256:54fd644e268346657619a987596e522e813d7abaaf92faafbe815b2b43cff16d"
    bootstrap = pathlib.Path(__file__).with_name("round3_boot.sh").read_text()
    if drive_job:
        bootstrap = bootstrap.replace('sleep 3900;', f'sleep {wall_limit-180};').replace('^bash pipeline/cuda/setup.sh', '^bash pipeline/cuda/drive_worker.sh|^bash pipeline/cuda/setup.sh')
    args = json.dumps(
        {
            "entrypoint": ["/opt/nvidia/nvidia_entrypoint.sh"],
            "cmd": ["/bin/bash", "-c", bootstrap],
        }
    )
if fit_job:
    (D / (name + ".owned")).touch(exist_ok=False)
allocated = time.monotonic()
hourly = fit_rate_cap
try:
    gpu = os.environ.get("GPU", "NVIDIA A40")
    cloud = os.environ.get("CLOUD", "SECURE")
    p = cli(
        "pod",
        "create",
        "--name",
        name,
        "--gpu-id",
        gpu,
        "--gpu-count",
        "1",
        "--cloud-type",
        cloud,
        "--image",
        image,
        "--container-disk-in-gb",
        os.environ.get("RS_CONTAINER_GB", "60"),
        "--volume-in-gb",
        "0",
        "--ports",
        "22/tcp",
        "--env",
        json.dumps({"PUBLIC_KEY": pub}),
        "--docker-args",
        args,
    )
    id = p["id"]
    hourly = (
        float(p["costPerHr"])
        + float(os.environ.get("RS_CONTAINER_GB", "60")) * 0.10 / 720
    )
    print("created", id, gpu, p.get("costPerHr"), flush=True)
    assert float(p["costPerHr"]) <= float(os.environ.get("RS_MAX_GPU_RATE", "1.1"))
    deadline = time.monotonic() + (720 if successor else 285)
    while time.monotonic() < deadline:
        info = cli("pod", "get", id)
        conn = info.get("ssh", {})
        ip = conn.get("ip")
        port = conn.get("port")
        if info.get("runtimeStatus") == "error":
            raise RuntimeError(info.get("runtimeStatusReason"))
        if ip and port:
            cmd = [
                "ssh",
                "-i",
                str(sshkey),
                "-p",
                str(port),
                "-o",
                "IdentitiesOnly=yes",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=4",
                "-o",
                "ServerAliveInterval=30",
                "-o",
                "ServerAliveCountMax=3",
                "-o",
                "StrictHostKeyChecking=accept-new",
                "-o",
                "UserKnownHostsFile=" + str(D / (name + ".known")),
                f"root@{ip}",
            ]
            if s.run(cmd + ["true"], stderr=s.DEVNULL).returncode == 0:
                break
        time.sleep(5)
    else:
        raise RuntimeError("direct SSH not available before boot deadline")
    if profile:
        try:
            observed_rate = cli("user")["currentSpendPerHr"]
        except (OSError, ValueError, KeyError, s.SubprocessError):
            observed_rate = None
        (D / (name + ".rate.json")).write_text(
            json.dumps(
                {
                    "gpu": gpu,
                    "pod": id,
                    "gpu_hourly": p["costPerHr"],
                    "account_hourly": observed_rate,
                }
            )
        )
    tarcmd = "tar xf -" if profile else "tar xzf -"
    s.run(
        cmd
        + [
            f"nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv && mkdir -p /root/rs && {tarcmd} -C /root/rs"
        ],
        stdin=bundle.open("rb"),
        timeout=480 if successor else 360 if profile else 140,
        check=True,
    )
    if fit_job:
        # Private Studio input archive relayed over SSH; never materialized on the controller.
        with s.Popen(
            studio_ssh + [os.environ["RS_STUDIO_STREAM"]], stdout=s.PIPE
        ) as src:
            s.run(
                cmd + ["tar xf - -C /root/rs"],
                stdin=src.stdout,
                check=True,
                timeout=1800 if drive_job else 480,
            )
            src.stdout.close()
            if src.wait(timeout=10):
                raise RuntimeError("Studio input stream failed")
    if drive_job:
        (D / (name + ".ssh.json")).write_text(json.dumps(cmd))
    remote_worker = cmd + [
        "cd /root/rs && export RS=$PWD && timeout 16200 bash pipeline/cuda/drive_worker.sh" if drive_job else "cd /root/rs && export RS=$PWD && timeout 3000 bash pipeline/cuda/setup.sh"
    ]
    if successor:
        import threading

        with s.Popen(remote_worker, stdout=s.PIPE) as child:
            deadline_timer = threading.Timer(16500 if drive_job else 3100, child.kill)
            deadline_timer.start()
            try:
                save_stream(child.stdout, name, cmd)
                code = child.wait()
            finally:
                deadline_timer.cancel()
                child.kill()
        worker = s.CompletedProcess(remote_worker, code)
    else:
        worker = s.run(remote_worker, timeout=3100)
    result = D / (name + "-results.tar.gz")
    with result.open("wb") as f:
        s.run(
            cmd
            + [
                (
                    "cd /root/rs/bench && tar --exclude=results/review -czf - results"
                    if successor
                    else "cd /root/rs/bench && tar czf - results"
                )
            ],
            stdout=f,
            check=True,
            timeout=180 if successor else 90,
        )
    print("results", result, flush=True)
    worker.check_returncode()
finally:
    delete(name)
    if fit_job:
        seconds = time.monotonic() - allocated
        (D / (name + ".meter.json")).write_text(
            json.dumps(
                dict(
                    name=name,
                    runtime_upper_s=seconds,
                    hourly=hourly,
                    meter_usd=seconds * hourly / 3600,
                    owned_remaining=mine(name),
                )
            )
        )
    (D / (name + ".complete")).touch()
    print("cleaned", name, cli("user")["clientBalance"], flush=True)
