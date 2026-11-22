"""Coordinate the six existing camera readers/writers, batching independent lens windows by shape."""

import argparse, collections, contextlib, json, os, queue, sys, threading, time, traceback
from pathlib import Path
from concurrent.futures import Future
import numpy as np
import torch

rs = Path(os.environ["RS"])
sys.path[:0] = [str(rs / "pipeline/depth"), str(rs / "pipeline/masks")]
import rscommon as rc, run_depth as depth, run_masks as masks, models
from seg import Segmenter, MOVABLE

rc.setup_env()
keys = json.loads((rs / "depth/keyframes.json").read_text())["frames"]
cams = list(keys)
assert len(cams) == 6 and all(len(keys[c]) == 12 for c in cams)
size = (724, 469)
ns = argparse.Namespace(
    size="724x469",
    width=518,
    window=5,
    window_multiview=3,
    win_spacing=1.0,
    posed=True,
    edge=0.06,
    max_depth=150.0,
    overwrite=True,
    limit=0,
    every=1,
    min_free=0,
    scale="none",
    scale_factor=1.0,
    cam_factors={},
)
traj = rc.trajectory()
kind = sys.argv[1]
inbox = queue.Queue()
acc = collections.Counter()
groups = collections.Counter()


@contextlib.contextmanager
def span(k):
    torch.cuda.synchronize()
    t = time.perf_counter()
    yield
    torch.cuda.synchronize()
    acc[k] += time.perf_counter() - t


class Request:
    name = "da3-large-1.1"
    kind = "multi"

    def __init__(self, c):
        self.camera = c

    def predict(self, *args, **kw):
        f = Future()
        inbox.put((self.camera, args, kw, f))
        return f.result(timeout=120)

    __call__ = predict


def capture(cam, K):
    try:
        if kind == "mask":
            masks.segment_camera(cam, keys[cam][:K], size, Request(cam), True)
        else:
            depth.process_camera(
                cam,
                keys[cam][:K],
                Request(cam),
                ns,
                traj,
                lambda: (_ for _ in ()).throw(
                    ValueError("unexpected monocular fallback")
                ),
            )
    except BaseException as exc:
        inbox.put((cam, None, traceback.format_exc(), None))


def mask_batch(req):
    images = [r[1][0][0] for r in req]
    h, w = images[0].shape[:2]
    cpu = torch.from_numpy(np.stack(images)).pin_memory()
    with span("transfer_and_device_prep"):
        x = cpu.to("cuda", non_blocking=True).permute(0, 3, 1, 2).half() / 255.0
        x = (x - mean) / std
    with span("gpu_forward"):
        out = seg.m(pixel_values=x)
    with span("gpu_post"):
        cls = out.class_queries_logits.float().softmax(-1)
        mk = torch.nn.functional.interpolate(
            out.masks_queries_logits.float(),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        ).sigmoid()
        decisions = []
        for cl, mask in zip(cls, mk):
            sc, lb = cl[:, :-1].max(-1)
            keep = sc > 0.5
            sc, lb, mask = sc[keep], lb[keep], mask[keep]
            idx = (sc[:, None, None] * mask).argmax(0) if len(sc) else None
            decisions.append(
                (sc, lb, mask > 0.5, idx, torch.argsort(sc, descending=True))
            )
    with span("transfer_and_device_prep"):
        decisions = [
            [v.cpu().numpy() if v is not None else None for v in r] for r in decisions
        ]
    result = []
    for sc, lb, mk, idx, order in decisions:
        sem = np.full((h, w), 255, np.uint8)
        inst = []
        for j in order:
            region = (idx == j) & mk[j]
            orig = mk[j].sum()
            area = region.sum()
            if orig == 0 or area == 0 or area / orig < 0.8:
                continue
            sem[region] = lb[j]
            if int(lb[j]) in MOVABLE:
                inst.append(dict(mask=region, label=int(lb[j]), score=float(sc[j])))
        result.append([dict(semantic=sem, instances=inst)])
    groups["6 x 1"] = groups.get("6 x 1", 0) + 1
    return result


def da_batch(req):
    buckets = collections.defaultdict(list)
    answer = [None] * len(req)
    for i, (_, args, kw, _) in enumerate(req):
        ims, ks, poses = args
        assert poses is not None
        ks = np.asarray(ks, np.float32)
        h, w = ims[0].shape[:2]
        cpu, ex, kin = da._preprocess_inputs(
            ims,
            models._w2c(poses).astype(np.float32),
            ks,
            max(h, w),
            "upper_bound_resize",
        )
        buckets[tuple(cpu.shape)].append((i, cpu, ex, kin, ks, h, w))
    for shape, items in buckets.items():
        with span("transfer_and_device_prep"):
            sets = [da._prepare_model_inputs(row[1], row[2], row[3]) for row in items]
            x = torch.cat([r[0] for r in sets])
            e = torch.cat([da._normalize_extrinsics(r[1].clone()) for r in sets])
            k = torch.cat([r[2] for r in sets])
            del sets
        with span("gpu_forward"), torch.autocast("cuda", dtype=torch.float16):
            raw = da.model(x, e, k, [], False, False, "first")
        groups[str((len(items), *shape))] += 1
        for b, (i, cpu, ex, kin, ks, h, w) in enumerate(items):
            with span("transfer_and_device_prep"):
                output = {
                    key: v[b : b + 1].float().cpu()
                    for key, v in raw.items()
                    if torch.is_tensor(v) and v.ndim and v.shape[0] == len(items)
                }
            pred = da._convert_to_prediction(output)
            pred = models._da3_align_scale(ex, kin, pred)
            da._add_processed_images(pred, cpu)
            res = []
            for j, K in enumerate(ks):
                d = pred.depth[j]
                assert np.isfinite(d).all() and np.max(d) > 0, (
                    "nonfinite/empty DA3 depth"
                )
                if d.shape != (h, w):
                    kd = K.copy()
                    kd[0] *= d.shape[1] / w
                    kd[1] *= d.shape[0] / h
                    d = rc.resample_depth(d, kd, K, (w, h), "linear")
                res.append(dict(depth=d.astype(np.float32), conf=None, sky=None))
            answer[i] = res
    return answer


if kind == "mask":
    seg = Segmenter(fp16=True)
    mean = torch.tensor(seg.proc.image_mean, device="cuda", dtype=torch.float16).view(
        1, 3, 1, 1
    )
    std = torch.tensor(seg.proc.image_std, device="cuda", dtype=torch.float16).view(
        1, 3, 1, 1
    )
else:
    da = models.load_model("da3-large-1.1").m
with torch.inference_mode():
    for K in (1, 12):
        acc.clear()
        groups.clear()
        torch.cuda.synchronize()
        start = time.perf_counter()
        workers = [
            threading.Thread(target=capture, args=(c, K), daemon=True) for c in cams
        ]
        for t in workers:
            t.start()
        for step in range(K):
            batch = []
            for c in cams:
                r = inbox.get(timeout=90)
                if r[3] is None:
                    raise RuntimeError(r[2])
                batch.append(r)
            assert len({r[0] for r in batch}) == 6
            outputs = mask_batch(batch) if kind == "mask" else da_batch(batch)
            for r, val in zip(batch, outputs):
                r[3].set_result(val)
        for t in workers:
            t.join(60)
            assert not t.is_alive()
        if not inbox.empty():
            raise RuntimeError(inbox.get()[2])
        torch.cuda.synchronize()
        total = time.perf_counter() - start
        row = dict(
            keyframes=K,
            seconds_per_keyframe={k: v / K for k, v in acc.items()},
            cpu_io_residual=(total - sum(acc.values())) / K,
            total=total / K,
            groups=dict(groups),
            gpu=torch.cuda.get_device_name(),
            precision="FP16 backbone autocast; SDK float32 pose and prediction heads"
            if kind == "depth"
            else "FP16 forward, float32 panoptic post",
        )
        print(row, flush=True)
        if K == 12:
            (rs / "bench/results" / f"{kind}.json").write_text(
                json.dumps(row, indent=2)
            )
