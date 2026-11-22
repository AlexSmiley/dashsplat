"""Uniform wrappers around the depth models tried in the bake-off.

Every wrapper takes rectified pinhole images (uint8 RGB, all the same size), their 3x3 intrinsics in
COLMAP pixel convention, and optionally camera-to-world poses (OpenCV axes, metres). It returns, for
each image, z-depth in metres on the same pixel grid (0 = invalid) and an optional confidence map.

    m = load_model("da3metric-large")
    out = m.predict(imgs, Ks, T_wc=None)   # list of dicts {depth, conf, sky}

Image size must be a multiple of 14 for the ViT models; callers choose it (see WORK_SIZES).
"""
import time

import numpy as np
import torch

from rscommon import resample_depth, stub_optional_modules

DEV = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

MODELS = {
    # name: (class, kwargs, licence)
    "da3metric-large": ("DA3Mono", dict(repo="depth-anything/DA3METRIC-LARGE"), "Apache-2.0"),
    "da3-large-1.1": ("DA3Multi", dict(repo="depth-anything/DA3-LARGE-1.1"), "CC BY-NC 4.0"),
    "da3-base": ("DA3Multi", dict(repo="depth-anything/DA3-BASE"), "Apache-2.0"),
    "da3-small": ("DA3Multi", dict(repo="depth-anything/DA3-SMALL"), "Apache-2.0"),
    "da3nested-1.1": ("DA3Multi", dict(repo="depth-anything/DA3NESTED-GIANT-LARGE-1.1", metric=True),
                      "CC BY-NC 4.0"),
    "mapanything": ("MapAnythingModel", dict(repo="facebook/map-anything"), "CC BY-NC 4.0"),
    "mapanything-apache": ("MapAnythingModel", dict(repo="facebook/map-anything-apache"), "Apache-2.0"),
    # full coverage: no MapAnything validity/edge masking (our own edge mask is applied downstream)
    "mapanything-nomask": ("MapAnythingModel", dict(repo="facebook/map-anything", mask=False), "CC BY-NC 4.0"),
    "moge2-vitl": ("MoGe2", dict(repo="Ruicheng/moge-2-vitl-normal"), "MIT code / weights see card"),
    "moge2-vitb": ("MoGe2", dict(repo="Ruicheng/moge-2-vitb-normal"), "MIT code / weights see card"),
    "unidepth-v2-l": ("UniDepth2", dict(repo="lpiccinelli/unidepth-v2-vitl14"), "CC BY-NC 4.0"),
}


def sync():
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    elif DEV.type == "mps":
        torch.mps.synchronize()


def empty_cache():
    if DEV.type in ("cuda", "mps"):
        getattr(torch, DEV.type).empty_cache()


def _w2c(T):
    T = np.asarray(T, np.float64)
    out = np.tile(np.eye(4), (len(T), 1, 1))
    R = T[:, :3, :3]
    out[:, :3, :3] = R.transpose(0, 2, 1)
    out[:, :3, 3] = -np.einsum("nji,nj->ni", R, T[:, :3, 3])
    return out


def _centres(E):
    R, t = E[:, :3, :3], E[:, :3, 3]
    return -np.einsum("nji,nj->ni", R, t)


def _da3_align_scale(extrinsics, intrinsics, prediction, align_to_input_ext_scale=True, **_):
    """Rescale DA3 depth to the metric frame of the input poses: median ratio of predicted to input
    pairwise camera baselines (pairs with no baseline, e.g. views of one lens at one time, are skipped).
    Taking the network depth as already in the normalised input frame was tried and is wrong (~10x)."""
    if extrinsics is None:
        return prediction
    prediction.intrinsics = intrinsics.numpy()
    E_in = extrinsics.numpy()
    E_pr = np.asarray(prediction.extrinsics)
    c_in, c_pr = _centres(E_in), _centres(E_pr)
    iu = np.triu_indices(len(c_in), 1)
    d_in = np.linalg.norm(c_in[:, None] - c_in[None], axis=-1)[iu]
    d_pr = np.linalg.norm(c_pr[:, None] - c_pr[None], axis=-1)[iu]
    ok = d_in > 0.05 * max(d_in.max(), 1e-9)
    s = float(np.median(d_pr[ok] / d_in[ok])) if ok.any() else 1.0
    prediction.scale_to_metric = 1.0 / s
    if align_to_input_ext_scale:
        prediction.depth = prediction.depth / s
        prediction.extrinsics = E_in[..., :3, :]
    return prediction


class DA3Mono:
    """DA3METRIC-LARGE: monocular metric depth = focal * net / 300 (README FAQ). Also predicts sky."""

    kind = "mono"

    def __init__(self, repo):
        stub_optional_modules()
        from depth_anything_3.api import DepthAnything3
        import depth_anything_3.utils.logger as lg
        try:
            lg.logger.level = 40
        except Exception:
            pass
        self.m = DepthAnything3.from_pretrained(repo).to(DEV).eval()

    @torch.inference_mode()
    def predict(self, imgs, Ks, T_wc=None):
        h, w = imgs[0].shape[:2]
        p = self.m.inference(list(imgs), process_res=max(w, h), process_res_method="upper_bound_resize")
        out = []
        for i, K in enumerate(Ks):
            d = p.depth[i]
            f = 0.5 * (K[0, 0] * d.shape[1] / w + K[1, 1] * d.shape[0] / h)
            d = d * f / 300.0
            sky = getattr(p, "sky", None)
            sky = sky[i] if sky is not None else None
            if d.shape != (h, w):
                Kd = K.copy()
                Kd[0] *= d.shape[1] / w
                Kd[1] *= d.shape[0] / h
                d = resample_depth(d, Kd, K, (w, h), "linear")
            out.append(dict(depth=d.astype(np.float32), conf=p.conf[i] if p.conf is not None else None,
                            sky=sky))
        return out


class DA3Multi:
    """DA3 any-view / nested models. With poses: pose-conditioned, depth aligned to the input pose scale.
    Without poses: plain multi-view (relative for any-view models, metric for the nested model)."""

    kind = "multi"

    def __init__(self, repo, metric=False):
        stub_optional_modules()
        from depth_anything_3.api import DepthAnything3
        self.metric = metric
        self.m = DepthAnything3.from_pretrained(repo).to(DEV).eval()
        # The stock alignment (evo Umeyama Sim3) fails on collinear camera centres, which is the
        # normal case for a car driving straight. Scale only needs the ratio of baselines.
        self.m._align_to_input_extrinsics_intrinsics = _da3_align_scale

    @torch.inference_mode()
    def predict(self, imgs, Ks, T_wc=None, use_pose=True, ref_view_strategy="first"):
        h, w = imgs[0].shape[:2]
        Ks = np.asarray(Ks, np.float32)
        kw = dict(process_res=max(w, h), process_res_method="upper_bound_resize",
                  ref_view_strategy=ref_view_strategy)
        if T_wc is not None and use_pose:
            kw.update(extrinsics=_w2c(T_wc).astype(np.float32), intrinsics=Ks, align_to_input_ext_scale=True)
        p = self.m.inference(list(imgs), **kw)
        out = []
        for i, K in enumerate(Ks):
            d = p.depth[i]
            if d.shape != (h, w):
                Kd = K.copy()
                Kd[0] *= d.shape[1] / w
                Kd[1] *= d.shape[0] / h
                d = resample_depth(d, Kd, K, (w, h), "linear")
            sky = getattr(p, "sky", None)
            out.append(dict(depth=d.astype(np.float32), conf=p.conf[i] if p.conf is not None else None,
                            sky=sky[i] if sky is not None else None))
        return out


class MapAnythingModel:
    """MapAnything with intrinsics, optional cam2world poses and the metric flag."""

    kind = "multi"

    def __init__(self, repo, mask=True):
        from mapanything.models import MapAnything
        self.m = MapAnything.from_pretrained(repo).to(DEV).eval()
        self.mask = mask

    @torch.inference_mode()
    def predict(self, imgs, Ks, T_wc=None, use_pose=True, amp=True):
        from mapanything.utils.image import preprocess_inputs
        h, w = imgs[0].shape[:2]
        views = []
        for i, (im, K) in enumerate(zip(imgs, Ks)):
            v = {"img": torch.from_numpy(np.ascontiguousarray(im)),
                 "intrinsics": torch.from_numpy(np.asarray(K, np.float32))}
            if T_wc is not None and use_pose:
                v["camera_poses"] = torch.from_numpy(np.asarray(T_wc[i], np.float32))
                v["is_metric_scale"] = torch.tensor([True])
            views.append(v)
        proc = preprocess_inputs(views, resize_mode="fixed_size", size=(w, h))
        for v in proc:
            for k, x in list(v.items()):
                if torch.is_tensor(x):
                    v[k] = x.to(DEV)
        preds = self.m.infer(proc, memory_efficient_inference=False, use_amp=amp, amp_dtype="bf16",
                             apply_mask=self.mask, mask_edges=self.mask)
        out = []
        for i, K in enumerate(Ks):
            pr = preds[i]
            d = pr["depth_z"][0, ..., 0].float().cpu().numpy()
            Ko = pr["intrinsics"][0].float().cpu().numpy()
            m = pr["mask"][0, ..., 0].cpu().numpy() if ("mask" in pr and self.mask) else None
            if m is not None:
                d = np.where(m > 0, d, 0)
            Ko = Ko.copy()
            Ko[0, 2] += 0.5  # MapAnything uses OpenCV pixel centres; ours are COLMAP (+0.5)
            Ko[1, 2] += 0.5
            d = resample_depth(d, Ko, np.asarray(K), (w, h), "nearest")
            conf = pr["conf"][0].float().cpu().numpy()
            conf = resample_depth(conf, Ko, np.asarray(K), (w, h), "nearest")
            out.append(dict(depth=d.astype(np.float32), conf=conf, sky=None))
        return out


class MoGe2:
    """MoGe-2 metric point map with the true horizontal field of view as input."""

    kind = "mono"

    def __init__(self, repo):
        from moge.model.v2 import MoGeModel
        self.m = MoGeModel.from_pretrained(repo).to(DEV).eval()

    @torch.inference_mode()
    def predict(self, imgs, Ks, T_wc=None, resolution_level=9, fp16=False):
        h, w = imgs[0].shape[:2]
        x = torch.from_numpy(np.stack(imgs)).to(DEV).permute(0, 3, 1, 2).float() / 255.0
        fov = torch.tensor([np.degrees(2 * np.arctan(w / (2 * K[0, 0]))) for K in Ks], dtype=torch.float32, device=DEV)
        o = self.m.infer(x, fov_x=fov, resolution_level=resolution_level, use_fp16=fp16)
        d = o["depth"].float().cpu().numpy()
        d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
        if "mask" in o:
            d = np.where(o["mask"].cpu().numpy(), d, 0)
        return [dict(depth=d[i].astype(np.float32), conf=None, sky=None) for i in range(len(imgs))]


class UniDepth2:
    """UniDepth v2 with known pinhole intrinsics."""

    kind = "mono"

    def __init__(self, repo):
        stub_optional_modules(("wandb",))
        from unidepth.models import UniDepthV2
        self.m = UniDepthV2.from_pretrained(repo).to(DEV).eval()

    @torch.inference_mode()
    def predict(self, imgs, Ks, T_wc=None):
        from unidepth.utils.camera import Pinhole
        x = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).to(DEV)
        K = torch.from_numpy(np.asarray(Ks, np.float32)).to(DEV)
        o = self.m.infer(x, Pinhole(K=K))
        d = o["depth"][:, 0].float().cpu().numpy()
        c = o.get("confidence")
        c = c[:, 0].float().cpu().numpy() if c is not None else [None] * len(imgs)
        return [dict(depth=d[i], conf=c[i], sky=None) for i in range(len(imgs))]


def load_model(name):
    cls, kw, lic = MODELS[name]
    t0 = time.time()
    m = globals()[cls](**kw)
    m.name, m.licence, m.load_s = name, lic, time.time() - t0
    return m
