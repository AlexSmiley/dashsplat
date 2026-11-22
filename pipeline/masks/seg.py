"""Panoptic segmentation for sky and movable-object masks (Mask2Former Swin-L, Cityscapes panoptic).

    seg = Segmenter()                 # loads facebook/mask2former-swin-large-cityscapes-panoptic on MPS
    out = seg(images)                 # list of dicts: semantic (H,W uint8 Cityscapes train ids, 255 = none),
                                      #   instances: list of {mask (H,W bool), label, score}
Cityscapes train ids: 0 road, 1 sidewalk, 2 building, 3 wall, 4 fence, 5 pole, 6 traffic light,
7 traffic sign, 8 vegetation, 9 terrain, 10 sky, 11 person, 12 rider, 13 car, 14 truck, 15 bus,
16 train, 17 motorcycle, 18 bicycle. Movable instances exclude train (see MOVABLE).
"""
import numpy as np
import torch

DEV = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
REPO = "facebook/mask2former-swin-large-cityscapes-panoptic"
LICENCE = "Mask2Former code MIT; weights trained on Cityscapes (non-commercial dataset terms)"
SKY = 10
ROAD = 0
# Cityscapes "train" (16) is left out: there are no trains on this drive, and the model labels the
# drive-through building's wall as one in the car park.
MOVABLE = {11: "person", 12: "rider", 13: "car", 14: "truck", 15: "bus", 17: "motorcycle", 18: "bicycle"}


class Segmenter:
    def __init__(self, fp16=False):
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
        self.proc = AutoImageProcessor.from_pretrained(REPO)
        self.m = Mask2FormerForUniversalSegmentation.from_pretrained(REPO).to(DEV).eval()
        self.dtype = torch.float16 if fp16 else torch.float32
        if fp16:
            self.m = self.m.half()

    @torch.inference_mode()
    def __call__(self, images, score_thresh=0.5, overlap_thresh=0.8):
        """images: list of RGB uint8 arrays of one size. Returns per-image semantic map and instances."""
        h, w = images[0].shape[:2]
        x = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).to(DEV).to(self.dtype) / 255.0
        mean = torch.tensor(self.proc.image_mean, device=DEV, dtype=self.dtype).view(1, 3, 1, 1)
        std = torch.tensor(self.proc.image_std, device=DEV, dtype=self.dtype).view(1, 3, 1, 1)
        x = (x - mean) / std
        out = self.m(pixel_values=x)
        cls = out.class_queries_logits.float().softmax(-1)  # (B, Q, C+1)
        masks = out.masks_queries_logits.float()  # (B, Q, h/4, w/4)
        masks = torch.nn.functional.interpolate(masks, size=(h, w), mode="bilinear", align_corners=False)
        masks = masks.sigmoid()
        res = []
        for b in range(len(images)):
            scores, labels = cls[b, :, :-1].max(-1)
            keep = scores > score_thresh
            sc, lb, mk = scores[keep], labels[keep], masks[b][keep]
            sem = np.full((h, w), 255, np.uint8)
            inst = []
            if len(sc):
                prob = sc[:, None, None] * mk
                idx = prob.argmax(0)
                order = torch.argsort(sc, descending=True)
                idx_np = idx.cpu().numpy()
                mk_np = (mk > 0.5).cpu().numpy()
                lb_np, sc_np = lb.cpu().numpy(), sc.cpu().numpy()
                for k in order.cpu().numpy():
                    region = idx_np == k
                    orig = mk_np[k].sum()
                    area = (region & mk_np[k]).sum()
                    if orig == 0 or area == 0 or area / orig < overlap_thresh:
                        continue
                    region = region & mk_np[k]
                    sem[region] = lb_np[k]
                    if int(lb_np[k]) in MOVABLE:
                        inst.append(dict(mask=region, label=int(lb_np[k]), score=float(sc_np[k])))
            res.append(dict(semantic=sem, instances=inst))
        return res
