"""Independent crop classifiers returning ranked (label, score) pairs."""

import json
import os
from pathlib import Path
import subprocess
import sys

MODEL = "google/siglip2-base-patch16-224"
REVISION = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
# Include plausible non-targets: a forced choice between mover classes would
# corroborate the false mask detections this stage is supposed to reject.
PROMPTS = {
    "car": "a car",
    "suv": "an SUV",
    "van": "a van",
    "truck": "a truck",
    "bus": "a bus",
    "motorhome": "a motorhome",
    "person": "a pedestrian",
    "bicycle": "a bicycle",
    "motorcycle": "a motorcycle",
    "cyclist": "a cyclist",
    "sign": "a road sign",
    "building": "a building",
    "vegetation": "trees or bushes",
    "road": "an empty road",
    "barrier": "a fence or guardrail",
    "background": "an indistinct patch of background scenery",
}


def vision(paths, root):
    exe = root / "classify"
    src = Path(__file__).with_name("classify.swift")
    if not exe.exists() or exe.stat().st_mtime < src.stat().st_mtime:
        subprocess.run(["swiftc", "-O", str(src), "-o", str(exe)], check=True)
    lines = subprocess.run(
        [str(exe)], input="\n".join(paths) + "\n", text=True,
        stdout=subprocess.PIPE, check=True,
    ).stdout.splitlines()
    return [[(k, float(v)) for k, v in json.loads(line)] for line in lines]


def siglip2(paths, device=None):
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor

    device = device or os.environ.get("MOVERS_CLASSIFIER_DEVICE") or (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    model = AutoModel.from_pretrained(MODEL, revision=REVISION).eval().to(device)
    processor = AutoProcessor.from_pretrained(MODEL, revision=REVISION)
    labels = list(PROMPTS)
    out = []
    with torch.inference_mode():
        text = processor.tokenizer(
            [f"This is a photo of {p}." for p in PROMPTS.values()],
            padding="max_length",
            max_length=64,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        # Reuse text embeddings for the whole drive. Relative scores (including
        # background prompts) are not calibrated Apple Vision confidences.
        text_features = model.text_model(**text).pooler_output
        text_features = torch.nn.functional.normalize(text_features, dim=-1)
        for i in range(0, len(paths), 8):
            images = []
            for path in paths[i : i + 8]:
                with Image.open(path) as im:
                    images.append(im.convert("RGB"))
            inputs = processor.image_processor(images, return_tensors="pt")
            pixels = inputs.pixel_values.to(device)
            features = model.vision_model(pixel_values=pixels).pooler_output
            logits = torch.nn.functional.normalize(features, dim=-1) @ text_features.T
            scores = (logits * model.logit_scale.exp()).softmax(dim=-1).cpu().tolist()
            for row in scores:
                ranked = sorted(zip(labels, row), key=lambda item: item[1], reverse=True)
                out.append(ranked)
    return out


def classify(paths, root, backend=None):
    backend = backend or os.environ.get("MOVERS_CLASSIFIER") or (
        "vision" if sys.platform == "darwin" else "siglip2"
    )
    if backend not in ("vision", "siglip2"):
        raise ValueError(f"Unknown classifier {backend!r}; expected vision or siglip2")
    if not paths:
        return []
    return vision(paths, root) if backend == "vision" else siglip2(paths)
