# Masks

Sky, road and movable-object instance masks for the keyframes in `$RS/depth/keyframes.json`, on the same grid as
the depth maps. Usage, output formats and the shared environment are in [../depth/README.md](../depth/README.md);
evaluation is in [../depth/REPORT.md](../depth/REPORT.md).

```sh
.venv/bin/python pipeline/masks/run_masks.py --cams all            # segmentation + track ids (resumable)
.venv/bin/python pipeline/masks/run_masks.py --cams all --motion   # moving / parked flags (after run_depth.py)
.venv/bin/python pipeline/masks/viz_masks.py front 4001 out.jpg    # overlay for one keyframe
```

- **Model:** Mask2Former Swin-L, Cityscapes panoptic (`facebook/mask2former-swin-large-cityscapes-panoptic`) via
  Hugging Face transformers on MPS, about 0.1 s per 724x469 image. Licence: MIT code; the weights are trained on
  Cityscapes, whose terms are non-commercial.
- **Classes:** sky (`masks/sky`), road (`masks/road`, extra), and the movable things person, rider, car, truck, bus,
  motorcycle, bicycle (`masks/dyn`, uint16 track id, 0 = background). Cityscapes "train" is dropped: the model calls
  the drive-through building's wall a train (`--prune` removes such instances from existing outputs).
- **Tracks:** ids are per camera and persist across keyframes by greedy mask IoU (> 0.15) within the groups vehicle /
  person / two-wheeler.
- **Moving vs parked (`--motion`):** each instance's nearest-half pixels are lifted to the world with its depth, the
  rig and the trajectory (cached per frame in `work/depth/cache/`). Track fragments are linked into objects when one
  starts within 1.5 s of another ending (same class group) and the first's extrapolated position lands within
  max(1.5 m, 8 % of range) of the second's start. An object is decided only when seen >= 3 times over >= 1 s:
  `moving = true` when its fitted speed is > 2 m/s and its displacement beats the noise (max of 1 m, twice the fit
  scatter, 5 % of range); `false` when it stays within the noise while the car moved >= 2 m (or over >= 3 s);
  otherwise, and for implausible speeds (> 35 m/s), `null`. Counts are in `tracks.json` -> `counts`.
- Pixels outside the calibration mask are 0 in every mask. On the rear fisheye the sky label misses parts of the
  periphery (the model was trained on pinhole images); depth beyond 150 m is dropped as a backstop.
