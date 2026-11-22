# Bird's-eye road texture refinement test

Does a small restoration model sharpen the 2 cm/px road mosaics without moving paint? Results and verdict: [REPORT.md](REPORT.md).

On Studio, from `$RS`:

```sh
PYTHONPATH=. envs/genfill/bin/python      -m pipeline.genfill.ground.run_difix    # Difix noref/ref x t99/t199
PYTHONPATH=. envs/genfill-edit/bin/python -m pipeline.genfill.ground.run_esrgan   # Real-ESRGAN x2plus
PYTHONPATH=. envs/genfill/bin/python      -m pipeline.genfill.ground.evaluate     # metrics + sheets -> work/genfill/ground/report
PYTHONPATH=. envs/genfill-edit/bin/python -m pipeline.genfill.ground.run_chase    # 8 chase renders sheet
```

- `common.py`: patch crops (1536 px = 30.7 m squares of `best.jpg` + `road-mask.png`), overlapped tiling with a linear feather, paint metrics.
- `locate.py`: draws the offline ego track with time labels on a mosaic, used to pick the crops.
- The bike-lane window (106–118 s, offline poses) was built with `accumulate_window` into `$RS/work/genfill/ground/mosaic/bike/` (24 s).
- `envs/genfill-edit` (uv, Python 3.12): torch 2.15.0, opencv-python-headless, spandrel 0.4.2, scipy (no basicsr needed). Weights: `$RS/third_party/Real-ESRGAN/weights/RealESRGAN_x2plus.pth` (BSD-3, GitHub release v0.2.1).
- Outputs: `$RS/work/genfill/ground/out/<patch>/<setting>.png` (2 cm, same pixel grid as the crop); `esrgan-x2-native-1cm.png` is the 1 cm output.
