# Six-camera calibration

From the Model 3 dashcam frames and telemetry, estimate a metric rig, rectify each lens, and validate against disjoint frames and a visual-only drive segment. Results and limitations for this drive are in [REPORT.md](REPORT.md).

On anomastudio, run from the Reality Splat root:

```sh
# Once: C++17 toolchain, Ceres sparse CPU backend and CMake.
brew install ceres-solver cmake
# Uses the project's existing Python 3.12, NumPy, SciPy and OpenCV.
bash pipeline/calib/run.sh
```

The default root is `$HOME/dev/reality-splat`; override with `RS`. Inputs: `data/drive/{frames,cams,meta.json,timeline.npz}`. Intermediates and compiled solver: `work/calib/`. Features/pairs are cached for this drive and selection; use a fresh work root for different frames or mask geometry. The final products are `calib/rig.json`, six byte masks, `calib/report.md`, and `calib/debug/`. No neural weights or GPU are used. BA defaults to 14 threads; override with `BA_THREADS`.

1. `features.py` samples moving windows on the shared clock, masks bodywork and builds RootSIFT tracks across cameras and time. Split-only pairs restore sequential coverage when every other keyframe is removed.
2. `solve.py` initializes metric points and runs `ba.cpp`, maximizing image consistency while allowing small 6-DoF corrections to the telemetry trajectory. The same solver fits two independent image subsets.
3. `ground.py` and `align.py` estimate road height from a photometric sweep, vehicle heading from translation, lateral centre from the side mounts, and axle location from turns subject to the rear overhang bound.
4. `scale.py` freezes the rig, discards telemetry poses and speeds, and reconstructs an unseen departure twice from arbitrary initial spacings. Telemetry is only the comparison standard, not a scale input to those solves.
5. `deliver.py` writes the PLAN schema, native-resolution fit and uncertainty tables, masked rectifications, and cross-lens overlays with the target lens excluded from triangulation.
