# Contributing

The most useful thing you can send is a result from a different car. Open an issue with the model, year and
hardware (HW3, HW4), the firmware, how many seconds you ran, which stages finished, the manifest from
`work/drives/<name>/manifest.json` (stage times and the rig check's numbers) and, if you are happy to share them, a few
stills from the viewer. Please do not post raw footage, GPS tracks or anything that shows other people's faces or
number plates.

The most useful code contribution is a calibration preset for another model: `calib/rig.json` plus the body masks
in `calib/masks/`, with the drive length and the held-out check from `pipeline/calib` (see
[pipeline/calib/REPORT.md](pipeline/calib/REPORT.md) for what those numbers mean). Everything in this repository was
built from one Model 3, so a second car's calibration tells us more than any other change.
