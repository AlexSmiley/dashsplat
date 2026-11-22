# How the project was run

dashsplat was built in about a day by a team of AI coding agents that Ryan Vogel directed. Two machines did the
work: **basebook**, his working Mac, held the git checkout, where the agents edited code and looked at results, and
**anomastudio** ("the Studio"), a Mac Studio with an M5 Ultra and 96 GB, ran everything heavy from its own copy in
`~/dev/reality-splat`. Code moved between them with `ferry`, his small tool for copying files and running commands
on his own machines. Rented NVIDIA GPUs came in on the second day for the port in `pipeline/cuda/`.

The file below was the repository's `AGENTS.md`: the standing rules every agent read before working. It is kept
here as a record of how the work was organised. It does not apply to your machine, and nothing in the code reads it.

---

# Reality Splat: rules for agents

## Keep basebook free

basebook (this Mac, where the repo lives) is Ryan's working machine. Heavy work
here makes everything else on it unusable. Run anything heavy on the Mac Studio
(`anomastudio`, `~/dev/reality-splat`) or a rented GPU, not here.

Heavy means: model inference or training, depth, masks, splat fitting, the image
model, video encoding or transcoding, frame-by-frame captures (headless Chrome,
4K renders), fframes or other film renders, large builds, and anything that
runs for more than a few minutes on several cores.

- Ship code with `ferry cp`, run with `ferry sh anomastudio '...'` (long jobs
  with `--detach`), and bring back only stills, contact sheets, small clips and
  final deliverables.
- On basebook: editing, git, small scripts, viewer builds, one browser tab for a
  spot check, and controlling remote jobs. If something heavy truly cannot run
  on the Studio, run one job at a time under `nice`, never in parallel, and say
  so in your report.
- Before starting a job here, check `uptime`. If the load average is above
  about 12, do not add to it.
- Disk here is tight. Keep large files (captures, renders, scenes, model
  weights, recordings) on the Studio, keep this checkout's untracked data under
  a few GB, and delete temporary recordings when done.
- The Studio is shared too: one GPU job at a time, low priority for encodes,
  and check what is already running before starting.
