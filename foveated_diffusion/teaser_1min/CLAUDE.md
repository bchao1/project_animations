# Foveated Diffusion — 1-minute teaser (teaser_1min)

Compact (~1 min) cut of `../teaser_5min/` (read-only source; do not edit it). Assets are read by absolute path
from `../teaser_5min/data/`; the only new asset (`data/generated/gaze_C.png`) is built on import.
The ground truth is `storyboard.md`; the current script is named at its top. Never overwrite a script version.

- Env: `imp_tome` conda env + `LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7` (set by `render.sh`).
- Render: `./render.sh l [script] [Scene ...]` (480p15) or `./render.sh h [script]` (1080p60). With no scene list it
  renders the 3 scenes in parallel and writes `media/final/FoveatedDiffusion_1min_<vN>_<res>.mp4`.
- No LaTeX: Pango markup via `tx()`/`mathify()`; CMU Serif; text appears instantly; HR `#448AFF`, LR `#00C853`.
- QA: `python scratch/qa_sheets.py media/videos/<script>/480p15 scratch/qa_vN [step_s]` → contact sheets per scene.
