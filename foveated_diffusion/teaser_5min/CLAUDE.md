# Foveated Diffusion — 5-minute teaser (teaser_5min)

Condensed (~5 min) cut of the ≈9 min 20 s explainer in `../opus5.5_test/` (read-only source; do not edit it).
Self-contained: `data/` (paper figure crops; `data/generated/` is rebuilt on import), `foveated-diffusion/`
(paper-finder records + PDF/text), `scratch/qa_sheets.py`.
The ground truth is `storyboard.md`; the current script is named at its top. Never overwrite a script version.

- Env: `imp_tome` conda env + `LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7` (set by `render.sh`).
- Render: `./render.sh l [script] [Scene ...]` (480p15) or `./render.sh h [script]` (1080p60). With no scene
  list it renders all 10 scenes in parallel and writes `media/final/FoveatedDiffusion_5min_<vN>_<res>.mp4`.
  If `data/generated/` is missing, import the script once first (parallel renders would race on building it).
- No LaTeX: Pango markup via `tx()`/`mathify()`; CMU Serif; text appears instantly; HR `#448AFF`, LR `#00C853`.
- QA: `python scratch/qa_sheets.py media/videos/<script>/480p15 scratch/qa_vN` → 2 s contact sheets per scene.
