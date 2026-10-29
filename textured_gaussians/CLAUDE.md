# Textured Gaussians teaser (CVPR 2025, arXiv 2411.18625)

Self-contained 3b1b-style explainer built from the paper (PDF + LaTeX source in `textured-gaussians/pdfs/`).
The ground truth for the animation is `storyboard.md`; the current script is named at its top.

- Environment: `imp_tome` conda env. Manim there needs `LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7`
  (the env ships only libffi 8; system pango/gobject want LIBFFI_BASE_7.0). `render.sh` sets it.
- Assets: `python build_assets.py` crops paper figures into `data/` and runs our 2D toy fit (torch) into
  `data/generated/` (numbers in `toy_results.json`). `render.sh` runs it if the json is missing.
- Render: `./render.sh l [script] [Scene ...]` (480p15) or `./render.sh h` (1080p60). With no scene list it renders all
  11 scenes in parallel (logs in `scratch/logs/`) and concatenates them into `media/final/`.
- No LaTeX: math is Pango markup via `tx()`; glyphs CMU Serif lacks are routed to Latin Modern Math by `mathify()`.
- Versioning: new fixes go into a new `animation_textured_gaussians_vN.py`; never overwrite; update `storyboard.md`.
- Paper-finder records (memory bank, mind graph, BibTeX, ingest summary) live in `textured-gaussians/`.
- QA: `scratch/qa_sheets.py <video dir> <out dir>` builds 2 s contact sheets per scene.
