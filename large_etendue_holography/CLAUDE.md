# Large Étendue 3D Holography explainer

Self-contained teaser for Chao et al., SIGGRAPH Asia 2024 (arXiv 2409.03143), built from the paper PDF + arXiv LaTeX source
in `large-etendue-holography/pdfs/`. The ground truth for the animation is `storyboard.md`; the current script is named at its top.

- Environment: `imp_tome` conda env. Manim there needs `LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7`
  (the env ships only libffi 8; system pango/gobject want LIBFFI_BASE_7.0). `render.sh` sets it.
- Render: `./render.sh l [script] [Scene ...]` (480p15) or `./render.sh h` (1080p60). With no scene list it renders all
  10 scenes in parallel (logs in `scratch/logs/`) and concatenates them into `media/final/`.
- No LaTeX: math is Pango markup via `tx()`; glyphs CMU Serif lacks (× → ≈ − ℱ ℋ 𝒫 𝒰 …) go to Latin Modern Math via `mathify()`.
- Computed visuals: `precompute_toy_cgh.py` (torch, GPU) → `data/generated/toy_cgh.npz`; `build_assets()` caches PNGs in
  `data/generated/` (delete `done_v1` to rebuild). Toy results are labelled "toy simulation"; all numbers on screen come from the paper.
- Paper-finder records (memory bank, mind graph, BibTeX, summary) live in `large-etendue-holography/`.
- QA: `scratch/qa_sheets.py <video dir> <out dir>` builds 2 s contact sheets per scene. Save each fix round as a new version file.
