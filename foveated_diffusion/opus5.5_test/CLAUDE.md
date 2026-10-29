# Foveated Diffusion explainer (opus5.5_test)

Self-contained: everything here was built only from the paper PDF (`foveated-diffusion/pdfs/`).
The ground truth for the animation is `storyboard.md`; the current script is named at its top.

- Environment: `imp_tome` conda env. Manim there needs `LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7`
  (the env ships only libffi 8; system pango/gobject want LIBFFI_BASE_7.0). `render.sh` sets it.
- Render: `./render.sh l [script] [Scene ...]` (480p15) or `./render.sh h` (1080p60). With no scene list it
  renders all 11 scenes in parallel and concatenates them into `media/final/`.
- No LaTeX: math is Pango markup via `tx()`; glyphs CMU Serif lacks (× ⊙ − → ≈ …) are routed to Latin Modern Math by `mathify()`.
- Paper-finder records (memory bank, mind graph, BibTeX, ingest notes) live in `foveated-diffusion/`.
- QA: `scratch/qa_sheets.py <video dir> <out dir>` builds 2 s contact sheets per scene.
