# Spectral Progressive Diffusion teaser

Self-contained 3b1b-style teaser for arXiv 2605.18736, built only from the paper (PDF + LaTeX source in
`spectral-progressive-diffusion/pdfs/`). The ground truth is `storyboard.md`; the current script is named at its top.

- Environment: `imp_tome` conda env. Manim there needs `LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7`
  (the env ships only libffi 8; system pango/gobject want LIBFFI_BASE_7.0). `render.sh` sets it.
- Render: `./render.sh l [script] [Scene ...]` (480p15) or `./render.sh h` (1080p60). With no scene list it renders all
  10 scenes in parallel (logs in `scratch/logs/`) and concatenates them into `media/final/`.
- No LaTeX: math is Pango markup via `tx()`; glyphs CMU Serif lacks (× − → ≈ ∝ …) go to Latin Modern Math via `mathify()`.
  Avoid √ (sits below the baseline) — write `(…)^1/2`.
- Spectra, noise sweeps and the DCT expansion demo are computed at import from `data/corgi_clean.png`; paper figures in
  `data/` are crops of the arXiv-source rasters. Every on-screen number must come from the paper (see the summary).
- Versioning: new fix round → new `animation_spectral_progressive_vN.py` (never overwrite) + update `storyboard.md`.
- Paper-finder records (memory bank, mind graph, BibTeX, summary): `spectral-progressive-diffusion/`.
- QA: `scratch/qa_sheets.py <video dir> <out dir>` builds 2 s contact sheets per scene. Keep scratch files in `scratch/`.
