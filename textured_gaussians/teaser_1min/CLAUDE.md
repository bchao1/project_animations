# Textured Gaussians — 1-minute teaser

Compact (~60 s) cut of the 5-min teaser in the parent folder (`../animation_textured_gaussians_v3.py`). Ground truth:
`storyboard.md` (current script named at its top). Never edit anything outside this folder.

- Env: `imp_tome` conda env + `LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7` (set in `render.sh`).
- Render: `./render.sh l [script]` (480p15) / `./render.sh h [script]` (1080p60); 3 scenes T1_Hook, T2_Method, T3_Payoff
  rendered in parallel, concatenated to `media/final/TexturedGaussians_1min_vN_<res>.mp4`.
- Assets are read by absolute path from `../data/` (paper crops) and `../data/generated/` (2D toy fit; numbers in
  `toy_results.json`). Do not regenerate them.
- No LaTeX: Pango markup via `tx()`; `mathify()` routes missing glyphs to Latin Modern Math. Helpers are copied from the
  5-min script, not imported.
- Versioning: fixes go into a new `animation_textured_gaussians_1min_vN.py`; update `storyboard.md`.
- QA: `python scratch/qa_sheets.py media/videos/<script>/480p15 scratch/qaN [step]` → contact sheets.
