# Self-Supervised Deep Learning for Fisheye Image Rectification (FE-GAN) teaser

Self-contained ~5 min 3Blue1Brown-style teaser for Chao*, Hsu*, Lee, Wang, *Self-Supervised Deep Learning for Fisheye Image
Rectification* (IEEE ICASSP 2020, pp. 2248–2252), built from the IEEE PDF (not on arXiv), the paper figures, the website
teaser GIF and the official code (github.com/ByronHsu/FEGAN, reference only).
The ground truth for the animation is `storyboard.md`; the current script is named at its top.

- Environment: `imp_tome` conda env. Manim there needs `LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7`
  (the env ships only libffi 8; system pango/gobject want LIBFFI_BASE_7.0). `render.sh` sets it.
- Render: `./render.sh l [script] [Scene ...]` (480p15) or `./render.sh h` (1080p60). With no scene list it
  renders all 10 scenes in parallel and concatenates them into `media/final/`. Default script = highest `animation_fisheye_vN.py`.
- Versioning: never overwrite `animation_fisheye_vN.py`; copy to vN+1, then update `storyboard.md`.
- No LaTeX: math is Pango markup via `tx()`; glyphs CMU Serif lacks (× − → ≈ ∑ 𝔼 ‖ ∇ ∫ ℒ ↦ ✓ …) go to Latin Modern Math
  and ✗ to DejaVu Sans via `mathify()`. No Unicode sub/superscript digits (₁ ²): use `sub()` / `sup()`. θ = `TH` (CMU italic).
- Geometry is real numpy: procedural street scene on the pinhole plane, lens family r = tan(κθ)/κ (pinhole → equidistant
  fisheye), exact rectifying flow, bilinear warps, 90° rotation checks, toy flow-loss values, strength family. All cached in
  `data/generated/` by `build_assets()` / `build_assets_v2()` (delete `done_v1` / `done_v2` to rebuild; ≈ 1 min).
  Paper figure crops: `scratch/crop_figs.py` → `data/` (from `fisheye-rectification/pdfs/figs/*_hi.png`).
  Website GIF: `data/icassp2020-result.gif`, frames `data/gif_frames/` (played with a pixel-array updater, see `live()`).
- Paper-finder records (memory bank, mind graph, BibTeX, summary, PDF, text, figures, page renders) live in
  `fisheye-rectification/`. Every number on screen must come from `summaries/chao2020selfsupervised.md`.
- The 1-minute cut lives in `teaser_1min/` (own storyboard, render.sh, script; reads `../data` read-only).
- QA: `scratch/qa_sheets.py <video dir> <out dir> [step]` builds contact sheets per scene. Keep scratch files in `scratch/`.
