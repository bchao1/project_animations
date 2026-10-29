# Random-phase Wave Splatting teaser

Self-contained ~5 min 3Blue1Brown-style teaser for Chao*, Yang*, Choi, Gopakumar, Koiso, Wetzstein, *Random-phase Wave
Splatting of Translucent Primitives for Computer-generated Holography* (arXiv 2508.17480, 2025; follow-up of Gaussian Wave
Splatting, sibling project `../gaussian_wave_splatting/`), built from the paper PDF + LaTeX source.
The ground truth for the animation is `storyboard.md`; the current script is named at its top.

- Environment: `imp_tome` conda env. Manim there needs `LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7`
  (the env ships only libffi 8; system pango/gobject want LIBFFI_BASE_7.0). `render.sh` sets it.
- Render: `./render.sh l [script] [Scene ...]` (480p15) or `./render.sh h` (1080p60). With no scene list it
  renders all 10 scenes in parallel and concatenates them into `media/final/`. Default script = highest `animation_rpws_vN.py`.
- Versioning: never overwrite `animation_rpws_vN.py`; copy to vN+1, then update `storyboard.md`.
- No LaTeX: math is Pango markup via `tx()`; glyphs CMU Serif lacks (× − → ≈ ∑ ∏ ℱ 𝒫 𝔼 ∠ ✓ …) go to Latin Modern Math
  and ✗ to DejaVu Sans via `mathify()`. No Unicode sub/superscript digits (₁ ²): use `sub()` / `sup()`.
- Wave optics is real numpy (angular spectrum method; RPWS back-to-front compositing with √ weights and √(1 − α) masks;
  GWS smooth-phase alpha wave blending; the 'GWS with √' ablation; time multiplexing over 24 frames). All toy results are
  cached in `data/generated/` by `build_assets()` (delete the `done_vN` flag to rebuild; ≈ 30 s).
  Paper figure crops: `scratch/crop_figs.py` → `data/` (from `random-phase-wave-splatting/pdfs/figs/*_hi.png`).
- Paper-finder records (memory bank, mind graph, BibTeX, summary, PDF, text, LaTeX source, v1 HTML text) live in
  `random-phase-wave-splatting/`. Every number on screen must come from `summaries/chao2025random.md`.
- The 1-minute cut lives in `teaser_1min/` (own storyboard, render.sh, script; reads `../data` read-only).
- QA: `scratch/qa_sheets.py <video dir> <out dir> [step]` builds contact sheets per scene. Keep scratch files in `scratch/`.
