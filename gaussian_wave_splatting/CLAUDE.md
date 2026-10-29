# Gaussian Wave Splatting teaser

Self-contained ~5 min 3Blue1Brown-style teaser for Choi*, Chao* et al., *Gaussian Wave Splatting for
Computer-Generated Holography* (ACM TOG / SIGGRAPH 2025, arXiv 2505.06582), built from the paper PDF + LaTeX source.
The ground truth for the animation is `storyboard.md`; the current script is named at its top.

- Environment: `imp_tome` conda env. Manim there needs `LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7`
  (the env ships only libffi 8; system pango/gobject want LIBFFI_BASE_7.0). `render.sh` sets it.
- Render: `./render.sh l [script] [Scene ...]` (480p15) or `./render.sh h` (1080p60). With no scene list it
  renders all 10 scenes in parallel and concatenates them into `media/final/`.
- Versioning: never overwrite `animation_gws_vN.py`; copy to vN+1, then update `storyboard.md`.
- No LaTeX: math is Pango markup via `tx()`; glyphs CMU Serif lacks (× − → ≈ ∑ ∏ ℱ 𝒫 𝒢 …) are routed to Latin Modern Math by `mathify()`.
- Wave optics is real numpy (angular spectrum method, closed-form GWS spectrum, Huygens sums); heavy arrays are cached in
  `data/generated/` by `build_assets()` (delete the `done_vN` flag to rebuild). Paper figure crops: `scratch/crop_figs.py` → `data/`.
- Paper-finder records (memory bank, mind graph, BibTeX, summary, PDF, text, LaTeX source) live in `gaussian-wave-splatting/`.
- QA: `scratch/qa_sheets.py <video dir> <out dir>` builds 2 s contact sheets per scene. Keep scratch files in `scratch/`.
