# Coded-aperture TDM light field display teaser

Self-contained ~5 min 3Blue1Brown-style teaser for Chao*, Liu*, Chen, *Time-Division Multiplexing Light Field Display With
Learned Coded Aperture* (IEEE TIP 32, 350–363, 2023; arXiv 2107.06205; National Taiwan University), built from the arXiv v3 PDF
(the IEEE version is paywalled; arXiv has no LaTeX source; no code or video exists).
The ground truth for the animation is `storyboard.md`; the current script is named at its top.

- Environment: `imp_tome` conda env. Manim there needs `LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7`. `render.sh` sets it.
- Render: `./render.sh l [script] [Scene ...]` (480p15) or `./render.sh h` (1080p60). With no scene list it renders all 10
  scenes in parallel and concatenates them into `media/final/CodedApertureLightFieldDisplay_<script>_<res>.mp4`.
  Default script = highest `animation_tdmlf_vN.py`.
- Versioning: never overwrite `animation_tdmlf_vN.py`; copy to vN+1, then update `storyboard.md`.
- No LaTeX: math is Pango markup via `tx()`; glyphs CMU Serif lacks (× − → ≈ ∑ ℱ ∇ 𝒲 ℒ ⊙ ∗ ✓ …) go to Latin Modern Math and ✗
  to DejaVu Sans via `mathify()`. No Unicode sub/superscript digits: use `sub()` / `sup()`. P̂ = italic P + U+0302.
- Toy optics are real computation, cached in `data/generated/` by `build_assets()` (flag `done_v1`, ≈ 5 min, torch on CPU +
  numpy; delete the flag to rebuild) and `build_assets_v2()` (flag `done_v2`): 9 × 9 layered light field, ray-optics
  TDM / CTDM image formation (paper Eqs. 8–10), a per-scene optimisation of 4 images + 4 apertures with the paper's
  focus-weighted loss (Eqs. 12–15, β = 2), and Fourier-optics PSFs of open / corner / Fig. 9 apertures. Prototypes in `scratch/`.
- Paper figures: `scratch/crop_figs.py` exports the embedded rasters from the PDF into `data/fig*.png`;
  `data/fig9_apertures_9x9.npy` = the digitised Fig. 9 binary apertures; website GIF in `data/tip2021.gif`, `data/gif_frames/`
  (frame 0 = baseline, frame 1 = CTDM-LFD, Fig. 12 agapanthus, near focus).
- Paper-finder records (memory bank, mind graph, BibTeX, summary, PDF, text, page renders) live in `coded-aperture-lf-display/`.
  Every number on screen must come from `summaries/chao2023time.md`.
- The 1-minute cut lives in `teaser_1min/` (own storyboard, render.sh, script; reads `../data` read-only).
- QA: `scratch/qa_sheets.py <video dir> <out dir> [step]` builds contact sheets per scene. Keep scratch files in `scratch/`.
