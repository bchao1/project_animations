# MPI holography teaser

Self-contained ~5 min 3Blue1Brown-style teaser for Chao, Seyb, Matsuda, Cossairt, Zhou, Lanman, Wetzstein, Kuo, Jang,
*Fast Wave-optics Rendering of Multiplane Images for 3D Holographic Displays* (arXiv 2607.19731, 2026; Stanford · Meta;
follow-up of RPWS, sibling project `../random_phase_wave_splatting/`), built from the paper PDF + LaTeX source.
The ground truth for the animation is `storyboard.md`; the current script is named at its top.

- Environment: `imp_tome` conda env. Manim there needs `LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7`
  (the env ships only libffi 8; system pango/gobject want LIBFFI_BASE_7.0). `render.sh` sets it.
- Render: `./render.sh l [script] [Scene ...]` (480p15) or `./render.sh h` (1080p60). With no scene list it
  renders all 10 scenes in parallel and concatenates them into `media/final/MPIHolography_<script>_<res>.mp4`.
  Default script = highest `animation_mpi_vN.py`.
- Versioning: never overwrite `animation_mpi_vN.py`; copy to vN+1 (patch scripts in `scratch/patch_vN.py`), then update `storyboard.md`.
- No LaTeX: math is Pango markup via `tx()`; glyphs CMU Serif lacks (× − → ≈ ∑ ℱ 𝒫 𝔼 △ ← ✓ …) go to Latin Modern Math
  and ✗ to DejaVu Sans via `mathify()`. No Unicode sub/superscript digits (₁ ²): use `sub()` / `sup()`. Long lines: `fit()`.
- Wave optics is real numpy (angular spectrum method, 3 wavelengths; 4-layer toy MPI composited back to front with paper
  Eq. 2; RGBD depth-slice baseline; 24-frame time multiplexing; pupil-sampled light fields and focal stacks). All toy results
  are cached in `data/generated/` by `build_assets()` (delete `done_v1` to rebuild; ≈ 3 min).
- Paper figure crops: `scratch/crop_figs.py` → `data/` (from `mpi-holography/pdfs/figs/*_hi.png`). Homepage teaser videos:
  `data/videos/*.mp4` → frames `data/clips/{lf,fs}.npy` (960×640), played by `video()` (live ImageMobject, ping-pong).
  Cairo cannot shear ImageMobjects: pseudo-3D layer sheets are pre-sheared PNGs (`layer*_sh.png`, `sheet()`).
- Paper-finder records (memory bank, mind graph, BibTeX, summary, PDF, text, LaTeX source, figure rasters) live in
  `mpi-holography/`. Every number on screen must come from `summaries/chao2026fast.md` (Fig. 3 values were read from the
  figure's vector data; table there).
- The 1-minute cut lives in `teaser_1min/` (own storyboard, render.sh, script; reads `../data` read-only).
- QA: `scratch/qa_sheets.py <video dir> <out dir> [step]` builds contact sheets per scene. Keep scratch files in `scratch/`.
