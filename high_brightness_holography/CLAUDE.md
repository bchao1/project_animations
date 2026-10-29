# High-Brightness Holographic Projection teaser

Self-contained ~5 min 3Blue1Brown-style teaser for Chao, Gopakumar, Choi, Wetzstein, *High-brightness holographic projection*,
Optics Letters 48(15), 4041 (2023), built from the main text (Wayback snapshot; Optica is captcha-walled), Supplement 1 and
Visualization 1 (figshare), and the website clip. Main-paper Figs. 1–5 were NOT obtainable; only supplement figures are used.
The ground truth for the animation is `storyboard.md`; the current script is named at its top.

- Environment: `imp_tome` conda env. Manim there needs `LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7`
  (the env ships only libffi 8; system pango/gobject want LIBFFI_BASE_7.0). `render.sh` sets it.
- Render: `./render.sh l [script] [Scene ...]` (480p15) or `./render.sh h` (1080p60). With no scene list it renders all
  10 scenes in parallel and concatenates them into `media/final/HighBrightnessHolography_<script>_<res>.mp4`.
  Default script = highest `animation_hbh_vN.py`.
- Versioning: never overwrite `animation_hbh_vN.py`; copy to vN+1, then update `storyboard.md`.
- No LaTeX: math is Pango markup via `tx()`; glyphs CMU Serif lacks (× − → ≈ ∑ ℒ ∂ ⇒ ∞ …) go to Latin Modern Math and ✗ to
  DejaVu Sans via `mathify()`. No Unicode sub/superscript digits (₁ ² ⁻¹): use `sub()` / `sup()`.
- Toy wave optics: `precompute_toy.py` (torch autograd SGD through the angular spectrum method; picks the GPU with most free
  memory, else CPU; ≈ 4 min) → `data/generated/toy_*.npz`. `build_assets()` in the script turns them into display arrays
  (delete `data/generated/done_v1` to rebuild). Everything simulated is labelled "toy simulation" on screen.
- Paper assets: `scratch/crop_figs.py` → `data/*.png` (supplement figure crops from
  `high-brightness-holography/pdfs/figs/*_hi.png`) and `data/video/<name>/%04d.jpg` (30 fps frames of `data/video/*.mp4`).
  Videos play via the `Clip` class (ImageMobject whose pixels advance with scene time); add it with `self.add`, not FadeIn.
- Paper-finder records (memory bank, mind graph, BibTeX, summary, supplement PDF + text, main-text HTML/text, figure rasters)
  live in `high-brightness-holography/`. Every number on screen must come from `summaries/chao2023high.md`
  (paper numbers) or from the toy list in `storyboard.md` (toy numbers, labelled as such).
- The 1-minute cut lives in `teaser_1min/` (own storyboard, render.sh, script; reads `../data` read-only).
- QA: `scratch/qa_sheets.py <video dir> <out dir> [step]` builds contact sheets per scene. Keep scratch files in `scratch/`.
