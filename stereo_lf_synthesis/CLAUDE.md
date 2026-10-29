# Stereo → light field synthesis teaser (ICIP 2021)

Self-contained ~5 min 3Blue1Brown-style teaser (+ 1-min cut in `teaser_1min/`) for Chao, Liu, Chen, *Robust Light Field
Synthesis From Stereo Images With Left-Right Geometric Consistency* (IEEE ICIP 2021, pp. 1844–1848, National Taiwan University).
Not on arXiv; IEEE Xplore blocks scripted downloads, so the PDF is the author copy from the `bchao1/old-website` GitHub repo.
The ground truth for the animation is `storyboard.md`; the current script is named at its top.

- Environment: `imp_tome` conda env. Manim there needs `LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7`. `render.sh` sets it.
- Render: `./render.sh l [script] [Scene ...]` (480p15) or `./render.sh h` (1080p60). With no scene list it renders all 11
  scenes in parallel and concatenates them into `media/final/StereoLightFieldSynthesis_<script>_<res>.mp4`.
  Default script = highest `animation_slf_vN.py`.
- Versioning: never overwrite a rendered `animation_slf_vN.py`; copy to vN+1, then update `storyboard.md`.
- Shared code: `slf_common.py` (style, `tx()`/`mathify()`, `Base` scene with captions, `ViewGrid`, caches). Both the 5-min
  and the 1-min scripts import it; change it with care (re-render both).
- No LaTeX: math is Pango markup via `tx()`; glyphs CMU Serif lacks (× − → ≈ ‖ ∇ ℒ · …) go to Latin Modern Math and ✓ ✗ to
  DejaVu Sans via `mathify()`. No Unicode sub/superscript digits: use `sub()` / `sup()`.
- Data: `data/hci/full_data/` = HCI 4D LF benchmark (24 scenes, CC BY-NC-SA 4.0), fetched with ranged reads by
  `scratch/remote_zip.py` (the full zip is 3.4 GB). Views (i, j) = HCI camera 9 i + j; S_l = V(4, 0), S_r = V(4, 8).
  `data/gif_frames/` = website teaser GIF = the paper's synthesized 9 × 9 origami light field (81 raster-ordered views; a
  resized, palette-quantised copy — use for display / refocus only, not for error numbers).
  Paper figures: `scratch/crop_figs.py` → `data/fig1_pipeline.png`, `fig2_error.png`, `fig4_refocus.png`, `data/fig3/`.
- Re-run: `scratch/repro/train_hci.py` trains the authors' `DisparityNet` + `LFRefineNetV2` (cloned read-only in
  `scratch/lf-flow/`) with the paper's losses on the HCI 20 / 4 split (GPU, ≈ 35 min for 8000 steps); outputs in
  `data/generated/repro/` (ckpt, log.json, metrics.json, origami tensors, training snapshots). Everything labelled
  "our re-run" on screen comes from there; paper numbers come from `stereo-lf-synthesis/summaries/chao2021robust.md`.
- Caches: `build_assets()` (flag `data/generated/done_v1`, ≈ 5 min, refocus stacks / EPI / PSAS shuffle) and
  `build_repro_assets()` (flag `done_v2`, needs the re-run). Delete a flag to rebuild.
- Paper-finder records (memory bank, mind graph, BibTeX, summary, PDFs, text, page renders) live in `stereo-lf-synthesis/`.
- QA: `scratch/qa_sheets.py <video dir> <out dir> [step]` builds contact sheets per scene. Keep scratch files in `scratch/`.
