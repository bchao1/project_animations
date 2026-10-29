# RPWS explainer — 3 × 1 min

Accompanies the Nature-format manuscript (`../data/main_old_nature_format.pdf`); never frame RPWS as a delta to GWS on screen.
Three standalone ≈ 1-min 3Blue1Brown-style explainers (+ a 3-min concatenation) for *Random-phase Wave Splatting*
(arXiv 2508.17480): 1 · how RPWS differs from GWS (the math), 2 · why random-phase alpha blending works,
3 · why back-to-front compositing works. Requested 2026-10-03 to replace the 5-min teaser in `..`, which only glossed
over these points. Goal: make RPWS look very novel by following the paper's exact math, with toy experiments that
replicate the paper's own toy figures. Look = the multisource explainer (`../../large_etendue_holography/`).
Ground truth for the animation: `storyboard.md` (current script named at its top).

- Sources: main paper LaTeX `../random-phase-wave-splatting/pdfs/src/` and the **supplement**
  `../data/Random_phase_Wave_Splatting (2).pdf` (text `../random-phase-wave-splatting/pdfs/rpws_supp_clean.txt`;
  Algs. 1–6, S3, S4, S6.1). Notes with every number: `../random-phase-wave-splatting/summaries/chao2025random.md`.
- Env: `imp_tome` + `LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7` (set by `render.sh`). Sims use torch on GPU
  (`CUDA_VISIBLE_DEVICES`, default 3).
- `sims.py` (≈ 2.5 min) writes the toy experiments to `data/generated/` (flag `done_sims`); the script's
  `build_assets()` turns them into display arrays (flag `done_v1`). Delete a flag to rebuild.
- Render: `./render.sh l|h [script] [Scene ...]`. Without a scene list it renders S00_Title, P1–P3 in parallel and
  writes `media/final/RPWS_Part{1,2,3}_<script>_<res>.mp4` and `RPWS_3min_<script>_<res>.mp4` (S00 + P1–P3).
- Versioning: never overwrite `animation_rpws_explainer_vN.py`; copy to vN+1 and update `storyboard.md`.
- No LaTeX: Pango markup via `tx()`. Do **not** route √ to Latin Modern Math (its radical renders oversized);
  CMU Serif's fallback is fine. No Unicode sub/superscript digits in `tx()` text (captions convert ₁ ₂ ²).
- `FrameSeq(..., tracker=kt)`: a ValueTracker is falsy (zero submobjects) — test `is not None`, never `or`.
- Hold every caption (`self.hold()`) before a visual change that follows it.
- Result videos: authors talk-deck clips in `data/deck/` (see storyboard "Talk deck review"); `build_videos()` caches frames.
- QA: `python scratch/qa_sheets.py media/videos/<script>/480p15 scratch/qa_vN 2`.
