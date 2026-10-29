# Gaussian Wave Splatting — 1-minute teaser

Compact (~60 s) cut of the 5-min teaser in the parent folder (`../animation_gws_v4.py`, `../storyboard.md`).
Ground truth: `storyboard.md` (current script named at its top). Scenes: T1_Problem, T2_Method, T3_Payoff.

- Env: `imp_tome` conda env + `LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7` (set in `render.sh`).
- Render: `./render.sh l|h [script] [Scene ...]` → `media/final/GaussianWaveSplatting_1min_vN_<res>.mp4` (all scenes only).
- Assets are read by absolute path from `../data` (figure crops) and `../data/generated` (caches built by the 5-min
  script; this script never writes them). Helpers are copied from the 5-min script, not imported.
- No LaTeX (Pango markup via `tx()`/`mathify()`), CMU Serif, text added instantly, captions ≤ ~10 words.
- Versioning: never overwrite `animation_gaussian_wave_splatting_1min_vN.py`; copy to vN+1 and update `storyboard.md`.
- QA: `scratch/qa_sheets.py <video dir> <out dir> [step]` builds contact sheets; keep scratch files in `scratch/`.
