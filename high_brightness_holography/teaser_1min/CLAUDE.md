# High-Brightness Holographic Projection — 1-minute teaser

Compact (~60 s) cut of the 5-min teaser in the parent folder (`../animation_hbh_v4.py`, `../storyboard.md`).
Ground truth: `storyboard.md` (current script named at its top). Scenes: T1_Hook, T2_Method, T3_Payoff.

- Env: `imp_tome` conda env + `LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7` (set in `render.sh`).
- Render: `./render.sh l|h [script] [Scene ...]` → `media/final/HighBrightnessHolography_1min_vN_<res>.mp4` (all scenes only).
- Assets are read by absolute path from `../data` (supplement crops, `video/<name>/` frames) and `../data/generated` (toy caches
  from `../precompute_toy.py` + the 5-min script's `build_assets()`, flag `done_v1`; this script never writes them).
  Helpers are copied from the 5-min script, not imported.
- No LaTeX (Pango markup via `tx()`/`mathify()`), CMU Serif, text added instantly, captions ≤ ~15 words, 1 equation.
- Numbers: paper numbers from `../high-brightness-holography/summaries/chao2023high.md`; toy numbers labelled "toy".
- Versioning: never overwrite `animation_hbh_1min_vN.py`; copy to vN+1 and update `storyboard.md`.
- QA: `scratch/qa_sheets.py <video dir> <out dir> [step]` builds contact sheets; keep scratch files in `scratch/`.
