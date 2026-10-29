# Coded-aperture TDM light field display — 1-minute teaser

Compact (~60 s) cut of the 5-min teaser in the parent folder (`../animation_tdmlf_v3.py`, `../storyboard.md`).
Ground truth: `storyboard.md` (current script named at its top). Scenes: T1_Hook, T2_Method, T3_Payoff.

- Env: `imp_tome` conda env + `LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7` (set in `render.sh`).
- Render: `./render.sh l|h [script] [Scene ...]` → `media/final/CodedApertureLightFieldDisplay_1min_vN_<res>.mp4` (all scenes only).
- Assets are read by absolute path from `../data` (figure crops, GIF frames, Fig. 9 apertures) and `../data/generated` (toy caches
  built by the 5-min script, flags `done_v1` + `done_v2`; this script never writes them). Helpers are copied from the 5-min
  script, not imported.
- No LaTeX (Pango markup via `tx()`/`mathify()`), CMU Serif, text added instantly, captions ≤ ~15 words, ≤ 1 equation.
- Versioning: never overwrite `animation_tdmlf_1min_vN.py`; copy to vN+1 and update `storyboard.md`.
- QA: `scratch/qa_sheets.py <video dir> <out dir> [step]` builds contact sheets (globs S*/T* scenes); keep scratch files in `scratch/`.
