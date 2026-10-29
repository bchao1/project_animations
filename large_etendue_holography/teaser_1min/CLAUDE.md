# Large Étendue Holography — 1-minute teaser

Compact (~57 s) cut of the 5-min explainer in `..` (`../animation_large_etendue_v3.py`). `storyboard.md` is the
ground truth; the current script is named at its top. Never overwrite a version: copy to `_vN+1.py` per fix round.

- Env: `imp_tome` conda env + `export LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7` (set in `render.sh`).
- Render: `./render.sh l|h [script] [Scene ...]` → scenes T1_Problem, T2_Method, T3_Payoff in parallel, concatenated to
  `media/final/LargeEtendueHolography_1min_vN_<res>.mp4`. Logs in `scratch/logs/`.
- Assets are read by absolute path from `../data` and `../data/generated` (cached by the 5-min script; not regenerated).
- No LaTeX (Pango markup via `tx()`), CMU Serif, text added instantly; toy visuals tagged "toy simulation";
  every number comes from the paper (`../large-etendue-holography/summaries/chao2024etendue.md`).
- QA: `python scratch/qa_sheets.py <video dir> <out dir> 1.0` → 1 s contact sheets per scene.
