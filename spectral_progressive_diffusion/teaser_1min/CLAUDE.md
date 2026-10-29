# Spectral Progressive Diffusion — 1-minute teaser

Compact (~63 s) cut of the 5-min teaser in the parent folder (`../animation_spectral_progressive_v4.py`). Ground truth
is `storyboard.md`; the current script is named at its top. Never edit anything outside this folder.

- Env: `imp_tome` conda env + `LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7` (both set by `render.sh`).
- Render: `./render.sh l|h [script] [Scene ...]`; with no scene list it renders T1_Hook, T2_Insight, T3_Method,
  T4_Payoff in parallel (logs in `scratch/logs/`) and concatenates to
  `media/final/SpectralProgressiveDiffusion_1min_vN_<res>.mp4`.
- Helpers are copied (not imported) from the 5-min script; images/corgi are read from `../data/` by absolute path.
- No LaTeX (Pango markup via `tx()`), CMU Serif, text added instantly, captions ≥ len/15 + 1 s.
- Numbers on screen come only from the paper (see `../spectral-progressive-diffusion/summaries/xiao2026spectral.md`).
- New fix round → new `animation_spectral_progressive_diffusion_1min_vN.py` + update `storyboard.md`.
- QA: `scratch/qa_sheets.py <video dir> <out dir> [step]` builds contact sheets.
