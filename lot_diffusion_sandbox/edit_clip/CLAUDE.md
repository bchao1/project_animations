# LoT multi-turn in-context editing clip (edit_clip/)

Started 2026-10-08. One 20-s 1080p60 clip explaining how the 100-turn bottle editing session works (the user's teaser
video): LoRA transfer base → distilled, in-context reference tokens in the LoT layout, down-projection (16 patches → 1
token), 4 steps + blend, paste → next turn's context, 100 chained turns, speed breakdown 195 s → 7.7 s → 1.8 s (≈109×).
Beats + numbers: `storyboard.md`. Current script: `render_edit_v4.py` (v1–v3 kept). Never overwrite a version.

- Data: `prep_data.py` (imp_tome, needs torch) reads ~/LoT-george-branch/results/2026-10-08_084925_ref-edit-9b-distilled-bottles-100turn
  (k4 arm) + base timings from results/2026-10-07_152323_ref-edit-multiturn-9b-bottles → `data/generated/`.
- Render: `~/miniconda3/envs/imp_tome/bin/python render_edit_vN.py h|l` → `media/LoT_edit_vN_<1080p60|540p15>.mp4`
  (~1 min for h, 48 workers); `... still 4.4 10.0` → `scratch/still_vN_<t>.png`. No manim: PIL 2× SSAA, CMU Serif.
- Load npz eagerly (`dict(np.load(...))`): a lazy NpzFile corrupts under fork.
