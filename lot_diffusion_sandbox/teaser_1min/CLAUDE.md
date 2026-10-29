# Level-of-Token Diffusion — 1-minute teaser (teaser_1min)

Ground truth: `storyboard.md` (current script named at its top). Style: `../STYLE_NOTES.md`. Never overwrite a
script version: copy to `animation_lot_1min_vN+1.py` per fix round and update `storyboard.md`.

- Env: `imp_tome` conda env + `LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7` (set by `render.sh`).
- Render: `./render.sh l|h [script] [Scene ...]` → S1_Insight, S2_Method, S3_Payoff in parallel (logs in
  `scratch/logs/`), concatenated to `media/final/LoTDiffusion_1min_vN_<res>.mp4` (all scenes only).
- Sources are read-only by absolute path from `..` (`data/img`, `data/clips`, `web/assets`, the PDF); generated
  assets (layout json, Fig. 7a crops, x_t frames) go to `data/generated/` on import.
- No LaTeX (Pango markup via `tx()`/`mathify()`), CMU Serif, text added instantly; numbers only from the paper / site
  (`../level-of-token-diffusion/summaries/lot_diffusion.md`) or computed from the drawn layout.
- QA: `python scratch/qa_sheets.py media/videos/<script>/480p15 scratch/qa_vN [step_s]` → contact sheets per scene.
- Do not read or reuse `/home/brianchc/project_animations/lot_diffusion/` (rejected earlier attempt).
