# LoT Diffusion — token-budget slider clips (budget_slider)

One 5-s 1080p60 clip per scene of paper Figs. 10–13 (16 clips), for the user to pick from.
Current script: `animation_lot_budget_v4.py` (v4, 2026-10-01: slider → filled amber progress bar with quarter ticks,
% inside the fill, "B / 4,096 tokens" above its end; v3, 2026-10-01: no header / caption, panels 5.7 wide, live % between
panels, slider = 0 → 4,096 amber progress arrow with "N tokens" riding the tip; v2 drops the "LoT-FLUX.2 4B · same prompt and seed · paper Fig. N"
label and the "Same prompt, same seed" caption prefix, user request 2026-09-30). Never overwrite a version.

- Data (read-only): George's sweep `/home/brianchc/LoT-george-branch/results/2026-09-04_001859_budget-sweep-100-nopeople-mask-vrs/outputs`,
  `sweep_<arm>/<item>/B<budget>/{grid.pt, out.png}`. Figs = `make_supp_figures.sh` pages: supp_mask_1/2 = Fig. 10/11,
  supp_vrs_1/2 = Fig. 12/13. Copied to `data/generated/<arm>/<item>/` on import.
- Clip: shaded vector token layout (left, leaves `(r0, r1, c0, c1)`, VRS is anisotropic) → generated image (right),
  v3 steps B = 1,024 / 1,792 / 2,560 / 3,328 / 4,096; holds 0.8 / 0.6 / 0.6 / 0.6 / 0.8 s, 0.4-s morphs.
- Render: `./render.sh l|h [script] [Scene ...]` → `media/final/<res>/LoT_budget_<Scene>.mp4`
  (one `--media_dir` per scene: parallel renders otherwise race on the shared text cache).
- Style: `../STYLE_NOTES.md`; helpers copied from `../teaser_5min/animation_lot_5min_v3.py`. Do not read `../../lot_diffusion/`.
