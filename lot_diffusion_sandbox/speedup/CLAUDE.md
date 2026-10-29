# LoT speedup lookup (shared by every clip in lot_diffusion_sandbox)

Use this instead of per-clip fits whenever a clip shows a speedup for a token count.
- `lot_speedup.py`: `speedup(tokens, full=None, modality="image"|"video")`; model 1/s = a + b f + c f² (f = tokens /
  full, s(1) = 1, a, b, c ≥ 0), non-negative least squares over the `"fit": true` points of `speedup_points.json`.
  image (LoT-FLUX.2 4B, of 4,096): 1/s = 0.213 + 0.769 f + 0.017 f² (Tab. 1, Fig. 4, Fig. 7a, Fig. 21; ±0.1×).
  video (LoT-Wan2.1 14B, of 75,600): 1/s = 0.775 f + 0.225 f² (Tab. 2 + Fig. 7b; only 25–66% tokens measured).
- `speedup_points.json`: every speedup the paper reports, with source. Fig. 19 / Fig. 22 points sit ~0.6× above the
  4B trend (setting unclear) and are kept but not fit.
- `speedup_table.csv` (1–100% tokens, both modalities) and `speedup_fit.png` from `python fit_speedup.py`.
- These are estimates for arbitrary token counts, not measurements of the shown item: write "≈" on screen.
