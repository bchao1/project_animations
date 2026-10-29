# LoT Diffusion — 20-s 80s-arcade clip (arcade_clip)

Current script: `render_arcade_v2.py` (v1 kept; never overwrite a version). No manim: numpy + PIL + cv2 frames piped to
ffmpeg (48 workers), chiptune SFX synthesized in numpy and muxed as AAC. Font: Press Start 2P (OFL) in `data/`.
Render: `~/miniconda3/envs/imp_tome/bin/python render_arcade_v2.py h|l` → `media/final/LoT_arcade_v2_<res>.mp4` (~40 s);
`... still 5.9 10.4` → `scratch/still_v2_<t>.png`.

- Image: DoF example `img_000000000025` (giraffe, 4B image), turbo depth cue; game green = neon in-focus band (44,208,165).
  Stages 4,096 → 1,606 (cap 2×2) → 1,027 (real layout) tokens; speedups 1.9× / 2.4× = Amdahl fit of paper Fig. 21 timings.
- Bonus: site video demo "platformer"; per-frame levels rebuilt from `generated-grid` colours by `prep_data.py`
  (`data/platformer_levels.npy`, total 34,698 = site num_leaves). 75,600 → 38,046 → 34,698; 2.2× / 2.5× interpolated
  from Tab. 2 (LoT-Wan2.1 14B). Video playback rate follows the speedup; loop seam hidden by a static glitch.
