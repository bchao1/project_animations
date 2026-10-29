# LoT Diffusion — 20-s Mondrian / De Stijl clip (mondrian_clip)

Current script: `render_mondrian_v2.py` (v1 kept; never overwrite a version). No manim: numpy + PIL + cv2 frames piped to
ffmpeg (Pool(10)), marimba / piano / brush-swish audio synthesized in numpy. Font: Jost variable (OFL) in `data/`.
Render: `~/miniconda3/envs/imp_tome/bin/python render_mondrian_v2.py h|l` → `media/final/LoT_mondrian_v2_<res>.mp4`;
`... still 6.4 10.6` → `scratch/still_v2_<t>.png`.

- The layout is the painting: black bars = token borders, colour = token size (1×1 white, 2×2 yellow, 4×4 blue, 8×8 red).
- Same content and numbers as `../arcade_clip` (giraffe DoF 4,096 → 1,606 → 1,027, 1.9× / 2.4×; platformer video
  75,600 → 38,046 → 34,698, 2.2× / 2.5×, levels from `../arcade_clip/data/platformer_levels.npy`).
