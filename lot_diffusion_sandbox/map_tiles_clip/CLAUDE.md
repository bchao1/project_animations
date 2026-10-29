# LoT Diffusion — 20-s web-map clip (map_tiles_clip)

Current script: `render_map_tiles_v2.py` (v1 kept; never overwrite a version). No manim: numpy + PIL + cv2 frames piped
to ffmpeg (Pool 10), UI sounds / tile-tick patter / pad synthesized in numpy, AAC stereo 48 kHz. Font: Inter (OFL) in `data/`.
Render: `~/miniconda3/envs/imp_tome/bin/python render_map_tiles_v2.py h|l` → `media/final/LoT_map_tiles_v2_<res>.mp4` (~1 min);
`... still 8.4 16.2` → `scratch/still_v2_<t>.png`; `scratch/sheet.py v2` makes contact sheets.

- Metaphor: tokens = map tiles, LoT = map level-of-detail. Grey placeholders stream in centre-out; tile-load durations are
  2.4 s / 2.4/1.9 / 2.4/2.4 so the loading itself shows the 1.9× / 2.4× speedups (illustrative, relative only).
- Same content as `../arcade_clip` (giraffe DoF 4,096 → 1,606 → 1,027; platformer 75,600 → 38,046 → 34,698, levels from
  `../arcade_clip/data/platformer_levels.npy`). Video loops on a 73-frame cycle with an 8-frame crossfade at the seam.
