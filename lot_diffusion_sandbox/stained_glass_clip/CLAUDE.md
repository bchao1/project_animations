# LoT Diffusion — 20-s stained-glass clip (stained_glass_clip)

Current script: `render_stained_glass_v2.py` (v1 kept; never overwrite a version). No manim: numpy + PIL + cv2 frames piped
to ffmpeg (Pool(10)); organ pad / glass chimes / bell synthesized in numpy with a noise-IR cathedral reverb. Font: Cinzel (OFL) in `data/`.
Render: `~/miniconda3/envs/imp_tome/bin/python render_stained_glass_v2.py h` → `media/final/LoT_stained_glass_v2_1080p60.mp4`
(~1.5 min); `... still 5.8 11.4` → `scratch/still_v2_<t>.png`.

- Tokens = glass panes, lead came = token borders; merges melt the lead between panes (gold glint).
- Content and numbers as ../arcade_clip: giraffe DoF 4,096 → 1,606 → 1,027 (1.9× / 2.4×), platformer video
  75,600 → 38,046 → 34,698 (2.2× / 2.5×, playback speed follows), levels from `../arcade_clip/data/platformer_levels.npy`.
