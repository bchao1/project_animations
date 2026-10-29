# LoT Diffusion — 20-s retro-desktop clip (retro_desktop_clip)

Current script: `render_retro_desktop_v2.py` (v1 kept: Pixelify Sans, whose "2" read like "Z"/"8"; never overwrite a version).
No manim: numpy + PIL + cv2 frames piped to ffmpeg (Pool(10)), UI sounds synthesized in numpy (chime, clicks, ding,
hard-drive chatter, tada). Generic 90s desktop (no real logos); text = aliased Liberation Sans (`d.fontmode = "1"`).
Render: `~/miniconda3/envs/imp_tome/bin/python render_retro_desktop_v2.py h` → `media/final/LoT_retro_desktop_v2_1080p60.mp4`
(~35 s); `... still 5.6 10.6` → `scratch/still_v2_<t>.png`.

- Same content as ../arcade_clip: giraffe DoF 4,096 → 1,606 → 1,027 tokens (1.9X / 2.4X), platformer video
  75,600 → 38,046 → 34,698 (2.2X / 2.5X; levels from ../arcade_clip/data/platformer_levels.npy).
- Beats: LoT Paint draws 1x1 tokens → Token budget slider drag 2x2 / 8x8 → Full model vs LoT progress race (2.4× ratio,
  "0:58" vs "0:24" are gag countdowns, not real timings) → image viewer → player + Tokens window, 2x2 / 4x4 buttons speed
  playback → About LoT.
