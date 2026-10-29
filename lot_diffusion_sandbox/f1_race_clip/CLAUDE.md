# LoT Diffusion — 20-s F1 race-broadcast clip (f1_race_clip)

Current script: `render_f1_race_v3.py` (v1, v2 kept; never overwrite a version). No manim: numpy + PIL + cv2 frames piped
to ffmpeg (Pool 10), stereo 48 kHz audio synthesized with numpy/scipy (engines panned FULL left / LoT right).
Fonts (OFL, `data/`): Barlow Condensed, Titillium Web. Render: `~/miniconda3/envs/imp_tome/bin/python render_f1_race_v3.py h|l`
→ `media/final/LoT_f1_race_v3_<res>.mp4` (~30 s); `... still 6.0 10.7` → `scratch/still_v3_<t>.png`.

- Same content as `../arcade_clip`: giraffe DoF (4B image) 4,096 → 1,606 → 1,027 tokens (1.9X / 2.4X);
  platformer video 75,600 → 38,046 → 34,698 (2.2X / 2.5X), levels from `../arcade_clip/data/platformer_levels.npy`.
- Race clock: full-res 5.78 s per image, LoT 5.78 / 2.4 = 2.41 s, gap +3.37 s; broadcast clock runs 1.6× real time.
  10 "laps" = 10 abstract compute sweeps (not the real step count). The FULL car has no image of its own, so its
  panel only dims to FINISHED.
