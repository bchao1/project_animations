# LoT Diffusion — 20-s falling-block puzzle clip, handheld-LCD look (blocks_clip)

Current script: `render_blocks_v2.py` (v1 kept; never overwrite a version). No manim: numpy + PIL + cv2 frames piped to
ffmpeg (Pool(10)), handheld-style sound (pulse / 4-bit wave / noise) synthesized in numpy, AAC stereo 48 kHz.
Font: Silkscreen (OFL) in `data/`. 4-shade LCD palette for UI; blocks keep the depth / image / video colours.
Render: `~/miniconda3/envs/imp_tome/bin/python render_blocks_v2.py h` → `media/final/LoT_blocks_v2_1080p60.mp4` (~40 s);
`... still 5.6 8.1` → `scratch/still_v2_<t>.png`.

- Content and speedups as `../arcade_clip` (giraffe DoF 4,096 → 1,606 → 1,027, 1.9X / 2.4X; platformer video
  75,600 → 38,046 → 34,698, 2.2X / 2.5X; levels from `../arcade_clip/data/platformer_levels.npy`).
- Mechanics: 1×1 rain fills the well bottom-up; merges = groups blink like a line clear, vanish, merged block drops in
  (drop time / current speed); every block flips into the image. Video loops at 73 frames with an 8-frame crossfade.
