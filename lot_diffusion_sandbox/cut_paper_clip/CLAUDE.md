# LoT Diffusion — 20-s cut-paper stop-motion clip (cut_paper_clip)

Current script: `render_cut_paper_v2.py` (v1 kept; never overwrite a version). No manim: numpy + PIL + cv2, 300 unique
poses at 15 fps each held 4 frames (60 fps), Pool(10) → ffmpeg; foley (snips, glue squeaks, stamp thunks, ukulele plucks,
xylophone) synthesized in numpy, stereo 48 kHz AAC. Fonts (OFL/Apache) in `data/`: Gochi Hand, Permanent Marker.
Render: `~/miniconda3/envs/imp_tome/bin/python render_cut_paper_v2.py h` → `media/final/LoT_cut_paper_v2_1080p60.mp4`
(~30 s); `... still 5.0 9.8` → `scratch/still_v2_<t>.png`.

- Same content as ../arcade_clip: giraffe DoF (4,096 → 1,606 → 1,027, 1.9X / 2.4X) and platformer video
  (75,600 → 38,046 → 34,698, 2.2X / 2.5X, levels from ../arcade_clip/data/platformer_levels.npy).
- Look: kraft table, construction-paper token squares (block-mean colour, hand-cut wobbly seams via a static domain warp),
  scissors cut the print, glue stick merges blurry pieces, red rubber stamps for speedups, pieces flip to the image.
