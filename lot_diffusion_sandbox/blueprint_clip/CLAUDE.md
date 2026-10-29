# LoT Diffusion — 20-s blueprint / architect clip (blueprint_clip)

Current script: `render_blueprint_v2.py` (v1 kept; never overwrite a version). No manim: numpy + PIL + cv2 frames piped
to ffmpeg (Pool(10)); foley (pen scratch, ruler clicks, eraser, paper rustle, stamp thump, chimes) + lo-fi pad synthesized
in numpy/scipy. Fonts (OFL) in `data/`: Architects Daughter (notes), IBM Plex Mono (title block, stamp).
Render: `~/miniconda3/envs/imp_tome/bin/python render_blueprint_v2.py h|l` → `media/final/LoT_blueprint_v2_<res>.mp4` (~30 s);
`... still 8.7 15.4` → `scratch/still_v2_<t>.png`.

- Sheet 1: giraffe DoF; depth drawn as contour survey, in-focus band hatched; plotter draws 4,096 1×1 grid → eraser REV 1
  (2×2, 1,606, 1.9×) → REV 2 (4×4/8×8, 1,027, 2.4×) with revision clouds + size callouts → APPROVED stamp → image built.
- Sheet 2: platformer video + per-frame plan (levels from `../arcade_clip/data/platformer_levels.npy`), 75,600 → 38,046
  (2.2×) → 34,698 (2.5×); playback speed follows. Speedup numbers as in `../arcade_clip/CLAUDE.md`.
