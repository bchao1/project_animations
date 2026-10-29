# LoT Diffusion — 20-s silver-age comic-book clip (comic_clip)

Current script: `render_comic_v2.py` (v1 kept; never overwrite a version). No manim: numpy + PIL + cv2 frames piped to
ffmpeg (Pool(10)), cartoon SFX (brass stabs, timpani, whooshes, boing, page flips, walking bass) synthesized in numpy.
Fonts (OFL) in `data/`: Bangers (titles/bursts/numbers), Comic Neue Bold (captions/balloons).
Render: `~/miniconda3/envs/imp_tome/bin/python render_comic_v2.py h|l` → `media/final/LoT_comic_v2_<res>.mp4` (~1 min);
`... still 7.0 10.95` → `scratch/still_v2_<t>.png`; `scratch/sheet.py v2` makes contact sheets.

- Look: every panel printed as CMYK halftone (rotated screens, misregistration); inside the token layout the dot pitch
  follows the token size (1×1 4 px … 8×8 16 px) and each token is a panel with an ink border (thicker when bigger).
- Content = arcade_clip: giraffe DoF 4,096 → 1,606 → 1,027 (ZOOM! 1.9X, KRAK! 2.4X); platformer video
  75,600 → 38,046 → 34,698 (WHOOSH! 2.2X, POW! 2.5X), levels from `../arcade_clip/data/platformer_levels.npy`.
- Speed lines are drawn before the panels so they only show in the gutters (v1 had them over the numbers).
