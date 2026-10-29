# LoT Diffusion — 20-s patchwork-quilt clip (quilt_clip)

Current script: `render_quilt_v2.py` (v1 kept; never overwrite a version). No manim: numpy + PIL + cv2 frames piped to
ffmpeg (Pool(10)), music-box / sewing-machine / pop SFX synthesized in numpy, AAC stereo 48 kHz.
Fonts (OFL) in `data/`: Baloo 2 ExtraBold (numbers, titles, satin-stitch look) + Patrick Hand (captions).
Render: `~/miniconda3/envs/imp_tome/bin/python render_quilt_v2.py h|l` → `media/final/LoT_quilt_v2_<res>.mp4` (~45 s);
`... still 9.0 11.8` → `scratch/still_v2_<t>.png`; `scratch/sheet.py v2` makes contact sheets.

- Tokens = fabric patches (7 procedural textures, tinted by the pastelized block-mean colour), seams = token borders with
  dashed stitches and puffy shading (distance transform), a needle rides each sweep, global wrinkle remap.
- Same content and numbers as `../arcade_clip`: giraffe DoF 4,096 → 1,606 → 1,027 (1.9× / 2.4×), then flip to the
  generated image; platformer 75,600 → 38,046 → 34,698 (2.2× / 2.5×), playback speed follows, loop seam is a 4-frame dissolve.
