# LoT Diffusion — 10-s LEGO clip (lego_clip)

Current script: `render_lego_v1.py`; never overwrite a version that was shown to the user (copy to `_v2` for a fix round).
No manim: 3D bricks in moderngl (headless EGL, `EGL_DEV=<gpu>` picks the device, default 5) + PIL text → ffmpeg pipe.
Render (conda `imp_tome`): `python render_lego_v1.py l|h` → `media/final/LoT_lego_v1_<540p30|1080p60>.mp4` + contact sheet
`scratch/sheet_v1_<res>.png`; `STILL=1.2,6.9 python render_lego_v1.py h` writes `scratch/still_v1_<t>.png` only.

- Data: corgi demo layout `../teaser_1min/data/generated/corgi_layout.json` (1,546 leaves of 64 × 64) and image
  `../data/img/paint0_result.png`; plate colour = mean image colour under the plate.
- Beats: 0–2.5 s 4,096 1 × 1 plates rain in (counter = landed plates) · 3–7 s plates fuse 1→2→4→8 along the real
  quadtree (lift, swap, drop) → 1,546, "2.6× fewer" (4,096 / 1,546) · 7–8.4 s camera to top-down, board left,
  generated image right, title.
- Geometry: stud pitch 1, plate height 0.4, stud r 0.30 h 0.18, 0.035 seam; 4096² PCF shadow map, sun from upper-left.
