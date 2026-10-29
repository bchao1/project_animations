# LoT website clip (website_clip/)

Started 2026-10-06. One 5-s square (1024 × 1024, 60 fps) website video: the giraffe example only (V1 v7's top-left tile,
item 025 / dof): depth map in its token layout → noisy tokens → image → mean-colour layout → fluid settle → 25.1% tick +
"≈2.5× faster". Count "1,027 of 4,096 tokens" from frame 0. No "depth map" label (user request).
- Current v3 = v2 look at 8 s (web_common_v3 timeline, physics 1.5×). v2 (5 s): image fills the frame (992 px, 16-px border); text overlaid in dark rounded boxes (count top-centre,
  "100%" top-left inside, "25.1%" above the level line, speedup box under the count). v1 = 640-px tile + text outside.
- `./render.sh l|h v3` → `media/LoT_website_v3_{15fps,60fps}.mp4`; `h` also writes the web copy `media/LoT_website_v3.mp4`
  (exactly 5 or 8 s, faststart).
- `web_common_v1.py` (geometry + timeline) imports `../social_launch/clip1_signal/signal_common_v7.py` and reuses tile 0 of
  its `physics_v6.npz` (tiles never touch in that sim). Physics played at PHYS_SPEED = 2× to fit 5 s.
- Frames: `render_frames_v1.py` → `data/generated/frames_v1/` (delete to re-render after timeline/geometry edits).
