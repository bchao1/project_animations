# LoT Diffusion — 3-post launch thread (social_launch)

Started 2026-10-01. `posts.md` = refined post text; `visual_plan.md` = what each clip shows; finals in `media/`.
- V1 `clip1_signal/` (`./render.sh l|h [v1…v7]`, current v7 = v6 items + restyled mask/box + frame 0 already tokenized, `signal_common_v7.py` + `render_tiles_v7.py` (physics_v6.npz); v6 = the six ICLR-supplementary images, `signal_common_v6.py` + `sim_v6.py` + `render_tiles_v6.py`; v5 = v4 tiles + animation_signal_settle_v5): `signal_common_v2.py` (items, geometry, timeline,
  OFFSET = 5.6 s onto settle2d v3 time), `sim_v2.py` (settle2d sim2d_v3 patched to the six items), `render_tiles_v2.py`.
- V2 `clip2_quadtree/` (`./render.sh l|h [script]`, current v8: q_common_v8 + render_frames_v8 + volume vol11c_31_g40 from `render_vol_ghost_v8.py` (v16 volume with the wireframe at v16 t 4.0–4.6; shared spacetime files untouched); v7 = V1-style layout sweep; v6 = merge-up quadtree): corgi 2D physics (`sim_left_v1.py`, settle2d sim2d_v3 patched to one item)
  + spacetime v16 colour volume `vol11c_31` (render it with `../../spacetime_clip/render_v16.sh h image 31` if missing);
  PRE = 3 s quadtree prefix before v16's t = 0.
- V3 `clip3_budget/make_v3.sh` (user's four clips reversed: budget high → low; make_v2.sh = low → high): ffmpeg xfade via the flex env (system ffmpeg lacks xfade) piped into system libx264.
Upstream clips change under other sessions (budget_slider v4, spacetime v16 appeared mid-build): re-check the newest
version before cutting. Never call video layouts octrees (per-frame grids). Separate from `../social/` (older launch set).
Speedups on screen come from `../speedup/lot_speedup.py` (shared fit of the paper's timings), purple #E040FB, with "≈".
