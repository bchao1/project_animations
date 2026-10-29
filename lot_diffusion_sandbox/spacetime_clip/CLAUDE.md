# LoT Diffusion — 10-s X clip: space-time view of a video's per-frame LoT layouts (spacetime_clip)

Batch (2026-10-01): `./render_batch.sh h [items]` → `media/final/batch/LoT_spacetime_v15_item<k>_1080p60.mp4` for the 10
site videos 0, 3, 4, 10, 14, 16, 22, 26, 37, 39 (all four layout sources; 35 skipped: its video opens on black frames).
`animation_lot_spacetime_v15.py` = v14 for any `$LOT_ITEM` (per-video SUBJECT caption + own token fraction, per-item
`lay_<k>/` 2D-layout cache); `render_volume_v10.py` = v9 + `$LOT_EGL_DEVICE`. Physics runs ~3–5 min per item when
10 share 4 GPUs (`BATCH_GPUS`, default 1 5 7 8); whole batch ≈ 10 min from scratch, ≈ 4 min if sims/volumes exist.

v14 = current: `animation_lot_spacetime_v14.py` + `render_volume_v9.py` (→ `vol9_31/`): v13 with 50%-opaque green
token faces before the collapse (user: the stack must feel like a solid). Otherwise identical to v13:
Previous (v13): `animation_lot_spacetime_v13.py` (scene `LoTSpaceTime`, 13.8 s) + `spacetime_common_v7.py` +
`sim_physics_v2.py` (→ `data/generated/physics2_31.npz`, ~100 s on one A6000, `SIM_GPU=5`) + `render_volume_v8.py`
(→ `vol8_31/`, 828 frames). Never overwrite a version.
v13 (user feedback 2026-10-01: the ease into the lattice slab looked unnatural): no snap — physics runs to the end
with lighter damping (viscosity 0.3 → 2.3 /s, pressure halves), so the surface keeps moving around the amber 36.8%
waterline. v12 = same physics + lattice snap (`sim_physics.py`, `snap_match.py`, common_v6, render_volume_v7).

Previous (v12): `animation_lot_spacetime_v12.py` (scene `LoTSpaceTime`, 13.8 s) + `spacetime_common_v6.py` +
`sim_physics.py` (GPU position-based dynamics, → `data/generated/physics_31.npz`, ~40 s on one A6000, `SIM_GPU=5`) +
`snap_match.py` (→ `snap_31.npz`) + `render_volume_v7.py` (→ `vol7_31/`, 828 frames). Never overwrite a version.
v12 (user feedback 2026-10-01: the v11 trickle "is not how water / sand / particles flow"): real particle physics —
27,795 particles (diameter grows 0.98 → 1.1 after release, so the settled packing is ≈ 1 cube per cell), gravity,
Jacobi contacts on a uniform grid, floor + walls, friction 30 → 0.3 /s (castle → liquefy), hydrostatic push down the
smoothed height gradient (one-sided at walls, else piles climb the corners), small horizontal jitter, viscosity at
the end; then a local lattice snap (integral min-cost flow between neighbouring columns + per-column height order;
median 0.7 cells). v11 = cellular-automaton trickle (settle_sim.py, common_v5, render_volume_v6, `vol6_31/`).

v10 story (user feedback 2026-10-01): frame 1 flat 2D face-on → frames step out toward the viewer (time points out of
the screen; octant x, y, time at the back-bottom-left corner of frame 1, drawn in 3D with depth) → every token shrinks
to a 1 × 1-token cube keeping its colour → cubes fall straight down into piles (one per token per (x, time) column)
→ piles flatten into a slab of height 27,795 / (80 · 21) = 16.54 of 45 → the amber count collapses onto the y axis
as "36.8%" under a "100%" tick and a wireframe of the full-resolution volume. The "no token spans time" footnote
fades out when the shrink starts (the flatten pools cubes across frames).
Render: `./render.sh l|h [script]` → `media/final/LoT_spacetime_<vN>_<res>.mp4` (renders the 3D volume first if missing).

- Data: site showcase item 31 (`../web/showcase.json`, `token_grid` = gzip-base64 uint8 (y0, y1, x0, x1) per latent frame;
  21 latent frames × 45 × 80; 27,795 of 75,600 tokens = 2.72×), video `../data/clips/show31_gen/` (81 frames).
  `LOT_ITEM=<k>` switches item (scored in `scratch/score.py`; item 10 robotics was 62% fine → solid brick, no visible tube).
- 3D: `render_volume.py` (moderngl, headless EGL, A6000) → `data/generated/vol_31/NNNN.png` 60 fps, 1256 × 810, 2× SSAA + MSAA.
  1 × 1 tokens opaque; larger tokens = glass boxes (faint faces, extent-coloured edges), sorted back to front.
  Timeline / camera / projection shared via `spacetime_common.py` (manim overlays use `project()`).
- Accuracy: every token is one latent frame thick (extent (1, e_h, e_w)); the tube is stacked per-frame layouts.
  On-screen footnote says so; never call them token cubes.
- manim gotcha: ImageMobject updaters must take `dt`, else `wait()` is treated as static and the frames freeze.

v16 (2026-10-01, item 31): fluid from the first landing (spacetime_common_v8 + sim_physics_v3: push/jitter/height map on landed cubes only, low friction from release, SETTLE 8.3–9.3 s); `./render_v16.sh h extent|image 31` → `media/final/LoT_spacetime_v16[_color]_item31_1080p60.mp4` (image = tokens in mean video colour, render_volume_v11 → vol11c_31).
