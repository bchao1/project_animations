# LoT Diffusion — 14.5-s X clip: six images settle into their token fraction (settle2d_clip)

2D version of `../spacetime_clip` (v13 logic). Current script: `animation_lot_settle2d_v2.py` (scene `LoTSettle2D`)
+ `settle2d_common_v2.py` (data, timeline, geometry, physics params, speedup fit) + `sim2d_v2.py` (GPU PBD, all six
images batched, → `data/generated/physics2d_v2.npz`, ~35 s) + `render_tiles_v2.py` (PIL 2× SSAA, 48 workers →
`tiles_v2/`, 870 frames, ~20 s). Render: `./render.sh l|h` → `media/final/LoT_settle2d_v2_<res>.mp4`. Never overwrite a version.

v2 (user feedback 2026-10-01: top must end flat by normal fluid flattening, no lattice-snap transition): ±20% random
disk sizes (POLY; equal disks crystallise and one-row steps never flow away — this was the main fix), hydrostatic
push 200 that doubles over SETTLE instead of halving, 3-column smoothing, jitter keeps 70%, damping 1.0. Surface std
1.18 → 0.40 rows (`scratch/rough.py <npz>`; ≈ 0.4 is the floor of a partly filled top row). Surface-weighted push
(SURF) made it rougher. Sweeps: `SIM_OVR='{"POLY":0.2}' SIM_OUT=scratch/sweep/x.npz python sim2d_v2.py`.
Side effect: a slosh runs up the left wall around t ≈ 7.5 s (a few squares splash up) before it flattens.

- Story per image: noise → generated image → diagonal sweep into its LoT layout (fill = mean image colour per token)
  → shrink to 1 × 1 tokens → fall / castle / liquefy / flow flat (same physics recipe as spacetime sim_physics_v2, 2D)
  → count collapses onto a tick at n / 64 rows → "≈X× speedup" under the image.
- Data: images_index.json items with 25–35% of 4,096 tokens, 4B images (`*_generated_4b.png`) + exact leaves
  `../data/layouts/`: sandwich 142 dof 1,024 · giraffe 025 dof 1,027 · hand 127 mask 1,339 · dog 042 mask 1,351 ·
  giraffe 144 mask 1,369 · dog 307 box 1,420. Dropped duplicates: giraffe 025 box, dog 307 mask.
- Disk diameter calibrated so the settled mean column top = n / 64: v1 D = 1.085 (± 0.15 rows), v2 D = 1.05 (± 0.24).
- Speedups are an Amdahl fit (a = 0.215) of paper Fig. 21 per-image timings (same as ../arcade_clip), not measured on
  these images; on-screen footnote says so.

v3 (2026-10-01): fluid from the first landing — push/jitter/height map on landed disks only, low friction from release, damping SETTLE 6.3–7.3 s at 2 /s, constant push (sweep scratch/sweep3/a: sloshing gone by 8 s, surface std 0.47). Files: settle2d_common_v3, sim2d_v3, render_tiles_v3, animation_lot_settle2d_v3 → media/final/LoT_settle2d_v3_1080p60.mp4.
