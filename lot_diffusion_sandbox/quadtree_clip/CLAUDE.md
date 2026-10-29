# LoT Diffusion — 10-s X clip: quadtree (graphics) → LoT layout → diffusion (quadtree_clip)

Current script: `animation_lot_quadtree_v2.py` (scene `LoTQuadtree`); never overwrite a version.
Render: `./render.sh l|h [script]` → `media/final/LoT_quadtree_<vN>_<res>.mp4`.
Assets: deer of paper Fig. 10 from `../budget_slider/data/generated/mask/27778182_deer/` (George's sweep) + its SAM3 mask.
Beats: 0–3 s region quadtree over the mask (to 2 × 2 cells) · 3–6.5 s cells morph into the B = 1,024 LoT layout →
DiT + LoRA → image · 6.5–10 s same layout at 1,792 … 4,096 tokens. No source / provenance labels (user preference).
