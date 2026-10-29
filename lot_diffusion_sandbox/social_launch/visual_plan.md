# Visuals — one clip per post, no footage shared between clips (built 2026-10-01)

All 1080p60 in `media/`. Every token in V1 and V2 is filled with its mean image / video colour.

## V1 — post 1 (16.0 s): `clip1_signal/` v7 → `media/V1_signal_settle_v7_1080p60.mp4`
v7 (2026-10-02): frame 0 is already the token layout on the signal (signal pixels inside the token boxes, counts at
N of 4,096 from the start; no full-grid sweep). Mask and box restyled to the depth map's turbo family on a dark navy
vignette: smooth teal → mint mask with a light rim + glow; box filled amber → red-orange with a light edge (no tags —
tokens chopped them). Hold 2.5 s → noise 2.5–3.0 → denoise 3.0–5.0 → rest as v6, 1.5 s earlier.
v6 (2026-10-02): the six images the user picked from the ICLR supplementary page (`clip1_signal/data/iclr/*.webp`,
pixel-exact = `data/img/img_<id>_<src>_generated.png`, not `_4b`): depth giraffe 025 (1,027) / wine 283 (2,047) |
mask zebra 034 (703) / cake 092 (2,506) | box dog 042 (2,320) / sandwich 142 (529). Physics re-simulated (sim_v6.py);
drawing and text identical to v5. Below: v5 description.
Columns by signal, two images each: depth map (wine 283, man 294) | semantic mask (zebra 154, snowboard 201) |
bounding box in orange (giraffe 025, dog 307). No captions, footnotes or speedups: only column names and one count
per tile. 0 s signal with its 64 × 64 grid, "4,096 tokens · 100% of 4,096" → 1–2 s sweep into the signal-colour
layout, count runs down to n → 2-s hold → noisy token layout → 2-s denoise into the image → hold → mean-colour
layout → shrink → settle2d v3 physics (re-simulated for these images, sim_v2.py) → % on a tick at the pile height.
v3: purple "≈N× faster" under every count (../speedup/lot_speedup.py), 344-px tiles.
v4: only "N tokens" until the settle; then % on the left tick + speedup under the tile; 390-px tiles.
v5: counts read "2,047 of 4,096 tokens" (v4 tiles, text only).
v1 (`V1_signal_settle_v1_*`, rejected): 3 × 2 by image, captions, no holds, mask and box both blue.

## V2 — post 2 (21.4 s): `clip2_quadtree/` v8 → `media/V2_quadtree_settle_v8_1080p60.mp4`
v8 (2026-10-02): the video's full-resolution wireframe box and both 100% marks fade in once the volume has stacked
(11.6–12.2 s), before the shrink / settle (v7: 16.5–17.1 s, after it). Video 100% label tracks its tick during the orbit.
v7 (2026-10-02): no quadtree merging. The layout forms exactly as in V1: full grid of outlines on the silhouette
from frame 0 ("4,096 of 4,096" / "75,600 of 75,600"), 1.0–2.0 s diagonal sweep into the layout outlines while the
counts run down smoothly (right 0.06 s later), 2-s hold, noise 4.0–4.5, denoise 4.5–6.5. Both 100% marks fade in
together with the video's wireframe (16.5–17.1 s); in v6 the image's appeared at 14.2 s, ahead of the video's.
Left image (corgi, painted detail map, 1,546 tokens), right video (spacetime item 31, 27,795 tokens). No captions.
0–2.7 s quadtree in quadtree_clip's style (dim grey detail silhouette, white cell outlines level by level, from the
exact leaves) → noisy token layout (counts run down from "4,096 · 100%" / "75,600 · 100%") → 2-s denoise into the
image / latent frame 1 → mean-colour layout → right crossfades into spacetime v16 colour volume (`vol11c_31`, from
its t = 1.0) and stacks in time → both shrink → both settle → % copies fly onto the ticks (37.7% / 36.8%).
v4: purple "≈2.0× faster" / "≈3.2× faster" under the counts (shared lookup).
v5: only "N tokens" until the settle; then % on the left tick + speedup below; image panel 5.4, volume 7.2 units.
v6: the quadtree runs bottom-up: full 1 × 1 grid (4,096 / 75,600) merges to 2 × 2 → 4 × 4 → 8 × 8 where the layout is
coarse (1.0 / 1.8 / 2.6 s), counts "N of 4,096 tokens" dropping 4,096 → 2,008 → 1,591 → 1,546 and 75,600 → 36,886 → 27,795.

## V3 — post 3 (19.1 s): `clip3_budget/make_v3.sh` → `media/V3_budget_v3_1080p60.mp4` (v2 = same, low → high)
budget_slider v4 renders picked by the user, 0.3-s crossfades: F10_1_Deer → F10_4_Mountain → F13_4_Flower →
F12_2_Tree, each reversed so the budget steps down 4,096 → 1,024. Re-run make_v3.sh whenever budget_slider's finals are re-rendered.
