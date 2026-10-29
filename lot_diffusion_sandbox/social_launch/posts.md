# LoT Diffusion — launch post (3-post thread), refined 2026-10-01

Edits vs. the draft are noted under each post. Character counts include spaces; post 1 is over 280, so it needs
X Premium (long post) or it folds at "Show more" right after the hook question.

## 1/3 🧵 (569 chars) — visual: `media/V1_signal_settle_v7_1080p60.mp4` (16.0 s)

Visual planning and layout design are becoming core building blocks of image and video generation, as in recent models like @reve and @ideogram_ai. We ask: can these signals also make generation more efficient, not just more controllable?

In our new work, Level-of-Token (LoT) Diffusion, we turn control signals such as semantic masks, bounding boxes, and depth maps into multi-resolution token layouts. Just as quadtrees and level of detail (LoD) speed up rendering in graphics, LoT speeds up generation by 2–3.5× while preserving image quality and controllability. 🧵

- "2–4×" → "2–3.5×": the paper's headline is up to 2.04× (images) and 3.53× (video). "4×" is not in the paper.
- Check the handles: @ideogram_ai is right as far as I know; confirm @reve is Reve's account.

## 2/3 (306 chars) — visual: `media/V2_quadtree_settle_v8_1080p60.mp4` (21.4 s)

Given a layout and a token budget, LoT builds a quadtree-like token grid: fine tokens where the layout asks for detail, coarse tokens everywhere else. For video, every frame gets its own grid.

A fine-tuned DiT reads only this short token sequence and still generates the image or video at full resolution.

- "quadtree or octree-like" → "quadtree-like … every frame gets its own grid": every video token is one latent frame
  thick (no token spans time), so the video layout is a stack of per-frame quadtrees, not an octree. Calling it an
  octree is the claim a reviewer would catch first.

## 3/3 (223 chars) — visual: `media/V3_budget_v3_1080p60.mp4` (19.1 s, budget steps down)

A single LoT model works across a wide range of token budgets and layouts.

The same layout can be tokenized coarse or fine: as the budget drops from 4,096 to 1,024 tokens, LoT still produces a coherent image at every step.

- Merged your two versions into one post. "Images or videos" → "image": the budget sweep is image-only, and images
  and videos use two different models (FLUX.2 [klein] 4B, Wan2.1 14B), so "a single model … images or videos" would mislead.

## Open
- No links/credits post yet (paper is under double-blind review). Add a 4/4 with 📄 / 🌐 once arXiv + project page exist.
