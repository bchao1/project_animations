# Storyboard — Level-of-Token Diffusion, 1-minute teaser

**Current version: v2** — `animation_lot_1min_v2.py`, 480p15 70.0 s (S1 ≈ 25 · S2 ≈ 23 · S3 ≈ 22). Rules: `../STYLE.md`.
Render `./render.sh l|h` → `media/final/LevelOfTokenDiffusion_1min_v2_<res>.mp4`. Helpers/beats from `../animation_lot_v2.py`.

| scene | beats |
|---|---|
| S1_Insight | title 72 bold + subtitle over a 16×16 token field that merges into the toy layout · corgi "the image" vs "what the model computes" (256 HR tokens) · captions: pays for every token equally / empty sky costs as much as her face · painted map crossfades over the image ("the plan") · tokens merge (2×2 then 4×4) → count "104 + 26 + 3 = 133 tokens" · rectangles merge → 126 tokens, toy note |
| S2_Method | noisy 8×8 latent → 2×2 patch → A_eᵀ → one token → DiT (LoRA) → h_(2,2) → 2×2 full-res flow; base models · paper Fig. 4 row (box / mask / texture / depth) |
| S3_Payoff | `b_speed` (Tab. 1/2 curves, 2.04× AMBER / 3.53× PURPLE) · `b_imgcompare` (Fig. 7a row) · end card: title 64 + HR tagline (+ authors / links when set), 5 s |
