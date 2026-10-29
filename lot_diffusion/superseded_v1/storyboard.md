# Storyboard — LoT Diffusion ≈5.5-minute teaser (3b1b style)

**Current version: v1** — `animation_lot_v1.py`, 10 scenes; 480p15 total 330.6 s (5:31). Render `./render.sh l|h`
→ `media/final/LevelOfTokenDiffusion_5min_v1_<res>.mp4`. Built by replaying clip bodies from `clips/` (see
`clips/storyboard.md`) with `extra` hold seconds, plus explainer-only beats. Caption hold = len/16 + 0.8 s.

| scene | ≈ s | content |
|---|---|---|
| S00_Title | 7 | Fig. 1 typography: token layout → crossfade to the LoT image → shrink left, title + two subtitles |
| S01_Problem | 41 | card "Part 1 · The problem" → C01 → attention cost (N → 4N tokens, N² → 16N² pairs), 2 captions → C02 |
| S02_Layouts | 24 | card "Part 2 · The idea" → C03 → 2 captions (partition → one token each; L vs HW / THW) |
| S03_Sources | 38 | C04 → four rule cards from Sec. A.3 (masks/boxes levels, VRS τ(R), DoF blur radius, painted quadtree) → C07 |
| S04_Method | 44 | card "Part 3 · The method" → two obstacles (decoder, denoiser) → flow matching / AsymFlow eqs → C05 → 4 details (Procrustes A_e, scale s_e, head init W_e = A_e W_out, LoRA) |
| S05_LoTDiT | 20 | C06 → training cards LoT-FLUX.2 / LoT-Wan2.1 |
| S06_Video | 45 | card "Part 4 · Results" → R03 → R05 → R04 → R07 |
| S07_Control | 39 | R08 → R10 → R09 |
| S08_Compare | 34 | C08 → R11 → ablation table (Tab. 3 at ≈1.52×: HPSv3 / FID / MUSIQ) |
| S09_Outlook | 38 | R02 → limitation (Fig. 22 sheet music, 3,189 vs 1,466 tokens) → outlook (3 bullets) → end card |
