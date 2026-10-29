# Storyboard — Level-of-Token Diffusion, 5-minute teaser (3b1b house style)

**Current version: v2** — `animation_lot_v2.py`, 480p15 ≈ 5:01 (S00 7.9 · S01 35.8 · S02 29.4 · S03 42.8 · S04 12.5 · S05 34.4 · S06 23.3 · S07 52.6 · S08 37.9 · S09 25.4 s). Rules: `STYLE.md`.
v1 (clip-montage approach, off-style) is kept in `superseded_v1/` for reference only.
Numbers: `level-of-token-diffusion/summaries/lot_diffusion.md`. Render: `./render.sh l|h`.
1-min cut: `teaser_1min/storyboard.md`. Thread clips: `clips/storyboard.md` (one clip = one beat below).

## Running example
The painted-demo image "a little girl riding an enormous friendly corgi" (site, LoT-FLUX.2): `data/img/paint0_result.png`
with its detail map `paint0_painting.png` (white = fine, yellow, orange, black = coarse). Toy layout on a 16 × 16 grid:
the painting's detail scores (black 0 · orange 0.3 · yellow 0.6 · white 1) area-averaged to 16 × 16, then the paper's
quadtree rule (Eq. 23, blocks 4×4 → 2×2 → 1×1; split 4×4 if max ≥ 0.2, split 2×2 if max ≥ 0.8). Tokens: 1×1 HR blue,
2×2 green, 4×4 dark green. Counts are computed in code and labelled "toy".

## Scenes
| scene | beats (captions ≈ 10 words; one visual transformed per beat) |
|---|---|
| S00_Title | title "Level-of-Token Diffusion" 72 bold y 2.75 · subtitle "Adaptive, Efficient Image and Video Generation" 32 y 1.95 · 16×16 LoTField hero (3.8 at (0, −0.75)) cycles uniform → plan A → plan B → plan A (retargets 0.6 s) · authors line (constant AUTHORS, omitted while unset) |
| S01_TokenCost | card "Part 1 · The problem / Every token costs the same" · header "1 · Every token costs the same" · **A** corgi 4.6 at (−3.5, 0.1) → 16×16 grid lines → patch-colour tokens → amber box "a token on her face", purple box "a token of empty sky" → recolour to HR, "16 × 16 = 256 tokens" · **B** attention matrix 4.0 at (3.4, −0.1), query box scan, "256 × 256 = 65,536 query–key pairs" → squares N / 4N, N² / 16N² · last caption is the question |
| S02_Plans | header "2 · Detail can be planned" · **A** graphics: level-of-detail meshes (near dense, far sparse) and variable-rate shading tiles, toy · **B** empty canvas + plan boxes "girl · corgi · grass · sky" → the agent's painted detail map (site) fades in on the canvas → bridge card "Can a pretrained DiT generate from tokens of different sizes?" |
| S03_Layouts | card "Part 2 · The method / Level-of-token layouts" · header "3 · Level-of-token layouts" · **A** 256 HR tokens (5.0 at (−3.4, 0.1)) + detail-map panel (2.4 at (3.6, 1.2)) → 4×4 merges, then 2×2 merges (ReplacementTransform, by distance from the subject) → count column 1×1 / 2×2 / 4×4 → boxed L → L / 256 · **B** two sky 4×4 pairs → 4×8, grass 2×2 pairs → 2×4; equation R_i = [u, u+e_h) × [v, v+e_w) ; e ∈ {1, 2, 4, 8} · **C** tokens flatten to a sequence row (y 2.75) → attention squares 65,536 → L² |
| S04_Naive | header "4 · Just plug it in?" · paper Fig. 21, rows woman / archway: token layout · direct HR prediction (RED note) · ours, with "1,111 tokens · 2.2× faster" / "1,752 · 1.8×" |
| S05_AsymFlow | header "5 · Patch-wise asymmetric flow" · **A** 8×8 noisy latent (noise colours) with a toy layout; one 2×2 token lifted: 4 cells → A_eᵀ → one token ∈ ℝ^D · **B** short sequence → DiT (LoRA) → head h_(2,2) → û_A (4 cells) → equation û = P_e û_A + (I − P_e)(x_t + û_A)/σ_t with coloured terms → full-resolution velocity (4 cells) → VAE decodes as usual |
| S06_SizeAware | header "6 · Telling the DiT each token's size" · **A** 8×8 lattice with indices, 2×4 token: top-left → centre dot, "RoPE index = patch centre"; 1×1 token keeps its pretrained RoPE · s(e) = (1, 2, 3, −1) bars → zero-init MLP g_φ · **B** DiT + LoRA box, "FLUX.2 klein 4B / 9B · Wan2.1 14B", "heads start from the pretrained head: W_e = A_e W_out" |
| S07_Layouts | header "7 · Layouts are just data" · **A** Fig. 4 deer row (4 columns: box / mask / texture / depth, compression · speed-up) · **B** Fig. 10 deer, budget sweep B = 1,024 … 4,096 (slider) · **C** Fig. 19 hydrant / bird / cat: fine box moves, subject follows; Tab. 4 44.9% → 66.8% · **D** driving video (site show04): layout video + LoT-Wan2.1 video, 1.97× fewer tokens |
| S08_Results | card "Part 3 / Results" · header "8 · Results" · **A** speed-up vs token compression (Tab. 1, Tab. 2) curves, "2.04×" AMBER / "3.53×" PURPLE big · **B** HPSv3 bars at 2.935× (Full / ToMe-SD / DDiT / Foveated / Ours) · **C** Fig. 7a row with speed-ups, RED notes · **D** site video comparison (Fig. 7b example) 4 playing videos, speed-ups 1.00 / 1.60 / 1.59 / 1.62× |
| S09_Outlook | header "9 · Limitations and outlook" · Fig. 22 sheet music (3,189 vs 1,466 tokens) · "Level-of-Token Diffusion in three ideas" (3 points) · headline "up to 2.04× faster images · up to 3.53× faster video · layout-adaptive" · end card |
