# Level-of-Token Diffusion (anonymous ICLR 2027 submission 7014) — numbers used on screen

Source: `../../7014_Level_of_Token_Diffusion.pdf` (text: `../../paper/lot_text.txt`) and the supplementary site
manifests mirrored in `../../web/`. Page numbers refer to the PDF.

## Method facts
- Layout: partition of the H × W token grid into L axis-aligned rectangles R_i = [u_i, u_i + e_h,i) × [v_i, v_i + e_w,i)
  (Eq. 2). Compression = HW / L (images), THW / L (video). (p. 4)
- Extents: LoT-Flux e_h, e_w ∈ {1, 2, 4, 8}; LoT-Wan (1, e_h, e_w), e_h, e_w ∈ {1, 2, 4}. (p. 7)
- Patch-wise asymmetric flow: x̄_t,Ri = A_eᵀ x_t^(i) (Eq. 7); û^(i) = P_e û_A^(i) + (I − P_e)(x_t^(i) + û_A^(i)) / σ_t
  (Eq. 9, Fig. 3); A_e fitted by orthogonal Procrustes on multi-scale VAE tokens; P_e = A_e A_eᵀ. (pp. 5–6)
- Extent heads h_e, one per extent, initialised W_e = A_e W_out (Eq. 10). Per-extent data scale s_e. (p. 6)
- Shape embedding s(e) = (log₂ e_h, log₂ e_w, log₂ e_h e_w, log₂ e_h / e_w), zero-initialised MLP g_φ added to the input
  embedding (Eq. 11–12). RoPE index at the patch centre c_i = (u_i + (e_h − 1)/2, v_i + (e_w − 1)/2); for 1 × 1
  tokens this recovers the pretrained RoPE (Eq. 13). (pp. 6–7)
- Models: FLUX.2-klein-base-4B (and 9B) at 1024², Wan2.1-T2V-14B, 81 frames at 720P; LoRA rank 256 on the
  transformer, per-extent heads and patch-size embedding fully fine-tuned. (p. 7, A.4)
- Layout sources: semantic masks, bounding boxes, texture variance (VRS), depth of field, painted / agent maps. (A.3)

## Results
- Tab. 1 (images, LoT-Flux2 4B vs baselines, same budgets): compression 1.893× → speed 1.560×; 2.406× → 1.818×;
  2.935× → 2.040×. At 2.935×: HPSv3 Full 10.4299 · ToMe-SD 0.9253 (2.003×) · DDiT 7.4075 (1.321×) · Foveated 7.8433
  (1.180×) · Ours 9.5522 (2.040×). FID Full 16.748 · Ours 13.345 · Foveated 16.124 · DDiT 28.156 · ToMe 43.528. (p. 8)
- Tab. 2 (video, LoT-Wan2.1 14B): budget 2× → 2.262×, 2.5× → 2.909×, 3× → 3.530× speed. Imaging quality at 3×:
  Full 0.6465 · ToMe 0.4208 · Foveated 0.4952 · Ours 0.5650. (p. 8)
- Headline wording: "up to 2.04× for images and 3.53× for videos" (p. 2).
- Tab. 3 ablation (≈1.52× compression): HPSv3 / FID / MUSIQ — Direct HR 7.0595 / 26.487 / 63.346 · Binary grid
  8.1955 / 20.603 / 66.443 · No size embedding 10.0284 / 14.584 / 69.433 · Ours 10.3477 / 13.800 / 70.410. (p. 9)
- Tab. 4 controllability (2,096 COCO prompts): target-region coverage overall FLUX.2 9B 44.94% → LoT-FLUX.2 9B 66.83%;
  4B 45.38% → 61.17%. (p. 33)
- Fig. 4 (deer row): box 1.73× compression ≈1.54× speed-up · mask 2.18× / 1.81× · texture variance 1.36× / 1.28× ·
  depth 3.11× / 2.29×. (p. 6)
- Fig. 7a (budget 1,269 tokens): Full 1.00× · ToMe-SD 2.10× · DDiT 1.90× · Foveated 1.63× · Ours 2.29×.
  Fig. 7b (budget 50,134 tokens): Full 1.00× · ToMe-SD 1.60× · Foveated 1.59× · Ours 1.62×. (p. 9)
- Fig. 10 (deer, mask-guided): budgets B = 1,024 / 1,792 / 2,560 / 3,328 / 4,096 tokens (4,096 = full). (p. 23)
- Fig. 19 rows: (a) fire hydrant 1,636 tokens 2.50× · (c) bird 1,222 3.35× · (i) cat 1,519 2.70×. (p. 34)
- Fig. 21 rows: 1,111 tokens · 2.2× faster; 1,752 · 1.8×; 758 · 2.8×. (p. 38)
- Fig. 22 limitation: 3,189 tokens (1.28×) vs 1,466 tokens (2.79×), broken staff lines. (p. 39)
- Site, painted demo "girl riding a corgi": 1,546 of 4,096 tokens (1×1: 1,312 · 2×2: 140 · 4×4: 79 · 8×8: 15).
- Site, driving showcase: show03 1.66×, show04 1.97× compression (semantic-mask layouts, LoT-Wan2.1 14B).
