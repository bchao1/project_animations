# [chao2025textured] Textured Gaussians for Enhanced 3D Scene Appearance Modeling

Chao, Tseng, Porzi, Gao, T. Li, Q. Li, Saraf, Huang, Kopf, Wetzstein, Kim (Stanford, Meta, UMD College Park)
CVPR 2025, pp. 8964-8974 · arXiv 2411.18625 · 40 pp. incl. supplement · project: https://textured-gaussians.github.io/
PDF: `../pdfs/chao2025textured.pdf` · full text: `../pdfs/chao2025textured.txt` · LaTeX: `../pdfs/src/` (main.tex, sec/*.tex, main.bib) · figures: `../pdfs/src/sec/figures/`

## Problem
In 3DGS every pixel covered by one Gaussian gets the same color up to the Gaussian falloff factor, and the per-Gaussian
scalar opacity only yields ellipsoidal footprints. Fine texture and non-ellipsoidal shapes therefore need many tiny Gaussians,
and quality collapses when the Gaussian budget is small. Prior fixes are either too smooth (Textured-GS, SH-varying color /
opacity, huang2024spatial) or restricted to simple objects (Texture-GS, one global unit-sphere texture, xu2024texturegs).
Idea: borrow texture mapping and alpha mapping from mesh graphics and give each Gaussian its own small texture.

## Method
**3DGS background (Sec. 3.1).**
- Gaussian: G_i(x) = exp(−½ (x − μ_i)ᵀ Σ_i⁻¹ (x − μ_i)), center μ_i ∈ ℝ³, covariance Σ = R S Sᵀ Rᵀ (rotation R, scale S).
- Projection: Σ′ = J W Σ Wᵀ Jᵀ (W world→camera, J local affine Jacobian); Σ_2D = [Σ′]_{1,2},{1,2} (upper-left 2×2 minor).
- Compositing (Eq. 3): c(p) = Σ_{i=1..K} c_i α_i ∏_{j=1..i−1} (1 − α_j), c_i from per-Gaussian SH + view direction.
- Alpha (Eq. 4): α_i = G_i^2D(p) · o_i (o_i = per-Gaussian opacity).

**Textured Gaussians (Sec. 3.2).**
- Each Gaussian gets a fixed-resolution texture map of size T × T × K, T ∈ ℕ, K ∈ {1, 3, 4} = alpha-only, RGB, RGBA (Fig. 2).
  Shape = per-Gaussian opacity × texture alpha; appearance = spatially constant SH (low-frequency + view-dependent / specular)
  + spatially varying texture RGB (high-frequency residual).
- Texture lives on plane P through μ_i spanned by the two major axes; normal n_i = axis with the smallest scale
  (implementation re-picks the two longest axes dynamically, which works better empirically).
- Ray-plane intersection (Eq. 5), ray from camera center o through pixel p:
  x = o + [((μ_i − o) · n_i) / ((p − o) · n_i)] · (p − o) / ‖p − o‖
- UV mapping (Eq. 6), then bilinear interpolation of texels:
  u = [m·σ_i¹ + (x − μ_i) · r_i¹] / (2·m·σ_i¹) · (T − 1)
  v = [m·σ_i² + (x − μ_i) · r_i²] / (2·m·σ_i²) · (T − 1)
  σ_i¹, σ_i² = scales of the two major axes, r_i¹, r_i² = their unit directions, m = texture extent multiplier.
  Implementation: m = 3, i.e. the texture spans ±3σ; intersections outside ±3σ are skipped (c_tex = 0), which avoids integer
  overflow in UV and stabilizes grazing views; the Gaussian's opacity there is < 0.01 anyway.
- Color (Eq. 7): c_i(p) = c_i^base + c_i^tex(u, v), c_base = SH color exactly as in 3DGS.
- Alpha (Eq. 8): α_i(p) = α_i^tex(u, v) · G_i(x) · o_i. Note G_i is the *3D* Gaussian evaluated at the intersection point x
  (exact, ray-based, following GOF), not the splatted 2D approximation of Eq. 4.
- Generalized compositing (Eq. 9): c_final(p) = Σ_{i=1..K} c_i(p) α_i(p) ∏_{j=1..i−1} (1 − α_j(p)).
  c_tex = 0 and α_tex = 1 recover 3DGS.
- Custom CUDA kernels for ray-Gaussian intersection, UV mapping, texture lookup and compositing (PyTorch front end).

**Optimization (Sec. 3.3 + supplement).**
- Loss L = λ L1 + (1 − λ) L_SSIM, λ = 0.8.
- Stage 1: plain 3DGS* for 30000 iterations (default 3DGS learning rates, ADC with GOF sum-of-magnitudes densification score
  Σ_j ‖dL/dp_j · dp_j/dμ′‖ instead of the magnitude-of-sums ‖Σ_j dL/dp_j · dp_j/dμ′‖).
- Stage 2: initialize from the stage-1 model and jointly optimize all Gaussian attributes + textures for another 30000
  iterations with ADC disabled (fixed Gaussian count for fair comparison). Joint single-stage optimization is called highly
  ill-posed; two stages converge faster and better.
- Texture lr 0.001 for all datasets (no per-dataset tuning); RGB texels initialized to 25/255 (texture learns the residual
  over the SH mean color); alpha texels initialized to 1.
- Training time ≈ 2× 3DGS (two stages). Nvidia H100 clusters.
- 3DGS* baseline = authors' GOF-style 3DGS (ray-based exact Gaussian evaluation + revised densification [GOF, AbsGS]).

## Datasets / protocol
Blender (8 synthetic scenes, white bg), Mip-NeRF 360 (all 9), DTU (5 scenes, 800×600, black bg, following Texture-GS),
Tanks and Temples (2), Deep Blending (2) = 26 test scenes; plus 2 custom room-scale 360° captures at a local art center,
"flower gallery" (art installation by Kija Lucas, Palo Alto Art Center) and "children's art". Mip-NeRF 360 / Instant-NGP
numbers from the 3DGS paper (scene datasets) and NeRFBaselines (Blender); DTU from the authors' own runs of released code.
For reduced-Gaussian models a fixed total texel budget is shared, so fewer Gaussians means larger T. Default Table 1 setup
uses 4×4 RGBA textures.

## Key numbers
**Table 1 (PSNR / SSIM / LPIPS; RGBA textures; same number of Gaussians as 3DGS*).**

| Method | Blender | Mip-NeRF 360 | DTU | Tanks and Temples | Deep Blending |
|---|---|---|---|---|---|
| Mip-NeRF 360 | 30.34 / 0.9510 / 0.0600 | 27.69 / 0.7920 / 0.2370 | 34.28 / 0.9641 / 0.0698 | 22.22 / 0.7590 / 0.2570 | 29.40 / 0.9010 / 0.2450 |
| Instant-NGP | 32.20 / 0.9590 / 0.0550 | 25.30 / 0.6710 / 0.3710 | 23.58 / 0.7653 / 0.2603 | 21.72 / 0.7230 / 0.3300 | 23.62 / 0.7970 / 0.4230 |
| Texture-GS | 28.97 / 0.9380 / 0.0550 | – | 30.03 / – / 0.1440 | – | – |
| 3DGS* | 33.08 / 0.9671 / 0.0440 | 27.26 / 0.8318 / 0.1871 | 33.54 / 0.9697 / 0.0551 | 24.18 / 0.8541 / 0.1754 | 28.04 / 0.8940 / 0.2707 |
| Ours | 33.24 / 0.9674 / 0.0428 | 27.35 / 0.8274 / 0.1858 | 33.61 / 0.9699 / 0.0556 | 24.26 / 0.8542 / 0.1684 | 28.33 / 0.8908 / 0.2699 |
| 3DGS* (10%) | 31.47 / 0.9590 / 0.0594 | 25.77 / 0.7796 / 0.2860 | 32.71 / 0.9627 / 0.0811 | 22.82 / 0.8020 / 0.2728 | 27.64 / 0.8853 / 0.3101 |
| Ours (10%) | 32.14 / 0.9629 / 0.0489 | 26.32 / 0.7976 / 0.2323 | 32.74 / 0.9661 / 0.0582 | 23.41 / 0.8259 / 0.2122 | 27.98 / 0.8898 / 0.2804 |
| 3DGS* (1%) | 26.89 / 0.9160 / 0.1165 | 22.37 / 0.6293 / 0.4774 | 30.88 / 0.9320 / 0.1581 | 19.90 / 0.6736 / 0.4406 | 23.97 / 0.8167 / 0.4337 |
| Ours (1%) | 28.11 / 0.9343 / 0.0849 | 23.73 / 0.7064 / 0.3365 | 32.43 / 0.9627 / 0.0694 | 21.10 / 0.7399 / 0.3104 | 24.83 / 0.8454 / 0.3552 |

PSNR gain over 3DGS*: 100%: +0.16 / +0.09 / +0.07 / +0.08 / +0.29 dB; 10%: +0.67 / +0.55 / +0.03 / +0.59 / +0.34 dB;
1%: +1.22 / +1.36 / +1.55 / +1.20 / +0.86 dB (mean ≈ +1.24 dB). At 100% ours loses to 3DGS* on Mip-NeRF 360 SSIM, Deep
Blending SSIM and DTU LPIPS; at 10% and 1% it is strictly better on every metric. Mip-NeRF 360 (the NeRF) still has the best
PSNR on Mip-NeRF 360, DTU and Deep Blending.

**ACCURACY NOTE.** The text (Sec. 4.1) says "nearly 2 dB improvements over 3DGS* when using 1% of the default number of
Gaussians", but Table 1's per-dataset 1% gains are Blender +1.22, Mip-NeRF 360 +1.36, DTU +1.55, T&T +1.20, Deep
Blending +0.86 dB (max 1.55, mean ≈ 1.24). Record both; if quoting a number in the animation, the table-backed
"up to ~1.5 dB (1.2-1.6 dB on four of five datasets)" is safer than "nearly 2 dB".

Figure 5 average Gaussian counts at 1%: ≈ 2.5k (object-level) and ≈ 39k (scene-level); custom captures (Fig. 6) ≈ 100k.

**Table 2 (same model size; PSNR, avg #Gaussians in M).**

| Dataset | 3DGS* | Alpha-only | RGBA |
|---|---|---|---|
| Blender | 33.08 (0.27) | 33.12 (0.19) | 33.03 (0.21) |
| Mip-NeRF 360 | 27.26 (4.4) | 27.37 (3.1) | 27.26 (3.5) |
| DTU | 33.53 (0.28) | 33.45 (0.24) | 33.41 (0.22) |
| Tanks and Temples | 24.17 (2.9) | 24.38 (2.6) | 24.28 (1.3) |
| Deep Blending | 28.04 (2.8) | 28.36 (1.6) | 28.52 (1.0) |

Alpha-only is best on Blender, Mip-NeRF 360, T&T; RGBA best on Deep Blending; 3DGS* best on DTU. Conclusion: there is a sweet
spot for splitting the byte budget between Gaussians and texels (Fig. 9: fitted quadratic over T vs #Gaussians at fixed size,
Blender and DTU).

**Efficiency (supplement Table A.2; 26 scenes, 100% Gaussians, 4×4 RGBA).**
#Gaussians 2.12M vs 2.12M · model size 125.3 MB (3DGS) vs 261.2 MB (ours) · GPU memory 2.4 vs 4.0 GB ·
inference 0.073 vs 0.123 s/frame. Model size relative to 3DGS at equal #Gaussians: alpha-only 1.27×, RGB 1.82×, RGBA 2.08×.
(The supplement text elsewhere says inference time is "approximately the same" as 3DGS, which Table A.2's 0.123 vs 0.073 s
contradicts; the table caption itself says slower.)

**Texture variant ablation (Fig. 8 main: Blender, T&T, Deep Blending; Table A.4 / Fig. A.7: all five).** RGBA best; alpha-only
beats RGB (and 3DGS*) while being one-third the size of RGB and one-fourth of RGBA texture storage, because alpha textures
change both shape (spatially varying opacity) and appearance (spatially varying compositing), while RGB-textured Gaussians
remain ellipsoids. Examples with complex geometry: Blender *ship*, lego truck in Mip-NeRF 360 *kitchen*.
PSNR at 1% (Blender / Mip-NeRF 360 / DTU): alpha 27.64 / 23.69 / 32.31, RGB 27.60 / 23.50 / 32.52, RGBA 28.11 / 23.73 / 32.43
(DTU is the exception where RGB > alpha).

**ACCURACY NOTE (Table A.4).** For the alpha-only / RGB / RGBA rows, the "Tanks and Temples" and "Deep Blending" columns appear
swapped relative to the header: e.g. RGBA (100%) lists T&T 28.34 / 0.8910 / 0.2699 and DB 24.26 / 0.8542 / 0.1685, whereas
Table 1 gives T&T 24.26 / 0.8542 / 0.1684 and DB 28.33 / 0.8908 / 0.2699; RGBA (1%) lists T&T 24.83 and DB 21.08 vs Table 1
T&T 21.10, DB 24.83. The 3DGS* rows in the same table are in the correct order. Read the texture-variant T&T numbers from the
fifth column and DB from the fourth.

**Other supplement results.**
- Table A.1: original 3DGS paper vs 3DGS*: Blender 33.32 vs 33.09; Mip-NeRF 360 27.21 / 0.815 / 0.214 vs 27.28 / 0.832 / 0.187;
  T&T 23.14 / 0.841 / 0.183 vs 24.18 / 0.854 / 0.175; DB 29.41 / 0.903 / 0.243 vs 28.04 / 0.894 / 0.271. Original 3DGS LPIPS
  is flagged as underestimated (Bulò et al., confirmed with 3DGS authors).
- Table A.3: 1 / 2 / 5 / 10 / 20 / 50 / 100% sweeps, fixed texel budget; ours better in PSNR and LPIPS throughout, gap shrinks
  with more Gaussians (Fig. A.4).
- Fixed #Gaussians (1%), varying T (shown T = 2, 5, 20, 40): quality rises with T, dips slightly at very high T (overfitting)
  (Fig. A.5, A.13-A.17). Fixed 4×4 textures, varying #Gaussians (1, 5, 20, 100%): quality rises with count (Fig. A.6, A.18-A.22).
- Effective-rank analysis: Textured Gaussians become mostly flat (rank 2); vanilla GS yields more needle-like (rank ≈ 1)
  Gaussians, justifying the planar texture on 3D Gaussians (Fig. A.2).
- Grazing-angle heuristics: ±3σ intersection clipping and SH base color both reduce artifacts (Fig. A.3).

**Minor internal inconsistencies noticed.** 3DGS* Blender 33.08 (Table 1) vs 33.09 (A.1/A.3/A.4); Mip-NeRF 360 27.26 vs 27.28;
DTU 33.54 (Table 1) vs 33.53 (Table 2). Ours (1%) Blender 28.11 (Table 1) vs 28.02 (A.3); Ours (10%) Mip-NeRF 360 26.32 (Table 1)
vs 26.45 (A.3). Supplement says "24 test scenes" in one place and "26" in another (8+9+5+2+2 = 26). Table 1 cites the NeRF paper
for the Mip-NeRF 360 dataset column. The source references `sec/figures/vary_num_gs_DTU.pdf` while the file on disk is
`vary_num_gs_dtu.pdf` (case mismatch).

## Limitations / future work
2D diffuse textures assume texture lies on a local surface and cannot model spatially varying specular color; extend to local
3D volume textures or 5D radiance fields, possibly via factorized representations (TensoRF, triplanes); time-varying textures
for dynamic scenes. Also (supplement): ≈ 2× training time, ≈ 2× model size / GPU memory with RGBA, slower inference.

## Figure list (source files in `../pdfs/src/sec/figures/`)
Main paper
- Fig. 1 `teaser.pdf` — RGBA Textured Gaussians vs 3DGS with the same Gaussian count; fine artwork detail recovered.
- Fig. 2 `rgba_textures.pdf` — four variants (none / alpha / RGB / RGBA): per-Gaussian texture (top) and rendered primitive (bottom).
- Fig. 3 `model_pipeline.pdf` — pipeline: ray-Gaussian intersection → RGBA texture lookup (c_tex, α_tex at x) → generalized compositing.
- Fig. 4 `num_gaussians_ablation.pdf` — PSNR vs number of Gaussians, 3DGS* vs ours, three scene-level datasets.
- Fig. 5 `main_qualitative.pdf` — benchmark NVS comparison at a small budget (≈ 2.5k object / 39k scene Gaussians).
- Fig. 6 `custom_main_figure.pdf` — custom art-gallery captures, ≈ 100k Gaussians, ours sharper.
- Fig. 7 `color_contribution.pdf` — alpha-modulated, composited c_base and c_tex for alpha / RGB / RGBA at 1% Gaussians vs 3DGS*.
- Fig. 8 `texture_map_ablation_main.pdf` — ΔPSNR over 3DGS* per texture variant, Blender / T&T / Deep Blending, 100% and 1%.
- Fig. 9 `same_model_size_ablation.pdf` — equal-size alpha-only models, T vs #Gaussians trade-off with fitted quadratic (Blender, DTU).

Supplement
- Fig. A.1 `custom_dataset_info.pdf` — sample training views and image counts of "flower gallery" and "children's art".
- Fig. A.2 `effective_rank.pdf` — effective-rank histograms: Textured Gaussians flat (rank 2) vs vanilla GS needle-like (≈ 1).
- Fig. A.3 `intersection_ablation.pdf` — no intersection clipping / no SH base color / full model at grazing views.
- Fig. A.4 `num_gaussians_ablation_supplement_full_metrics.pdf` — PSNR / SSIM / LPIPS vs #Gaussians, all five datasets.
- Fig. A.5 `fix_num_gs_vary_tex_res_ablation_supplement_full_metrics.pdf` — metrics vs texture resolution at 1% Gaussians.
- Fig. A.6 `fix_tex_res_vary_num_gs_ablation_supplement_full_metrics.pdf` — metrics vs #Gaussians at fixed 4×4 textures.
- Fig. A.7 `texture_map_ablation.pdf` — ΔPSNR per texture variant, all five datasets, 100% and 1%.
- Figs. A.8-A.12 `extra_nvs_{blender,mipnerf360,dtu,tandt,db}.pdf` — qualitative 3DGS* vs ours at 100% and 1% Gaussians, per dataset.
- Figs. A.13-A.17 `vary_tex_res_{blender,mipnerf360,dtu,tandt,db}.pdf` — qualitative, fixed #Gaussians, T = 2, 5, 20, 40.
- Figs. A.18-A.22 `vary_num_gs_{blender,mipnerf360,dtu,tandt,db}.pdf` — qualitative, fixed 4×4 textures, 1 / 5 / 20 / 100% Gaussians.
- Figs. A.23-A.27 `tex_ablation_{blender,mipnerf360,dtu,tandt,db}.pdf` — qualitative alpha / RGB / RGBA vs 3DGS* at 1%.
- Figs. A.28-A.32 `color_contribution_{blender,mipnerf360,dtu,tandt,db}.pdf` — per-dataset c_base / c_tex decompositions at 1%.

Tables: 1 (NVS comparison, label tab:same_num_gs), 2 (same model size), A.1 (original 3DGS vs 3DGS*), A.2 (efficiency),
A.3 (full #Gaussians sweep), A.4 (texture variants, T&T / DB columns swapped for textured rows).
