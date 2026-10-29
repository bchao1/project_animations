# Robust Light Field Synthesis From Stereo Images With Left-Right Geometric Consistency

Chun-Hao (Brian) Chao, Chang-Le Liu, Homer H. Chen · National Taiwan University · IEEE ICIP 2021, pp. 1844–1848
Sources: author PDF `../pdfs/chao2021robust.pdf` (5 pages; text `../pdfs/chao2021robust.txt`; page renders `../pdfs/pages/`;
embedded figures `../pdfs/figs/`), code github.com/bchao1/lf-flow (cloned read-only to `../../scratch/lf-flow/`),
website teaser GIF bchao1.github.io/images/icip2021-results.gif.

**Every number shown on screen must come from this file.**

## One-line
Two photos (a stereo pair) → a full 9 × 9 (or 7 × 7) light field: warp each photo with a learned disparity map, merge the two
warped light fields by how far each view is from each photo, and clean up occlusions with a cheap 3D CNN.

## Problem framing (Sec. 1)
- LF imaging → refocusing, view change. Capture: plenoptic cameras (spatial-angular resolution trade-off [Ng 2005]) or camera
  arrays (bulky, costly).
- Angular super-resolution (e.g. 3 × 3 → 7 × 7, "reconstruct a 7*7 light field by interpolating 3*3 sampled views") assumes a
  grid of views can be captured.
- Single image (Srinivasan et al. ICCV 2017): absolute depth from one image is ill-posed → hard for AR.
- Stereo: easy to capture (stereo cameras, or translate one phone camera horizontally); absolute depth available → AR headsets.
- Prior stereo → LF: Zhang et al. CVPR 2015 (phase-based, iterative, slow, only uses the left view); Zhou et al. 2018 MPI
  (needs camera poses at train + test time).

## Contributions (verbatim structure)
1. High-quality LF via left-right consistency of stereo images + distance-weighted alpha blending.
2. Patch-wise spatial-angular shuffling (PSAS) 3D CNN refinement with far fewer parameters.
3. Generalises to real-world stereo; refocused images look good.

## Method (Sec. 2)
Notation: LF F ∈ R^{N×N×H×W×3}; view V(i, j), 0 ≤ i, j < N. S_l = V(N/2, 0), S_r = V(N/2, N−1) (middle row, left- and
right-most views; code mode `stereo_wide`).
1. **Disparity + warping**: g(M_lr) = D_lr ∈ R^{H×W×1}, M_lr = concat(S_l, S_r) ∈ R^{H×W×6}. Backward-warp S_l to every view
   → F_l. Same network with swapped input order: g(M_rl) = D_rl → warp S_r → F_r.
   Code: disparity ∈ tanh·max_disparity (HCI 32, INRIA-DLFD 40, INRIA-Lytro 10) divided by the stereo baseline (8 views) = unit
   (per-view-step) disparity; view (i, j) = grid_sample(S, x + D·(j − j_s), y + D·(i − i_s)).
2. **Losses**: Eq. 1 L_c = ‖F_l − F_t‖₁ + ‖F_r − F_t‖₁ (left-right consistency: both must match the target).
   Eq. 2 L_t = ‖∇x D_lr‖₁ + ‖∇y D_lr‖₁ + ‖∇x D_rl‖₁ + ‖∇y D_rl‖₁ (TV smoothness).
3. **Distance-weighted alpha blending**: Eq. 3 K_l(i, j) = ‖(i, j) − (N/2, 0)‖₂, Eq. 4 K_r(i, j) = ‖(i, j) − (N/2, N−1)‖₂,
   A_l = K_r / (K_r + K_l), Eq. 5 F_c = F_l · A_l + F_r · (1 − A_l).
   Motivation Fig. 2: MSE of views from F_l grows with distance from the left view (columns to the right), F_r mirrored; the
   blended LF has low error everywhere (9 × 9 heat maps).
4. **Refinement r = PSAS 3DCNN**: five PSAS blocks; block = 3D conv → axis transpose → 3D conv (3 × 3 × 3). M ∈ R^{N²×H×W×3};
   reshape to (N′ × N′) × (N′ × H/N′) × (N′ × W/N′) × 3, swap axes 0/2 with 1/4, reshape back to N′² × H × W × 3.
   Code (`LFRefineNetV2.spatio_angular_shuffle`): (b, n, n, c, n, h/n, n, w/n) → permute (0, 4, 6, 3, 1, 5, 2, 7): new
   "view" (p, q) = spatial patch (p, q) of size h/n × w/n, tiled from all n × n angles → a 3 × 3 conv on it mixes angular
   neighbours; applying the shuffle again restores the LF. Code net: conv1 → shuffle → conv2 → shuffle → conv3 → shuffle → conv4
   → shuffle → conv5 (tanh), residual add to F_c, clamp to [−1, 1]; views are the Conv3d channels, colour is the depth axis;
   hidden channels = num_views (81). Requires H, W divisible by N (135 = 15 · 9, 504 = 56 · 9).
   Eq. 6 L_r = ‖F_o − F_t‖₁; Eq. 7 L = w_c L_c + w_t L_t + w_r L_r.

## Implementation (Sec. 3.2)
Disparity net similar to Srinivasan et al.; tanh last layer; ELU; Adam β₁ = 0.9, β₂ = 0.999 (ε printed as "10⁸", a typo for
10⁻⁸); w_c = 1, w_t = 0.001, w_r = 1; 2000 epochs; lr 0.001; random 135 × 135 patches. End to end from scratch.

## Data (Sec. 3.1)
INRIA Lytro 1G (46 train / 17 test), HCI 4D LF (20 / 4), Stanford LF Archive (9 / 2). INRIA small disparity, Stanford large.
HCI 20 / 4 = `additional` (16) + `training` (4) / `test` (4: bedroom, bicycle, herbs, origami).

## Results
**Table 1** (PSNR / SSIM):
| Method | INRIA | HCI | Stanford |
|---|---|---|---|
| Srinivasan (full LF, 1 view) | 27.43 / 0.7959 | 23.60 / 0.5805 | 17.75 / 0.5846 |
| Kalantari (full LF, 4 corners) | 27.08 / 0.7565 | 23.18 / 0.5032 | 23.78 / 0.7167 |
| **Ours (full LF, 2 views)** | **32.18 / 0.8672** | **29.73 / 0.8277** | **25.85 / 0.8337** |
| Zhang (horizontal views) | 28.87 / 0.9119 | 23.06 / 0.8362 | 25.72 / **0.9328** |
| **Ours (horizontal views)** | **34.18 / 0.9400** | **31.50 / 0.9118** | **29.06** / 0.9195 |

Derived gains over the best full-LF baseline: INRIA +4.75 dB (vs Srinivasan 27.43), HCI +6.13 dB (vs Srinivasan 23.60),
Stanford +2.07 dB (vs Kalantari 23.78). Horizontal views vs Zhang: +5.31 / +8.44 / +3.34 dB. Zhang keeps the higher SSIM on
Stanford (0.9328 vs 0.9195) — do not claim a clean sweep on SSIM.

**Table 2** (INRIA, 7 × 7): Srinivasan 2.94M params, 0.113 s (GPU); Kalantari 1.64M, 34.51 s (GPU); Zhang non-learning,
514.84 s (CPU); Ours 1.74M, 0.235 s. Text: "44% fewer parameters compared to Srinivasan" — 1 − 1.74/2.94 = 40.8 %, so the 44 %
is not consistent with the table; on screen show the raw 1.74M vs 2.94M. Derived: 34.51 / 0.235 ≈ 147× faster than Kalantari
(one pass for the whole LF vs one view at a time); slower than Srinivasan (two warps + merge).

**Fig. 3**: top-left and bottom-right views, GT / Kalantari / Srinivasan / Ours; scenes identified: HCI bicycle + origami
(row 1), INRIA Lytro android-on-guitar + pot (row 2), Stanford flower + amethyst (row 3). The caption says the rows are
"INRIA, HCI, and Stanford", but bicycle and origami are HCI test scenes — on screen say only "INRIA, HCI and Stanford scenes".
**Fig. 4**: real smartphone stereo (camera translated horizontally), model trained on INRIA, refocused front / middle / back.

**Table 3** (ablation, PSNR / SSIM, INRIA | HCI):
| Variant | INRIA | HCI |
|---|---|---|
| w/o left-right consistency | 32.07 / 0.8641 | 29.71 / **0.8338** |
| w/o disparity smoothing | 32.11 / 0.8648 | 29.48 / 0.8176 |
| F_l only (left view) | 30.67 / 0.8613 | 27.50 / 0.8316 |
| F_r only (right view) | 31.14 / 0.8307 | 27.41 / 0.7105 |
| (F_l + F_r)/2 average | 31.92 / 0.8659 | 28.95 / 0.8019 |
| baseline 3DCNN (Srinivasan's) | 31.33 / 0.8603 | 29.54 / 0.8211 |
| **full model** | **32.18 / 0.8672** | **29.73** / 0.8277 |
Derived (PSNR, INRIA / HCI): alpha vs average +0.26 / +0.78; full vs F_l only +1.51 / +2.23; PSAS vs baseline 3DCNN +0.85 /
+0.19; LR consistency +0.11 / +0.02; smoothing +0.07 / +0.25. Biggest wins: using both views + alpha blending, PSAS on INRIA.

## Our re-run (not paper numbers; label "our re-run" on screen)
`scratch/repro/train_hci.py`: authors' `DisparityNet(81)` + `LFRefineNetV2(81, 81, 81)` from lf-flow, paper losses and
weights, HCI 20 / 4 split, 135 × 135 crops, batch 8, Adam lr 1e-3, 8000 steps (≈ 3200 crop-epochs), A6000.
Parameter count: 852,514 (g, incl. its unused attention head) + 886,788 (PSAS) = 1,739,302 = the paper's 1.74M (Table 2).
Code vs paper: code PSAS = 5 convs with 4 shuffles between them (paper text: five blocks of conv-transpose-conv); code
halves L_c (× 0.5) — the re-run uses the paper's Eq. 1 without the 0.5. Test metrics: `data/generated/repro/metrics.json`.

## Website teaser
`icip2021-results.gif`: 81 frames, 100 ms, 504 × 504, raster order over the 9 × 9 grid (row jumps every 9 frames) = the
synthesized light field of HCI origami. Verified: shift-and-add over the 81 frames refocuses the wall (a ≈ −1.5 px/view) and
the origami crane (a ≈ +1 px/view).

## Weak spots / assumptions (for the analyst view, not all on screen)
- Supervised: needs dense LF ground truth; trained per dataset (disparity scale `max_disparity` per dataset).
- Only horizontal stereo baseline; vertical parallax is extrapolated from the same horizontal disparity (Lambertian, 2-plane
  geometry assumption) — refinement must fix the rest.
- Backward warping with a single disparity per source view cannot fill disocclusions; blending relies on the other view seeing
  them.
- Fixed geometric alpha (not content-aware); the code has an unused learned attention branch (`learned_alpha`).
- PSAS needs H, W divisible by N.

## Follow-up
ISMAR / TVCG 2022 "View Synthesis and Optical Distortion Compensation for Near-Eye Light Field AR Display" (Chao, Chen, Liu,
Hsu, Shih, Chen) builds the stereo → LF synthesis into a near-eye LF AR display system.
