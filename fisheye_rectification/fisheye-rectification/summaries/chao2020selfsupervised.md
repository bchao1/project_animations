# chao2020selfsupervised — Self-Supervised Deep Learning for Fisheye Image Rectification

Brian Chao*, Pin-Lun Hsu*, Hung-Yi Lee, Yu-Chiang Frank Wang. (* equal contribution)
1 Department of Electrical Engineering, National Taiwan University, Taipei, Taiwan; 2 ASUS Intelligent Cloud Services (Wang).
IEEE ICASSP 2020, pages 2248–2252. DOI 10.1109/ICASSP40776.2020.9054191. Not on arXiv; no open-access copy.
PDF: `../pdfs/chao2020selfsupervised.pdf` (IEEE Xplore, fetched from the Stanford network); text `../pdfs/chao2020selfsupervised.txt`;
figures at native resolution `../pdfs/figs/fig1_architecture_hi.png`, `fig2_results_hi.png`; page renders `page1..5.png` (300 dpi).
Code: https://github.com/ByronHsu/FEGAN (cloned to `../../scratch/FEGAN`, read-only reference).
Website teaser: https://bchao1.github.io/images/icassp2020-result.gif (142 × 142, 2 frames × 800 ms: fisheye church → rectified).

## Problem
Single-image fisheye rectification. Fisheye cameras have a large field of view but strong geometric distortion, "especially
around the outer regions of the image"; this can degrade recognition and segmentation "since the pin-hole camera model
assumption cannot be held" (Sec. 1). Early works need intrinsic/extrinsic parameters from multi-view images [1, 2]; single-image
classical methods (division model [3], curved arcs [4], Hough transform [5], plumb-line [4, 6, 7]) rely on hand-crafted low-level
features. Deep methods [8, 9, 10] need supervision: "ground truth distortion-free images, distortion parameters, line segments,
segmentation maps, ground truth flow, or paired image data", and "all of these works assume that there is only one fisheye
distortion model".

## Method: Fisheye GAN (FE-GAN) (Sec. 2, Fig. 1)
- Domains: distorted X, distortion-free Y; unpaired training sets {x_i}, i = 1..N ⊂ X and {y_i} ⊂ Y.
- Generator G = (U, W). Flow-prediction module U : X → F outputs a distortion flow f_i = U(x_i), an H × W × 2 map ("pixel-wise image
  coordinate mapping between x_i and the predicted distortion-free image x̃_i"). Warping module W : (X, F) → Y, x̃_i = W(x_i, f_i).
  (Code: `fake = F.grid_sample(real, flow + grid)`, i.e. x̃(p) = x(p + f_p) with bilinear sampling; flow predicted at half
  resolution and upsampled — code detail, not on screen.)
- U: cross-connection (FlowNet-like [12]) coarse-to-fine architecture + self-attention layers [18] ("fisheye distortion is viewed as
  an image-level characteristic"). W: Spatial Transformer Network [19], fully differentiable resampling (Sec. 3.2).
- Discriminator D = (D_adv, D_cls), sharing initial layers, down-sampling / strided convolutions. D_adv: real vs fake (G(x) vs y).
  D_cls: distorted vs distortion-free; trained adversarially — views distortion-free images as real and generated outputs as distorted.
- Eq. 1: L_adv(G, D) = E_y[log D_adv(y)] + E_x[log(1 − D_adv(G(x)))]
- Eq. 2: L_cls(G, D) = E_y[log D_cls(y)] + E_x[log(1 − D_cls(x))] + E_x[log(1 − D_cls(G(x)))]
- Cross-rotation consistency (inspired by GcGAN [13]): rotate by θ ∈ {90, 180, 270} degrees (R_θ).
  Eq. 3: L_geo_c(G) = E_x[‖R_θ(G(x)) − G(R_θ(x))‖₁] + E_x[‖G(x) − R_−θ(G(R_θ(x)))‖₁]
  Eq. 4: L_flow_c(G) = E_x[‖U(x) − U(R_θ(x))‖₁] ("Since the fisheye distortion flow is radial-symmetric, it is expected to be
  invariant to the rotated versions of the input image").
- Intra-image warping consistency, assuming the optical axis goes through the image centre:
  radial-symmetric (flow points outwards or inwards w.r.t. the centre; equal magnitudes for equidistant pixels) and smooth.
  Eq. 5: L_smooth(G) = E_x[∫_f ‖∇_x f_p‖₁ + ‖∇_y f_p‖₁]
  Eq. 6: L_radial(G) = E_x[∫_f |f_p·p / (‖f_p‖·‖p‖)| + ∫_r Var({‖f_p‖ : ‖p‖ = r})]
  NOTE: as typeset, minimising the |cos| term would push flow to be tangential; the text says flow points in/outwards and the
  code (`radial_constraint`) returns −mean|cos| (i.e. maximises alignment). On screen we describe the intent ("|cos| → 1:
  flow along the radius", "variance on each circle → 0") and show the two ingredients, not a sign-ambiguous typeset form.
- Eq. 7: G = arg min_G max_D L(G, D), L = sum of losses (1)–(6).
- Loss weights exist only in the code (train_gcgan.sh: λ_gc 1, λ_smooth 2, λ_crossflow 2, λ_radial 0.5, λ_rot 0.1) — NOT on screen.

## Experiments (Sec. 3)
- Datasets built from MCindoor20000 [14], Wireframe [15], LSUN Churches [16]. Fisheye transform of Basu & Licardie [17] with
  randomly determined parameters → **1000 fisheye images per dataset**. Ground-truth pairs are NOT used for training
  (only for evaluation).
- Baselines: CycleGAN [20], UNIT [21] (quantitative + visual); Aleman-Flores [5] (visual only, Fig. 2).
- Metrics: PSNR ↑, SSIM ↑, RMSE ↓ between outputs and ground-truth distortion-free images.

### Table 1 (PSNR / SSIM / RMSE)
| Method | MCindoor | Wireframe | LSUN Churches |
|---|---|---|---|
| CycleGAN | 11.404 / 0.335 / 71.902 | 11.505 / 0.257 / 68.549 | 10.982 / 0.193 / 72.683 |
| UNIT | 12.959 / 0.44 / 61.387 | 8.965 / 0.215 / 93.236 | 10.336 / 0.235 / 79.812 |
| FE-GAN (ours) | **15.215 / 0.498 / 45.672** | **12.846 / 0.352 / 59.887** | **12.781 / 0.377 / 59.492** |
(UNIT MCindoor SSIM is printed as "0.44" in the paper; shown as 0.44.)

Derived (on screen, labelled as over the best baseline per dataset):
- MCindoor: best baseline PSNR UNIT 12.959 → +2.256 dB (shown "+2.3 dB"); vs CycleGAN +3.811 dB.
- Wireframe: best baseline CycleGAN 11.505 → +1.341 dB ("+1.3 dB").
- LSUN Churches: best baseline CycleGAN 10.982 → +1.799 dB ("+1.8 dB").
- RMSE MCindoor: 45.672 vs best baseline 61.387 (UNIT) → −15.715 ("−15.7").

### Table 2 ablation (MCindoor; PSNR / SSIM / RMSE)
| Variant | PSNR | SSIM | RMSE |
|---|---|---|---|
| Ours* (w/o self-attention layer) | 12.388 | 0.361 | 62.184 |
| Ours* (w/o cross-rotation consistency) | 11.021 | 0.324 | 72.889 |
| Ours* (w/o intra-image warping) | 12.38 | 0.303 | 62.319 |
| Ours (full version) | 15.215 | 0.498 | 45.672 |
Derived: removing cross-rotation costs 15.215 − 11.021 = 4.194 dB ("−4.2 dB"), the largest drop; w/o intra-image warping has
the lowest SSIM (0.303). (The paper prints 12.38, not 12.380.)

### Fig. 2
5 rows (fisheye input, CycleGAN, UNIT, Aleman-Flores, ours) × 9 examples (3 MCindoor, 3 Wireframe, 3 LSUN Churches).
Paper: CycleGAN/UNIT translate style (colour, texture), not geometry; UNIT hallucinates; Aleman-Flores (single distortion
model, fixed number of parameters) under-corrects. Crops in `data/fig2_<row>_<mc|wf|ch><k>.png`.

## Claims used on screen
- "among the first to propose self-supervised deep learning models for fisheye image rectification" (contributions).
- No ground-truth camera parameters and no distorted / ground-truth image pairs during training.
- Predicting a flow instead of the parameters of one distortion model (paper: prior works assume one model).
- Assumption: optical axis through the image centre (Sec. 2.2). Training/evaluation data: synthetic fisheye transform [17].

## Background facts (textbook optics, not from the paper; shown as standard projection models)
- Pinhole (rectilinear): r = f·tan θ. Equidistant fisheye: r = f·θ. Equisolid: r = 2f·sin(θ/2). Stereographic: r = 2f·tan(θ/2).
  Orthographic: r = f·sin θ. θ = angle of the incoming ray to the optical axis; r = distance from the image centre.

## Every number on screen and where it comes from
Paper: Table 1 (27 values above), Table 2 (12 values above), "1000 fisheye images per dataset" (Sec. 3.1),
"θ ∈ {90°, 180°, 270°}" (Sec. 2.2, Eq. 3), "H × W × 2" flow (Sec. 2.1), "IEEE ICASSP 2020, pp. 2248–2252", equation numbers (1)–(7).
Derived from Table 1/2: +2.3 dB (MCindoor, vs UNIT 12.959), +1.3 dB (Wireframe, vs CycleGAN 11.505), +1.8 dB (LSUN Churches,
vs CycleGAN 10.982), "1.3 to 2.3 dB", "4.2 dB" (w/o cross-rotation), "+2.3 dB" in the 1-min cut.
Textbook optics (labelled as projection models): r = f tan θ, f θ, 2f sin(θ/2), 2f tan(θ/2), f sin θ; "180° view", "90°",
circle radius f π/2 (image of θ = 90° under r = f θ).
Website/citation: "Basu & Licardie (1995)" = paper ref. [17]; refs [8, 9, 10] named FishEyeRecNet, Xue et al., Li et al.

## Toy computations in the video (ours, labelled "toy" on screen)
Details in `../../storyboard.md` ("Toy computations"). Values shown:
- ray angle at the image edge / corner: 45° / 55° (pinhole, f = half width) → 57° / 81° (equidistant fisheye); live in between.
- S06: mean |R(G(x)) − G(R(x))| (0–255): radial rectifier 0.00 (2e−6), x-only rectifier 27.0.
- S07: smoothness 23.15 / 0.66 / 0.66, direction |cos| 0.92 / 0.00 / 1.00, circle variance 19.37 / 0.10 / 0.10
  (noisy / swirl / radial), in pixels of a 512² image.
- S07: flow strength 0.00–1.50 × true flow; error axis 0–50 (blurred RMSE: 38.6 at 0, min 1.9 at 0.99, 42.2 at 1.5).
- Arrow scale factors ×0.5, ×0.7, ×1.6; toy grid 512 × 512.
