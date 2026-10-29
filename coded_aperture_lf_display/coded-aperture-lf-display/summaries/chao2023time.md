# [chao2023time] Time-Division Multiplexing Light Field Display With Learned Coded Aperture

Brian Chao* (Chun-Hao Chao on the paper), Chang-Le Liu*, Homer H. Chen (Fellow, IEEE) — National Taiwan University
(* co-first authors; corresponding author H. H. Chen). IEEE Transactions on Image Processing **32**, 350–363 (2023),
DOI 10.1109/TIP.2022.3203210, IEEE Xplore document 9987726 (paywalled; the homepage lists it as "IEEE TIP, 2022").
arXiv 2107.06205: v1 13 Jul 2021 ("Learning Coded Apertures for Time-Division Multiplexing Light Field Display"),
v3 23 Aug 2022 (final title; the version ingested here, 15 pp., journal layout with page numbers 11–25).
PDF: `../pdfs/chao2023time.pdf` · text: `../pdfs/chao2023time.txt` · page renders: `../pdfs/pages/pNN.png` (110 dpi)
and `pNN_hi.png` (400 dpi) · embedded rasters: `../pdfs/figs/raw_*.png`. No LaTeX source on arXiv (PDF-only submission).
No code repository or supplementary video exists (searched GitHub + web, 2026-09-27).
Homepage teaser: https://bchao1.github.io/images/tip2021.gif — 256 × 256, 2 frames × 1000 ms, near / far refocus of the
SIGGRAPH16 agapanthus scene (same scene as Fig. 12 bottom / Fig. 17c). Homepage blurb: "Using coded apertures and a
learning-based light field display optimization pipeline to reproduce correct defocus blurs for better accommodation cues."

## Problem (Sec. I, III)
- Stereoscopic displays give binocular disparity (drives **vergence**) but show images at a fixed focal distance →
  wrong **accommodation** cue → vergence-accommodation conflict (VAC): eye strain, headache, even loss of depth perception.
- Alternatives: varifocal [1]–[3] and multifocal [4]–[12] displays need eye trackers / screen stacks; holographic [13]–[15]
  need high-resolution SLMs. Integral-imaging light field displays (II-based LFDs) give correct cues without these.
- "Light field display" in this paper = viewpoint interval smaller than the pupil diameter → user can focus at any depth
  (not multi-view displays for different viewing positions, which do not solve accommodation).
- **SDM-LFD** (space-division, microlens array; Fig. 1b): panel split into sub-areas, one per view → spatial–angular
  trade-off, needs high-precision microlens manufacturing. "For a 3×3 light field display, the actual area that is perceived
  by the viewer is only one-ninth of the total area of the display panel." Spatial resolution = display resolution divided by
  number of lenses.
- **TDM-LFD** (time-division; Fig. 1c): a switchable shutter at the aperture, synchronised with the panel; view L_s shown
  while the small rectangular window at s is open (e.g. top-left view ↔ top-left window). Full panel resolution per view.
  Drawbacks: aperture window small (smaller than the angular sampling interval) → little light; **aliasing in defocus regions
  due to low angular resolution** (Fig. 2a) — sparse views give ghost copies instead of blur.

## Image formation (Sec. III, Eqs. 1–8)
- Generic AR display (Fig. 1a): display panel → main lens → intermediate image plane → eyepiece → virtual image plane.
  Object / image distances z_O, z_I; refocusing by the eye = shifting the intermediate image plane.
- Coherent point source, Fourier optics (Goodman): Eqs. 1–5. M = −z_I / z_O; h = impulse response of the lens system
  = ℱ{ P(λz_I x̃_p, λz_I ỹ_p) · W(z_I, λz_I x̃_p, λz_I ỹ_p) } (pupil × aberration/defocus function),
  W = exp[ j (k/2)(1/z_O + 1/z_I − 1/F)((λz_I x̃_p)² + (λz_I ỹ_p)²) ] (Eq. 5), k = 2π/λ.
- Incoherent display → intensities (Eq. 6): I_I(x_I, y_I) = |h(x_O, y_O)|² ∗ I_O(x_O/M, y_O/M).
- Light field L on a regular grid G of viewpoints on the (s, t) plane, period Δs = (Δs, Δt). Per view a shifted rect aperture
  (Eq. 7): P_s(x) = rect((x − s ⊙ Δs) / aperture width). Refocused retinal image at z_I (Eq. 8):
  **R_{z_I} = Σ_{s∈G} ( |h_{s,z_I}|² ∗ L_s )**.
- Nomenclature (Table I): L(x, y, s, t) 4D light field, (x, y) spatial, (s, t) angular; L_s(x) sub-aperture view;
  I_c = {I_c,1 … I_c,N} encoded light field; P pupil function; ℱ, ℱ⁻¹ non-unitary Fourier transforms.

## Method: CTDM-LFD (Sec. IV; Fig. 1d, Fig. 3)
- Replace grid apertures P_s with learned coded apertures **P̂ = {P̂_1, …, P̂_k}** and the raw views with an encoded light field
  produced by a learned mapping **f** (Table II). Hyper-parameters (Table III): **n** = number of sub-aperture views sampled
  from the light field; **k** = number of encoded images = number of coded apertures (k need not equal n); **l** = spatial
  resolution of the coded apertures (P̂_i ∈ ℝ^{l×l}).
- **Light field encoding network f** (Sec. IV-A, Figs. 5–6): n views V_1..V_n concatenated to M_in ∈ ℝ^{H×W×3n} → CNN →
  M_out ∈ ℝ^{H×W×3k}, split into k images I_c,1..I_c,k ∈ ℝ^{H×W×3}. "These images are not light field views, but rather encoded
  versions of them"; no direct supervision (trained through the focal stack). Architecture: input Conv2d (3, 3, 128, 3n) →
  **10 residual blocks** (Conv2d 3×3 128→128, BatchNorm, ReLU, Conv2d, BatchNorm, skip) → output Conv2d (3, 3, 3k, 128);
  output clamped to [0, 1] with a sigmoid.
- **PSF per display plane** (Eq. 9): h_{P̂_i, z_I} = ℱ{ P̂_i(λz_I x̃_p, λz_I ỹ_p) W(z_I, …) }; each P̂_i optimizable.
  Output of frame i at distance d: |h_{P̂_i,d}|² ∗ I_c,i.
- **Persistence of vision** (Eq. 10): panel shows I_c,1..I_c,k in sequence, shutter switches to the matching P̂_i; the set P̂ is
  fixed for a display. Perceived: **R_{z_I} = Σ_{t=1}^{k} |h_{P̂_t, z_I}|² ∗ I_{c,t} ∈ ℝ^{H×W×3}**.
- **Focal stack** (Sec. IV-D): perceived light field ≡ continuous focal stack; train on a discrete focal stack
  F_c = {R_{z_I,1} … R_{z_I,m}} (m refocused images); ground truth F_g from the full-angular-resolution light field via Eq. 8.
- **Loss** (Sec. IV-E): L1 (Eq. 11) L = ‖F_c − F_g‖₁ → weighted (Eq. 12) L = 𝒲 ⊙ ‖F_c − F_g‖₁, focus measure (Eq. 13)
  U_j = ‖∇_x R_{z_I,j}‖₁ + ‖∇_y R_{z_I,j}‖₁, weights (Eq. 14) 𝒲_j = e^{β Ū_j}, Ū_j = (U_j − min U_j)/(max U_j − min U_j) (Eq. 15).
  β = 0 → all-ones weights; large β emphasises in-focus pixels. Fig. 4 shows normalised focus measures of two refocused images
  (clock scene): high-focus regions move front → back.
- Binary apertures (Sec. V-G2): during training P̂ passed through S(x) = 1/(1 + e^{−tx}) with large temperature t; at inference
  binarised at 0.5 (values > 0.5 → 1).

## Experimental setup (Sec. V)
- Datasets: **INRIA-syn** (INRIA synthetic LF) 39 scenes, 9×9 angular, split 31/8; **INRIA-real** (INRIA Lytro) 63 scenes,
  7×7, split 51/12; **SIGGRAPH16** (Kalantari et al.) 130 scenes, 8×8, split 100/30.
- View sampling (Fig. 7): n = 4 → the four corner views; n = 9 → 3×3 grid (8×8 case approximately 3×3).
  Settings compared: CTDM-LFD (n = 4, k = 4) and (n = 9, k = 8). SDM designs divide the panel into n² patches (n = 2, 3).
- Metrics: PSNR, SSIM, LPIPS on focal stacks computed from the displayed light field (not a single refocused image).
- Training: β = 2 (Secs. V-D, V-E), trained end-to-end from scratch for **10000 epochs**, Adam (β1 = 0.9, β2 = 0.999,
  ε = 1e−8), learning rate 0.001.

## Results
### Table V — PSNR / SSIM / LPIPS (focal stacks)
| Method | INRIA-syn | INRIA-real | SIGGRAPH16 |
|---|---|---|---|
| Baseline (n = 4) | 28.56 / 0.8485 / 0.2626 | 26.39 / 0.8932 / 0.1434 | 32.92 / 0.9285 / 0.1199 |
| CTDM-LFD (n = 4, k = 4) | **32.29 / 0.9374 / 0.1284** | **30.21 / 0.9602 / 0.0736** | **37.57 / 0.9705 / 0.03119** |
| Baseline (n = 9) | 34.73 / 0.9688 / 0.1385 | 36.32 / 0.9872 / 0.0363 | 40.61 / **0.9858** / 0.0397 |
| CTDM-LFD (n = 9, k = 8) | **36.41 / 0.9858 / 0.0632** | **39.97 / 0.9891 / 0.0186** | **41.21** / 0.9739 / **0.0107** |
CTDM-LFD better on all metrics / datasets except SSIM for n = 9 on SIGGRAPH16.
Derived gains (PSNR, CTDM − baseline): n = 4: +3.73 (INRIA-syn), +3.82 (INRIA-real), +4.65 (SIGGRAPH16) dB;
n = 9: +1.68, +3.65, +0.60 dB.
### Qualitative
- Fig. 2: refocused images from TDM-LFD (a) vs CTDM-LFD (b) (INRIA-syn scene, bunny/green object).
- Fig. 10 (INRIA-syn, n = 4): Baseline / Ours / GT with zoom patches — TDM-LFD "severe aliasing in defocus regions due to
  undersampling"; ours "preserves the sharpness of in-focus regions and has natural blur in defocus regions".
- Fig. 11 (INRIA Lytro, n = 4, k = 4; Totoro path + "B S N MOM" letters, near / far focus), Fig. 12 (SIGGRAPH16; poppy,
  agapanthus; near / far focus).
- Fig. 17: continuous focal stacks (n = 4, k = 4), **nine focal planes** per stack, in-focus region moves front → back.
### Optics simulation (Sec. V-F; Figs. 8, 9, 13; Table IV)
- Zemax design: **unit-magnification relay lens**, symmetric, only spherical surfaces, common glasses (Ohara S-NBM51,
  S-LAL56, LAL58); binary aperture at the centre (orange plane in Fig. 8). Optimised **9×9 binary** apertures (l = 9), 4 shown
  in Fig. 9. Table IV (first three lenses, R / T / Nd / Vd): 41.066 / 6.007 / 1.67790 / 50.72; −41.115 / 1.000; −27.799 / 2.912 /
  1.61340 / 44.27; 19.276 / 1.000; 18.834 / 2.000 / 1.69350 / 50.81; 54.311 / 1.000. Fig. 8 scale bar 10 mm.
- Fig. 13: near / far focus, ours vs GT (flower basket scene) — "bokeh … much more natural compared to that of TDM-LFD".
- "Not limited to the relay lens. For most imaging or projector systems, we can similarly put the aperture at their Fourier plane."
### Ablations (Sec. V-G, INRIA-syn; all n = 4, k = 4, l = 9 unless stated)
- **Table VI** (learned components, continuous aperture): Baseline 28.56 / 0.8485 / 0.2626; learned P̂ 29.85 / 0.8762 / 0.2277;
  learned f and P̂ 32.29 / 0.9374 / 0.1284. Text: learned P̂ "over 1 dB" (derived +1.29); adding f "further … nearly 3 dB"
  (derived +2.44 over learned P̂; +3.73 over baseline). Fig. 14 visual.
- **Table VII** (aperture values): binary 31.24 / 0.9249 / 0.1415; continuous 32.29 / 0.9374 / 0.1284 (grey-scale LCoS →
  continuous; binary liquid-crystal array → binary). Fig. 15.
- **Table VIII** (symmetry: three apertures mirrored from the first): symmetric 31.06 / 0.9283 / 0.1413; free 32.29 / 0.9374 / 0.1284.
- **Table IX** (n, k; l = 9), PSNR / SSIM / LPIPS:
  n = 4: k = 4 32.29 / 0.9374 / 0.1284; k = 8 33.12 / 0.9304 / 0.1428; k = 16 32.43 / 0.9341 / 0.1284.
  n = 9: k = 4 33.27 / 0.9723 / 0.0759; k = 8 36.41 / 0.9858 / 0.0632; k = 16 34.15 / 0.9795 / 0.0693.
  (Text-extraction order is ambiguous; this assignment is read from the page render p11: row n = 4 → 32.29, 33.12, 32.43.)
  Larger n better; larger k not necessarily better; required frame rate grows linearly with k.
- **Table X** (β): 0 → 31.25 / 0.9245 / 0.1561; 2 → 32.29 / 0.9374 / 0.1284; 5 → 32.07 / 0.9376 / 0.1248. PSNR highest at β = 2,
  in-focus detail better preserved at β = 5; β is user-tunable. Fig. 16.

## Limitations / discussion (as stated)
- Frame rate: "the required frame rate of CTDM-LFD increases linearly with k" → trade image quality vs refresh rate.
- Binary apertures slightly worse than continuous (31.24 vs 32.29 dB); Zemax could only simulate binary.
- Only 4 / 9 sparse views used for a fair comparison with SDM; using all views "could be better".
- Evaluated in simulation (Fourier-optics model + Zemax optics simulation); no physical prototype in the paper.

## Numbers used on screen (and where they come from)
5-min video (`../../animation_tdmlf_v3.py`) and 1-min cut (`../../teaser_1min/animation_tdmlf_1min_v3.py`):
- **Bibliographic**: "IEEE Transactions on Image Processing 32 (2023)", "32, 350–363 (2023)" (Crossref, DOI 10.1109/TIP.2022.3203210);
  "arXiv 2107.06205"; "National Taiwan University"; "* equal contribution" (co-first authors, p. 1 footnote).
- **Sec. III-A**: "For a 3 × 3 light field, each view is only one-ninth of the panel" / "each view gets 1/9 of the panel".
- **Eqs. 8, 10, 12, 13, 14** shown (typeset from p. 4–6); Eq. 9 shown in simplified form "PSF = |ℱ{P · W(z)}|²".
- **Settings**: n = 4 corner views (Sec. V-D, Fig. 7), k = 4 (Table V / Fig. 3 / Fig. 17), n = 9, k = 8 (Table V), l = 9 → "9 × 9"
  apertures (Fig. 9, Table VI note), β = 2 (Sec. V-C), "nine focal planes" / "focal plane k of 9" (Fig. 17 text, p. 9).
- **Table V** (all shown, S07): 28.56/0.8485/0.2626, 26.39/0.8932/0.1434, 32.92/0.9285/0.1199; 32.29/0.9374/0.1284,
  30.21/0.9602/0.0736, 37.57/0.9705/0.03119; 34.73/0.9688/0.1385, 36.32/0.9872/0.0363, 40.61/0.9858/0.0397;
  36.41/0.9858/0.0632, 39.97/0.9891/0.0186, 41.21/0.9739/0.0107.
  Derived: "+3.7 dB (INRIA-syn)" = 32.29 − 28.56 = 3.73; "+3.8 dB (INRIA-real)" = 30.21 − 26.39 = 3.82;
  "+4.7 dB (SIGGRAPH16)" = 37.57 − 32.92 = 4.65; "about 4 dB". "Better on every metric except one SSIM entry" = paper text
  (n = 9, SIGGRAPH16, SSIM 0.9739 vs 0.9858).
- **Table VI** (S08 bars): 28.56 / 29.85 / 32.29 dB; derived "+1.29 dB" (29.85 − 28.56; paper: "over 1 dB") and "+2.44 dB"
  (32.29 − 29.85; paper: "nearly 3 dB" — the video shows the exact table difference).
- **Table VII**: "31.24 vs 32.29 dB" (binary vs continuous); "binary apertures cost about 1 dB" (1.05).
- **Table VIII**: "31.06 vs 32.29 dB" (symmetric vs free).
- **Table X**: "β = 0 / 2 / 5: 31.25 / 32.29 / 32.07 dB"; "β = 2 gives the best PSNR" (paper text).
- **Table IX** (n = 4 row): "k = 4 / 8 / 16: 32.29 / 33.12 / 32.43 dB"; "frame rate grows (linearly) with k" (Sec. V-G4).
- **β = 2 → "up to e² ≈ 7.4 times more"**: derived from Eq. 14 with Ū ∈ [0, 1] (weights between 1 and e² = 7.39).
- **Figures shown**: Fig. 1(b), 1(c), 2(a), 2(b), 3, 8, 9, 10 (row 1 + zooms), 11 (Lytro letters near focus + zoom), 12 (via the
  website GIF: agapanthus near focus, baseline / ours), 13 (near / far, ours), 17 (chairs, letters, poppy rows).
- **Toy numbers** (not from the paper, labelled "toy" on screen): toy focal-stack PSNR 22.0 dB (baseline) and 30.4 dB (learned
  images + apertures); iteration counts 0–2999; defocus 0–18 waves; disparities ±1 px per aperture cell.
