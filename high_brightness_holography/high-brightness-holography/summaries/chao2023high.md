# [chao2023high] High-brightness holographic projection

Brian Chao, Manu Gopakumar, Suyeon Choi, Gordon Wetzstein (Electrical Engineering Department, Stanford University).
Optics Letters 48(15), 4041–4044 (2023), DOI 10.1364/OL.489617. Received 20 Mar 2023, revised 20 Jun, accepted 28 Jun,
published 25 Jul 2023. Funding (acknowledgments): SGF, NSF GRFP, Meta PhD fellowship, Kwanjeong; thanks Sony and Samsung.

## Sources actually obtained (2026-09-27)
- **Main text**: Optica's page is behind a Radware captcha (curl, WebFetch and headless Chrome all blocked). The full HTML
  text (all paragraphs, Eqs. 1–4, all five figure captions, references) was recovered from the Wayback Machine snapshot of
  2023-07-26 → `../pdfs/chao2023high_fulltext_wayback.html`, plain text `../pdfs/chao2023high.txt`.
- **Main-paper figures 1–5 (images) and the main PDF: NOT obtained** (captcha; the image URLs were never archived).
  Hence numbers that only appear inside main Figs. 2–5 (e.g. the PSNR/LPIPS and "+x%" labels of Fig. 5, the Fig. 3 / 4c
  curves) are unknown and are not used. Only their captions are used.
- **Supplement 1** (18 pp., CC-BY 4.0, figshare 10.6084/m9.figshare.23596260) → `../pdfs/chao2023high_supplement.pdf`,
  text `../pdfs/chao2023high_supplement.txt`, figures `../pdfs/figs/figS*_hi.png` (native-resolution embedded rasters;
  vector Figs. S2–S5 rendered at 400 dpi). Page renders `../pdfs/figs/supp_pNN_{hi,lo}.png`.
- **Visualization 1** (figshare 10.6084/m9.figshare.22290445, `supplemental_video_small.mp4`, 720 × 450, 60 fps, 31.8 s)
  → `../../data/video/visualization1.mp4`: split-screen wipes, left label "with loss", right "without loss", six captured
  images (monarch, macaw, bunny, parrot, eagle, swallowtail).
- **Website teaser** `https://bchao1.github.io/images/OL2023.mp4` (600 × 600, 60 fps, 5.3 s) → `../../data/video/OL2023.mp4`:
  crop of the swallowtail wipe (a brighter version sweeps in from the left to the right edge, holds, then sweeps back).
  The brighter half is the "with loss" side (same as Visualization 1).

## Problem
Projectors should combine image quality, brightness and light efficiency. Conventional (category i) displays form images by
**amplitude modulation**: they attenuate light (LCD/DMD), so light from the illumination engine is thrown away. Dual-display
systems (category ii: phase SLM for light reallocation + amplitude SLM, e.g. Damberg 2016, Hoskinson 2010) are bright but cost
more, are bulkier, and have synchronization issues. **Holography** can "reallocate" light anywhere within the sub-hologram of a
point, but prior holographic projectors had speckle / ripple artifacts and never optimized light efficiency in the CGH.

## Method
- Fresnel holography projector (Fig. 1): laser → SLM phase φ → propagate z to the hologram plane → projection lens → screen.
- **Eq. 1**: u_z(x, y, λ) = f(u_SLM(x, y, λ), z), u_SLM = e^{iφ(x, y, λ)} u_src(x, y, λ). f = propagation + projection;
  f_ASM (angular spectrum method) or f_model (camera-calibrated learned propagation model, same design as Choi et al. 2022).
- Learned model: capture train/test sets of SLM phase patterns + amplitude images; fit with SGD (as Choi et al.).
- Phase optimization: minimize L_recon(s · |f_model(e^{iφ}, z)|, a_target), s = scale factor optimized with φ.
- **Eq. 2**: φ^(k) = φ^(k−1) − α (∂L/∂φ)^T L(s · |f_model(u_SLM, z)|, a_target)  (SGD, learning rate α).
- With only L_recon (e.g. L2) "good image quality can be achieved … yet much light is distributed outside of the region of
  interest".
- **Eq. 3** (light-efficiency loss): L_reg = L2(|f(u_SLM, z)|, √(Σ|u_src|² / Σ a_target²) · a_target) — forces all
  illumination power Σ|u_src|² into the ROI while keeping pixel-wise accuracy. Σ|u_src|² is calibrated (supp. 1-D).
- **Eq. 4**: L = L_recon + γ L_reg.

## Hardware (main text + supplement Sec. 1)
- SLM: **TI DLP6750Q1EVM**, **1280 × 800**, pixel pitch **10.8 µm**, **4 bits** per pixel (supp: 16 levels, Gumbel-Softmax as
  in Choi et al. 2022).
- Laser: **FISBA RGBeam** fiber-coupled, three diodes, max output **50 mW**, **636.4 / 517.7 / 440.8 nm** (R / G / B).
- Projection lens **Nikon AF Nikkor 50 mm f/1.4D**, screen **87 cm** from the lens; viewing region **12 cm × 21 cm**
  (supp: 12.4 × 21.1 cm², ROI 700 × 1190 SLM pixels = 7.56 × 12.85 mm²; d1 = 5.3 cm; magnification M = 87 / 5.3 = **16.4**).
- Collimation lens 100 mm; beam splitter; aperture in front of the projection lens filters higher diffraction orders (Fig. S1).
- Colours optimized separately, captured monochrome, combined in post. **Eight-frame time multiplexing** for all holograms.
- Camera: FLIR Grasshopper 2.3 MP colour USB3.
- Propagation distance z ≈ **20 cm**; calibrated **19.7 / 19.9 / 20.1 cm** (R / G / B) ↔ α = 0.633 / 0.639 / 0.643.
- Homography: 18 × 11 dot grid, dots 70 px apart. Laser power calibration: zero-phase pattern capture → Σ|u_src|².

## Sub-hologram cone and d* (supplement Sec. 2)
- Max diffraction angle θ = sin⁻¹(λ / 2p). A pixel's light reaches a disk of radius L on the image plane, L / d = tan θ (Eq. S3).
- Normalized distance **d* = R / tan(sin⁻¹(λ / 2p))**, R = √(H² + W²) p / 2 (Eq. S4); d = α d*. α = 1: centre pixel's cone covers
  the whole image region.
- Prototype: p = 10.8 µm, R = √(7.56² + 12.85²) / 2 = **7.45 mm**, λ = **517 nm** → **d* = 31.11 cm**. Light efficiency peaks
  around **20 cm = 0.643 d*** (Fig. S4). Main text: long distance needed, **z ≥ 0.6 d***.
- **Fig. S4** (simulation, whole test set) — values read from the vector plot's path coordinates (PyMuPDF get_drawings, axes
  calibrated from tick labels 0.0–1.0 / 50–90): x = 0.032, 0.161, 0.322, 0.643, 0.965 d* (= 1, 5, 10, 20, 30 cm);
  **without loss 45, 60, 66, 67, 63 %; with loss 66, 80, 89, 93, 91 %**.

## Software (supplement Sec. 3)
- PyTorch. Phase init U(−0.5, 0.5), s initialized to 1 and updated every iteration in closed form
  s* = a_reconᵀ a_target / a_targetᵀ a_target (Eq. S5 as printed); **2000 iterations, learning rate 0.02**.
- Alg. S1 (SGD-ASM / SGD-model), Alg. S2 (SGD-CITL: replace(|û|, √I_captured) keeps model gradients).
- Learned model (Fig. S5): CNN_SLM → ASM(z) → CNN_target, lut, a_src, φ_src; trained at one z per colour. Batch size **4**,
  learning rate **4e−4**; **4800** SGD-ASM phases per colour channel with captured intensities on the projection screen;
  split train / val / test **8 : 1 : 1**; training ≈ **24 hours** on an NVIDIA A6000.
- Main text limitation: optimizing a phase pattern takes **about 20 minutes** on a single NVIDIA RTX A6000.

## Brightness definitions (supplement Sec. 4)
- s ≈ laser amplitude needed to match the target; small s = low laser power needed = high brightness.
- Full-colour brightness = **1 / (s_R² + s_G² + s_B²)**.
- Conventional display: s² = max(I) / max(L), brightness = 1 / Σ_c max(I_c)/max(L_c) (Eq. S6); max(L) = P / (H × W).
- **Fig. S6** (simulation, fixed power budget, labels read from the figure): conventional vs holographic at 0.17 d*, 0.33 d*,
  0.67 d*: monarch **+1042 %, +1180 %, +1225 %**; macaw **+503 %, +596 %, +663 %**.

## Results
- Main text: loss + long propagation distance "improves image brightness and light usage by **around 30%**"; Fig. 5:
  "effectively increasing the overall image brightness by around 30%", "stray light can be completely eliminated outside of
  the region of interest"; LPIPS retained.
- Fig. 3 caption (simulation): brightness stays constant with sparsity for conventional displays, **hyperbolic increase** for
  holographic ones; holographic displays keep an efficiency of **over 70%**, conventional ones go to **nearly 0%** for
  extremely sparse content; the loss improves efficiency and brightness at every sparsity level. Sparsity = ratio of
  zero-valued pixels. Application: automotive **head-up displays** (sparse text and pictograms).
- Fig. 4 caption: ASM-optimized captures show severe fringe and chromatic artifacts at long distance plus peripheral falloff
  (laser non-uniformity); the learned model fixes both. Loss performance vs distance: "improves from nearly 0% (no light
  reallocation) to **30%**" for sparse images.
- Light efficiency definition: total power leaving the display / total power output by the illumination engine.
- **Fig. S7** (captured, SGD-CITL): long 0.64 d* shows wave-like fringes vs short 0.31 d*.
- **Fig. S8** (captured, loss weight γ = 0.0, 1.0, 100.0, Inf; PSNR/LPIPS in blue, brightness gain in red):
  monarch 19.49/0.44 | +17% 20.10/0.42 | +25% 19.82/0.44 | +23% 19.75/0.45;
  bunny 20.52/0.62 | +18% 20.26/0.59 | +31% 18.74/0.62 | +31% 18.73/0.64;
  eagle 17.86/0.53 | +13% 18.58/0.50 | +26% 18.02/0.52 | +27% 17.78/0.55.
- **Fig. S9** (captured; columns ASM / CITL / Model; top = without loss, bottom = with loss):
  castle: 12.93/0.81 → +0% 13.60/0.79 | 15.95/0.75 → +14% 15.73/0.74 | 16.26/0.64 → +24% 16.14/0.67;
  cat: 15.24/0.75 → +7% 15.91/0.73 | 20.50/0.69 → +25% 19.40/0.63 | 21.41/0.54 → +33% 19.45/0.56;
  swallowtail: 12.28/0.70 → +7% 13.04/0.69 | 16.68/0.66 → +26% 16.22/0.62 | 15.99/0.62 → +32% 15.99/0.59.
  SGD-CITL corrects the illumination non-uniformity but generates fringes; SGD-model is clean with the best brightness gain.
- **Fig. S10** (captured SGD-model, without → with): bunny1 20.80/0.62 → +36% 18.55/0.62; bunny2 17.99/0.74 → +26%
  17.05/0.73; castle 16.26/0.64 → +24% 16.14/0.67; parrot 20.62/0.68 → +34% 19.18/0.65; cat 21.41/0.54 → +33% 19.45/0.56;
  lake 16.77/0.61 → +23% 16.78/0.62; beach 17.73/0.69 → +30% 17.02/0.69; swallowtail 15.99/0.62 → +32% 15.99/0.59;
  glasses (sparse) 18.80/0.50 → +9% 18.93/0.48. Caption: "around **20–30%** for all natural images"; sparse case less significant.
- Supplement Sec. 9: native SLM 800 × 1280, ROI 700 × 1190; ROI larger than the SLM → bad quality (Figs. S11, S13).
  Brightness of a 2D sinusoid improves **10-fold** for a small ROI of **100 × 160** pixels with the loss (Fig. S12, simulation).
- Supplement Sec. 10: average brightness improvement over the whole test set: **66% in simulation, 33% captured**.

## Limitations / outlook (main text)
- ~20 min per phase pattern on one RTX A6000 → an inverse model (image → phase) left for future work.
- Zeroth-order undiffracted light reduces dynamic range / contrast → higher-efficiency SLMs or off-axis setup with a phase
  ramp separating the DC term.
- Video not done; in theory: adjust laser power per frame for the different scale factors s.
- Loss weight is application-dependent: large weight when brightness matters more than quality (e.g. outdoor under sunlight).

## Numbers used on screen (5-min and 1-min videos) and their sources
Paper numbers:
- "Optics Letters 48(15), 4041 (2023)" — citation.
- "around 30% brighter" — main text (twice) and Fig. 5 caption.
- "33% captured / 66% simulated, test-set average" — supplement Sec. 10.
- "over 70%" vs "nearly 0%" efficiency — Fig. 3 caption.
- "+1042% / +1180% / +1225%" (and macaw +503 / +596 / +663%) — labels inside Fig. S6 (shown as the figure itself).
- θ = sin⁻¹(λ/2p), d* formula, p = 10.8 µm, λ = 517 nm, R = 7.45 mm, d* = 31.11 cm, z ≈ 20 cm ≈ 0.64 d*, z ≥ 0.6 d* —
  supplement Sec. 2, main text.
- Fig. S4 curve values 45–67% / 66–93% at 1–30 cm — digitized from the vector plot (see above).
- Eqs. 1–4, "2000 iterations, learning rate 0.02", brightness = 1/(s_R² + s_G² + s_B²) — main text, supplement Secs. 3–4.
- Hardware: 1280 × 800, 10.8 µm, 4 bits, 50 mW, 636.4 / 517.7 / 440.8 nm, 50 mm f/1.4, 87 cm, 12 cm × 21 cm,
  19.7 / 19.9 / 20.1 cm, 8 frames, magnification 16.4 — main text + supplement Sec. 1.
- Model training: 4800 phases per colour, batch 4, lr 4e−4, 8 : 1 : 1, ≈ 24 h on an A6000 — supplement Sec. 3B.
- Fig. S9 / S10 / S8 in-figure labels (PSNR/LPIPS, +x%) — shown as the paper's figures; the storyboard names the ones
  mentioned in captions: castle +0% (ASM) / +14% (CITL) / +24% (model); bunny +36%, cat +33%, swallowtail +32%; glasses +9%;
  γ sweep monarch +17 / +25 / +23%.
- "10-fold brighter for a 100 × 160 ROI" — supplement Sec. 9 (Fig. S12).
- "about 20 minutes per phase pattern (RTX A6000)" — main text.
Toy-simulation numbers (computed by `../../precompute_toy.py`, labelled "toy simulation" on screen) are listed in
`../../storyboard.md` ("Toy simulations"), never presented as paper results.
