# Paper Memory Bank
Last updated: 2026-09-29

### [chao2021robust] Robust Light Field Synthesis From Stereo Images With Left-Right Geometric Consistency
- **Authors**: Chun-Hao (Brian) Chao, Chang-Le Liu, Homer H. Chen (National Taiwan University)
- **Venue**: IEEE ICIP 2021, Anchorage, pp. 1844–1848, DOI 10.1109/ICIP42928.2021.9506423
- **URL**: https://ieeexplore.ieee.org/document/9506423 · author PDF https://raw.githubusercontent.com/bchao1/old-website/HEAD/assets/pdf/icip2021.pdf · code https://github.com/bchao1/lf-flow · website teaser https://bchao1.github.io/images/icip2021-results.gif
- **Status**: analyzed (full PDF in pdfs/, text, page renders, embedded figures; summary in summaries/)
- **Topics**: light field synthesis, stereo, view synthesis, disparity warping, 3D CNN refinement
- **Abstract**: Synthesizes a full N × N light field from one stereo pair: a shared disparity CNN warps each stereo view into a light field (left-right consistency loss), the two are merged by distance-weighted alpha blending, and a patch-wise spatial-angular shuffling (PSAS) 3D CNN refines occlusions / non-Lambertian effects.
- **Notes**: Not on arXiv. IEEE Xplore blocks scripted access (AWS WAF challenge), bchao1.github.io/papers/*.pdf now 404; the author PDF (title metadata "ICIP_2021_light_field_v7") is still in the bchao1/old-website repo. The website GIF = 81 raster-ordered 504 × 504 synthesized views of HCI "origami" (verified: row jumps every 9 frames; shift-and-add refocuses wall ↔ origami).
---

### [chao2022view] View Synthesis and Optical Distortion Compensation for Near-Eye Light Field AR Display
- **Authors**: Chun-Hao Chao, Yi-Chou Chen, Chang-Le Liu, Tz-Wei Hsu, Kuang-Tsu Shih, Homer H. Chen
- **Venue**: ISMAR 2022 / IEEE TVCG (journal extension)
- **URL**: https://raw.githubusercontent.com/bchao1/old-website/HEAD/assets/pdf/tvcg2022.pdf
- **Status**: discovered (PDF + text in pdfs/chao2022tvcg_related.*)
- **Topics**: light field synthesis from stereo, light field AR display, distortion compensation
- **Abstract**: Extends the ICIP pipeline into a stereo → light field system for near-eye LF AR glasses, plus optical distortion compensation of the display.
- **Notes**: Same code base (lf-flow). Mentioned only as follow-up in the outlook.
---

### [srinivasan2017learning] Learning to Synthesize a 4D RGBD Light Field from a Single Image
- **Authors**: P. P. Srinivasan, T. Wang, A. Sreelal, R. Ramamoorthi, R. Ng
- **Venue**: ICCV 2017
- **Status**: discovered
- **Topics**: single-image light field synthesis, depth estimation
- **Abstract**: Predicts 4D ray depths from one image, renders a Lambertian LF, then a 3D CNN adds occlusions / non-Lambertian residuals.
- **Notes**: Baseline in Table 1 (1 view); ICIP disparity net g follows its design; "baseline 3DCNN" ablation = its refinement net. 2.94M params, 0.113 s.
---

### [kalantari2016learning] Learning-Based View Synthesis for Light Field Cameras
- **Authors**: N. K. Kalantari, T.-C. Wang, R. Ramamoorthi
- **Venue**: ACM TOG (SIGGRAPH Asia) 2016
- **Status**: discovered
- **Topics**: light field angular super-resolution, view synthesis
- **Abstract**: Two CNNs (disparity, color) synthesize each novel view from the 4 corner views.
- **Notes**: Baseline in Table 1 (4 corner views); one view per pass → 34.51 s for 7 × 7.
---

### [zhang2015light] Light Field from Micro-Baseline Image Pair
- **Authors**: Zhoutong Zhang, Y. Liu, Q. Dai
- **Venue**: CVPR 2015
- **Status**: discovered
- **Topics**: light field from stereo, phase-based view synthesis
- **Abstract**: Non-learning, phase-based iterative disparity refinement from two micro-baseline views; warps the left view.
- **Notes**: Baseline (horizontal views only); 514.84 s on CPU.
---

### [zhou2018stereo] Stereo Magnification: Learning View Synthesis using Multiplane Images
- **Authors**: T. Zhou, R. Tucker, J. Flynn, G. Fyffe, N. Snavely
- **Venue**: ACM TOG (SIGGRAPH) 2018
- **Status**: discovered
- **Topics**: MPI, stereo view extrapolation
- **Notes**: Needs camera poses at train and test time (ICIP's argument against it).
---

### [wu2017light] Light Field Reconstruction Using Deep Convolutional Network on EPI — Wu et al., CVPR 2017
- **Status**: discovered · **Topics**: angular super-resolution, EPI
---

### [wang2018end] End-to-End View Synthesis for Light Field Imaging with Pseudo 4DCNN — Wang et al., ECCV 2018
- **Status**: discovered · **Topics**: angular super-resolution
---

### [liu2020light] Light Field Synthesis by Training Deep Network in the Refocused Image Domain — C.-L. Liu, K.-T. Shih, J.-W. Huang, H. H. Chen, IEEE TIP 29, 2020
- **Status**: discovered · **Topics**: angular super-resolution (same NTU group)
---

### [honauer2016dataset] A Dataset and Evaluation Methodology for Depth Estimation on 4D Light Fields — Honauer, Johannsen, Kondermann, Goldluecke, ACCV 2016
- **Status**: used as data (HCI 4D LF benchmark; 9 × 9 views, 512 × 512; CC BY-NC-SA 4.0). The paper's 20/4 split = additional (16) + training (4) / test (4).
---

### [ng2005light] Light Field Photography with a Hand-held Plenoptic Camera — R. Ng et al., Stanford Tech Report / thesis 2005
- **Status**: discovered · **Topics**: plenoptic camera, refocusing, spatial-angular trade-off
---

### [chao2023time] Time-Division Multiplexing Light Field Display With Learned Coded Aperture — Chao*, Liu*, Chen, IEEE TIP 32, 2023
- **Status**: analyzed (see ../../coded_aperture_lf_display/) · **Topics**: light field display (same group, same era)
---
