# Paper Memory Bank
Last updated: 2026-09-24

### [chao2024etendue] Large Étendue 3D Holographic Display with Content-adaptive Dynamic Fourier Modulation
- **Authors**: Brian Chao, Manu Gopakumar, Suyeon Choi, Jonghyun Kim, Liang Shi, Gordon Wetzstein (Stanford University; NVIDIA; MIT)
- **Venue**: SIGGRAPH Asia 2024 Conference Papers (Tokyo), DOI 10.1145/3680528.3687600; arXiv 2409.03143 (v1 2024-09-05, v2 2024-11-24, cs.GR)
- **URL**: https://arxiv.org/abs/2409.03143 (project: https://bchao1.github.io/holo_dfm/, video: https://www.youtube.com/watch?v=Rt1f18nBkAc)
- **Citations**: n/a
- **Status**: analyzed (full text + LaTeX source ingested; see summaries/chao2024etendue.md)
- **Topics**: holographic near-eye displays, étendue expansion, multi-source illumination, Fourier-plane modulation, light field CGH
- **Abstract**: Current holographic near-eye displays support only a small étendue, forcing a tradeoff between field of view and eyebox size. Existing étendue expansion is limited in image quality or needs extremely fast SLMs. The paper combines multiple coherent sources with content-adaptive amplitude modulation of the hologram spectrum in the Fourier plane, and optimizes time-multiplexed phase + amplitude patterns with a pupil-aware gradient-descent CGH algorithm supervised by a large-baseline target light field. Better image quality and étendue than baselines in simulation and on a prototype.
- **Notes**: Source paper for the animation in this folder. Prototype: 3×3 fiber-tip source array (8.17 mm spacing, 200 mm collimator, 2.34° incidence), TI DLP6750Q1EVM 4-bit phase SLM, 1080p LCoS amplitude SLM at the Fourier plane (75 mm FT lens), 7.78° diagonal FoV, 8.53 × 8.53 mm eyebox.
---

### [kuo2020mask] High Resolution Étendue Expansion for Holographic Displays
- **Authors**: Grace Kuo, Laura Waller, Ren Ng, Andrew Maimone
- **Venue**: ACM TOG 39(4), Article 66 (SIGGRAPH 2020)
- **URL**: https://dl.acm.org/doi/10.1145/3386569.3392414
- **Status**: discovered
- **Topics**: post-SLM étendue expansion, scattering masks
- **Abstract**: A static scattering element in front of the SLM expands the cone of light (étendue), decoupling FoV and eyebox; new algorithms support dense photorealistic imagery at the SLM's native resolution with up to 20 dB PSNR gain over baseline scattering methods.
- **Notes**: Baseline II ("single source with phase mask") in the paper; mask-based methods lack degrees of freedom → low contrast, speckle.
---

### [jo2022binary] Multi-illumination 3D holographic display using a binary mask
- **Authors**: Youngjin Jo, Dongheon Yoo, Dukho Lee, Minkwan Kim, Byoungho Lee
- **Venue**: Optics Letters 47(10), 2482–2485, 2022
- **URL**: https://doi.org/10.1364/OL.455348
- **Status**: discovered
- **Topics**: pre-SLM étendue expansion, multi-source illumination, Fourier mask
- **Abstract**: Multiple laser diodes illuminate the SLM from different angles simultaneously to expand bandwidth; a fixed random binary mask at the Fourier plane breaks the correlation between the repeated spectrum copies.
- **Notes**: Direct inspiration; baseline IV. Mask is static and not content-adaptive; the target paper replaces it with a dynamic amplitude SLM.
---

### [lee2020wide] Wide-angle speckleless DMD holographic display using structured illumination with temporal multiplexing
- **Authors**: Byounghyo Lee, Dongheon Yoo, Jinsoo Jeong, Seungjae Lee, Dukho Lee, Byoungho Lee
- **Venue**: Optics Letters 45(8), 2148–2151, 2020
- **URL**: https://doi.org/10.1364/OL.390552
- **Status**: discovered
- **Topics**: steered illumination, time multiplexing, Fourier filtering
- **Abstract**: A 2D array of laser diodes switched on sequentially provides temporal directional illumination for a DMD hologram; a programmable polarization shutter at the Fourier plane, synchronized with the sources, filters the HDOs.
- **Notes**: Baseline VII (steered illumination with shifting Fourier filter); needs a very fast SLM.
---

### [jang2018hoe] Holographic Near-Eye Display with Expanded Eye-Box
- **Authors**: Changwon Jang, Kiseung Bang, Gang Li, Byoungho Lee
- **Venue**: ACM TOG 37(6), Article 195 (SIGGRAPH Asia 2018)
- **URL**: https://doi.org/10.1145/3272127.3275069
- **Status**: discovered
- **Topics**: eyebox expansion, pupil tracking, MEMS steering
- **Abstract**: Pupil tracking plus a MEMS mirror shifts the exit pupil over an expanded eye-box; a pupil-shifting HOE reduces form factor.
- **Notes**: Representative steered-illumination approach (pre-SLM expansion).
---

### [grace2023multi] Multisource Holography
- **Authors**: Grace Kuo, Florian Schiffers, Douglas Lanman, Oliver Cossairt, Nathan Matsuda
- **Venue**: ACM TOG 42(6), Article 203 (SIGGRAPH Asia 2023); arXiv 2309.10816
- **URL**: https://arxiv.org/abs/2309.10816
- **Status**: discovered
- **Topics**: speckle reduction, multi-source illumination, dual SLM
- **Abstract**: An array of sources plus two sequential phase SLMs lets each source be controlled nearly independently, suppressing speckle in a single frame without losing resolution.
- **Notes**: The paper explicitly distinguishes itself: closely spaced sources for speckle → étendue unchanged; here sources are far apart and the second modulator is an amplitude SLM in the Fourier plane. Schematic style borrowed from this paper.
---

### [choi2022flex] Time-Multiplexed Neural Holography
- **Authors**: Suyeon Choi, Manu Gopakumar, Yifan Peng, Jonghyun Kim, Matthew O'Toole, Gordon Wetzstein
- **Venue**: SIGGRAPH 2022 Conference Proceedings, Article 32; arXiv 2205.02367
- **URL**: https://arxiv.org/abs/2205.02367
- **Status**: discovered
- **Topics**: time-multiplexed CGH, quantized fast SLMs, light field supervision
- **Abstract**: CGH framework for fast, heavily quantized phase SLMs with camera-calibrated wave propagation models; supports 2D, 2.5D, focal-stack and 4D light field supervision; time multiplexing of T frames averaged by the eye.
- **Notes**: Source of the time-multiplexed image formation model and the quantization handling for the 4-bit TI phase SLM.
---

### [schiffers2023stochastic] Stochastic Light Field Holography
- **Authors**: Florian Schiffers, Praneeth Chakravarthula, Nathan Matsuda, Grace Kuo, Ethan Tseng, Douglas Lanman, Felix Heide, Oliver Cossairt
- **Venue**: IEEE ICCP 2023; arXiv 2307.06277
- **URL**: https://arxiv.org/abs/2307.06277
- **Status**: discovered
- **Topics**: light field CGH, pupil sampling
- **Abstract**: Supervises hologram optimization with light-field-refocused photographs from stochastically sampled pupil states, giving correct parallax and focus cues.
- **Notes**: Pupil-supervision idea behind the paper's stochastic, one-view-per-iteration H2LF (binary pupil mask in the Fourier plane).
---

### [gopakumar2021unfiltered] Unfiltered holography: optimizing high diffraction orders without optical filtering for compact holographic displays
- **Authors**: Manu Gopakumar, Jonghyun Kim, Suyeon Choi, Yifan Peng, Gordon Wetzstein
- **Venue**: Optics Letters 46(23), 5822–5825, 2021
- **URL**: https://doi.org/10.1364/OL.442851
- **Status**: discovered
- **Topics**: higher diffraction orders (HDOs), wave propagation models
- **Abstract**: Models and optimizes HDOs of the pixelated SLM instead of optically filtering them.
- **Notes**: HDO propagation model used in the paper.
---
