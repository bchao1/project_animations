# Paper Memory Bank
Last updated: 2026-09-27

### [chao2023high] High-brightness holographic projection
- **Authors**: Brian Chao, Manu Gopakumar, Suyeon Choi, Gordon Wetzstein (Electrical Engineering Department, Stanford University)
- **Venue**: Optics Letters 48(15), 4041–4044 (1 Aug 2023; online 25 Jul 2023). DOI 10.1364/OL.489617, PMID 37527113. Received 20 Mar 2023, revised 20 Jun, accepted 28 Jun 2023.
- **URL**: https://opg.optica.org/ol/fulltext.cfm?uri=ol-48-15-4041&id=535470 (Optica page is behind a Radware captcha; full text obtained from the Wayback Machine snapshot of 2023-07-26). Supplement 1 (CC-BY 4.0): https://doi.org/10.6084/m9.figshare.23596260. Visualization 1: https://doi.org/10.6084/m9.figshare.22290445. Website teaser clip: https://bchao1.github.io/images/OL2023.mp4
- **Citations**: 12 (Semantic Scholar, 2026-09)
- **Status**: analyzed (main text + supplement + both videos; see summaries/chao2023high.md). Main-paper figure images (Figs. 1–5) could NOT be obtained (captcha; not archived).
- **Topics**: holographic projection, light reallocation, light-efficiency loss, phase-only CGH (SGD), camera-calibrated learned wave propagation, camera-in-the-loop
- **Abstract**: Fresnel-holography projector (phase-only SLM, long ~20 cm propagation, projection lens). A light-efficiency loss forces |f(u_SLM)| to equal the target scaled so that all source power lands in the image region; with a camera-calibrated learned propagation model this raises captured brightness by around 30% at similar image quality.
- **Notes**: Source paper for this animation project.
---

### [peng2020neural] Neural Holography with Camera-in-the-loop Training
- **Authors**: Yifan Peng, Suyeon Choi, Nitish Padmanaban, Gordon Wetzstein
- **Venue**: ACM TOG 39(6) (SIGGRAPH Asia 2020)
- **URL**: https://doi.org/10.1145/3414685.3417802
- **Status**: discovered
- **Topics**: camera-in-the-loop, CGH, learned propagation
- **Abstract**: Introduces CITL hologram optimization and a learned, camera-calibrated wave propagation model (HoloNet).
- **Notes**: Supplement ref [1]; CITL (Alg. S2), homography calibration.
---

### [choi2021neural] Neural 3D Holography
- **Authors**: Suyeon Choi, Manu Gopakumar, Yifan Peng, Jonghyun Kim, Gordon Wetzstein
- **Venue**: ACM TOG 40(6), 240 (SIGGRAPH Asia 2021)
- **URL**: https://doi.org/10.1145/3478513.3480542
- **Status**: discovered
- **Topics**: learned wave propagation model, CGH
- **Abstract**: Camera-calibrated CNN-augmented ASM model for accurate 3D holography.
- **Notes**: Main-text ref [10].
---

### [choi2022time] Time-multiplexed Neural Holography
- **Authors**: Suyeon Choi, Manu Gopakumar, Yifan Peng, Jonghyun Kim, Matthew O'Toole, Gordon Wetzstein
- **Venue**: SIGGRAPH 2022 Conference Proceedings; arXiv 2205.02367
- **URL**: https://arxiv.org/abs/2205.02367
- **Status**: discovered
- **Topics**: time multiplexing, quantized SLMs (Gumbel-Softmax), learned propagation model
- **Abstract**: Optimizes several heavily quantized phase patterns shown in quick succession.
- **Notes**: The projector's model architecture (CNN_SLM, ASM, CNN_target) and 4-bit Gumbel-Softmax optimization follow this paper (main ref [11], supplement ref [5]).
---

### [damberg2016high] High Brightness HDR Projection Using Dynamic Freeform Lensing
- **Authors**: Gerwin Damberg, James Gregson, Wolfgang Heidrich
- **Venue**: ACM TOG 35(3), 24 (2016)
- **URL**: https://doi.org/10.1145/2857051
- **Status**: discovered
- **Topics**: light reallocation, dual-modulation projection
- **Abstract**: Phase SLM as freeform lens steers light to bright regions, followed by an amplitude modulator.
- **Notes**: Example of the "dual-display" category (ii) the paper argues against (cost, size, sync).
---

### [hoskinson2010light] Light Reallocation for High Contrast Projection Using an Analog Micromirror Array
- **Authors**: Reynald Hoskinson, Boris Stoeber, Wolfgang Heidrich, Sidney Fels
- **Venue**: ACM TOG 29(6), 165 (2010)
- **Status**: discovered
- **Topics**: light reallocation, projection
- **Notes**: Dual-display category, main ref [4].
---

### [chan2022holocurtains] Holocurtains: Programming Light Curtains via Binary Holography
- **Authors**: Dorian Chan, Srinivasa G. Narasimhan, Matthew O'Toole
- **Venue**: CVPR 2022
- **Status**: discovered
- **Topics**: holographic light reallocation for sensing
- **Notes**: Main ref [5]: experimentally demonstrated holographic light reallocation for 3D sensing.
---

### [shi2021towards] Towards real-time photorealistic 3D holography with deep neural networks
- **Authors**: Liang Shi, Beichen Li, Changil Kim, Petr Kellnhofer, Wojciech Matusik
- **Venue**: Nature 591, 234 (2021)
- **Status**: discovered
- **Topics**: AI-driven CGH, sub-hologram / near-SLM setting
- **Notes**: Main ref [13]; example of a near-SLM setting with little light reallocation.
---

### [gerchberg1972practical] Gerchberg–Saxton phase retrieval
- **Authors**: R. W. Gerchberg, W. O. Saxton
- **Venue**: Optik 35, 237 (1972)
- **Status**: discovered
- **Topics**: phase retrieval, classic CGH
- **Notes**: Main ref [14]; classic CGH algorithms not optimized to use the full laser power.
---

### [kavakli2023multicolor] Multi-color Holograms Improve Brightness in Holographic Displays
- **Authors**: Koray Kavaklı, Liang Shi, Hakan Urey, Wojciech Matusik, Kaan Akşit
- **Venue**: SIGGRAPH Asia 2023
- **URL**: https://doi.org/10.1145/3610548.3618135
- **Status**: discovered
- **Topics**: holographic display brightness
- **Notes**: Concurrent/follow-up brightness work found in the search; not cited by chao2023high.
---
