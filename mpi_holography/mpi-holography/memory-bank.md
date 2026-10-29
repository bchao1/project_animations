# Paper Memory Bank
Last updated: 2026-09-27

### [chao2026fast] Fast Wave-optics Rendering of Multiplane Images for 3D Holographic Displays
- **Authors**: Brian Chao (Stanford University and Meta; corresponding author), Dario Seyb, Nathan Matsuda, Oliver Cossairt, Yang Zhou, Douglas Lanman (Meta), Gordon Wetzstein (Stanford University), Grace Kuo, Changwon Jang (Meta)
- **Venue**: arXiv preprint 2607.19731 (cs.GR), v1 22 Jul 2026. LaTeX uses acmtog in `nonacm` mode; author homepage lists it as "arXiv, 2026". No project page.
- **URL**: https://arxiv.org/abs/2607.19731 ; homepage teaser videos https://bchao1.github.io/images/mpi_light_field.mp4, https://bchao1.github.io/images/mpi_focal_stack.mp4
- **Citations**: n/a (new)
- **Status**: analyzed (PDF, text, LaTeX source, figure rasters, vector plot data; see summaries/chao2026fast.md)
- **Topics**: computer-generated holography, multiplane images, layer-based CGH, random-phase holograms, wave-optics alpha compositing, time multiplexing
- **Abstract**: Converts multiplane images (a handful of RGBA layers) into random-phase, time-multiplexed holograms by propagating back to front with the angular spectrum method and compositing with RPWS's sqrt-transmittance / sqrt-colour random-phase alpha blending. Quality close to RPWS (Gaussian splats), far above RGBD layer CGH at depth edges, up to 250,000x faster than RPWS.
- **Notes**: Source paper for this animation project. Follow-up of RPWS (chao2025rpws) with the same compositing rule applied to MPI layers instead of millions of primitives.
---

### [chao2025rpws] Random-phase Wave Splatting of Translucent Primitives for Computer-generated Holography
- **Authors**: Brian Chao*, Jacqueline Yang*, Suyeon Choi, Manu Gopakumar, Ryota Koiso, Gordon Wetzstein
- **Venue**: arXiv 2508.17480, 2025
- **URL**: https://arxiv.org/abs/2508.17480
- **Status**: analyzed (full record in ../../random_phase_wave_splatting/random-phase-wave-splatting/)
- **Topics**: primitive-based CGH, random-phase alpha wave blending, time multiplexing
- **Abstract**: Random-phase holograms from translucent 2D primitives; back-to-front wave compositing with sqrt masks; exact alpha blending in expectation over frames.
- **Notes**: The quality baseline (and the compositing rule) of the MPI paper; "extremely slow runtime" with millions of Gaussians.
---

### [zhou2018mpi] Stereo Magnification: Learning View Synthesis using Multiplane Images
- **Authors**: Tinghui Zhou, Richard Tucker, John Flynn, Graham Fyffe, Noah Snavely
- **Venue**: ACM TOG 37(4), Article 65 (SIGGRAPH 2018)
- **URL**: https://doi.org/10.1145/3197517.3201323
- **Status**: discovered
- **Topics**: multiplane images, view synthesis
- **Abstract**: Introduces the MPI (fronto-parallel RGBA layers) predicted by a network from a stereo pair; novel views via homography warps + over-compositing.
- **Notes**: MPI representation used by the paper; its predictor produced the MPIs of the paper's Fig. 6.
---

### [choi2025gaussian] Gaussian Wave Splatting for Computer-Generated Holography
- **Authors**: Suyeon Choi*, Brian Chao*, Jacqueline Yang, Manu Gopakumar, Gordon Wetzstein
- **Venue**: ACM TOG 44(4) (SIGGRAPH 2025); arXiv 2505.06582
- **URL**: https://arxiv.org/abs/2505.06582
- **Status**: analyzed (record in ../../gaussian_wave_splatting/)
- **Topics**: primitive-based CGH, Gaussian splatting
- **Abstract**: Closed-form Gaussian wavefronts, alpha wave blending, smooth-phase holograms.
- **Notes**: Primitive-based CGH that the MPI paper cites as millions-of-primitives cost.
---

### [peng2020neural] Neural Holography with Camera-in-the-loop Training
- **Authors**: Yifan Peng, Suyeon Choi, Nitish Padmanaban, Gordon Wetzstein
- **Venue**: ACM TOG 39(6), 2020
- **URL**: https://doi.org/10.1145/3414685.3417802
- **Status**: discovered
- **Topics**: SGD phase retrieval, layer-based (RGBD) CGH
- **Abstract**: SGD-based phase-only hologram optimization and camera calibration.
- **Notes**: RGBD layer-based CGH baseline family; SGD used to make phase-only patterns for the prototype.
---

### [choi2022time] Time-multiplexed Neural Holography
- **Authors**: Suyeon Choi, Manu Gopakumar, Yifan Peng, Jonghyun Kim, Matthew O'Toole, Gordon Wetzstein
- **Venue**: ACM SIGGRAPH 2022 Conference Proceedings
- **URL**: https://arxiv.org/abs/2205.02367
- **Status**: discovered
- **Topics**: time multiplexing, light-field CGH (STFT)
- **Abstract**: Several fast-SLM frames averaged in intensity; focal-stack and 4D light-field supervision.
- **Notes**: STFT operator used for light-field reconstruction in Eq. 3.
---

### [schiffers2023stochastic] Stochastic Light Field Holography
- **Authors**: Florian Schiffers, Praneeth Chakravarthula, Nathan Matsuda, Grace Kuo, Ethan Tseng, Douglas Lanman, Felix Heide, Oliver Cossairt
- **Venue**: IEEE ICCP 2023
- **URL**: https://arxiv.org/abs/2307.06277
- **Status**: discovered
- **Topics**: light-field holography, pupil sampling
- **Abstract**: Pupil-sampled light-field supervision for holograms.
- **Notes**: Pupil-masking operator in Eq. 3.
---

### [porter1984over] Compositing Digital Images
- **Authors**: Thomas Porter, Tom Duff
- **Venue**: SIGGRAPH 1984 (Computer Graphics 18(3))
- **URL**: https://doi.org/10.1145/964965.808606
- **Status**: discovered
- **Topics**: alpha compositing
- **Abstract**: The "over" operator for RGBA images.
- **Notes**: MPIs are rendered back to front with "over"; the paper's Eq. 2 is its wave-optics analogue.
---

### [zhan2025complexradiance] Complex-Valued Holographic Radiance Fields
- **Authors**: Yicheng Zhan, Dong-Ha Shin, Seung-Hwan Baek, Kaan Akşit
- **Venue**: arXiv 2506.08350, 2025
- **URL**: https://arxiv.org/abs/2506.08350
- **Status**: discovered
- **Topics**: holographic radiance fields, layered complex compositing
- **Abstract**: Scene-specific complex-valued Gaussians composited along discrete planes.
- **Notes**: Concurrent work; learned complex scene model vs the MPI paper's off-the-shelf layers.
---

### [chu2025artifact] Artifact-Resilient Real-Time Holography
- **Authors**: Victor Chu, Oscar Pueyo-Ciutad, Ethan Tseng, Florian Schiffers, Grace Kuo, Nathan Matsuda, Albert Redo-Sanchez, Douglas Lanman, Oliver Cossairt, Felix Heide
- **Venue**: ACM TOG 44(6), Article 219, Dec 2025
- **URL**: https://doi.org/10.1145/3763361
- **Status**: discovered
- **Topics**: real-time random-phase hologram networks
- **Abstract**: Real-time random-phase hologram synthesis network.
- **Notes**: Named as a future direction to reach real time.
---
