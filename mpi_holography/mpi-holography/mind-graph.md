# Mind Graph
Last updated: 2026-09-27

### Layer-based Computer-Generated Holography
- **Description**: CGH that propagates a small number of depth layers (RGBD slices, MPIs) to the SLM
- **Related topics**: [Multiplane Images], [Random-phase Alpha Wave Compositing], [Primitive-based CGH]
- **Key papers**:
  - [chao2026fast] MPI holography (arXiv 2026) — MPIs + random-phase back-to-front wave compositing, up to 250,000x faster than RPWS
  - [peng2020neural] Neural holography (TOG 2020) — RGBD / layer-based CGH baseline family
- **Other relevant papers**:
  - [zhan2025complexradiance] Complex-valued holographic radiance fields — concurrent, learned plane-wise complex compositing

### Multiplane Images
- **Description**: Scene as a handful of fronto-parallel RGBA layers; novel views by homography + over-compositing
- **Related topics**: [Layer-based Computer-Generated Holography]
- **Key papers**:
  - [zhou2018mpi] Stereo magnification (TOG 2018) — introduces MPIs
  - [porter1984over] Porter–Duff compositing — the "over" operator
- **Other relevant papers**:
  - [chao2026fast] MPI holography — MPIs as the intermediate representation for CGH

### Random-phase Alpha Wave Compositing
- **Description**: Occlusion with random-phase wavefronts: sqrt(1 − α) masks, sqrt colour weights, back to front, exact in expectation over frames
- **Related topics**: [Layer-based Computer-Generated Holography], [Time Multiplexing]
- **Key papers**:
  - [chao2025rpws] RPWS (arXiv 2025) — the compositing rule
  - [chao2026fast] MPI holography — applies it to MPI layers

### Primitive-based CGH
- **Description**: Sum of millions of primitive wavefronts (points, polygons, Gaussians)
- **Related topics**: [Random-phase Alpha Wave Compositing]
- **Key papers**:
  - [choi2025gaussian] Gaussian wave splatting (TOG 2025)
  - [chao2025rpws] RPWS — quality baseline, slow

### Time Multiplexing
- **Description**: Several holograms averaged in intensity (speckle, light fields, exact blending)
- **Related topics**: [Random-phase Alpha Wave Compositing]
- **Key papers**:
  - [choi2022time] Time-multiplexed neural holography (SIGGRAPH 2022) — STFT light-field operator
  - [schiffers2023stochastic] Stochastic light field holography (ICCP 2023) — pupil-sampled views
- **Other relevant papers**:
  - [chu2025artifact] Artifact-resilient real-time holography — real-time random-phase networks (future direction)
