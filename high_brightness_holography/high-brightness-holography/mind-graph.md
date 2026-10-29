# Mind Graph
Last updated: 2026-09-27

### Holographic Projection and Light Reallocation
- **Description**: Phase-only holograms steer light instead of blocking it, so sparse or dark content can be much brighter than on attenuating displays
- **Related topics**: [Light-efficiency Objectives for CGH], [Dual-modulation / Freeform-lensing Projectors]
- **Key papers**:
  - [chao2023high] High-brightness holographic projection (Opt. Lett. 2023) — light-efficiency loss + learned propagation model, ~30% brighter captured images
  - [chan2022holocurtains] Holocurtains (CVPR 2022) — holographic light reallocation for 3D sensing
- **Other relevant papers**:
  - [kavakli2023multicolor] Multi-color holograms improve brightness (SIGGRAPH Asia 2023)

### Light-efficiency Objectives for CGH
- **Description**: Losses that force the reconstructed field to put all source power inside the region of interest
- **Related topics**: [Holographic Projection and Light Reallocation], [Camera-calibrated Wave Propagation]
- **Key papers**:
  - [chao2023high] L_reg = L2(|f(u_SLM)|, sqrt(Σ|u_src|² / Σa²) · a_target), total L = L_recon + γ L_reg

### Camera-calibrated Wave Propagation / Camera-in-the-loop
- **Description**: Learn the real display's propagation from captured SLM-pattern/image pairs, or use camera feedback during optimization
- **Related topics**: [Light-efficiency Objectives for CGH]
- **Key papers**:
  - [peng2020neural] Neural holography with CITL training (TOG 2020)
  - [choi2021neural] Neural 3D holography (TOG 2021)
  - [choi2022time] Time-multiplexed neural holography (SIGGRAPH 2022) — model architecture and 4-bit optimization used by chao2023high
- **Other relevant papers**:
  - [shi2021towards] Tensor Holography (Nature 2021)
  - [gerchberg1972practical] Gerchberg–Saxton (1972)

### Dual-modulation / Freeform-lensing Projectors
- **Description**: A phase SLM redistributes light onto an amplitude SLM that forms the image
- **Related topics**: [Holographic Projection and Light Reallocation]
- **Key papers**:
  - [damberg2016high] High brightness HDR projection with freeform lensing (TOG 2016)
  - [hoskinson2010light] Light reallocation with an analog micromirror array (TOG 2010)
