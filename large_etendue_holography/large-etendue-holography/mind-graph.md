# Mind Graph
Last updated: 2026-09-24

### Holographic Near-Eye Displays
- **Description**: Near-eye VR/AR displays that synthesize a coherent wavefront with a phase SLM (true 3D, focus cues, parallax)
- **Related topics**: [Étendue Expansion], [Light Field / Pupil-aware CGH]
- **Key papers**:
  - [chao2024etendue] Large Étendue 3D Holography with Dynamic Fourier Modulation (SIGGRAPH Asia 2024) — multi-source + Fourier amplitude SLM
  - [choi2022flex] Time-Multiplexed Neural Holography (SIGGRAPH 2022) — fast quantized SLMs, time multiplexing

### Étendue Expansion
- **Description**: Beating G = λ²·N_x·N_y (FoV × eyebox fixed by SLM pixel count)
- **Parent topic**: [Holographic Near-Eye Displays]
- **Subtopics**:
  - **Post-SLM (masks, pupil replication, HDOs)**: [kuo2020mask] scattering mask (TOG 2020)
  - **Pre-SLM (steered / multi-source illumination)**: [jang2018hoe] MEMS steering (TOG 2018); [lee2020wide] sequential LD array + shutter (OL 2020); [jo2022binary] simultaneous sources + random binary Fourier mask (OL 2022); [chao2024etendue] simultaneous sources + dynamic content-adaptive Fourier amplitude SLM
- **Contrast**: [grace2023multi] Multisource Holography (TOG 2023) — multi-source for speckle, not étendue

### Light Field / Pupil-aware CGH
- **Description**: Optimizing holograms against a 4D light field via pupil masks in the Fourier (eyebox) plane
- **Related topics**: [Étendue Expansion]
- **Key papers**:
  - [schiffers2023stochastic] Stochastic Light Field Holography (ICCP 2023)
  - [gopakumar2021unfiltered] Unfiltered Holography (OL 2021) — HDO modeling
  - [chao2024etendue] stochastic single-view H2LF with 9×9 pupils
