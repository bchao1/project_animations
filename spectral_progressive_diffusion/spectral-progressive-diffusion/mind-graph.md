# Mind Graph
Last updated: 2026-09-24

### Efficient Visual Generation
- **Description**: Reducing the inference cost of diffusion / flow-matching DiTs for images and video
- **Subtopics**: [Progressive-Resolution Sampling], [Spectral Analysis of Diffusion]
- **Key papers**:
  - [xiao2026spectral] Spectral Progressive Diffusion (arXiv 2026) — resolution grows along one denoising trajectory, schedule from the power spectrum

### Spectral Analysis of Diffusion
- **Description**: Frequency-domain view of the forward/reverse process; power-law image spectra vs flat noise
- **Parent**: [Efficient Visual Generation]
- **Related topics**: [Progressive-Resolution Sampling]
- **Key papers**:
  - [dieleman2024spectral] Diffusion is spectral autoregression (blog 2024) — RAPSD argument, low→high frequency generation
  - [rissanen2023generative] Inverse Heat Dissipation (ICLR 2023) — explicit coarse-to-fine generative process
  - [falck2025fourier] A Fourier Space Perspective on Diffusion Models (arXiv 2025) — per-frequency SNR, equal-rate forward process
  - [xiao2026spectral] — Prop. 1/2: per-frequency δ-optimal activation time → per-resolution transition time

### Progressive-Resolution Sampling
- **Description**: Run early (noisy) steps at low resolution and upsample later
- **Parent**: [Efficient Visual Generation]
- **Subtopics**:
  - *Trained from scratch / custom architecture*: [jin2025pyramidal] Pyramidal Flow Matching (ICLR 2025)
  - *Training-free on pretrained DiTs*: [tian2025bottleneck] Bottleneck Sampling (2025); [jeong2026ralu] RALU (CVPR 2026); [zhang2025frecas] FreCaS (ICLR 2025, needs fully denoised low-res image)
  - *Training-free + fine-tuning, principled schedule*: [xiao2026spectral]
