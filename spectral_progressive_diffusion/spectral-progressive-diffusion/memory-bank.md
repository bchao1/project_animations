# Paper Memory Bank
Last updated: 2026-09-24

### [xiao2026spectral] Spectral Progressive Diffusion for Efficient Image and Video Generation
- **Authors**: Howard Xiao, Brian Chao, Lior Yariv, Gordon Wetzstein (Stanford University)
- **Venue**: arXiv preprint 2605.18736 (v1 2026-05-18, v3 2026-08-17), cs.CV; NeurIPS 2026 preprint style
- **URL**: https://arxiv.org/abs/2605.18736 (project: https://howardxiao.ca/speed/)
- **Citations**: n/a
- **Status**: analyzed (full text + LaTeX source ingested; see summaries/xiao2026spectral.md)
- **Topics**: spectral autoregression, progressive-resolution sampling, efficient DiT inference, flow matching
- **Abstract**: Diffusion models implicitly generate content autoregressively in frequency: low frequencies
  resolve early, high frequencies late, so high-resolution compute on noise-dominated frequencies is redundant.
  The paper progressively grows resolution along the denoising trajectory of pretrained models, using a
  spectral noise expansion (embed the low-res spectrum, pad new high-frequency slots with t·ε, invert,
  then rescale + re-index the timestep) and a δ-optimal resolution schedule derived from the power spectrum.
  Supports training-free acceleration and a LoRA fine-tuning recipe.
- **Notes**: Up to 7.09× wall-clock speedup on FLUX.1-dev (S=3, 7× tier), 2.03× (S=2) / 2.54× (S=3) on
  WAN 2.1 720p. Source paper for the animation in this project folder.
---

### [dieleman2024spectral] Diffusion is spectral autoregression
- **Authors**: Sander Dieleman
- **Venue**: Blog post, 2024-09-02
- **URL**: https://sander.ai/2024/09/02/spectral-autoregression.html
- **Status**: discovered
- **Topics**: spectral autoregression, power spectra of natural images
- **Abstract**: Shows with radially averaged power spectral density (RAPSD) plots that natural image spectra follow
  a power law while Gaussian noise is flat, so adding noise drowns high frequencies first; diffusion
  sampling therefore behaves like approximate autoregression from low to high frequency.
- **Notes**: The core motivation of xiao2026spectral (cited as [9]); the “PSD vs flat noise spectrum” picture
  used in Part 2–3 of the animation.
---

### [rissanen2023generative] Generative Modelling With Inverse Heat Dissipation
- **Authors**: Severi Rissanen, Markus Heinonen, Arno Solin
- **Venue**: ICLR 2023 (arXiv 2206.13397)
- **URL**: https://arxiv.org/abs/2206.13397
- **Status**: discovered
- **Topics**: coarse-to-fine generation, heat equation, spectral view of diffusion
- **Abstract**: Diffusion-like generative model that stochastically reverses the heat equation, which erases
  fine-scale information; the model builds images coarse-to-fine and shows emergent colour/shape disentanglement.
- **Notes**: Cited with Dieleman as the observation that diffusion generates low frequencies first.
---

### [falck2025fourier] A Fourier Space Perspective on Diffusion Models
- **Authors**: Fabian Falck, Teodora Pandeva, Kiarash Zahirnia, Rachel Lawrence, Richard Turner, Edward Meeds, Javier Zazo, Sushrut Karmalkar
- **Venue**: arXiv preprint 2505.11278, 2025
- **URL**: https://arxiv.org/abs/2505.11278
- **Status**: discovered
- **Topics**: Fourier analysis of the forward process, per-frequency SNR
- **Abstract**: Under the DDPM forward process, high-frequency components lose SNR faster and earlier than low ones
  because data variance decays in Fourier space; proposes a forward process that corrupts all frequencies at the same rate.
- **Notes**: Cited for the power-law exponent range β ∈ [2, 3] and as a spectral noise-schedule method that
  improves quality but not efficiency.
---

### [jeong2026ralu] Training-free Mixed-Resolution Latent Upsampling for Spatially Accelerated Diffusion Transformers (RALU)
- **Authors**: Wongi Jeong, Kyungryeol Lee, Hoigi Seo, Se Young Chun
- **Venue**: CVPR 2026 (Highlight); arXiv 2507.08422 (v1 title: “Upsample What Matters: Region-Adaptive Latent Sampling …”)
- **URL**: https://arxiv.org/abs/2507.08422
- **Status**: discovered
- **Topics**: training-free progressive resolution, noise-timestep matching
- **Abstract**: Region-Adaptive Latent Upsampling: early upsampling only in artifact-prone (edge) regions plus
  Noise-and-Timestep matching before/after upsampling; spatial acceleration of FLUX-class DiTs.
- **Notes**: Main baseline in Table 1 of xiao2026spectral; evaluation protocol and baseline rows are copied from RALU's Table 1.
---

### [tian2025bottleneck] Training-free Diffusion Acceleration with Bottleneck Sampling
- **Authors**: Ye Tian, Xin Xia, Yuxi Ren, Shanchuan Lin, Xing Wang, Xuefeng Xiao, Yunhai Tong, Ling Yang, Bin Cui
- **Venue**: arXiv preprint 2503.18940, 2025
- **URL**: https://arxiv.org/abs/2503.18940
- **Status**: discovered
- **Topics**: training-free progressive resolution (high–low–high)
- **Abstract**: High-resolution denoising at the start and end, low resolution in the middle; tuned transition
  points and timestep shifts; up to 3× (image) / 2.5× (video).
- **Notes**: Baseline in Table 1 of xiao2026spectral; example of an S=2-style heuristic schedule.
---

### [jin2025pyramidal] Pyramidal Flow Matching for Efficient Video Generative Modeling
- **Authors**: Yang Jin, Zhicheng Sun, Ningyuan Li, Kun Xu, Hao Jiang, Nan Zhuang, Quzhe Huang, Yang Song, Yadong Mu, Zhouchen Lin
- **Venue**: ICLR 2025 (arXiv 2410.05954)
- **URL**: https://arxiv.org/abs/2410.05954
- **Status**: discovered
- **Topics**: pyramid-stage flow matching, trained-from-scratch progressive resolution
- **Abstract**: Reinterprets the denoising trajectory as a series of pyramid stages where only the last runs at full
  resolution; stages are interlinked; single DiT trained end-to-end.
- **Notes**: Representative of progressive-resolution methods that need training from scratch / custom architecture.
---

### [zhang2025frecas] FreCaS: Efficient Higher-Resolution Image Generation via Frequency-aware Cascaded Sampling
- **Authors**: Zhengqiang Zhang, Ruihuang Li, Lei Zhang
- **Venue**: ICLR 2025 (arXiv 2410.18410)
- **URL**: https://arxiv.org/abs/2410.18410 (OpenReview: https://openreview.net/forum?id=TsBDfe8Ra5)
- **Status**: discovered
- **Topics**: training-free frequency-aware cascaded sampling
- **Abstract**: Training-free higher-resolution generation: cascaded stages of increasing resolution that progressively expand
  frequency bands, with frequency-aware classifier-free guidance (FA-CFG) steering new detail into the expanded band.
- **Notes**: Cited as a training-free spectral method that needs a fully denoised low-res image first, limiting efficiency.
---
