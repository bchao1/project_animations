# Paper Memory Bank
Last updated: 2026-09-24

### [chao2026foveated] Foveated Diffusion: Efficient Spatially Adaptive Image and Video Generation
- **Authors**: Brian Chao*, Lior Yariv*, Howard Xiao, Gordon Wetzstein (Stanford)
- **Venue**: arXiv preprint 2603.23491, 2026
- **URL**: https://arxiv.org/abs/2603.23491 (project: https://bchao1.github.io/foveated-diffusion/, code: https://github.com/bchao1/foveated_diffusion)
- **Citations**: n/a (Semantic Scholar rate-limited)
- **Status**: analyzed (full text ingested for animation; see summaries/chao2026foveated.md)
- **Topics**: foveation, efficient visual generation, mixed-resolution tokens, DiT post-training
- **Abstract**: Allocates DiT tokens non-uniformly with a foveation mask (HR tokens in fovea, 2x2-merged LR tokens in periphery), post-trains FLUX.2 Klein 4B / Wan2.1 1.3B with LoRA on mixed-resolution-consistent targets. Up to 2x (image) / 4x (video) speedup at perceptual parity.
- **Notes**: Source paper for the animation in this folder.
---

### [wu2025crpa] One Attention, One Scale: Phase-Aligned Rotary Positional Embeddings for Mixed-Resolution Diffusion Transformer
- **Authors**: H. Wu, J. Xu, Q. Miao, D. Samaras, H. Le
- **Venue**: arXiv preprint 2511.19778, 2025
- **URL**: https://arxiv.org/abs/2511.19778
- **Status**: discovered
- **Topics**: mixed-resolution RoPE
- **Abstract**: Phase-aligned RoPE: align key RoPE phases to the query's resolution so each attention computation lives on one scale.
- **Notes**: Mixed-resolution RoPE used by Foveated Diffusion [ref 88].
---

### [guenter2012foveated] Foveated 3D Graphics
- **Authors**: B. Guenter, M. Finch, S. Drucker, D. Tan, J. Snyder
- **Venue**: ACM TOG 31(6), 2012
- **Status**: discovered
- **Topics**: foveated rendering
- **Abstract**: Classic gaze-contingent foveated rendering with eccentricity-dependent resolution layers.
- **Notes**: Graphics inspiration for Foveated Diffusion.
---
