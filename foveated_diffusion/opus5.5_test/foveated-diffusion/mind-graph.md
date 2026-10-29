# Mind Graph
Last updated: 2026-09-24

### Foveated Generation
- **Description**: Generative models that allocate compute according to gaze / visual eccentricity
- **Related topics**: [Foveated Rendering], [Mixed-Resolution Positional Encoding]
- **Key papers**:
  - [chao2026foveated] Foveated Diffusion (arXiv 2026) — first foveated DiT framework; mixed-resolution-consistent training targets

### Foveated Rendering
- **Description**: Graphics pipelines that shade/rasterize at lower rate in the periphery
- **Related topics**: [Foveated Generation]
- **Key papers**:
  - [guenter2012foveated] Foveated 3D Graphics (TOG 2012) — canonical foveated rendering

### Mixed-Resolution Positional Encoding
- **Description**: RoPE variants for tokens of different spatial sizes in one sequence
- **Related topics**: [Foveated Generation]
- **Key papers**:
  - [wu2025crpa] Phase-Aligned RoPE (arXiv 2025) — used by Foveated Diffusion
