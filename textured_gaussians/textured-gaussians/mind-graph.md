# Mind Graph
Last updated: 2026-09-24

### Gaussian Splatting Primitives
- **Description**: Explicit radiance-field representations built from anisotropic Gaussians rendered by differentiable splatting / alpha compositing
- **Related topics**: [Appearance Modeling in 3DGS], [Surface-Aligned / 2D Gaussians], [NVS Benchmarks]
- **Key papers**:
  - [kerbl20233dgs] 3D Gaussian Splatting (TOG / SIGGRAPH 2023) — base representation: one SH color + one opacity + ellipsoidal footprint per Gaussian
  - [chao2025textured] Textured Gaussians (CVPR 2025) — generalizes the per-Gaussian appearance model; 3DGS is the special case c_tex = 0, α_tex = 1

### Appearance Modeling in 3DGS
- **Description**: Making color / opacity vary within a single primitive instead of being constant up to the Gaussian falloff
- **Related topics**: [Gaussian Splatting Primitives], [Textured / Billboard Primitives (Concurrent)]
- **Key papers**:
  - [chao2025textured] Textured Gaussians (CVPR 2025) — local T×T alpha / RGB / RGBA texture per Gaussian, sampled at the ray-plane intersection; alpha-only is the best quality-per-byte variant
  - [huang2024texturedgs] Textured-GS (arXiv 2024 / GI 2025) — per-Gaussian SH for color and opacity evaluated at intersection-dependent directions; smooth, low-frequency variation only
  - [xu2024texturegs] Texture-GS (ECCV 2024) — Gaussians as geometry + one global texture via learned UV MLP; limited to simple object-centric scenes; Table 1 baseline

### Textured / Billboard Primitives (Concurrent)
- **Description**: Contemporaneous (late 2024) works attaching texture or alpha maps to planar / 2D primitives
- **Related topics**: [Appearance Modeling in 3DGS], [Surface-Aligned / 2D Gaussians]
- **Key papers**:
  - [rong2024gstex] GStex (WACV 2025) — per-primitive RGB textures on 2DGS surfels
  - [svitov2025bbsplat] BBSplat (ICCV 2025) — textured planar billboards with RGB + alpha maps, sparsity-driven compression
  - [xu2024svgs] SVGS / SuperGaussians (TVCG 2026) — spatially varying color / opacity functions (bilinear, movable kernels, tiny MLPs) on 2D surfels
  - [chao2025textured] Textured Gaussians (CVPR 2025) — distinguishes itself by alpha textures and the alpha / RGB / RGBA vs model-size analysis
  - (also cited, not recorded here: HDGS, Song et al. 2024, textured 2D Gaussians)

### Surface-Aligned / 2D Gaussians
- **Description**: Flattened or ray-evaluated Gaussians with exact ray-primitive intersection, originally aimed at geometry / surface extraction
- **Related topics**: [Gaussian Splatting Primitives], [Textured / Billboard Primitives (Concurrent)]
- **Key papers**:
  - [huang20242dgs] 2D Gaussian Splatting (SIGGRAPH 2024) — planar Gaussian disks with ray-splat intersection; base of GStex and SVGS
  - [yu2024gof] Gaussian Opacity Fields (TOG / SIGGRAPH Asia 2024) — ray-based exact Gaussian evaluation and sum-of-magnitudes densification; Textured Gaussians' 3DGS* baseline follows it
  - [chao2025textured] Textured Gaussians (CVPR 2025) — textures live on the plane of each Gaussian's two major axes; optimization yields mostly flat (effective rank 2) Gaussians

### NVS Benchmarks
- **Description**: Standard novel-view-synthesis datasets and NeRF baselines used for evaluation
- **Related topics**: [Gaussian Splatting Primitives]
- **Key papers**:
  - [barron2022mipnerf360] Mip-NeRF 360 (CVPR 2022) — dataset (9 scenes) and Table 1 baseline
  - [chao2025textured] Textured Gaussians (CVPR 2025) — evaluates on Blender (8), Mip-NeRF 360 (9), DTU (5), Tanks and Temples (2), Deep Blending (2) + 2 custom art-gallery captures
