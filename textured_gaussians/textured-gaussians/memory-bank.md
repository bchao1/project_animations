# Paper Memory Bank
Last updated: 2026-09-24

### [chao2025textured] Textured Gaussians for Enhanced 3D Scene Appearance Modeling
- **Authors**: Brian Chao, Hung-Yu Tseng, Lorenzo Porzi, Chen Gao, Tuotuo Li, Qinbo Li, Ayush Saraf, Jia-Bin Huang, Johannes Kopf, Gordon Wetzstein, Changil Kim (Stanford, Meta, UMD College Park)
- **Venue**: CVPR 2025, pp. 8964-8974 (arXiv 2411.18625, v1 Nov 2024, v2 May 2025)
- **URL**: https://arxiv.org/abs/2411.18625 (project: https://textured-gaussians.github.io/, CVF open access: https://openaccess.thecvf.com/content/CVPR2025/html/Chao_Textured_Gaussians_for_Enhanced_3D_Scene_Appearance_Modeling_CVPR_2025_paper.html)
- **Citations**: n/a (not queried)
- **Status**: analyzed (full text + LaTeX source ingested; see summaries/chao2025textured.md)
- **Topics**: 3DGS appearance modeling, per-primitive texture / alpha mapping, novel-view synthesis
- **Abstract**: Augments every 3D Gaussian with a small T×T alpha, RGB or RGBA texture map, sampled at the ray / Gaussian-plane intersection, so a single primitive can carry spatially varying color and opacity (non-ellipsoidal shapes). Two-stage training on top of a GOF-style 3DGS; consistent gains over 3DGS* at equal Gaussian count, large gains with 1-10% of the Gaussians; alpha-only textures give the best quality per byte.
- **Notes**: Source paper for the animation in this folder.
---

### [kerbl20233dgs] 3D Gaussian Splatting for Real-Time Radiance Field Rendering
- **Authors**: Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, George Drettakis
- **Venue**: ACM TOG 42(4), 2023 (SIGGRAPH 2023); arXiv 2308.04079
- **URL**: https://arxiv.org/abs/2308.04079 (project: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- **Status**: discovered
- **Topics**: Gaussian splatting primitives, real-time NVS
- **Abstract**: Represents a scene as anisotropic 3D Gaussians (position, covariance = R S Sᵀ Rᵀ, opacity, SH color) optimized from SfM points with adaptive density control and rendered by a tile-based differentiable splatting rasterizer at real-time rates.
- **Notes**: The base representation Textured Gaussians extends (bib key kerbl3Dgaussians in the paper). Each 3DGS Gaussian has one color and an ellipsoidal footprint; Textured Gaussians lifts both limits. Original 3DGS numbers are compared in supplement Table A.1 (and its LPIPS values flagged as underestimated).
---

### [huang2024texturedgs] Textured-GS: Gaussian Splatting with Spatially Defined Color and Opacity
- **Authors**: Zhentao Huang, Minglun Gong (University of Guelph)
- **Venue**: arXiv 2407.09733, 2024; later in Proceedings of Graphics Interface 2025 (GI '25, ACM DL doi 10.1145/3769872.3769903)
- **URL**: https://arxiv.org/abs/2407.09733 (code: https://github.com/ZhentaoHuang/Textured-GS)
- **Status**: discovered
- **Topics**: appearance modeling in 3DGS, spatially varying color / opacity
- **Abstract**: Gives each Gaussian per-primitive SH coefficients for both color and opacity, evaluated with a direction derived from the ray-Gaussian intersection, so color and opacity vary smoothly across a single Gaussian.
- **Notes**: Cited in the paper as huang2024spatial. Closest prior "spatially varying Gaussian" work; the paper argues its SH-based variation is too smooth / low-frequency to represent complex textures, unlike explicit texel grids. Not to be confused with Texture-GS (Xu et al.).
---

### [xu2024texturegs] Texture-GS: Disentangling the Geometry and Texture for 3D Gaussian Splatting Editing
- **Authors**: Tian-Xing Xu, Wenbo Hu, Yu-Kun Lai, Ying Shan, Song-Hai Zhang
- **Venue**: ECCV 2024 (LNCS 15083, pp. 37-53); arXiv 2403.10050
- **URL**: https://arxiv.org/abs/2403.10050 (project: https://slothfulxtx.github.io/TexGS/)
- **Status**: discovered
- **Topics**: appearance modeling in 3DGS, global texture / UV mapping, editing
- **Abstract**: Gaussians represent geometry only; appearance lives in a single global 2D texture reached via a learned UV-mapping MLP (with a local Taylor expansion at ray-Gaussian intersections), enabling texture swapping / painting with real-time rendering.
- **Notes**: Baseline row in Table 1 (Blender 28.97 / 0.9380 / 0.0550 from GStex authors' runs; DTU 30.03 / – / 0.1440 as reported). The unit-sphere texture parameterization limits it to simple object-centric scenes; Textured Gaussians instead uses independent local per-Gaussian textures. DTU preprocessing (800×600, black bg) follows this paper.
---

### [yu2024gof] Gaussian Opacity Fields: Efficient Adaptive Surface Reconstruction in Unbounded Scenes
- **Authors**: Zehao Yu, Torsten Sattler, Andreas Geiger
- **Venue**: ACM TOG 43(6), 2024 (SIGGRAPH Asia 2024); arXiv 2404.10772
- **URL**: https://arxiv.org/abs/2404.10772 (project: https://niujinshuchong.github.io/gaussian-opacity-fields/)
- **Status**: discovered
- **Topics**: surface-aligned / ray-based Gaussians, densification
- **Abstract**: Evaluates Gaussians exactly along rays (ray-Gaussian intersection instead of 2D splat approximation), defines an opacity field whose level set gives surfaces extracted with Marching Tetrahedra, and improves densification with a sum-of-gradient-magnitudes criterion.
- **Notes**: The paper's 3DGS* baseline and the stage-1 model "closely follow" GOF: ray-based exact Gaussian evaluation α = G(x)·o and the revised densification score (also AbsGS). Ray-Gaussian intersection is reused for texture UV lookup.
---

### [huang20242dgs] 2D Gaussian Splatting for Geometrically Accurate Radiance Fields
- **Authors**: Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, Shenghua Gao
- **Venue**: SIGGRAPH 2024 Conference Papers (doi 10.1145/3641519.3657428); arXiv 2403.17888
- **URL**: https://arxiv.org/abs/2403.17888
- **Status**: discovered
- **Topics**: surface-aligned / 2D Gaussians, geometry reconstruction
- **Abstract**: Collapses 3D Gaussians into oriented planar 2D Gaussian disks (surfels) with explicit ray-splat intersection plus depth-distortion and normal-consistency losses, giving view-consistent geometry and accurate meshes.
- **Notes**: Cited as Huang2DGS2024. Textured Gaussians maps textures onto the plane of the two major axes of each 3D Gaussian, effectively a flattened 3D Gaussian; supplement (Fig. A.2) shows optimization naturally yields many effective-rank-2 (flat) Gaussians. GStex and SVGS build directly on 2DGS.
---

### [rong2024gstex] GStex: Per-Primitive Texturing of 2D Gaussian Splatting for Decoupled Appearance and Geometry Modeling
- **Authors**: Victor Rong, Jingxiang Chen, Sherwin Bahmani, Kiriakos N. Kutulakos, David B. Lindell
- **Venue**: WACV 2025; arXiv 2409.12954
- **URL**: https://arxiv.org/abs/2409.12954 (project: https://lessvrong.com/cs/gstex)
- **Status**: discovered
- **Topics**: textured / billboard primitives (concurrent)
- **Abstract**: Attaches a per-primitive RGB texture to each 2D Gaussian (2DGS) so appearance is decoupled from geometry, reaching high-fidelity appearance with fewer primitives and enabling texture editing.
- **Notes**: Concurrent work. Differences claimed by Textured Gaussians: alpha textures and an analysis of alpha / RGB / RGBA variants vs model size. GStex authors' runs of Texture-GS supply the Blender row of Table 1.
---

### [svitov2025bbsplat] BillBoard Splatting (BBSplat): Learnable Textured Primitives for Novel View Synthesis
- **Authors**: David Svitov, Pietro Morerio, Lourdes Agapito, Alessio Del Bue
- **Venue**: ICCV 2025, pp. 25029-25039; arXiv 2411.08508
- **URL**: https://arxiv.org/abs/2411.08508 (project: https://david-svitov.github.io/BBSplat_project_page/)
- **Status**: discovered
- **Topics**: textured / billboard primitives (concurrent)
- **Abstract**: Replaces Gaussians with optimizable textured planar billboards carrying RGB textures and alpha maps that control shape; drop-in for GS pipelines, supports 2DGS-style mesh extraction, and a sparsity regularizer allows up to 17× compression vs 3DGS.
- **Notes**: Concurrent work (cited as svitov2025billboardsplattingbbsplatlearnable, listed as arXiv 2025 in the paper's bib). Also uses alpha maps, so it is the closest concurrent design; Textured Gaussians keeps the Gaussian falloff and SH base color and studies the alpha-only variant.
---

### [xu2024svgs] SVGS (formerly SuperGaussians): Enhancing Gaussian Splatting Using Primitives with Spatially Varying Colors
- **Authors**: Rui Xu, Wenyue Chen, Jiepeng Wang, Yuan Liu, Peng Wang, Cheng Lin, Shiqing Xin, Xin Li, Wenping Wang, Taku Komura (v1 as "SuperGaussians" listed Lin Gao instead of Cheng Lin, with a different order)
- **Venue**: IEEE TVCG 2026 (vol. 32, no. 7, pp. 6972-6985 per a PubMed listing surfaced by search; not opened directly); arXiv 2411.18966
- **URL**: https://arxiv.org/abs/2411.18966 (project: https://ruixu.me/html/SuperGaussians/index.html)
- **Status**: discovered
- **Topics**: textured / billboard primitives (concurrent), spatially varying color
- **Abstract**: Gives each 2D Gaussian surfel spatially varying color and opacity via bilinear interpolation, movable kernels, or tiny neural networks; movable kernels perform best for NVS while keeping 2DGS-quality geometry.
- **Notes**: Concurrent work, cited as xu2024SuperGaussians (arXiv 2024 title "SuperGaussians"). Same goal as Textured Gaussians; different parameterization of intra-primitive variation.
---

### [barron2022mipnerf360] Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields
- **Authors**: Jonathan T. Barron, Ben Mildenhall, Dor Verbin, Pratul P. Srinivasan, Peter Hedman
- **Venue**: CVPR 2022; arXiv 2111.12077
- **URL**: https://arxiv.org/abs/2111.12077
- **Status**: discovered
- **Topics**: NVS benchmarks, NeRF baselines
- **Abstract**: Extends mip-NeRF to unbounded 360° scenes with a non-linear scene contraction, online proposal-network distillation and a distortion regularizer; introduces the Mip-NeRF 360 dataset.
- **Notes**: Both a baseline row in Table 1 (best PSNR on Mip-NeRF 360, DTU, Deep Blending) and the source of one of the five benchmarks (all 9 scenes used). Note Table 1's header cites mildenhall2020nerf for the Mip-NeRF 360 dataset column, a citation slip.
---
