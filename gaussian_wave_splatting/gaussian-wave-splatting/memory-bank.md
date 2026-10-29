# Paper Memory Bank
Last updated: 2026-09-24

### [choi2025gaussian] Gaussian Wave Splatting for Computer-Generated Holography
- **Authors**: Suyeon Choi*, Brian Chao*, Jacqueline Yang, Manu Gopakumar, Gordon Wetzstein (Stanford University; * equal contribution)
- **Venue**: ACM Transactions on Graphics 44(4), Article 111 (SIGGRAPH 2025), DOI 10.1145/3731163; arXiv 2505.06582 (cs.GR, physics.optics)
- **URL**: https://arxiv.org/abs/2505.06582 (project: https://bchao1.github.io/gaussian-wave-splatting/, code: https://github.com/computational-imaging/hsplat)
- **Citations**: n/a
- **Status**: analyzed (full text + LaTeX source ingested; see summaries/choi2025gaussian.md)
- **Topics**: computer-generated holography, Gaussian splatting, primitive-based CGH, occlusion in wave optics
- **Abstract**: Turns optimized 2D Gaussian splatting scenes into holograms. Derives a closed-form angular spectrum of an arbitrarily rotated/scaled/translated 2D Gaussian at the SLM plane (Gaussian Wave Splatting), a wave-optics counterpart of volume rendering (alpha wave blending) for occlusion, and a fast Fourier-domain approximation based on order-invariant transparency implemented in CUDA (~30x faster than exact). Validated in simulation and on a benchtop holographic display prototype; 5-11 dB PSNR over point- and polygon-based CGH.
- **Notes**: Source paper for the animation in this project.
---

### [kerbl3Dgaussians] 3D Gaussian Splatting for Real-Time Radiance Field Rendering
- **Authors**: Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, George Drettakis
- **Venue**: ACM TOG 42(4) (SIGGRAPH 2023); arXiv 2308.04079
- **URL**: https://arxiv.org/abs/2308.04079
- **Status**: discovered
- **Topics**: Gaussian splatting, novel-view synthesis
- **Abstract**: Represents scenes as anisotropic 3D Gaussians initialized from SfM points, optimized with interleaved density control, and rendered with a fast visibility-aware tile rasterizer (>= 100 fps at 1080p) at state-of-the-art quality.
- **Notes**: Origin of the Gaussian scene representation GWS builds on; GWS borrows its front-to-back alpha compositing and the 1/255 alpha threshold.
---

### [huang20242d] 2D Gaussian Splatting for Geometrically Accurate Radiance Fields
- **Authors**: Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, Shenghua Gao
- **Venue**: ACM SIGGRAPH 2024 Conference Papers; arXiv 2403.17888
- **URL**: https://arxiv.org/abs/2403.17888
- **Status**: discovered
- **Topics**: Gaussian splatting, surface reconstruction
- **Abstract**: Collapses 3D Gaussians into oriented planar 2D Gaussian disks (center, two tangent vectors, two scales), with perspective-accurate ray-splat intersection plus depth-distortion and normal-consistency losses, giving view-consistent geometry.
- **Notes**: GWS uses 2DGS because flat primitives have well-defined angular spectra on planes (like polygons in polygon-based CGH).
---

### [matsushima2009extremely] Extremely high-definition full-parallax computer-generated hologram created by the polygon-based method
- **Authors**: Kyoji Matsushima, Sumio Nakahara
- **Venue**: Applied Optics 48(34), H54-H63, 2009
- **URL**: https://doi.org/10.1364/AO.48.000H54
- **Status**: discovered
- **Topics**: polygon-based CGH
- **Abstract**: Computes full-parallax holograms of polygon meshes by computing each polygon's field via tilted-plane angular-spectrum diffraction; demonstrates billion-pixel-scale CGH.
- **Notes**: Polygon-based baseline in Fig. 5 of GWS.
---

### [matsushima2014silhouette] Silhouette method for hidden surface removal in computer holography and its acceleration using the switch-back technique
- **Authors**: Kyoji Matsushima, Masaki Nakamura, Sumio Nakahara
- **Venue**: Optics Express 22(20), 24450-24465, 2014
- **URL**: https://doi.org/10.1364/OE.22.024450
- **Status**: discovered
- **Topics**: occlusion in CGH, polygon-based CGH
- **Abstract**: Hidden-surface removal for polygon CGH: propagate the field from back to front and mask it with each object's binary silhouette aperture; switch-back technique accelerates the masking.
- **Notes**: Mesh baseline (Tables 1, 3; Figs. 3, 4). GWS shows binary silhouettes cannot represent smooth Gaussian falloffs / alpha blending.
---

### [chen2009computer] Computer generated hologram with geometric occlusion using GPU-accelerated depth buffer rasterization for three-dimensional display
- **Authors**: Rick H.-Y. Chen, Timothy D. Wilkinson
- **Venue**: Applied Optics 48(21), 4246-4255, 2009
- **URL**: https://doi.org/10.1364/AO.48.004246
- **Status**: discovered
- **Topics**: point-based CGH, occlusion
- **Abstract**: Point-cloud CGH with occlusion from a GPU depth-buffer visibility test; points act as opaque disks.
- **Notes**: Point-based baseline (Tables 1, 3; Fig. 5).
---

### [matsushima2009band] Band-limited angular spectrum method for numerical simulation of free-space propagation in far and near fields
- **Authors**: Kyoji Matsushima, Tomoyoshi Shimobaba
- **Venue**: Optics Express 17(22), 19662-19673, 2009
- **URL**: https://doi.org/10.1364/OE.17.019662
- **Status**: discovered
- **Topics**: wave propagation, angular spectrum method
- **Abstract**: Band-limits the ASM transfer function to avoid aliasing so FFT-based propagation stays accurate at long distances.
- **Notes**: Propagation operator P(.; z) in GWS Eq. 3-4.
---

### [maimone2017holographic] Holographic near-eye displays for virtual and augmented reality
- **Authors**: Andrew Maimone, Andreas Georgiou, Joel S. Kollin
- **Venue**: ACM TOG 36(4) (SIGGRAPH 2017)
- **URL**: https://doi.org/10.1145/3072959.3073624
- **Status**: discovered
- **Topics**: holographic near-eye displays, point-based CGH, double-phase encoding
- **Abstract**: Near-eye holographic display prototypes with aberration correction, focus cues and compact form factors; real-time point-based CGH and double-phase amplitude encoding for phase-only SLMs.
- **Notes**: GWS encodes its complex SLM field to phase-only with DPAC following this work.
---

### [mcguire2013weighted] Weighted Blended Order-Independent Transparency
- **Authors**: Morgan McGuire, Louis Bavoil
- **Venue**: Journal of Computer Graphics Techniques 2(2), 2013 (the GWS bibliography lists it as 2(4))
- **URL**: https://jcgt.org/published/0002/02/09/
- **Status**: discovered
- **Topics**: order-independent transparency, real-time rendering
- **Abstract**: Approximates transparency compositing with a weighted sum that does not depend on primitive order, enabling single-pass GPU transparency.
- **Notes**: Inspiration for fast GWS (2DGS-OIT: c = sum c_i G_i o_i) which makes the Fourier-domain sum per-frequency parallel.
---
