# Mind Graph
Last updated: 2026-09-24

### Primitive-based Computer-Generated Holography
- **Description**: CGH algorithms that sum the wavefronts of scene primitives (points, polygons, Gaussians) at the SLM plane
- **Related topics**: [Occlusion in Wave Optics], [Wave Propagation], [Gaussian Scene Representations]
- **Key papers**:
  - [choi2025gaussian] Gaussian Wave Splatting (TOG/SIGGRAPH 2025) — closed-form 2D Gaussian → hologram transform, alpha wave blending, fast OIT CUDA variant
  - [chen2009computer] Point-cloud CGH with depth-buffer occlusion (Appl. Opt. 2009)
  - [matsushima2009extremely] Polygon-based full-parallax CGH (Appl. Opt. 2009)
  - [maimone2017holographic] Holographic near-eye displays (TOG 2017) — point-based CGH, DPAC
  - Subtopic: **Occlusion in Wave Optics**
    - [matsushima2014silhouette] Silhouette method — binary-aperture masking
    - [choi2025gaussian] Alpha wave blending — smooth alpha, front-to-back transmittance

### Wave Propagation
- **Description**: Numerical free-space diffraction (angular spectrum, Fresnel), incl. tilted planes
- **Related topics**: [Primitive-based Computer-Generated Holography]
- **Key papers**:
  - [matsushima2009band] Band-limited angular spectrum method (Opt. Express 2009)

### Gaussian Scene Representations
- **Description**: Explicit Gaussian primitives optimized from photos for novel-view synthesis
- **Related topics**: [Primitive-based Computer-Generated Holography], [Order-Independent Transparency]
- **Key papers**:
  - [kerbl3Dgaussians] 3D Gaussian Splatting (TOG 2023)
  - [huang20242d] 2D Gaussian Splatting (SIGGRAPH 2024) — flat disks used by GWS

### Order-Independent Transparency
- **Description**: Compositing approximations that drop primitive ordering for parallelism
- **Related topics**: [Gaussian Scene Representations]
- **Key papers**:
  - [mcguire2013weighted] Weighted blended OIT (JCGT 2013) — basis of 2DGS-OIT / fast GWS
