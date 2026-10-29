# Mind Graph
Last updated: 2026-09-25

### Primitive-based Computer-Generated Holography
- **Description**: CGH that sums wavefronts of scene primitives (points, polygons, Gaussians, triangles) at the SLM
- **Related topics**: [Phase Distributions of Holograms], [Alpha Blending of Wavefronts], [Splatting Scene Representations]
- **Key papers**:
  - [chao2025random] Random-phase Wave Splatting (arXiv 2025) — random-phase holograms from any translucent 2D primitive
  - [choi2025gaussian] Gaussian Wave Splatting (TOG/SIGGRAPH 2025) — smooth-phase Gaussian holograms, closed-form spectrum
  - [matsushima2005computer] Polygon CGH with shade and texture (Appl. Opt. 2005) — random-phase polygons (Polygons-RP baseline)

### Phase Distributions of Holograms (smooth vs random phase)
- **Description**: The phase on object surfaces sets the hologram's angular spectrum: eyebox size, defocus blur, parallax, speckle
- **Related topics**: [Primitive-based Computer-Generated Holography], [Time Multiplexing / Partial Coherence]
- **Key papers**:
  - [kim2022accommodative] Accommodative holography (TOG 2022) — phase distribution drives accommodation
  - [kim2024holographic] Holographic parallax (TOG 2024) — parallax across the eyebox improves realism
  - [chao2025random] RPWS — full-bandwidth random-phase splatting
- **Other relevant papers**:
  - [schiffers2023stochastic] Stochastic light field holography (ICCP 2023) — pupil-sampled supervision

### Time Multiplexing / Partial Coherence
- **Description**: Several holograms shown in quick succession average in intensity (speckle reduction, light fields)
- **Related topics**: [Phase Distributions of Holograms], [Alpha Blending of Wavefronts]
- **Key papers**:
  - [choi2022time] Time-multiplexed neural holography (SIGGRAPH 2022)
  - [chao2025random] RPWS — time multiplexing = exact alpha blending in expectation

### Alpha Blending of Wavefronts (occlusion)
- **Description**: How translucent primitives occlude each other in the wave domain
- **Related topics**: [Primitive-based Computer-Generated Holography]
- **Key papers**:
  - [choi2025gaussian] Alpha wave blending — amplitude domain, front to back (smooth phase)
  - [chao2025random] Random-phase alpha wave blending — intensity domain (sqrt weights), back to front with sqrt masks

### Splatting Scene Representations
- **Description**: Explicit translucent primitives optimized from photos
- **Related topics**: [Primitive-based Computer-Generated Holography]
- **Key papers**:
  - [huang20242d] 2D Gaussian Splatting (SIGGRAPH 2024)
  - [held2025triangle] Triangle Splatting (arXiv 2025) — soft-edged triangles, used by RPWS-Δ
