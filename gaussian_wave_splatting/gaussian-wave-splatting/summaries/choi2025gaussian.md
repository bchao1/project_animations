# [choi2025gaussian] Gaussian Wave Splatting for Computer-Generated Holography

Suyeon Choi*, Brian Chao*, Jacqueline Yang, Manu Gopakumar, Gordon Wetzstein (Stanford; * equal contribution)
ACM TOG 44(4), Article 111 (SIGGRAPH 2025), DOI 10.1145/3731163 · arXiv 2505.06582 (10 May 2025) · 59 pp. incl. supplement
PDF: `../pdfs/choi2025gaussian.pdf` · text: `../pdfs/choi2025gaussian.txt` · LaTeX source: `../pdfs/src/` · figure rasters: `../pdfs/figs/`
Project: https://bchao1.github.io/gaussian-wave-splatting/ · code: https://github.com/computational-imaging/hsplat

## Problem
Holographic near-eye displays need a CGH algorithm that turns content into an SLM pattern. Existing CGH
is either very slow and needs unusual inputs (dense light fields, focal stacks + SGD), or uses inputs that
lack occlusion / view-dependent effects (RGB-D, point clouds) or photorealism (per-face-colored meshes).
Meanwhile Gaussian splatting gives photoreal, explicit 3D scenes from a few photos. Goal: turn Gaussians
directly into holograms, with occlusion.

## Background
- 2DGS primitive: center μ ∈ R³, tangent vectors t_u, t_v, scales s_u, s_v.
- Ray-based compositing (Eq. 1): c = Σ_i c_i α_i Π_{j<i} (1 − α_j), with α_i = G_i(u, v) · o_i (Eq. 2).
- Angular spectrum method (Eqs. 3-4): P(u; z) = ∬ F(u) H(k, z) e^{j(k_x x + k_y y)} dk_x dk_y,
  H(k, z) = e^{j k_z z} if √(k_x² + k_y²) < 2π/λ else 0, |k| = 2π/λ.

## Method
1. **Alpha wave blending** (Eqs. 5-6), wave counterpart of volume rendering, composited at the SLM (z = 0):
   u_SLM(x) = Σ_i c_i o_i P( u_i(x; z_i) T_i(x) ; −z_i ),  T_i(x) = Π_{j<i} (1 − o_j |u_j(x; z_j)|).
   Front-to-back (near SLM → far), global transmittance map maintained. Reduces to ray alpha blending if
   propagation and phase are ignored (u_i = G_i). Supplement: recursion collapses into a sum because
   propagations compose (P(P(u; a); b) = P(u; a + b)).
   Prior silhouette methods (Matsushima 2014) mask with binary apertures, back to front → slow and
   wrong for smooth Gaussian falloff.
2. **Gaussian Wave Splatting equation** (closed form, avoids per-primitive FFT + nonlinear tilted remap):
   - Canonical Gaussian (Eq. 7): u_c = e^{−(x_c² + y_c²)/2} ↔ Û_c(k_c) = 2π e^{−2π² k_cᵀ k_c}.
   - Hologram space: x = R_i S_i x_c + μ_i, Σ_i = R_i S_i S_iᵀ R_iᵀ (z-scale 0 → flat disk).
   - Affine theorem (Eq. 8): scale → Û_o(k_o) = det(S) Û_c(S k_o).
   - Rotation: wave vectors rotate too, k_o = R_i⁻¹ k (tilted-plane rotational transform, Matsushima 2003);
     translation → phase ramp e^{j k·μ_i}.
   - Result (Eqs. 9-10): Û_i(k) = det(J) det(S_i) Û_c(S_i R_i⁻¹ k) e^{j k·μ_i}
     = 2π det(J) det(S_i) e^{−2π² kᵀ Σ_i k} e^{j k·μ_i},  det(J) = k_oz / k_z.
   - Reference plane wave along k_r → shift of the angular spectrum by k_r. Phases of primitives matched so
     wavefronts interfere constructively (smooth-phase holograms). Generalizes to polygons by swapping S
     for the triangle's affine map (supplement).
3. **Fast GWS** (Sec. 4.2): in Fourier domain (Eq. 11) T_i becomes a convolution, which blocks per-frequency
   parallelism. Drop it (T_i ≈ 1, Eq. 12): Û_SLM(k) ≈ Σ_i c_i o_i Û_i(k) H(k, −z_i) → embarrassingly parallel
   per frequency; custom CUDA kernels. Spatial analogue (Eq. 13) = order-invariant transparency (McGuire &
   Bavoil 2013): c ≈ Σ c_i G_i o_i. Scenes re-optimized as **2DGS-OIT** with **view-dependent opacity**
   (opacity as SH coefficients, α_i = G_i · SH(o_i, d)) to restore ordering cues (Fig. 2).
4. **Phase-only encoding**: double phase-amplitude coding (DPAC; Hsueh 1978, Maimone 2017), two phase values
   per complex sample interlaced as a checkerboard.

## Setup
Scenes: NeRF synthetic Blender + Mip-NeRF 360. 2DGS via gsplat; 3DGS-MCMC densification with capped
Gaussian count. Baselines: point cloud (Chen & Wilkinson 2009; SfM points / points sampled on NeRF2Mesh
meshes), polygon/silhouette (Matsushima 2009/2014; NeRF2Mesh meshes with per-face colours). All in PyTorch
(`hsplat` library), fast GWS in CUDA.
Prototype: FISBA READYBeam Bio2 fiber-coupled lasers 638 / 520 / 488 nm; Holoeye PLUTO phase-only LCoS SLM,
1920 × 1080, 8 µm pitch, 8-bit; FLIR Grasshopper3 2.3 MP camera; colour by separate exposures per wavelength.
Depth range ≈ 1 cm in hologram space ↔ optical infinity to ≈ 0.5 m in view space. Spatially varying LUT
calibration, 3π phase range, Fourier filter.

## Key numbers
- Table 1 (simulated, PSNR / SSIM / LPIPS):
  | Method | Blender | Mip-NeRF 360 |
  |---|---|---|
  | Chen & Wilkinson 2009 – point cloud | 18.34 / 0.77 / 0.20 | 9.45 / 0.074 / 0.70 |
  | Matsushima 2014 – mesh | 20.79 / 0.81 / 0.18 | 15.77 / 0.27 / 0.65 |
  | Ours (fast) – Gaussians | 26.55 / 0.92 / 0.10 | 20.70 / 0.59 / 0.42 |
  | Ours (exact) – Gaussians | 27.30 / 0.91 / 0.11 | 22.37 / 0.58 / 0.40 |
  "5–11 dB PSNR improvement over baseline methods."
- Table 2 (runtime; baselines quoted from their papers):
  Chen 2009: 1024 × 1280, 15k points, 2.47 s · Matsushima 2014: 2000 × 2000, 15k triangles, 13.5 s ·
  Ours fast: 1024 × 1280 15k Gaussians 6.84 s; 2000 × 2000 18.06 s ·
  Ours exact: 1024 × 1280 180 s; 2000 × 2000 470 s → fast ≈ 30× faster than exact (180/6.84 ≈ 26×, 470/18.06 ≈ 26×; paper states "nearly 30×").
- Fig. 4 toy scene: red Gaussian at zero-disparity plane, blue Gaussian 2 cm behind; partially coherent,
  24 frames; baselines need a 0.1 threshold.
- Not real time yet (limitation).

## Figures
1. Teaser — multiview captures → Gaussian scene → GWS hologram (phase map, −π…π) → captured front/back focus (garden).
2. OIT ablation (kitchen): naïve OIT vs OIT + view-dependent opacity.
3. Coordinate systems (canonical → object → hologram space); plane wave illuminating tilted 2D Gaussian, propagating to SLM.
4. Wave blending ablation (stump): naïve / silhouette (binary aperture) / alpha wave blending / target.
5. Captured focal stacks (lego, materials): point / polygon / fast GWS / exact GWS, target renderings + rear/middle/front insets.
6. Captured focal stacks of exact GWS from 3 novel views (garden), rear/middle/front slices.
7. Parallax toy (red/blue Gaussians): RGB-D / silhouette / ours / target, left/center/right views.
Supplement: setup photo, runtime heat maps (Gaussians 1e4–1e6 × resolution up to 2048), LUTs, per-scene tables, holographics pipeline.

## Limitations / future work
Focus on in-focus imagery: smooth-phase → unnatural defocus blur, limited eyebox (random-phase extension
is future work); DPAC encoding could be replaced by learned / gradient-based encoders; CUDA version not yet
real time; limited étendue → only 0th-order SH for view-dependent effects on current hardware.
