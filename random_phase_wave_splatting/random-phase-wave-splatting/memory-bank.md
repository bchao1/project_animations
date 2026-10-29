# Paper Memory Bank
Last updated: 2026-09-25

### [chao2025random] Random-phase Wave Splatting of Translucent Primitives for Computer-generated Holography
- **Authors**: Brian Chao*, Jacqueline Yang*, Suyeon Choi, Manu Gopakumar, Ryota Koiso, Gordon Wetzstein (Stanford University; Koiso also KDDI Research; * equal contribution)
- **Venue**: arXiv preprint 2508.17480 (cs.GR), v1 24 Aug 2025 (titled "Random-phase Gaussian Wave Splatting for Computer-generated Holography"), v3 11 Dec 2025. No journal reference on arXiv; the author's homepage lists it as "arXiv, 2025". The LaTeX uses the acmtog class in `nonacm` mode.
- **URL**: https://arxiv.org/abs/2508.17480 (no project page / code yet: "Source code and example datasets will be made public.")
- **Citations**: n/a
- **Status**: analyzed (PDF, text, LaTeX source ingested; see summaries/chao2025random.md)
- **Topics**: computer-generated holography, random-phase holograms, primitive-based CGH, alpha blending of wavefronts, time multiplexing, Gaussian / triangle splatting
- **Abstract**: Random-phase Wave Splatting (RPWS) converts any 3D representation made of translucent 2D primitives (2D Gaussians, soft-edged triangles) into time-multiplexed random-phase holograms. Core: back-to-front wave compositing with sqrt transmittance masks, and intensity-domain ("sqrt") alpha blending that is exact in expectation over random phases, so time multiplexing is the alpha-blending mechanism, not just speckle reduction. Fills the SLM bandwidth (large eyebox), natural defocus blur and parallax; 24.87 / 21.21 dB light-field PSNR (Blender / Mip-NeRF 360, 24 frames) vs 14.35 / 10.50 dB for GWS.
- **Notes**: Source paper for this animation project. Direct follow-up of GWS (choi2025gaussian).
---

### [choi2025gaussian] Gaussian Wave Splatting for Computer-Generated Holography
- **Authors**: Suyeon Choi*, Brian Chao*, Jacqueline Yang, Manu Gopakumar, Gordon Wetzstein (Stanford University)
- **Venue**: ACM TOG 44(4), Article 111 (SIGGRAPH 2025), DOI 10.1145/3731163; arXiv 2505.06582
- **URL**: https://arxiv.org/abs/2505.06582
- **Status**: analyzed (full record in ../../gaussian_wave_splatting/gaussian-wave-splatting/)
- **Topics**: primitive-based CGH, Gaussian splatting, alpha wave blending
- **Abstract**: Closed-form angular spectrum of a 2D Gaussian at the SLM, alpha wave blending (front-to-back, amplitude domain) for occlusion, fast Fourier-domain OIT variant; smooth-phase holograms.
- **Notes**: The baseline RPWS generalizes. Its supplement sketched random phase for a single Gaussian; RPWS shows that naively adding random phase to GWS (GWS-NRP) gives wrong colours, and that GWS's smooth phase gives a small eyebox, little parallax and ringing defocus.
---

### [choi2022time] Time-multiplexed Neural Holography: A flexible framework for holographic near-eye displays with fast heavily-quantized spatial light modulators
- **Authors**: Suyeon Choi, Manu Gopakumar, Yifan Peng, Jonghyun Kim, Matthew O'Toole, Gordon Wetzstein
- **Venue**: ACM SIGGRAPH 2022 Conference Proceedings; arXiv 2205.02367
- **URL**: https://arxiv.org/abs/2205.02367
- **Status**: discovered
- **Topics**: time multiplexing, partially coherent holography, light-field CGH (STFT)
- **Abstract**: Optimizes multiple quantized phase patterns shown in rapid succession on fast SLMs so their intensities average into the target; supports 2D, RGBD, focal-stack and 4D light-field supervision.
- **Notes**: RPWS uses its STFT operator to reconstruct 4D light fields; RPWS reinterprets time multiplexing as exact alpha blending.
---

### [kim2024holographic] Holographic Parallax Improves 3D Perceptual Realism
- **Authors**: Dongyeon Kim, Seung-Woo Nam, Suyeon Choi, Jong-Mo Seo, Gordon Wetzstein, Yoonchan Jeong
- **Venue**: ACM TOG 43(4) (SIGGRAPH 2024); arXiv 2404.11810; DOI 10.1145/3658168
- **URL**: https://arxiv.org/abs/2404.11810
- **Citations**: 43 (Semantic Scholar, 2026-09)
- **Status**: discovered
- **Topics**: holographic parallax, eyebox, perceptual studies, fast FLCoS SLM
- **Abstract**: Shows with user studies that holograms reproducing parallax across the pupil/eyebox look more realistic; light-field holograms on a 3600 fps FLCoS.
- **Notes**: Motivation for RPWS's large eyebox; RPWS cites its 3600 fps FLCoS as a path to many time-multiplexed frames.
---

### [kim2022accommodative] Accommodative holography: improving accommodation response for perceptually realistic holographic displays
- **Authors**: Dongyeon Kim, Seung-Woo Nam, Byounghyo Lee, Jong-Mo Seo, Byoungho Lee
- **Venue**: ACM TOG 41(4) (SIGGRAPH 2022); DOI 10.1145/3528223.3530147
- **URL**: https://doi.org/10.1145/3528223.3530147
- **Status**: discovered
- **Topics**: accommodation, defocus cues, random vs smooth phase
- **Abstract**: Shows that the hologram's phase distribution (angular spread) determines whether it can drive the eye's accommodation; smooth-phase holograms give weak defocus cues.
- **Notes**: Cited by RPWS for why smooth-phase holograms have a small eyebox and poor focus cues.
---

### [schiffers2023stochastic] Stochastic Light Field Holography
- **Authors**: Florian Schiffers, Praneeth Chakravarthula, Nathan Matsuda, Grace Kuo, Ethan Tseng, Douglas Lanman, Felix Heide, Oliver Cossairt
- **Venue**: IEEE ICCP 2023; arXiv 2307.06277
- **URL**: https://arxiv.org/abs/2307.06277
- **Status**: discovered
- **Topics**: light-field holography, pupil sampling, parallax
- **Abstract**: Supervises hologram optimization with light-field refocused photographs from stochastically sampled pupil states, giving correct parallax and focus cues.
- **Notes**: Cited by RPWS among works showing smooth-phase holograms are undesirable (bandwidth, defocus, pupil sensitivity).
---

### [matsushima2005computer] Computer-generated holograms for three-dimensional surface objects with shade and texture
- **Authors**: Kyoji Matsushima
- **Venue**: Applied Optics 44(22), 4607-4614, 2005
- **URL**: https://doi.org/10.1364/AO.44.004607
- **Status**: discovered
- **Topics**: polygon-based CGH, random phase diffuse surfaces
- **Abstract**: Polygon-based CGH with diffuse (random-phase) surface functions giving shading and texture on tilted polygons.
- **Notes**: Basis of the Polygons-RP baseline (with the silhouette method, Matsushima 2014) in RPWS Table 1 and Figs. 4-6.
---

### [held2025triangle] Triangle Splatting for Real-Time Radiance Field Rendering
- **Authors**: Jan Held, Renaud Vandeghen, Adrien Deliege, Abdullah Hamdi, Silvio Giancola, Anthony Cioppa, Andrea Vedaldi, Bernard Ghanem, Andrea Tagliasacchi, Marc Van Droogenbroeck
- **Venue**: arXiv 2505.19175 (2025)
- **URL**: https://arxiv.org/abs/2505.19175
- **Status**: discovered
- **Topics**: triangle primitives, differentiable rendering, radiance fields
- **Abstract**: Optimizes a soup of soft-edged triangles (smoothness parameter σ) end-to-end as differentiable splats; sharp edges, game-engine compatible.
- **Notes**: The soft-edged triangle primitives used for RPWS-Δ (qualitative results, Figs. 5-6).
---

### [huang20242d] 2D Gaussian Splatting for Geometrically Accurate Radiance Fields
- **Authors**: Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, Shenghua Gao
- **Venue**: ACM SIGGRAPH 2024 Conference Papers; arXiv 2403.17888
- **URL**: https://arxiv.org/abs/2403.17888
- **Status**: discovered
- **Topics**: Gaussian splatting
- **Abstract**: Flat, oriented 2D Gaussian disks with ray-splat intersection.
- **Notes**: The primitive of RPWS-GS (as in GWS).
---
