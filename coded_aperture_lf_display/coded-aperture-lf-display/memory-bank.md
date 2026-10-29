# Paper Memory Bank
Last updated: 2026-09-27

### [chao2023time] Time-Division Multiplexing Light Field Display With Learned Coded Aperture
- **Authors**: Brian Chao* (Chun-Hao Chao), Chang-Le Liu*, Homer H. Chen (National Taiwan University; * co-first authors; corresponding author H. H. Chen)
- **Venue**: IEEE Transactions on Image Processing, vol. 32, pp. 350–363, 2023 (DOI 10.1109/TIP.2022.3203210, IEEE Xplore 9987726; the author homepage lists it as "IEEE TIP, 2022"). arXiv 2107.06205 (v1 13 Jul 2021 as "Learning Coded Apertures for Time-Division Multiplexing Light Field Display"; v3 23 Aug 2022 = final title).
- **URL**: https://arxiv.org/abs/2107.06205 · https://ieeexplore.ieee.org/document/9987726 (paywalled). No code repository or supplementary video found (GitHub user bchao1 has none; web search 2026-09-27).
- **Citations**: n/a (Semantic Scholar rate-limited)
- **Status**: analyzed (arXiv v3 PDF + extracted text + page renders; no LaTeX source on arXiv — PDF-only submission; see summaries/chao2023time.md)
- **Topics**: light field displays, vergence-accommodation conflict, coded aperture, time-division multiplexing, end-to-end optics optimization, digital refocusing
- **Abstract**: CTDM-LFD: a near-eye light field display that shows k encoded images in sequence, each through its own learned coded aperture (l × l = 9 × 9 pattern at the relay lens's aperture / Fourier plane); the eye integrates them (persistence of vision) into a refocused image with natural defocus blur. Encoding CNN f (n sparse views → k images) and apertures P̂ are learned end-to-end through a Fourier-optics image-formation model with a focus-weighted focal-stack L1 loss. Beats the baseline TDM-LFD on INRIA-syn / INRIA-real / SIGGRAPH16 (e.g. 32.29 vs 28.56 dB PSNR, INRIA-syn, n = k = 4); verified with a Zemax relay-lens design using binary 9 × 9 apertures.
- **Notes**: Source paper for this animation project. Author's website teaser: https://bchao1.github.io/images/tip2021.gif (2 frames, near/far refocus of the agapanthus scene).
---

### [levoy1996light] Light Field Rendering
- **Authors**: Marc Levoy, Pat Hanrahan
- **Venue**: SIGGRAPH 1996
- **URL**: https://doi.org/10.1145/237170.237199
- **Status**: discovered
- **Topics**: light fields, two-plane parameterization, refocusing by shift-and-add
- **Abstract**: 4D light field L(x, y, s, t); new views by resampling; digital refocusing via shift-and-add of sub-aperture views.
- **Notes**: Ref. [26] of chao2023time; its L(x, y, s, t) notation (Table I) is used in the paper and in our video.
---

### [ng2005fourier] Fourier Slice Photography
- **Authors**: Ren Ng
- **Venue**: SIGGRAPH 2005
- **URL**: https://doi.org/10.1145/1186822.1073256
- **Status**: discovered
- **Topics**: refocusing, Fourier slice theorem, light field photography
- **Abstract**: A photograph is a 2D slice of the 4D light field's Fourier transform; fast digital refocusing.
- **Notes**: Ref. [27]; refocusing assumes a dense light field — the sparse-view case is what CTDM-LFD targets.
---

### [lanman2013near] Near-Eye Light Field Displays
- **Authors**: Douglas Lanman, David Luebke
- **Venue**: ACM TOG 32(6) (SIGGRAPH Asia 2013)
- **URL**: https://doi.org/10.1145/2508363.2508366
- **Status**: discovered
- **Topics**: integral-imaging near-eye display, microlens arrays, accommodation cues
- **Abstract**: Microlens array over a microdisplay gives a thin HMD with focus cues, at reduced spatial resolution.
- **Notes**: Ref. [22]; the space-division (SDM) design whose spatial/angular trade-off motivates TDM / CTDM.
---

### [grosse2010coded] Coded Aperture Projection
- **Authors**: Max Grosse, Gordon Wetzstein, Anselm Grundhöfer, Oliver Bimber
- **Venue**: ACM TOG 29(3), 2010
- **URL**: https://doi.org/10.1145/1805964.1805966
- **Status**: discovered
- **Topics**: coded aperture, projector depth of field
- **Abstract**: Programmable coded apertures in projectors with pre-compensated images to extend depth of field.
- **Notes**: Ref. [45]; "seminal work" that inspired using coded apertures in a display.
---

### [liang2008programmable] Programmable Aperture Photography: Multiplexed Light Field Acquisition
- **Authors**: Chia-Kai Liang, Tai-Hsu Lin, Bing-Yi Wong, Chi Liu, Homer H. Chen
- **Venue**: ACM TOG 27(3) (SIGGRAPH 2008)
- **URL**: https://doi.org/10.1145/1360612.1360654
- **Status**: discovered
- **Topics**: coded aperture, multiplexed light field capture
- **Abstract**: Captures light fields at full sensor resolution by time-multiplexing programmable aperture patterns.
- **Notes**: Ref. [43]; same lab (H. H. Chen, NTU) — CTDM-LFD is its display-side counterpart.
---

### [inagaki2018learning] Learning to Capture Light Fields Through a Coded Aperture Camera
- **Authors**: Yasutaka Inagaki, Yuto Kobayashi, Keita Takahashi, Toshiaki Fujii, Hajime Nagahara
- **Venue**: ECCV 2018
- **Status**: discovered
- **Topics**: learned coded aperture, light field acquisition
- **Abstract**: Jointly learns aperture patterns and a CNN reconstruction for coded-aperture light field capture.
- **Notes**: Ref. [46]; learning-based coded aperture inspiration.
---

### [maruyama2019display] A 3-D Display Pipeline from Coded-Aperture Camera to Tensor Light-Field Display Through CNN
- **Authors**: Kohei Maruyama, Yasutaka Inagaki, Keita Takahashi, Toshiaki Fujii, Hajime Nagahara
- **Venue**: ICIP 2019
- **Status**: discovered
- **Topics**: coded aperture, tensor light field display
- **Abstract**: End-to-end CNN from coded-aperture captures to layered (tensor) display patterns.
- **Notes**: Ref. [44]; one of the few coded-aperture + light-field-display works.
---

### [wang2021time] Time-Multiplexed Integral Imaging Based Light Field Displays
- **Authors**: Xuan Wang, Hong Hua
- **Venue**: Proc. SPIE 11765, 2021
- **Status**: discovered
- **Topics**: time-multiplexed integral imaging displays
- **Abstract**: Time-multiplexing to relax the spatial–angular resolution trade-off of integral-imaging LFDs.
- **Notes**: Ref. [16].
---

### [kalantari2016learning] Learning-Based View Synthesis for Light Field Cameras
- **Authors**: Nima Khademi Kalantari, Ting-Chun Wang, Ravi Ramamoorthi
- **Venue**: ACM TOG 35(6) (SIGGRAPH Asia 2016)
- **Status**: discovered
- **Topics**: light field view synthesis, Lytro dataset
- **Abstract**: CNN view synthesis for Lytro light fields; its dataset = "SIGGRAPH16" in chao2023time (130 scenes, 8 × 8).
- **Notes**: Ref. [55]; dataset source.
---

### [zabels2019ar] AR Displays: Next-Generation Technologies to Solve the Vergence–Accommodation Conflict
- **Authors**: Roberts Zabels et al.
- **Venue**: Applied Sciences 9(15), 2019
- **Status**: discovered
- **Topics**: VAC survey, varifocal / multifocal / light field / holographic displays
- **Abstract**: Survey of AR display approaches for resolving VAC.
- **Notes**: Ref. [50].
---
