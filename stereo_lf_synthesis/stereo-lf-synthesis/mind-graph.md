# Mind Graph
Last updated: 2026-09-29

### Light field synthesis from extremely sparse views
- **Description**: Build a dense N × N light field from 1, 2 or 4 captured views.
- **Related topics**: [Angular super-resolution], [Stereo view synthesis], [Light field displays]
- **Key papers**:
  - [chao2021robust] Robust LF Synthesis From Stereo (ICIP 2021) — 2 views, left-right consistency, alpha blending, PSAS 3D CNN
  - [srinivasan2017learning] 4D RGBD LF from a Single Image (ICCV 2017) — 1 view; depth is ill-posed
  - [kalantari2016learning] Learning-Based View Synthesis (SIGGRAPH Asia 2016) — 4 corner views
- **Other relevant papers**:
  - [zhang2015light] LF from Micro-Baseline Image Pair (CVPR 2015) — 2 views, non-learning, left view only
  - [chao2022view] View Synthesis and Distortion Compensation for LF AR (ISMAR/TVCG 2022) — follow-up

### Angular super-resolution (dense grid → denser grid)
- **Description**: Interpolate e.g. 3 × 3 → 7 × 7 views; assumes a grid can be captured.
- **Key papers**: [wu2017light] EPI CNN (CVPR 2017); [wang2018end] Pseudo 4DCNN (ECCV 2018)
- **Other relevant papers**: [liu2020light] refocused-image-domain training (TIP 2020)

### Stereo view synthesis / extrapolation
- **Description**: Novel views from a stereo pair via depth, MPIs or warping.
- **Key papers**: [zhou2018stereo] Stereo Magnification (SIGGRAPH 2018) — needs camera poses
- **Other relevant papers**: [zhang2015light]; [chao2021robust]

### Light field capture and datasets
- **Key papers**: [ng2005light] plenoptic camera; [honauer2016dataset] HCI 4D benchmark
- **Other relevant papers**: INRIA Lytro 1G dataset; Stanford (New) Light Field Archive

### Light field displays (downstream use)
- **Key papers**: [chao2023time] CTDM-LFD (TIP 2023); [chao2022view] LF AR glasses (TVCG 2022)
