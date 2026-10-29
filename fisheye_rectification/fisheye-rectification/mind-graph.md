# Mind Graph
Last updated: 2026-09-27

### Fisheye / lens-distortion rectification
- **Description**: Recover a pinhole (distortion-free) image from a wide-angle fisheye image.
- **Related topics**: [Unpaired image-to-image translation], [Differentiable warping]
- **Key papers**:
  - [chao2020selfsupervised] FE-GAN (ICASSP 2020) — self-supervised, unpaired, predicts a pixel flow; no camera parameters
  - [li2019blind] Blind geometric distortion correction (CVPR 2019) — supervised pixel-flow regression
- **Other relevant papers**:
  - [yin2018fisheyerecnet] FishEyeRecNet — supervised parameters + segmentation
  - [xue2019learning] straight-line supervision
  - [aleman2014automatic] classical plumb-line / division model
  - [fitzgibbon2001simultaneous] division model
  - [basu1995alternative] fisheye transform (data synthesis)
  - [feng2023simfir] SimFIR — later self-supervised representation learning

### Unpaired image-to-image translation
- **Description**: Learn X → Y mappings from two unpaired image sets with adversarial losses plus consistency constraints.
- **Related topics**: [Fisheye rectification]
- **Key papers**:
  - [zhu2017cyclegan] CycleGAN — cycle consistency
  - [fu2019gcgan] GcGAN — geometry (rotation/flip) consistency; inspires FE-GAN's cross-rotation loss
- **Other relevant papers**:
  - [liu2017unit] UNIT — shared latent space

### Differentiable warping
- **Description**: Resample an image with a predicted coordinate field, trainable end to end.
- **Key papers**:
  - [jaderberg2015stn] Spatial Transformer Networks — FE-GAN's warp module W
