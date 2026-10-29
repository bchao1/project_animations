# Paper Memory Bank
Last updated: 2026-09-27

### [chao2020selfsupervised] Self-Supervised Deep Learning for Fisheye Image Rectification
- **Authors**: Brian Chao*, Pin-Lun Hsu*, Hung-Yi Lee, Yu-Chiang Frank Wang (* equal contribution; National Taiwan University; Wang also ASUS Intelligent Cloud Services)
- **Venue**: IEEE ICASSP 2020, pp. 2248–2252, DOI 10.1109/ICASSP40776.2020.9054191
- **URL**: https://ieeexplore.ieee.org/document/9054191 · code https://github.com/ByronHsu/FEGAN · website teaser https://bchao1.github.io/images/icassp2020-result.gif
- **Citations**: 32 (OpenAlex, 2026-09)
- **Status**: analyzed (full PDF in pdfs/, text extracted, figures extracted; summary in summaries/)
- **Topics**: fisheye rectification, unpaired image-to-image translation, GAN, self-supervised learning, flow-based warping
- **Abstract**: FE-GAN rectifies a single fisheye image by predicting a pixel-level distortion flow (U-Net-like U + STN warp W), trained adversarially on unpaired fisheye / distortion-free sets with cross-rotation and intra-image warping (radial + smoothness) consistency.
- **Notes**: Not on arXiv; no open-access copy (Unpaywall/OpenAlex: closed). PDF obtained from IEEE Xplore via the Stanford network.
---

### [zhu2017cyclegan] CycleGAN — Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks
- **Authors**: Zhu, Park, Isola, Efros · **Venue**: ICCV 2017 · **Status**: discovered
- **Topics**: unpaired image translation · **Notes**: baseline in Table 1 / Fig. 2; the FEGAN code base derives from pytorch-CycleGAN-and-pix2pix.
---

### [liu2017unit] UNIT — Unsupervised Image-to-Image Translation Networks
- **Authors**: Liu, Breuel, Kautz · **Venue**: NIPS 2017 · **Status**: discovered
- **Topics**: unpaired image translation · **Notes**: baseline in Table 1 / Fig. 2.
---

### [fu2019gcgan] GcGAN — Geometry-Consistent GANs for One-Sided Unsupervised Domain Mapping
- **Authors**: Fu, Gong, Wang, Batmanghelich, Zhang, Tao · **Venue**: CVPR 2019 · **Status**: discovered
- **Topics**: geometric consistency, unpaired translation · **Notes**: inspiration for FE-GAN's cross-rotation consistency (Eq. 3).
---

### [jaderberg2015stn] Spatial Transformer Networks
- **Venue**: NIPS 2015 · **Status**: discovered · **Notes**: basis of FE-GAN's differentiable warping module W.
---

### [yin2018fisheyerecnet] FishEyeRecNet
- **Venue**: ECCV 2018 · **Status**: discovered · **Notes**: supervised; predicts distortion parameters guided by semantic segmentation.
---

### [xue2019learning] Learning to Calibrate Straight Lines for Fisheye Image Rectification
- **Venue**: CVPR 2019 · **Status**: discovered · **Notes**: supervised with line-segment annotations.
---

### [li2019blind] Blind Geometric Distortion Correction on Images Through Deep Learning
- **Venue**: CVPR 2019 · **Status**: discovered · **Notes**: regresses pixel-wise flow with ground-truth flow supervision.
---

### [aleman2014automatic] Automatic Lens Distortion Correction Using One-Parameter Division Models
- **Venue**: IPOL 2014 · **Status**: discovered · **Notes**: classical (Hough / plumb-line, one-parameter division model) baseline in Fig. 2.
---

### [basu1995alternative] Alternative Models for Fish-eye Lenses
- **Venue**: Pattern Recognition Letters 1995 · **Status**: discovered · **Notes**: the fisheye transform used (with random parameters) to synthesise the paper's 1000 fisheye images per dataset.
---

### [fitzgibbon2001simultaneous] Division model for lens distortion
- **Venue**: CVPR 2001 · **Status**: discovered
---

### [feng2023simfir] SimFIR
- **Venue**: ICCV 2023 · **Status**: discovered · **Notes**: later self-supervised fisheye rectification work (follow-up direction; not used on screen).
---
