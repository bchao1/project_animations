# CameraCtrl: Enabling Camera Control for Video Diffusion Models

## 1. Bibliographic info
- **Authors:** Hao He (CUHK, Shanghai AI Lab), Yinghao Xu (Stanford), Yuwei Guo (CUHK, Shanghai AI Lab), Gordon Wetzstein (Stanford), Bo Dai (Shanghai AI Lab), Hongsheng Li (CUHK, corresponding), Ceyuan Yang (Shanghai AI Lab, corresponding)
- **Venue:** ICLR 2025 (conference paper)
- **arXiv:** 2404.02101 (v2, 13 Mar 2025)
- **Project page:** https://hehao13.github.io/projects-CameraCtrl/

## 2. Core idea (lay version)
You draw the path a virtual camera should fly (pan, dolly forward, orbit), and CameraCtrl makes an
existing AI video generator film its scene along exactly that path, so the model acts less like a
clip generator and more like a world you can steer a camera through.

## 3. Method
- **Problem:** text/image-to-video diffusion models (AnimateDiff, SVD) have no precise way to take a
  user camera trajectory. Earlier methods (AnimateDiff MotionLoRA: 8 fixed moves; MotionCtrl: raw
  pose numbers) generalize poorly or ignore small camera motions.
- **Camera representation = Plücker embeddings.** For every pixel (u,v), a 6-D ray
  p = (o x d, d), where o is the camera center and d is the normalized world-space ray direction
  RK^-1[u,v,1]^T + t. Each frame becomes a 6 x h x w "ray map", and a trajectory becomes an
  n x 6 x h x w tensor. This gives per-pixel geometry and uniform value ranges, unlike raw K/[R|t]
  values or Euler angles.
- **Camera encoder (Phi_c):** a T2I-Adapter-style encoder (pixel-unshuffle, conv, 4 ResNet scales)
  with a **temporal attention block after each conv block**. It takes *only* the Plücker embeddings
  (not the noisy latents, as ControlNet would), which avoids leaking training-set appearance and helps
  it generalize across domains. It outputs multi-scale features that match the U-Net resolutions.
- **Fusion into temporal attention:** camera features are added element-wise to the latent features,
  passed through a learnable linear layer, and fed into the first temporal attention layer of each
  temporal block, in both the U-Net encoder and decoder. Temporal layers suit camera motion, which is
  a global, frame-to-frame change.
- **Plug-and-play:** the base video model stays frozen (only the encoder and linear fusion layers
  are trained), so one module works with AnimateDiff, personalized checkpoints (RealisticVision,
  ToonYou), SVD (I2V), and alongside SparseCtrl (RGB/sketch conditioning).
- **Data study:** they compared Objaverse (synthetic, most diverse cameras, large appearance gap),
  MVImageNet (real, mostly horizontal orbits), and RealEstate10K (real, varied trajectories).
  **RealEstate10K wins** because its appearance is close to the base model's data and its cameras
  are diverse. Adding ACID gave no gain.
- **Metrics:** TransErr and RotErr, computed by running COLMAP on the generated videos and comparing
  against the input poses, plus FVD, CLIPSIM, Frame Consistency, and a new Object Dynamic Degree
  (ODD).
- **Training:** ~65K RealEstate10K clips, AdamW, 50K steps, batch 32. T2V uses AnimateDiff V3,
  16 frames at 256x384, 16 A100s, ~25 h. I2V uses SVD, 14 frames at 320x576, 32 A100s, ~40 h.
- **Extras:** camera-motion intensity can be scaled via the translation spacing. Pans and zooms can
  also be produced purely by editing the intrinsics (principal point, focal length).

## 4. Key numbers / claims
- **T2V (AnimateDiff):** TransErr **12.98** vs MotionCtrl 14.02; RotErr **1.29** vs 1.58.
  CLIPSIM 0.301 and FC 0.941 are on par with or better than the base model.
- **I2V (SVD):** TransErr **9.02** vs 10.21; RotErr **1.18** vs 1.41; FVD **360.3** vs 386.2
  (MotionCtrl) and 371.2 (plain SVD).
- **User study** (50 users): I2V **73.1%** vs 26.9% (MotionCtrl). T2V basic moves 43.6% vs 37.0%
  (MotionCtrl) vs 19.4% (AnimateDiff). T2V complex trajectories 72.4% vs 27.6%.
- **Ablations:** Plücker beats raw values, Euler angles, and direction+origin (RotErr 1.29 vs
  1.43-1.57). T2I-Adapter+temporal beats ControlNet (FVD 222 vs 284-296). Temporal-attention
  injection beats spatial injection. RealEstate10K beats MVImageNet and Objaverse (Objaverse cannot
  even be measured).
- COLMAP lower bound on real RealEstate10K clips: TransErr 6.93, RotErr 1.02.
- **Failure mode:** large rotations are under-shot (a requested 150 degree pan yields ~90 degrees),
  because RealEstate10K has few big rotations.

## 5. Figures
All pages below are 0-based indices into the 32-page PDF.

| Fig | Page | Caption summary | Teaser (1-2 s)? |
|---|---|---|---|
| 1 | 1 | Overview: trajectory plots (camera frustums with a red arrow) next to generated frame strips: hot-air balloon (AnimateDiff), B&W man at the sea (RealisticVision), surfing lion (SVD I2V), house (SparseCtrl). Left side shows "Camera Control Model + Video Diffusion". | **YES, best.** Trajectory plus frames reads instantly. **SAVED** |
| 2 | 4 | Framework: (a) Plücker embedding goes into the camera encoder, then into the temporal attention of a frozen U-Net; (b) injection detail: add, then linear, then temporal attention. | Moderate. Good for a "how" beat but dense. Just the Plücker ray-map piece could be re-animated. |
| 3 | 7 | Qualitative MotionCtrl vs CameraCtrl, T2V and I2V ("Rocky coastline with crashing waves"). MotionCtrl rotates the scene or misses the small left move. | Weak/moderate. A comparison needs time to read. |
| 4 | 9 | Applications: turtle (AnimateDiff), neon cyberpunk city (RealisticVision), anime girl (ToonYou), boy under sun with sketch (SparseCtrl), misty lake (SVD), each with its trajectory. | **YES.** Striking style range under one control module. **SAVED** |
| 5 | 17 | Dataset samples: Objaverse (objects on white), MVImageNet (object orbits), RealEstate10K (house interiors). | Weak. Useful only if the video discusses data. |
| 6 | 19 | Schematic of camera representations: repeated K/R/t, Euler angles, per-pixel Plücker map. | Moderate. Nice for explaining "a ray per pixel", but better redrawn. |
| 7 | 20 | Representation ablation frames (raw matrix vs direction+origin vs Plücker), staircase interior. | Weak. |
| 8 | 21 | AnimateDiff vs CameraCtrl on basic pans (pyramid, dog on grass). | Weak. |
| 9 | 22 | MotionCtrl vs CameraCtrl T2V (horse, snowy pines, villa, apple). | Weak. |
| 10 | 23 | MotionCtrl vs CameraCtrl I2V (coastline, turtle, flower pot, pyramid). | Weak. |
| 11 | 24 | RealEstate10K-domain results (bedrooms, kitchens, living rooms). | Weak/moderate. Realistic but bland interiors. |
| 12 | 25 | Same prompt (horse on grassland) with different trajectories gives the same scene seen along different paths. | Moderate/strong. Directly shows "same world, different camera". |
| 13 | 26 | Natural objects and scenes (sunflower, squirrel, elven palace, snowy forest, lake, balloon). | Moderate. Pretty montage. |
| 14 | 27 | Stylized scenes with RealisticVision (storm coastline, mecha, neon street, rabbit). | Moderate. |
| 15 | 28 | Cartoon characters with ToonYou. | Weak/moderate. |
| 16 | 28 | SVD I2V results (cheese house explosion, Van Gogh Seine, aquarium, fireworks). | Moderate. |
| 17 | 29 | CameraCtrl + SparseCtrl (RGB/sketch conditions; autumn landscape, spiderman, cyberpunk zoom). | Weak/moderate. |
| 18 | 30 | Camera-motion intensity (1x vs 4x translation spacing, pan-down and zoom-in). | Weak. |
| 19 | 30 | Camera motion via intrinsics only (principal-point shift gives pans; focal length gives zoom). | Weak. |
| 20 | 31 | Failure cases: 100 and 150 degree rotations are under-rotated. | Weak (not for a teaser). |

**Saved crops (200 dpi):**
- `/home/brianchc/project_animations/world_models/data/figs/cameractrl_fig1.png` (Fig. 1, page 1):
  4 rows of trajectory plot plus 5 frames, with the "Camera Control Model + Video Diffusion" label.
- `/home/brianchc/project_animations/world_models/data/figs/cameractrl_fig4.png` (Fig. 4, page 9):
  5 rows (turtle, cyberpunk city, anime girl, boy/sketch, misty lake), each with a trajectory plot.

## 6. BibTeX
```bibtex
@inproceedings{he2025cameractrl,
  title     = {CameraCtrl: Enabling Camera Control for Video Diffusion Models},
  author    = {He, Hao and Xu, Yinghao and Guo, Yuwei and Wetzstein, Gordon and Dai, Bo and Li, Hongsheng and Yang, Ceyuan},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2025},
  note      = {arXiv:2404.02101}
}
```
