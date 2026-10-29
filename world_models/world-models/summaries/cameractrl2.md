# CameraCtrl II: Dynamic Scene Exploration via Camera-controlled Video Diffusion Models

## 1. Bibliographic info
- **Authors:** Hao He (CUHK, ByteDance Seed), Ceyuan Yang (ByteDance Seed, corresponding),
  Shanchuan Lin (ByteDance Seed), Yinghao Xu (Stanford), Meng Wei (ByteDance), Liangke Gui
  (ByteDance Seed), Qi Zhao (ByteDance Seed), Gordon Wetzstein (Stanford), Lu Jiang (ByteDance Seed),
  Hongsheng Li (CUHK)
- **Venue:** ICCV 2025
- **arXiv:** 2503.10592 (v1, 13 Mar 2025)
- **Project page:** https://hehao13.github.io/Projects-CameraCtrl-II/

## 2. Core idea (lay version)
Starting from one photo, you steer a camera clip by clip through a living, moving scene (turn left
around pedestrians, then back onto the sidewalk), and the model keeps generating a consistent world
along your path. Camera control stops being a single shot and becomes open-ended exploration.

## 3. Method
- **Two problems with earlier camera control** (CameraCtrl, MotionCtrl, AC3D): (1) adding camera
  control kills scene dynamics, since training on static RealEstate10K/DL3DV makes the model produce
  frozen worlds; (2) only short single clips are possible (25 frames for CameraCtrl, 49 for AC3D),
  so there is no way to keep exploring.
- **RealCam dataset (dynamic videos with camera labels):** real in-the-wild videos (indoor, aerial,
  street). TMO motion segmentation masks the movers, and RAFT flow on the static background measures
  camera motion to keep only videos that move enough. Per-frame cameras come from VGGSfM.
- **Metric scale calibration:** SfM scale is arbitrary. Depth Pro metric depth is aligned to SfM
  depth via a Huber loss plus RANSAC on keyframes, and the translation is rescaled, t -> s*t, so all
  scenes share a metric scale.
- **Trajectory distribution balancing:** each trajectory is segmented at keypoints (angle between
  fitted lines above a threshold), then classified by main direction x main turn (N x M bins).
  Over-represented types such as "forward" are pruned by importance weight.
- **Lightweight camera injection:** the Plücker ray maps (6 x h x w per latent frame, one pose every
  4 frames to match the causal VAE) go through a *single new patchify layer* whose output is added
  to the visual tokens *before the first DiT block only*. This avoids CameraCtrl-style multi-layer
  encoders that over-constrain motion.
- **Joint training plus camera CFG:** the full ~3B-parameter internal text-to-video DiT is fine-tuned
  on 4:1 camera-labeled:unlabeled video (unlabeled clips get an all-zero Plücker input). This keeps
  generality and enables **classifier-free guidance on the camera** (w_text=7.5, w_cam=8.0).
- **Clip-wise autoregressive extension (exploration):** clean tokens from the last frames of the
  previous clip are concatenated with the noised tokens of the new clip, plus a binary mask channel.
  Loss is computed only on the new clip. All poses are expressed relative to the first frame of the
  first clip (global reference) to avoid drift. This is trained jointly with single-clip generation.
- **Distillation for interactivity:** progressive distillation cuts 96 NFEs (32 uncond + 32 text-CFG
  + 32 cam-CFG) to 16. APT adversarial post-training reaches 1 step, trading some accuracy.
- **Training:** stage 1 is single-clip at 192x320, 100K steps, batch 640, 64 H100s. Stage 2 is
  384x640 plus video extension, 50K steps, batch 512, 128 H100s. The Euler sampler uses 32 steps.

## 4. Key numbers / claims
- **I2V vs CameraCtrl / MotionCtrl** (Tab. 2): FVD **73.11** vs 199.53 / 221.23. Motion strength
  **698.51** vs 133.37 / 102.21, about 5x more dynamic than CameraCtrl. TransErr **0.1527** vs
  0.2812 / 0.3221. RotErr **1.58** vs 2.81 / 2.78. Geometric consistency (VGGSfM success rate)
  **88.70** vs 52.12 / 57.87. Appearance consistency across clips **0.889** vs 0.778 / 0.743.
- **T2V vs AC3D:** FVD 641 vs 987; motion 574 vs 162; TransErr 0.189 vs 0.298; RotErr 1.66 vs 2.98;
  geometric consistency 85.0 vs 69.2.
- **Speed** (4 s, 12 fps video, 4 H800s): **13.83 s -> 2.61 s** with progressive distillation
  (small accuracy loss), and **0.59 s** with 1-step APT (TransErr 0.25, RotErr 2.56, so degraded).
- **Ablations:** training only on static RealEstate10K collapses motion (129 vs 307). Without scale
  calibration, errors rise (TransErr 0.212 vs 0.183). Without distribution balancing, RotErr is
  4.56 vs 1.74. Multi-layer injection cuts motion to 247. Without joint training, motion and accuracy
  both drop. Per-clip reference frames or a noised-condition extension lower appearance consistency.
- Generated videos are 3D-consistent enough for FLARE to reconstruct clean point clouds.
- **Limitation:** the model has no physical awareness. A trajectory through a fence makes the fence
  "dissolve" instead of the camera stopping (Fig. 7).

## 5. Figures
All pages below are 0-based indices into the 13-page PDF.

| Fig | Page | Caption summary | Teaser (1-2 s)? |
|---|---|---|---|
| 1 | 0 | Interactive exploration: starting street image, then 4 chained clips. The user "thinks" ("I need to turn left to avoid these people", "return to my original path", ...) and draws a new trajectory each time. Clips stay consistent and pedestrians keep moving. | **YES, best.** It is the world-model story in one image: user steers, world continues. **SAVED** |
| 2 | 2 | Dataset curation pipeline: motion segmentation, VGGSfM, metric depth, scale calibration, distribution balancing, RealCam. Small single-column diagram. | Weak/moderate. Could be a quick icon beat. |
| 3 | 3 | Architecture: (a) Plücker embedding goes through a camera patchify layer, is added to the visual patchify output, then enters the pretrained DiT; (b) video extension with clean previous-clip tokens + noised current tokens + mask, and loss only on the current clip. | Moderate. Clean diagram, but too dense for 1-2 s. Panel (b) idea is worth re-animating. |
| 4 | 6 | Qualitative comparison, "Cars and a bus move along a city street": CameraCtrl ignores the upward motion and is static; AC3D misses the forward motion and the bus; CameraCtrl II follows the path with moving cars. | Moderate. A before/after is possible but needs labels. |
| 5 | 8 | Diverse scenes with trajectories: Minecraft-style village, B&W foggy 19th-century London street, abandoned hospital, fantasy forest trail, anime palace (I2V, first frame outlined in red). | **YES.** Vivid, game-like "explore any world" montage. **SAVED** |
| 6 | 8 | 3D reconstruction: generated frames (living room, dining room, courtyard, burger close-up) next to FLARE point clouds. | Strong/moderate. "Generated video becomes 3D" is a nice 1 s beat. |
| 7 | 9 | Failure case: camera path goes through a fence and the fence structure breaks down. | Weak (limitation only). |

**Saved crops (200 dpi):**
- `/home/brianchc/project_animations/world_models/data/figs/cameractrl2_fig1.png` (Fig. 1, page 0):
  thought bubbles, user trajectories, and 4 rows x 5 frames of chained street-scene clips with
  dashed continuation arrows.
- `/home/brianchc/project_animations/world_models/data/figs/cameractrl2_fig5.png` (Fig. 5, page 8):
  5 rows x 6 frames (Minecraft village, foggy London, hospital, forest trail, anime palace), each
  with its trajectory plot.

## 6. BibTeX
```bibtex
@inproceedings{he2025cameractrl2,
  title     = {CameraCtrl II: Dynamic Scene Exploration via Camera-controlled Video Diffusion Models},
  author    = {He, Hao and Yang, Ceyuan and Lin, Shanchuan and Xu, Yinghao and Wei, Meng and Gui, Liangke and Zhao, Qi and Wetzstein, Gordon and Jiang, Lu and Li, Hongsheng},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2025},
  note      = {arXiv:2503.10592}
}
```
