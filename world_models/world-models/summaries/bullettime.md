# BulletTime: Decoupled Control of Time and Camera Pose for Video Generation

## 1. Bibliographic info

- **Authors:** Yiming Wang (1,2), Qihang Zhang (3,2)\*, Shengqu Cai (2)\*, Tong Wu (2)†, Jan Ackermann (2)†, Zhengfei Kuang (2)†, Yang Zheng (2)†, Frano Rajič (1)†, Siyu Tang (1), Gordon Wetzstein (2)
  - 1 ETH Zurich, 2 Stanford University, 3 CUHK. \*, † = equal contribution.
- **Venue:** CVPR 2026 (pp. 18319-18330 per README bibtex)
- **arXiv:** 2512.05076 (v2, 23 Aug 2026), cs.CV
- **Project page:** https://19reborn.github.io/Bullet4D/
- **Code:** https://github.com/19reborn/BulletTime
- **Checkpoint:** https://huggingface.co/19reborn/bulletTime_ckpt
- **Dataset:** https://huggingface.co/datasets/19reborn/bulletTime_dataset
- **Data simulator:** https://github.com/19reborn/BulletTime_dataSimulator
- **YouTube demo:** https://www.youtube.com/watch?v=HS5nUlmH4CE

## 2. Core idea (lay version)

Normal video generators tie "when things happen" to "where the camera is" (both move forward together frame by frame), so BulletTime gives the model two separate dials, one for world time and one for camera position. You can freeze or slow a moment and still fly the camera around it, like the bullet-time shot in *The Matrix*, starting from any single ordinary video.

## 3. Method

- **Problem framing:** in standard video diffusion, "video time" (the frame index) mixes together two different things: *world time* τ (how far the scene's dynamics have progressed) and *camera pose* c (the viewpoint). BulletTime treats (τ, c) as two separate, orthogonal conditioning signals that together form a controllable 4D coordinate.
- **Continuous world-time input:** each output frame i gets a real-valued world time τ_i. Changing the spacing τ_{i+1} - τ_i gives slow motion, speed-up, pausing (frozen time) or reversal.
- **Time-RoPE:** rotary positional encoding computed on continuous world time instead of integer frame indices. Attention logits depend on τ_i - τ_j, and it adds no parameters. Time differences are scaled by FPS, so under uniform time it reduces exactly to standard RoPE, which keeps the pretrained prior intact.
- **Time-AdaLN:** a 1D conv encodes per-frame timestamps down to the latent temporal resolution. MLPs then predict scale/shift that modulate LayerNorm features. This recovers frame-level timing that is lost to VAE/patchify temporal downsampling, and it beats cross-attention and channel-addition conditioning.
- **4D-RoPE:** Time-RoPE is fused with a camera-aware rotary transform (following "Cameras as Relative Positional Encoding", Li et al. 2025) into one unified 4D positional encoding inside attention.
- **Camera-AdaLN:** per-pixel Plücker ray embeddings pass through a 2D conv camera encoder, then MLPs predict scale/shift. This is a branch parallel to Time-AdaLN. (Per the README, the released code applies the time/camera AdaLN only before attention, which gives comparable results with about half the added parameters.)
- **Synthetic 4D-controlled dataset:** built in Blender with the PointOdyssey framework. It has about 2k scenes (80 environments: 60 HDR outdoor and 20 indoor; 100 mocap-animated characters). Each scene is rendered with 3 camera trajectories (multi-waypoint, orbit, static) × 3 timing patterns (linear plus 2 drawn from slow-mo, pause, random warp, spline warp), for about 20k videos with camera and world-time labels. Time and camera vary independently, which provides the supervision needed to disentangle them.
- **Base model and setup:** fine-tuned from CogVideoX-5B-T2V (README: CogVideoX1.5-5B). For video-to-video, source and target tokens are concatenated along the frame axis (ReCamMaster-style). Output is 81 frames at 384×640. Training is progressive (half resolution, then full), 40K iterations, batch size 64, AdamW, lr 2e-5.
- **Inference:** input is a single video plus a per-frame camera c2w JSON plus a per-frame world-time curve (.npy). The output is a new video for each (camera, timing) pair. The repo ships HTML tools for interactively designing camera paths and time curves.

## 4. Key numbers / claims

- **Synthetic benchmark (Tab. 1, 500 unseen videos):** PSNR / SSIM / LPIPS: **Ours 24.57 / 0.6905 / 0.1265** vs ReCamMaster\* 21.86 / 0.5852 / 0.1846 vs TrajectoryCrafter\* 17.72 / 0.4917 / 0.3431. Baselines get time control through RIFE time-remapping of the input. ReCamMaster was fine-tuned on BulletTime's data; its released checkpoint scores 19.67 PSNR.
- **Real-world (Tab. 2, 100 ViPE videos):** RotErr **1.47** vs 2.98 (ReCamMaster\*) vs 5.44 (TrajCrafter\*). TransErr **1.32** vs 1.85 vs 3.31. FVD **2292** vs 2325 vs 2399. KVD **139.1** vs 146.1 vs 150.2. BulletTime has the best temporal flicker, motion smoothness, and subject/background consistency. TrajectoryCrafter is slightly higher on aesthetic/imaging quality.
- **Disentanglement (Tab. 3):** background consistency with the camera fixed and time varied, mPSNR **28.29** vs 25.80 (ReCamMaster\*), mLPIPS 0.1119 vs 0.1527.
- **Ablation (Tab. 4, time-only):** Time-RoPE + AdaLN reaches **32.15 PSNR** vs RoPE + CrossAttention 23.86. Time-RoPE alone gets 30.45 and beats every learnable variant that uses standard RoPE.
- **Ablation (Tab. 5, 4D):** full model 23.45 PSNR. Without 4D-RoPE it drops to 21.98; without camera/time AdaLN it drops to 22.74.
- **Qualitative claims:** trained only on synthetic human-centric data, the model generalizes to real videos, animals (a husky), fights, pouring liquids (latte art), and unseen time curves. It needs no 4D reconstruction or post-processing.
- **Limitations:** relies on synthetic data. Generation is parallel (non-autoregressive), so it cannot produce very long videos. Hands can look wrong, and backgrounds unseen in the input come out low-detail.

## 5. Figures

| Page (0-based) | Figure | Caption summary | Teaser (1-2 s)? |
|---|---|---|---|
| 0 | Fig. 1 | Soccer player input video (coupled camera, uniform time; color-coded time ticks) leads to "4D-Controlled Video Generation" with 3 outputs: static camera + non-uniform time, Moving Camera-1, and Moving Camera-2, each with a 3D trajectory inset. | **YES, the best one.** It shows the whole concept in one grid. Saved as `bt_fig1.png`. |
| 3 | Fig. 2 | Method overview. A 4D Control box (RoPE_c/MLP_c, RoPE_t/MLP_t) feeds a DiT block (LayerNorm, Scale&Shift, Attention, FFN). Noisy latent plus conditional video, a 4D Controlled Dataset strip (4 clips, each with a camera inset and time ticks), and the diffusion loss. | Moderate. Good for a "how" beat, but too dense for 1-2 s. |
| 4 | Fig. 3 | Synthetic comparison: GT vs TrajectoryCrafter vs ReCamMaster vs Ours. | No. |
| 5 | Fig. 4 | Real-world qualitative comparison (Ours / ReCamMaster / TrajectoryCrafter; camera and time control) showing baseline artifacts. | No (possibly a quick comparison flash). |
| 6 | Fig. 5 | 4D control: husky, boxing, and latte pour. Segments are labeled "fixed camera, changing time" and "**BulletTime**: moving camera, fixed time" (with a bullet graphic), plus camera trajectory insets. | **YES, second best.** It uses the bullet-time branding directly on real clips. Saved as `bt_fig5.png`. |
| 6 | Fig. 6 | Time control generalization: 3 generations (frames 30-60) from the same input under different speed curves. | Moderate. The speed-curve plots could be animated. |
| 7 | Fig. 7 | Disentanglement: camera fixed, time-control A/B/C, frame 40. ReCamMaster drifts geometrically; Ours stays consistent. | No. |
| 12 | Fig. B.1 | Failure cases (hands, unseen background). | No. |
| 13 | Fig. C.1 | Dataset sample. Left: same timing, cameras A/B/C. Right: different timing A/B/C, same camera (synthetic character). | Moderate. It explains the decoupled data idea clearly, but the visuals are plain. |
| 13 | Fig. C.2 | Grid of diverse synthetic dataset samples. | No. |
| 14 | Fig. D.1 | ReCamMaster artifacts under time remapping, with vs without time control. | No. |

Saved crops (200 dpi):
- `/home/brianchc/project_animations/world_models/data/figs/bt_fig1.png`: Fig. 1 teaser (page 0, clip rect 54,230,562,478 pt)
- `/home/brianchc/project_animations/world_models/data/figs/bt_fig5.png`: Fig. 5 BulletTime results (page 6, clip rect 52,60,562,295 pt)

## 6. Project-page videos (https://19reborn.github.io/Bullet4D/static/videos/...)

- **intro_public_1.mp4**: hero video at the top of the page, autoplays. Caption: "TL;DR Time- and camera-controlled 4D video generation. Given a single input video where camera motion is entangled with uniform temporal sampling, our method synthesizes new videos that enable decoupled control over world time and camera pose." (Likely the animated version of Fig. 1.)
- **bullet_time_full.mp4**: directly below the hero. Caption: "Our method enables bullet-time effects (i.e., freely moving the camera while freezing or slowing scene dynamics) for a wide range of real-world videos." This is the best source for bullet-time footage.
- **method.mp4**: under the "Method Overview" heading. Caption: "Overview of our 4D-controllable Diffusion Transformer." This is an animated Fig. 2: time via RoPE_t plus AdaLN/MLP_t, camera via RoPE_c plus MLP_c, fused 4D positional encoding, trained on the synthetic 4D dataset.
- **diverse_cam_full.mp4**: in the Results section, heading "Diversity of Camera Motions" (played at 1.25x). Caption: "Input Videos from ReCamMaster." It shows the same inputs re-rendered along many different camera trajectories.
- **diverse_time_full.mp4**: in the Results section, heading "Diversity of Time Control" (1.25x). Caption: "Input video from ReCamMaster, and Wan2.2." It shows the same input under varied world-time curves (slow-mo, pause, speed changes).
- The page also has a JS gallery of generated/input toggle pairs under `static/assets/{wan,veo,...}/`. Inputs come from Wan2.2 (e.g., martial arts, latte), Google Veo3 (e.g., fence), and real-world ViPE clips. Some are labeled "Video stabilization".

## 7. BibTeX

```bibtex
@inproceedings{wang2026bullettime,
  title     = {BulletTime: Decoupled Control of Time and Camera Pose for Video Generation},
  author    = {Wang, Yiming and Zhang, Qihang and Cai, Shengqu and Wu, Tong and Ackermann, Jan and Kuang, Zhengfei and Zheng, Yang and Raji{\v{c}}, Frano and Tang, Siyu and Wetzstein, Gordon},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {18319--18330},
  year      = {2026},
  note      = {arXiv:2512.05076}
}
```
