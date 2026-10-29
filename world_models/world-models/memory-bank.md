# Paper Memory Bank
Last updated: 2026-09-26

### [he2025cameractrl] CameraCtrl: Enabling Camera Control for Video Diffusion Models
- **Authors**: Hao He, Yinghao Xu, Yuwei Guo, Gordon Wetzstein, Bo Dai, Hongsheng Li, Ceyuan Yang
- **Venue**: ICLR 2025
- **URL**: https://arxiv.org/abs/2404.02101 · https://hehao13.github.io/projects-CameraCtrl/
- **Status**: summarized (summaries/cameractrl.md)
- **Topics**: camera control, video diffusion, world generation
- **Abstract**: Plug-in camera module for a frozen video diffusion model; camera paths become per-pixel Plücker ray maps fed to temporal attention.
- **Notes**: Seg 1 of the 10 s clip (with CameraCtrl II). Fig. 1 row used as paper figure.
---

### [he2025cameractrl2] CameraCtrl II: Dynamic Scene Exploration via Camera-controlled Video Diffusion Models
- **Authors**: Hao He, Ceyuan Yang, Shanchuan Lin, Yinghao Xu, Meng Wei, Liangke Gui, Qi Zhao, Gordon Wetzstein, Lu Jiang, Hongsheng Li
- **Venue**: ICCV 2025
- **URL**: https://arxiv.org/abs/2503.10592 · https://hehao13.github.io/Projects-CameraCtrl-II/
- **Status**: summarized (summaries/cameractrl2.md)
- **Topics**: camera control, dynamic scene exploration, world models
- **Abstract**: Clip-by-clip camera-steered exploration of a dynamic, persistent scene; RealCam dataset with metric-scale cameras, camera CFG, distillation.
- **Notes**: Demo clip `citywalk1.mp4` (generated panel) is the seg 1 video. Fig. 1 rows used as paper figure.
---

### [xie2026generatedreality] Generated Reality: Human-centric World Simulation using Interactive Video Generation with Hand and Camera Control
- **Authors**: Linxi Xie, Lisong C. Sun, Ashley Neall, Tong Wu, Shengqu Cai, Gordon Wetzstein
- **Venue**: CVPR 2026
- **URL**: https://arxiv.org/abs/2602.18422 · https://codeysun.github.io/generated-reality/
- **Status**: summarized (summaries/generated_reality.md)
- **Topics**: interactive world generation, hand-object interaction, XR
- **Abstract**: Video world model conditioned on tracked head + hand pose (2D skeleton + 3D joints), distilled to real time (11 FPS) for VR.
- **Notes**: Seg 2 video = teaser_final2.mp4 22.25–24.6 s (petting the dog, headset inset). Fig. 1 left panel used.
---

### [wang2026bullettime] BulletTime: Decoupled Control of Time and Camera Pose for Video Generation
- **Authors**: Yiming Wang, Qihang Zhang, Shengqu Cai, Tong Wu, Jan Ackermann, Zhengfei Kuang, Yang Zheng, Frano Rajič, Siyu Tang, Gordon Wetzstein
- **Venue**: CVPR 2026
- **URL**: https://arxiv.org/abs/2512.05076 · https://github.com/19reborn/BulletTime · https://19reborn.github.io/Bullet4D/
- **Status**: summarized (summaries/bullettime.md)
- **Topics**: space-time control, 4D video generation, bullet time
- **Abstract**: Separate world-time and camera-pose controls (Time-RoPE, 4D-RoPE, Time/Camera-AdaLN) on CogVideoX, trained on synthetic 4D-controlled data.
- **Notes**: Seg 3 video = intro_public_1.mp4 13.25–16.25 s. Fig. 1 + Fig. 5 row used.
---

### [wu2025spmem] Video World Models with Long-term Spatial Memory
- **Authors**: Tong Wu, Shuai Yang, Ryan Po, Yinghao Xu, Ziwei Liu, Dahua Lin, Gordon Wetzstein
- **Venue**: NeurIPS 2025
- **URL**: https://arxiv.org/abs/2506.05284 · https://spmem.github.io/
- **Status**: summarized (summaries/spmem.md)
- **Topics**: world models, spatial memory, long-horizon consistency
- **Abstract**: Working memory + static point-cloud spatial memory + episodic keyframes keep revisited places consistent (~7 dB PSNR gain on revisits).
- **Notes**: Seg 4 video = demo_page.mp4 5.6–8.8 s (memory diagram + generated rider). Teaser figure (hi-res site jpg) used.
---
