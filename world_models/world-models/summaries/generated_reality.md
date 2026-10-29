# Generated Reality: Human-centric World Simulation using Interactive Video Generation with Hand and Camera Control

## 1. Bibliographic info
- **Authors:** Linxi Xie (1,2,*,†), Lisong C. Sun (1,*), Ashley Neall (1,3,*,†), Tong Wu (1), Shengqu Cai (1), Gordon Wetzstein (1)
  - 1 Stanford University, 2 NYU Shanghai, 3 UNC Chapel Hill. * equal contribution; † work done as visiting researcher at Stanford.
- **Venue:** CVPR 2026 (the project page labels it "CVPR 2026 Findings Paper")
- **arXiv:** 2602.18422 (v2, 23 Aug 2026), cs.CV
- **Project:** https://codeysun.github.io/generated-reality/ (the page links to the code)

## 2. Core idea (lay version)
You put on a VR headset, and instead of loading a 3D game world, an AI video generator paints the world live around you, following where you look and exactly how your fingers move, so you can pet a dog, swing a sword or open a door that never existed.

## 3. Method
- **Problem:** Current video world models (Genie, Oasis, etc.) take only coarse controls like keyboard, mouse or text. XR needs control from the user's actual tracked body, especially the fingers, for dexterous hand-object interaction.
- **Inputs tracked by a commercial headset (Meta Quest 3):** a 6-DoF head/camera pose and joint-level hand pose for both hands. Hands use the UmeTrack model: wrist translation (xyz) + wrist quaternion + 20 finger-joint angles per hand ("hand pose parameters", HPP).
- **First systematic study of hand-pose conditioning in video DiTs.** HPP is injected four ways: token concatenation (PlayerOne-style), token addition (ReCamMaster-style), AdaLN (PEVA-style) and cross-attention. These are compared against video-type conditions: a binary hand mask (InterDyn) and a 2D skeleton video (ControlNet-style). **Token addition is best** of the HPP options. Cross-attention and AdaLN do worse than no conditioning at all.
- **Hybrid 2D-3D hand conditioning (key contribution):** a 2D hand-skeleton video rendered from the user's viewpoint is VAE-encoded and channel-concatenated with the video latents, and 3D HPP features from a 1D-conv "motion encoder" are added to the patch tokens. The 2D part gives spatial grounding. The 3D part removes depth ambiguity and self-occlusion, and keeps working when the hand is only partly in frame near the edges.
- **Joint head + hand control:** the head pose becomes per-frame Plücker ray embeddings (camera encoder). The final token is patchify([z_video, z_skeleton]) + E_motion(HPP) + E_cam(Plücker), fed to the DiT.
- **Iterative encoder training:** training both encoders jointly from scratch was unstable, because head motion and hand motion are ambiguous. So they are trained separately first (camera encoder initialized from Wan "FUN"), then fine-tuned together. LoRA stages are chained: skeleton, then hybrid, then joint.
- **Backbone and data:** Wan2.2 14B I2V (MoE with high-noise and low-noise experts), LoRA rank 32, 480x480, trained on HOT3D egocentric hand-object mocap clips (5,824 five-second clips). Also validated on GigaHands (8x larger).
- **Real-time system (distillation):** a bidirectional Wan2.2 **5B** teacher with head and hand conditioning is distilled into a **causal 5B autoregressive student via Self-Forcing** (DMD-style). It generates 12-frame chunks with per-frame hand and head conditioning, and supports both I2V and T2V.
- **XR integration:** a Unity app on Quest 3 streams tracked poses to a remote H100 server, where a circular buffer holds the latest poses. Generated chunks stream back to the headset, giving a closed loop: move your hands and head, and the world responds.
- **Generalization:** training uses only studio HOT3D footage, but the foundation model's prior lets text prompts (LLM-augmented) produce open-world scenes: alien planets, dungeons, dogs, cats, golf and more.

## 4. Key numbers and claims
- **11 FPS real-time with 1.4 s latency** on a single (remotely streamed) H100. Latency comes mostly from generating and decoding a 12-frame chunk. The conditioning adds only **0.002 s**.
- **User study** (11 subjects, ages 22-30; tasks: "push the green button", "open the jar", "turn the steering wheel"; 8 s per task; baseline = head-pose only + text prompt):
  - Task success **71.2% (ours) vs 3.0% (baseline)**
  - Perceived control (7-pt Likert) **4.21 vs 1.74**
- **Table 1 (HOT3D, Wan2.2 14B):** Hybrid gives MPJPE 12.23 mm, MPVPE 9.10, 2D L2 11.50 px, the best on all hand metrics. The unconditioned baseline scores 17.86 / 12.29 / 67.50, and skeleton-only scores 12.38 / 9.25 / 11.72. The estimator lower bound is 9.42 / 7.74 / 9.08. FVD falls from 601.6 to 383.7.
- **Table 2 (joint control):** JointCtrl has the best video quality (PSNR 18.60, LPIPS 0.280, SSIM 0.617). Camera error is 0.25 m / 2.79 deg, close to CameraCtrl's 0.23 / 2.77. Hand error is 12.81 mm, close to hand-only's 12.23. The hand-only model has 13.4 deg rotation error.
- **GigaHands (Wan2.2 5B):** Hybrid beats 2D-only by 10% on MPJPE, 11% on MPVPE and **34% on 2D error**.
- **Limitations:** resolution, latency, no stereo, drift after a few seconds, DMD mode-seeking and over-saturation, and weak long-range hand-object-object dependencies.

## 5. Figures (page index 0-based)
| Fig | Page | Caption summary | Teaser-worthy (1-2 s)? |
|---|---|---|---|
| 1 | 0 | Concept: a prompt (image or text "waving flag... alien landscape") plus head/hand pose trajectories and 3D hand renders produce "Generated Reality" filmstrips (car driving; astronaut hand waving a flag on an alien planet) over a collage of other scenes (dog, winter forest, dungeon). | **YES, best single hero shot.** Saved as `data/figs/gr_fig1.png` |
| 2 | 2 | Pipeline: a person in a VR headset; camera 6-DoF goes to the camera encoder; hand HPP (20 joint angles + wrist translation/quaternion) goes to the motion encoder; 2D skeleton frames go to the 3D VAE; all are summed (+) with noisy video tokens into DiT blocks; the output is an egocentric frame of hands pouring a mug. | Yes for a method beat. Use the site's clean version `data/figs/gr_pipeline.png` (4378x1460 RGBA, **transparent background**, white text on dark design, so composite it on black/dark). |
| 3 | 4 | Qualitative hand-conditioning comparison (Baseline / HPP / Video / Hybrid). GT hand in red, prediction in orange, overlap in green. Hybrid holds up at frame edges. | Weak (small, technical) |
| 4 | 6 | Joint hand-camera control: GT / Camera-Ctrl / Hand-Ctrl / Joint-Ctrl rows. Hand-only reaches for the wrong object. | Weak-moderate |
| 5 | 7 | User study: generated screenshots for the 3 tasks (button, jar, steering wheel), and a photo of a user in the Quest headset whose hands appear in the generated video in real time. | Moderate. The "human in headset" photo is a good concept beat, but the image is small. |
| 6 | 7 | Bar charts: task accuracy and perceived control (ours vs baseline). | Moderate as a stat beat (better re-drawn as animated bars: 3% vs 71%, 1.74 vs 4.21) |
| 7 | 12 | User-study qualitative frames, baseline vs ours, for 3 tasks. | Weak |
| 8 | 13 | Diverse T2V generations with 2D hand-skeleton overlay, 8 rows x 5 frames: astronaut flag, petting golden retriever, dungeon sword fight, A-frame cottage, golf swing, pushing door into winter forest, highway driving, cat wand toy. | **YES, best "variety" montage.** Saved as `data/figs/gr_fig8.png`. Individual rows crop well as 1 s strips. |
| 9 | 14 | GigaHands qualitative comparison (scenes 1-2). | No |
| 10 | 15 | GigaHands qualitative comparison (scenes 3-4). | No |

Tables 1-2 are on page 5, and Tables 3-4 on page 11.

Saved crops (200 dpi, checked visually):
- `/home/brianchc/project_animations/world_models/data/figs/gr_fig1.png` (Fig. 1 teaser, without caption and affiliation line)
- `/home/brianchc/project_animations/world_models/data/figs/gr_fig8.png` (Fig. 8 full grid with bolded prompt lines, without figure caption)
- `/home/brianchc/project_animations/world_models/data/figs/gr_pipeline.png` (project-site pipeline, transparent background)

## 6. Project-page demo videos
All the T2V clips sit in a carousel with a toggle that overlays the 2D hand-pose conditioning (3D hand and head conditioning is not shown). Every clip is egocentric and driven by live-tracked hands and head. The descriptions below come from the page's prompts. Where I describe the actual motion, I inferred it from the prompt and Fig. 8.
- **teaser_final2.mp4:** the hero video at the top of the page, captioned "Generated Reality is a concept that turns human-tracked data into immersive experiences." The HTML gives no further description. It is most likely a montage of tracked head and hands driving the generated scenes (compare the Fig. 1 collage).
- **boat.mp4:** on a sailboat at sea, the viewer's gloved hands pull a halyard rope hand-over-hand, and the white mainsail rises and fills with wind. Midday sun, deep blue ocean, seabirds.
- **boxing.mp4:** a boxing ring under spotlights. The viewer wears red gloves and faces a huge upright cat opponent that is also in boxing gloves, staring into the camera.
- **cat.mp4:** a sunlit living room. The viewer flicks a cat-wand toy with a pom-pom, and a domestic cat crouches, pounces and swipes at it (Fig. 8, last row).
- **clap.mp4:** a wooden A-frame glass cottage on a Mexican peninsula at sunset with turquoise water, palms and flowers. The prompt is purely scenic; the file name and the Fig. 8 row suggest the user's hands come together to clap in the foreground.
- **dog.mp4:** a sunny park. The viewer bends down and pets a friendly golden retriever, which pants and wags (Fig. 8, row 2).
- **golf.mp4:** a lush golf course (the prompt mentions Kenyan golfers, a buggy and a caddy). The viewer's hands grip and swing a golf club (Fig. 8, row 5).
- **push2.mp4:** a plain hallway with a wooden door. The viewer pushes it open onto a snowy magical pine forest with a glowing lamppost (Fig. 8, row 6).
- **space.mp4:** astronaut gloves grip and wave a flag bearing an alien sigil, on a colorful alien planet where alien humanoids watch in the distance (Fig. 1 / Fig. 8, row 1).
- **torch.mp4:** golden-hour Hawaii beach. The viewer holds a flickering tiki torch while a beach party, a ukulele circle and planted torches sit in the background, with natural handheld head motion.
- Also on the page but not in the list: **fight.mp4** (torchlit dungeon: torch in the left hand, longsword in the right, an armored soldier charges; Fig. 8 row 3); ablation clips `hybrid/*-clip-0000XX.mp4` (GT vs Base / HPP / 2D / Ours) and `joint/*` (GT vs Camera-Ctrl / Hand-Ctrl / Ours); **user_study_video.mp4** (the live Quest 3 system at 11 FPS with tracked hands and head, streaming to the headset); plus `static/images/user_study_results.png`.

## 7. BibTeX
```bibtex
@inproceedings{xie2026generatedreality,
  author    = {Xie, Linxi and Sun, Lisong C. and Neall, Ashley and Wu, Tong and Cai, Shengqu and Wetzstein, Gordon},
  title     = {Generated Reality: Human-centric World Simulation using Interactive Video Generation with Hand and Camera Control},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026},
  note      = {arXiv:2602.18422}
}
```

## Teaser notes (10 s)
Suggested beats: (1) a person in a headset moves their hands; (2) a skeleton overlay turns into generated hands in an alien world waving a flag; (3) a fast montage of Fig. 8 rows or clips (dog, sword, door into snow, cat); (4) a stat card: "11 FPS, 71% vs 3% task success".
