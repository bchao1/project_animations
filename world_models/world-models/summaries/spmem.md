# Video World Models with Long-term Spatial Memory (SPMem)

## 1. Bibliographic info
- **Title:** Video World Models with Long-term Spatial Memory (confirmed from paper text, PDF p.0)
- **Authors:** Tong Wu\*¹, Shuai Yang\*²˒⁴, Ryan Po¹, Yinghao Xu¹, Ziwei Liu⁵, Dahua Lin³˒⁴, Gordon Wetzstein¹ (\* equal contribution)
  - ¹ Stanford University, ² Shanghai Jiao Tong University, ³ The Chinese University of Hong Kong, ⁴ Shanghai AI Laboratory, ⁵ S-Lab, Nanyang Technological University
  - (The project page lists the affiliation numbers differently, SJTU=3 and CUHK=2. The institutions are the same.)
- **Venue:** NeurIPS 2025 (poster). The local text is arXiv v1 (5 Jun 2025), which gives no venue. The venue was checked against neurips.cc/virtual/2025/poster/118886, the NeurIPS 2025 proceedings, and the GitHub README "[NeurIPS 2025]".
- **arXiv:** 2506.05284 (cs.CV)
- **Project:** https://spmem.github.io/ | Code: https://github.com/spmem/spmem
- Acknowledgment: supported by Google. Ryan Po is supported by a Stanford Graduate Fellowship.

## 2. Core idea (lay version)
AI "world models" imagine a video world as you move through it, but they only look back a few frames, so if you turn around and come back, the street you just saw has changed. This paper gives the model a lasting 3D memory of the scene (a point-cloud map plus a few snapshots), so a place looks the same when you return to it.

## 3. Method
- **Problem:** Autoregressive video world models condition on only a few recent frames because attention cost grows quadratically. Anything outside that window is forgotten, so revisits are inconsistent.
- **Three memory types, based on how human memory works:**
  - **Working memory (short-term):** the most recent context frames (the last 5 frames/latents), concatenated with the target tokens along the frame axis. This keeps motion continuous.
  - **Spatial memory (long-term):** a global **static point cloud** of the generated world, built up autoregressively. It is rendered along the requested camera trajectory, with empty regions left black, to act as geometric guidance.
  - **Episodic memory (long-term, sparse):** a set of historical keyframes. A frame is added when a mask-based visibility check shows that the newly revealed (unseen) area is larger than a threshold. These frames keep visual detail and identities that the sparse point cloud loses.
- **Keeping memory static:** new frames are converted to point maps (CUT3R online at inference, Mega-SaM for building the dataset) and merged with **TSDF fusion**. Dynamic objects give inconsistent depth across frames, get low confidence, and are suppressed. Only static structure goes into memory.
- **Conditioning architecture:** the backbone is CogVideoX-5B-I2V (initialized from DiffusionAsShader). The point-cloud render is encoded by the 3D VAE and processed by a **ControlNet-style condition DiT**: a copy of the first 18 DiT blocks, whose outputs are added to the main DiT through zero-initialized linear layers. Episodic reference frames are VAE-encoded and fed in through new **historical cross-attention** layers (video tokens are queries, reference tokens are keys/values). Action text is encoded by T5.
- **Autoregressive loop:** generate a chunk, predict its point maps, align them, TSDF-fuse them into global memory, render memory along the next camera path, generate the next chunk. CUT3R's recurrent state is carried across steps so all chunks share one coordinate frame.
- **Dataset:** 90K samples built from MiraData. Each 97-frame clip is split into 49 source frames and 48 target frames (sharing one frame). Mega-SaM gives poses and depth, TSDF fusion of the source frames gives the static point cloud, which is rendered along the target trajectory. Qwen writes action captions for the target frames.
- **Training:** 49 frames at 480x720, 6,000 iterations, learning rate 2e-5, batch size 8, 8x A100.

## 4. Key numbers / claims
- **View recall consistency** (the same pose visited on a forward trajectory and then its reverse; 500 held-out MiraData clips):
  - Ours: PSNR **19.10**, SSIM **0.647**, LPIPS **0.307**
  - Best baseline (Wan2.1-Inpainting): 12.16 / 0.451 / 0.588. DaS: 12.01 / 0.451 / 0.587. TrajectoryCrafter: 11.71 / 0.438 / 0.600.
  - That is about +7 dB PSNR, and LPIPS is roughly halved.
- **User study** (20 experts, 14 cases, average rank 1-4, higher is better): Cam-Acc 3.63, Stat-Cons 3.38, Dyn-Plaus 3.40. The best baseline (DaS) scored 2.57 / 2.44 / 2.70.
- **VBench:** best aesthetic quality (0.584), temporal flickering (0.758), motion smoothness (0.989) and subject consistency (0.936). Wan2.1 is slightly better on imaging quality and background consistency; the authors attribute this to Wan producing more static scenes.
- **Ablation (Table 3):** removing working memory or episodic memory hurts every VBench metric (for example, flickering drops to 0.674 without working memory and 0.726 without episodic memory).
- **Stated limitations:** TSDF fusion fails for abrupt, large camera jumps (Fig. 6, Spiderman swinging), which leaves memory sparse. The method does not address drift (quality loss over time). PSNR is still "far from perfect."

## 5. Figures
| Page (0-based) | Fig | Caption summary | 1-2 s teaser visual? |
|---|---|---|---|
| 0 | Fig 1 | Teaser. Global static point cloud with camera path, then episodic, working and spatial memory tiles, then dynamic video generation. Two rows: hooded figure in an autumn courtyard, horseback rider on a city street. | **YES, the best.** One image shows the whole idea: 3D map + memory + generated frames. **Saved as sm_fig1.png** |
| 3 | Fig 2 | System overview. Condition DiT (point-cloud render) + denoising DiT with historical cross-attention, recent context, T5 action text, point-cloud update loop. | Medium. It is a busy block diagram; useful only as a brief "how it works" beat. |
| 5 | Fig 3 | Dataset pipeline. Mega-SaM, TSDF fusion, static vs dynamic point cloud, render along target path, Qwen action caption. | Weak-medium. Pipeline diagram (same as the site's data.jpg). |
| 6 | Fig 4 | Qualitative comparison with Wan2.1, TrajectoryCrafter, DaS and Ours. Rows: camera accuracy (look-up), **view recall consistency (first visit vs revisit of the angel monument)**, action accuracy ("crouching, crossing a log bridge"). | **YES.** The first-time vs recall rows show the "revisiting a place should look the same" point directly, and the baselines visibly fail. **Saved as sm_fig4.png** |
| 7 | Fig 5 | Memory ablation (Witcher-like scene). Static point map, w/o working memory, w/o episodic memory, full model, GT. Red boxes mark failures, yellow boxes mark successes. | Medium. Good for "each memory matters," but small panels. |
| 8 | Fig 6 | Failure case. Spiderman swinging: an abrupt trajectory leads to a sparse memory and ghosting. | No (limitations). |
| 9 | Fig S1 | Autoregressive point cloud fusion. Two filmstrips (night city street walk; horseback rider in a valley) above the fused long-term point clouds with the colored camera trajectory. | **Strong runner-up.** Great for a "memory building up" beat. Also saved as sm_figS1.png |

Saved files (200 dpi, cropped to the figure only, checked visually):
- `/home/brianchc/project_animations/world_models/data/figs/sm_fig1.png` (1118x445): Fig 1 teaser
- `/home/brianchc/project_animations/world_models/data/figs/sm_fig4.png` (1118x885): Fig 4 qualitative comparison
- `/home/brianchc/project_animations/world_models/data/figs/sm_figS1.png` (1100x545): Fig S1 point cloud fusion (bonus)
- `/home/brianchc/project_animations/world_models/data/figs/sm_teaser.jpg` (3368x1306): the site's high-res version of Fig 1. **Use this rather than the PDF crop.**
- `/home/brianchc/project_animations/world_models/data/figs/sm_data.jpg` (3324x1554): the site's high-res version of Fig 3 (data pipeline)

## 6. Project-page videos (https://spmem.github.io/static/videos/...)
Page structure: title, abstract, **wm-demo** (hero video, no caption), "TL;DR" with **demo_page**, "Data Construction" (data.jpg), "Long-term Video Generation" (long_1-8 in a 2-column grid), "Static Geometry-grounded Video Generation" (carousel of short_1-12), BibTeX.

- **wm-demo.mp4:** hero/overview video below the abstract. No caption. Likely the main summary reel.
- **demo_page.mp4:** in the TL;DR section: "Our video generation framework relies on long-term spatial memory, short-term working memory, and sparse episodic memory." Likely a walkthrough of the three memories.
- **Long-term Video Generation** (long autoregressive rollouts):
  - long_1: futuristic car driving through a sunny, palm-lined modern city
  - long_2: two hands rising from the ground holding glowing orbs in a bleak mountain landscape
  - long_3: forest journey from a sunny day into snow, ending at a wooden cabin
  - long_4: black sports car on a scenic mountain highway, passing other cars
  - long_5: horseback rider on a grassy mountain path under an overcast sky
  - long_6: first-person walk through a red-rock desert with moving shadows
  - long_7: aerial view of a beachside promenade with palms, shops, cars and pedestrians
  - long_8: horseback rider moving through varied urban scenes, from cobblestone streets to a modern city
- **Static Geometry-grounded Video Generation** (short clips, carousel):
  - short_1: game-like sunlit ruined street with a horse-drawn carriage and pedestrians in period clothing
  - short_2: overcast early-1900s cobblestone street with carriages and tram tracks
  - short_3: dense mossy forest with sunlight through the canopy, slow camera move
  - short_4: backpacker walking a sunlit path through green landscape
  - short_5: aerial view of a snowy forest with a vehicle on a winding road
  - short_6: grocery store fruit section with price tags
  - short_7: aerial view of a red car on a rural countryside road
  - short_8: river flowing over stones through a lush forest
  - short_9: first-person shooter view of an overgrown post-apocalyptic area with stone stairs
  - short_10: two boxers facing off in a gym ring
  - short_11: aerial view of a graffiti-covered building on a coastal hill by a beach
  - short_12: people ice skating at a sunny outdoor rink with mountains behind

## 7. BibTeX
Proceedings version (recommended):
```bibtex
@inproceedings{wu2025spmem,
  title     = {Video World Models with Long-term Spatial Memory},
  author    = {Wu, Tong and Yang, Shuai and Po, Ryan and Xu, Yinghao and Liu, Ziwei and Lin, Dahua and Wetzstein, Gordon},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2025},
  note      = {arXiv:2506.05284}
}
```
Project-page version:
```bibtex
@article{wu2025video,
  title   = {Video world models with long-term spatial memory},
  author  = {Wu, Tong and Yang, Shuai and Po, Ryan and Xu, Yinghao and Liu, Ziwei and Lin, Dahua and Wetzstein, Gordon},
  journal = {arXiv preprint arXiv:2506.05284},
  year    = {2025}
}
```

## Teaser notes (10 s)
Suggested beats:
1. A camera moves forward and then turns back. A baseline "forgets" the scene (use the Fig 4 recall row).
2. The point cloud builds up along the trajectory (Fig S1 / Fig 1 left).
3. Three memory tiles light up: working, spatial, episodic (Fig 1 middle).
4. The generated rider/courtyard video plays, and the revisited view matches.
5. Title card with the "+7 dB PSNR on revisits" stat.

Clips to use: long_8 or long_5 (horseback rider, matching Fig 1/S1), plus the wm-demo hero video.
