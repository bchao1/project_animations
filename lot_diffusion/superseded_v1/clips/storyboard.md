# Storyboard — LoT Diffusion 10-second thread clips

**Current version: v1** — `animation_lot_clips_v1.py`, 19 scenes, each exactly 10.0 s (±1 frame), 1920×1080 60 fps.
Render `./render.sh l|h [Scene ...]` → `media/final/<res>/LoT_<Scene>.mp4`. Thread use: `../social/threads.md`.

Common: black bg, CMU Serif; headline y 3.42 (size 30–34); captions y −3.45…−3.62; tag "Level-of-Token Diffusion"
(14, dim, italic) bottom-right at y −3.87. First frame is always informative (X autoplay / thumbnail).
Token colours by log2(area): 1×1 #448AFF, 1×2 #29B6F6, 2×2 #00C853, 2×4 #9CCC65, 4×4 #FFCA28, 4×8 #FFA726, 8×8 #B388FF.
Website layout videos (showcase `*_lay`, 3D-scene grids) keep the site's own colours.

| clip | content (sources) | on-screen numbers |
|---|---|---|
| C01_EqualCompute | dog (site images.json 042, semantic mask): image + uniform 64×64 mean-colour tokens → markers background vs fur → LayoutMorph to the real layout → level colours + legend | 4,096 → 1,351 tokens, 3.0× fewer |
| C02_Lineage | LOD mesh (Clark 1976), VRS tiles (2014), quadtree over Fig. 10 alps (Shapiro 1993) → shrink → boxes → layout → living room (site bbox) | — |
| C03_AnySize | real VRS video layout show22 slice 0 (45×80): uniform → layout; zoom 7×10 window on a 4×1 token; R_i formula; extents | 3,600 → 1,562, 2.30× |
| C04_Sources | paper Fig. 4 deer row: box / mask / texture / depth layout → generation, columns in sequence | 1.73/1.54, 2.18/1.81, 1.36/1.28, 3.11/2.29 (compression/speed-up) |
| C05_AsymFlow | 8×8 noisy latent with a 24-token layout → 2×2 patch → A_eᵀ → token → DiT → h_(2,2) → û_A → dense velocity; Eq. 9 | L = 24 of 64 (illustrative) |
| C06_ShapeRoPE | 8×8 lattice, 2×4 token at (2,1): top-left → centre (2.5, 2.5); 1×1 = pretrained RoPE; s(e) bars (1, 2, 3, −1); zero-init MLP g_φ; h⁰ equation | — |
| C07_Budget | Fig. 10 deer, B = 1,024 / 1,792 / 2,560 / 3,328 / 4,096, slider, crossfades | budgets |
| C08_Numbers | scatter speed-up vs HPSv3 (Tab. 1 @ 2.935×) and vs VBench imaging (Tab. 2 @ 3×) | 2.04×/9.55, 3.53×/0.565 etc. |
| R01_PaintGUI | recording 3 (manual GUI, 57 s) at 5.8×, cover-cropped; 3 numbered steps | — |
| R02_Agent | recording 0 canvas crop at 6× (frozen at 4.3 s, before the page scrolls) → paint0 LayoutMorph → result | 4,096 → 1,546, 2.6× |
| R03_Scene3D | toy-train demo: generated (big) + detail map + temporal grid, 0.6× speed | 50,832 / 75,600 (1.49×) |
| R04_Games | platformer (grid video), monkey warrior (show14), desert coupe (show26, own LayoutVideo) | 2.18×, 2.02×, 2.16× |
| R05_Driving | show03 day / show04 rainy night: layout video + generated, class legend | 1.66×, 1.97× |
| R06_Robotics | show02 (mask) / show10 (boxes) | 1.50×, 1.50× |
| R07_DepthOfField | show30/31/39 generated + own LayoutVideo from exact leaves; blur-radius formula | 2.13×, 2.72×, 2.46× |
| R08_MoveTheBox | Fig. 19 rows a / c / i: 1×1 box (detected yellow bbox, snapped to 2×2 lattice) moves A → B, LoT image changes, dense FLUX.2 9B right; Tab. 4 | 1,636/2.50×, 1,222/3.35×, 1,519/2.70×; 44.9% → 66.8% |
| R09_Typography | site typography 82 / 40 / 86: grid → LoT image; dense 40 inset | tokens, s vs s, 4.56× / 2.26× / 3.17× |
| R10_Boxes | site bbox examples reading-corner / coastal / still-life: box overlay → layout → image | 2,454 / 1,962 / 1,953 of 6,144 |
| R11_VideoBaselines | site comparison 0 (raft): full-res / ToMe-SD / Foveated / LoT, 2×2, 0.6× | 1,108 / 402 / 407 / 386 s, 2.44× |
