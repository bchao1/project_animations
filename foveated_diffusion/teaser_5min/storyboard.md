# Storyboard — Foveated Diffusion 5-minute teaser (3b1b style)

**Current version: v2** — `animation_foveated_diffusion_5min_v2.py` (480p15 total 298.4 s ≈ 4 min 58 s)
Render: `./render.sh l` (480p15) · `./render.sh h` (1080p60). Env: `imp_tome` + `LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7`.
Final cut: `media/final/FoveatedDiffusion_5min_v2_1080p60.mp4` (10 scenes concatenated, 298.7 s ≈ 4 min 59 s, 1920×1080, 60 fps, h264). Target 4:30–5:30.

Condensed from `../opus5.5_test/` v4 (11 scenes, ≈ 9 min 20 s). Source paper: Chao, Yariv, Xiao, Wetzstein,
*Foveated Diffusion: Efficient Spatially Adaptive Image and Video Generation*, arXiv 2603.23491.
Images in `data/` are crops of the paper's figures; `data/generated/` is built by `build_assets()` on import.

## Version history
- v1 — first teaser cut, 10 scenes, 295.1 s at 480p15.
- v2 — S01 face/door boxes stay until their caption is read (no stale caption over the recolour);
  S02 1.5 s hold before the gaze moves + 1 s after; S06 denoise steps 0.4 → 0.55 s and a 1.2 s hold on the
  decoded x^high / x^low; S08 user-study bars shortened (x0 −2.8, W 5.4) with p-value notes to the right,
  so the 50% line no longer crosses “p = 0.48: indistinguishable”.

## Condensing plan (long cut → teaser)
| Long cut | s | Teaser | v2 s | What changes |
|---|---|---|---|---|
| S00 Title | 8 | S00_Title | 5.6 | shorter fovea wander (5 steps) |
| S01 TokenCost | 62 | S01_TokenCost | 36.1 | card 1.6 s; face/door beat in one caption; attention scan 2 moves not 4; resolution + video in one caption |
| S02 Vision | 51 | S02_Vision | 26.7 | no part card; one gaze move; foveated-rendering eq built faster, one caption; question card 2.2 s |
| S03 Tokenization + S04 RoPE | 44 + 56 | S03_Tokenization | 35.7 | RoPE reduced to one caption during the flatten-to-sequence beat |
| S05 Naive | 35 | S04_Naive | 16.3 | all 6 images at once, 2 captions |
| S06 Training | 79 | S05_Training | 39.5 | faster build, fewer captions, shorter t-slider, details card dropped (LoRA rank 32 + base models folded into one caption) |
| S07 Generation | 57 | S06_Generation | 31.7 | denoise 8 × 0.55 s, split+decode one caption, blend one caption |
| S08 Masks | 59 | S07_Masks | 42.7 | Fig. 9 grid at once; saliency + applications in 2 captions (YOLO dropped); spline path shorter |
| S09 Results | 64 | S08_Results | 37.1 | Table 1 → 4-line HPSv2.1 summary beside the plot; qualitative grid dropped (already shown in S04/S07); user study + video kept, faster |
| S10 Outlook | 46 | S09_Outlook | 26.9 | seams one caption; three-level grid dropped; gaze-following field; summary + end card shorter |

Part cards only at the three acts: "Part 1 · The problem" (S01), "Part 2 · The method" (S03), "Part 3 · Results" (S08).
Caption hold: len/16 + 0.8 s (long cut: len/15 + 1 s).

## Global style (unchanged from the long cut)
| Property | Value |
|---|---|
| Font | CMU Serif; Latin Modern Math for × ⊙ − → ≈ ∼ ∈ 𝒩 ℒ 𝔼 ′ ≪ ≤ ‖ · ↓ |
| Math | Pango markup, no LaTeX |
| Background | black |
| HR / LR | `#448AFF` / `#00C853` |
| Other | dim `#8A8A8A`, ghost `#3A3A3A`, red `#FF5252`, amber `#FFCA28` (image), purple `#E040FB` (video), LoRA `#FF7043`, DiT fill `#1E2A33` |
| Header | top-left, y 3.62, size 24, dim, “k · title” |
| Caption | y −3.5, size 28 (26 on two centered lines), max width 13.2 |
| Text | instant (`self.add`) |

Common mask: 8×8 blocks of a 16×16 grid, r = 2.0 → m = 48 HR, 52 LR, L = 100 of 256 (39%).

## Scenes (v2, durations from the 480p15 render)
Layouts are the long cut's unless stated; only timing/captions changed.

### S00_Title (5.6 s)
Title (72, bold, y 2.75), subtitle (32, y 1.95), authors (22, dim, y −3.3). 16×16 `FoveaField` (size 3.8 at (0, −0.75))
fades in 1.5 s; fovea path (4,4)→(3.5,3)→(3,5)→(4.5,5.5)→(5,3.5)→(4,4), 0.5 s per step.

### S01_TokenCost (36.1 s) — card “Part 1 · The problem / Every token costs the same” (1.6 s), header “1 · …”
Dog image (4.6 at (−3.5, 0.1)) → 16×16 grid lines (1.0 s) → mean-colour tokens (1.2 s). Face (4,8) amber + door (2,1)
purple boxes with legend at x ≈ 1.0. Recolour to HR; “h × w = 16 × 16 = 256 tokens” (y 2.55); attention matrix
(4.0 at (3.4, −0.1)) + “256 × 256 = 65,536 query–key pairs”; query box/row scan (5,8)→(2,3)→(11,12).
Squares N / 4N (side 0.9 at (1.2, 0.9), 3.6 at (4.6, 0)). Captions: DiT grid · both tokens cost the same ·
self-attention every step · 2× res → 4× tokens, 16× attention, video × frames · “does the viewer see every token equally?”

### S02_Vision (26.7 s) — header only
Acuity plot (schematic, Axes 5.6×3.4 at (−3.6, 0.35)), fovea band, labels. Farmer (4.6 at (3.4, 0.25)), red dot,
gaze blur A → dashed circle → move to (0.80, 0.22) with blur B. Foveated-rendering equation (42, y 2.35) + 5-image
visual equation (1.9 each, 0.4 s per item). Card “Can a diffusion model spend its tokens the same way?” (2.2 s).

### S03_Tokenization (35.7 s) — card “Part 2 · The method / Foveated tokenization”
Grid 16×16 (5.0 at (−3.4, 0.1)), mask panel (2.4 at (3.6, 1.2)) + legend; periphery dim + outline; 2×2 → 1 merge (2.8 s).
Counting at x ≈ 3.3: m = 48 · 208/4 = 52 · boxed L = 100 · 100/256 ≈ 39%. Flatten to a 100-token row (y 2.75, 2.2 s) with
the **RoPE beat as one caption**: “Phase-aligned RoPE (Wu et al., 2025) keeps positions consistent across the two scales.”
Attention squares: 65,536 → 10,000 pairs, ≈ 15%.

### S04_Naive (16.3 s) — header “4 · Just plug it in?  Naïve mixed-resolution denoising”
Rows Full high-res (y 1.35) / Naïve (y −1.2), images 2.35 at x −1.7, 0.95, 3.6 (horse, guitar, farmer), all 6 fade in together;
red notes: duplicated horse · two guitars · two hat brims. 2 captions (failure; 2 × 2 scale mismatch).

### S05_Training (39.5 s) — header “5 · Foveated training: mixed-resolution-consistent targets”
Same layout as long-cut S06: x / Down(x) → E → z_high 16×16, z_low 8×8 → M panel → z_fov grid via Merge.
Flow matching: z_fov to (−4.3, 0.35), equation (y 2.55), t slider 0→1→0.55 (1.5 + 0.9 s), DiT + LoRA box, v_θ grid, loss (y −2.45).
Details card dropped; caption: “Only LoRA adapters (rank 32) are trained on a frozen DiT: FLUX.2 Klein 4B for images, Wan2.1 1.3B for video.”

### S06_Generation (31.7 s) — header “6 · Foveated generation”
Noise grid (2.6 at (−5.0, 0.6)), DiT box, 8 denoise steps × 0.55 s; split into z_high / z_low (1.6 at (0.9, 1.55|−1.25)),
decoders D (x 2.5) → x^high / x^low (x 4.35), 1.2 s hold. Blend equation (38, y 2.35) + visual equation (0.4 s per item).
Hero output 4.9 tall at (−2.0, 0.1) with blocky fovea outline; “48 high-res + 52 low-res → 100 tokens instead of 256”.

### S07_Masks (42.7 s) — header “7 · Foveation masks are just data”
Fig. 9 grid (Position / Size / Disjoint regions, Naïve|Ours, 9 pairs of 1.62) fades in at once (1.8 s), 2 captions.
Saliency (Random vs Saliency-guided, 3 rows of 1.55) + Applications 2×2 (2.1 each): VR gaming, robotics simulation,
driving simulation, aerial footage; 2 captions (DeepGaze; weapon/gripper/van). YOLO/bounding-box caption dropped.
Video path: 5.0×2.8 frame at (−2.3, 1.2), 3 control points + natural cubic spline (no radius rings), fovea circle travels
2.2 s and returns while 5 `vid_fox_ours` frames (1.44 tall, y −1.95) fade in.

### S08_Results (37.1 s) — card “Part 3 / Results” (1.4 s), header “8 · Results”
Speedup plot (Axes 5.4×4.2 at (−3.4, 0.2), Fig. 6 vertices): image 2.2×, video 4.2× at 25%.
Right column (x 3.5) replaces Table 1: “Image quality: HPSv2.1 ↑” · full high-res 0.280 · **ours 0.279 – 0.280** (HR) ·
naïve 0.268 – 0.275 (dim) · “at 26–43% of the tokens (1.6–2.1× faster)”.
Qualitative 2×6 grid dropped. User study: protocol strip (fixate → image A 1 s → fixate → image B 1 s → A or B?) at y 2.0;
bars x0 −2.8, W 5.4 at y 0.25/−0.8/−1.85: 47.3% (p = 0.48: indistinguishable), 87.4%, 90.8% (p < 0.0001), notes right of bars,
dashed 50% line. Video: fox rows Full / Naïve / Ours (5 frames 1.22 tall), VBench 0.941 / 0.907 / 0.945,
“3.5× faster at 38% of the tokens (Wan2.1 1.3B, 480p)”.

### S09_Outlook (26.9 s) — header “9 · Limitations and outlook”
Fig. 28 crops (image 4.0 at (−4.6, 0.3), video strip at (−1.5, 0.3)) + seams / cause / fix text (all at once), 1 caption.
Three-level grid dropped. Gaze-following `FoveaField` (4.4 at (0, 0.2)) + red dot, path (3.2,2.6)→(2.6,4.6)→(4.2,5.6)→(5.4,4.2)→(4.6,2.8)→(4,4),
0.55 s per step, label “tokens follow an eye tracker” (y −2.4). Summary: three ideas (1.6 s each) + “≈ 2× faster images · ≈ 4× faster
video · perceptually indistinguishable” (3 s); end card (title, arXiv 2603.23491 · bchao1.github.io/foveated-diffusion, authors, 3.5 s).
