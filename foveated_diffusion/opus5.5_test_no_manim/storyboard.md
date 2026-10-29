# Foveated Diffusion — 1-minute teaser

**Current version: v1** (`animation_teaser_v1.py`)

Only reference: `opus5.5_test/foveated-diffusion/pdfs/chao2026foveated.pdf`.
Text is pulled into `ref/paper.txt`. Figures are pulled from the PDF's embedded
images into `ref/imgs/x<xref>.png`, using PyMuPDF.

No Manim. Frames are drawn with pycairo. Text and math use matplotlib mathtext:
CMU Serif for body text, the Computer Modern `cm` fontset for math. Frames are
piped to `/usr/bin/ffmpeg` (libx264).

## Render
```
conda activate hsplat
python animation_teaser_v1.py --quality l    # media/videos/480p15/FoveatedDiffusionTeaser_v1.mp4 (854x480)
python animation_teaser_v1.py --quality h    # media/videos/1080p60/FoveatedDiffusionTeaser_v1.mp4
python animation_teaser_v1.py --quality h --stills 10,20   # PNG stills -> media/stills/
```

## Style
- Black background; white text that appears instantly (no text fades)
- HR token `#448AFF`; LR token `#00C853`; neutral gray `#9E9E9E`; LoRA badge `#FF9800`; gaze dot `#FF5252`
- Tokens are rounded squares with corner radius = 10% of the side. In a grid, each token is 80% of its cell (10% margin on each side). In a 1D sequence, the gap is 20% of the side.
- Design canvas is 1920×1080 and is scaled to the output resolution

## Token example (used throughout)
- 16×16 HR grid (hw = 256) and an 8×8 grid of 2×2 blocks
- Fovea = the 12 central blocks, where (block offset)² ≤ 2.5 → m = 48 HR tokens
- Periphery = 52 LR tokens → **L = 48 + 52 = 100** (39% of 256)
- Sequence order: block raster order. A fovea block contributes 4 HR tokens; a periphery block contributes 1 LR token.
- Circle drawn on images: radius 0.25 × image side, centered

## Timeline (60.0 s)
| # | Scene | Start–End | Content |
|---|---|---|---|
| 1 | Title | 0–4 | Title, subtitle, authors, "Stanford University". Token strip at y=820: 5 blue tokens (36 px) in the center and 7 green tokens (60 px) on each side, popping in from the center outward. |
| 2 | Motivation | 4–11.5 | Header "Human vision is foveated". Left: lion `x993` (660 px) with blur that grows with distance from a red gaze dot moving on a Lissajous path, plus a dashed fovea circle. Right (t>2.4): "attention cost ∝ N²". N grows 4→16 with an N×N gray pair grid. Bottom (t>5): "Why generate every pixel at full resolution?" |
| 3 | Tokenization | 11.5–23 | Legend. Image (560 px) → "VAE encode + patchify" → 16×16 blue grid (0.8–1.7 s wave). Mask circle drawn at 2.5 s. From 4.0–6.6 s, periphery blocks merge 4 blue → 1 green, in an outward wave. Grid-label sequence: "Standard DiT: h×w = 256" → "1 low-res token per 2×2 block" → "L = m + (hw−m)/4 = 48 + 52 = 100". At 7.4–9.0 s, tokens fly into a 1D row at y=880 (side 14). Captions: "L = 100 tokens instead of 256" and a gray RoPE note. |
| 4 | Generation | 23–35 | Row moves to y=250 and turns to noise. DiT box "(LoRA, mixed-resolution RoPE)" with loop arrows. 8 discrete steps, t = 1.00→0.00, over 1.0–4.2 s; token colors go from gray noise to blue/green. From 4.8–6.3 s, Split sends HR tokens to z_high (16×16 at 240 px) and LR tokens to z_low (8×8). D decoders give x_high (fovea only; periphery at 18% opacity) and x_low (half size). Up, ⊙M′, ⊙(1−M′), then + gives the output (440 px): half-res periphery plus sharp fovea, with a dashed circle. Eq. 7 appears below at 8.8 s. |
| 5 | Training | 35–43 | Training image x (`x990`) → Down → x_low. Each goes through E to give z_high (16×16) and z_low (8×8). From 3.0–4.4 s, Merge(M) moves fovea HR tokens and periphery LR tokens into z0_fov (unused tokens dim to 35%). Then "add noise" → DiT box with orange LoRA badge → v_θ. Flow-matching loss shows at 5.3 s. Gray caption about cross-resolution consistency. |
| 6 | Image results | 43–47 | Columns: Full high-res / Naïve mixed-res / **Ours** (blue). Rows: horse and dog, cropped from Fig. 5 (`x1018`), 370 px each. |
| 7 | Video results | 47–51 | Fox clip from Fig. 8 (`x1050`, rows 0–2), cycling 5 frames at 2.5 fps. Three 560×322 panels. "3.5× faster", with "at 38% of the tokens, VBench quality on par". |
| 8 | Numbers | 51–56.5 | Speedup bars at 25% tokens: Image >2×, Video >4× (Fig. 6). Table 1 line: 30% tokens → 1.98×, HPSv2.1 0.279 vs. 0.280. 2AFC user study bars (Fig. 7): ours 47.3% vs. full high-res 52.7%; ours 87.4% vs. naïve 12.6%. Application strip (Fig. 1c): saliency-guided, multiple foveae, VR gaming, robotics, self-driving, aerial. |
| 9 | End | 56.5–60 | Title, "Spend tokens where the eye looks.", "Up to 2× faster images · 4× faster videos", project URL (blue). |

## Asset map (from the PDF)
- `x993` — Fig. 2 generation output lion. Used as x_high, x_low (downsampled by 2×), and the output composite.
- `x990` — Fig. 2 training image x
- `x1018` — Fig. 5 grid (3 rows × 6 columns of 512 px cells). Column order: Full, Naïve, Ours, Full, Naïve, Ours.
- `x1050` — Fig. 8 video strips (6 rows × 5 frames of 640×368). Row order: Full, Naïve, Ours (fox), then Full, Naïve, Ours (chef).
- `x820–x833` — Fig. 1(c) application images
