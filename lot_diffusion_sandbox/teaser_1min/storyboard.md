# Storyboard — Level-of-Token Diffusion, 1-minute teaser (3b1b style)

**Current version: v4** — `animation_lot_1min_v4.py` (480p15 total 64.3 s: S1 27.9 + S2 17.5 + S3 18.9)
Render: `./render.sh l` (480p15) · `./render.sh h` (1080p60) → `media/final/LoTDiffusion_1min_vN_<res>.mp4`.
Final cut: `media/final/LoTDiffusion_1min_v4_1080p60.mp4` (3 scenes, 64.4 s, 1920×1080, 60 fps, h264;
1080p scenes 28.0 / 17.6 / 18.8 s).
Env: `imp_tome` + `LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7`. Target 55–65 s (hard max 75 s).
Style: `../STYLE_NOTES.md` (helpers copied from the foveated 1-min v4).

Paper: *Level-of-Token Diffusion*, anonymous ICLR 2027 submission 7014; supplementary site
iclr-submission7014.github.io/supplementary.

**Takeaway.** Where detail matters is often known before generation. LoT turns that into a layout of tokens of
different sizes — fine where detail matters, coarse elsewhere — and a fine-tuned pretrained DiT still predicts
full-resolution flow for every patch → up to 2.04× faster images, 3.53× faster video, close to full-resolution quality.

**Running example (all three scenes).** The site's painted demo “A girl riding a corgi” (LoT-FLUX.2, 1024²):
painted detail map `../data/img/paint0_canvas.png` → layout → result `../data/img/paint0_result.png`.
The layout is our own manim build from the site's token grid (`../web/assets/b72bc…webp`, 896 px = 64 × 14 px):
`build_assets()` reads the smoothed detail field D off its grey background and applies the paper's quadtree rule
(A.3, Eq. 23) with fitted thresholds t₈, t₄, t₂ = 0.02, 0.44, 0.74 (not shown on screen). It reproduces the site's
leaf counts exactly — 1,312 · 140 · 79 · 15 tokens of extent 1, 2, 4, 8 = **1,546 of 4,096** (asserted) — and its grid
lines (checked by overlay). Merge passes shown in S1: 4,096 → 2,008 → 1,591 → 1,546 (computed from that layout).

## Assets (`data/generated/`, built on import)
- `corgi_layout.json` — the 1,546 leaves (row, col, extent) on the 64 × 64 grid.
- `fig7a_{0..4}.png` — Fig. 7a panels cropped from the rendered PDF page 9 (8× zoom). The embedded rasters differ
  from what the page shows (the DDiT / Foveated panels are overlaid images), so we crop the rendered page.
- `xt_{0..4}.png` — x_t = (1 − t)·x₀ + t·ε of the result image, t = 1, .75, .5, .25, 0 (seed 7), labelled “illustration”.
- Video: `../data/clips/show03_{lay,gen}` (site showcase item 3, 81 frames, played at 15 fps via a pixel-array updater).

## Numbers on screen
| Number | Source |
|---|---|
| 64 × 64 = 4,096 tokens (1024² FLUX.2), 1,546 tokens, 1,312 / 140 / 79 / 15 | site painted demo (`token_count`, `leaf_counts`, `uniform_fine_token_count`) |
| 2.65× fewer tokens, 38% of the tokens | 4,096 / 1,546 = 2.649, 1,546 / 4,096 = 37.7% |
| 4,096 → 2,008 → 1,591 → 1,546; “16 tokens on the face / of sky” | computed from the reproduced layout |
| up to 2.04× faster images, 3.53× faster video, “at ≈ 3× fewer tokens” | Tab. 1 (2.935× → 2.040×), Tab. 2 (3× → 3.530×), abstract/§1 |
| Fig. 7a: budget 1,269 tokens; speedups 1.00 / 2.10 / 1.90 / 1.63 / 2.29× | paper Fig. 7a (p. 9) |
| “noisy low-res regions” (ToMe-SD, DDiT), “distorted at token borders” (Foveated) | Fig. 7 caption wording |
| HPSv3 at 2.94× fewer tokens: full 10.43 · ours 9.55 · best baseline 7.84 (Foveated) | Tab. 1 |
| driving clip: 1.66× fewer tokens | site video showcase item 3 (semantic-mask layout, LoT-Wan2.1 14B) |

## Style
As `../STYLE_NOTES.md`: black, CMU Serif (+ Latin Modern Math), Pango markup only, text instant, captions y −3.5 /
28 / held ≥ len/16 + 0.8 s, no headers or part cards. Extent colours 1×1 `#448AFF`, 2×2 `#00C853`, 4×4 `#1B7F3B`,
8×8 `#0E5A2A`; highlights amber `#FFCA28` (face, grass token, image numbers) / purple `#E040FB` (sky, video numbers).
Token side = e·c − 0.2c (10% margin), corner radius 10% of side; 1 × 1 tokens are plain squares (radius sub-pixel).

## Beats (v4; durations from the 480p15 render)

### S1_Insight (27.9 s) — hook + key insight
1. **Title (0–3.9 s).** “Level-of-Token Diffusion” (72 bold, y 2.75) + subtitle “adaptive, efficient generation from
   multiresolution token layouts” (32, y 1.95). Emblem: corgi layout (3.8 at (0, −0.75)) fades in as 64 coarse
   8 × 8 tokens (1.0 s), splits 8 → 4 → 2 → 1 (0.6 s each, outward from the subject), 0.8 s hold, fade 0.5 s.
2. **Hook (3.9–13.5 s).** Left: result image (4.6 at (−3.4, 0.35)), “the image”. Right: 64 × 64 grid of 1 × 1 tokens
   (4.6 at (3.4, 0.35)), “what the model computes”, count “**4,096** tokens, every denoising step” (y −2.3).
   Caption “A diffusion transformer gives every patch the same token.”
   Amber 4 × 4 box on the girl's face (rows 11–14, cols 30–33) and purple 4 × 4 box on sky (rows 4–7, cols 4–7) in both
   panels; key under the image “□ 16 tokens on the face  □ 16 tokens of sky”.
   Caption “The girl's face and the empty sky cost exactly the same.”
3. **Insight (13.5–27.9 s).** Image → painted detail map (cross-fade 0.8 s), label “detail map, painted before
   generation”, legend “detail ■0 ■0.3 ■0.6 ■1” (y −2.3). Caption “But where detail matters is often known in advance.”
   Caption “Level-of-Token layout: fine tokens where detail matters, coarse elsewhere.” Three merge passes (1.1 s each,
   outward from the subject), count 4,096 → 2,008 → 1,591 → 1,546; the face box keeps its 16 tokens, the sky box grows
   to the 8 × 8 token it ends up in. Boxes fade; extent legend “■ 1 × 1: 1,312 · ■ 2 × 2: 140 · ■ 4 × 4: 79 · ■ 8 × 8: 15”
   (y −2.78). Caption “1,546 tokens instead of 4,096: 2.65× fewer.” 1.5 s hold; everything but the layout fades.

### S2_Method (17.5 s) — one picture: short sequence in, full-resolution flow out
Layout slides from (3.4, 0.35) to (−4.65, 0.2) and shrinks 4.6 → 3.2 (0.9 s), label “LoT layout”.
1. Arrow “1,546 / tokens” → `DiT` + `LoRA` (1.6 × 2.2 at (−1.35, 0.2)), dim “pretrained / FLUX.2 · Wan2.1” below.
   Caption “A pretrained DiT, fine-tuned once, reads only these tokens.”
2. Amber outline on one 4 × 4 grass token (bottom-left); its copy flies into the DiT; arrow → “head 4 × 4” box
   (amber stroke, (0.3, 0.2)) → fan of 16 fine blue tokens (≈ (1.72, 0.2)), dim “flow for all / 16 positions”.
   Caption “A size-specific head returns full-resolution flow for each patch.”
3. Output (3.2 at (4.65, 0.2)) “full-resolution output”, dim “illustration of denoising”; the fan lands as an amber
   4 × 4 square at the grass patch's position; x_t frames t = 1 → 0 (0.45 s each).
   Caption “Every step runs on 38% of the tokens; the image stays full resolution.” 1.5 s hold, fade 0.6 s.

### S3_Payoff (18.9 s) — results + end card
1. **Speed (0–5.9 s).** Site clip show03 as layout (y 1.15) over video (y −1.0), 1.95 tall at x −3.5, side labels
   “layout” / “video”, header “LoT-Wan2.1 14B · semantic-mask layout”, dim “1.66× fewer tokens · supplementary website”.
   Right: “2.04×” (amber 84 bold) faster images, “3.53×” (purple) faster video, dim “FLUX.2 klein 4B · Wan2.1 14B /
   at ≈ 3× fewer tokens (paper Tab. 1, 2)”. Caption “Fewer tokens per step, faster generation.” 5 s of playback.
2. **Quality (5.9–13.9 s).** Fig. 7a row: 5 panels 2.3 tall, pitch 2.52, y 0.55; names above, “k× speedup” below,
   red notes under ToMe-SD / DDiT / Foveated; blue frame + bold blue label on “Ours (LoT)”; dim header
   “image generation, same budget: 1,269 tokens (paper Fig. 7a)”; HPSv3 line (y −2.35).
   Caption “Same token budget: the baselines break, LoT keeps the detail.” 2 s hold past the caption.
3. **End card (13.9–18.9 s).** Final layout emblem (1.9 at (0, 2.3)), title (64 bold, y 0.75), blue tagline
   “fine tokens where detail matters · up to 2.04× faster images · 3.53× faster video”, “Anonymous ICLR 2027 submission 7014”,
   dim “iclr-submission7014.github.io/supplementary”; 5 s hold.

## Version history
- v1 — first cut, 63.5 s. QA: face/sky boxes single cells (invisible), extent legend crowding the caption,
  S2 “pretrained FLUX.2 / Wan2.1 + LoRA” overlapping the layout, S2 output not fading (5 stacked frames),
  S1 closing caption repeating the count.
- v2 — 4 × 4-token boxes (face stays 16 tokens, sky box grows to its 8 × 8 token), panels at y 0.35, legend y −2.78,
  base-model label on two lines, one x_t layer at a time, “layout / video” clip labels, red notes 20 pt. 64.3 s.
- v3 — face / sky labels moved out of the image into a key under it (amber was unreadable on the sky). S1 only rendered.
- v4 — face box moved 2 columns right onto the face (rows 11–14, cols 30–33; all 1 × 1). 64.3 s.
