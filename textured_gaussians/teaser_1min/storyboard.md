# Storyboard — Textured Gaussians, 1-minute teaser

**Current version: v4** — `animation_textured_gaussians_1min_v4.py`
Final cut: `media/final/TexturedGaussians_1min_v4_1080p60.mp4` (64.8 s, 1920 × 1080, 60 fps).
Render: `./render.sh l` (480p15 test) · `./render.sh h` (1080p60 final) → `media/final/TexturedGaussians_1min_vN_<res>.mp4`.
Target 55–70 s (hard max 75 s). Condensed from the 5-min teaser (`../animation_textured_gaussians_v3.py`, `../storyboard.md`).

Takeaway: a plain 3D Gaussian has one colour and an elliptical footprint; attaching a small RGBA texture to every Gaussian
lets each primitive carry spatially varying colour (RGB) and non-ellipsoidal shape (alpha), so quality is much higher at the
same, very small Gaussian budget.

Numbers shown (all verified against `../textured-gaussians/summaries/chao2025textured.md`):
- ≈ 100k Gaussians for both models in the custom captures (Fig. 6 text, “on average around 100k”).
- Table 1, PSNR gain over 3DGS* at 1 % of the Gaussians: Blender +1.22, Mip-NeRF 360 +1.36, DTU +1.55, T&T +1.20,
  Deep Blending +0.86 dB (range quoted as “+0.86 to +1.55 dB”, not the text's “nearly 2 dB”).
- Fig. 5 average counts at 1 %: ≈ 2.5k per object, ≈ 39k per scene.
- Our own 2D toy (labelled “our 2D toy”): 64 Gaussians, same stage-1 start, 19.29 dB plain vs 25.89 dB RGBA 8 × 8
  (`../data/generated/toy_results.json`).

Style: identical to the 5-min version (CMU Serif, Pango markup via `tx()`, black bg, GS `#90CAF9`, ours `#FFB74D`, dim
`#8A8A8A`, header top-left y 3.62 size 24, caption y −3.5 size 28, held ≥ len/15 + 1 s, text added instantly).

## Scenes (actual durations from the v4 1080p60 render: 22.4 + 21.8 + 20.6 = 64.8 s)

### T1_Hook (22.4 s: hook 0–8.4 s, idea 8.4–22.4 s)
1. Hook (≈ 8 s). Title “Textured Gaussians” (60 bold, y 3.1). Fig. 6 children's-art pair (portrait close-ups):
   `fig6_art2_3dgs` / `fig6_art2_ours`, 3.2 tall at x ∓3.45, labels “3DGS*” (GS) / “Textured Gaussians” (ours).
   Caption: “Same ~100k Gaussians. One is blurry.” → “Why? Look at one Gaussian.”
2. Key insight (≈ 15 s). Header “The idea”. Left: plain orange splat (3.4 at (−4.3, 0.35)) with dashed 2σ ellipse,
   label “3DGS: one color, one ellipse”. Right: a textured splat (3.4 at (1.4, 0.35)) fed by an RGB tile and an α tile
   (1.25 each, x 5.3, y 1.2 / −0.6) via arrows; label “+ RGBA texture”. Caption “Plain Gaussian: one color, elliptical
   footprint.” Then ValueTrackers morph α 1 → 5-point star while the colour texture (flower-wallpaper crop, T = 16)
   fades in. Captions: “Give every Gaussian its own small RGBA texture.” → “Alpha reshapes it; RGB paints detail.”

### T2_Method (21.8 s: mechanism 0–13.5 s, toy 13.5–21.8 s)
3. Method in one picture (≈ 12 s). Header “How it works”. Colour row (y 1.0): base colour + texture colour = per-pixel
   colour; alpha row (y −1.45): texture alpha × Gaussian falloff = per-pixel opacity; arrows into one textured Gaussian
   (3.0 at (4.3, −0.2)). One equation (y 2.6): αᵢ(p) = αᵢᵗᵉˣ(u, v) · 𝒢ᵢ(x) · oᵢ.
   v4 layout: tiles 1.9 at x −5.6/−3.2/−0.8, rows y 0.9 / −1.5, final Gaussian 3.4 at (4.3, −0.3); base colour orange,
   texture = saturated children's-art crop residual. Captions (each held before the next row appears): “Color: base color
   plus texture color.” → “Opacity: texture alpha times the Gaussian falloff.” (+ equation) → “Then blended front to
   back, just like 3DGS.”
4. Proof in 2D (≈ 10 s). Header “Our 2D toy: 64 Gaussians fit a photo”. Target / plain / RGBA (3.0 tall at x −4.3/0/4.3);
   stage-2 frames of plain and RGBA animate together from the same start, PSNR counters (3 s). 2σ outlines flash on both.
   Captions: “Same 64 Gaussians, same starting fit.” → “Still 64 Gaussians: 19.3 dB vs 25.9 dB.”

### T3_Payoff (20.6 s: bars 0–9.7 s, Fig. 5 pairs 9.7–15.5 s, end card 15.5–20.6 s)
5. Payoff (≈ 15 s). Header “Paper results · equal Gaussian count”. Bar chart (axes 9 × 3.3 at y 0): PSNR gain over
   3DGS* at equal count (Table 1), 5 datasets, paired bars 100 % (GS: +0.16/+0.09/+0.07/+0.08/+0.29) and 1 % (ours:
   +1.22/+1.36/+1.55/+1.20/+0.86); legend y 2.95, chart title above axes. Caption “1% of the Gaussians: +0.86 to +1.55 dB PSNR.” → “The fewer Gaussians,
   the bigger the gain.” Then Fig. 5 pairs (2.4 tall at x ∓3.3): Deep Blending drjohnson + DTU buddha,
   labels “3DGS*” / “Textured Gaussians”. Caption “1% budget: ≈39k Gaussians per scene, ≈2.5k per object.”
6. End card (≈ 5 s). Title + subtitle, CVPR 2025, “arXiv 2411.18625 · textured-gaussians.github.io”, authors,
   affiliations.

## Version history
- v1 — first cut, 60.3 s. Problems: T2 rows/equation appeared before the first caption was read (caption hold only
  runs at the next caption); T2 caption promised a ray that is never shown; final textured Gaussian small and dark;
  T1 splats small; T3 “~” glyph rendered as a raised tilde; “fewer Gaussians, bigger gain” not backed by the chart.
- v2 — explicit `_hold_caption()` between T2 reveals, captions rewritten to match visuals, bigger/brighter T1 and T2
  splats (saturated texture), T3 adds 100 % bars + legend, “~” → “≈”, larger Fig. 5 pairs. 65.2 s.
- v3 — T3 legend collided with the +1.55 label / chart title: axes shortened to 3.3, legend moved to y 2.95.
- v4 — T3 centre dataset labels on two lines (“Deep Blending” nearly touched the images). Final 1080p60: 64.8 s.
