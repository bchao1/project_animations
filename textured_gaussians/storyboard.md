# Storyboard — Textured Gaussians teaser (3b1b style)

**Current version: v3** — `animation_textured_gaussians_v3.py`
Render: `./render.sh l` (480p15 test) · `./render.sh h` (1080p60 final). Env: `imp_tome` conda env with
`LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7` (baked into `render.sh`; imp_tome ships only libffi 8).
Final cut: `media/final/TexturedGaussians_animation_textured_gaussians_v3_1080p60.mp4` (11 scenes, ≈ 5 min 20 s).

Source: Chao, Tseng, Porzi, Gao, Li, Li, Saraf, Huang, Kopf, Wetzstein, Kim, *Textured Gaussians for Enhanced 3D
Scene Appearance Modeling*, CVPR 2025, arXiv 2411.18625. PDF, extracted text and LaTeX source:
`textured-gaussians/pdfs/`. Assets: `python build_assets.py` (imp_tome, CUDA optional) writes
- `data/fig*.png` — crops of the rasters embedded in the source's figure PDFs (Fig. 1, 2, 3, 5, 6, 7, A-tex-ablation);
- `data/generated/` — our own 2D toy experiment (`toy_results.json` holds every toy number shown).

Toy experiment (ours, labelled as such on screen): 128 × 128 crop of the teaser's children's-art wall; N 2D Gaussians
alpha-composited front to back in index order (Eq. 3/4); L1 loss, Adam. Plain sweep N = 16/64/256/1024, 3000 iters →
16.9 / 19.2 / 22.5 / 29.3 dB. Two-stage recipe like the paper: stage 1 plain N = 64 (1500 iters, 18.98 dB), stage 2 from
that state for 1500 iters with 8 × 8 textures (UV = local coords / σ, texture spans ±3σ, bilinear, α = α_tex·G·o,
c = c_base + c_tex): none 19.29 · alpha 20.21 · RGB 24.74 · RGBA 25.89 dB.

## Version history
- v1 — first full cut, 11 scenes, 7:06 (too long).
- v2 — trimmed to 5:20 (shorter captions, faster anims); fixed S03 text off right edge, S07 ellipse outlines spilling out of
  images (now rasterized inside each image), S08 bar labels overlapping (two-line names), S09 same, S10 text off-screen /
  overlapping (2 × 2 cost grid + limitation block), S04 renders larger/brighter, S05 diagram scaled 1.3× with faint texture
  support on the Gaussian plane, S08 qualitative rows re-laid (scene names in the centre gap).
- v3 — S07 outlines thinner; S08 "PSNR (dB)" label merged into the x-axis caption (it collided with a plot title);
  S06 colour row holds 2.8 s before the alpha row appears.

## Global style
| Property | Value |
|---|---|
| Font | CMU Serif; Latin Modern Math for glyphs CMU lacks (× ⊙ − → ≈ ∑ ∏ 𝒢 ℒ ⊤ ′ ½ …) via `mathify()` |
| Math | Pango markup, no LaTeX. Subscript `rise=-3000` + `<small><small>`; superscript `rise=4500`; bold = vectors |
| Background | black |
| Colors | 3DGS*/plain `#90CAF9`, ours `#FFB74D`, alpha-only `#87CEEB`, RGB `#32CD32`, RGBA `#FFA07A` (paper's bar colours), ray `#FFEB3B`, axes r¹ `#FF5252`, r² `#69F0AE`, n `#448AFF`, dim `#8A8A8A`, box `#1E2A33` |
| Text | instant (`self.add`), never animated in; bare `&` auto-escaped in `tx()` |
| Header | top-left, y = 3.62, size 24, dim: “k · title” |
| Caption | bottom, y = −3.5, size 28 (26 on 2 centred lines), max width 13.2; held ≥ len/15 + 1 s |
| Part card | “Part k” (30, dim) over title (50, bold), 1.8 s |
| Splat renderer | `splat_layers()` numpy: α = clip(o·G·α_tex·[|u|,|v| ≤ 3], 0, 0.99), colour = base + c_tex (bilinear, texel centres at ±3σ) |

## Scenes (durations from the v3 480p15 render)

### S00_Title (7.0 s)
Title (66 bold, y 3.0), subtitle (34, y 2.3); toy fits 3.3 tall at x ∓1.9, y −0.25: `toy_final_none` “64 Gaussians” then
`toy_final_rgba` “64 textured Gaussians”; 11 authors on 2 lines + “Stanford University · Meta · University of Maryland,
College Park · CVPR 2025” (21, dim, centred y −2.95).

### S01_Hook (21.9 s) — header “Textured Gaussians · CVPR 2025”
`fig1_teaser` (Fig. 1 rendered, width 13.4, y 0.55). Then Fig. 6 children's-art pair (3.4 tall at x ∓3.45): “3DGS*” /
“Textured Gaussians (RGBA)”. Captions: 3DGS builds scenes from blobs · same count, 3DGS blurs · custom capture ≈ 100k
Gaussians (paper: “on average around 100k”) · give every Gaussian a texture map.

### S02_Splatting (32.8 s) — “1 · Gaussian splatting recap”
A: live 2D Gaussian image (3.8 at (−3.6, 0.15), ValueTrackers θ, σ₁, σ₂, 2σ ellipse + r¹/r² arrows) rotating/stretching;
equations (30, left-aligned at x ≈ 2.95): 𝒢(x) = exp(−½(x−μ)ᵀΣ⁻¹(x−μ)), Σ = RSSᵀRᵀ, Σ′ = JWΣWᵀJᵀ, Σ₂D = upper-left 2 × 2.
B: canvas of 5 splats (4.2 at (−4.3, 0.25)), pixel p at canvas (0.05, 0); side view ray y 1.55 from camera x −0.9 to 6.3,
the 3 splats hit (α = 0.63, 0.60, 0.67, computed) at x 1.0/2.9/4.8; eq. c(p) = ∑ cᵢαᵢ∏(1−αⱼ), αᵢ = 𝒢ᵢ²ᴰ(p)·oᵢ;
travelling dot drives transmittance bar (T 1 → 0.37 → 0.15 → 0.05) and accumulated colour swatch; weights 0.63 + 0.22 + 0.10.

### S03_Limitation (29.9 s) — “2 · The limitation”
Orange splat (3.2 at (−4.0, 1.0)) with dashed scanline; plot below: colour constant vs opacity Gaussian. Right edge: the
paper's two properties (26). Red square + GS-blue circle “a sharp square corner cannot be one ellipse”.
Toy sweep: target + N = 16/64/256/1024 (2.35 tall, x −5.4…5.4, y 0.35) with PSNR from json. Caption: 3DGS* averages
4.4 M Gaussians on Mip-NeRF 360 (Table 2, #GS column).

### S04_Textures (35.1 s) — “3 · Textured Gaussians”
Eq. “texture map per Gaussian: 𝒯 × 𝒯 × K, K ∈ {1, 3, 4}” (y 2.75). 4 columns x = ∓5.1, ∓1.7: RGB + α tiles (1.2 each,
y 1.4, T = 16), rendered splat (2.75, y −0.95): none / alpha (smooth noise 0.15–1) / RGB (painting residual) / RGBA
(after the paper's Fig. 2). Morph: splat 4.2 at (−3.3, 0.2) whose α_tex blends 1 → 5-point star → 1 → crescent (T = 32),
live α tile 2.2 at (3.6, 0.55), eq. αᵢ(p) = αᵢᵗᵉˣ(u,v)·𝒢ᵢ(x)·oᵢ.

### S05_UVMapping (29.5 s) — “4 · Ray–Gaussian intersection and UV mapping”
Parallel projection P(w) = (−1.75, −0.05) + 1.3·(X + 0.5Y, Z + 0.28Y). Camera o (−3.45, −0.5, 0), image plane X = −2.15,
μ (0.9, 0.35, 0.25), n ∝ (−0.9, 0.3, 0.2), σ¹ 0.42, σ² 0.3, m = 3. Plane carries the texture at 22% (support ±3σ) and the
textured splat (α = α_tex·G). Axes r¹/r²/n. Texture panel (T = 16 flower-wallpaper crop, 2.6 at (5.1, 0.85)). A point path
on the Gaussian plane drives pixel p, ray, hit x, (u, v) readout, and a 2 × 2 texel box. Eqs: x = o + t d,
t = (μ − o)·n / (d·n) (y 2.7); u = (mσ¹ + (x − μ)·r¹)/(2mσ¹)·(𝒯 − 1), v likewise, m = 3 (y −2.3).
(Eq. 5 is written in the equivalent normalized-direction form.)

### S06_Appearance (31.9 s) — “5 · Color, opacity, compositing”
Eq. 7 (y 2.55 left), Eq. 8 (right). Colour row y 0.75: base (SH) + texture = per-pixel colour (1.75 tiles); alpha row
y −1.75: texture alpha × 𝒢·o = per-pixel opacity; arrows to one textured Gaussian (3.2 at (4.2, −0.5)).
Then Eq. 9 (38, y 2.2), “c_tex = 0 and α_tex = 1 ⟹ original 3DGS”; two-stage boxes (Stage 1 plain 3DGS, 30,000 its,
densification on → Stage 2 + textures, 30,000 more, densification off); ℒ = λℒ₁ + (1 − λ)ℒ_SSIM, λ = 0.8.

### S07_ToyFit (31.7 s) — “6 · Our 2D toy: 64 Gaussians, with and without textures”
Images 2.6 tall at x −5.55/−2.78/0/2.78/5.55, y 0.3. Stage-1 frames (every 50 its) with iteration/PSNR readout; then 4
stage-2 variants animate together with PSNR counters (19.29 / 20.21 / 24.74 / 25.89 dB). 2σ outlines rasterized
(`ellipse_overlay`) on the plain and RGBA fits. Captions note: toy RGB > alpha, paper's 3D result is the reverse; toy numbers
are ours and textures add parameters.

### S08_Results (46.1 s) — “7 · Novel-view synthesis results”
Datasets: Blender 8, Mip-NeRF 360 9, DTU 5, Tanks and Temples 2, Deep Blending 2, 2 custom rooms; baseline 3DGS* (GOF-style).
PSNR vs % Gaussians (supp. Table A.3, 1/2/5/10/20/50/100%) for Mip-NeRF 360, T&T, DB (3.6 × 3.0 axes at x −4.6/0/4.6).
Table 1 gain bars (100% vs 1%): +0.16/+1.22, +0.09/+1.36, +0.07/+1.55, +0.08/+1.20, +0.29/+0.86. Caption uses the table
range (+0.86 to +1.55 dB), not the text's “nearly 2 dB”. Qualitative Fig. 5 pairs (2.4 tall at x ∓3.3, y 1.4/−1.22):
lego + train, then drjohnson + counter; “about 2.5k per object, 39k per scene”.

### S09_Ablations (32.3 s) — “8 · Alpha, RGB or RGBA?”
ΔPSNR at 1% (supp. Table A.4; its T&T/DB columns are swapped vs the header, read in Table 1 order): Blender 0.75/0.71/1.22,
T&T 1.03/0.83/1.18, DB 0.71/0.67/0.86. Side: channels 1/3/4, model size vs 3DGS 1.27×/1.82×/2.08×. Lego + ship crops
(3DGS*, alpha, RGB, RGBA at 1%). Table 2 (same model size) with best per row in bold.

### S10_Outlook (21.5 s) — “9 · Costs and limitations”
2 × 2 grid (supp. Table A.2, 2.12 M Gaussians, 4 × 4 RGBA): 125.3 → 261.2 MB, 2.4 → 4.0 GB, 0.073 → 0.123 s/frame, training
≈ 2×. Limitations (Sec. 5): diffuse 2D textures, no spatially varying specular → 3D volume textures / 5D radiance fields
(TensoRF, triplanes), time-varying textures. Summary (3 points), end card: title, CVPR 2025, arXiv 2411.18625 ·
textured-gaussians.github.io, authors, affiliations.
