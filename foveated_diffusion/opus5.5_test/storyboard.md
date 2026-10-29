# Storyboard — Foveated Diffusion explainer (3b1b style)

**Current version: v4** — `animation_foveated_diffusion_v4.py`
Render: `./render.sh l` (480p15 test) · `./render.sh h` (1080p60 final). Env: `imp_tome` conda env with
`LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7` (baked into `render.sh`; imp_tome ships only libffi 8).
Final cut: `media/final/FoveatedDiffusion_animation_foveated_diffusion_v4_1080p60.mp4` (11 scenes concatenated, ≈ 9 min 20 s).

Source: Chao, Yariv, Xiao, Wetzstein, *Foveated Diffusion: Efficient Spatially Adaptive Image and
Video Generation*, arXiv 2603.23491. PDF + extracted text: `foveated-diffusion/pdfs/`. Images in
`data/` are crops of the paper's figures, except fig5/fig9/fig10 (original result PNGs since v6); `data/generated/` is built by `build_assets()` on import
(delete `data/generated/done_v1` to rebuild).

## Version history
- v1 — first full cut, 11 scenes, ≈ 6.4 min.
- v2 — caption auto-hold (15 chars/s + 1 s) and two-line wrap; fixed overlaps (S01 arrows over tokens,
  S06 v-label vs equation, S03/S06/S09/S10 text off-screen), S02 gaze dot z-order, S07 DiT box turning white.
- v3 — S01 query row scans 4 tokens; big square grows from its own center; S08 application rows raised;
  S10 summary narrower.
- v4 — S04 rows moved up (region labels no longer touch captions); captions never break inside “N × M”;
  two-line captions are centered (split at a space outside all markup elements).
- v5 — slide-deck clips (`animation_foveated_diffusion_v5.py`, S01–S10, `--save_sections`): `CAPTIONS = False` (no captions/headers/part cards), one `next_section` per visual step ending on the finished frame (0.3 s hold), S03 attention-squares "pairs" beat removed (100-token row centred + labelled), S01 attention/N² beats isolated in their own sections; manifest `slides_v5/sections.json`.
- v6: clean images (`animation_foveated_diffusion_v6.py`). The fig5, fig9 and fig10 images in `data/` were PDF crops with 1 to 2 px white gutter lines at the edges. They are now the original result PNGs from `foveated_diffusion-private/results/eccv2026_final_v2/` (1024 px, LANCZOS to 512). The map is in `data/_pdfcrop_backup_2026-10-14/sources.json`. The old crops and old generated assets are backed up in the same folder. Only S01, S02, S05, S08 and S09 were re-rendered (`slides_v6/render_v6.sh`). Section names and order match v5. Manifest `slides_v6/sections.json` points to v6 files for those scenes and to v5 files for the rest.

## Global style
| Property | Value |
|---|---|
| Font | CMU Serif (text); Latin Modern Math for glyphs CMU lacks: × ⊙ − → ≈ ∼ ∈ 𝒩 ℒ 𝔼 ′ ≪ ≤ ‖ · ↓ |
| Math | Pango markup, no LaTeX. Subscript: `rise=-3000` + `<small><small>`; superscript: `rise=4500` |
| Background | black |
| HR token | `#448AFF`; LR token `#00C853`; 3rd level (S10 only) `#1B7F3B` |
| Other colors | dim text `#8A8A8A`, ghost slots `#3A3A3A`, red `#FF5252` (fixation dot, naïve failure labels), amber `#FFCA28` (image curve, face key, spline), purple `#E040FB` (video curve, door key), LoRA tag `#FF7043`, DiT box fill `#1E2A33` |
| Token | `RoundedRectangle`, square, corner radius 10% of side |
| Grid | cell c = size/n; HR side 0.8c (10% margin); LR token covers 2×2 cells, side 1.6c |
| 1-D row (S04) | side 0.8, gap 0.16 (20%), pitch 0.96 |
| Content tokens (S06, S07) | fill = mean image color of the patch, stroke = HR blue (1.2) / LR green (1.6) |
| Text | instant (`self.add`), never animated in |
| Header | top-left, y = 3.62, size 24, dim: “k · title” |
| Caption | bottom, y = −3.5, size 28 (26 when split into 2 centered lines), max width 13.2, bottom ≥ −3.85; held ≥ len/15 + 1 s |
| Part card | “Part k” (30, dim) over title (50, bold, ≤ 12.5 wide), 2.4 s, instant |

Common foveation mask: 8×8 blocks of a 16×16 grid, blocks whose centre is within r = 2.0 of the
fovea centre → 12 blocks (4×4 minus corners) → **m = 48 HR tokens, 52 LR tokens, L = 100 of 256 (39%)**.
Fovea centre (block coords, row, col): S03 (3, 4) on the dog; S06 (4, 4) on the training lion; S07 (3, 3) on the output lion.

## Scenes (durations from the 480p15 render)

### S00_Title (8.3 s)
Title “Foveated Diffusion” (72, bold, y 2.75), subtitle (32, y 1.95), authors (22, dim, y −3.3).
16×16 `FoveaField` (size 3.8, centre (0, −0.75)) fades in, then the fovea wanders along
(4,4)→(3.5,3)→(3,2.5)→(3.5,5)→(4.5,5.5)→(5,4)→(4,3.5)→(4,4); blocks split 1→4 / merge 4→1 (0.55 s per step).

### S01_TokenCost (62 s) — “1 · Every token costs the same”
1. `fig5_dog_full` (4.6 tall at (−3.5, 0.1)); 16×16 grid lines drawn; tokens with mean patch colours fade in, image fades out.
2. Face token (4,8) boxed amber, door token (2,1) boxed purple; legend squares + labels at x ≈ 1.0 (no arrows).
3. Tokens recolour to HR blue; “h × w = 16 × 16 = 256 tokens” (y 2.55).
4. Attention matrix (`generated/attn.png`, softmax of token-colour similarity, 4.0 at (3.4, −0.1)) + “256 × 256 = 65,536 query–key pairs”.
   Query box + matrix row highlight scan tokens (5,8)→(2,3)→(9,12)→(13,5)→(7,8).
5. Squares: side 0.9 at (1.2, 0.9) “N tokens / N² pairs”; side 3.6 at (4.6, 0) grows from centre, “2× resolution: 4N tokens / 16 N² pairs”.
Captions: DiT grid of latent tokens · each token = a patch · both tokens cost the same · full resolution every step · self-attention · highlighted row · N × N · 2×res → 4× tokens, 16× attention · video × frames · “does the viewer see every token equally?”

### S02_Vision (51 s) — “2 · Where you look matters”
1. Acuity plot (schematic): Axes 5.6×3.4 at (−3.6, 0.35), x −40…40°, curve 1/(1+|e|/2.5), fovea band |e| < 5° (HR, 18% opacity), labels fovea / periphery / “schematic”.
2. `fig5_farmer_full` (4.6 at (3.4, 0.25)), red dot (r 0.1, z-index 10) at (0.50, 0.55); crossfade to gaze-blurred `gaze_A` (blur σ = 0.045·max(0, e − 40 px), ≤ 9 px); dashed fovea circle; gaze moves to (0.80, 0.22) with `gaze_B`.
3. Foveated rendering: equation x^fov = M ⊙ x^high + (1 − M) ⊙ Up(x^low) (42, y 2.35); visual equation of 5 images (1.9 each, y 0): M, x^high, 1 − M, Up(x^low) (8× down, nearest up), x^fov.
4. Card: “Can a diffusion model spend its tokens the same way?” (40, ≤ 12.5 wide).

### S03_Tokenization (44 s) — “3 · Foveated tokenization”
Grid 16×16, size 5.0 at (−3.4, 0.1). Mask panel: 16×16 binary cells (size 2.4 at (3.6, 1.2)), “M ∈ {0,1}^(h×w)”, M(i,j)=1/0 legend.
Periphery dimmed (30%) + white Union outline of the 12 blocks; periphery 2×2 → 1 green via `ReplacementTransform`, lagged outward from the fovea (4 s).
Counting (x ≈ 3.3): m = 48 · (h·w − m)/4 = 208/4 = 52 · boxed L = m + (h·w − m)/4 = 100 · 100/256 ≈ 39%.
Tokens flatten into a 100-token row (side 0.1, pitch 0.125, y 2.75, raster order over blocks).
Attention squares from top-left (−4.2, 2.2): full 4.4 (HR 25%) vs foveated 4.4·100/256 = 1.72; “65,536 → 10,000 pairs, ≈ 15%”.

### S04_RoPE (56 s) — “4 · Mixed-resolution positional encoding”
Row (a) y 1.1: 8 HR tokens 0…7; dials (r 0.3, y 2.4) rotate by 0.62·index.
Tokens 4,5 and 6,7 merge into green tokens at slots 4 and 6 (dashed ghosts at 5, 7), dials show “?”.
Row “high-res query” y −0.3: 0 1 2 3 (HR) | 4, 6 (LR) with ghosts 5, 7.
Row “low-res query” y −1.6: HR 0 at slot 0, 1 at slot 2 (ghosts 1, 3) | LR 2, 3 at slots 4, 6.
Region rules at y −2.2 (“high-res region” blue, “low-res region” green). Mirrors paper Fig. 3 / Wu et al. 2025.

### S05_Naive (35 s) — “5 · Naïve mixed-resolution denoising”
Rows “Full high-res” (y 1.35) and “Naïve mixed-res” (y −1.2), images 2.35 at x −1.7, 0.95, 3.6: horse, guitar, farmer (Fig. 5).
Red labels: duplicated horse · two guitars · two hat brims. Captions explain the 2 × 2 scale mismatch.

### S06_Training (79 s) — “6 · Foveated training: mixed-resolution-consistent targets”
x = `lion_train` (1.9 at (−5.7, 1.3)) → bicubic → Down(x) (0.95 at (−5.7, −1.6)); encoders E at x −3.95;
z_high 16×16 (size 1.9 at (−1.9, 1.3)), z_low 8×8 (size 1.9 at (−1.9, −1.6)); M panel (1.1 at (0.55, −0.15));
z_fov grid (2.6 at (3.5, −0.15)) filled by copies: fovea tokens from z_high, periphery from z_low (unused ones dimmed to 20%).
Flow matching: z_fov moves to (−4.3, 0.35) size 2.8; equation z_t = (1 − t) z_0 + t z_1 (y 2.55); t slider (length 2.8, y −1.65) animates 0→1→0.55 with colour interpolation clean↔noise;
DiT + LoRA box (1.7×2.2 at (0.2, 0.35)); v_θ grid (2.8 at (4.4, 0.35), colours 0.5 + 0.9·(noise − clean)) with label below; loss ℒ = 𝔼‖v_θ − (z_1 − z_0)‖² (y −2.45).
Details card: FLUX.2 Klein 4B · 90k images · 1024² / Wan2.1 1.3B · Vchitect-T2V · 480p / LoRA rank 32 · 10k steps · 8 × H100 / random circles, random gaze paths.

### S07_Generation (57 s) — “7 · Foveated generation”
Noise foveated grid (2.6 at (−5.0, 0.6)); DiT box (1.4×1.8 at (−2.2, 0.6)) with in/back arrows; t counter 1.00 → 0.00 in 8 steps (0.6 s each, DiT box pulses grey), tokens become `lion_out` patch colours.
Split into z_high (16×16, 1.6 at (0.9, 1.55), ghosts outside fovea) and z_low (8×8, 1.6 at (0.9, −1.25), ghosts inside fovea).
Decoders D at x 2.5 → x^high (fovea-only RGBA, 1.6) and x^low (half-res, 0.8) at x 4.35.
Blend: equation (7) + visual equation (M′, x^high, 1 − M′, Up(x^low), x^fov; 1.9 each); hero output 4.9 tall at (−2.0, 0.1) with blocky fovea outline; “48 high-res + 52 low-res = 100 tokens instead of 256”.

### S08_Masks (59 s) — “8 · Foveation masks are just data”
Fig. 9 grid: groups Position / Size / Disjoint regions at x −4.45, 0, 4.45; Naïve/Ours columns ±0.85; rows y 1.55, −0.13, −1.81; images 1.62.
Saliency: “Random masks” (x −5.55) vs “Saliency-guided” (x −3.7), 3 rows of 1.55 (flower, Fig. 10a); Applications 2×2 (2.1 each at (0.4|3.9, 1.3|−1.35)): VR gaming, robotics simulation, driving simulation, aerial footage.
Video path: 5.0×2.8 frame at (−2.3, 1.2); control points (0.2,0.35,r0.16), (0.55,0.7,r0.22), (0.82,0.4,r0.14); natural cubic spline; white fovea circle travels 3 s + back 2 s; 5 `vid_fox_ours` frames (1.44 tall) at y −1.95.

### S09_Results (64 s) — “9 · Results”
Speedup plot (Axes 5.4×4.2 at (−3.4, 0.2); x = 100 − token ratio): curve vertices read from the paper's Fig. 6 vectors; image (amber) ends 2.2×, video (purple) 4.2× at 25%.
Table 1 (right): tokens / speedup / HPSv2.1 naïve / ours — 100% 1.00× full 0.280 · 43% 1.61× 0.268 0.279 · 30% 1.98× 0.270 0.279 · 26% 2.08× 0.275 0.280; FID footnote.
Qualitative: 2 rows × (Full, Naïve, Ours) × 2 subjects (horse, dog / guitar, farmer), images 1.95.
User study: protocol strip (fixate 5 s → image A 1 s → fixate 1 s → image B 1 s → A or B?) and bars (width 6.2 from x −2.2): Ours vs Full 47.3% (p = 0.48), Ours vs Naïve 87.4%, Full vs Naïve 90.8% (p < 0.0001), dashed 50% line.
Video: fox rows Full / Naïve / Ours (5 frames, 1.22 tall, y 1.75, 0.4, −0.95); VBench subject consistency 0.941 / 0.907 / 0.945; “3.5× faster at 38% of the tokens (Wan2.1 1.3B, 480p)”.

### S10_Outlook (46 s) — “10 · Limitations and outlook”
Fig. 28 crops (image 4.0 at (−4.6, 0.3), video strip at (−1.5, 0.3)) + text: seams / cause / fix.
Three-level grid (size 4.2 at (−3.5, 0.25)): HR core, LR ring, 4×4-cell coarse tokens (side 3.2c, `#1B7F3B`).
Gaze-following `FoveaField` (4.2 at (3.5, 0.25)) with red gaze dot on path (3.2,2.6)→(2.6,4.6)→(4.2,5.6)→(5.4,4.2)→(4.6,2.8)→(4,4).
Summary: three numbered ideas (27, left edge −6.0) + “≈ 2× faster images · ≈ 4× faster video · perceptually indistinguishable”; end card (no header): title, arXiv 2603.23491 · bchao1.github.io/foveated-diffusion, authors.
