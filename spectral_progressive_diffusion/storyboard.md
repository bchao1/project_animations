# Storyboard — Spectral Progressive Diffusion teaser (3b1b style)

**Current version: v4** — `animation_spectral_progressive_v4.py`
Render: `./render.sh l` (480p15 test) · `./render.sh h` (1080p60 final). Env: `imp_tome` conda env with
`LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7` (baked into `render.sh`). Scenes render in parallel; logs in `scratch/logs/`.
Final cut: `media/final/SpectralProgressiveDiffusion_animation_spectral_progressive_v4_1080p60.mp4`
(10 scenes concatenated, ≈ 5 min 12 s).

Source: Xiao, Chao, Yariv, Wetzstein, *Spectral Progressive Diffusion for Efficient Image and Video
Generation*, arXiv 2605.18736. PDF, text and LaTeX source: `spectral-progressive-diffusion/pdfs/`
(summary with all numbers: `spectral-progressive-diffusion/summaries/xiao2026spectral.md`).
Every number on screen is from the paper (Tables 1–4, Figs. 3–5, Sec. 3–4) except those explicitly labelled as
our demo on the corgi image (β ≈ 3.0 for this image, t = 0.9 frontier example, κ = 1.50 / t̃ = 0.50 worked example
— the last is plain arithmetic with the paper's Eq. 5–6).

## Data
- `data/corgi_clean.png` — clean corgi from paper Fig. 1 (`pdfs/src/figures/pipeline.001.jpeg`, crop 2983–3526 × 279–822, → 512²).
- `data/{buddha,canyon}_{flux1x,ralu158,ours166,flux484,ralu498,ours577}.png`, `{city,aurora}_{pixelgen1x,ours16}.png`,
  `coral_{wan1x,wan2x,ours203}_{0..3}.png`, `fall_*` — embedded rasters of `figures/main_qual.pdf` (Fig. 3).
- `data/pt_{crow,phone,camera,bus}_{full,d0001,d001,d01,d05,d1}.png` — `supp_figures/spectral_passthrough.pdf` (Fig. 5).
- `data/fig4b_editing.png` — Fig. 4(b) rendered from the vector PDF at 400 dpi (keeps its text).
- Computed at import (`SpecData`, `expansion_demo`): orthonormal 2D FFT of the centred corgi, radially averaged PSD
  (fit β ≈ 3.0, A ≈ 1.1e4), noisy states x_t = (1 − t)x_0 + tε (seed 7), SNR = 1 “frontier” ω*(t) from the monotone
  PSD envelope, low-pass reconstructions, and the DCT spectral-expansion demo (128², t_i = 1/3, r = 2, seed 3).

## Version history
- v1 — first full cut, 10 scenes, 6:55.
- v2 — log-log axes: x-axis moved to the bottom (y range offset, `ax.p`); frontier sweeps parameterised by log ω*
  (`t_of_w`) so every grid size gets screen time; trimmed captions (5:10); S05 notation x̃ = κ·x, t̃ = κ·t;
  S06 δ readout moved inside the plot, s₁/s₂ labels separated; S07 diagram narrowed, equations fit;
  S08 7.09× label no longer on the RALU dot; S09 editing figure re-rendered with its text; S01 box includes labels.
- v3 — S02 paper-β note split in two lines (was clipped); S03 clean spectrum held 2.5 s before noise appears;
  S04 caption “t ≈ 0.9”; S05 demo at t_i = 1/3 (κ = 1.50, t̃ = 0.50) and DCT colour range −1.4…1.4 so the
  low-frequency corner and the dog are visible; S08 label sizes; S09 figure width 10.6.
- v4 — S07 closing caption made exact (PixelGen LoRA 0.913 is near, not above, the 0.921 baseline):
  “Same speedup, better quality: LoRA lifts ImageReward back to (or near) the full model.”
- v5 — slide-deck clips (S00–S08): `CAPTIONS = False` (no captions/headers/part cards, no S04 “→ Part 5/6” card), one manim section per visual step ending on the finished frame (0.3 s hold, fades start the next section, no trailing fade-outs), S01 attention “N² → 16 N² pairs” labels removed; render `scratch/render_v5.sh h <Scenes>` (`--save_sections`), manifest `slides_v5/sections.json`.
- v6: result images in `data/` are the true source images, not PDF crops. The crops had a 1–2 px white gutter line and the paper's inset boxes. Sources: `/local/howard/.../qualitative_results/{baseline,pixelgen_baseline,supp_motivation}` (from `~/wavelet-diffusion/figure_scripts/main_qual.sh` and `ablation_qual.sh`) and Wan frames 0, 10, 30, 60 of `/miele/liory/wan_paper_eval/r720_s5_*/p036_seed36.mp4` (coral) and `p031_seed31.mp4` (fall). `fig4b_editing.png` is a new vector render clipped to the image grid. Backups and the full list: `data/_pdfcrop_backup_2026-10-14/` (`SOURCES.tsv`). Only S06 and S08 use these images, so only they were re-rendered: `scratch/render_v6.sh h S06_Schedule S08_Results`. Manifest `slides_v6/sections.json` (`scratch/qa_sections_v6.py`) points S00–S05 and S07 to the v5 render.
- v7: S06 and S08 load `data/*_box.png`. These are the clean v6 images with the paper's inset boxes drawn back on. Box positions come from `sharpness_based_inset_boxes(_consensus)` in `~/wavelet-diffusion/figure_scripts/make_figure_grid.py`. FLUX and passthrough rows use one consensus pair per row. PixelGen uses one pair per image. Colors are #FF4500 (orange, first box) and #FFD700 (yellow). Line width is 2 px on the downsampled paper cell, scaled to the native size (8 px at 1024 for FLUX, 4 px for PixelGen and passthrough). Script: `scratch/render_v7.sh h S06_Schedule S08_Results`. Manifest `slides_v7/sections.json` (`scratch/qa_sections_v7.py`).

## Global style
| Property | Value |
|---|---|
| Font | CMU Serif; Latin Modern Math for × − → ≈ ∼ ∈ 𝒩 ℒ 𝔼 ≤ ≥ ‖ · ∝ ≲ (no √ — it drops below baseline; use (…)^1/2) |
| Math | Pango markup via `tx()`; sub: rise −3000 + small small; sup: rise 4500 |
| Background | black |
| Colours | signal `#58C4DD`, noise `#FC6255`, frontier / ours `#FFCA28`, ours S=3 `#FF9100`, RALU `#E040FB`, Bottleneck `#A1887F`, baseline `#B0BEC5`, LoRA `#FF7043`, dim `#8A8A8A`; stages s₁ `#5B8FD9`, s₂ `#E8955A`, s₃ `#D9534F` (as in paper Fig. 1) |
| Spectra | log10 power, magma colormap, range −4 … 3.5 (FFT) / −1.4 … 1.4 (DCT demo) |
| Text | instant (`self.add`), never animated |
| Header | top-left, y = 3.62, size 24, dim: “k · title” |
| Caption | bottom, y = −3.5, size 28 (26 two centred lines), max width 13.2; held ≥ len/15 + 1 s |
| Part card | “Part k” (30, dim) over title (50 bold), 1.8 s |

## Scenes (durations from the v4 480p15 render)

### S00_Title (8.9 s)
Title (60 bold, y 2.8), subtitle (32, y 2.05), authors (24 dim, y −2.95), “Stanford University” (22 dim, y −3.4).
Left (−1.9, −0.45), 3.0: progressive trajectory image at grid n ∈ {32…512} (oracle x_0 block-mean + block-mean noise ×k);
right (1.9, −0.45): FFT of x_t, dimmed outside the current Nyquist square, yellow frontier circle. log ω* 3 → 256 in 6 s.

### S01_Cost (26.5 s) — “1 · Every step pays full price”
8×8 token grid (2.0 at (−4.6, 0.5)) → 16×16 (4.0 at (0.3, 0.5)); “self-attention: N² pairs → 16 N² pairs” at x 4.9.
Then x_t thumbnails t = 1.0 … 0.0 (1.75, y 1.2, pitch 2.1), denoising arrow y −0.1, 12×12 token grids under each (y −1.15),
red box around t = 1.0, 0.8 (with labels).

### S02_Spectrum (28.9 s) — “2 · Images are mostly low frequency”
Corgi (3.0 at (−5.3, 0.5)) → “FFT” → spectrum (3.0 at (−1.35, 0.5)); log-log axes (4.8 × 3.4 at (3.95, 0.45)), |ω| 1…256,
P 1e−5…10^4.5. Ring sweep draws the radially averaged PSD (blue); dashed fit; label “P_ω ∝ |ω|^−β, β ≈ 3.0 for this image”
inside plot; two-line note “paper’s fits: β = 1.92 (FLUX latents), 2.42 (Wan latents), 2.45 (pixels)” at y −2.05.
Low-pass reconstruction inside a growing circle (R 3 → 24 → 256) with dashed marker on the PSD.

### S03_Noise (32.7 s) — “3 · Noise drowns the high frequencies first”
Equation x_t = (1 − t)x_0 + tε (34, y 2.85). x_t (3.0 at (−5.3, 0.3)) with t readout, its spectrum (3.0 at (−1.55, 0.3))
with frontier circle; axes (4.6 × 3.4 at (4.0, 0.3)): signal (1 − t)²P_ω (blue), noise t² (red, flat), yellow crossing dot,
red shading above the crossing; SNR_ω(t) formula at (4.0, −2.2). t: 0 (hold 2.5 s) → 0.97 (4.5 s); then log ω* → 256 in 7 s.
Citation note “Rissanen et al. 2023 · Dieleman 2024” at (−3.4, −2.2).

### S04_Frontier (30.4 s) — “4 · Grow the grid with the frontier”
Fourier plane (3.6 at (−3.3, 0.35)), dashed Nyquist squares for n = 32, 64, 128, 256, solid white square = current grid,
frontier circle. Pixel panel (3.6 at (2.6, 0.35)) at the current grid (nearest), label “n × n grid, t = …”, and
“1/256 … all of the full-resolution pixels”. Sweep log ω* 6 → 256 in 8 s (32 → 64 → 128 → 256 → 512).
Closing card: “How do we grow without breaking the noise? → Part 5 / When exactly should we grow? → Part 6”.

### S05_Expansion (39.3 s) — “5 · Spectral noise expansion”
Row y 0.8: x_{t_i} 64² (1.9 at x −5.55) → T_Φ → DCT coefficients (1.9 at −2.9) → copy flies into the top-left quadrant of the
128² coefficient canvas (2.9 at 0.75), then the full canvas fades in with “t_i ε” in the new quadrant → T_Φ⁻¹ → x at 128²
(2.9 at 4.95). Step labels (i)/(ii)/(iii) at y −0.95.
Alignment: “x̃ = κ · x_{t_i}, t̃_i = κ · t_i”, “κ = r/(1 + (r − 1)t_i), r = s_{i+1}/s_i”, “e.g. r = 2, t_i = 1/3 → κ = 1.50, t̃_i = 0.50”
(centred at y 1.95). Three 2.0 images at y −1.05: expanded + aligned | a true x_t̃ at 128² | bilinear upsampling instead.

### S06_Schedule (40.2 s) — “6 · When to grow: a schedule from the spectrum”
Prop. 1 statement (26, y 2.8) and t_ω = 1/[1 + (δ/(P_ω(1 + P_ω − δ)))^1/2] (32 amber, y 2.1).
Axes 5.6 × 2.9 at (−2.6, −0.55): t_ω vs |ω| (log, 1…256) from the paper’s pixel fit (A = 3745.46, β = 2.4493);
dashed lines at s₁ω_max = 64, s₂ω_max = 128 (ω_max = 256), dots + t₁*, t₂* read-outs (no numbers shown);
Prop. 2 block at (4.2, −0.3); δ readout inside plot; δ sweep 0.01 → 0.1 → 0.001 → 0.01.
Timeline bar (y 1.9, t = 1 at x −6 → t = 0 at x 6) with s₁ = 1/4 (1/16 of the tokens), s₂ = 1/2 (1/4), s₃ = 1 (all tokens)
at the δ = 0.01 transition times; Fig. 5 bus row (6 × 1.95 at y −1.15): full res, δ = 0.0001 … 0.1.

### S07_FineTune (31.9 s) — “7 · Training-free, or fine-tuned”
Lanes s₁/s₂/s₃ (y 1.8, 0.6, −0.6), t-axis x = −5 + (1 − t)·5.9; stage segments with expand + align arrows jumping back to
t̃ = 2t/(1 + t); labels x̃_{t̃_{i−1}}, x_{t_i}. Equations block (≤ 5.2 wide at (4.3, 0.55)): x_{t_i} = (1 − t_i)x_0 + t_iε,
v^{s_i} = (x̃ − x_{t_i})/(t̃_{i−1} − t_i), ℒ. LoRA line: rank 32 · 2000 iterations · 5K synthetic images · ≈ 2 h on 8 H100s.
Bars (Table 2 ImageReward, axis from 0.70): Z-Image 50 steps 0.965 / TF 1.65× 0.904 / LoRA 1.65× 0.982;
PixelGen 25 steps 0.921 / TF 1.60× 0.799 / LoRA 1.55× 0.913.

### S08_Results (43.9 s) — “8 · Results”
Scatter (8.0 × 4.5 at (−1.3, 0.45)), speedup vs ImageReward, Table 1: FLUX 50/10/7 steps (1.00, 1.095), (4.84, 0.981),
(6.62, 0.920); Bottleneck (4.67, 0.889), (6.64, 0.792); RALU (1.58, 1.028), (4.98, 1.022), (6.69, 0.999);
Ours S=2 (1.66, 1.049), (5.77, 1.059), (6.78, 1.039); Ours S=3 (6.09, 1.042), (7.09, 1.015). Legend at (5.3, 1.3).
Fig. 3a grid: 6 columns (FLUX 1×, RALU 1.58×, Ours 1.66×, FLUX 4.84×, RALU 4.98×, Ours 5.77×) × buddha/canyon, 2.1 each.
Fig. 3b: PixelGen 1× vs Ours 1.6× (city, aurora; 2.4 each). Fig. 3c: coral 3 rows × 4 frames (1.25 tall) with row labels;
VBench image quality 0.6133 / 0.5706 / 0.6114 and “S = 3: 2.54×”.

### S09_Outlook (29.4 s) — “9 · Takeaways”
Fig. 4(b) editing (width 10.6). Limitations (3 bullets, Sec. 6). Three ideas + “up to 7× faster images · 2.5× faster video ·
training-free or fine-tuned” (intro claims). End card: title, “arXiv 2605.18736 · howardxiao.ca/speed”, authors · Stanford.
