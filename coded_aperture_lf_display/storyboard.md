# Storyboard — Coded-aperture TDM light field display teaser (3b1b style)

**Current version: v3** — `animation_tdmlf_v3.py`
Render: `./render.sh l` (480p15 test) · `./render.sh h` (1080p60 final). Env: `imp_tome` conda env with
`LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7` (baked into `render.sh`).
`render.sh` first imports the script once (builds the cached toy assets), then renders the 10 scenes in parallel
(logs in `media/log_<script>_<res>_<Scene>.txt`) and concatenates them.
Individual scene MP4s: `media/videos/animation_tdmlf_v3/<res>/S0k_*.mp4`.
Final cut: `media/final/CodedApertureLightFieldDisplay_animation_tdmlf_v3_1080p60.mp4` (10 scenes, 1920×1080 @ 60 fps,
5 min 26.2 s = 326.2 s). 480p15 test cut: 5 min 28.7 s.

Source: Brian Chao* (Chun-Hao Chao on the paper), Chang-Le Liu*, Homer H. Chen, *Time-Division Multiplexing Light Field
Display With Learned Coded Aperture*, IEEE Transactions on Image Processing 32, 350–363 (2023), DOI 10.1109/TIP.2022.3203210;
arXiv 2107.06205 (v3, Aug 2022 = the ingested text). National Taiwan University. The IEEE version is paywalled; the arXiv v3 is
the final accepted text. No LaTeX source on arXiv, no code, no supplementary video.
PDF, text, page renders: `coded-aperture-lf-display/pdfs/`; ingest notes with every number used:
`coded-aperture-lf-display/summaries/chao2023time.md`.
`data/fig*.png` are the paper's embedded figure rasters at native resolution (exported by `scratch/crop_figs.py`);
`data/fig9_apertures_9x9.npy` = the four binary 9 × 9 apertures of Fig. 9, digitised cell by cell (no ambiguous cells).
`data/tip2021.gif` + `data/gif_frames/` = the website teaser (2 frames × 1 s): frame 0 = Fig. 12 agapanthus **baseline**
near focus, frame 1 = **CTDM-LFD** near focus (matched by mean abs. difference to the Fig. 12 panels, 1.0 vs 2.6 grey levels).
`data/generated/` is built on import by `build_assets()` (flag `done_v1`, ≈ 5 min: torch CPU + numpy) and `build_assets_v2()`
(flag `done_v2`, weight maps); file-locked.

## Version history
- v1 — first full cut, 10 scenes, 6 min 39 s (480p15). QA: far too long (captions long); S04 equation touching the panel
  labels and side labels clipped by the first panel; S05 "encoded images"/"apertures" labels running together; S06 weight maps
  too dark and row labels shown before their images; S07 Table V columns overlapping (three numbers per cell) and Fig. 10 zooms
  colliding with the source note; S01 accommodation line hidden under the screen line; toy 'learned P̂ only' panel
  contradicted the paper (21.8 vs 22.0 dB in the toy), so it was removed.
- v2 — captions shortened throughout; S01 bigger eye, "sharp"/"blur spot" readout, labelled cyan focus lines; S02 LF-display
  diagram with the display closer to the eye; S03 Fig. 2 (a) and (b) side by side; S04 equation moved up, side labels dropped;
  S06 weight maps with gamma 0.5 (`wmap2.npy`); S07 Table V as PSNR (bold) + SSIM / LPIPS (small grey); 5 min 33 s.
- v3 — S02 refocus captions merged; S07 Fig. 11 second row = the paper's zoom on the far letters (ghosting now obvious),
  Fig. 17 swept once; S08 one hold removed. 480p15: 5 min 28.7 s. Rendered 1080p60: 5 min 26.2 s.

## Global style
| Property | Value |
|---|---|
| Font | CMU Serif (text); Latin Modern Math for glyphs CMU lacks (× − → ≈ ∑ ∏ ℱ ∇ 𝒲 ℒ ⊙ ∗ ✓ ⇒ …, `MATHCH`); DejaVu Sans for ✗ |
| Math | Pango markup, no LaTeX. `sub()` rise −3000 + `<small><small>`; `sup()` rise 4500; P̂ = italic P + U+0302 (renders in CMU Serif) |
| Background | black |
| Colours | dim `#8A8A8A`, red `#FF5252` (baseline TDM-LFD / failure / VAC), green `#69F0AE` (CTDM-LFD / correct), amber `#FFCA28` (headline, rays, vergence), cyan `#4DD0E1` (apertures, accommodation), blue `#448AFF` (iris, pupils), orange `#FF9E40` (retina) |
| Apertures | 9 × 9 cells, open = cyan, closed = near-black, thin grid (`aperture_rgb`, nearest-neighbour scaling) |
| PSFs | `inferno`, sqrt intensity, each frame normalised to its 99.9th percentile |
| Text | instant (`self.add`), never animated in |
| Header | top-left, y = 3.62, size 24, dim: “k · title” |
| Caption | bottom, y = −3.5, size 28 (26 when split into 2 centred lines), max width 13.2, bottom ≥ −3.85; held ≥ len/17 + 0.8 s; `hold()` waits for it before the next visual change |
| Part card | “Part k” (30, dim) over title (50, bold, ≤ 12.5 wide), 1.5 s, instant |
| Toy notes | 16 pt dim, “toy …” under every simulated panel |
| Focus ruler | `focus_ruler`: near (star, amber) left · middle (leaf, green) · far (letters, cyan) right, white dot = current focus |

## Toy simulations (all cached in data/generated/)
**Light field** (`toy_layers`, `render_views`, numpy): 192 × 192 canvas (16-px border cropped → 160 × 160 shown), 9 × 9 views
(u, v ∈ −4…4 = the paper's l = 9 aperture grid, INRIA-syn angular resolution). Three layers composited back to front with
periodic sub-pixel FFT shifts of (u, v) × disparity: far letter wall (“LIGHT FIELD / DISPLAY / CODED / APERTURE / TIME /
DIVISION”, d = −1 px per cell), middle striped green leaf on the display plane (d = 0), near orange checkered star (d = +1).
**Image formation** (torch FFT, ray-optics limit of Eqs. 8–10): a cell (v, u) seen at focus a contributes
exp(+j2π(f_x u + f_y v)a)·sinc(f_x a)sinc(f_y a) (shifted box of width |a|).
- Ground truth (Eq. 8, all 81 views through their own cells, averaged). Baseline TDM-LFD: the 4 corner views (n = 4, paper
  Fig. 7) through the 4 corner cells, averaged (brightness-normalised).
- CTDM-LFD (Eq. 10): R_a = Σ_t I_t ∗ PSF_t,a / 81 (one fully open aperture = target brightness, so apertures must pass light).
- Toy optimisation: k = 4 images (sigmoid-parametrised, init = mean of the corner views) and 4 apertures (sigmoid of N(0, 0.5²)
  logits) fitted to the one scene with the paper's focus-weighted L1 loss (Eqs. 12–15, β = 2) over 11 focal planes
  a ∈ [−1.25, 1.25]; Adam, lr 0.03 (apertures) / 0.02 (images), 3000 iterations, seed 0. The paper instead trains a 10-block CNN f
  over a dataset for 10000 epochs (Adam, lr 0.001) — stated on screen. Snapshots at 40 log-spaced iterations; PSNR log every 10.
- Toy focal-stack PSNR (11 planes, 160 × 160): baseline 22.0 dB, learned images + apertures 30.4 dB (shown); apertures-only
  (views unchanged) 21.8 dB — not shown (the toy's fixed-brightness model penalises blurring raw views; the paper's +1.29 dB
  ablation is shown from Table VI instead).
- Display sweeps: 57 focal planes a ∈ [−1.4, 1.4] (`sweep_{gt,base,fP,P}.npy`); time-multiplex cycle at near focus
  (`cycle_cum.npy`, running sum of the 4 frames).
**Fourier-optics PSFs** (numpy, `_psf_one`): 1024² grid, 9 × 9 aperture with 24-sample cells, defocus phase
exp(jπ W20 b (ρ/R)²) averaged over b ∈ {0.85, 0.92, 1, 1.08, 1.15} (small band, incoherent light), W20 = 0…18 waves (37 steps),
|FFT|², central 768² box-averaged to 192². Apertures: open, 4 corner cells, paper Fig. 9 aperture 1.
**Weight maps** (v2, numpy): U = |∂x R| + |∂y R| of the toy GT at near / far focus, W = exp(2 Ū), shown ((W − 1)/(e² − 1))^0.5.

## Scenes (durations from the 480p15 render of v3)

### S00_Title (8.6 s)
Title (50 bold, fitted ≤ 12.8 wide, y 3.05), subtitle “with Learned Coded Aperture” (36, y 2.38). Website GIF 3.1 tall at
(0, −0.08), toggling every 1 s like the original; label switches “baseline TDM-LFD: aliased blur” (red) / “CTDM-LFD (this paper):
natural blur” (green); note “near focus, SIGGRAPH16 light field (paper Fig. 12; teaser from bchao1.github.io)”. Authors,
“National Taiwan University · IEEE Transactions on Image Processing 32 (2023)”, “* equal contribution”.

### S01_Eye (47.5 s) — “1 · How your eyes sense depth”
A. Side-view eye (circle R 1.75 at (3.3, 0.3), blue iris, lens ellipse whose thickness follows 1/f, orange retina). Thin-lens
rays (7, amber) from a star at x = −5.6 → sharp (green dot “sharp”); star moves to −1.2 → rays meet behind the retina, red
“blur spot” bar; lens accommodates → sharp again. Captions: light enters · bring it closer · accommodation, blur is a depth cue.
B. Top view: two eyes, star, amber vergence lines + cyan dashed “focus” line at the star, “real world: both at the same
distance ✓”. Stereo display: screen at y 1.9 with the two eye images (red/blue dots), virtual star at y −0.1, vergence to the star,
focus line at the screen, red bracket + “vergence-accommodation conflict (VAC)”, “eye strain · headache · even loss of depth
perception”; alternatives list (varifocal / multifocal: eye trackers or screen stacks · holographic: high-resolution SLMs ·
light field displays: none of these).

### S02_LightField (44.5 s) — “2 · Light fields and refocusing”
A. LF-display diagram: virtual point at x −5, display at 0.6, eye at 4.5; three coloured views enter one pupil and converge on the
retina (“3 views inside one pupil”). B. 9 × 9 grid of toy views (0.39 each at x −4.3), L(x, y, s, t) + legend; refocused image
(all 81 views, 3.3 tall at (2.6, 0.75)) with live focus sweep near → far → near and the focus ruler. C. “all 81 views” vs “only
the 4 corner views” (3.3 tall at x ∓2.2) with “views used” aperture icons, sweep near → far → near → middle: ghost copies.

### S03_Displays (36.1 s) — “3 · Space- vs time-division multiplexing”
A. Paper Fig. 1(b) (w 5.6) + a 3 × 3 tiled panel of toy views, “each view gets 1/9 of the panel” (paper: one-ninth for 3 × 3).
B. Paper Fig. 1(c) + live panel cycling the 4 corner views with the matching shutter window lit (“time slot k of 4”), Eq. 8.
C. Paper Fig. 2 (a) TDM-LFD / (b) CTDM-LFD refocused (2.9 tall) + ✓ full resolution / ✗ tiny window / ✗ defocus aliasing.

### S04_Aperture (22.0 s) — “4 · Blur takes the shape of the aperture”
“PSF = |ℱ{P · W(z)}|²” (y 2.85). Three live PSF panels 2.8 tall at x −4.4 / 0 / 4.4 (y 0.3): open aperture / 4 corner windows (TDM,
red) / coded aperture (paper Fig. 9, green), aperture icons below (y −1.7); “defocus: N waves” (amber); sweep 0 → 18 → 6 → 18.

### S05_CTDM (31.9 s) — “5 · Coded time-division multiplexing”
Pipeline: 4 corner views (n = 4) → amber box “encoding network f” → 4 toy encoded images (1.0, x −2.3) with their learned
apertures (x −0.95), “t = 1…4”; arrow “lens + eye” → retina image (3.0 at (3.9, 0.4)) showing the running sum after t of 4 frames
while an amber box walks down the rows; Eq. 10 (y −2.2). Then paper Fig. 3 (w 11.5): f and P̂ learned together.

### S06_Learning (49.3 s) — “6 · Learning the display end to end”
A. Eq. 12 + legend (F_c, F_g), Eqs. 13–14; toy GT at near / far focus with weight maps 𝒲 (1.9 each), “toy scene, β = 2”;
“up to e² ≈ 7.4 times more” (β = 2 from the paper; e² = 7.39 derived).
B. Toy optimisation: 4 apertures (1.3 each, 2 × 2 at x −5.55/−4.05) morphing between snapshots, near-focus retina image (3.0),
PSNR axes (0–3000 iterations, 14–34 dB) with the green curve and the red dashed toy baseline (22.0 dB); “iteration N” counter;
iteration 0 → 60 in 2.5 s, then → 2999 in 6.5 s (ease x^2.2).
C. baseline TDM-LFD (4 windows) / learned images + apertures P̂ / ground truth (81 views), 3.4 tall, live focus sweep
near → far → near → far, “toy PSNR 22.0 dB / 30.4 dB”.

### S07_Results (51.0 s) — “7 · Results”
1. Table V: 4 rows × 3 datasets, PSNR (24 pt, red baseline / green CTDM) with SSIM / LPIPS (15 pt grey) below; amber headline
   “n = 4 views: +3.7 dB (INRIA-syn) · +3.8 dB (INRIA-real) · +4.7 dB (SIGGRAPH16)”.
2. Fig. 10 INRIA-syn chairs: baseline / ours / GT (w 3.2) + zoom patches (w 1.5).
3. Fig. 11 INRIA Lytro, near focus: full images (w 2.55) + the paper's zoom on the far letters (baseline “MOM” in ghost copies).
4. Fig. 17: chairs / letters / poppy focal stacks (3.4 tall), frame-swapped through the 9 focal planes (“focal plane k of 9”).
5. Zemax: Fig. 8 relay lens, Fig. 9 binary apertures, Fig. 13 near / far focus (ours).

### S08_Ablations (24.1 s) — “8 · What matters”
Bar chart of Table VI (28.56 / 29.85 / 32.29 dB, bars grow in), captions +1.29 dB / +2.44 dB (derived). Right list (22 pt):
binary vs continuous 31.24 vs 32.29 dB (Table VII) · mirrored vs free 31.06 vs 32.29 dB (Table VIII) · β = 0 / 2 / 5:
31.25 / 32.29 / 32.07 dB (Table X) · k = 4 / 8 / 16 (n = 4): 32.29 / 33.12 / 32.43 dB (Table IX), frame rate grows with k.

### S09_Outlook (13.7 s) — “9 · Summary”
Three contributions; trade-off (more frames k need a faster panel and shutter) and “validated with Fourier-optics and Zemax
simulations”; amber “+3.7 dB over TDM-LFD with 4 views (INRIA-syn) · natural defocus blur”. End card: GIF (CTDM frame), title,
“IEEE Transactions on Image Processing 32, 350–363 (2023) · arXiv 2107.06205”, authors, NTU, * equal contribution; 4 s.

## Accuracy notes
- Paper numbers on screen: Table V (all 12 cells, 3 metrics), Table VI, VII, VIII, IX (n = 4 row), X; “3 × 3 → one-ninth of the
  panel” (Sec. III-A); β = 2 (Sec. V-C); n = 4 corner views / k = 4 (Sec. V-D, Fig. 7); l = 9 (Fig. 9); nine focal planes (Fig. 17);
  IEEE TIP 32, 350–363 (2023) (Crossref); arXiv 2107.06205. Derived: +3.7 / +3.8 / +4.7 dB (= 32.29 − 28.56, 30.21 − 26.39,
  37.57 − 32.92), +1.29 dB and +2.44 dB (Table VI differences), e² ≈ 7.4 (β = 2 in Eq. 14). “Better on every metric except one
  SSIM entry” = the paper's own statement (n = 9, SIGGRAPH16 SSIM).
- Everything else (toy PSNRs 22.0 / 30.4 dB, iterations, defocus in waves, disparities) is a parameter or output of our toy
  simulations and is labelled “toy” on screen.
- The toy is per-scene and optimises the encoded images directly; the paper learns a CNN f. The toy's brightness model (one open
  aperture = target brightness) is our choice; the paper does not state its normalisation.
- The Zemax result and the Fig. 9 apertures are binary (Zemax cannot simulate continuous apertures, Sec. V-F); Table V uses
  continuous apertures.
- The paper reports no physical prototype and no runtimes; the video says “validated with Fourier-optics and Zemax simulations”.
