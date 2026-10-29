# Storyboard — Large Étendue 3D Holography explainer (3b1b style)

**Current version: v3** — `animation_large_etendue_v3.py`
Render: `./render.sh l` (480p15 test) · `./render.sh h` (1080p60 final). Env: `imp_tome` conda env with
`LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7` (baked into `render.sh`; imp_tome ships only libffi 8).
Final cut: `media/final/LargeEtendueHolography_animation_large_etendue_v3_1080p60.mp4` (10 scenes, ≈ 5 min 22 s).

Source: Chao, Gopakumar, Choi, Kim, Shi, Wetzstein, *Large Étendue 3D Holographic Display with Content-adaptive
Dynamic Fourier Modulation*, SIGGRAPH Asia 2024, arXiv 2409.03143. PDF, extracted text and LaTeX source:
`large-etendue-holography/pdfs/`. Summary with every number used: `large-etendue-holography/summaries/chao2024etendue.md`.

Assets
- `data/*.png` — paper figures: raster panels extracted from the arXiv figure PDFs (teaser spectra/views without the
  vector pupil circles, Fig. 6 / Fig. 7 rasters, raw prototype photo) and rasterized Figs. 2–5 (`baseline.png`, `hw_setup.png`, …).
- `data/generated/toy_cgh.npz` — `precompute_toy_cgh.py` (≈ 2 min on one GPU): toy re-implementation of the paper's
  image formation (64 × 64 phase SLM sampled 4× for HDOs, 3 × 3 incoherent sources shifted by one order cell,
  20 × 20 Fourier amplitude mask, 9 × 9 pupils of radius 0.2345 × eyebox, stochastic one-view loss, Adam, 3000 its)
  for configs I, III, IV, V (1 frame) and V (6 frames). Only used for illustrative visuals, always labelled
  “toy simulation”. Its own PSNRs (12.0 / 16.6 / 17.5 / 19.1 / 25.0 dB) are **not** shown in the video.
- `data/generated/*.png` — built by `build_assets()` on import (delete `data/generated/done_v1` to rebuild).

## Version history
- v1 — first full cut, 10 scenes, ≈ 5 min 57 s.
- v2 — title fits frame; S01 wave panel smaller/left, narrow aperture in grating mode so the ±θ beams separate,
  intensity-weighted brightness, thicker θ guides; S02 equations scaled to ≤ 6.4 wide, FoV label off the wedge;
  S04 equation no longer clipped, “cell = λf/p” left of the spectrum, filter no longer hides the box; S06 diagram
  spacing (Fourier-lens label below), view panel shown from the start, Jo et al. label split; captions shortened,
  title cards 1.8 s; ≈ 5 min 22 s.
- v3 — `Base.wait` renders real frames (frozen_frame=False) so ValueTracker-driven images/labels update during
  waits (S06 masks were frozen in v2); S02 curve note moved inside the plot (two lines).

## Global style
| Property | Value |
|---|---|
| Font | CMU Serif; Latin Modern Math for × → ≈ − ∈ ∼ ℱ ℋ 𝒫 𝒰 (via `mathify`) |
| Math | Pango markup, no LaTeX. Subscript `rise=-3000` + `<small><small>`; superscript `rise=4500` |
| Background | black |
| Light / spectra | laser green `#39E27D`, colormap `gmap` (black → green → white) |
| Other colours | amber `#FFCA28` (eyebox, amplitude SLM, θ guides), lens cyan `#4DD0E1`, SLM grey `#90A4AE`, red `#FF5252` / blue `#448AFF` (pupils, as in paper Fig. 1), purple `#E040FB`, dim `#8A8A8A`, ghost `#3A3A3A`; source tints `#FFB74D`, green, `#4FC3F7` |
| Text | instant (`self.add`), never animated in |
| Header | top-left, y = 3.62, size 24, dim: “k · title” |
| Caption | y = −3.5, size 28 (26 when split into 2 centred lines), max width 13.2; held ≥ len/15 + 1 s |
| Part card | “Part k” (30, dim) over title (50, bold), 1.8 s |
| Toy visuals | always tagged “toy simulation” (18, dim) |

## Scenes (durations from the 480p15 render)

### S00_Title (9.3 s)
Title (54 bold, ≤ 12.8 wide, y 2.85), subtitle (38, y 2.1); computed Fourier plane (3.0 tall, y −0.1) whose 9 shifted
spectrum copies appear one by one (`ms_1…ms_9`, 4 s); authors (24, y −2.35), “Stanford University · NVIDIA · MIT”, “SIGGRAPH Asia 2024”.

### S01_Diffraction (40 s) — “1 · Diffraction and pixel pitch”
Live `WaveField` (3.7 tall at (−2.95, 0.25); 300 × 620 array, λ = 6 px; plane wave left of the SLM, 1-D angular-spectrum
propagation to the right, animated in time). SLM pixels drawn from the actual phase (`always_redraw`).
1. Lens mode: per-pixel quantized lens phase focuses light to “a 3D point” (focus 260 px).
2. Grating mode (aperture 0.13 H): alternating pixels (period 2p) → two beams; `sin θ = λ / 2p` (46, x 4.15),
   amber dashed guides at ±θ drawn from the formula lie on the computed beams; readout “p = … px θ = …°”
   (simulation units); p animates 10 → 6.5 → 16 → 10.
3. “real SLM: p = 10.8 µm, λ = 632.8 nm → θ ≈ ±1.7°” (paper Fig. 3 parameters, θ = asin(λ/2p) = 1.68°).

### S02_Etendue (34 s) — “2 · Étendue”
Left schematic (SLM at x −6.4, eyepiece at distance G = 0.8 + 0.22 w, eyebox at 2G, green light polygon, amber eyebox bar
∝ w, dashed FoV wedge with the true FoV(w)). Right: equations G = 4A sin²θ · G_SLM = 4LxLy sin²θ = λ²NxNy ·
FoV = 2 tan⁻¹(Np/2g), w = gλ/p (x 3.75, y 2.65/1.95/1.3). Axes 5.0 × 2.75 at (3.85, −1.15), eyebox 0–10 mm, FoV 0–40°,
α = 1 curve FoV(w) = 2 atan(Nλ/2w) with N = 1000, λ = 632.8 nm (matches paper Fig. 3); amber rectangle (0,0)–(w, FoV)
driven by the same tracker w: 2 → 7.5 → 1.1 → 3 mm. Caption: 1080p SLM ≈ 80° FoV with < 1 mm eyebox (paper, Related Work).

### S03_Prior (25 s) — “3 · Stretching étendue: prior approaches”
Paper Fig. 4 (13.4 wide, y 0.55) with boxes: II red “static mask → low contrast, speckle”; III+IV amber “all sources at
once → copies, few degrees of freedom”; VI+VII purple “steered → needs very fast SLMs”. Caption on pupil replication.

### S04_FourierLens (43 s) — “4 · The Fourier plane is the eyebox”
Ray diagram: SLM (x −6.2), lens (x −3.9), Fourier plane (x −1.6); three parallel bundles (0, +0.20, −0.32 rad) converge
to points; “direction θ → point u = f tan θ ≈ λf · f_x”. Computed |ℱ{pixelated SLM field}|² (toy I, raw, log; 3.9 at
(3.7, 0.25)) with 4 × 4 order-cell grid, white central cell, “cell = λf/p”, HDO label; Fourier filter fades in.
Then captured single-source spectrum (paper Fig. 1b raster, 4.8 wide) + view; pupil circle (r = 0.188 W) moves from the
red position (0.506 W) to the blue position (0.748 W) as in the paper figure, view cross-fades center → right (Fig. 1c, d).

### S05_MultiSource (40 s) — “5 · Multi-source illumination”
Side view: 3 sources (x −6.4), collimator (x −5.0), phase SLM (x −3.3) with tinted tilted beams; Eq. 2 second line
𝒰_slm^(j) = ℱ{e^{iφ} u_src^(j) e^{i k^(j)·x}}, “a tilt in space = a shift in frequency”; 3 × 3 source icon lights up as
the computed Fourier plane (4.3 at (3.6, 0.2)) accumulates the 9 shifted copies; amber expanded-eyebox square (3 cells).
Captions: tilts matched to ±1st orders tile copies; eyebox 3× per axis, étendue α² = 9×.
Then toy target view vs toy “multi-source, no Fourier filter” view (3.3 each at x ±3.4).

### S06_FourierMod (31 s) — “6 · A second SLM in the Fourier plane”
Top diagram (y 2.0): sources → collimator → phase SLM → Fourier lens → amplitude SLM → eye; right notes: Fourier plane =
eyebox plane, 20 × 20 Fourier display (simulation), 1080p LCoS (prototype). Panels (2.35, y −0.65, x −5.0/−1.75/1.5/4.9):
spectrum of all 9 sources × mask 𝒫^(t) = modulated spectrum → 6-frame toy view; masks cycle t = 1…6 (0.55 s / 0.45 s).
Eq. f^(j)(φ, 𝒫, z) = ℱ⁻¹{𝒰_slm^(j) · ℋ(z) · 𝒫} (y −2.55). Contrast: fixed random binary mask (Jo et al. 2022) + its toy view, then back to ours.

### S07_Optimization (33 s) — “7 · Computer-generated holography”
Eq. 3 (28, y 2.75, ≤ 13.4 wide). Eyebox square 3.7 at (−4.2, −0.2) over dim toy spectrum, 9 × 9 pupil dots, blue pupil
circle (4 mm of 8.53 mm → r = 0.2345 E). Target view p vs hologram through pupil p (toy, 6 frames; 2.6 each at x 0.5 / 3.95).
7 random pupil jumps (stochastic single-view supervision), then a continuous sweep along the middle row (25 views) showing parallax.

### S08_Results (50 s)
1. “8 · Prototype”: Fig. 5 photo (8.6 wide at (−2.35, 0.35)) + specs: 3 × 3 fiber tips 8.17 mm apart · 200 mm collimator
   → 2.34° · TI DLP phase SLM (4-bit) · 1080p LCoS amplitude SLM · 75 mm FT lens · diagonal FoV 7.78° · eyebox 8.53 × 8.53 mm.
2. “Captured: single source vs. ours”: Fig. 1 b–g rasters (2.45 tall), red/blue pupils; blue pupils move to the right position.
3. Table 1 bars (PSNR × 0.22 from x −1.4): I 16.60/0.23, II 14.61/0.15, III 13.35/0.24, IV 18.63/0.26, VI 20.02/0.43 (9 fr),
   VII 23.95/0.51 (9 fr), ours 21.05/0.39 (1 fr), 24.65/0.66 (6 fr).
4. Fig. 6 crops (center + right view) for I, III, IV, VII, ours-6 with the figure's per-scene PSNR (16.13, 11.28, 18.27, 25.07, 26.01 dB).
5. Fig. 7 crops (front focus + right-view insets) for I, III, IV, ours-1, ours-6 with Table 2: 12.72/0.19, 12.46/0.17, 13.83/0.23, 14.43/0.40.

### S09_Outlook (18 s) — “9 · Limitations and outlook”
Bullets: benchtop → nanophotonic phased arrays, waveguides · 240 Hz amplitude display → ∼13.33 Hz color-sequential 6-frame ·
LCoS > 720 Hz · no real-time synthesis yet. End card (no header): title, 9-copy spectrum, authors,
“SIGGRAPH Asia 2024 · arXiv 2409.03143 · bchao1.github.io/holo_dfm”.
