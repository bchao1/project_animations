# Storyboard — Spectral Progressive Diffusion, 1-minute teaser

**Current version: v5** — `animation_spectral_progressive_diffusion_1min_v5.py`
Final cut: `media/final/SpectralProgressiveDiffusion_1min_v5_1080p60.mp4` — 63.2 s, 1920×1080, 60 fps
(beat durations below are from the v5 1080p60 render).
Render: `./render.sh l` (480p15) · `./render.sh h` (1080p60). Final: `media/final/SpectralProgressiveDiffusion_1min_vN_<res>.mp4`.

Condensed from the 5-min teaser (`../animation_spectral_progressive_v4.py`, `../storyboard.md`); same style
(CMU Serif, Pango markup, black background, signal blue / noise red / frontier yellow, stage colours of paper Fig. 1).
No headers, no part cards. Captions at y = −3.5, held ≥ len/15 + 1 s. Text appears instantly.

**Takeaway:** diffusion resolves low frequencies first and high frequencies last, so early high-resolution compute is
spent on noise; grow the resolution along the denoising path (spectral noise expansion + a schedule from the power
spectrum) → up to 7× faster images and 2.5× faster video while preserving quality.

## Numbers on screen (all from the paper, arXiv 2605.18736)
- 7.09× wall-clock (Ours S = 3, FLUX.1-dev 1024², Table 1); scatter points of Table 1 (speedup, ImageReward).
- Fig. 3 labels: FLUX 1×, FLUX 4.84× (10 steps), RALU 4.98×, Ours 5.77×.
- "up to 7× faster images · 2.5× faster video" (intro claim, Sec. 1).
- Schedule timeline: s = 1/4, 1/2, 1 (1/16, 1/4, all tokens) — paper's S = 3 scales; switch times from Eq. 9/10
  with the paper's pixel fit (A = 3745.46, β = 2.4493, Table 4), δ = 0.01 (paper default). No times printed.
- Corgi demo spectra/sweeps are computed (as in the 5-min version); no numbers from them are shown.

## Version history
- v1 — first cut from the plan (60.5 s).
- v2 — T1 “denoising” label moved to the right end (was clipped by the red box); T2 0.6 s hold on the clean image +
  “yellow: signal = noise” label; T4 Fig. 3 labels on two lines (name / speed) — they overlapped.
- v3 — T3 schedule and T4 scatter held 1.2 s / 1.0 s past the caption reading time; Fig. 3 column pitch 2.5.
- v4 — T1 token grids lowered (y −1.65) so “denoising” clears them.
- v5 — T1 caption made exact: “Early on, that compute is spent mostly on noise.” (t = 0.8 is not pure noise).

## Beats (v5: 63.2 s)

### T1_Hook (9.5 s) — title + the problem
Title “Spectral Progressive Diffusion” (54 bold, y 3.0) + subtitle (28, y 2.35) stay up.
Row of x_t thumbnails t = 1.0 … 0.0 (1.6 each, y 0.55, labels above), grey “denoising” arrow (y −0.55),
12×12 token grids under each (1.2, y −1.65), “denoising” under the arrow’s right end. Caption 1: “Every denoising step pays for the full-resolution grid.”
Red box around the t = 1.0, 0.8 columns. Caption 2: “Early on, that compute is spent mostly on noise.”

### T2_Insight (12.0 s) — noise drowns the high frequencies first (from S03)
x_t (3.0 at (−5.3, 0.4)) with t readout, its FFT (3.0 at (−1.55, 0.4)) with the yellow frontier circle, log-log
axes (4.6 × 3.4 at (4.0, 0.4)): signal (1 − t)² P_ω (blue), noise t² (red), yellow crossing dot, red shading.
“yellow: signal = noise” under the spectrum. Hold 0.6 s, then t: 0 → 0.97 in 3 s. Caption: “Noise drowns the high frequencies first.”
Then run the sampler: frontier log ω* → 256 in 6 s. Caption: “Denoising resolves coarse structure first, fine detail last.”

### T3_Method (19.4 s) — grow the grid (from S04, S05, S06)
A. Fourier plane (3.4 at (−3.3, 0.45)) with dashed Nyquist squares, solid current-grid square, frontier circle;
   pixel panel (3.4 at (2.6, 0.45)) at the current grid, “n × n grid” above, “1/256 … all of the pixels” below.
   Sweep 6.5 s (32 → 512). Caption: “So grow the grid as the frontier expands.”
B. Clear. Top row (y 1.35): x_{t_i} 64² (1.9) → 128² DCT canvas (2.4; low corner = DCT of x_{t_i}, new slots
   labelled t_i ε) → x at 128² (2.4). Caption: “Spectral noise expansion: pad new frequencies with noise.”
C. Same screen, bottom: stage timeline bar (y −1.35) t = 1 → t₁* → t₂* → t = 0 with s₁ = 1/4, s₂ = 1/2, s₃ = 1 and
   token fractions; dim note “leave a scale once its highest frequency rises above the noise”. Kept equation-free.
   Caption: “When to grow: read off the power spectrum.” Held 1.2 s past reading time.

### T4_Payoff (22.4 s) — results + end card (from S08, S09)
A. Scatter speedup vs ImageReward (Table 1), FLUX fewer-steps dashed, Bottleneck, RALU, Ours S = 2/3 with
   “7.09×” label. Caption: “Best quality at every speed tier, up to 7.09× faster.”
   Held 1.0 s past reading time.
B. Fig. 3 grid: 4 columns (FLUX, 50 steps 1× · FLUX, 10 steps 4.84× · RALU 4.98× · Ours 5.77×; two-line labels)
   × buddha / canyon, 2.2 each, pitch 2.5; yellow frame around Ours; source note “paper Fig. 3 · FLUX.1-dev, 1024²”.
   Caption: “Faster than the baselines, with quality preserved.”
C. Headline “up to 7× faster images · 2.5× faster video” (40 bold yellow), “training-free on pretrained models, or
   fine-tuned with a LoRA”, “FLUX.1-dev · Z-Image · PixelGen · Wan 2.1” (3.8 s).
D. End card (5 s): title, “arXiv 2605.18736 · howardxiao.ca/speed”, authors · Stanford University.
