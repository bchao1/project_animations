# Spectral Progressive Diffusion for Efficient Image and Video Generation — ingest summary

- **Authors**: Howard Xiao, Brian Chao, Lior Yariv, Gordon Wetzstein — Stanford University
- **arXiv**: 2605.18736 (v3, 17 Aug 2026) · project: https://howardxiao.ca/speed/
- **Sources ingested**: `pdfs/xiao2026spectral.pdf` (51 pages), `pdfs/xiao2026spectral.txt` (PyMuPDF text),
  `pdfs/src/` (arXiv LaTeX: `sections/*.tex`, `tables/*.tex`, `figures/`, `supp_figures/`).

## Problem
DiT self-attention cost grows quadratically with token count, and resolution / video length keep growing.
Diffusion models generate "autoregressively in frequency" (Rissanen et al. 2023; Dieleman 2024): low
frequencies are resolved early, high frequencies only late. So during early steps most of the frequencies a
high-resolution grid can represent are pure noise, and computing them at full resolution is redundant.
Prior progressive-resolution methods either need new architectures / training from scratch (Pyramidal Flow,
Scale-space diffusion, …) or depend on heuristics for when/how to upsample (Bottleneck Sampling, RALU, …).

## Preliminaries
- Flow matching: **x_t = (1 − t) x_0 + t ε**, t ∈ [0, 1], ε ~ N(0, I); network v_θ(x_t, t) predicts ε − x_0,
  loss 𝔼‖v_θ(x_t, t) − (ε − x_0)‖². Sampling integrates the probability-flow ODE from t = 1 to t = 0.
- Power spectrum under an orthonormal spectral basis Φ = {φ_ω} (Fourier / DCT / Haar):
  **P_ω = 𝔼|⟨x_0, φ_ω⟩|²**; natural images: **P_ω ∝ |ω|^(−β)**, β typically in [2, 3].
  Gaussian noise has a flat spectrum, so high frequencies are noise-dominated early.
- Per-frequency forward process: x_t^(ω) = (1 − t) x_0^(ω) + t ε^(ω), ε^(ω) ~ N(0, 1) i.i.d. (orthonormality).
- Per-frequency SNR (App. A.1): **SNR_ω(t) = (1 − t)²/t² · P_ω**.
- Nyquist: an H × W grid represents radial frequency up to ω_max(H, W) = min(H, W)/2; stage at scale s
  represents Ω_s = {|ω| ≤ s · ω_max}.

## Method
### Stages
S scales 0 < s_1 < … < s_S = 1 and transition times 1 > t_1 > … > t_{S−1} > 0; stage i runs at (s_i H, s_i W)
for t ∈ (t_i, t_{i−1}].

### Spectral noise expansion (Sec. 3.1, Fig. 1 right)
At transition time t_i:
1. ξ = T_Φ(x_{t_i}^{s_i}) — spectrum on Ω_{s_i}.
2. Embed ξ in the low-frequency part of the Ω_{s_{i+1}} spectrum; fill the new slots Ω_{s_{i+1}} \ Ω_{s_i} with **t_i ε**, ε ~ N(0, 1).
3. x_{t_i}^{s_{i+1}} = T_Φ^{−1}(ξ′) at resolution (s_{i+1} H, s_{i+1} W).

### Timestep alignment
Padding raises the overall noise level, so rescale and re-index (r = s_{i+1}/s_i):
- **x̃ = r / (1 + (r − 1) t_i) · x_{t_i}^{s_{i+1}}**
- **t̃_i = r t_i / (1 + (r − 1) t_i)**
Derivation (App. A.2): clean coefficients at shared frequencies scale by r across resolutions
(orthonormal identification); match noise coefficient κ t_i = t̃ and signal κ(1 − t_i) = r(1 − t̃).
Example (our arithmetic): r = 2, t_i = 0.5 → t̃ = 2/3, κ = 4/3.
Default T_Φ = DCT (real-valued, supports non-power-of-two scaling); DWT, FFT perform similarly (ablation).

### δ-optimal resolution schedule (Sec. 3.2)
- **Prop. 1** (per-frequency activation time): with x_0^(ω) ~ N(0, P_ω), for all t ≥ t_ω,
  𝔼|v*^(ω)(x_t, t) − ε^(ω)|² ≤ δ, where
  **t_ω = 1 / (1 + √(δ / (P_ω (1 + P_ω − δ))))**.
  Lemma: 𝔼|v* − ε|² = SNR_ω(1 + P_ω)/(1 + SNR_ω). Intuition: for t ≥ t_ω the optimal velocity at ω is ≈ the noise.
- **Prop. 2** (per-resolution transition time): P_ω monotone decreasing ⇒ t_ω decreasing in |ω| ⇒
  **t_i* = min_{ω ∈ Ω_{s_i}} t_ω = t_{ω = s_i · ω_max(H, W)}**.
- Single hyperparameter δ, independent of S (prior methods: S = 2 or hand-tuned t's). Default δ = 0.01.
- Measured power-law fits (Table 4, P_ω ≈ A|ω|^(−β), centered samples):
  FLUX.1-dev (= Z-Image VAE) latent β = 1.9155, A = 203.62, R² = 0.978 (100K images);
  WAN latent β = 2.4227, A = 219.48, R² = 0.999 (10K videos);
  Pixel 512² β = 2.4493, A = 3745.46, R² = 1.000 (100K images).
  Fig. 2(a) labels: P_ω ∝ ω^−1.92 (FLUX image latents), ω^−2.42 (Wan video latents).
- Empirical support (App. B, Fig. 5 "spectral noise passthrough"): keep full resolution at every step, but for
  each frequency whose activation time has not been reached (t > t_ω) replace the component by t · T_Φ(ε)^(ω)
  (ε = the initial noise at t = 1). δ ∈ [0.0001, 0.001]: almost no visible difference;
  clear transition to blurry images around δ = 0.05 ("CHOOLBUS" instead of "SCHOOLBUS").

### Spectral-transformation-based fine-tuning (Sec. 3.3)
Training-free use assumes the model works at all scales s_i, and the schedule may not match training dynamics.
Stage i: straight path from the expanded, aligned x̃_{t̃_{i−1}}^{s_i} to the standard FM state
x_{t_i}^{s_i} = (1 − t_i) x_0^{s_i} + t_i ε^{s_i}; target **v^{s_i} = (x̃_{t̃_{i−1}}^{s_i} − x_{t_i}^{s_i}) / (t̃_{i−1} − t_i)**;
sample x_t^{s_i} = x_{t_i}^{s_i} + (t − t_i) v^{s_i}; loss 𝔼‖v_θ(x_t^{s_i}, t) − v^{s_i}‖². Same ε^{s_i} used for both ends.
LoRA rank 32, 2000 iterations, 5K synthetic images from the base model (MS-COCO prompts disjoint from eval),
8 × H100, effective batch 8, ≈ 2 h. PixelGen (x_0-prediction) supervised with converted x_0 targets.

## Experimental setup
- Models: FLUX.1-dev (training-free, 1024²), Z-Image (fine-tuning, 1024²), PixelGen-XXL/16 T2I (pixel, 512²),
  WAN 2.1-T2V-1.3B (video, 720P). δ = 0.01, DCT throughout.
- FLUX scales: S = 2 → latent 64², 128² (pixels 512², 1024²); S = 3 → latent 32², 64², 128² (256², 512², 1024²).
- Metrics: ImageReward, CLIP-IQA, NIQE, T2I-CompBench, GenEval (RALU protocol; 5,000 MS-COCO prompts); VBench for video (200 prompts).
  Speedups = wall-clock vs 50-step FLUX on A100 (FLUX), vs H100 baselines elsewhere.

## Key numbers
**Table 1 — FLUX.1-dev training-free (speedup / TFLOPs / ImageReward):**
| Tier | Method | Speedup | TFLOPs | ImageReward |
|---|---|---|---|---|
| 1× | FLUX 50 steps | 1.00× | 2991.01 | 1.095 |
| | RALU | 1.58× | 1749.94 | 1.028 |
| | Ours S=2 | 1.66× | 1755.22 | 1.049 |
| 5× | FLUX 10 steps | 4.84× | 610.02 | 0.981 |
| | Bottleneck | 4.67× | 571.23 | 0.889 |
| | RALU | 4.98× | 540.47 | 1.022 |
| | Ours S=2 | 5.77× | 500.34 | 1.059 |
| | Ours S=3 | 6.09× | 469.15 | 1.042 |
| 7× | FLUX 7 steps | 6.62× | 431.45 | 0.920 |
| | Bottleneck | 6.64× | 431.52 | 0.792 |
| | RALU | 6.69× | 426.01 | 0.999 |
| | Ours S=2 | 6.78× | 427.03 | 1.039 |
| | Ours S=3 | 7.09× | 406.24 | 1.015 |
Text: up to 7.36× FLOPs speedup and 7.09× wall-clock speedup (S=3).

**Table 2 — fine-tuning (speedup / ImageReward):** Z-Image 50 steps 1.00× 0.965; TF S=2 1.65× 0.904; LoRA S=2 1.65× 0.982;
TF S=3 1.74× 0.875; LoRA S=3 1.74× 0.954. PixelGen 25 steps 1.00× 0.921; TF S=2 1.60× 0.799; LoRA S=2 1.55× 0.913.
Table 6 extras: Z-Image LoRA δ=0.05 S=2 2.01×; 7-step tier LoRA S=2 7.81× (IR 0.918), TF S=3 8.08×.

**Table 3 — WAN 2.1 720P (VBench):** 50 steps 1.00× (Subject 0.9492, Background 0.9621, Motion 0.9874, Dynamic 0.4800,
Aesthetic 0.5993, Image quality 0.6133); 25 steps 2.00× (0.9434, 0.9597, 0.9879, 0.4250, 0.5893, 0.5706);
Ours S=2 2.03× (0.9462, 0.9598, 0.9859, 0.4950, 0.5975, 0.6114); Ours S=3 2.54× (0.9299, 0.9499, 0.9815, 0.5700, 0.5696, 0.5990).
Headline in intro: "speedups of up to 7× on image generation and 2.5× on video generation".

**Ablations (Table 8, FLUX 7-step baseline, H100):** increasing δ or S → faster, gradual quality loss; DCT ≈ DWT ≈ FFT.
e.g. DCT S=2: δ=0.001 1.00× IR 1.032; δ=0.01 1.19× 1.039; δ=0.1 1.62× 0.966. S=2 gives most of the speedup; higher S diminishing.

**Application**: frequency-based image editing (noise the low-frequency part of T_Φ(x_in), expand, resume with a new prompt)
beats SDEdit on texture/style edits (Fig. 4b, "Minecraft", "ceramic").

## Limitations (Sec. 6)
Training-free use assumes the model generalizes across resolutions; continuous schedules may hit resolutions unseen in
training (few transitions near the training distribution suffice). Fine-tuning reduces the gap; pretraining with the method
is a promising direction. Temporal frequencies for video are future work.

## Figures
- Fig. 1 — pipeline: timeline 1 → t1 → t2 → t3 → 0 with s = 0.25/0.5/0.75/1, corgi denoising, P(ω) ∝ ω^−β curve split into
  stages; right: spectral noise expansion (x^{s_i}_{t_i} → T_Φ → padded spectrum with t_i ε → T_Φ^{-1} → x^{s_{i+1}}_{t_i}).
- Fig. 2 — (a) latent power spectra (FLUX image, Wan video) with fits ω^−1.92, ω^−2.42; (b) DWT sub-bands of latents/pixels over denoising steps.
- Fig. 3 — qualitative: FLUX (1×) / RALU (1.58×) / Ours (1.66×) / FLUX (4.84×) / RALU (4.98×) / Ours (5.77×); PixelGen (1×) vs Ours (1.6×);
  Wan (1×) / Wan (2.0×) / Ours (2.03×) video frames (coral reef, waterfall).
- Fig. 4 — (a) ImageReward vs δ for S = 2/3/4 and DCT/DWT/FFT; (b) frequency-based editing vs SDEdit.
- Fig. 5 — spectral noise passthrough over δ ∈ {0.0001, 0.001, 0.01, 0.05, 0.1}.
- Figs. 6–8 FLUX, 9–11 Z-Image (fine-tuned, Ours*), 12–14 PixelGen qualitative; 15–17 ablations; 18–23 editing.
- Tables 1–8 as above.

## Crops used by the animation (in ../data/)
See `storyboard.md` of the project; all images come from embedded rasters of `figures/main_qual.pdf`,
`supp_figures/spectral_passthrough.pdf`, `supp_figures/flux_supp_1.pdf`, `figures/ablation_freq_edit.pdf`.
