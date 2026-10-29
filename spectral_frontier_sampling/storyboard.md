# Storyboard: Spectral Frontier Sampling (SFS), 3b1b style

**Current version: v2**: `animation_sfs_v2.py`. Two videos:
- Teaser (about 70 s): scenes `T00`…`T04`.
- Method video (about 3–4 min): scenes `M00`…`M07`.

Render: `./render.sh l teaser` or `./render.sh h method` (see `render.sh`).

Source: the paper draft `prompt-adaptive-speed-cvpr-2027/submission/adaptive-speed-cvpr2027/` (commit 1a85ba2,
"Reframe as Spectral Frontier Sampling"), sections 1 and 3. Data come from the research repo
(`experiments/fractional_expansion/`, findings 2026-10-13 fractional-expansion-envelope and 2026-10-15
spectrogram-motivation). Planning decisions: the grilling session of 2026-10-15 (Q1–Q17, below).

## Decisions (grilling, 2026-10-15)
- Two videos (teaser + method), with section markers for slide export later.
- 3b1b style of the SPEED animation (`../spectral_progressive_diffusion`): black background, CMU Serif, signal blue,
  noise red, ours yellow. SPEED staircases are dashed grey, ours is solid yellow.
- Spectrogram: x = denoising step (0–49), y = radial frequency bin, compute-spaced (area under a staircase = its
  FLOPs). Color = log10 power of x̂₀ of the full-res run (magma, black = no signal yet).
- Teaser part 1: SPEED 2-stage → SPEED 3-stage → ours. Waste = the area inside a staircase where x̂₀ has no power yet.
- Teaser part 2: forest → kitchen (early) → barn (late). Only the spectrogram, our staircase, a faint fixed SPEED-3
  staircase, and a thumbnail.
- Numbers: paper headline numbers (1.7× vs 3-stage, 4.7–5.0× vs full res, HPSv3 −0.19) and one-prompt compute
  numbers labelled "one prompt".
- Short simple-English captions, `CAPTIONS` switch.
- Method video: short primer, then all six estimation parts in paper order, then frontier tracking with the 2× cap.
  Real data for the estimation and tracking scenes. At most one short formula per idea.
- Round 3 (after the oracle check, finding `2026-10-15_online-vs-oracle-first-expansion`): the kitchen (online 16,
  oracle 26) and barn (online 38, oracle 31) picks were estimator noise, not content. Per-sample examples are now
  forest → busy street 2348 → border collie 1026 (online and oracle agree on the direction). Running example of the
  method video: busy street 2348 (first fire at step 26; estimate 0.39 ± 0.25 vs true 0.36 log10 at bin 32). The
  true final power is drawn as a faint dashed check curve.
- Thumbnails next to a spectrogram show the full-res image (the spectrograms are full-res runs). The full-res forest
  is letterboxed; that is the real model output.
- No author names (anonymous submission). Closing card has SPEED-3 vs ours image pairs.

## Prompts (klein 2K, Aesthetic-4K eval, N = 50, CFG 4)
| idx | short name | first expansion (ours) | compute speedup: SPEED-2 / SPEED-3 / ours |
|---|---|---|---|
| 002580 | forest sunset | step 32 | 2.38× / 2.88× / 6.07× |
| 002348 | busy street | step 27 (oracle 24) | 2.38× / 2.88× / 3.80× |
| 001026 | border collie | step 36 (oracle 33) | 2.38× / 2.88× / 7.19× |
| 000577 | kitchen (not used: overestimate) | step 16 (oracle 26) | 2.38× / 2.88× / 3.90× |

SPEED schedules are the same for all prompts: 2-stage 128 → 256 at step 36; 3-stage 64 → 128 → 256 at steps 27, 36.

## Data (`data/`)
- `sfs_data.npz` (`scratch/prep_data.py`): per prompt `p<idx>_{P_x0, P_final, sigmas, speed2, speed3, cont2x, *_x}`
  and `compute_of_bin`.
- `method_<idx>.npz` (`scratch/prep_method.py`): every ContRule intermediate per step (ring C, rel, ok, VAE-mapped
  values, estimate and sd at every candidate boundary, u, g, fire, next), checked against the rule's trace.
- `final_{fullres,speed3,cont2x}_<idx>.png`: 768 px copies of the stored 2K results.
- `capture/<src>_<idx>/` (`scratch/capture.py`, GPU): per-step x_t and x̂₀ PNGs (512 px), `steps.npz` (ring stats
  per step on the current grid), `trace.json` (the rule's own per-step record).

## Teaser
### T00_Title (3 s)
"Spectral Frontier Sampling", subtitle "Online, per-sample resolution for diffusion models".

### T01_Spectrogram (about 15 s)
Left: x̂₀ of the full-res forest run sharpens step by step. Right: the spectrogram fills one column per step.
Captions: "Diffusion makes an image from coarse to fine." / "Each column shows the power at each frequency, step by
step." / "Black means no signal yet. Fine detail appears only near the end."

### T02_Waste (about 25 s)
1. SPEED 2-stage staircase. The black area under it fills with red hatching. "SPEED computes these frequencies, but
   they hold no signal yet." Label "2.4× faster (compute, one prompt)".
2. SPEED 3-stage. Less red. "More stages waste less. But each stage is a hand-picked choice." "2.9×".
3. Ours (yellow) follows the edge of the colored region. Almost no red. "Our grid grows with the frontier: the
   highest frequency with signal." "6.1×".

### T03_PerSample (about 20 s)
Boxes clear, a faint SPEED-3 stays. The spectrogram morphs forest → kitchen (the yellow curve moves left) → barn
(moves right). Thumbnail of each final image. "A detailed image needs high frequencies early." / "A smooth image can
stay small longer." / "SPEED uses one schedule for all images. Ours finds one per image, while it samples."

### T04_End (5 s)
"1.7× faster than a 3-stage schedule · 4.7–5.0× faster than full resolution" (FLUX.2 [klein], 2048²).

## Method video
### M00_Title
### M01_Primer: the frontier
Kitchen P_final on log-log axes (signal power (1−t)²P). Noise power t² is a flat line. As t drops, the noise line
drops, and the crossing (frontier) slides right. Then a vertical slice of the spectrogram is this spectrum.
Formula: SNR = (1−t)² P / t².
### M02_Grid: a grid keeps only low frequencies
Token grid of side n keeps frequencies up to n/2. Two decisions: when to grow, and to which size.
### M03_Problem: the needed power is unknown
Three P_final curves (kitchen, forest, barn) fan out. The dataset mean (SPEED) mistimes both. The boundary band is
still noise on the coarse grid.
### M04_Estimate: online power estimation (kitchen, grid 64, real ring stats)
1. Clean estimate x̂₀ = x_t − t·v (x_t and x̂₀ images).
2. Measure power on low rings 2–12 (C, dots with error bars). Unreliable rings fade.
3. VAE map: coarse dots shift to the full-res scale.
4. Correlation across frequencies: the estimate at every target frequency, with an error band.
5. Residual: only the part the coarse grid cannot predict (u) is tested.
6. Margin: test the lower edge of the band (z = 0.5).
### M05_Rule: when to grow
Real trace: the threshold line drops step by step. The rule fires when the boundary passes the test.
### M06_Tracking: to which size
Candidates every 8 latent pixels. The grid grows to the first candidate that fails. New frequencies are filled with
noise. The 2× cap example.
### M07_Loop: close the loop + results
Repeating M04–M06 every step draws the full staircase on the spectrogram. Results card + image pairs.

## v2 changes (user feedback, 2026-10-15)
- All spectrograms show log10 SNR of x_t, (1 − σ)² P_final / σ² (model-free), in magma, range 1e-4 … 100. The colour
  bar marks 0.01. A white dashed contour marks where P_final passes SPEED's δ = 0.01 test, P ≥ P*(σ_{k+1}) ("signal
  emerges"). Reason: on the x̂₀-power plot the staircases often sat below the coloured edge.
- SPEED staircases are solid grey, so the dashed white line is only the δ contour.
- T01: image x_t (real decode, full-res run) on top, its 2D spectrum (2D log SNR from the final latent's 2D power,
  `P2D_final` in `capture/full_<idx>/steps.npz`) below, labels "image", "spectrum", "spectrogram". At step 40 a
  ring on the 2D spectrum sweeps out and fills that step's column, with a dashed link from ring to cell. Then all
  columns fill. x_t stays mostly noise until about step 40 (the shifted σ schedule); captions say so.
- M01: the left image is the real noisy x_t of the street at the current step (`capture/full_002348`).
- M04 VAE map: scatter of grid-64 vs full-res power at ring 8 over the 600 round-trip latents, with the rule's line
  (λ 0.666, shift −0.774 log10); today's value is read off the line (`data/vae_scatter.npz`).
- M04 residual: a 1D cartoon (edge: harmonics above the grid limit follow the pattern; random texture: no pattern),
  then real street detail in the octave [n/4, n/2) below the boundary, pixel view, upper-left crop: true detail =
  clean guess + still new. Step 26 (grid 64): 85 % new. Step 39 (grid 160): 52 % new. The rule's u (latent, band
  [7n/16, n/2)) is 0.96 and 0.81 (`data/resid.json`, `scratch/prep_extra.py`).
- M04 legend replaces the overlapping inline labels. M04 test: the "noise level" label sits at the right end.

## Version history
- v1: first cut. 480p durations: teaser 1:56, method 5:48. Method scenes M04 (test), M05, M06 use zoomed spectrum
  plots. M06 shows two tracking examples: step 34 (112 → 128, one candidate passes) and step 40 (184 → 256, all
  pass).
- v2: see "v2 changes" above.
