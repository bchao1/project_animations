# Storyboard — Stereo → light field synthesis teaser (3b1b style)

**Current version: v1** — `animation_slf_v1.py`
Render: `./render.sh l` (480p15 test) · `./render.sh h` (1080p60 final). Env: `imp_tome` + `LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7`
(baked into `render.sh`). `render.sh` first builds the caches, then renders the 11 scenes in parallel (logs in
`media/log_<script>_<res>_<Scene>.txt`) and concatenates them into `media/final/StereoLightFieldSynthesis_<script>_<res>.mp4`.
480p15 durations (v1): S00 8.1 · S01 30.4 · S02 31.9 · S03 32.4 · S04 24.3 · S05 36.2 · S06 34.0 · S07 14.6 · S08 48.4 ·
S09 18.0 · S10 12.1 s = 4 min 50 s.

Source: Chun-Hao (Brian) Chao, Chang-Le Liu, Homer H. Chen, *Robust Light Field Synthesis From Stereo Images With Left-Right
Geometric Consistency*, IEEE ICIP 2021, pp. 1844–1848, National Taiwan University. Author PDF (bchao1/old-website repo),
text, page renders, figures: `stereo-lf-synthesis/pdfs/`; every paper number: `stereo-lf-synthesis/summaries/chao2021robust.md`.
Code: github.com/bchao1/lf-flow (cloned in `scratch/lf-flow/`, read-only).

## Takeaway
A stereo pair is easy to capture but gives only 2 of the 81 views of a 9 × 9 light field. One disparity CNN, run with the pair
in both orders, warps each photo into a full light field (left-right consistency loss: both must match the target); the two
are merged by distance-weighted alpha blending (each view trusts the closer photo), and a patch-wise spatial-angular shuffling
(PSAS) 3D CNN fixes occlusions. +4.75 / +6.13 / +2.07 dB over the best 1-view / 4-view baseline with 1.74M parameters.

## Real computation on screen (no toy stand-ins)
- **Ground truth**: HCI 4D LF benchmark scene “origami” (test split), 9 × 9 views, 512² cropped to 504² (PSAS needs 9 | H, W).
  Parallax sweep, shift-and-add refocusing (31 slopes a ∈ [−1.6, 1.4] px per view; wall at −1.5, crane at +1.0), an
  epipolar-plane image (scanline y = 372, x 112–364, views (4, 0)…(4, 8)), and the authors' `spatio_angular_shuffle` applied to it
  (patch (6, 3), 56 × 56 px tiles).
- **Paper output**: website GIF = the paper's synthesized origami light field (81 raster-ordered views), refocused by
  shift-and-add (title, results, end card).
- **Our re-run** (`scratch/repro/train_hci.py`): authors' DisparityNet + PSAS (1.74M params) trained with the paper's losses
  (w_c = 1, w_t = 0.001, w_r = 1) on the paper's HCI split (20 train = additional + training, 4 test), 135² crops, batch 8,
  Adam 1e-3, 8000 steps on one A6000 (≈ 35 min). Shown: D_lr / D_rl, F_l warps and error maps along the middle row, per-view
  MSE heat maps (mean over 4 test scenes) for F_l / F_r / F_c, blending ablation PSNRs, zoom of view (0, 7) (window with the
  largest F_l → F_c error drop), training snapshots and PSNR curve, final test PSNR. Always labelled “our re-run”.
  Re-run test PSNR (mean of 4 HCI test scenes, 504² crops): F_l 28.66, F_r 28.55, average 30.21, F_c 30.91, F_o 30.91 dB.
  The PSAS refiner stayed ≈ identity (F_o = F_c from step ~800), so refinement gains on screen are the paper's (Table 3), and
  the video says so.

## Global style
| Property | Value |
|---|---|
| Font | CMU Serif; Latin Modern Math for × − → ≈ ‖ ∇ ℒ · … (`MATHCH`); DejaVu Sans for ✓ ✗ |
| Math | Pango markup, no LaTeX. `sub()` rise −3000 + `<small><small>` |
| Background | black |
| Colours | dim `#8A8A8A`; cyan `#4DD0E1` = left view S_l / D_lr / F_l; orange `#FF9E40` = right view S_r / D_rl / F_r; green `#69F0AE` = ours / blended / refined; amber `#FFCA28` = headlines, network g, selections; red `#FF5252` = errors, losses; blue = Srinivasan (1 view), purple = Kalantari (4 views) |
| Heat maps | per-view MSE, `turbo`, shared scale for F_l / F_r / F_c, nearest-neighbour 9 × 9 cells, white dots = input views |
| Error images | |F − F_t| mean over RGB, sqrt(clip(e / 0.25)), `inferno` |
| Disparity | `viridis`, shared 1–99 % range of D_lr and D_rl |
| Text | instant (`self.add`), never animated in |
| Header | top-left, y = 3.62, size 24, dim: “k · title” |
| Caption | bottom, y = −3.5, size 28 (26 when split), max width 13.2, held ≥ len/17 + 0.8 s; `hold()` waits before the next change |
| Part card | “Part k” (30, dim) over title (50, bold), 1.5 s |

## Scenes
### S00_Title
Title (48 bold) + subtitle; GIF card 3.3 tall cycling views along a circle over the 9 × 9 grid; mini grid with the two input
views (amber) and the current view (white box), “2 input views / 79 synthesized”; authors, NTU · IEEE ICIP 2021.

### S01_LightField — “1 · Light fields”
9 × 9 GT thumbnails (0.37 each) + live big view with an amber selector (middle row, then around the rim): parallax. Then the big
view becomes the live shift-and-add refocus with the focus slider (back wall ↔ crane) and the refocus formula. Then capture is
hard: plenoptic-camera sketch (main lens, microlenses, sensor) vs 5 × 5 camera array.

### S02_Sparse — “2 · Synthesis from sparse views”
Four 9 × 9 input maps: 3 × 3 grid (angular super-resolution) · 4 corners (Kalantari 2016) · 1 centre (Srinivasan 2017, ✗ depth
ill-posed) · 2 views = stereo pair (this paper, green box). Pros of stereo + prior stereo work (Zhang 2015: iterative, 514.84 s
CPU, left view only; Stereo Magnification needs poses).

### S03_Disparity — “3 · Disparity and warping”
S_l = V(4, 0), S_r = V(4, 8) with the scanline; EPI rows fade in one by one; three dashed slope lines drawn from our re-run's
D_lr (x = 150, 238, 330). Then D_lr (viridis) = g(S_l, S_r), warp equation, live F_l view + error map while the selector walks
(4, 0) → (4, 8) → (0, 8).

### S04_Consistency — “4 · Left-right consistency”
Two rows: (S_l, S_r) → g → D_lr → warp → F_l mosaic; (S_r, S_l) → same g → D_rl → F_r; target F_t mosaic, red L_c arrows,
Eq. 1, Eq. 2.

### S05_Blending — “5 · Distance-weighted alpha blending”
Heat maps F_l, F_r (+ colour bar) at y 1.45, then paper Fig. 2 (h 1.75) below; Eqs. 3–4, A_l grey map with values 1.0 / 0.5 /
0.0 on the middle row, A_l formula + Eq. 5; blended F_c heat map; bottom row fades, then our re-run PSNRs (F_l only / average /
distance-weighted, before refinement) + paper Table 3 (31.92 / 28.95 → 32.18 / 29.73).

### S06_Refine — “6 · Patch-wise spatial-angular shuffling (PSAS)”
GT thumbnails with the same patch marked in all 81 views; 81 markers fly into a 9 × 9 mosaic, replaced by the real shuffled
image; block diagram 3D conv → shuffle → 3D conv (× 5). Then zoom (our re-run, view (0, 7)): F_l only / F_c blended / F_t + error
maps; Eq. 7 with weights; paper Table 3: PSAS 32.18 / 29.73 vs Srinivasan's 3D CNN 31.33 / 29.54, plus the note that our short
re-run's refiner stayed close to identity.

### S07_Rerun — “7 · A short re-run on HCI”
Live training snapshots (F_o centre view + D_lr, 25 log-spaced steps) with the training-PSNR curve; final test PSNR vs the
paper's 29.73 dB; 852,514 + 886,788 = 1.74M parameters (= Table 2).

### S08_Results — “8 · Results”
Table 1 bar chart (full LF PSNR) + headline +4.75 / +6.13 / +2.07 dB; Zhang comparison (horizontal views, with the Stanford SSIM
caveat). Fig. 3 crops: origami, then amethyst (GT / Kalantari / Srinivasan / Ours). Table 2: params + log-scale time bars
(≈ 147× faster than Kalantari). Refocus of the paper's synthesized origami light field + paper Fig. 4 (phone stereo).

### S09_Ablations — “9 · Ablations (paper Table 3)”
Horizontal INRIA bars (from 30 dB) + HCI numbers for all 7 variants.

### S10_Outlook — “10 · Summary”
Three contributions, headline gains, limits, ISMAR / TVCG 2022 follow-up; end card with the GIF.

## Version history
- v1 — first full cut. QA fixes before the final render: captions use subscript markup (no raw “S_l”); S01 selector = amber over
  black outline, slider/equation moved below the image, plenoptic sketch labelled; S05 re-laid out (Fig. 2 smaller, bottom row
  cleared before the ablation numbers); S06 zoom switched from F_c → F_o (refiner no-op in the re-run) to F_l → F_c, labels moved
  off the header; S07 PSNR curve uses a trailing mean (no edge dip); S08 Fig. 3 crops 2.3 tall, refocus note under the slider,
  Fig. 4 at 3.9 tall; S10 summary split into short lines. 480p15: 4 min 50 s. Rendered 1080p60: 4 min 48.1 s
  (`media/final/StereoLightFieldSynthesis_animation_slf_v1_1080p60.mp4`). Known nit: the S03 EPI title touches the right edge.
  Render note: S04 (4 mosaics × 81 thumbnails) takes ≈ 45 min at 1080p60 — pre-composite the mosaics in v2.
