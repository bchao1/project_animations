# Storyboard — FE-GAN (Self-Supervised Deep Learning for Fisheye Image Rectification), 1-minute teaser

**Current version: v2** — `animation_fisheye_1min_v2.py`
Final cut: `media/final/FisheyeRectification_1min_v2_1080p60.mp4` (3 scenes, 1920×1080 @ 60 fps, 57.1 s; T1 18.8 · T2 23.6 · T3 14.7 s).
Render: `./render.sh l` (480p15) · `./render.sh h` (1080p60) → `media/final/FisheyeRectification_1min_vN_<res>.mp4`.
Individual scene MP4s: `media/videos/animation_fisheye_1min_v2/<res>/T*.mp4`.
Condensed from the 5-min teaser (`../animation_fisheye_v3.py`, `../storyboard.md`); style, helpers and cached toy assets
(`../data/generated/`, built by the 5-min script) are reused read-only.

Source: Chao*, Hsu*, Lee, Wang, *Self-Supervised Deep Learning for Fisheye Image Rectification*, IEEE ICASSP 2020.
Numbers: `../fisheye-rectification/summaries/chao2020selfsupervised.md`.

## Takeaway
Fisheye lenses bend straight lines. FE-GAN predicts a pixel flow and warps the image through it; it trains on unpaired
images with a discriminator plus geometric self-supervision (rotation consistency, radial + smooth flow), so it needs no
camera parameters and no paired ground truth → best PSNR / SSIM / RMSE on three datasets, +2.3 dB PSNR on MCindoor.

## Numbers on screen (all from the paper)
- “+2.3 dB PSNR over the best baseline (MCindoor, paper Table 1)” = 15.215 (FE-GAN) − 12.959 (UNIT) = 2.256 dB.
- “Best PSNR, SSIM and RMSE on MCindoor, Wireframe and LSUN Churches” — Table 1 (FE-GAN best in all 9 columns).
- Toy computations (lens morph, “ray angle at the image corner” 55° → 81°, flow fields) are labelled “toy”.

## Style
Same as the 5-min version: black background, CMU Serif (Latin Modern Math / DejaVu Sans for missing glyphs via `mathify`),
Pango markup only (no LaTeX), text added instantly, colours dim #8A8A8A / orange #FF9E40 (fisheye) / cyan #4DD0E1 (pinhole) /
green #69F0AE (FE-GAN) / red #FF5252 (baselines) / amber #FFCA28 (flow, headline). No part cards, no headers (teaser pacing).
Caption: bottom (y −3.5), size 28, held ≥ len/17 + 0.8 s; `hold()` before visual changes.

## Beats (durations from the v2 480p15 render: T1 19.0 s · T2 23.7 s · T3 14.9 s = 57.6 s)

### T1_Hook (19.0 s) — hook
1. (0–6 s) Title / subtitle; website GIF 3.0 tall toggling every 0.8 s, label “fisheye input” / “rectified by FE-GAN”;
   authors, “National Taiwan University · ASUS Intelligent Cloud Services · IEEE ICASSP 2020”.
   Caption: “Fisheye lenses see wide, but they bend every straight line.”
2. (6–19 s) Toy morph pinhole → fisheye (5.0 tall at (−3.0, 0.25)) with the live world grid and two amber world lines;
   slider pinhole ↔ fisheye and “ray angle at the image corner”. Captions: “Same scene, same focal length: a pinhole view
   becomes a fisheye view.” → “Undoing it usually needs calibration, or paired ground truth that is hard to get.”

### T2_Method (23.7 s) — method in two pictures (one equation)
3. (0–11 s) fisheye x → U → predicted flow f = U(x) (colour + 11 × 11 arrows) → W → rectified x̃ (row-by-row reveal);
   x̃(p) = x(p + f_p). Captions: “FE-GAN predicts a pixel flow, then warps the fisheye image through it.” → “A GAN trained on
   unpaired images: its discriminator asks, distorted or distortion-free?”
4. (11–23.7 s) Left: “cross-rotation consistency”, radial flow turning 90° onto its faint copy, R_θ(G(x)) = G(R_θ(x)).
   Right: “intra-image warping consistency”, ring of equal radial arrows, “radial and smooth flow”. Captions: “No pairs
   needed. Rotate the input by 90°, and the output must simply rotate;” → “and a fisheye flow must point along the radius,
   with one length per circle.”

### T3_Payoff (14.9 s) — payoff + end card
5. (0–9.5 s) Paper Fig. 2: 5 columns (fisheye input, CycleGAN, UNIT, Aleman-Flores, FE-GAN) × 2 rows (church, MCindoor),
   2.05 each; amber “+2.3 dB PSNR over the best baseline (MCindoor, paper Table 1)”. Captions: “Straighter results than
   CycleGAN, UNIT and a classical method.” → “Best PSNR, SSIM and RMSE on MCindoor, Wireframe and LSUN Churches.”
6. (9.5–14.9 s) End card: title, IEEE ICASSP 2020, code URL, authors, affiliations, * equal contribution (4.5 s hold).

## Version history
- v1 — first cut, 51.8 s (480p15). QA: under 55 s; “paper Fig. 2” note near the second row.
- v2 — explicit holds after the title and morph captions (extra waits had been absorbed by caption holds), second payoff
  caption, Fig. 2 note moved to the top right; 57.6 s (480p15). Rendered 1080p60.
