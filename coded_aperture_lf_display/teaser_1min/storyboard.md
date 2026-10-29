# Storyboard — Coded-aperture TDM light field display, 1-minute teaser

**Current version: v3** — `animation_tdmlf_1min_v3.py`
Final cut: `media/final/CodedApertureLightFieldDisplay_1min_v3_1080p60.mp4` (3 scenes, 1920×1080 @ 60 fps, 63.5 s). Render: `./render.sh l` (480p15) · `./render.sh h` (1080p60) → `media/final/CodedApertureLightFieldDisplay_1min_vN_<res>.mp4`.
Individual scene MP4s: `media/videos/animation_tdmlf_1min_v3/<res>/T*.mp4`.
Condensed from the 5-min teaser (`../animation_tdmlf_v3.py`, `../storyboard.md`); style, helpers (copied) and cached toy assets
(`../data/generated/`, built by the 5-min script) are reused read-only.

Source: Chao*, Liu*, Chen, *Time-Division Multiplexing Light Field Display With Learned Coded Aperture*, IEEE TIP 32 (2023),
arXiv 2107.06205. Numbers: `../coded-aperture-lf-display/summaries/chao2023time.md`.

## Takeaway
Stereo displays cause the vergence-accommodation conflict; light field displays let the eye refocus, but a time-division
display with few views turns defocus blur into ghost copies. Blur takes the shape of the aperture, so CTDM-LFD shows k encoded
images, each through its own learned coded aperture, and the eye sums them; network and apertures are learned end to end →
natural blur and about 4 dB more PSNR with 4 views.

## Numbers on screen (all from the paper)
- “+3.7 dB (INRIA-syn) · +3.8 dB (INRIA-real) · +4.7 dB (SIGGRAPH16)”, “PSNR over TDM-LFD with the same 4 views (paper Table V)”:
  32.29 − 28.56, 30.21 − 26.39, 37.57 − 32.92 (Table V, n = 4, k = 4).
- “n = 4, k = 4” (Fig. 17 / Table V settings), “focal plane k of 9” (Fig. 17 shows nine planes), “IEEE TIP 2023”,
  “IEEE Transactions on Image Processing 32 (2023) · arXiv 2107.06205”.
- Toy panels (refocus sweep, PSFs, time-multiplex cycle, training) are labelled “toy”.

## Style
Same as the 5-min version: black background, CMU Serif (Latin Modern Math / DejaVu Sans for missing glyphs via `mathify`),
Pango markup only (no LaTeX), text added instantly, colours dim #8A8A8A / red #FF5252 (baseline) / green #69F0AE (CTDM) /
amber #FFCA28 / cyan #4DD0E1 (apertures, accommodation). No part cards, no headers (teaser pacing). One equation (Eq. 10).
Caption: bottom (y −3.5), size 28, held ≥ len/17 + 0.8 s; `hold()` before visual changes.

## Beats (durations from the v3 480p15 render: T1 21.1 s · T2 24.9 s · T3 17.9 s = 63.9 s)

### T1_Hook (21.1 s)
1. (0–6 s) Title / subtitle; website GIF (3.0 tall) toggling baseline (red label “baseline TDM-LFD: ghosted blur”) and CTDM-LFD
   (green “CTDM-LFD (this paper): natural blur”) every 1 s; authors · NTU · IEEE TIP 2023. Caption “A light field display that
   gets defocus blur right.”
2. (6–12 s) Top-view VAC diagram: eyes, screen, virtual star, amber vergence lines, cyan focus line at the screen, red bracket;
   labels “accommodation: eyes focus on the screen” / “vergence-accommodation conflict” / “vergence: eyes point at the object”.
   Caption “Stereo headsets make your eyes converge at one depth but focus at another.”
3. (12–21 s) Toy refocus: “dense light field: 81 views” vs “sparse display: 4 views” (3.4 tall), focus ruler, sweep
   near → far → near → far. Caption “Light field displays let the eye refocus. But with few views, blur breaks into ghosts.”

### T2_Method (24.9 s)
4. (0–10 s) Fourier-optics PSFs (open / 4 windows (TDM) / coded aperture (paper Fig. 9)), defocus 0 → 18 → 8 → 18 waves.
   Captions “Blur takes the shape of the aperture: four tiny windows give four ghost dots.” → “A coded aperture instead spreads
   each point into a pattern we can design.”
5. (10–18 s) Time multiplexing: 4 toy encoded images + learned apertures (t = 1…4) → “eye sums the frames” → retina running sum
   (two cycles); Eq. 10. Caption “Show k encoded images, each through its own coded aperture. The eye sums them.”
6. (18–25 s) Toy training: 4 apertures morph from noise into blobs, near-focus retina image sharpens, “training step N”.
   Caption “A network and the apertures are learned end to end through a Fourier-optics model.”

### T3_Payoff (17.9 s)
7. (0–6.5 s) Paper Fig. 11 zoom (near focus, far letters): baseline / CTDM-LFD / ground truth (3.1 wide) + amber headline.
   Caption “Natural blur instead of ghosts, and about 4 dB better on three datasets.”
8. (6.5–12.5 s) Paper Fig. 17 focal stacks (chairs, letters, poppy) swept through the 9 planes. Caption “Sweep the focus: sharp
   where you look, smooth blur everywhere else.”
9. (12.5–17.9 s) End card: GIF (CTDM frame), title, IEEE TIP 32 (2023) · arXiv 2107.06205, authors, NTU · * equal contribution (5 s).

## Version history
- v1 — first cut, 44.6 s (480p15): too short; T2 “encoded images” / “coded apertures” labels overlapping.
- v2 — added the VAC beat (T1), a second PSF caption and sweep (T2), the Fig. 17 sweep (T3); two-line column labels; 63.9 s.
- v3 — VAC labels reworded (“accommodation: eyes focus on the screen” / “vergence: eyes point at the object”); 63.9 s (480p15).
  Rendered 1080p60: 63.5 s.
