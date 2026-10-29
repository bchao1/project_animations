# Storyboard — Stereo → light field synthesis, 1-minute teaser

**Current version: v1** — `animation_slf_1min_v1.py`
Render: `./render.sh l` (480p15) · `./render.sh h` (1080p60) → `media/final/StereoLightFieldSynthesis_1min_vN_<res>.mp4`.
Individual scene MP4s: `media/videos/animation_slf_1min_vN/<res>/T*.mp4`.
Condensed from the 5-min teaser (`../animation_slf_v*.py`, `../storyboard.md`); style, helpers and caches come from
`../slf_common.py` / `../data/generated/` (read-only here).

Source: Chao, Liu, Chen, *Robust Light Field Synthesis From Stereo Images With Left-Right Geometric Consistency*, IEEE ICIP 2021.
Numbers: `../stereo-lf-synthesis/summaries/chao2021robust.md`.

## Takeaway
Two photos → a full 9 × 9 light field: warp each photo with a learned disparity map (run both ways, kept consistent), blend
by distance, refine with a patch-shuffling 3D CNN; +4.75 / +6.13 / +2.07 dB over 1-view and 4-view baselines, 1.74M params.

## Numbers on screen (all from the paper)
“+4.75 dB INRIA · +6.13 dB HCI · +2.07 dB Stanford” (Table 1: 32.18 − 27.43, 29.73 − 23.60, 25.85 − 23.78), bar values of
Table 1, “1.74M parameters” (Table 2), “IEEE ICIP 2021”. Panels from our re-run of the authors' code are labelled as such.

## Style
Same as the 5-min version (see `../storyboard.md`); no part cards, no headers (teaser pacing); one equation line (A_l, Eq. 5).

## Beats (480p15 v1: T1 17.7 s · T2 28.9 s · T3 16.5 s = 63.1 s)
### T1_Hook
1. Title / subtitle, GIF card cycling the paper's synthesized views + mini grid (2 input / 79 synthesized). “Two photos in, a
   full 9 × 9 light field out.”
2. GT 9 × 9 grid with only the stereo pair lit (others at 22 % opacity). “A light field is a grid of views. A stereo pair gives
   you only 2 of the 81.”
3. All views lit + live GT refocus sweep wall → crane with the focus slider. “With all 81 you can refocus after the shot, and
   change viewpoint.”

### T2_Method
4. S_l + D_lr (re-run), live warped view + error map as the selector walks (4, 0) → (4, 8). “A CNN predicts disparity; warping
   the photo fills every view, but errors grow with distance.”
5. Heat maps F_l, F_r (re-run, 4 HCI test scenes), then the blended F_c + A_l / Eq. 5. Two captions.
6. PSAS: patch markers fly from the 81 thumbnails into the shuffled mosaic; warped-only vs blended zoom + error maps (re-run;
   its refiner stayed ≈ identity) and “PSAS refiner vs a standard 3D CNN: +0.85 dB (paper, INRIA)”.

### T3_Payoff
7. Table 1 bars + amber gains + “1.74M parameters”.
8. Refocus of the paper's synthesized light field (GIF), wall → crane.
9. End card: GIF, title, IEEE ICIP 2021 · code link, authors · NTU (4.5 s).

## Version history
- v1 — first cut, 63.1 s (480p15). Rendered 1080p60: 62.6 s (`media/final/StereoLightFieldSynthesis_1min_v1_1080p60.mp4`).
