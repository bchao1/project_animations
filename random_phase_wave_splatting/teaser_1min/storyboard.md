# Storyboard — Random-phase Wave Splatting, 1-minute teaser

**Current version: v2** — `animation_rpws_1min_v2.py`
Final cut: `media/final/RandomPhaseWaveSplatting_1min_v2_1080p60.mp4` (3 scenes, 1920×1080 @ 60 fps, 56.7 s; T1 19.7 · T2 23.8 · T3 13.2 s).
Render: `./render.sh l` (480p15) · `./render.sh h` (1080p60) → `media/final/RandomPhaseWaveSplatting_1min_vN_<res>.mp4`.
Individual scene MP4s: `media/videos/animation_rpws_1min_v2/<res>/T*.mp4`.
Condensed from the 5-min teaser (`../animation_rpws_v3.py`, `../storyboard.md`); style, helpers and cached toy
wave-optics assets (`../data/generated/`, built by the 5-min script) are reused read-only.

Source: Chao*, Yang*, Choi, Gopakumar, Koiso, Wetzstein, *Random-phase Wave Splatting of Translucent Primitives for
Computer-generated Holography*, arXiv 2508.17480 (2025). Numbers: `../random-phase-wave-splatting/summaries/chao2025random.md`.

## Takeaway
Smooth-phase holograms (like Gaussian Wave Splatting) squeeze light into a tiny eyebox. Random phase fills the eyebox
(parallax, natural blur), but breaks alpha blending with a cross term; RPWS blends in intensity (√ weights), composites back
to front, and time-multiplexes frames so the blend is exact on average → +10.5 dB light-field PSNR over GWS.

## Numbers on screen (all from the paper)
- “+10.5 dB light-field PSNR over GWS” / “24.87 vs 14.35 dB, Blender scenes, 24 frames” — Table 1 (RPWS-GS 24 frames vs GWS,
  simulated 10 × 10 light fields, Blender). 24.87 − 14.35 = 10.52 dB.
- Toy simulations (eyebox, frame averaging, alpha toy) are labelled as such.

## Style
Same as the 5-min version: black background, CMU Serif (Latin Modern Math / DejaVu Sans for missing glyphs via `mathify`),
Pango markup only (no LaTeX), text added instantly, colours dim #8A8A8A / red #FF5252 (GWS) / green #69F0AE (RPWS) /
amber #FFCA28 / cyan #4DD0E1 (pupil). No part cards, no headers (teaser pacing).
Caption: bottom (y −3.5), size 28, held ≥ len/17 + 0.8 s; `hold()` before visual changes.

## Beats (actual durations from the v2 480p15 render: T1 19.8 s · T2 23.9 s · T3 13.3 s = 57.0 s)

### T1_Hook (19.8 s) — hook
1. (0–4.8 s) Title / subtitle; two live panels 2.7 tall at (∓1.7, −0.05): one random-phase frame vs running average
   (1 → 24 frames); authors, “Stanford University · KDDI Research · arXiv 2508.17480 (2025)”.
   Caption: “Holograms from splats, done with random phase and many frames.”
2. (4.8–19.8 s) Eyebox: smooth (GWS, red) | random (RPWS, green) — spectrum + live view each (2.8 tall, x −5.2 / −2.2 / 2.2 / 5.2),
   cyan pupil sliding 0 → −0.72 → +0.72 → −0.72 → 0. Captions: “Smooth-phase holograms send light straight ahead, into a
   tiny eyebox.” → “Move your eye, and the image vanishes.” → “Random phase fills the whole eyebox: bright everywhere, with parallax.”

### T2_Method (23.9 s) — key insight + method in one picture (one equation)
3. (0–9 s) The only equation: |w₁e^{iφ₁} + w₂e^{iφ₂}|² = w₁² + w₂² + 2w₁w₂cos(φ₁ − φ₂) (38, cross term amber);
   “smooth phase: cos = 1 ⇒ amplitudes add (GWS)” (red) / “random phase: 𝔼[cos] = 0 ⇒ intensities add, on average” (green).
   Captions: “But random phase breaks alpha blending …” → “Key insight: blend in intensity (√ weights), and let frame averaging
   cancel the cross term.”
4. (9–23.9 s) Toy: target / naïve random phase / RPWS (2.8 tall) + error maps, frames averaged 1 → 24 (5.5 s).
   Captions: “Average frames: naïve random phase stays wrong; RPWS converges to the target.” → “Time multiplexing, composited
   back to front, is what makes the blending exact.”

### T3_Payoff (13.3 s) — payoff + end card
5. (0–7.8 s) Paper Fig. 1: GWS front focus vs RPWS back focus (width 6.3, y 1.1); amber “+10.5 dB light-field PSNR over GWS” (36)
   and “24.87 vs 14.35 dB, Blender scenes, 24 frames (paper Table 1)”. Caption: “Natural defocus blur and a full eyebox, from
   Gaussians or triangles.”
6. (7.8–13.3 s) End card: title, subtitle, arXiv 2508.17480, authors, Stanford University · KDDI Research · * equal contribution (5 s hold).

## Version history
- v1 — first cut, 54.3 s (480p15). QA: “simulated holograms, paper Fig. 1” overlapped the headline; slightly under 55 s.
- v2 — T3 images moved up / headline down, payoff hold 2.5 s, slower pupil sweep and frame averaging; 57.0 s (480p15). Rendered 1080p60.
