# Storyboard — Gaussian Wave Splatting, 1-minute teaser

**Current version: v3** — `animation_gaussian_wave_splatting_1min_v3.py`
Final cut: `media/final/GaussianWaveSplatting_1min_v3_1080p60.mp4` (3 scenes, 1920×1080 @ 60 fps, 59.8 s).
Render: `./render.sh l` (480p15) · `./render.sh h` (1080p60) → `media/final/GaussianWaveSplatting_1min_vN_<res>.mp4`.
Condensed from the 5-min teaser (`../animation_gws_v4.py`, `../storyboard.md`); style, helpers and cached
wave-optics assets (`../data/generated/`, built by the 5-min script, flag `done_v3`) are reused read-only.

Source: Choi*, Chao*, Yang, Gopakumar, Wetzstein, *Gaussian Wave Splatting for Computer-Generated Holography*,
ACM TOG 44(4) / SIGGRAPH 2025, arXiv 2505.06582. Numbers: `../gaussian-wave-splatting/summaries/choi2025gaussian.md`.

## Takeaway
Holograms need the full wavefield of a 3D scene. Gaussian Wave Splatting turns each Gaussian splat into its
wavefield in closed form (with occlusion via alpha wave blending) and computes it fast in the Fourier domain
→ photoreal, occlusion-aware holograms from neural-rendering scenes.

## Numbers on screen (all from the paper)
- “5–11 dB higher PSNR than point / polygon CGH” — Table 1 caption (“5–11dB PSNR improvement over baseline methods”), simulated.
- “6.84 s vs 180 s · nearly 30× faster” — Table 2 (1024 × 1280, 15k Gaussians) and Sec. 5 text (“nearly 30× speedup”).
- Toy simulations are labelled as such.

## Style
Same as the 5-min version: black background, CMU Serif (Latin Modern Math for missing glyphs via `mathify`),
Pango markup only (no LaTeX), text added instantly, colours dim #8A8A8A / red #FF5252 / green #69F0AE /
amber #FFCA28 / cyan #4DD0E1 / purple #E040FB. No part cards, no headers (teaser pacing).
Caption: bottom (y −3.5), size 28, ≤ ~10 words, held ≥ len/17 + 0.8 s.

## Beats (actual durations from the v3 1080p60 render: T1 20.2 s · T2 24.0 s · T3 15.6 s = 59.8 s)

### T1_Problem (20.2 s) — hook + problem
Actual: beat 1 0–4.7 s · beat 2 4.7–10.9 s · beat 3 10.9–20.2 s.
1. (0–4.5 s) Title “Gaussian Wave Splatting” / “for Computer-Generated Holography”; live three-source
   wavefield (`three_U.npy`) in the middle; authors · SIGGRAPH 2025 in dim.
   Caption: “A hologram recreates the light wave of a 3D scene.”
2. (4.5–12 s) Hologram phase (`holo_phase.png`) → arrow 𝒫(·; z) → live reconstruction (`focus_frames.npy`),
   z readout; sweep 0 → 12 mm (star sharp) → 36 mm (“3D” sharp). Caption: “Holograms need the full wavefield of a 3D scene.”
3. (12–20 s) Paper Fig. 5 target renderings: point cloud (“holes”, red) · mesh (“overly smooth”, red) ·
   Gaussian splats (“photoreal”, green). Captions: “Classic CGH uses points or meshes: holes, blur.” →
   “Neural rendering gives photoreal Gaussians. Can holograms use them?”

### T2_Method (24.0 s) — key insight + mechanism
Actual: beat 4 0–4.5 s · beat 5 4.5–9.0 s · beat 6 9.0–14.5 s · beat 7 14.5–18.7 s · beat 8 18.7–24.0 s.
4. (0–6 s) Live 2D Gaussian ↔ its spectrum (numpy FFT, hue = phase); stretch, rotate, shift.
   Caption: “Key fact: a Gaussian's Fourier transform is a Gaussian.”
5. (6–11 s) Side view: plane wave hits a tilted splat, its wavefield travels to the SLM (Huygens sum, live).
   Caption: “So each splat's wavefield has a closed form.”
6. (11–17 s) The one equation, terms coloured + labelled (tilt / scale / shape / position and depth);
   below it the formula's SLM field of one splat → refocused splat. Caption: “One formula per splat, no per-primitive FFT.”
7. (17–22 s) Occlusion toy: naïve sum (red) · alpha wave blending (green) · ray-traced target.
   Caption: “Alpha wave blending: near splats occlude far ones.”
8. (22–28 s) Frequency grids: exact fills one Gaussian at a time, fast lights all frequencies at once;
   amber “6.84 s vs 180 s · nearly 30× faster”. Caption: “In the Fourier domain, all frequencies run in parallel.”

### T3_Payoff (15.6 s) — results + end card
Actual: beat 9 0–5.5 s · beat 10 5.5–10.6 s · beat 11 (end card, 5 s hold) 10.6–15.6 s.
9. (0–7 s) Captured lego (Fig. 5): point-based · polygon-based · GWS; amber “5–11 dB higher PSNR than point / polygon CGH”.
   Caption: “Photoreal 3D holograms on a real display.”
10. (7–14 s) Captured garden focal stack (Fig. 6): rear → middle → front focus crossfades.
    Caption: “One captured hologram, refocused through the scene.”
11. (14–19 s) End card: title, subtitle, authors, Stanford · ACM TOG 44(4), SIGGRAPH 2025,
    arXiv 2505.06582 · bchao1.github.io/gaussian-wave-splatting, code github.com/computational-imaging/hsplat.

## Version history
- v1 — first cut, 58.5 s (480p15). QA: toy note under the hologram touched the left frame edge; payoff beat too short to read.
- v2 — hologram/reconstruction 3.8 tall, occlusion toy 3.1 tall, payoff hold 4.5 s, focal slices 1.1 s; 60.5 s. QA: toy note
  moved under the arrow overlapped both images.
- v3 — toy note split into two lines under the arrow (no overlap). Rendered 1080p60: 59.8 s.
