# Storyboard — MPI holography, 1-minute teaser

**Current version: v2** — `animation_mpi_1min_v2.py`
Final cut: `media/final/MPIHolography_1min_v2_1080p60.mp4` (3 scenes, 1920×1080 @ 60 fps, 55.7 s; T1 18.7 · T2 23.6 · T3 13.4 s).
Render: `./render.sh l` (480p15) · `./render.sh h` (1080p60) → `media/final/MPIHolography_1min_vN_<res>.mp4`.
Individual scene MP4s: `media/videos/animation_mpi_1min_v2/<res>/T*.mp4`.
Condensed from the 5-min teaser (`../animation_mpi_v3.py` / `v4`, `../storyboard.md`; v4 only changes S07, not used here); style, helpers and cached toy
wave-optics assets (`../data/generated/`, built by the 5-min script) are reused read-only.

Source: Chao, Seyb, Matsuda, Cossairt, Zhou, Lanman, Wetzstein, Kuo, Jang, *Fast Wave-optics Rendering of Multiplane Images
for 3D Holographic Displays*, arXiv 2607.19731 (2026). Numbers: `../mpi-holography/summaries/chao2026fast.md`.

## Takeaway
RGBD-layer holograms are fast but tear at depth edges; Gaussian holograms (RPWS) are accurate but take hours. Feeding the
hologram a multiplane image (a handful of RGBA layers) and compositing random-phase waves back to front (propagate, × √(1 − α),
+ √(αc)e^{iφ}), averaged over 24 frames, gives RPWS-like focus and parallax up to 250,000× faster.

## Numbers on screen (all from the paper)
- “up to 250,000× faster than RPWS” — abstract / text (vs 4-layer MPI).
- “0.05 s (4 MPI layers) vs ≈ 4 h (RPWS)” — values read from the vector data of paper Fig. 3 (0.050 s; 1.41 × 10⁴ s).
- 24 frames (paper T = 24); toy parameters (4 / 9 / 14 / 20 mm, 638 / 520 / 450 nm, 8 µm pixels) are labelled toy simulation.

## Style
Same as the 5-min version: black background, CMU Serif (Latin Modern Math / DejaVu Sans for missing glyphs via `mathify`),
Pango markup only, text added instantly; MPI green, RGBD red, RPWS orange, amber headline. No part cards, no headers.
Caption: bottom (y −3.5), size 28, held ≥ len/17 + 0.8 s; `hold()` before visual changes.

## Beats (durations from the v2 480p15 render: T1 18.9 s · T2 23.9 s · T3 13.5 s = 56.2 s)

### T1_Hook (18.9 s)
1. (0–6.5 s) Title / subtitle, homepage videos playing (light field | focal stack, 3.0 tall), authors, affiliations.
   Caption: “Holograms you can refocus and look around, rendered from multiplane images.”
2. (6.5–18.9 s) Paper Fig. 1: “RGBD layers: fast” (red) | “RPWS (Gaussians): accurate” (orange), width 5.6.
   Captions: “Today's holograms: RGBD layers are fast but leak light at depth edges.” → “Gaussian-based RPWS is accurate,
   but splats millions of primitives: hours per hologram.”

### T2_Method (23.9 s) — one equation
3. (0–5.5 s) Toy image → 4 sheared RGBA layers: “The idea: feed the hologram a multiplane image, a handful of RGBA layers.”
4. (5.5–16 s) Eq. 2 at top; the 5 running-wave panels (layer 4 → … → SLM) appear with amber 𝒫 arrows. Captions:
   “Back to front: propagate, dim by √(1 − α), add each layer's random-phase wave.” → “One FFT propagation per layer, not one
   wavefront per primitive.”
5. (16–23.9 s) Toy focal sweep RGBD | MPI (24 frames each), 4 → 20 → 9 mm: “Averaged over 24 frames: RGBD tears at depth
   edges, the MPI blurs naturally.”

### T3_Payoff (13.5 s)
6. (0–8.5 s) Fig. 1 MPI back / front focus (width 5.0); amber “up to 250,000× faster than RPWS” (40 bold) and “0.05 s (4 MPI
   layers) vs ≈ 4 h (RPWS); values read from paper Fig. 3”. Caption: “Quality close to RPWS, far better than RGBD, and up to
   250,000× faster.”
7. (8.5–13.5 s) End card: title, subtitle, arXiv 2607.19731, 9 authors, Stanford University · Meta (4.5 s hold).

## Version history
- v1 — first cut, 54.9 s (480p15); T2's first panel appeared before its caption; slightly under 55 s.
- v2 — caption before the first compositing panel, longer focal sweep (3.5 + 2.5 s) and payoff hold (3 s); 56.2 s (480p15).
  Rendered 1080p60: 55.7 s.
