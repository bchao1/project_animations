# Storyboard — Large Étendue 3D Holography, 1-minute teaser

**Current version: v2** — `animation_large_etendue_holography_1min_v2.py`
Final cut: `media/final/LargeEtendueHolography_1min_v2_1080p60.mp4` (57.3 s, 1920 × 1080, 60 fps).
Actual durations (1080p60): T1_Problem 17.2 s · T2_Method 22.7 s · T3_Payoff 17.5 s.
Render: `./render.sh l` (480p15) · `./render.sh h` (1080p60) → `media/final/LargeEtendueHolography_1min_vN_<res>.mp4`.
Target 55–70 s, hard max 75 s.

Condensed from the 5-min explainer (`../animation_large_etendue_v3.py`, `../storyboard.md`). Same style: CMU Serif,
Pango markup (no LaTeX), black background, laser green / amber / cyan palette, text added instantly.
Assets are read by absolute path from `../data/` and `../data/generated/` (nothing regenerated).

Numbers on screen (all from the paper, Chao et al. SIGGRAPH Asia 2024, arXiv 2409.03143):
- 1080p SLM: 80° FoV ↔ eyebox < 1 mm (Sec. 2, Related Work).
- FoV<sub>x</sub>·FoV<sub>y</sub>·w² ≈ λ²N<sub>x</sub>N<sub>y</sub> (Sec. 4, Eq. 6); plot curve uses Fig. 3 parameters (1000 × 1000, 10.8 µm, 632.8 nm).
- α × α sources → étendue × α²; 3 × 3 → 9× (Sec. 4).
- Table 1 (sim., 9 × 9 LF PSNR): I 16.60, IV 18.63, VII 23.95 (9 frames), ours 24.65 (6 frames); "fewer frames (6 vs. 9)".
- Experimental captures: paper Fig. 1 (b–g).

## Scenes (3) and beats — actual durations (v2, planned in brackets)

### T1_Problem (17.2 s) [≈ 19 s]
1. Hook (≈ 8.6 s incl. 0.5 s fade). Title (bold 44, y 3.25) + subtitle (26, dim, y 2.6). Below: captured single-source Fourier spectrum
   (Fig. 1b, 4.6 wide, left) + view through the pupil (Fig. 1c/d, 4.6 wide, right), labels above, source note. Red pupil centred (≈ 4 s),
   then it slides to the blue position while the view fades to the dark right view (2.2 s).
   Captions: "Holograms show true 3D, but to a tiny eyebox" · "Move the eye a few millimetres: the image vanishes".
2. Étendue (≈ 8.6 s). Left: SLM → eyepiece → eyebox schematic (from S02) driven by eyebox w. Right: FoV-vs-eyebox curve with
   amber rectangle (same tracker). Top: the one equation FoV<sub>x</sub>·FoV<sub>y</sub>·w² ≈ λ²N<sub>x</sub>N<sub>y</sub> + dim
   "étendue, fixed by the SLM pixel count". w: 2 → 7.5 → 1.1 mm (3 s each).
   Captions: "SLM pixels cap the étendue: FoV × eyebox" · "A 1080p SLM: 80° FoV leaves under 1 mm of eyebox".

### T2_Method (22.7 s) [≈ 26 s]
3. Key insight (≈ 8.9 s). S05 side view (3 tinted sources → collimator → phase SLM) + computed Fourier plane (toy) that
   accumulates 9 shifted copies (0.33 s each); 3 × 3 icon lights up; amber "expanded eyebox" square (label below the image frame).
   Dim label "a tilt in space = a shift in frequency".
   Captions: "Idea: light the SLM from many angles at once" · "Tilts tile spectrum copies: eyebox 3×, étendue 9×".
4. Catch + fix (≈ 13.8 s: catch 4.2, fix 4.9, pupil 4.4, fade 0.5). S06 top diagram (sources → phase SLM → Fourier lens → amplitude SLM → eye).
   Panel row: spectrum (all 9 sources) … view. First only spectrum + toy view without mask (red "no Fourier mask").
   Caption "But the copies overlap: ghosts, low contrast". Then mask 𝒫(t) and modulated spectrum panels appear, masks cycle
   (0.4 s / frame, clock updater), view switches to ours (6 frames). Caption "Fix: a second SLM applies a content-adaptive mask per frame".
   Then blue pupil circle on the modulated spectrum sweeps across the eyebox, view follows (toy V6 path, parallax).
   Caption "Both SLMs optimized jointly, for every pupil position".

### T3_Payoff (17.5 s) [≈ 20 s]
5. Captured (≈ 5.9 s). Fig. 1 rows Conventional / Ours: spectrum, center view (red), right view (blue); blue pupils slide right.
   Caption "Real prototype: move the pupil, ours stays bright". Header "Experimental captures (paper Fig. 1)".
6. Numbers (≈ 6.0 s). Left: big "9×" + "étendue" + "3 × 3 lasers (paper Sec. 4)". Right: 4 PSNR bars (I, IV, VII 9 fr, ours 6 fr).
   Caption "Best light-field quality, with fewer frames (6 vs. 9)".
7. End card (5.5 s). Tagline "more sources for étendue · a dynamic Fourier mask for image quality". Title, subtitle, 9-copy spectrum, authors, "SIGGRAPH Asia 2024 · arXiv 2409.03143 ·
   bchao1.github.io/holo_dfm".

## Version history
- v1 — first cut, 55.9 s (480p15). QA: "one source" label sat on the spectrum image, "expanded eyebox" label inside
  the image, captured-results beat and bar chart too short to take in.
- v2 — removed "one source" label; "expanded eyebox" moved below the frame; captured beat +1.6 s (1.0 s before and
  2.2 s after the pupil move); bars held 1 s after the caption. 57.9 s at 480p15, 57.3 s at 1080p60.
