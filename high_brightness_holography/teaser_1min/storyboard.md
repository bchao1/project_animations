# Storyboard — High-Brightness Holographic Projection, 1-minute teaser

**Current version: v2** — `animation_hbh_1min_v2.py`
Final cut: `media/final/HighBrightnessHolography_1min_v2_1080p60.mp4` (3 scenes, 1920×1080 @ 60 fps, 57.9 s; T1 20.7 · T2 21.2 · T3 16.0 s).
Render: `./render.sh l` (480p15) · `./render.sh h` (1080p60) → `media/final/HighBrightnessHolography_1min_vN_<res>.mp4`.
Individual scene MP4s: `media/videos/animation_hbh_1min_v2/<res>/T*.mp4`.
Condensed from the 5-min teaser (`../animation_hbh_v4.py (v3 when this cut was made; same assets)`, `../storyboard.md`); style, helpers and cached toy assets
(`../data/generated/`) are reused read-only.

Source: Chao, Gopakumar, Choi, Wetzstein, *High-brightness holographic projection*, Optics Letters 48(15), 4041 (2023),
+ Supplement 1 + Visualization 1. Numbers: `../high-brightness-holography/summaries/chao2023high.md`.

## Takeaway
Conventional projectors block light; a phase-only hologram steers it, but still leaks some outside the picture. A
light-efficiency loss (match a target rescaled to carry all of the laser's power) plus a camera-calibrated propagation model
puts the light inside the image: around 30% brighter captured images at the same laser power, image quality retained.

## Numbers on screen
Paper: “Optics Letters 48(15), 4041 (2023)”; “+24%” (learned model) and “+0%” (ideal ASM model), castle with the loss,
paper Fig. S9 labels; “around 30% brighter, image quality retained” (main text, Fig. 5 caption); “test-set average: +33%
captured, +66% in simulation” (supplement Sec. 10). Paper Fig. S9 images carry their own PSNR/LPIPS/gain labels.
Toy (labelled): 34% light through the amplitude panel vs 79% landing in the image (1D toy); 64% → 94% light in the image (2D toy).

## Style
Same as the 5-min version: black background, CMU Serif (+ Latin Modern Math / DejaVu Sans via `mathify`), Pango markup only,
text added instantly, colours dim #8A8A8A / red #FF5252 (conventional, without the loss) / green #69F0AE (hologram, with the
loss) / amber #FFCA28 (headline) / cyan #4DD0E1 (image region). No part cards, no headers.
Caption: bottom (y −3.5), size 28, held ≥ len/17 + 0.8 s; `hold()` before visual changes.

## Beats (durations from the v2 1080p60 render)

### T1_Hook (20.7 s)
1. (0–4 s) Title (52 bold) + authors (y 2.3); website clip `OL2023` (captured swallowtail wipe) 3.3 tall; note “captured; the
   brighter half uses the light-efficiency loss”; “Stanford University · Optics Letters 48(15), 4041 (2023)”.
   Caption: “A projector that puts all of its light where the picture is.”
2. (4–20.7 s) 1D toy x–z maps 3.4 × 4.8 at x ∓3.4: conventional amplitude panel (“light that gets through: 34%”) vs
   phase-only hologram (“light that lands in the image: 79%”). Captions: blocking light → redirecting light → “But even a
   hologram lets some light stray outside the picture.”

### T2_Method (21.2 s) — the one equation
Eq. 3 (L_reg, 30 pt, y 2.85). Toy 2D SGD live, log scale, 3.2 tall at x ∓3.35: “usual image loss only” vs “+ light-efficiency
loss”, “light in the image” 64% vs 94% (5.5 s sweep). Then paper Fig. S9 castle, with the loss: ideal model / camera in the
loop / learned model (width 4.0). Captions: “Key idea: rescale the target…” → “The border goes dark… (toy: 64% to 94%)” →
“On real hardware, it needs a camera-calibrated model…” → “With a learned model: +24% brighter. With the ideal model: +0%.”

### T3_Payoff (16.0 s)
Visualization 1 (7.0 wide, y 1.15; monarch → macaw → bunny wipes) + amber “around 30% brighter, image quality retained” (36)
+ dim “test-set average: +33% captured, +66% in simulation (supplement)”. End card 4 s: title, Optics Letters 48(15), 4041
(2023), authors, Stanford University · bchao1.github.io.

## Version history
- v1 — first cut, 52.6 s (480p15). QA: T2 image labels touching the equation; T3 headline crowding the video's source line;
  under 55 s.
- v2 — T2 panels 3.2 tall at y 0.05; T3 video 7.0 wide at y 1.15, headline / sub-line moved down, extra caption so the video
  plays ~10 s; 58.2 s (480p15), 57.9 s (1080p60). Rendered 1080p60.
