# Storyboard — Level-of-Token Diffusion, 5-minute teaser (3b1b style)

**Current version: v3** — `animation_lot_5min_v3.py` (480p15 total 277.8 s ≈ 4 min 38 s)
Render: `./render.sh l` (480p15) · `./render.sh h` (1080p60) → `media/final/LoTDiffusion_5min_vN_<res>.mp4`.
Final cut: `media/final/LoTDiffusion_5min_v3_1080p60.mp4` (10 scenes, 278.6 s ≈ 4 min 39 s, 1920×1080, 60 fps, h264).
Env: `imp_tome` + `LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7`. Target 4:30–5:30.
Style: `../STYLE_NOTES.md`; helpers and the `LoTField` / layout code come from `../teaser_1min/animation_lot_1min_v4.py`
(headers, part cards and `title_card` from the foveated 5-min v2). Captions ≤ ~16 words, held ≥ len/16 + 0.8 s.

**Takeaway (same as the 1-min).** Where detail matters is often known before generation. LoT turns it into a
layout of variable-size tokens; patch-wise asymmetric flow lets a fine-tuned pretrained DiT predict full-resolution
flow for every patch from the short sequence → up to 2.04× faster images, 3.53× faster video, close to full quality.

**Running example.** The corgi painted demo and its reproduced layout (as in the 1-min) carry S00–S06 and the outlook.

**Colour-shaded tokens (new in the 5-min, user request, after the foveated S02 lion tokens).** Every token grid on
screen is `LoTField(..., shade=True)`: fill = mean colour of the result image over the token's patch (64 × 64 box-mean
`cells.npy`, averaged per block), outline = extent colour (1×1 `#448AFF` · 2×2 `#00C853` · 4×4 `#1B7F3B` · 8×8
`#0E5A2A`), outline width 0.3 / 0.9 / 1.6 / 2.2 at panel size 4.6 (scaled with the panel). The girl and corgi show in
fine tokens; sky and meadow become flat blocks, so the layout and the content read at once.

Part cards: “Part 1 · The problem” (S01), “Part 2 · The method” (S03), “Part 3 · Results” (S07). Headers “k · title”.

## Assets (`data/generated/`, built on import)
- `corgi_layout.json`, `detail_D.npy` / `detail_D.png` (64 × 64 detail field, nearest-upsampled), `cells.npy`.
- `fig7a_{0..4}.png`, `fig7b_{m}_{k}.png` (m = Full, ToMe-SD, Foveated, Ours; k = 3 frames): rendered PDF page 9 crops, 8×.
- `fig4_{0..7}.png`: deer row of `../data/img/fig4_full.png` (layout, generation) × (box, mask, texture, depth).
- `blocky.png`: layout-average of the result (not used on screen in v3; S04 builds the blocky view from tokens).
- Clips: `../data/clips/show{03,02,14}_{lay,gen}` (site showcase items 3, 2, 14).

## Numbers on screen
| Number | Source |
|---|---|
| 64 × 64 = 4,096, 1,546, 1,312 / 140 / 79 / 15, 2.65×, 64 → 211 → 562 → 1,546 (split passes) | site demo + reproduced layout |
| 4,096² ≈ 16.8 M pairs; 1,546² / 4,096² ≈ 14% | arithmetic |
| Fig. 4 deer: 1.73× · 1.54×, 2.18× · 1.81×, 1.36× · 1.28×, 3.11× · 2.29× | paper Fig. 4 |
| clips: 1.66× (show03), 1.50× (show02), 2.02× (show14) fewer tokens | site showcase `compression` |
| Tab. 1 HPSv3 / speed at 1.893, 2.406, 2.935× for ToMe-SD, DDiT, Foveated, Ours; full 10.43 | paper Tab. 1 |
| Tab. 2 at 3×: speed 3.53 / 3.31 / 3.28×; imaging 0.647 / 0.565 / 0.495 / 0.421 | paper Tab. 2 |
| Fig. 7a 1,269 tokens (1.00 / 2.10 / 1.90 / 1.63 / 2.29×); Fig. 7b 50,134 tokens (1.00 / 1.60 / 1.59 / 1.62×) | paper Fig. 7 |
| Tab. 3 HPSv3: 7.06 / 8.20 / 10.03 / 10.35 (≈ 1.52× compression) | paper Tab. 3, A.13 |
| Fig. 22: 3,189 tokens (1.28×) vs 1,466 (2.79×) | paper Fig. 22 |
| LoRA rank 256, FLUX.2 klein 4B, Wan2.1 14B | §5, A.4 |
| toy check max \|û − (ε − x₀)\| ≈ 4 × 10⁻¹⁶ | computed (S05 toy) |

## Scenes (v3; durations from the 480p15 render)

### S00_Title (6.2 s)
Title (72 bold, y 2.75) + subtitle (32, y 1.95) + “Anonymous ICLR 2027 submission 7014” (22 dim, y −3.3). Shaded layout
emblem (3.8 at (0, −0.6)) fades in as 64 coarse tokens and splits 8 → 4 → 2 → 1 (0.8 s + 0.3 s each), 1.4 s hold.

### S01_Cost (32.9 s) — card “Part 1 · The problem / Every token costs the same”
Result image (4.6 at (−3.4, 0.35)) → 64 × 64 grid lines → shaded 1 × 1 tokens (the image stays readable) →
“h × w = 64 × 64 = 4,096 tokens”. Face / sky 4 × 4 boxes with key “16 tokens on the girl's face / of empty sky” (x 1.2).
Schematic attention square (3.6 at (3.5, 0.5), “4,096 × 4,096 ≈ 16.8 M query–key pairs”) with query-token / row scan
(3 positions). N / 4N squares. Captions: grid of latent tokens · 16 × 16-pixel patch of the 1024² image · both cost the
same · attention every step · 2× resolution → 4× tokens, 16× attention, video × frames.

### S02_Known (18.1 s) — header only
Prompt excerpt box (x −3.35; amber = detail phrases, purple = defocused sky) → arrow → painted detail map (4.2 at
(3.6, 0.45)) with brush legend; dim line of other layout sources (masks · boxes · texture variance · depth of field).
Question card “Can a diffusion model spend its tokens where the plan says?” (2.2 s).

### S03_Layout (37.2 s) — card “Part 2 · The method / Level-of-Token layout”
Left: detail field D (64 × 64 greys). Right: shaded layout built live: 64 tokens of 8 × 8 → 211 → 562 → 1,546
(1.2 s per pass), count below, extent key. Then left: extent shapes (1,1) (2,2) (4,4) (8,8) (2,4) at true relative size
(U = 0.17 per cell) + L = 1,546 · HW / L = 2.65× · attention pairs ≈ 14% (fit before arranging).

### S04_Naive (23.9 s) — header “4 · Just feed big tokens to the DiT?”
Shaded layout (right) → each token becomes a flat square of its patch (“toy: one value per token”, red “blocky”) next to
the real LoT output (left). Captions: a coarse token is one vector · decode as is → blocky · the decoder needs every
position and the pretrained DiT only predicts flow for its tokens · LoT predicts full-resolution flow for every patch.

### S05_AsymFlow (39.5 s) — core toy, header “5 · Patch-wise asymmetric flow”
Shaded layout (2.6 at (−5.0, 1.55)) with the amber 4 × 4 grass token; its 16 cells (contrast-stretched colours of the
result image) at t = 0.6 as tiles → Aₑᵀ → one token (toy A = patch mean) → DiT + LoRA → “head 4 × 4” → clean estimate
tiles; Eq. 9 colour-coded (amber = from the head, blue = completed from x_t); toy check ≈ 4 × 10⁻¹⁶. Then the toy
sampler (40 Euler steps, oracle head): 5 snapshots t = 1 … 0 of patch + token. Footnote on the toy assumptions.

### S06_DiT (32.9 s) — header “6 · Telling the DiT what each token is”
Shaded layout → 4 input projections (“in e×e”, extent-coloured) → DiT + LoRA (× N blocks) → heads h_(e,e) → full-resolution
flow. Shape embedding s(e) formula (y 2.75). RoPE panel: 1 × 1, 2 × 2, 4 × 4 tokens with centre dots on a fine grid +
c = (u + (e_h − 1)/2, v + (e_w − 1)/2). Training caption (LoRA rank 256, heads, shape MLP; FLUX.2 klein 4B, Wan2.1 14B).

### S07_Sources (22.5 s) — card “Part 3 · Results / Layouts from anywhere”
Fig. 4 deer: 4 (layout | generation) pairs 1.5 each with compression · speedup (amber). Then 3 site clips (driving mask,
robotics mask, game box), layout over video, playing at 15 fps, with compression; source note at y −1.85.

### S08_Results (35.9 s) — header “8 · Results”
(a) HPSv3-vs-speedup plot of Tab. 1 (DDiT, Foveated, Ours; ToMe-SD below the axis, noted), full-resolution dot;
legend column at x 1.1 with the Tab. 2 video summary. (b) Fig. 7a row. (c) Fig. 7b (4 methods × 3 frames, 1.05 tall,
“noisy” on ToMe-SD). (d) Tab. 3 ablation bars.

### S09_Outlook (28.7 s) — header “9 · Limitations and outlook”
Fig. 22 sheet music (4 panels 2.6), “broken staff lines”; plan → compute → image (painted map → shaded layout → result);
three ideas + “up to 2.04× faster images · 3.53× faster video”; end card with the shaded layout emblem (4.5 s).

## Version history
- v1 — first cut, 10 scenes, 277.8 s. QA: 1 × 1 outlines too heavy (the image read blue), S03 extent shapes far too
  large and the text column clipped, S05 “head 4 × 4” overflowing its box, S07 source note on the clip labels,
  S08 legend column clipped right, S09 source note under the red note. S05 crashed once on a 2-D centre (fixed in v1).
- v2 — thinner outlines (0.3 / 0.9 / 1.6 / 2.2), true-scale extent shapes, two-line head box, moved notes. 277.8 s.
- v3 — S03 equations and S08 video lines fitted before aligning (they were shifted off-frame). 277.8 s.
- v4 (2026-10-14, slide deck) — `animation_lot_5min_v4.py` + `render_v4.sh h` (S02–S08 only, `--save_sections`):
  `CAPTIONS = False` (no captions, headers or part cards; S02 question card kept as its own section), every scene cut
  into one-visual-step sections that end on the finished visual (0.3-s hold; fades open the next section, no fade to
  black at scene ends), S03 drops the "attention pairs ≈ 14%" line (no quadratic-cost claim on screen), S06 training
  caption dropped (no visual of its own); S02 prompt hand-wrapped and fitted (the v3 box ran off the left edge). Sections + last frames: `slides_v5/`, manifest `../slides_v5/sections.json`.
- v5 (2026-10-14, slide deck) uses `animation_lot_5min_v5.py` + `render_v5.sh h S07_Sources S08_Results`. It fixes faint
  white and black lines on the figure panels. v4 cropped Fig. 4, 7a and 7b panels out of the paper's figure rasters.
  Those crops kept 1 to 2 px of gutter and were resampled twice. v5 loads the files the paper's figure scripts placed
  (LoT repo `results/figure4/assets`, `results/paper_figures_456_portable/assets`). The DDiT and Foveated Fig. 7a
  panels exist only inside the PDF, so v5 extracts those two rasters (xref 493, 494). Sources are listed in
  `data/generated/true_sources.json`. The old crops are in `data/generated/_pdfcrop_backup_2026-10-14/`.
  Re-rendered sections: S07 00 and 01, S08 02, 03 and 04 (01 and 04 only change in their opening fade). S08 00 and 01
  came out byte-identical. Old section files are in `slides_v5/_pdfcrop_backup_2026-10-14/`.
