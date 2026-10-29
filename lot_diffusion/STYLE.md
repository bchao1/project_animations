# House style for every LoT Diffusion video (distilled from the other 20 teasers in this repo)

Source of these rules: the 1-min and 5-min teasers of foveated_diffusion, spectral_progressive_diffusion,
gaussian_wave_splatting, random_phase_wave_splatting, textured_gaussians, large_etendue_holography, mpi_holography,
high_brightness_holography, coded_aperture_lf_display and fisheye_rectification (scripts, storyboards, rendered
frames). The closest sibling is **foveated_diffusion/teaser_5min + teaser_1min**: when a rule below is ambiguous,
do what those two scripts do. Every LoT video (5-min, 1-min, 10-s thread clips) follows this file. No exceptions,
no extra decoration (no corner tags, no brand watermarks, no top headlines over every frame).

## 1. What a frame looks like
- Black background. One main visual, 0–2 secondary panels, generous empty space. Visuals never overlap.
- Text: CMU Serif via Pango `MarkupText` (`tx()`), no LaTeX. Glyphs CMU lacks go to Latin Modern Math (`mathify`).
  Sub/superscripts with `sub()`/`sup()` spans, never Unicode ₁ ². Variables italic `<i>x</i>`.
- **All text appears instantly** (`self.add`). Only images, tokens, arrows, shapes and plots animate.
- Caption: one sentence, white, 28 pt, centred at y −3.5; wider than 13.2 → two lines at 26 pt; bottom ≥ −3.85.
  ~10 words (5–16), plain present tense, "X: Y" colon style, no "we". One caption on screen at a time.
  Reading hold `len/16 + 0.8` s is enforced before the next caption / visual change / clear (`hold()`).
- Header (5-min only): `"k · title"`, 24 pt, DIM `#8A8A8A`, top-left, y 3.62. Plain claim or question. Stays for
  the whole scene. The 1-min has no headers. Thread clips use one fixed header (see §6).
- Labels: 20–26 pt, lowercase noun phrases ("what you see", "what the model computes", "a token on her face"),
  above panels (`buff` 0.15–0.25) in DIM or in the method colour; failure notes 22 pt RED under images
  ("noisy low-res regions"); source/toy notes 16–18 pt DIM ("paper Fig. 7", "toy layout on a 16 × 16 grid").
- Numbers: headline numbers big and in the accent colour (e.g. `tx("2.04×", 84, AMBER, bold=True)` + "faster
  images" 32), inline numbers `col("256", HR)`, derived key line boxed. Speed-ups written "N.NN×".

## 2. Palette (foveated values; tokens keep the sibling's convention)
HR / 1×1 token `#448AFF` · 2×2 token `#00C853` · 4×4 token `#1B7F3B` (LR3) · 8×8 token `#0E4A22`
DIM `#8A8A8A` · GHOST `#3A3A3A` (frames, empty tokens) · RED `#FF5252` (baseline failure) · AMBER `#FFCA28`
(image headline, highlights) · PURPLE `#E040FB` (video) · LORA `#FF7043` · BOX `#1E2A33` (DiT box fill, stroke
`#90A4AE`) · VAE trapezoid fill `#37474F` stroke `#B0BEC5`. Rectangular tokens take the colour of their area
(2 cells → between blue and green, 8 cells → between green and dark green). Ours = HR blue, bold label.

## 3. Tokens
Vector `RoundedRectangle`s, corner radius 10% of the short side, 10% cell margin (1×1 side = 0.8·cell). Toy grids are
small: 16 × 16 (or 8 × 8 for method close-ups), 4.4–5.0 wide. Patch colours via `patch_colors()` of a real image;
noise via `noise_colors()`. Merges are `ReplacementTransform` of the child tokens into the parent token, in
`LaggedStart` ordered by distance from the subject (1.5–2.8 s). Toy layouts are labelled as toy; real layouts
(paper figures, website videos) are shown as the paper shows them.

## 4. Motion and pacing
- Build one visual and transform it in place across the beat (the 3b1b move): image → grid over it → patch-colour
  tokens → recolour → merge → count. Keep the object; do not cut to a new slide mid-idea.
- Images `FadeIn` 0.4–0.8 s; token grids `LaggedStart(FadeIn, lag 0.002–0.02)` 0.8–1.5 s; arrows `GrowArrow`
  0.3–0.5 s; boxes/outlines/axes `Create` 0.3–0.6 s; curves `Create` 0.8–1.8 s; bars `GrowFromEdge` 0.5–0.7 s;
  `Indicate` 0.7 s; sweeps with a `ValueTracker` 1.5–6 s.
- Beat ends with `clear_all()` (uncaption + FadeOut 0.5–0.8 s). `keep=[...]` only to carry the object into the
  next beat. Scenes are hard cuts.
- Never: `Write`, animated text, camera moves, audio, crossfades between unrelated slides.

## 5. Structure
**5-min (4:30–5:30):** `S00_Title … S09_Outlook`, 10 scenes. Part cards only at the three acts
(`title_card("Part 1 · The problem", …)`, `"Part 2 · The method"`, `"Part 3" / "Results"`), 1.4–1.6 s.
S00: title 72 bold y 2.75, subtitle 32 y 1.95, animated token-field hero (≈3.8 wide at y −0.75), authors line
22 DIM y −3.3. S01 ends on a question; a bridge card (40 pt, 2.2 s) may close S02. Method: one mechanism per scene,
≤ 1 equation per beat, colour-coded symbols. Results: plots re-drawn in manim from paper numbers (manual tick
labels), a number column or bars, paper-figure grids with coloured method labels and a dim source line.
S09: limitation beat → "Level-of-Token Diffusion in three ideas" (numbered, 1.6–1.9 s each) → accent headline
line → end card (title 60 bold, links 28 DIM, authors 24 DIM), 3.5–5 s, no caption.
**1-min (55–70 s):** 3 scenes `S1_Insight` / `S2_Method` / `S3_Payoff`, 6–11 beats, no part cards, no headers.
Beat 1 is the title over the animated hero (no separate card). ≤ 1 equation. Payoff = plot + big numbers,
one labelled comparison row, end card 5 s.
**10-s thread clips:** see §6.

## 6. Thread clips (this project's addition, same grammar)
Each clip is one beat of the 5-min teaser, rendered by the same code (`clips/` imports the 5-min script's beat
methods), 8–14 s, with self-contained captions (no "So…" or "It…" that needs a previous beat). First frame already
shows the beat's main visual (X shows the first frame as the thumbnail). Only addition: the header line reads
`"Level-of-Token Diffusion"` (24 pt DIM, y 3.62) so a clip seen alone is attributable. Ends on the finished
visual held ≥ 1.5 s (no fade to black).

## 7. Accuracy and process
- Every number on screen comes from `level-of-token-diffusion/summaries/lot_diffusion.md` (paper + site); toy counts
  are labelled toy and must be arithmetically right.
- `storyboard.md` is written first and kept as ground truth; scripts are versioned `*_vN.py`, never overwritten.
- 480p15 test → contact sheets (`scratch/qa_sheets.py`, every 2 s) → fix overlaps, off-frame text, visuals that
  change before the caption is read → 1080p60.
