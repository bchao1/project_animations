# Style notes — Level-of-Token Diffusion teasers

Distilled from the finished teasers in `/home/brianchc/project_animations/` (scripts, storyboards, CLAUDE.md, and
contact sheets of `media/final/` for both the 5-min and `teaser_1min/` cuts of: large_etendue_holography,
mpi_holography, high_brightness_holography, random_phase_wave_splatting, gaussian_wave_splatting,
fisheye_rectification, foveated_diffusion/teaser_{5min,1min}, spectral_progressive_diffusion).
The LoT teasers copy their helpers from `foveated_diffusion/teaser_1min/animation_foveated_diffusion_1min_v4.py`
(closest sibling: token-efficient DiTs) and follow these rules.

## What the references do (and what we copy)

**Structure.**
- 1-min: 3 scenes, ≈ 55–65 s (hard max 75 s). *S1 hook + insight* (≈ 24 s) → *S2 method in one picture*
  (≈ 15 s) → *S3 payoff + end card* (≈ 20 s). No headers, no part cards.
- 5-min: ≈ 10 scenes, 4:30–5:30. Header top-left “k · title” (24, dim, y 3.62); part cards only at the three acts
  (“Part 1 · The problem”, “Part 2 · The method”, “Part 3 · Results”); summary “in three ideas” + end card.
- Opening: bold title (72) + subtitle (32) over a small animated emblem of the running example
  (foveated: a `FoveaField` whose fovea wanders; spectral: growing grid + spectrum). Closing: end card with title,
  a coloured one-line tagline, authors/venue, URL (dim), held ≈ 5 s.
- One running example, transformed in place: foveated = the farmer (gaze blur) whose 16 × 16 all-HR grid morphs
  into the foveated field under the gaze; spectral = the corgi through its spectrum; fisheye = one toy building
  morphing pinhole → fisheye. The next beat always reuses the objects of the previous one (slide, shrink, recolour),
  never a hard cut to an unrelated picture.

**Toy first, paper last.** The idea is always explained with a computed toy or a hand-built infographic
(tokens as rounded squares, trapezoid encoders, a `DiT` box with an orange `LoRA` tag, live counters).
Paper figures / website media appear only in the payoff (1–2 beats), each with a tiny dim source label
(“paper Fig. 7a”, “simulated, paper Fig. 1”, “experimentally captured …”). Toy panels carry a dim footnote
(“toy simulation: …” / “illustration”).

**Numbers.** Every number on screen comes from the paper (or is computed live from the drawn toy and says so).
Payoff pattern: one plot or bar chart + big coloured numbers (84 bold) with a dim qualifier line under them
(“with 25% of the tokens”), then one qualitative row with a red failure note on the baseline.

**Pacing.** One caption per beat, bottom centre, ≤ ~10 words in the 1-min (≤ ~16 in the 5-min); held
≥ len/16 + 0.8 s (`_hold_caption`); visual changes wait for the caption. Animations are short (0.3–1.5 s);
counts/labels snap in instantly. Fade-outs 0.5–0.6 s between beats (`clear_all`).

**Composition.** Two-panel layouts are the default (left = what you see / source, right = what the model computes),
panels ≈ 4.6 units, centres x ≈ ±3.4, labels 26 dim above panels, counts 28 below. Nothing overlaps; captions never
cover content (content lives in y ∈ [−2.9, 3.2]).

## Render setup (identical to the references)
- Env: conda `imp_tome` + `export LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7` (set by `render.sh`).
- `render.sh [l|h] [script] [Scene …]`: renders all scenes in parallel with `--media_dir media`, then concatenates
  them (ffmpeg concat, `-c copy`) into `media/final/LoTDiffusion_<1min|5min>_<vN>_<480p15|1080p60>.mp4`.
- Assets built once on import into `data/generated/` (render.sh imports the script once before the parallel renders).
- QA: `scratch/qa_sheets.py <video dir> <out dir> [step]` → 4-column contact sheets per scene (copied verbatim).
- Versioning: `animation_lot_1min_vN.py` / `animation_lot_5min_vN.py`; a fix round = a new file; `storyboard.md`
  names the current version at the top and keeps a version history.

## Visual constants
| Property | Value |
|---|---|
| Background | black |
| Font | CMU Serif; Latin Modern Math for × ⊙ − → ≈ ∼ ∈ ℒ 𝔼 ′ ≪ ≤ ‖ · ↓ (`mathify`) |
| Math | Pango markup via `tx()` / `sub()` / `sup()`; no LaTeX; ≤ 1 equation per screen, none in the 1-min |
| Text | appears instantly (`self.add`), never written/faded in |
| Title / subtitle | 72 bold, y 2.75 / 32, y 1.95 |
| Caption | y −3.5, size 28 (26 on two centred lines), max width 13.2 |
| Footnote / source label | 18–20, dim `#8A8A8A` |
| Tokens | `RoundedRectangle`, corner radius 10% of side, 10% cell margin on every side (a token of extent e covers e cells and has side e·c − 0.2c) |
| dim / ghost / red | `#8A8A8A` / `#3A3A3A` / `#FF5252` (failure notes) |
| amber / purple | `#FFCA28` image numbers, highlight 1 / `#E040FB` video numbers, highlight 2 (as in foveated) |
| DiT box / LoRA | fill `#1E2A33`, stroke `#90A4AE` / `#FF7043` tag |

**Token-extent colours (new, extends the foveated HR/LR pair).** Size already encodes the level; colour just
separates fine from coarse, keeping the sibling's convention (blue = full resolution, green = merged):
| Extent | 1 × 1 | 2 × 2 | 4 × 4 | 8 × 8 |
|---|---|---|---|---|
| Colour | `#448AFF` (foveated HR) | `#00C853` (foveated LR) | `#1B7F3B` (foveated LR3) | `#0E5A2A` |

The painted detail map keeps the paper's own brush colours (black / orange / yellow / white = detail 0 / 0.3 / 0.6 / 1).

## Rules for this paper
- **One takeaway:** tell the model where detail matters and it spends tokens only there — a Level-of-Token layout
  of variable-size tokens, read by a fine-tuned pretrained DiT that still predicts full-resolution flow for every
  patch → up to 2.04× faster images and 3.53× faster video at quality close to full resolution.
- **One running example:** the site's painted demo “a girl riding a corgi” (LoT-FLUX.2): painted detail map →
  its real 64 × 64 layout (1,546 of 4,096 tokens: 1,312 · 140 · 79 · 15 of extents 1, 2, 4, 8) → generated image.
  The layout is rebuilt in manim from the site's own token grid (quadtree reproduced exactly, see storyboard).
- **Zero tangents.** No related-work tour, no RoPE derivation in the 1-min, no training-recipe details beyond
  “LoRA on a pretrained DiT”, no ablation tables in the 1-min.
- **Numbers source:** `level-of-token-diffusion/summaries/lot_diffusion.md`, checked against the paper (all values used
  match; note that §5.2 prose says “1.32–2.04×” / “1.58–3.53×” while Tables 1–2 give ours 1.560–2.040× / 2.262–3.530×;
  we quote the tables and the abstract-level “up to 2.04× / 3.53×”).
