# Animation Storyboard
Only generate the animation described in this file. Do not generate any other parts of the project.

**Current implementation: `animation_tokenization_v9.py`**


## Explanation
Generate the Foveated Tokenization part of the animation.

Section titles ("Normal Tokenization" / "Foveated Tokenization") appear instantly at the top of each half (y=3.5). There is no divider between the halves.

### Phase 1 — Grid overlays

Both images fade in simultaneously. Each image is 80% of its half-canvas width (`IMG_W = HALF * 0.80`), centered vertically at y=0.

**Left — Normal tokenization:**
- A 6×6 grid of high-resolution (blue, `#448AFF`) cells overlays the image.
- Each cell is 90% of the cell width/height (10% margin), `fill_opacity=0.30`.
- All 36 cells fade in with `lag_ratio=0.02` over 1.2s.

**Right — Foveated tokenization:**
- The center 2×2 block of the 6×6 grid → 4 high-resolution (blue) cells, `fill_opacity=0.42`.
- The remaining 8 macro-blocks (each spanning 2×2 original cells) → 8 low-resolution (green, `#00C853`) cells, `fill_opacity=0.30`.
- LR cell size: `2*cw - margin`, where `margin = cw * 0.10` (same absolute margin as HR cells).
- HR and LR cells fade in together with `lag_ratio=0.03` over 1.3s.

### Phase 2 — Dimension labels + Legend (3-second pause)

After both grids appear (0.4s wait), H/W dimension braces animate in on the **normal grid** (run_time=0.5s):
- `W` brace: spans the grid horizontally on the **right** side; label to the right of brace (font size 32)
- `H` brace: spans the grid vertically on the **bottom** side; label below brace (font size 32)

A legend then appears instantly (`self.add()`):
- Blue square + "High-resolution tokens" (white, font size 20)
- Green square + "Low-resolution tokens" (white, font size 20)
- Items arranged horizontally with `buff=0.9` between them.
- **Vertical position**: `legend_y = (img_bottom + canvas_bottom) / 2` — exactly midway between image bottom and canvas bottom.
- **Horizontal position**: right-edge margin equals bottom-edge margin (`bottom_margin = legend.get_bottom()[1] + frame_height/2`; `legend.set_x(frame_width/2 - bottom_margin - legend.width/2)`).

**Wait 3 seconds** after the legend appears.

### Phase 3 — Token sequence comparison

The images, section titles, legend, and H/W braces/labels all fade out simultaneously (no divider to fade since it was removed).

All grid cells morph simultaneously (`self.play(*transforms, run_time=2.0)`) into four horizontal token rows. Each token travels a randomly curved arc path (`path_arc` ∈ [−π/2.5, π/2.5], fixed seed=7), producing an organic "fuzzy" motion. All tokens start and end at the same time. Final positions are always correct — grid cell k lands on its designated 1D slot.

**`SIDE_BIG = (FW * 0.80) / (12 + 11 * 0.20)`** — sized so 12 tokens fill 80% of canvas width. All four rows share this token size, making the 3× count difference immediately obvious.

**Normal group (top, 3 rows of 12):**
- Row 1 at y=2.4, Row 2 at y=2.4−ROW_SEP, Row 3 at y=2.4−2×ROW_SEP
- `ROW_SEP = SIDE_BIG * 1.30` (vertical distance between row centers)
- All 36 blue (`#448AFF`) tokens; `lg[0..11]` → row 1, `lg[12..23]` → row 2, `lg[24..35]` → row 3
- Labeled "Normal Tokenization: H × W" (white, font size 28) above row 1.

**Foveated sequence (bottom, 1 row of 12, y=−2.2):**
- 12 tokens: 4 green (LR) + 4 blue (HR) + 4 green (LR), gap = 20% of token side.
- Same width as each normal row — visually one-third the height of the normal group.
- Labeled "Foveated Tokenization: L ≪ H × W" (white, font size 28) above the row.

Token ordering in the foveated sequence matches the grid's row-major traversal:
- `rl_cells[0..3]` → fov positions 0–3 (top-left LR macro-blocks)
- `rh[0..3]` → fov positions 4–7 (center HR cells)
- `rl_cells[4..7]` → fov positions 8–11 (bottom-right LR macro-blocks)

Both labels appear instantly (`self.add()`) after the morph, then wait 3 seconds.

Label vertical position: `LBL_Y = SEQ_Y + SIDE_BIG/2 + SIDE_BIG*0.40 + 0.18`


## MUST CONFORM GUIDELINES
- Visuals should never overlap
- Text should just appear, not using animations.
- Use the CMU-serif font. Make sure that the font is correctly used.
- High-res tokens should be blue (`#448AFF`) and low-res tokens green (`#00C853`)
- Tokens should have slightly rounded edges (10% corner radius), and exactly square, even after flattened into a 1D sequence
- Only show number of tokens in labels, no "X fewer" comparisons
- Legend labels use plural: "High-resolution tokens" / "Low-resolution tokens"
- Gap between tokens in 1D sequence = 20% of token side


## Output Format
Generate the following video formats:
- 480p, 15fps first for testing
- 1080p, 60fps second for final version

Additional outputs:
- Always save the animation script in a file called `animation_XXX.py`. Do not overwrite existing files. XXX is the name of the animation.
- After every new animation version, update this `storyboard.md` to accurately reflect the current implementation.
