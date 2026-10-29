# LoT multi-turn editing clip — storyboard (current: `render_edit_v3.py`)

20 s, 1920 × 1080, 60 fps, black, CMU Serif, house tokens (rounded, 10 % margin, mean image colour).
Data: the 100-turn bottle session, LoT k = 4 arm (`~/LoT-george-branch/results/2026-10-08_084925_ref-edit-9b-distilled-bottles-100turn`),
packed by `prep_data.py`. Layout: image panel left (540 px), context + target token panels centre (330 px),
DiT box right of them, live readout far right. One caption per beat, bottom centre.

| t (s) | beat | on screen | caption |
|---|---|---|---|
| 0.0–3.0 | LoRA transfer | turn-0 image, instruction "turn 1 · Replace the red wine …", glass region outlined; DiT box "base · 50 steps + CFG" with LoT LoRA tag → tag lifts, box becomes "distilled · 4 steps", tag drops back. Dim note under box: LoRA + projections trained on the base model, text → image only | Same LoT LoRA: base → distilled model |
| 3.0–5.8 | in-context | image copy flies into the context panel as 4,096 1 × 1 tokens, diagonal sweep into the LoT layout (1 × 1 glass, 2 × 2 ring, 4 × 4 rest), count 4,096 → 826 of 4,096; layout copies down as noise = target; arrows into the DiT; readout "1,652 tokens per pass, full-res 8,192" | In-context editing, never trained |
| 5.8–8.6 | compressor | one 4 × 4 token on the bottle label highlighted; callout over the (dimmed) image: 16 patches → down-projection → 1 token | Down-projection: 16 patches → 1 token |
| 8.6–11.6 | 4 steps + blend | step 1–4 / 4; edit tokens denoise into orange juice, tokens outside the edit snap back to the noised source each step (white flash); timer 0 → 1.9 s | 4 steps; outside the edit, blend in the source |
| 11.6–13.8 | paste + chain | target decodes, edit region flies into the image (turn 1 output); turn 2 instruction; image copy → context (new layout) | The result is the next turn's context |
| 13.8–16.8 | 100 turns | accelerating flip through turns 2–100 (image, both layouts, counts, instruction), readout turn N / 100 and summed DiT time (179 s) | 100 chained turns, 1.8 s each |
| 16.8–20.0 | why fast | panels fade; turn-100 image stays; seconds per turn: full-res base 50 steps + CFG 195 s → ÷25 → full-res distilled 4 steps 7.7 s → ÷4.3 → LoT distilled 1.8 s (avg 797 + 797 of 4,096 + 4,096 tokens); big purple "≈109× faster per turn" | Fewer steps × fewer tokens |

Numbers: per-turn DiT seconds averaged over the session's turns (k4 1.79 s and full 7.68 s on idle GPUs, 100 turns;
base 194.9 s = recipe9b reencode_k1_refdense_noblend, 5 turns, results/2026-10-07_152323, GPUs shared but
consistent with 25 × 7.68 s = 100 vs 4 network passes). Token counts from each turn's meta.json.

## Version history
- v1 (2026-10-08): first build.
- v2 (2026-10-08, user feedback: too sparse; LoRA transfer unclear; 1×1 / 2×2 / 4×4 not explained; % of full res is key):
  image 780 px + 390-px panels with overlay labels; beat A = base box and distilled box side by side, LoRA tag flies
  across ("same LoRA, no retraining"); beat C = 3-row callout with real patches (4 × 4 background 16 → 1 via
  down-projection, 2 × 2 ring 4 → 1, 1 × 1 edited object full res) + persistent legend under the image; each panel has
  a readout "N of 4,096 tokens", % bar, "X% of full res"; final card rows carry passes + tokens (100% / 19%).
- v3 (2026-10-08, user: v2 looks ugly next to the other sandbox clips, tangents, nothing aligned): rebuilt on the V1
  look — three equal 560-px panels in one centred row (context · current image | target | result), white label above,
  "N of 4,096 tokens (X%)" below, "+" / "→" in the gaps; no readout column, DiT box, bars, legend or tiny dim notes.
  Beats = the user's points: A LoRA base → distilled (two boxes, tag moves); B in-context editing (current image →
  context tokens 100% → 20%, target noise in the same layout); C down-projection (one 4 × 4 token's 16 patches spread,
  merge into 1 token; glass outlined "1 × 1 · full res"); D 4 steps + result; E chain 100 turns; F 25× × 4.3× = ≈109×.
- v4 (2026-10-08, user): instruction as a top text box "“Change the … to …”" (old/new from the ctx-sweep per_turn.csv,
  `data/generated/change_text.json`); large bold "turn N" box at the bottom-right of each panel (only while it has content).
- v5 (2026-10-14, slide deck): `render_edit_v5.py` = v4 with `CAPTIONS = False` (no bottom caption); one section,
  `media/LoT_edit_v5_1080p60.mp4` (copied to `slides_v5/`, manifest `../slides_v5/sections.json`).
