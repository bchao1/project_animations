# Masking Strategies Animation Storyboard

**Current implementation: `animation_mask_training_v5.py`**

## Overview
Three-phase animation comparing randomized-mask training and saliency-guided training.
Canvas: 1920×1080, black background, CMU Serif font throughout.
All arrows and lines: stroke_width=5.0, tip_length=0.25.

---

## Phase 1 — Input Image (3s total)

No title. Input image (`saliency_masking/image.jpg`, width=5.25) centered at ORIGIN.
"Input Image" label (28pt) above image (buff=0.18). Image fades in over 0.8s, then wait 2.0s.

---

## Phase 2 — Randomized-Mask Training (~8s total)

**Transition (0.7s):** Input image slides to (−5.0, 0) and scales to width=2.6;
"Input Image" label (old) fades out simultaneously.
New "Input Image" label (22pt) appears instantly after slide completes.
Title "Randomized-Mask Training" (42pt, y=3.4) appears AFTER image has settled.

**Layout:**
- TITLE_Y = 3.4 (title bottom ≈ 3.10; label top ≈ 2.76; top mask top ≈ 2.20 → 0.34-gap chain ✓)
- Input image: center (−5.0, 0), width=2.6 → right edge x=−3.7; white border.
- 3 mask images only (no circle) centered at x=4.5, y = 1.3 / −0.7 / −2.7 (spacing=2.0)
  - W_EACH = 1.8 (maximised: 0.37 margin top label→title, 0.44 margin bottom mask→canvas)
  - Inter-image gap = 0.2 units (spacing 2.0 = W_EACH 1.8 + gap 0.2)
  - Asset: `random_masking/{0000,0001,0002}/foveation_mask_pix64_up512.png`
  - Each image has white border (stroke_width=5.0)
- Single shared "Foveation Mask" label (22pt) above the TOP mask only (column header), buff=0.18

**Branching arrows** (white, stroke_width=5.0, tip_length=0.25):
- Feeder line: (img.right, 0) → (−1.0, 0) — exact, no offset
- Vertical stem: (−1.0, −2.7) → (−1.0, 1.3)
- Three arrows: (−1.0, y) → (mask.left − 0.08, y) for y ∈ {1.3, −0.7, −2.7} — exact T-junction

**Sequence:**
1. Create feeder + stem (0.4s)
2. LaggedStart create 3 branches (lag=0.25, total 0.7s)
3. LaggedStart FadeIn 3 pairs (lag=0.3, total 1.0s)
4. Wait 0.8s
5. Pulsing ×5 in order [2, 0, 1, 2, 0]:
   - `ShowPassingFlash` (white, width=12, time_width=0.15, run_time=0.65) travels
     along polyline path: feeder → down/up stem to branch y → along branch to samp_lx
   - Then pair scales 1.08× (0.12s) and back (0.12s) — no color change
6. Wait 1.5s

---

## Phase 3 — Saliency-Guided Training (~7s total)

**Transition (0.7s):** Title morphs to "Saliency-Guided Training" (42pt, y=3.4). Phase 2 elements fade out.
Input image and "Input Image" label remain.

**Linear pipeline (left → right), all images white-bordered:**

| Element | Asset | Center | Width |
|---|---|---|---|
| Input image | stays from Phase 2 | (−5.0, 0) | 2.6 |
| Saliency map | `saliency_masking/saliency.png` | (0, 0) | 2.5 |
| Foveation mask | `saliency_masking/foveation_mask_pix64_up512.png` | (4.2, 0) | 2.5 |

Mask-only output (no circle image). Width=2.5 matches saliency map.
"Foveation Mask" label (22pt) above mask, buff=0.18 — same format as "Saliency Map".
"Saliency Map" label (22pt) above saliency map, buff=0.18.

**Arrows:**
- Arrow 1: (img_rx+0.08, 0) → (sal.left−0.08, 0) — label "Saliency\nDetection" (20pt) above, buff=0.15
- Arrow 2: (sal.right+0.08, 0) → (fov.left−0.08, 0) — label "Thresholding" (20pt) above, buff=0.15

**Sequence:**
1. Create arrow 1 (0.5s); add "Saliency\nDetection" label instantly
2. FadeIn saliency (0.5s); add "Saliency Map" label instantly; wait 0.3s
3. Create arrow 2 (0.5s); add "Thresholding" label instantly
4. FadeIn foveation mask (0.5s); add "Foveation Mask" label instantly
5. Wait 3.0s

---

## Output Format
See top-level `CLAUDE.md` for render formats, media folder location, versioning, and storyboard update rules.

## Assets
- `data/saliency_masking/image.jpg` — input lynx photo
- `data/saliency_masking/saliency.png` — heatmap
- `data/saliency_masking/foveation_mask_pix64_up512.png` — pixelated binary mask
- `data/random_masking/{0000,0001,0002}/circle.jpg` — image with random foveal circle
- `data/random_masking/{0000,0001,0002}/foveation_mask_pix64_up512.png` — random binary masks
